#!/usr/bin/env python3
import csv
import struct
import sys
import time
from collections import Counter, defaultdict
from datetime import timezone, timedelta
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RECORD_SIZE = 72
READ_CHUNK_RECORDS = 1_000_000
READ_CHUNK_BYTES = RECORD_SIZE * READ_CHUNK_RECORDS
NSE_EPOCH_OFFSET_NS = (315_532_800 - 19_800) * 1_000_000_000
IST = timezone(timedelta(hours=5, minutes=30))

MT_NEW_ORDER    = ord("N")
MT_MOD_ORDER    = ord("M")
MT_CXL_ORDER    = ord("X")
MT_TRADE        = ord("T")
MT_TRADE_CANCEL = ord("C")
MT_SPD_NEW      = ord("G")
MT_SPD_MOD      = ord("H")
MT_SPD_CXL      = ord("J")
MT_SPD_TRADE    = ord("K")
MT_HEARTBEAT    = ord("Z")

ORDER_TYPES     = frozenset([MT_NEW_ORDER, MT_MOD_ORDER, MT_CXL_ORDER, MT_SPD_NEW, MT_SPD_MOD, MT_SPD_CXL])
TRADE_TYPES     = frozenset([MT_TRADE, MT_TRADE_CANCEL, MT_SPD_TRADE])
HEARTBEAT_TYPES = frozenset([MT_HEARTBEAT])
ALLOWED_TYPES   = ORDER_TYPES | TRADE_TYPES | HEARTBEAT_TYPES

# ALL 4 instruments — was hardcoded to FUTSTK only
INSTRUMENTS = {"FUTSTK", "FUTIDX", "OPTSTK", "OPTIDX"}

EXPECTED_BODY_BYTES_BY_TYPE = {
    MT_NEW_ORDER: 29, MT_MOD_ORDER: 29, MT_CXL_ORDER: 29,
    MT_SPD_NEW:   29, MT_SPD_MOD:   29, MT_SPD_CXL:   29,
    MT_TRADE: 36,     MT_TRADE_CANCEL: 36, MT_SPD_TRADE: 36,
    MT_HEARTBEAT: 4,
}

RAW_DTYPE = np.dtype([
    ("cap_s",       "<u4"),
    ("_pad0",       "<u4"),
    ("cap_ns_part", "<u4"),
    ("_pad1",       "<u4"),
    ("_meta",       "10u1"),
    ("msg_len",     "<i2"),
    ("stream_id",   "<i2"),
    ("seq_no",      "<i4"),
    ("msg_type",    "u1"),
    ("body",        "37u1"),
])
assert RAW_DTYPE.itemsize == RECORD_SIZE

KEEP_SCHEMA = pa.schema([
    ("capture_ns",    pa.int64()),
    ("exchange_ns",   pa.int64()),
    ("stream_id",     pa.int16()),
    ("seq_no",        pa.int32()),
    ("msg_type",      pa.string()),
    ("token",         pa.int32()),
    ("side",          pa.string()),
    ("order_id",      pa.float64()),
    ("buy_order_id",  pa.float64()),
    ("sell_order_id", pa.float64()),
    ("price",         pa.int32()),
    ("quantity",      pa.int32()),
])

_ORDER_ARR     = np.array(sorted(ORDER_TYPES),     dtype=np.uint8)
_TRADE_ARR     = np.array(sorted(TRADE_TYPES),     dtype=np.uint8)
_HEARTBEAT_ARR = np.array(sorted(HEARTBEAT_TYPES), dtype=np.uint8)
_ALLOWED_ARR   = np.array(sorted(ALLOWED_TYPES),   dtype=np.uint8)

_MT_CHAR = np.empty(256, dtype=object)
for _c in ALLOWED_TYPES:
    _MT_CHAR[_c] = chr(_c)


def _require(condition: bool, msg: str):
    if not condition:
        raise ValueError(msg)


def sanitize_symbol(symbol: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in symbol)


# ---------------------------------------------------------------------------
# FIXED: load_segment_maps — now loads ALL 4 instruments (was FUTSTK only)
# ---------------------------------------------------------------------------
def load_futstk_maps(contract_csv_path: Path):
    """
    Load token metadata for ALL 4 instruments: FUTSTK, FUTIDX, OPTSTK, OPTIDX
    Returns:
        token_to_meta              : {token -> (symbol, expiry)}
        expected_tokens_by_expiry  : {expiry -> set(token)}
        expected_symbols_by_expiry : {expiry -> set(symbol)}
    """
    token_to_meta              = {}
    expected_tokens_by_expiry  = defaultdict(set)
    expected_symbols_by_expiry = defaultdict(set)
    inst_count                 = Counter()

    with open(contract_csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row_num, row in enumerate(reader, start=2):
            if not row or len(row) < 6:
                continue
            token_str  = row[2].strip()
            instrument = row[3].strip()
            symbol     = row[4].strip()
            expiry_str = row[5].strip()

            # ← FIXED: was "if instrument != 'FUTSTK': continue"
            if instrument not in INSTRUMENTS:
                continue

            try:
                token  = int(token_str)
                expiry = int(expiry_str)
            except ValueError:
                raise ValueError(f"Bad token/expiry at row {row_num}: {row}")

            prev    = token_to_meta.get(token)
            current = (symbol, expiry)
            if prev is not None and prev != current:
                raise RuntimeError(
                    f"Token {token} maps inconsistently: {prev} vs {current}"
                )

            token_to_meta[token] = current
            expected_tokens_by_expiry[expiry].add(token)
            expected_symbols_by_expiry[expiry].add(symbol)
            inst_count[instrument] += 1

    if not token_to_meta:
        raise RuntimeError(
            f"No tokens found for instruments {INSTRUMENTS} in contract CSV."
        )

    print(f"[CONTRACT] Loaded {len(token_to_meta):,} tokens across "
          f"{len(expected_tokens_by_expiry)} expiries")
    for inst in sorted(INSTRUMENTS):
        print(f"  {inst:10s}: {inst_count.get(inst, 0):,} tokens")

    return token_to_meta, expected_tokens_by_expiry, expected_symbols_by_expiry


# ---------------------------------------------------------------------------
# FULLY VECTORIZED parse
# ---------------------------------------------------------------------------
def parse_chunk_projected(buf: bytes) -> tuple[pa.Table, Counter]:
    n_aligned = len(buf) - (len(buf) % RECORD_SIZE)
    _require(n_aligned == len(buf), f"Internal error: chunk not record-aligned ({len(buf)} bytes).")
    if n_aligned == 0:
        return (
            pa.table({f.name: pa.array([], type=f.type) for f in KEEP_SCHEMA}, schema=KEEP_SCHEMA),
            Counter(),
        )

    raw = np.frombuffer(buf, dtype=RAW_DTYPE)
    n   = len(raw)

    bad_pad0 = np.where(raw["_pad0"] != 0)[0]
    if len(bad_pad0):
        raise ValueError(f"Non-zero pad0 found; first index={int(bad_pad0[0])}")
    bad_pad1 = np.where(raw["_pad1"] != 0)[0]
    if len(bad_pad1):
        raise ValueError(f"Non-zero pad1 found; first index={int(bad_pad1[0])}")

    mt = raw["msg_type"]
    invalid_mask = ~np.isin(mt, _ALLOWED_ARR)
    if invalid_mask.any():
        print(
            f"[WARN] Skipping {invalid_mask.sum()} records with unknown msg_type "
            f"(codes: {np.unique(mt[invalid_mask]).tolist()})",
            flush=True,
        )
        raw = raw[~invalid_mask]
        mt  = raw["msg_type"]
        n   = len(raw)
        if n == 0:
            return (
                pa.table({f.name: pa.array([], type=f.type) for f in KEEP_SCHEMA}, schema=KEEP_SCHEMA),
                Counter(),
            )

    msg_len = raw["msg_len"]
    bad_msg_len_rows = np.where(msg_len <= 0)[0]
    if len(bad_msg_len_rows):
        raise ValueError(f"Non-positive msg_len found; first index={int(bad_msg_len_rows[0])}")

    for t, expected_min in EXPECTED_BODY_BYTES_BY_TYPE.items():
        rows = np.where(mt == t)[0]
        if len(rows):
            bad = rows[msg_len[rows] < expected_min]
            if len(bad):
                raise ValueError(
                    f"msg_len smaller than decoder expectation for msg_type={chr(t)}; "
                    f"first index={int(bad[0])}, msg_len={int(msg_len[bad[0]])}, "
                    f"expected_at_least={expected_min}"
                )

    capture_ns = (
        raw["cap_s"].astype(np.int64) * 1_000_000_000
        + raw["cap_ns_part"].astype(np.int64)
    )

    body = raw["body"]

    is_order     = np.isin(mt, _ORDER_ARR)
    is_trade     = np.isin(mt, _TRADE_ARR)
    is_heartbeat = np.isin(mt, _HEARTBEAT_ARR)

    exchange_ns_raw = body[:, 0:8].copy().view(np.int64).reshape(n) + NSE_EPOCH_OFFSET_NS
    exchange_ns = np.where(is_order | is_trade, exchange_ns_raw, np.int64(0))

    token_order = body[:, 16:20].copy().view(np.int32).reshape(n)
    token_trade = body[:, 24:28].copy().view(np.int32).reshape(n)
    token_col   = np.where(is_order, token_order, np.where(is_trade, token_trade, np.int32(0)))

    price_order = body[:, 21:25].copy().view(np.int32).reshape(n)
    price_trade = body[:, 28:32].copy().view(np.int32).reshape(n)
    price_hb    = body[:,  0: 4].copy().view(np.int32).reshape(n)
    price_col   = np.where(is_order, price_order, np.where(is_trade, price_trade, price_hb))

    qty_order = body[:, 25:29].copy().view(np.int32).reshape(n)
    qty_trade = body[:, 32:36].copy().view(np.int32).reshape(n)
    qty_col   = np.where(is_order, qty_order, np.where(is_trade, qty_trade, np.int32(0)))

    oid_raw  = body[:, 8:16].copy().view(np.float64).reshape(n)
    order_id = np.where(is_order, oid_raw, np.nan)

    buy_oid_raw  = body[:,  8:16].copy().view(np.float64).reshape(n)
    sell_oid_raw = body[:, 16:24].copy().view(np.float64).reshape(n)
    buy_oid  = np.where(is_trade, buy_oid_raw,  np.nan)
    sell_oid = np.where(is_trade, sell_oid_raw, np.nan)

    side_byte = body[:, 20].astype(np.uint8)
    bad_side  = np.where(is_order & ~np.isin(side_byte, [ord("B"), ord("S")]))[0]
    if len(bad_side):
        raise ValueError(
            f"Invalid side byte for order message at row={int(bad_side[0])}: "
            f"{int(side_byte[bad_side[0]])!r}"
        )

    side_col = np.full(n, "", dtype=object)
    side_col[is_order & (side_byte == ord("B"))] = "B"
    side_col[is_order & (side_byte == ord("S"))] = "S"

    msg_type_col = _MT_CHAR[mt]

    unique, cnts = np.unique(mt, return_counts=True)
    counts = Counter({chr(int(u)): int(c) for u, c in zip(unique, cnts)})

    table = pa.table(
        {
            "capture_ns":    pa.array(capture_ns,   type=pa.int64()),
            "exchange_ns":   pa.array(exchange_ns,  type=pa.int64()),
            "stream_id":     pa.array(raw["stream_id"].astype(np.int16), type=pa.int16()),
            "seq_no":        pa.array(raw["seq_no"].astype(np.int32),    type=pa.int32()),
            "msg_type":      pa.array(msg_type_col,  type=pa.string()),
            "token":         pa.array(token_col,     type=pa.int32()),
            "side":          pa.array(side_col,      type=pa.string()),
            "order_id":      pa.array(order_id,      type=pa.float64()),
            "buy_order_id":  pa.array(buy_oid,       type=pa.float64()),
            "sell_order_id": pa.array(sell_oid,      type=pa.float64()),
            "price":         pa.array(price_col,     type=pa.int32()),
            "quantity":      pa.array(qty_col,       type=pa.int32()),
        },
        schema=KEEP_SCHEMA,
    )
    return table, counts


# ---------------------------------------------------------------------------
# Streaming reader
# ---------------------------------------------------------------------------
def parse_cap_batches(cap_path: Path):
    leftover         = b""
    msg_type_counts  = Counter()
    rows_total       = 0
    bytes_read       = 0
    start_time       = time.time()
    last_print       = start_time
    file_size        = cap_path.stat().st_size

    with open(cap_path, "rb") as f:
        while True:
            chunk = f.read(READ_CHUNK_BYTES)
            if not chunk:
                break
            data      = leftover + chunk
            remainder = len(data) % RECORD_SIZE
            leftover  = data[-remainder:] if remainder else b""
            aligned   = data[:-remainder] if remainder else data
            if aligned:
                table, counts = parse_chunk_projected(aligned)
                msg_type_counts.update(counts)
                rows_total += table.num_rows
                bytes_read += len(aligned)
                yield table
            now = time.time()
            if (now - last_print) > 5:
                elapsed = now - start_time
                rate_mb = (bytes_read / (1024 ** 2)) / elapsed if elapsed else 0.0
                pct     = min(bytes_read / file_size * 100.0, 100.0) if file_size else 0.0
                print(
                    f"  {cap_path.name}: {pct:5.1f}% | "
                    f"{bytes_read / (1024**3):.2f} / {file_size / (1024**3):.2f} GB | "
                    f"{rate_mb:,.0f} MB/s | rows={rows_total:,}",
                    flush=True,
                )
                last_print = now

    if len(leftover) > 0:
        print(
            f"[WARN] {cap_path.name}: trailing_bytes={len(leftover)}; "
            f"dropping exactly 1 incomplete final record",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Shard writer — now handles all 4 instruments
# ---------------------------------------------------------------------------
def process_cap_into_futstk_shards(
    cap_path: Path,
    archive_stem: str,
    token_to_meta: dict,
    shard_root: Path,
    row_count_csv_path: Path | None = None,
):
    row_counts = defaultdict(int)

    # Pre-build vectorized lookup maps once (not per batch)
    TOKEN_MAP_SIZE = 1_000_000
    meta_to_gid   = {}
    gid_to_meta   = []
    token_gid_map = np.full(TOKEN_MAP_SIZE, -1, dtype=np.int32)

    for tok, meta in token_to_meta.items():
        if meta not in meta_to_gid:
            meta_to_gid[meta] = len(gid_to_meta)
            gid_to_meta.append(meta)
        gid = meta_to_gid[meta]
        if 0 <= tok < TOKEN_MAP_SIZE:
            token_gid_map[tok] = gid

    for batch_idx, table in enumerate(parse_cap_batches(cap_path), start=1):
        token_arr = table.column("token").to_numpy(zero_copy_only=False).astype(np.int32)

        # Vectorized gid lookup — replaces np.isin + Python loop
        in_range = (token_arr >= 0) & (token_arr < TOKEN_MAP_SIZE)
        gid_arr  = np.full(len(token_arr), -1, dtype=np.int32)
        gid_arr[in_range] = token_gid_map[token_arr[in_range]]

        known_mask    = gid_arr >= 0
        if not known_mask.any():
            continue

        known_gids    = gid_arr[known_mask]
        known_indices = np.where(known_mask)[0]

        # Vectorized group-by: argsort by gid, then split
        sort_order     = np.argsort(known_gids)
        sorted_gids    = known_gids[sort_order]
        sorted_indices = known_indices[sort_order]
        split_points   = np.where(np.diff(sorted_gids))[0] + 1

        for gid_group, idx_group in zip(
            np.split(sorted_gids,   split_points),
            np.split(sorted_indices, split_points),
        ):
            gid            = int(gid_group[0])
            symbol, expiry = gid_to_meta[gid]
            safe_symbol    = sanitize_symbol(symbol)

            idx_array  = pa.array(idx_group, type=pa.int64())
            subtable   = table.take(idx_array)

            out_dir    = shard_root / str(expiry) / safe_symbol
            out_dir.mkdir(parents=True, exist_ok=True)
            shard_path = out_dir / f"{archive_stem}_b{batch_idx:04d}.parquet"

            pq.write_table(subtable, str(shard_path), compression="snappy")
            row_counts[(expiry, symbol)] += subtable.num_rows

        if batch_idx % 10 == 0:
            print(f"[PROCESS] {cap_path.name} batch={batch_idx}")

    if row_count_csv_path is not None:
        import csv as _csv
        write_header = not row_count_csv_path.exists()
        with open(row_count_csv_path, "a", encoding="utf-8", newline="") as f:
            w = _csv.writer(f)
            if write_header:
                w.writerow(["archive_stem", "expiry", "symbol", "safe_symbol", "rows"])
            for (expiry, symbol), rows in sorted(row_counts.items()):
                w.writerow([archive_stem, expiry, symbol, sanitize_symbol(symbol), rows])

    return row_counts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) != 6:
        print(
            "Usage:\n"
            "  python mtbt_parser_fast_up.py "
            "<input_cap> <contract_csv> <shard_root> <archive_stem> <row_count_csv>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_cap     = Path(sys.argv[1])
    contract_csv  = Path(sys.argv[2])
    shard_root    = Path(sys.argv[3])
    archive_stem  = sys.argv[4]
    row_count_csv = Path(sys.argv[5])

    token_to_meta, _, _ = load_futstk_maps(contract_csv)
    shard_root.mkdir(parents=True, exist_ok=True)
    row_count_csv.parent.mkdir(parents=True, exist_ok=True)

    process_cap_into_futstk_shards(
        cap_path=input_cap,
        archive_stem=archive_stem,
        token_to_meta=token_to_meta,
        shard_root=shard_root,
        row_count_csv_path=row_count_csv,
    )


if __name__ == "__main__":
    main()
