#!/usr/bin/env python3
"""
allseg_extract_full.py
──────────────────────
Full pipeline for all 4 instruments with new directory structure.

Output structure:
    /media/prateek/Data/parsed_data/{date}/FUTSTK/{symbol}/{date}_FUTSTK_{symbol}_{expiry}.parquet
    /media/prateek/Data/parsed_data/{date}/FUTIDX/{symbol}/{date}_FUTIDX_{symbol}_{expiry}.parquet
    /media/prateek/Data/parsed_data/{date}/OPTSTK/{symbol}/{date}_OPTSTK_{symbol}_{expiry}.parquet
    /media/prateek/Data/parsed_data/{date}/OPTIDX/{symbol}/{date}_OPTIDX_{symbol}_{expiry}.parquet

Usage:
    /home/alpha/myenv/bin/python -u allseg_extract_full.py --date 20260320
    /home/alpha/myenv/bin/python -u allseg_extract_full.py --date 20260320 --clean
"""

import argparse
import csv
import ctypes
import gc
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _release_arrow_memory():
    gc.collect()
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _trim_malloc():
    """Thread-safe malloc_trim only — no gc.collect()."""
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PARSER_SCRIPT          = Path("/media/svipl/Data/historical_data/mtbt_parser_fast_up.py")
XZ_THREADS_PER_ARCHIVE = 3
PARALLEL_ARCHIVES      = 8
MERGE_WORKERS          = 3
PUBLISH_WORKERS        = 16

INSTRUMENTS   = {"FUTSTK", "FUTIDX", "OPTSTK", "OPTIDX"}
COMPRESSION   = "zstd"
TOKEN_LENGTH  = 5
ORDER_ID_COLS = ("order_id", "buy_order_id", "sell_order_id")
PARQUET_MAGIC = b"PAR1"

# Base output path — local storage
PARSED_DATA_ROOT = Path("/media/svipl/Data/shared")

# NAS output path
PARSED_DATA_NAS  = Path("/mnt/historical_data/tbt_data/parsed_data")

# Instruments to skip when publishing to shared path
SHARED_SKIP_INSTRUMENTS = {"OPTSTK", "OPTIDX"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cap_stem_from_name(filename: str) -> str:
    stem = Path(filename).name
    for ext in (".tar.xz", ".tar.gz", ".tar.bz2", ".tgz", ".cap"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    return stem


def free_disk_gb(path: Path) -> float:
    stat = os.statvfs(str(path))
    return (stat.f_bavail * stat.f_frsize) / (1024 ** 3)


# ---------------------------------------------------------------------------
# Load contract metadata — all 4 instruments
# ---------------------------------------------------------------------------
def load_segment_maps(contract_csv_path: Path):
    token_to_meta              = {}
    token_to_instrument        = {}
    expected_tokens_by_expiry  = defaultdict(set)
    expected_symbols_by_expiry = defaultdict(set)
    tokens_by_instrument       = defaultdict(set)

    with open(contract_csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or len(row) < 6:
                continue
            token_str  = row[2].strip()
            instrument = row[3].strip()
            symbol     = row[4].strip()
            expiry_str = row[5].strip()

            if instrument not in INSTRUMENTS:
                continue

            token  = int(token_str)
            expiry = int(expiry_str)

            token_to_meta[token]       = (symbol, expiry, instrument)
            token_to_instrument[token] = instrument
            expected_tokens_by_expiry[expiry].add(token)
            expected_symbols_by_expiry[expiry].add(symbol)
            tokens_by_instrument[instrument].add(token)

    tokens_by_instrument = {k: frozenset(v) for k, v in tokens_by_instrument.items()}

    from collections import Counter
    inst_count = Counter(v[2] for v in token_to_meta.values())
    print(f"[CONTRACT] Loaded {len(token_to_meta):,} tokens across "
          f"{len(expected_tokens_by_expiry)} expiries")
    for inst in sorted(INSTRUMENTS):
        print(f"  {inst:10s}: {inst_count.get(inst, 0):,} tokens")

    return (token_to_meta, expected_tokens_by_expiry, expected_symbols_by_expiry,
            token_to_instrument, tokens_by_instrument)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def prepare_local_dirs(dirs: dict, clean_start: bool = False):
    if clean_start and dirs["work"].exists():
        print(f"[SETUP] Removing existing work dir: {dirs['work']}")
        shutil.rmtree(dirs["work"])
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Discover archives
# ---------------------------------------------------------------------------
def discover_cap_archives(source_root: Path):
    files = sorted(source_root.glob("*.cap.tar.xz"))
    if not files:
        raise RuntimeError(f"No .cap.tar.xz files found under {source_root}")
    print(f"[SOURCE] Found {len(files)} input .cap.tar.xz files")
    for f in files:
        print(f"  {f.name}  ({f.stat().st_size / 1e9:.2f} GB)")
    return files


def get_completed_archives(shard_dir: Path) -> set:
    completed = set()
    if not shard_dir.exists():
        return completed
    for parquet_file in shard_dir.rglob("*.parquet"):
        stem  = parquet_file.stem
        parts = stem.rsplit('_b', 1)
        if len(parts) == 2 and len(parts[1]) == 4 and parts[1].isdigit():
            completed.add(parts[0])
        else:
            completed.add(stem)
    return completed


# ---------------------------------------------------------------------------
# Worker: copy -> extract -> parse -> delete  (one archive)
# ---------------------------------------------------------------------------
def _process_one_archive_worker(args):
    src, stage_dir, extract_dir, shard_dir, row_count_csv, contract_csv = args
    src = Path(src)

    # Skip copy — extract directly from NAS source (no pointless NAS→NAS copy)
    extract_subdir = Path(extract_dir) / src.name.replace(".cap.tar.xz", "")
    extract_subdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tar",
         f"--use-compress-program=xz -T{XZ_THREADS_PER_ARCHIVE} -d",
         "-xf", str(src),
         "-C", str(extract_subdir)],
        check=True,
    )

    cap_files = list(extract_subdir.glob("*.cap"))
    if len(cap_files) != 1:
        raise RuntimeError(
            f"Expected 1 .cap in {extract_subdir}, found {len(cap_files)}"
        )
    cap_path     = cap_files[0]
    archive_stem = cap_stem_from_name(str(src))

    cmd = [
        sys.executable,
        str(PARSER_SCRIPT),
        str(cap_path),
        str(contract_csv),
        str(shard_dir),
        archive_stem,
        str(row_count_csv),
    ]
    subprocess.run(cmd, check=True)

    shutil.rmtree(extract_subdir)

    return src.name


# ---------------------------------------------------------------------------
# Filter helpers — same as old, just count dropped rows
# ---------------------------------------------------------------------------
def filter_table(table: pa.Table) -> tuple:
    """Drop heartbeats (msg_type=Z) and TEST symbols."""
    before     = table.num_rows
    dropped_hb = 0
    dropped_ts = 0

    if "msg_type" in table.schema.names:
        mask       = pc.not_equal(table.column("msg_type"), pa.scalar("Z"))
        table      = table.filter(mask)
        dropped_hb = before - table.num_rows

    if "symbol" in table.schema.names:
        before_ts  = table.num_rows
        mask       = pc.invert(pc.starts_with(table.column("symbol"), pattern="TEST"))
        table      = table.filter(mask)
        dropped_ts = before_ts - table.num_rows

    return table, dropped_hb, dropped_ts


# ---------------------------------------------------------------------------
# Merge shards
# ---------------------------------------------------------------------------
def _merge_one_symbol_streaming(
    symbol_dir: Path,
    final_dir: Path,
    date: str,
    expiry: int,
    tokens_by_instrument: dict,
) -> tuple:
    symbol        = symbol_dir.name
    shards        = sorted(symbol_dir.glob("*.parquet"))
    writers       = {}
    out_paths     = {}
    total_drop_hb = 0
    total_drop_ts = 0

    # Build per-instrument pyarrow token sets once (accurate — no range limits)
    inst_pa_sets = {
        inst: pa.array(list(tokens), type=pa.int32())
        for inst, tokens in tokens_by_instrument.items()
    }

    for shard in shards:
        try:
            table = pq.read_table(shard)
        except Exception as e:
            print(f"  [WARN] Cannot read {shard}: {e}")
            continue

        table, drop_hb, drop_ts = filter_table(table)
        total_drop_hb += drop_hb
        total_drop_ts += drop_ts

        if table.num_rows == 0 or "token" not in table.schema.names:
            del table
            continue

        token_col = table.column("token")

        # Use pc.is_in for accurate token matching — no tokens dropped due to range limits
        for inst, inst_pa_set in inst_pa_sets.items():
            mask = pc.is_in(token_col, value_set=inst_pa_set)
            if not pc.any(mask).as_py():
                continue

            inst_table = table.filter(mask)
            if inst_table.num_rows == 0:
                del inst_table
                continue

            if inst not in writers:
                out_dir  = final_dir / inst / symbol
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"__tmp_{inst}_{symbol}.parquet"
                writers[inst]   = pq.ParquetWriter(
                    str(out_path), inst_table.schema, compression=COMPRESSION
                )
                out_paths[inst] = out_path

            writers[inst].write_table(inst_table)
            del inst_table

        del table, token_col
        _trim_malloc()

    for inst, writer in writers.items():
        writer.close()

    # Rename tmp → final inside thread (each symbol has its own paths, no conflicts)
    final_paths = {}
    for inst, tmp_path in out_paths.items():
        final_name = f"{date}_{inst}_{symbol}_{expiry}.parquet"
        final_path = tmp_path.parent / final_name
        tmp_path.rename(final_path)
        final_paths[inst] = final_path

    del writers, out_paths, inst_pa_sets
    gc.collect()
    _trim_malloc()

    return final_paths, total_drop_hb, total_drop_ts


def _merge_one_expiry_symbol(
    expiry_dir: Path,
    final_dir: Path,
    date: str,
    tokens_by_instrument: dict,
) -> tuple:
    expiry      = int(expiry_dir.name)
    symbol_dirs = sorted([p for p in expiry_dir.iterdir() if p.is_dir()])
    result      = {inst: {} for inst in INSTRUMENTS}
    total_drop_hb = 0
    total_drop_ts = 0

    symbol_dirs_with_shards = [d for d in symbol_dirs if any(d.glob("*.parquet"))]
    total_syms = len(symbol_dirs_with_shards)
    print(f"[MERGE]  expiry={expiry}  symbol_dirs={total_syms}")

    with ThreadPoolExecutor(max_workers=MERGE_WORKERS) as ex:
        futures = {
            ex.submit(
                _merge_one_symbol_streaming,
                sym_dir, final_dir, date, expiry, tokens_by_instrument,
            ): sym_dir
            for sym_dir in symbol_dirs_with_shards
        }
        completed_count = 0
        for future in as_completed(futures):
            sym_dir = futures[future]
            final_paths, drop_hb, drop_ts = future.result()
            total_drop_hb += drop_hb
            total_drop_ts += drop_ts
            for inst, path in final_paths.items():
                result[inst][sym_dir.name] = path
            completed_count += 1
            if completed_count % 50 == 0:
                print(f"  [PROGRESS] expiry={expiry}  {completed_count}/{total_syms} symbols done")

    gc.collect()

    print(f"[MERGE]  expiry={expiry} done  "
          f"dropped_heartbeat={total_drop_hb:,}  "
          f"dropped_test={total_drop_ts:,}")

    return expiry, result


def merge_expiry_shards(
    shard_dir: Path,
    final_dir: Path,
    date: str,
    tokens_by_instrument: dict,
) -> dict:
    expiry_dirs = sorted(
        [p for p in shard_dir.iterdir() if p.is_dir()],
        key=lambda p: int(p.name),
    )

    shard_counts = {p: sum(1 for _ in p.rglob("*.parquet")) for p in expiry_dirs}
    expiry_dirs_sorted = sorted(expiry_dirs, key=lambda p: shard_counts[p])

    total       = len(expiry_dirs_sorted)
    final_files = {inst: {} for inst in INSTRUMENTS}

    print(f"[MERGE]  {total} expiries  (parallel symbols, MERGE_WORKERS={MERGE_WORKERS})")
    for i, expiry_dir in enumerate(expiry_dirs_sorted, 1):
        print(f"\n[MERGE]  [{i}/{total}] expiry={expiry_dir.name}  "
              f"shards={shard_counts[expiry_dir]}")
        try:
            expiry, result = _merge_one_expiry_symbol(
                expiry_dir, final_dir, date,
                tokens_by_instrument,
            )
            for inst in INSTRUMENTS:
                final_files[inst][expiry] = result.get(inst, {})
        except Exception:
            import traceback
            print(f"\n[ERROR] Merge failed for expiry {expiry_dir.name}!")
            print(traceback.format_exc())
            raise

        _trim_malloc()

    return final_files


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def run_sanity_checks(final_files: dict, row_count_csv: Path, output_root: Path, date: str) -> bool:
    print(f"\n{'=' * 65}")
    print("[SANITY] Running post-write sanity checks ...")
    print(f"{'=' * 65}")

    all_passed    = True
    parquet_total = 0
    csv_rows      = []

    row_count_cache: dict[tuple, int] = {}

    for inst in sorted(INSTRUMENTS):
        for expiry, symbols in sorted(final_files[inst].items()):
            for symbol, path in sorted(symbols.items()):
                print(f"\n  {inst} / {expiry} / {path.name}")
                magic_ok = False
                rows     = 0
                tok_ok   = False
                try:
                    with open(path, "rb") as f:
                        header = f.read(4)
                        f.seek(-4, 2)
                        footer = f.read(4)
                    magic_ok = (header == PARQUET_MAGIC and footer == PARQUET_MAGIC)
                    if not magic_ok:
                        print(f"  [FAIL] magic bytes  header={header}  footer={footer}")
                        all_passed = False
                    else:
                        print(f"  [PASS] magic bytes  PAR1 ok")

                    meta = pq.read_metadata(path)
                    rows = sum(
                        meta.row_group(i).num_rows
                        for i in range(meta.num_row_groups)
                    )
                    row_count_cache[(inst, expiry, symbol)] = rows

                    if rows == 0:
                        print(f"  [FAIL] row count = 0")
                        all_passed = False
                    else:
                        print(f"  [PASS] row count = {rows:,}")
                    parquet_total += rows

                    table      = pq.read_table(path, columns=["token"])
                    tokens_int = np.asarray(
                        table.column("token").to_numpy(zero_copy_only=False), dtype=np.int64
                    )
                    bad        = int(((tokens_int < 10_000) | (tokens_int > 999_999)).sum())
                    five_digit = int(((tokens_int >= 10_000)  & (tokens_int <= 99_999)).sum())
                    six_digit  = int(((tokens_int >= 100_000) & (tokens_int <= 999_999)).sum())
                    tok_ok     = (bad == 0)
                    if not tok_ok:
                        print(f"  [FAIL] token length: {bad:,} tokens out of range (not 5 or 6 digits)")
                        all_passed = False
                    else:
                        print(f"  [PASS] token length: {five_digit:,} x 5-digit  {six_digit:,} x 6-digit  (all valid)")
                    del table, tokens_int

                except Exception as e:
                    print(f"  [ERROR] {e}")
                    all_passed = False

                status = "PASS" if (magic_ok and rows > 0 and tok_ok) else "FAIL"
                csv_rows.append([
                    inst, expiry, symbol, path.name,
                    "PASS" if magic_ok else "FAIL",
                    rows,
                    "PASS" if tok_ok else "FAIL",
                    status,
                ])

    print(f"\n{'─' * 65}")
    print("  ROW COUNT SUMMARY PER INSTRUMENT")
    print(f"{'─' * 65}")
    for inst in sorted(INSTRUMENTS):
        inst_total = sum(
            count
            for (i, _e, _s), count in row_count_cache.items()
            if i == inst
        )
        print(f"  {inst:10s} : {inst_total:>15,} rows")
    print(f"  {'─' * 45}")
    print(f"  {'TOTAL':10s} : {parquet_total:>15,} rows  (FUTSTK+FUTIDX+OPTSTK+OPTIDX)")

    print(f"\n{'─' * 65}")
    print("  ROW COUNT CROSS-CHECK  (.cap parser CSV vs parquet)")
    print(f"{'─' * 65}")

    parser_total = 0
    if row_count_csv.exists():
        import csv as csv_mod
        try:
            with open(row_count_csv, "r") as f:
                reader = csv_mod.reader(f)
                for row in reader:
                    if not row:
                        continue
                    try:
                        parser_total += int(row[-1].strip())
                    except ValueError:
                        continue
            print(f"  Parser CSV         : {row_count_csv.name}")
            print(f"  Parser total rows  : {parser_total:>15,}  (messages read from .cap files)")
        except Exception as e:
            print(f"  [WARN] Could not read CSV: {e}")
    else:
        print(f"  [WARN] Parser CSV not found: {row_count_csv}")

    print(f"  Parquet total rows : {parquet_total:>15,}  (FUTSTK+FUTIDX+OPTSTK+OPTIDX)")

    if parser_total > 0:
        if parser_total == parquet_total:
            match_str = "YES"
            print(f"  Match              : YES -- no data lost in .cap -> parquet")
        else:
            match_str  = "NO"
            diff       = abs(parser_total - parquet_total)
            pct        = diff / parser_total * 100
            status_str = "MORE" if parquet_total > parser_total else "FEWER"
            print(f"  Match              : NO -- parquet has {diff:,} {status_str} rows ({pct:.4f}%)")
            print(f"  NOTE: small diff expected — heartbeat + TEST rows were dropped")
            all_passed = False
    else:
        match_str = "UNKNOWN"
        print(f"  Match              : UNKNOWN — parser CSV missing or empty")

    print(f"\n{'=' * 65}")
    print("[SANITY] ALL PASSED ✓" if all_passed else "[SANITY] SOME CHECKS FAILED — review above")
    print(f"{'=' * 65}")

    sanity_csv = output_root / f"sanity_check_{date}.csv"
    import csv as _csv
    try:
        with open(sanity_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["instrument", "expiry", "symbol", "filename",
                        "magic_bytes", "row_count", "token_check", "status"])
            for r in csv_rows:
                w.writerow(r)

            w.writerow([])
            w.writerow(["── ROW COUNT SUMMARY PER INSTRUMENT ──"])
            for inst in sorted(INSTRUMENTS):
                inst_total = sum(
                    count
                    for (i, _e, _s), count in row_count_cache.items()
                    if i == inst
                )
                w.writerow([inst, "", "", "", "", inst_total, "", ""])
            w.writerow(["TOTAL", "", "", "", "", parquet_total,
                        "", "PASS" if all_passed else "FAIL"])

            w.writerow([])
            w.writerow(["── ROW COUNT CROSS-CHECK ──"])
            w.writerow(["parser_csv",    row_count_csv.name])
            w.writerow(["parser_total",  "", "", "", "", parser_total,  "", ""])
            w.writerow(["parquet_total", "", "", "", "", parquet_total, "", ""])
            w.writerow(["match",         "", "", "", "", match_str,     "", ""])

        print(f"[SANITY] Results saved -> {sanity_csv}")
    except Exception as e:
        print(f"[WARN] Could not write sanity CSV: {e}")

    return all_passed


# ---------------------------------------------------------------------------
# Size report
# ---------------------------------------------------------------------------
def size_report(source_root: Path, final_files: dict, date: str):
    src_total = sum(
        f.stat().st_size for f in source_root.glob("*.cap.tar.xz")
    )

    print(f"\n{'=' * 65}")
    print("[SIZE REPORT]")
    print(f"  Input  .cap.tar.xz total : {src_total / 1e9:.3f} GB")
    print()

    out_total = 0
    for inst in sorted(INSTRUMENTS):
        inst_total = sum(
            p.stat().st_size
            for symbols in final_files[inst].values()
            for p in symbols.values()
        )
        out_total += inst_total
        num_expiries = len(final_files[inst])
        num_symbols  = sum(len(s) for s in final_files[inst].values())
        print(f"  {inst:10s} : {inst_total / 1e9:.3f} GB  "
              f"({num_expiries} expiries, {num_symbols} symbol files)")

    print(f"  {'─' * 50}")
    print(f"  TOTAL output : {out_total / 1e9:.3f} GB")
    ratio = out_total / src_total if src_total else 0
    print(f"  Ratio        : {ratio:.3f}  "
          f"({(1 - ratio) * 100:.1f}% smaller than source)")
    print(f"{'=' * 65}")

    futstk_snappy_dir = source_root / "parquet" / "extract" / "FUTSTK"
    if futstk_snappy_dir.exists():
        snappy_total = 0
        for f in futstk_snappy_dir.glob("*.parquet"):
            if not f.is_file():
                continue
            with open(f, "rb") as fh:
                fh.seek(-4, 2)
                footer = fh.read(4)
            if footer == PARQUET_MAGIC:
                snappy_total += f.stat().st_size

        futstk_zstd_total = sum(
            p.stat().st_size
            for symbols in final_files.get("FUTSTK", {}).values()
            for p in symbols.values()
        )

        if snappy_total > 0 and futstk_zstd_total > 0:
            saving = (1 - futstk_zstd_total / snappy_total) * 100
            print(f"\n{'=' * 65}")
            print("[FUTSTK SNAPPY vs ZSTD COMPARISON]")
            print(f"  Old FUTSTK snappy : {snappy_total / 1e9:.3f} GB")
            print(f"  New FUTSTK zstd   : {futstk_zstd_total / 1e9:.3f} GB")
            print(f"  Saving            : {saving:.1f}% smaller with zstd")
            print(f"{'=' * 65}")


# ---------------------------------------------------------------------------
# Copy support docs
# ---------------------------------------------------------------------------
def copy_support_docs(source_root: Path, output_root: Path):
    support_dir = output_root / "support_docs"
    support_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[SUPPORT_DOCS] Copying support files to {support_dir} ...")
    copied  = 0
    skipped = 0
    for src in source_root.iterdir():
        if not src.is_file():
            continue
        if src.name.endswith(".cap.tar.xz"):
            skipped += 1
            continue
        dst = support_dir / src.name
        try:
            shutil.copy2(src, dst)
            print(f"  [SUPPORT_DOCS] {src.name}  ({src.stat().st_size / 1e6:.2f} MB)")
            copied += 1
        except Exception as e:
            print(f"  [WARN] Could not copy {src.name}: {e}")
    print(f"[SUPPORT_DOCS] Done — {copied} files copied, {skipped} .cap.tar.xz skipped")


# ---------------------------------------------------------------------------
# Publish to shared — FUTSTK and FUTIDX only (OPTSTK and OPTIDX skipped)
# ---------------------------------------------------------------------------
def _publish_one(args):
    src, dst_dir = args
    src     = Path(src)
    dst     = dst_dir / src.name
    tmp_dst = dst_dir / (src.name + ".part")
    shutil.copy2(src, tmp_dst)
    os.replace(tmp_dst, dst)
    return src.name


def publish_to_nas(final_files: dict, row_count_csv: Path,
                   output_root: Path, date: str):
    print(f"\n[PUBLISH] Publishing to {output_root} ...")
    print(f"[PUBLISH] Skipping instruments on shared: {sorted(SHARED_SKIP_INSTRUMENTS)}")

    tasks = []
    for inst, expiry_symbols in final_files.items():
        if inst in SHARED_SKIP_INSTRUMENTS:
            print(f"  [PUBLISH] Skipping {inst} on shared path")
            continue
        for expiry, symbols in expiry_symbols.items():
            for symbol, path in symbols.items():
                out_dir = output_root / inst / symbol
                out_dir.mkdir(parents=True, exist_ok=True)
                tasks.append((str(path), out_dir))

    with ThreadPoolExecutor(max_workers=PUBLISH_WORKERS) as ex:
        for fname in ex.map(_publish_one, tasks):
            print(f"  [PUBLISH] {fname}")

    if row_count_csv.exists():
        dst_csv = output_root / row_count_csv.name
        shutil.copy2(row_count_csv, dst_csv)
        print(f"  [PUBLISH] {row_count_csv.name}")


# ---------------------------------------------------------------------------
# Publish to NAS — /mnt/historical_data/parsed_data
# All 4 instruments + sanity CSV + support_docs (no change)
# ---------------------------------------------------------------------------
def publish_to_parsed_nas(final_files: dict, row_count_csv: Path,
                           output_root: Path, date: str):
    nas_root = PARSED_DATA_NAS / date
    print(f"\n[NAS PUBLISH] Publishing to {nas_root} ...")

    nas_root.mkdir(parents=True, exist_ok=True)

    # ── Parquet files — all 4 instruments ─────────────────────────────────
    tasks = []
    for inst, expiry_symbols in final_files.items():
        for expiry, symbols in expiry_symbols.items():
            for symbol, path in symbols.items():
                out_dir = nas_root / inst / symbol
                out_dir.mkdir(parents=True, exist_ok=True)
                tasks.append((str(path), out_dir))

    with ThreadPoolExecutor(max_workers=PUBLISH_WORKERS) as ex:
        for fname in ex.map(_publish_one, tasks):
            print(f"  [NAS PUBLISH] {fname}")

    # ── Row count CSV ──────────────────────────────────────────────────────
    if row_count_csv.exists():
        dst_csv = nas_root / row_count_csv.name
        shutil.copy2(row_count_csv, dst_csv)
        print(f"  [NAS PUBLISH] {row_count_csv.name}")

    # ── Sanity check CSV ───────────────────────────────────────────────────
    sanity_csv = output_root / f"sanity_check_{date}.csv"
    if sanity_csv.exists():
        dst_sanity = nas_root / sanity_csv.name
        shutil.copy2(sanity_csv, dst_sanity)
        print(f"  [NAS PUBLISH] {sanity_csv.name}")
    else:
        print(f"  [WARN] sanity_check_{date}.csv not found at {sanity_csv} — skipping")

    # ── Support docs ───────────────────────────────────────────────────────
    support_src = output_root / "support_docs"
    if support_src.exists():
        support_dst = nas_root / "support_docs"
        support_dst.mkdir(parents=True, exist_ok=True)
        copied = 0
        for f in support_src.iterdir():
            if f.is_file():
                shutil.copy2(f, support_dst / f.name)
                print(f"  [NAS PUBLISH] support_docs/{f.name}")
                copied += 1
        print(f"  [NAS PUBLISH] support_docs/ — {copied} files copied")
    else:
        print(f"  [WARN] support_docs/ not found at {support_src} — skipping")

    print(f"[NAS PUBLISH] Done -> {nas_root}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: all 4 instruments, zstd, new directory structure"
    )
    parser.add_argument("--date", required=True,
                        help="Date YYYYMMDD, e.g. 20260320")
    parser.add_argument("--clean", action="store_true",
                        help="Force clean start, delete existing work dir")
    args = parser.parse_args()
    date = args.date

    source_root    = Path(f"/mnt/historical_data/tbt_data/raw_data/{date}")
    contract_csv   = source_root / f"fo_contract_stream_info_{date}.csv"
    output_root    = PARSED_DATA_ROOT / date
    local_work_dir = Path(f"/media/svipl/Data/allseg_{date}_work")

    dirs = {
        "work":    local_work_dir,
        "stage":   local_work_dir / "stage_archives",
        "extract": local_work_dir / "extracted_caps",
        "shard":   local_work_dir / "shards",
        "final":   local_work_dir / "final",
        "log":     local_work_dir / "logs",
    }

    print("=" * 65)
    print("[INFO] FULL PIPELINE — all 4 instruments, zstd")
    print(f"[INFO] DATE              = {date}")
    print(f"[INFO] INSTRUMENTS       = {sorted(INSTRUMENTS)}")
    print(f"[INFO] COMPRESSION       = {COMPRESSION}")
    print(f"[INFO] PARALLEL_ARCHIVES = {PARALLEL_ARCHIVES}")
    print(f"[INFO] SOURCE_ROOT       = {source_root}")
    print(f"[INFO] OUTPUT_ROOT       = {output_root}")
    print(f"[INFO] LOCAL_WORK        = {local_work_dir}")
    print(f"[INFO] SHARED_SKIP       = {sorted(SHARED_SKIP_INSTRUMENTS)}  (not pushed to shared)")
    print("=" * 65)

    prepare_local_dirs(dirs, clean_start=args.clean)
    output_root.mkdir(parents=True, exist_ok=True)

    (_, _, _, _, tokens_by_instrument) = load_segment_maps(contract_csv)

    source_archives = discover_cap_archives(source_root)
    row_count_csv   = dirs["log"] / f"allseg_row_counts_{date}.csv"
    total           = len(source_archives)

    completed_stems = get_completed_archives(dirs["shard"])
    if completed_stems:
        print(f"\n[RESUME] {len(completed_stems)} archives already done — skipping")

    remaining = [
        src for src in source_archives
        if cap_stem_from_name(src.name) not in completed_stems
    ]
    skipped = total - len(remaining)
    print(f"\n[PIPELINE] {total} total | {skipped} skipped | {len(remaining)} remaining")

    if not remaining:
        print("[PIPELINE] All archives done, skipping to merge...")
    else:
        worker_args = [
            (
                str(src),
                str(dirs["stage"]),
                str(dirs["extract"]),
                str(dirs["shard"]),
                str(row_count_csv),
                str(contract_csv),
            )
            for src in remaining
        ]
        completed_count = skipped
        with ProcessPoolExecutor(max_workers=PARALLEL_ARCHIVES) as ex:
            futures = {ex.submit(_process_one_archive_worker, arg): arg
                       for arg in worker_args}
            for future in as_completed(futures):
                archive_name = future.result()
                completed_count += 1
                print(
                    f"[DONE] {completed_count}/{total} {archive_name} | "
                    f"disk free: {free_disk_gb(dirs['work']):.1f} GB"
                )

    print(f"\n[MERGE] Merging shards by instrument / expiry / symbol ...")
    final_files = merge_expiry_shards(
        dirs["shard"], dirs["final"], date,
        tokens_by_instrument,
    )

    print(f"\n[OUTPUT SUMMARY]")
    for inst in sorted(INSTRUMENTS):
        print(f"  {inst}:")
        for expiry, symbols in sorted(final_files[inst].items()):
            for symbol, path in sorted(symbols.items()):
                size_mb = path.stat().st_size / 1e6
                print(f"    {path.name}  ({size_mb:.1f} MB)")

    passed = run_sanity_checks(final_files, row_count_csv, output_root, date)
    size_report(source_root, final_files, date)
    publish_to_nas(final_files, row_count_csv, output_root, date)
    copy_support_docs(source_root, output_root)

    # FIX: pass output_root so NAS publish can find sanity CSV + support_docs
    publish_to_parsed_nas(final_files, row_count_csv, output_root, date)

    print(f"\n[CLEANUP] Deleting local work dir: {local_work_dir}")
    shutil.rmtree(local_work_dir)

    if passed:
        print(f"\n[SUCCESS] {date} extraction complete.")
        print(f"[SHARED]  {output_root}")
        print(f"          +-- FUTSTK/ FUTIDX/  (OPTSTK & OPTIDX skipped)")
        print(f"          +-- support_docs/")
        print(f"          +-- sanity_check_{date}.csv")
        print(f"[NAS]     {PARSED_DATA_NAS / date}")
        print(f"          +-- FUTSTK/ FUTIDX/ OPTSTK/ OPTIDX/")
        print(f"          +-- support_docs/")
        print(f"          +-- sanity_check_{date}.csv")
    else:
        sys.exit("\n[FAILED] Sanity checks failed — review output above.")


if __name__ == "__main__":
    main()
