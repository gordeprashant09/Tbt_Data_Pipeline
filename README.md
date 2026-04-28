# NSE F&O Historical Data Pipeline

Full end-to-end pipeline that reads raw NSE market-capture (`.cap`) files, parses them into structured Parquet files, and organises the output by **instrument → symbol → expiry**.

---

## Project Files

| File | Role |
|---|---|
| `mtbt_parser_fast_up.py` | Low-level parser — reads binary `.cap` records, decodes each message, writes shard Parquet files |
| `allseg_date_inst_symbol_svipl.py` | Full pipeline orchestrator — extract → parse → merge → sanity-check → publish to **local + NAS** |
| `allseg_date_inst_symbol_fast_svipl.py` | Same as above but with higher parallelism; publishes to **local only** (no NAS mirror) |

---

## Architecture Overview

```
NAS: /mnt/historical_data/{date}/
        *.cap.tar.xz  ──┐
        fo_contract_stream_info_{date}.csv
                        │
          ┌─────────────▼──────────────┐
          │   allseg_date_inst_symbol  │   (orchestrator)
          │   _svipl.py  /  _fast_     │
          └──────┬──────────────┬──────┘
                 │              │
         8 parallel        contract CSV
         workers           loaded once
                 │
    ┌────────────▼────────────┐
    │  _process_one_archive   │  per .cap.tar.xz
    │  1. xz extract (3T)     │
    │  2. mtbt_parser_fast_up │  ← subprocess
    │  3. delete extracted    │
    └────────────┬────────────┘
                 │  shards/
                 │  {expiry}/{symbol}/{archive}_b{batch}.parquet
                 │
    ┌────────────▼────────────┐
    │   merge_expiry_shards   │  sequential, streaming
    │   filter heartbeat + TEST│
    │   split by instrument   │
    └────────────┬────────────┘
                 │  final/
                 │  {inst}/{symbol}/{date}_{inst}_{symbol}_{expiry}.parquet
                 │
    ┌────────────▼────────────┐
    │  sanity_checks          │
    │  size_report            │
    │  publish_to_nas         │
    │  copy_support_docs      │
    └─────────────────────────┘
```

---

## Pipeline Steps in Detail

### Step 1 — Archive Extraction
Each `.cap.tar.xz` archive is extracted in parallel (8 workers, 3 xz threads each) directly from the NAS mount — no intermediate copy step.

### Step 2 — Binary Parsing (`mtbt_parser_fast_up.py`)
Reads the raw `.cap` file in 1 million-record chunks. Each 72-byte record is decoded via a NumPy structured dtype. Outputs shard Parquet files keyed by `{expiry}/{symbol}/`.

**Message types decoded:**

| Code | Type |
|---|---|
| `N` | New Order |
| `M` | Modify Order |
| `X` | Cancel Order |
| `T` | Trade |
| `C` | Trade Cancel |
| `G/H/J` | Spread New/Mod/Cancel |
| `K` | Spread Trade |
| `Z` | Heartbeat |

**Parquet schema per shard:**

| Column | Type | Description |
|---|---|---|
| `capture_ns` | int64 | Capture timestamp (nanoseconds) |
| `exchange_ns` | int64 | Exchange timestamp (nanoseconds, IST) |
| `stream_id` | int16 | Feed stream identifier |
| `seq_no` | int32 | Message sequence number |
| `msg_type` | string | Single character: N/M/X/T/C/G/H/J/K/Z |
| `token` | int32 | NSE instrument token |
| `side` | string | `B` (buy) / `S` (sell) — orders only |
| `order_id` | float64 | Order identifier — orders only |
| `buy_order_id` | float64 | Buyer order ID — trades only |
| `sell_order_id` | float64 | Seller order ID — trades only |
| `price` | int32 | Price in paise |
| `quantity` | int32 | Quantity |

### Step 3 — Merge Shards
Shards are merged **sequentially per expiry, streaming per symbol** to keep memory bounded. During merge:
- **Heartbeat rows** (`msg_type = Z`) are **dropped** (counted, not saved)
- **TEST symbol rows** (`symbol` starts with `TEST`) are **dropped** (counted, not saved)
- Rows are split into 4 instrument buckets by token lookup

### Step 4 — Sanity Checks
For every output Parquet file the pipeline verifies:
- Magic bytes `PAR1` at header and footer
- Row count > 0
- All token values are 5 or 6 digits (10,000 – 999,999)
- Cross-check: total parquet rows == total rows logged by parser CSV

Results are written to `sanity_check_{date}.csv`.

### Step 5 — Publish
Files are copied atomically (`.part` → rename) to the output root. Support docs (contract CSV, etc.) are copied to `support_docs/`. The `_svipl` version also mirrors to the NAS at `/mnt/historical_data/parsed_data/`.

### Step 6 — Cleanup
The entire local work directory is deleted after successful publish.

---

## Output Structure

```
/media/svipl/Data/shared/{date}/
├── FUTSTK/
│   └── {SYMBOL}/
│       └── {date}_FUTSTK_{SYMBOL}_{expiry}.parquet
├── FUTIDX/
│   └── {SYMBOL}/
│       └── {date}_FUTIDX_{SYMBOL}_{expiry}.parquet
├── OPTSTK/
│   └── {SYMBOL}/
│       └── {date}_OPTSTK_{SYMBOL}_{expiry}.parquet
├── OPTIDX/
│   └── {SYMBOL}/
│       └── {date}_OPTIDX_{SYMBOL}_{expiry}.parquet
├── support_docs/
│   └── fo_contract_stream_info_{date}.csv  (+ other non-.cap files)
├── allseg_row_counts_{date}.csv
└── sanity_check_{date}.csv
```

NAS mirror (`_svipl` only):
```
/mnt/historical_data/parsed_data/{date}/
└── (same FUTSTK / FUTIDX / OPTSTK / OPTIDX structure)
```

---

## Script Comparison

| Setting | `_svipl.py` | `_fast_svipl.py` |
|---|---|---|
| `PARALLEL_ARCHIVES` | 8 | 8 |
| `MERGE_WORKERS` | 3 | 8 |
| `PUBLISH_WORKERS` | 16 | 16 |
| NAS mirror publish | ✅ Yes (`/mnt/historical_data/parsed_data`) | ❌ No |
| Heartbeat rows | Dropped | Dropped |
| TEST rows | Dropped | Dropped |
| Sanity CSV | ✅ Yes | ✅ Yes |
| Support docs copy | ✅ Yes | ✅ Yes |

---

## Usage

```bash
# Standard run
python -u allseg_date_inst_symbol_svipl.py --date 20260320

# Force clean start (deletes existing work dir first)
python -u allseg_date_inst_symbol_svipl.py --date 20260320 --clean

# Faster merge, local output only
python -u allseg_date_inst_symbol_fast_svipl.py --date 20260320
```

The parser is called automatically as a subprocess — do not run it directly unless debugging a single `.cap` file:

```bash
python mtbt_parser_fast_up.py \
    <input.cap> \
    <fo_contract_stream_info_{date}.csv> \
    <shard_root/> \
    <archive_stem> \
    <row_count.csv>
```

---

## Prerequisites

```bash
pip install pyarrow numpy
```

- Python 3.10+
- `xz` available in PATH
- Source archives at `/mnt/historical_data/{date}/*.cap.tar.xz`
- Contract CSV at `/mnt/historical_data/{date}/fo_contract_stream_info_{date}.csv`
- Local work disk: `/media/svipl/Data/` with sufficient free space (~2× raw input size)

---

## Key Design Decisions

**Vectorized parsing** — the entire `.cap` chunk is read into a single NumPy structured array and decoded with array operations; no per-record Python loops.

**Token lookup array** — a pre-built `token_gid_map[1_000_000]` int32 array replaces `np.isin()` in the hot path, reducing per-batch overhead from O(N × instruments) to O(N).

**Sequential merge** — expiries are merged one at a time to prevent Arrow off-heap buffers accumulating across threads. `malloc_trim(0)` is called after each symbol to return freed pages to the OS.

**Atomic publish** — each file is written as `.part` then renamed, so a partial run never leaves a corrupt file at the destination.

**Resume support** — completed archive stems are detected from existing shard files; interrupted runs automatically skip already-processed archives.

---

## Instruments Covered

| Instrument | Description |
|---|---|
| `FUTSTK` | Single-stock futures |
| `FUTIDX` | Index futures (NIFTY, BANKNIFTY, etc.) |
| `OPTSTK` | Single-stock options |
| `OPTIDX` | Index options |
