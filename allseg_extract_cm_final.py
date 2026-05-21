#!/usr/bin/env python3
"""
allseg_extract_cm.py
====================
CM (Cash Market) extraction pipeline — parallel version.

Processes efvi_YYYYMMDD_166xx.cap.tar.xz files (CM segment only).
Uses cm_contract_stream_info_YYYYMMDD.csv for token→symbol+series mapping.

Output structure:
  NAS parsed_data:
    /mnt/historical_data/tbt_data/parsed_data/YYYYMMDD/CM_Data/
      └── YYYYMMDD_BANDHANBNK_EQ.parquet
      └── YYYYMMDD_RELIANCE_EQ.parquet
      └── ...
  NAS tar bundle:
    /mnt/historical_data/tbt_data/parsed_data/YYYYMMDD/
      └── YYYYMMDD_cm_parsed.tar
  Shared folder:
    /mnt/historical_data/tbt_data/parsed_data/YYYYMMDD/CM_Data/
      └── sanity_check_cm_YYYYMMDD.csv

Usage:
  python allseg_extract_cm.py --date 20260327
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

PARSER_SCRIPT     = Path("/media/svipl/Data/historical_data/mtbt_parser_cm.py")
PYTHON            = sys.executable

# NAS paths
NAS_RAW_ROOT      = Path("/mnt/historical_data/tbt_data/raw_data")
NAS_PARSED_ROOT   = Path("/mnt/historical_data/tbt_data/parsed_data")

# Shared folder (local) — same as FO pipeline
SHARED_ROOT       = Path("/media/svipl/Data/shared")

# Local work dir (fast SSD)
LOCAL_WORK_BASE   = Path("/media/svipl/Data")

# Parallel archives to extract simultaneously
PARALLEL_ARCHIVES = 8

# CM archive pattern — only _166xx files
CM_ARCHIVE_PATTERN = "*_166*.cap.tar.xz"

# XZ threads per archive
XZ_THREADS_PER_ARCHIVE = 3


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cmd(cmd: list, check: bool = True, **kwargs):
    return subprocess.run(cmd, check=check, **kwargs)


def discover_cm_archives(source_root: Path) -> list[Path]:
    """Find all _166xx.cap.tar.xz files — CM segment only."""
    files = sorted(source_root.glob(CM_ARCHIVE_PATTERN))
    log(f"Found {len(files)} CM archives (_166xx) in {source_root}")
    return files


# ══════════════════════════════════════════════════════════════════
# WORKER — extract + parse one archive
# ══════════════════════════════════════════════════════════════════

def _process_one_archive(args):
    """
    Worker function for ProcessPoolExecutor.
    Extracts one .cap.tar.xz → parses → writes shards.
    Returns (archive_stem, row_count_csv_path, success)
    """
    src, stage_dir, extract_dir, shard_dir, row_count_csv, contract_csv = args

    archive_stem = src.stem.replace(".cap.tar", "").replace(".cap", "")
    extract_subdir = extract_dir / archive_stem
    extract_subdir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract .cap.tar.xz → .cap
    try:
        subprocess.run(
            [
                "tar",
                f"--use-compress-program=xz -T{XZ_THREADS_PER_ARCHIVE} -d",
                "-xf", str(src),
                "-C", str(extract_subdir),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Extract failed: {src.name}: {e.stderr.decode()[:200]}", flush=True)
        return (archive_stem, None, False)

    # Step 2: Find .cap file
    cap_files = list(extract_subdir.glob("*.cap"))
    if not cap_files:
        print(f"[ERROR] No .cap file found after extract: {src.name}", flush=True)
        return (archive_stem, None, False)
    cap_path = cap_files[0]

    # Step 3: Parse .cap → shards
    try:
        subprocess.run(
            [
                PYTHON, str(PARSER_SCRIPT),
                str(cap_path),
                str(contract_csv),
                str(shard_dir),
                archive_stem,
                str(row_count_csv),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Parse failed: {src.name}", flush=True)
        return (archive_stem, row_count_csv, False)

    # Step 4: Cleanup .cap file
    try:
        shutil.rmtree(extract_subdir)
    except Exception:
        pass

    print(f"[DONE] {src.name}", flush=True)
    return (archive_stem, row_count_csv, True)


# ══════════════════════════════════════════════════════════════════
# MERGE SHARDS → FINAL PARQUET
# ══════════════════════════════════════════════════════════════════

def merge_shards_to_parquet(shard_dir: Path, final_dir: Path, date: str):
    """
    Merge per-archive shards into single parquet per symbol+series.
    Output: final_dir/YYYYMMDD_SYMBOL_SERIES.parquet
    """
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
    except ImportError:
        log("[ERROR] pyarrow not available for merge")
        return {}

    final_dir.mkdir(parents=True, exist_ok=True)
    row_counts = {}

    # Each subdir = one symbol_series
    symbol_dirs = [d for d in shard_dir.iterdir() if d.is_dir()]
    log(f"  Merging {len(symbol_dirs)} symbol+series groups...")

    for sym_dir in sorted(symbol_dirs):
        safe_name  = sym_dir.name  # e.g. BANDHANBNK_EQ
        shards     = sorted(sym_dir.glob("*.parquet"))
        if not shards:
            continue

        out_path = final_dir / f"{date}_{safe_name}.parquet"

        tables = []
        for shard in shards:
            try:
                tables.append(pq.read_table(str(shard)))
            except Exception as e:
                log(f"  [WARN] Failed to read shard {shard.name}: {e}")

        if not tables:
            continue

        merged = pa.concat_tables(tables)
        pq.write_table(merged, str(out_path), compression="snappy")
        row_counts[safe_name] = merged.num_rows

        if len(row_counts) % 100 == 0:
            log(f"  Merged {len(row_counts)}/{len(symbol_dirs)} symbols...")

    log(f"  Merge complete: {len(row_counts)} parquet files written")
    return row_counts


# ══════════════════════════════════════════════════════════════════
# SANITY CHECK CSV
# ══════════════════════════════════════════════════════════════════

def write_sanity_check(final_dir: Path, date: str, row_counts: dict):
    sanity_path = final_dir / f"sanity_check_cm_{date}.csv"
    with open(sanity_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["safe_name", "rows", "file_size_mb"])
        for safe_name, rows in sorted(row_counts.items()):
            parquet_path = final_dir / f"{date}_{safe_name}.parquet"
            size_mb = round(parquet_path.stat().st_size / (1024**2), 2) if parquet_path.exists() else 0
            w.writerow([safe_name, rows, size_mb])
    log(f"  Sanity check: {sanity_path}")
    return sanity_path


# ══════════════════════════════════════════════════════════════════
# NAS PUBLISH — tar bundle + copy to parsed_data
# ══════════════════════════════════════════════════════════════════

def publish_to_nas(final_dir: Path, date: str, nas_parsed_root: Path):
    """
    1. Copy parquets to shared/YYYYMMDD/CM_Data/  ← FIRST (fast, local SSD→SSD)
    2. Create YYYYMMDD_cm_parsed.tar from final_dir
    3. Copy tar + sanity_check to NAS parsed_data/YYYYMMDD/
    4. Delete local tar after NAS copy (saves disk space)
    """
    nas_out_dir = nas_parsed_root / date
    nas_out_dir.mkdir(parents=True, exist_ok=True)

    sanity_src = final_dir / f"sanity_check_cm_{date}.csv"

    # ── Step 1: Copy parquets to shared FIRST (before tar/NAS) ───────────
    shared_out_dir = SHARED_ROOT / date / "CM_Data"
    shared_out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[SHARED] Copying parquets to {shared_out_dir} ...")
    t0 = time.time()
    count = 0
    for parquet in sorted(final_dir.glob("*.parquet")):
        shutil.copy2(str(parquet), str(shared_out_dir / parquet.name))
        count += 1
    if sanity_src.exists():
        shutil.copy2(str(sanity_src), str(shared_out_dir / sanity_src.name))
    elapsed = round(time.time() - t0)
    log(f"[SHARED] Copied {count} parquet files in {elapsed}s")
    log(f"[SHARED] {shared_out_dir}")

    # ── Step 2: Create tar bundle ─────────────────────────────────────────
    tar_name = f"{date}_cm_parsed.tar"
    tar_path = final_dir.parent / tar_name

    log(f"[TAR] Creating {tar_name} ...")
    t0 = time.time()
    run_cmd([
        "tar", "-cf", str(tar_path),
        "-C", str(final_dir.parent),
        final_dir.name,
    ])
    elapsed = round(time.time() - t0)
    size_gb = tar_path.stat().st_size / (1024**3)
    log(f"[TAR] Created: {tar_name} ({size_gb:.2f} GB) in {elapsed}s")

    # ── Step 3: Copy tar to NAS ───────────────────────────────────────────
    log(f"[NAS] Copying tar to {nas_out_dir} ...")
    t0 = time.time()
    tmp_nas = nas_out_dir / (tar_name + ".part")
    shutil.copy2(str(tar_path), str(tmp_nas))
    os.replace(str(tmp_nas), str(nas_out_dir / tar_name))
    elapsed = round(time.time() - t0)
    log(f"[NAS] Copied tar in {elapsed}s")

    # Copy sanity check to NAS
    if sanity_src.exists():
        shutil.copy2(str(sanity_src), str(nas_out_dir / sanity_src.name))
        log(f"[NAS] Copied sanity check")

    # ── Step 4: Delete local tar (no longer needed) ───────────────────────
    try:
        tar_path.unlink()
        log(f"[TAR] Local tar deleted: {tar_path.name}")
    except Exception as e:
        log(f"[WARN] Could not delete local tar: {e}")

    log(f"[NAS]    {nas_out_dir}")


# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CM data extraction pipeline")
    parser.add_argument("--date", required=True, help="Date in YYYYMMDD format")
    args = parser.parse_args()
    date = args.date

    t_start = time.time()
    log("=" * 65)
    log(f"  CM PIPELINE — {date}")
    log(f"  Archives pattern : {CM_ARCHIVE_PATTERN}")
    log(f"  Parallel workers : {PARALLEL_ARCHIVES}")
    log("=" * 65)

    # Paths
    source_root  = NAS_RAW_ROOT / date
    work_dir     = LOCAL_WORK_BASE / f"cm_{date}_work"
    extract_dir  = work_dir / "extracted_caps"
    shard_dir    = work_dir / "shards"
    final_dir    = work_dir / "CM_Data"
    row_count_csv = work_dir / "logs" / f"row_counts_cm_{date}.csv"

    for d in [extract_dir, shard_dir, final_dir, row_count_csv.parent]:
        d.mkdir(parents=True, exist_ok=True)

    # Contract file
    contract_csv = source_root / f"cm_contract_stream_info_{date}.csv"
    if not contract_csv.exists():
        log(f"[ERROR] CM contract not found: {contract_csv}")
        sys.exit(1)

    # Discover CM archives
    archives = discover_cm_archives(source_root)
    if not archives:
        log(f"[ERROR] No CM archives found in {source_root}")
        sys.exit(1)

    # Build worker args
    worker_args = [
        (src, work_dir, extract_dir, shard_dir, row_count_csv, contract_csv)
        for src in archives
    ]

    # Parallel extraction + parsing
    log(f"[EXTRACT+PARSE] Processing {len(archives)} CM archives ({PARALLEL_ARCHIVES} parallel)...")
    t0 = time.time()
    success_count = 0
    fail_count    = 0

    with ProcessPoolExecutor(max_workers=PARALLEL_ARCHIVES) as executor:
        futures = {executor.submit(_process_one_archive, arg): arg[0].name for arg in worker_args}
        for future in as_completed(futures):
            archive_name = futures[future]
            try:
                stem, _, ok = future.result()
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
                    log(f"[FAIL] {archive_name}")
            except Exception as e:
                fail_count += 1
                log(f"[ERROR] {archive_name}: {e}")

    elapsed = round(time.time() - t0)
    log(f"[EXTRACT+PARSE] Done in {elapsed}s — {success_count} OK, {fail_count} failed")

    # Merge shards → final parquets
    log("[MERGE] Merging shards into final parquets...")
    t0 = time.time()
    row_counts = merge_shards_to_parquet(shard_dir, final_dir, date)
    elapsed = round(time.time() - t0)
    log(f"[MERGE] Done in {elapsed}s — {len(row_counts)} parquet files")

    # Sanity check
    write_sanity_check(final_dir, date, row_counts)

    # Publish to NAS
    log("[NAS PUBLISH] Publishing to NAS...")
    publish_to_nas(final_dir, date, NAS_PARSED_ROOT)

    # Cleanup work dir
    log(f"[CLEANUP] Deleting work dir: {work_dir}")
    try:
        shutil.rmtree(work_dir)
    except Exception as e:
        log(f"[WARN] Cleanup failed: {e}")

    total = round(time.time() - t_start)
    log("=" * 65)
    log(f"[SUCCESS] CM extraction complete for {date} in {total}s")
    log(f"  NAS: {NAS_PARSED_ROOT / date / f'{date}_cm_parsed.tar'}")
    log("=" * 65)


if __name__ == "__main__":
    main()
