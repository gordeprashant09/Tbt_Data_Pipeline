#!/usr/bin/env python3
"""
allseg_extract_combined.py
==========================
Combined FO + CM extraction pipeline.

Runs both FO and CM pipelines in parallel as separate processes.
Each segment is independent — if one fails, the other continues.

Usage:
  python allseg_extract_combined.py --date 20260327              # run both
  python allseg_extract_combined.py --date 20260327 --segment FO # run FO only
  python allseg_extract_combined.py --date 20260327 --segment CM # run CM only
  python allseg_extract_combined.py --date 20260327 --clean      # force clean restart

Output:
  shared/DATE/
    FUTSTK/, FUTIDX/, CM_Data/
    sanity_check_DATE.csv
    sanity_check_cm_DATE.csv
    support_docs/

  NAS parsed_data/DATE/
    DATE_parsed.tar
    DATE_cm_parsed.tar
    sanity_check_DATE.csv
    sanity_check_cm_DATE.csv

Status file:
  /home/alpha/allseg_DATE_work/status.json
  {"FO": "done", "CM": "failed"} etc
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

PYTHON        = sys.executable
SCRIPT_DIR    = Path("/media/svipl/Data/historical_data")
FO_SCRIPT     = SCRIPT_DIR / "allseg_extract_full_tar_alpha_final.py"
CM_SCRIPT     = SCRIPT_DIR / "allseg_extract_cm_final.py"
LOG_DIR       = Path("/media/svipl/Data/logs")
WORK_BASE     = Path("/home/alpha")


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_status(status_file: Path) -> dict:
    if status_file.exists():
        try:
            with open(status_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def write_status(status_file: Path, status: dict):
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Combined FO+CM extraction pipeline")
    parser.add_argument("--date",    required=True, help="Date YYYYMMDD")
    parser.add_argument("--segment", choices=["FO", "CM"], default=None,
                        help="Run only FO or CM (default: both)")
    parser.add_argument("--clean",   action="store_true",
                        help="Force clean restart — ignore existing status/shards")
    args = parser.parse_args()
    date = args.date

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    work_dir    = WORK_BASE / f"allseg_{date}_work"
    status_file = work_dir / "status.json"

    # ── Determine which segments to run ──────────────────────────
    status = {} if args.clean else read_status(status_file)

    if args.segment:
        # explicit --segment flag
        segments_to_run = [args.segment]
    else:
        # run both — skip already done ones unless --clean
        segments_to_run = []
        for seg in ["FO", "CM"]:
            if status.get(seg) == "done" and not args.clean:
                log(f"[SKIP] {seg} already done (status.json) — use --clean to rerun")
            else:
                segments_to_run.append(seg)

    if not segments_to_run:
        log(f"[INFO] Nothing to run — both FO and CM already done for {date}")
        log(f"[INFO] Use --clean to force rerun")
        sys.exit(0)

    log("=" * 65)
    log(f"  COMBINED PIPELINE — {date}")
    log(f"  Segments to run  : {segments_to_run}")
    log(f"  FO script        : {FO_SCRIPT}")
    log(f"  CM script        : {CM_SCRIPT}")
    log(f"  Work dir         : {work_dir}")
    log(f"  Clean start      : {args.clean}")
    log("=" * 65)

    # ── Validate scripts exist ────────────────────────────────────
    for seg, script in [("FO", FO_SCRIPT), ("CM", CM_SCRIPT)]:
        if seg in segments_to_run and not script.exists():
            log(f"[ERROR] Script not found: {script}")
            sys.exit(1)

    # ── Launch processes ──────────────────────────────────────────
    processes = {}
    log_files = {}

    for seg in segments_to_run:
        script    = FO_SCRIPT if seg == "FO" else CM_SCRIPT
        log_path  = LOG_DIR / f"{seg.lower()}_{date}.log"
        cmd       = [PYTHON, "-u", str(script), "--date", date]
        if args.clean and seg == "FO":
            cmd.append("--clean")

        log(f"[LAUNCH] {seg} → {log_path}")
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes[seg]  = proc
        log_files[seg]  = log_file
        log(f"[LAUNCH] {seg} PID={proc.pid}")

    if len(segments_to_run) > 1:
        log(f"\n[INFO] Both processes running in parallel")
        log(f"[INFO] FO log : {LOG_DIR}/fo_{date}.log")
        log(f"[INFO] CM log : {LOG_DIR}/cm_{date}.log")
        log(f"[INFO] Waiting for both to complete ...\n")
    else:
        seg = segments_to_run[0]
        log(f"\n[INFO] Log : {LOG_DIR}/{seg.lower()}_{date}.log")
        log(f"[INFO] Waiting for {seg} to complete ...\n")

    # ── Wait + poll both processes ────────────────────────────────
    t_start   = time.time()
    results   = {}
    remaining = dict(processes)

    while remaining:
        for seg, proc in list(remaining.items()):
            ret = proc.poll()
            if ret is not None:
                elapsed = round(time.time() - t_start)
                if ret == 0:
                    results[seg] = "done"
                    log(f"[DONE] {seg} finished successfully in {elapsed}s ✅")
                else:
                    results[seg] = "failed"
                    log(f"[FAIL] {seg} FAILED (exit code {ret}) after {elapsed}s ❌")
                    log(f"[FAIL] Check log: {LOG_DIR}/{seg.lower()}_{date}.log")
                del remaining[seg]
        if remaining:
            time.sleep(10)

    # ── Close log files ───────────────────────────────────────────
    for f in log_files.values():
        f.close()

    # ── Update status file ────────────────────────────────────────
    status.update(results)
    write_status(status_file, status)
    log(f"\n[STATUS] Written to {status_file}")
    log(f"[STATUS] {json.dumps(status, indent=2)}")

    # ── Final summary ─────────────────────────────────────────────
    total = round(time.time() - t_start)
    log("\n" + "=" * 65)
    log(f"  COMBINED PIPELINE COMPLETE — {date}  (total: {total}s)")
    log("=" * 65)

    all_done   = all(v == "done"   for v in results.values())
    any_failed = any(v == "failed" for v in results.values())

    if all_done:
        log(f"[SUCCESS] All segments completed successfully ✅")
    if any_failed:
        failed = [s for s, v in results.items() if v == "failed"]
        log(f"[FAILED] Segments failed: {failed}")
        log(f"[RETRY]  Rerun with: python allseg_extract_combined.py --date {date} --segment {failed[0]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
