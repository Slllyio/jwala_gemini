"""
run_backfill.py
===============
E-Netra V4 — Historical Backfill Runner.

Iterates over all Mar-Ki-Mahu beats in guna_beats.geojson and runs the
daemon in --dry-run mode for a list of known DW pass dates in Feb 2026.

Usage:
    # Dry-run (no DB write, no state update) — safe to run anytime:
    python scripts/run_backfill.py

    # Live run (writes to PostGIS, updates state DB):
    python scripts/run_backfill.py --live

    # Limit to one beat (for testing):
    python scripts/run_backfill.py --beat "Goumukh"

    # Custom date window:
    python scripts/run_backfill.py --dates 2026-02-01 2026-02-14 2026-02-22
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DAEMON = _PROJECT_ROOT / "scripts" / "guna_ews_daemon.py"
_BEATS  = _PROJECT_ROOT / "data" / "aoi" / "guna_beats.geojson"

# Known DW pass dates for Feb 2026 (Sentinel-1/DW revisit ≈ every 6 days)
FEB_2026_PASS_DATES = [
    "2026-02-01",
    "2026-02-05",
    "2026-02-09",
    "2026-02-12",
    "2026-02-14",
    "2026-02-17",
    "2026-02-20",
    "2026-02-22",
]


def load_beats(beat_filter: str | None = None) -> list[str]:
    with open(_BEATS, encoding="utf-8") as f:
        fc = json.load(f)
    beats = [
        feat["properties"]["Beat"]
        for feat in fc["features"]
        if feat.get("properties", {}).get("Beat")
    ]
    if beat_filter:
        beats = [b for b in beats if beat_filter.lower() in b.lower()]
    return sorted(set(beats))


def run_one(beat: str, date_str: str, dry_run: bool) -> int:
    cmd = [sys.executable, str(_DAEMON), "--beat", beat, "--date", date_str]
    if dry_run:
        cmd.append("--dry-run")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="E-Netra V4 Backfill Runner")
    parser.add_argument("--live",  action="store_true",
                        help="Live mode: write to DB and update state")
    parser.add_argument("--beat",  default=None,
                        help="Limit to beats matching this substring")
    parser.add_argument("--dates", nargs="+", default=FEB_2026_PASS_DATES,
                        metavar="YYYY-MM-DD",
                        help="Space-separated list of dates to process")
    args = parser.parse_args()

    dry_run = not args.live
    beats   = load_beats(args.beat)
    dates   = args.dates

    mode_tag = "DRY-RUN" if dry_run else "LIVE"
    print(f"E-Netra V4 Backfill — {mode_tag}")
    print(f"  Beats : {len(beats)}")
    print(f"  Dates : {dates}")
    print(f"  Total : {len(beats) * len(dates)} daemon calls")
    print()

    stats = {"ok": 0, "fail": 0, "skip": 0}
    for i, beat in enumerate(beats, 1):
        for date_str in dates:
            print(f"  [{i:3d}/{len(beats)}] {beat:<25s} | {date_str}", end="  ", flush=True)
            rc = run_one(beat, date_str, dry_run)
            if rc == 0:
                stats["ok"] += 1
                print("OK")
            else:
                stats["fail"] += 1
                print(f"FAILED (rc={rc})")

    print()
    print("═" * 60)
    print(f"Backfill complete:  OK={stats['ok']}  FAIL={stats['fail']}")
    print("═" * 60)

    if stats["fail"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
