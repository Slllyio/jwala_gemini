"""
test_all_beats.py — Run the GEE-SRC rules engine across all 139 Guna beats.

Usage:
    python scripts/test_all_beats.py [--date DATE] [--range RANGE_NAME]
                                     [--limit N] [--s1-drop DB] [--out CSV]

Outputs:
    outputs/beat_test_results.csv   — per-beat signal readings + alert decision
    outputs/beat_test_summary.txt   — human-readable summary
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, date
from pathlib import Path

# ── project root so we can import src.* if needed ──────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_DATE       = "2026-02-23"        # anchor = today
BEATS_GEOJSON      = ROOT / "data" / "aoi" / "guna_beats.geojson"
DEFAULT_OUT_CSV    = ROOT / "outputs" / "beat_test_results.csv"
DEFAULT_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "best_detect.pth"
DEFAULT_CONFIG     = ROOT / "config.yaml"

# Regex to parse the [INFO] Signals line from generate_alerts output
_SIGNAL_RE = re.compile(
    r"Signals:\s*(\{[^\n]+\})"
)
_ALERT_AREA_RE  = re.compile(r"Alert area.*?([\d.]+)\s*ha")
_NO_ALERT_RE    = re.compile(r"No alert.*?area\s+([\d.]+)\s*ha")
_ALERT_FIRE_RE  = re.compile(r"\[rules\] Alert generated|Alert sent|ALERT")
_ENGINE_ERR_RE  = re.compile(r"\[ERROR\].*(?:Engine error|GEE error):(.+)")


def load_beats(geojson_path: Path, range_filter: str | None = None) -> list[dict]:
    """Return list of {range, beat, area_ha, geometry} dicts."""
    with open(geojson_path) as f:
        gj = json.load(f)

    beats = []
    for feat in gj["features"]:
        p = feat["properties"]
        range_name = (p.get("Range") or "").strip()
        beat_name  = (p.get("Beat")  or "").strip()
        area_ha    = float(p.get("Beat_Ar") or 0)
        if range_filter and range_name.lower() != range_filter.lower():
            continue
        beats.append({
            "range":    range_name,
            "beat":     beat_name,
            "area_ha":  area_ha,
            "beat_key": f"{range_name}/{beat_name}",
        })
    return beats


def run_beat(
    beat_key: str,
    date: str,
    config: Path,
    checkpoint: Path,
    s1_drop: float,
    out_dir: Path,
    dry_run: bool = True,
) -> dict:
    """Run generate_alerts.py for one beat. Returns a dict of parsed results."""
    cmd = [
        sys.executable, "-m", "src.inference.generate_alerts",
        "--config",     str(config),
        "--checkpoint", str(checkpoint),
        "--date",       date,
        "--beat",       beat_key,
        "--out-dir",    str(out_dir / "beat_runs"),
        "--s1-drop",    str(s1_drop),
    ]
    if dry_run:
        cmd.append("--dry-run")

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=300,          # 5-min GEE timeout per beat
        )
        elapsed = time.monotonic() - t0
        output  = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return {
            "status": "TIMEOUT",
            "elapsed_s": 300,
            "error": "subprocess timeout",
        }
    except Exception as exc:
        return {
            "status": "ERROR",
            "elapsed_s": 0,
            "error": str(exc),
        }

    # ── parse signal values ──────────────────────────────────────────────
    signals: dict = {}
    m = _SIGNAL_RE.search(output)
    if m:
        try:
            signals = json.loads(m.group(1).replace("None", "null"))
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")

    # ── parse alert status ───────────────────────────────────────────────
    if _ALERT_FIRE_RE.search(output):
        alert_status = "ALERT"
        # try to grab area
        am = _ALERT_AREA_RE.search(output)
        flagged_ha = float(am.group(1)) if am else None
    else:
        nam = _NO_ALERT_RE.search(output)
        flagged_ha = float(nam.group(1)) if nam else 0.0
        alert_status = "NO_ALERT" if result.returncode != 0 else "CLEAN"

    # ── parse errors ─────────────────────────────────────────────────────
    err_match = _ENGINE_ERR_RE.search(output)
    error_msg = err_match.group(1).strip() if err_match else ""
    if error_msg:
        alert_status = "GEE_ERROR"

    return {
        "status":       alert_status,
        "flagged_ha":   flagged_ha,
        "elapsed_s":    round(elapsed, 1),
        "error":        error_msg,
        "dNDVI":        signals.get("dNDVI_mean"),
        "dNBR":         signals.get("dNBR_mean"),
        "dVH_db":       signals.get("dVH_mean_db"),
        "cusum":        signals.get("cusum_mean"),
        "dDW_trees":    signals.get("dDW_trees_mean"),
    }


def main():
    parser = argparse.ArgumentParser(description="Run rules engine on all Guna beats")
    parser.add_argument("--date",       default=DEFAULT_DATE,
                        help="Anchor date YYYY-MM-DD (default: today)")
    parser.add_argument("--range",      default=None,
                        help="Filter to a single range by name")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Stop after N beats (for quick testing)")
    parser.add_argument("--s1-drop",    type=float, default=2.0,
                        help="S1 VH drop threshold in dB (default: 2.0)")
    parser.add_argument("--out",        default=str(DEFAULT_OUT_CSV),
                        help="Output CSV path")
    parser.add_argument("--config",     default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--no-dry-run", action="store_true",
                        help="Actually write alerts to DB / send Telegram")
    args = parser.parse_args()

    beats = load_beats(BEATS_GEOJSON, range_filter=args.range)
    if not beats:
        print(f"[ERROR] No beats found (range filter: {args.range})")
        sys.exit(1)

    if args.limit:
        beats = beats[: args.limit]

    print(f"[test_all_beats] anchor={args.date}  beats={len(beats)}  s1_drop={args.s1_drop} dB")
    print(f"  Output -> {args.out}")
    print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "range", "beat", "beat_area_ha",
        "status", "flagged_ha", "elapsed_s",
        "dNDVI", "dNBR", "dVH_db", "cusum", "dDW_trees",
        "error",
    ]

    counts = {"ALERT": 0, "NO_ALERT": 0, "GEE_ERROR": 0, "TIMEOUT": 0, "ERROR": 0, "CLEAN": 0}
    out_dir = ROOT / "outputs"

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, beat in enumerate(beats, 1):
            print(f"[{i:3d}/{len(beats)}] {beat['beat_key']:<40}", end="", flush=True)

            result = run_beat(
                beat_key   = beat["beat_key"],
                date       = args.date,
                config     = Path(args.config),
                checkpoint = Path(args.checkpoint),
                s1_drop    = args.s1_drop,
                out_dir    = out_dir,
                dry_run    = not args.no_dry_run,
            )

            counts[result["status"]] = counts.get(result["status"], 0) + 1

            status_icon = {
                "ALERT":     "[ALERT]",
                "NO_ALERT":  "[OK]   ",
                "CLEAN":     "[OK]   ",
                "GEE_ERROR": "[ERR]  ",
                "TIMEOUT":   "[TOUT] ",
                "ERROR":     "[FAIL] ",
            }.get(result["status"], "?")

            sig_str = ""
            if result.get("dNDVI") is not None:
                sig_str = (
                    f"dNDVI={result['dNDVI']:+.3f}  "
                    f"dVH={result['dVH_db']:+.2f}dB  "
                    f"cusum={result['cusum']:.3f}"
                )
            print(f"  {status_icon} {result['status']:<10}  {sig_str}  ({result['elapsed_s']}s)")

            writer.writerow({
                "range":        beat["range"],
                "beat":         beat["beat"],
                "beat_area_ha": beat["area_ha"],
                "status":       result["status"],
                "flagged_ha":   result.get("flagged_ha", ""),
                "elapsed_s":    result["elapsed_s"],
                "dNDVI":        result.get("dNDVI", ""),
                "dNBR":         result.get("dNBR", ""),
                "dVH_db":       result.get("dVH_db", ""),
                "cusum":        result.get("cusum", ""),
                "dDW_trees":    result.get("dDW_trees", ""),
                "error":        result.get("error", ""),
            })
            csvfile.flush()

    # ── summary ─────────────────────────────────────────────────────────────
    total = len(beats)
    summary_lines = [
        "",
        "=" * 60,
        f"  GUNA ALL-BEATS TEST RESULTS  |  anchor: {args.date}",
        "=" * 60,
        f"  Total beats run : {total}",
        f"  [ALERT]  ALERT        : {counts.get('ALERT', 0)}",
        f"  [OK]     No alert     : {counts.get('NO_ALERT', 0) + counts.get('CLEAN', 0)}",
        f"  [ERR]    GEE errors   : {counts.get('GEE_ERROR', 0)}",
        f"  [TOUT]   Timeouts     : {counts.get('TIMEOUT', 0)}",
        f"  [FAIL]   Other errors : {counts.get('ERROR', 0)}",
        f"  Results CSV     : {out_path}",
        "=" * 60,
    ]
    summary = "\n".join(summary_lines)
    print(summary)

    summary_path = out_path.with_suffix(".summary.txt")
    summary_path.write_text(summary)
    print(f"  Summary -> {summary_path}")


if __name__ == "__main__":
    main()
