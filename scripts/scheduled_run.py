"""
Scheduled Pipeline Run
======================

Wrapper for automated/scheduled pipeline execution.
Calculates the date range for the last N days, runs the pipeline,
and ingests results into the database.

Usage:
    python scripts/scheduled_run.py --config config.yaml
    python scripts/scheduled_run.py --config config.yaml --days 32

Scheduling:
    Windows:  schtasks /create /tn "VanSuraksha" /tr "python scripts/scheduled_run.py" /sc weekly /d MON
    Linux:    0 2 */14 * * cd /app && python scripts/scheduled_run.py >> /var/log/vs.log 2>&1
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def compute_date_range(days_back: int = 16):
    """Compute date range for last N days of imagery."""
    end = datetime.utcnow()
    start = end - timedelta(days=days_back)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def update_config_dates(config_path: str, start: str, end: str, tmp_path: str):
    """Create a temporary config with updated date range."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["gee"]["date_range"] = [start, end]
    cfg["_scheduled_run"] = {
        "triggered_at": datetime.utcnow().isoformat(),
        "date_range": [start, end],
    }

    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return cfg


def run_pipeline(config_path: str, steps: list, run_name: str):
    """Execute the pipeline with tracker."""
    cmd = [
        sys.executable, "scripts/run_pipeline.py",
        "--config", config_path,
        "--run-name", run_name,
        "--steps",
    ] + steps

    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def run_db_ingest(config_path: str, geojson_dir: str):
    """Ingest any new GeoJSON outputs into the database."""
    geojson_files = list(Path(geojson_dir).glob("*.geojson"))
    if not geojson_files:
        log.info("No GeoJSON files to ingest.")
        return

    for gjf in geojson_files:
        log.info(f"Ingesting: {gjf.name}")
        cmd = [
            sys.executable, "scripts/db_sink.py",
            "--config", config_path,
            "--geojson", str(gjf),
        ]
        subprocess.run(cmd, capture_output=False)


def main():
    parser = argparse.ArgumentParser(
        description="Van Suraksha -- Scheduled Pipeline Run")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--days", type=int, default=16,
                        help="Number of days back to process (default: 16)")
    parser.add_argument("--steps", nargs="+",
                        default=["data", "preprocess", "detect", "viz"],
                        help="Pipeline steps to run")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip DB ingestion after pipeline")
    args = parser.parse_args()

    start_date, end_date = compute_date_range(args.days)
    log.info(f"Scheduled run: {start_date} -> {end_date}")

    # Create temporary config with updated dates
    tmp_config = "config_scheduled.yaml"
    cfg = update_config_dates(args.config, start_date, end_date, tmp_config)

    run_name = f"scheduled_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"

    # Run pipeline
    exitcode = run_pipeline(tmp_config, args.steps, run_name)

    # Ingest into DB
    if exitcode == 0 and not args.skip_ingest:
        out_dir = cfg.get("paths", {}).get("output_dir", "outputs")
        run_db_ingest(tmp_config, out_dir)

    # Cleanup temp config
    try:
        os.remove(tmp_config)
    except OSError:
        pass

    if exitcode != 0:
        log.error(f"Scheduled run failed (exit code {exitcode})")
        sys.exit(exitcode)

    log.info(f"[DONE] Scheduled run complete: {run_name}")


if __name__ == "__main__":
    main()
