"""
End-to-End Pipeline Runner
==========================
Convenience script to run the full pipeline step by step.
Now integrated with PipelineTracker for structured logging,
DB persistence, and auto-notifications.

Usage:
    python scripts/run_pipeline.py --config config.yaml --steps all
    python scripts/run_pipeline.py --config config.yaml --steps data
    python scripts/run_pipeline.py --config config.yaml --steps train detect viz
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
import argparse
import subprocess
import logging
from pathlib import Path

from src.monitoring.tracker import PipelineTracker
from src.monitoring.notifier import Notifier

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Always run subprocesses with the same Python interpreter that is running
# this script (respects .venv / conda / ROCm env automatically).
_PYTHON = sys.executable

# Resolve the correct PROJ data directory once and propagate it to every
# child process so PostgreSQL's older PROJ.db doesn't shadow it.
# Priority: rasterio's proj_data (version-matched to its embedded GDAL wheel)
#           → then pyproj's proj_dir as fallback.
def _build_child_env() -> dict:
    """Return os.environ copy with PROJ_DATA / PROJ_LIB pinned to the venv."""
    env = os.environ.copy()
    proj_dir = None
    try:
        import rasterio as _rt
        _rt_proj = os.path.join(os.path.dirname(_rt.__file__), "proj_data")
        if os.path.isdir(_rt_proj):
            proj_dir = _rt_proj
    except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")
    if proj_dir is None:
        try:
            import pyproj
            proj_dir = pyproj.datadir.get_data_dir()
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")
    if proj_dir:
        env["PROJ_DATA"] = proj_dir   # PROJ >= 9
        env["PROJ_LIB"]  = proj_dir   # PROJ <  9 compat alias
        log.debug(f"PROJ data dir pinned to: {proj_dir}")
    return env

_CHILD_ENV = _build_child_env()


def _cmd(args: list) -> list:
    """Replace a leading 'python' token with the current interpreter path."""
    if args and args[0] == "python":
        return [_PYTHON] + args[1:]
    return args


def run_step(step_name: str, cmd: list, cfg: dict, tracker: PipelineTracker):
    """Run a pipeline step as a subprocess with tracker integration."""
    cmd = _cmd(cmd)   # always use the active venv/conda Python
    log.info(f"\n{'='*60}")
    log.info(f"  STEP: {step_name.upper()}")
    log.info(f"{'='*60}")
    log.info(f"  CMD: {' '.join(cmd)}")

    tracker.begin_step(step_name, inputs={"cmd": " ".join(cmd)})

    result = subprocess.run(cmd, capture_output=False, env=_CHILD_ENV)
    if result.returncode != 0:
        exc = RuntimeError(f"Pipeline step '{step_name}' failed with code {result.returncode}")
        tracker.fail_step(exc)
        log.error(f"  [FAIL] Step failed: {step_name}")
        raise exc

    tracker.end_step(outputs={"returncode": 0})
    log.info(f"  [OK] Step complete: {step_name}")


def main():
    parser = argparse.ArgumentParser(description="Prithvi Forest Change Pipeline Runner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--steps", nargs="+", default=["all"],
        choices=["all", "data", "preprocess", "train_detect", "train_predict",
                 "train_eo2", "detect", "predict", "viz", "filter", "ingest",
                 "alerts", "beat_report"],
        help="Pipeline steps to run"
    )
    parser.add_argument("--test_mode", action="store_true",
                        help="Run in test mode (small data download)")
    parser.add_argument("--run-name", default=None,
                        help="Custom name for this pipeline run (default: auto)")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip logging to database")
    # ── Date / AOI overrides (take precedence over config.yaml) ──────────────
    parser.add_argument("--date-from", default=None, metavar="YYYY-MM-DD",
                        help="Override gee.date_range start (e.g. 2026-01-01)")
    parser.add_argument("--date-to",   default=None, metavar="YYYY-MM-DD",
                        help="Override gee.date_range end   (e.g. 2026-02-15)")
    parser.add_argument("--aoi",       default=None,
                        help="Override gee.aoi_asset (GEE asset path)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides into the live cfg dict so every downstream step sees them
    if args.aoi:
        cfg["gee"]["aoi_asset"] = args.aoi
        log.info(f"AOI override: {args.aoi}")
    if args.date_from or args.date_to:
        dr = cfg["gee"]["date_range"]
        cfg["gee"]["date_range"] = [
            args.date_from or dr[0],
            args.date_to   or dr[1],
        ]
        log.info(f"Date range override: {cfg['gee']['date_range']}")

    # Initialize tracker with optional notifier
    run_name = args.run_name or f"pipeline_{'_'.join(args.steps)}"
    notifier = Notifier.from_config(args.config)
    tracker = PipelineTracker(
        run_name=run_name,
        output_dir=cfg.get("monitoring", {}).get("run_log_dir", "outputs/pipeline_runs"),
        notifier=notifier,
    )

    run_all = "all" in args.steps
    steps = args.steps

    try:
        # -- Step 1: GEE Data Fetch --
        if run_all or "data" in steps:
            gee_project = cfg.get("gee", {}).get("gee_project", "van-suraksha-alert")
            cmd = ["python", "src/data/gee_fetch.py", "--config", args.config,
                   "--project", gee_project]
            if args.test_mode:
                cmd.append("--test")
            run_step("GEE Data Fetch", cmd, cfg, tracker)

        # -- Step 2: Preprocess --
        if run_all or "preprocess" in steps:
            run_step("Preprocessing", [
                "python", "src/data/preprocess.py", "--config", args.config
            ], cfg, tracker)

        # -- Step 3a: Train Change Detection --
        if run_all or "train_detect" in steps:
            run_step("Train: Change Detection", [
                "python", "src/train/train.py",
                "--config", args.config,
                "--task", "detect"
            ], cfg, tracker)

        # -- Step 3b: Train Prediction --
        if run_all or "train_predict" in steps:
            run_step("Train: Future Prediction", [
                "python", "src/train/train.py",
                "--config", args.config,
                "--task", "predict"
            ], cfg, tracker)

        # -- Step 3c: EO-2.0 Fine-tune (opt-in, never runs under 'all') --
        # Requires: pip install terratorch  +  config.yaml  use_eo2: true
        if "train_eo2" in steps:
            eo2_variant = cfg.get("model", {}).get("eo2_variant", "600M")
            run_step("Train: EO-2.0 Fine-tune", [
                "python", "src/train/train.py",
                "--config", args.config,
                "--task", "detect",
                # train.py reads use_eo2 from config; pass a reminder flag for logs
                "--tag", f"eo2_{eo2_variant}",
            ], cfg, tracker)

        # -- Step 4: Run Detection Inference --
        if run_all or "detect" in steps:
            import glob
            raw = cfg["paths"]["raw_dir"]
            tifs = sorted(glob.glob(os.path.join(raw, "hls_*.tif")))
            out = cfg["paths"]["output_dir"]
            ckpt = os.path.join(cfg["paths"]["checkpoint_dir"], "best_detect.pth")

            if tifs and Path(ckpt).exists():
                run_step("Change Detection Inference", [
                    "python", "src/inference/detect.py",
                    "--config", args.config,
                    "--checkpoint", ckpt,
                    "--input"] + tifs + [
                    "--output", os.path.join(out, "change_map_latest.tif")
                ], cfg, tracker)
            else:
                log.warning(f"Skipping detect -- no data/checkpoint found")

        # -- Step 5: Run Prediction Inference --
        if run_all or "predict" in steps:
            import glob
            raw = cfg["paths"]["raw_dir"]
            tifs = sorted(glob.glob(os.path.join(raw, "hls_*.tif")))
            out = cfg["paths"]["output_dir"]
            ckpt = os.path.join(cfg["paths"]["checkpoint_dir"], "best_predict.pth")

            if tifs and Path(ckpt).exists():
                run_step("Future Risk Prediction", [
                    "python", "src/inference/predict.py",
                    "--config", args.config,
                    "--checkpoint", ckpt,
                    "--input"] + tifs + [
                    "--output", os.path.join(out, "risk_map.tif")
                ], cfg, tracker)
            else:
                log.warning(f"Skipping predict -- no data/checkpoint found")

        # -- Step 6: Visualization --
        if run_all or "viz" in steps:
            run_step("Dashboard Generation", [
                "python", "src/viz/dashboard.py",
                "--config", args.config,
            ], cfg, tracker)

        # -- Step 7: Alert Filtering --
        if "filter" in steps:
            run_step("Alert Filtering", [
                "python", "scripts/filter_alerts.py",
                "--config", args.config,
            ], cfg, tracker)

        # -- Step 8: DB Ingestion --
        if "ingest" in steps:
            run_step("DB Ingestion", [
                "python", "scripts/db_sink.py",
                "--config", args.config,
                "--init",
            ], cfg, tracker)

        # -- Step 8b: Alert Generation (generate_alerts.py sweeps the date window) --
        # Iterates through every bi-weekly anchor date in the configured date range,
        # runs model inference + S1 filter + CuSum + zone risk, and writes alerts.
        if run_all or "alerts" in steps:
            date_range = cfg["gee"]["date_range"]
            date_from  = date_range[0]   # e.g. "2026-01-01"
            date_to    = date_range[1]   # e.g. "2026-02-15"
            aoi_asset  = cfg["gee"]["aoi_asset"]
            ckpt       = os.path.join(cfg["paths"]["checkpoint_dir"], "best_detect.pth")
            out_dir    = os.path.join(cfg["paths"]["output_dir"], "alerts")

            if not Path(ckpt).exists():
                log.warning(f"Alert generation skipped — checkpoint not found: {ckpt}")
            else:
                # Build a weekly anchor-date list across the window
                from datetime import date, timedelta
                import datetime as _dt
                d_from = _dt.date.fromisoformat(date_from)
                d_to   = _dt.date.fromisoformat(date_to)
                # Saturdays ~ every 7 days; for a 6-week window that gives ~6 chips
                anchors = []
                cur = d_from
                while cur <= d_to:
                    anchors.append(cur.isoformat())
                    cur += timedelta(days=14)   # bi-weekly composites
                if not anchors or anchors[-1] != d_to.isoformat():
                    anchors.append(d_to.isoformat())

                log.info(
                    f"Alert sweep: {aoi_asset}  "
                    f"{date_from} → {date_to}  ({len(anchors)} anchors)"
                )

                for anchor in anchors:
                    run_step(f"Alert Generation [{anchor}]", [
                        "python", "src/inference/generate_alerts.py",
                        "--config",     args.config,
                        "--checkpoint", ckpt,
                        "--date",       anchor,
                        "--out-dir",    out_dir,
                    ], cfg, tracker)

        # -- Step 9: Beat PDF Report --
        if "beat_report" in steps:
            beat_cmd = [
                "python", "scripts/beat_report.py",
                "--config", args.config,
                "--out-dir", os.path.join(
                    cfg.get("paths", {}).get("output_dir", "outputs"),
                    "reports"
                ),
            ]
            # Optional: restrict to a single beat, or set a custom look-back window
            if steps_opts := cfg.get("pipeline", {}).get("beat_report", {}):
                if days := steps_opts.get("days"):
                    beat_cmd += ["--days", str(days)]
                if beat := steps_opts.get("beat"):
                    beat_cmd += ["--beat", beat]
            run_step("Beat PDF Report", beat_cmd, cfg, tracker)

    except Exception as e:
        log.error(f"Pipeline failed: {e}")
    finally:
        # Always save report
        report_path = tracker.save_report()
        log.info(f"Run report: {report_path}")

        # Log to DB unless --no-db
        if not args.no_db:
            try:
                import psycopg
                db = cfg.get("database", {})
                conn = psycopg.connect(
                    host=db.get("host", "localhost"),
                    port=db.get("port", 5432),
                    dbname=db.get("dbname", "gis_projects"),
                    user=db.get("user", "postgres"),
                    password=os.environ.get("VS_DB_PASSWORD", db.get("password", "")),
                )
                tracker.log_to_db(conn)
                conn.close()
            except Exception as db_err:
                log.warning(f"DB logging skipped: {db_err}")

    summary = tracker.summary()
    status = summary["status"]
    if status == "success":
        log.info("\n[DONE] Pipeline complete!")
    else:
        log.error(f"\n[FAIL] Pipeline finished with {summary['steps_failed']} failed steps")
        sys.exit(1)


if __name__ == "__main__":
    main()
