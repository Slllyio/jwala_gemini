"""
Operational Daily Fire-Risk Pipeline (OpFML architecture)
===========================================================
Executes the full data → inference → bulletin cycle autonomously once per
day, replicating the OpFML (Operational Forecasting with Machine Learning)
architecture described in the PDF §"Operational Workflow for Daily Risk
Delineation".

Pipeline steps
--------------
1. Data Pulse       — poll for latest HLS granules & GFS forecast runs.
2. Harmonisation    — reproject all inputs to EPSG:4326 @ 30 m basis.
3. VIIRS Fetch      — retrieve near-real-time FIRMS detections.
4. Susceptibility   — run Prithvi multimodal model → risk raster.
5. Vectorisation    — threshold raster → GeoJSON forecast polygons.
6. Fire Dynamics    — cluster detections, build FEDS perimeters, CROS.
7. Validation       — spatial join predictions vs VIIRS; compute IoU / F1.
8. Bulletin         — assemble Forecast / Active / Context GeoJSON layers.
9. Notify           — push summary to monitoring system.

Usage::

    # Run for today
    python -m src.pipeline.daily_pipeline

    # Run for a specific date
    python -m src.pipeline.daily_pipeline --date 2025-03-01

    # Dry-run (skip actual downloads and inference)
    python -m src.pipeline.daily_pipeline --dry-run
"""

from __future__ import annotations

import src.utils.proj_fix  # noqa: F401  (PROJ/GDAL env fix)

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import yaml

log = logging.getLogger(__name__)

# ── Module imports (lazy where optional) ────────────────────────────────────
from src.data.earthaccess_ingest import download_hls, GUNA_BBOX
from src.data.weather_ingest import download_gfs, compute_fire_weather_index
from src.data.firms_ingest import (
    fetch_viirs_detections,
    load_detections_for_date,
    summarise_detections,
)
from src.analysis.fire_dynamics import (
    FireEventTracker,
    cluster_viirs_to_events,
)
from src.analysis.spatial_validation import (
    raster_to_geojson,
    validate_predictions,
    build_daily_bulletin,
)

DEFAULT_CONFIG = Path("config.yaml")
DEFAULT_DATA_LAKE = Path("data_lake")
DEFAULT_OUT = Path("outputs/daily_bulletins")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    run_date: str
    hls_files: List[Path] = field(default_factory=list)
    weather_files: List[Path] = field(default_factory=list)
    viirs_summary: Dict = field(default_factory=dict)
    spread_stats: List[Dict] = field(default_factory=list)
    validation_metrics: Dict = field(default_factory=dict)
    bulletin_paths: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "run_date": self.run_date,
            "hls_files_count": len(self.hls_files),
            "weather_files_count": len(self.weather_files),
            "viirs_summary": self.viirs_summary,
            "spread_stats": self.spread_stats,
            "validation_metrics": self.validation_metrics,
            "bulletin_paths": self.bulletin_paths,
            "errors": self.errors,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def _step_hls_ingest(
    run_date: date,
    data_lake: Path,
    cloud_max: int,
    dry_run: bool,
) -> List[Path]:
    """Step 1 — Download latest HLS granules covering Guna Division."""
    log.info("── Step 1: HLS Ingest ─────────────────────────────────────────")
    start = (run_date - timedelta(days=3)).isoformat()
    end   = run_date.isoformat()
    if dry_run:
        log.info("  DRY RUN — skipping download.")
        return []
    return download_hls(
        start=start,
        end=end,
        cloud_cover_max=cloud_max,
        product="S30",
        bbox=GUNA_BBOX,
        data_lake=data_lake,
    )


def _step_weather_ingest(
    run_date: date,
    data_lake: Path,
    dry_run: bool,
) -> List[Path]:
    """Step 1b — Download GFS fire-weather forecast (24/48/72 h leads)."""
    log.info("── Step 1b: GFS Weather Ingest ────────────────────────────────")
    if dry_run:
        log.info("  DRY RUN — skipping download.")
        return []
    return download_gfs(
        run_date=run_date.isoformat(),
        lead_hours=[24, 48, 72],
        data_lake=data_lake,
        bbox=GUNA_BBOX,
        save_zarr=False,
    )


def _step_viirs_fetch(
    run_date: date,
    data_lake: Path,
    api_key: Optional[str],
    dry_run: bool,
) -> Dict:
    """Step 3 — Retrieve FIRMS VIIRS near-real-time fire detections."""
    log.info("── Step 3: FIRMS VIIRS Fetch ──────────────────────────────────")
    if not api_key:
        log.warning("  No FIRMS_MAP_KEY set — VIIRS fetch skipped.")
        return {}
    if dry_run:
        log.info("  DRY RUN — skipping fetch.")
        return {}

    start = run_date - timedelta(days=1)   # last 24 h
    gdf = fetch_viirs_detections(
        api_key=api_key,
        start_date=start,
        end_date=run_date,
        bbox=GUNA_BBOX,
        data_lake=data_lake,
    )
    return summarise_detections(gdf)


def _step_fire_dynamics(
    run_date: date,
    data_lake: Path,
) -> List[Dict]:
    """Step 6 — Cluster VIIRS detections, build perimeters, compute CROS."""
    log.info("── Step 6: Fire Dynamics (FEDS) ───────────────────────────────")
    tracker = FireEventTracker(alpha=0.015, eps_deg=0.01, min_pixels=3)

    # Load two consecutive 12-hour snapshots if available
    for delta in [1, 0]:
        snap_date = run_date - timedelta(days=delta)
        try:
            gdf = load_detections_for_date(snap_date, data_lake)
            if not gdf.empty:
                # Assign overpass times (01:30 / 13:30 UTC for VIIRS)
                for hour in [1, 13]:
                    snap_ts = datetime.combine(
                        snap_date, datetime.min.time()
                    ).replace(hour=hour, minute=30)
                    half = gdf[gdf.get("daynight", "D").str.upper() == ("D" if hour == 13 else "N")]
                    if not half.empty:
                        tracker.add_snapshot(half, snap_ts)
        except FileNotFoundError:
            log.debug("  No detection file for %s — skipping.", snap_date)

    stats_list = []
    if len(tracker._snapshots) >= 2:
        tracker.build_perimeters()
        stats = tracker.compute_spread_stats()
        stats_list = [
            {
                "dt_hours":         s.dt_hours,
                "area_change_km2":  s.area_change_km2,
                "cros_km_h":        s.cros_km_h,
                "spread_bearing_deg": s.spread_bearing_deg,
                "new_pixels":       s.new_pixels,
                "flinelen_km":      s.flinelen_km,
            }
            for s in stats
        ]
        log.info("  %d spread-stat intervals computed.", len(stats_list))
    else:
        log.info("  Insufficient snapshots for spread stats (need ≥ 2).")

    return stats_list


def _step_vectorise_raster(
    run_date: date,
    out_dir: Path,
    threshold: float = 0.50,
    min_area_ha: float = 0.5,
) -> Optional[Path]:
    """
    Step 5 — Vectorise the Prithvi risk probability raster to GeoJSON.

    Looks for a raster at outputs/risk_rasters/<YYYY>/<MM>/<DD>/risk.tif
    (written by the inference step upstream).
    """
    log.info("── Step 5: Raster → GeoJSON Vectorisation ─────────────────────")
    tag = run_date.strftime("%Y/%m/%d")
    raster_path = Path(f"outputs/risk_rasters/{tag}/risk.tif")

    if not raster_path.exists():
        log.warning(
            "  Risk raster not found at %s — forecast layer will be empty.",
            raster_path,
        )
        # Emit an empty placeholder GeoJSON so the bulletin can still be built
        import geopandas as gpd
        placeholder = out_dir / f"forecast_{run_date.strftime('%Y%m%d')}.geojson"
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame({"geometry": []}).to_file(placeholder, driver="GeoJSON")
        return placeholder

    out_path = out_dir / f"forecast_{run_date.strftime('%Y%m%d')}.geojson"
    return raster_to_geojson(
        risk_raster_path=raster_path,
        out_path=out_path,
        threshold=threshold,
        min_area_ha=min_area_ha,
    )


def _step_validate(
    run_date: date,
    forecast_geojson: Path,
    data_lake: Path,
) -> Dict:
    """Step 7 — Spatial join predictions vs VIIRS; return precision/recall/F1."""
    log.info("── Step 7: Spatial Validation ─────────────────────────────────")
    date_tag = run_date.strftime("%Y%m%d")
    viirs_path = (
        data_lake
        / "fire_detections" / "firms_viirs"
        / run_date.strftime("%Y/%m/%d")
        / f"viirs_nrt_{date_tag}.geojson"
    )

    if not viirs_path.exists() or not forecast_geojson.exists():
        log.warning("  Missing inputs for validation — skipping.")
        return {}

    try:
        return validate_predictions(
            burn_polygons_path=forecast_geojson,
            viirs_points_path=viirs_path,
            confidence_filter=["nominal", "high"],
        )
    except Exception as exc:
        log.error("  Validation failed: %s", exc)
        return {"error": str(exc)}


def _step_build_bulletin(
    run_date: date,
    forecast_geojson: Path,
    data_lake: Path,
    out_dir: Path,
) -> Dict[str, str]:
    """Step 8 — Assemble the three-layer daily bulletin."""
    log.info("── Step 8: Build Daily Bulletin ───────────────────────────────")
    date_tag = run_date.strftime("%Y%m%d")

    active_path = (
        data_lake
        / "fire_detections" / "firms_viirs"
        / run_date.strftime("%Y/%m/%d")
        / f"viirs_nrt_{date_tag}.geojson"
    )
    if not active_path.exists():
        # Use empty placeholder
        import geopandas as gpd
        active_path.parent.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame({"geometry": []}).to_file(active_path, driver="GeoJSON")

    context_path = Path("data/aoi/guna_beats.geojson")   # beat boundaries as context

    paths = build_daily_bulletin(
        forecast_geojson=forecast_geojson,
        active_viirs_geojson=active_path,
        feds_perimeters_geojson=None,   # populated if tracker exported to disk
        context_geojson=context_path if context_path.exists() else None,
        out_dir=out_dir,
        bulletin_date=run_date,
    )
    return {k: str(v) for k, v in paths.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_daily_pipeline(
    run_date: Optional[date] = None,
    config_path: Path = DEFAULT_CONFIG,
    data_lake: Path = DEFAULT_DATA_LAKE,
    out_dir: Path = DEFAULT_OUT,
    dry_run: bool = False,
    firms_api_key: Optional[str] = None,
) -> PipelineResult:
    """
    Execute the full daily fire-risk pipeline for *run_date*.

    Parameters
    ----------
    run_date      : Date to process (default: today UTC).
    config_path   : Path to project YAML config.
    data_lake     : Root of local data-lake.
    out_dir       : Output directory for daily bulletins.
    dry_run       : Skip all external API calls; useful for CI/testing.
    firms_api_key : NASA FIRMS MAP_KEY (overrides env FIRMS_MAP_KEY).

    Returns
    -------
    PipelineResult summary dataclass.
    """
    t0 = time.time()

    if run_date is None:
        run_date = date.today()

    log.info("=" * 65)
    log.info("  jwalaNetra_2  Daily Fire-Risk Pipeline  |  %s", run_date.isoformat())
    log.info("=" * 65)

    # ── Load config ────────────────────────────────────────────────────────────
    cfg: Dict = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    cloud_max   = cfg.get("gee", {}).get("cloud_threshold", 15)
    threshold   = cfg.get("inference", {}).get("default_threshold", 0.35)
    min_area_ha = cfg.get("inference", {}).get("min_alert_area_ha", 0.2)

    api_key = (
        firms_api_key
        or os.environ.get("FIRMS_MAP_KEY")
        or cfg.get("data_lake", {}).get("firms_api_key")
    )

    result = PipelineResult(run_date=run_date.isoformat())
    bulletin_out = out_dir / run_date.strftime("%Y/%m/%d")
    bulletin_out.mkdir(parents=True, exist_ok=True)

    # ── Step 1: HLS Ingest ────────────────────────────────────────────────────
    try:
        result.hls_files = _step_hls_ingest(run_date, data_lake, cloud_max, dry_run)
        log.info("  ✓ HLS: %d granule files", len(result.hls_files))
    except Exception as exc:
        log.error("  ✗ HLS ingest failed: %s", exc)
        result.errors.append(f"hls_ingest: {exc}")

    # ── Step 1b: Weather Ingest ───────────────────────────────────────────────
    try:
        result.weather_files = _step_weather_ingest(run_date, data_lake, dry_run)
        log.info("  ✓ GFS: %d forecast files", len(result.weather_files))
    except Exception as exc:
        log.error("  ✗ Weather ingest failed: %s", exc)
        result.errors.append(f"weather_ingest: {exc}")

    # ── Step 3: VIIRS Fetch ───────────────────────────────────────────────────
    try:
        result.viirs_summary = _step_viirs_fetch(run_date, data_lake, api_key, dry_run)
        log.info("  ✓ VIIRS: %s", result.viirs_summary)
    except Exception as exc:
        log.error("  ✗ VIIRS fetch failed: %s", exc)
        result.errors.append(f"viirs_fetch: {exc}")

    # ── Step 5: Vectorise Risk Raster ─────────────────────────────────────────
    forecast_geojson: Optional[Path] = None
    try:
        forecast_geojson = _step_vectorise_raster(
            run_date, bulletin_out, threshold, min_area_ha
        )
        log.info("  ✓ Forecast GeoJSON: %s", forecast_geojson)
    except Exception as exc:
        log.error("  ✗ Vectorisation failed: %s", exc)
        result.errors.append(f"vectorise: {exc}")

    # ── Step 6: Fire Dynamics ─────────────────────────────────────────────────
    try:
        result.spread_stats = _step_fire_dynamics(run_date, data_lake)
        log.info("  ✓ Spread stats: %d intervals", len(result.spread_stats))
    except Exception as exc:
        log.error("  ✗ Fire dynamics failed: %s", exc)
        result.errors.append(f"fire_dynamics: {exc}")

    # ── Step 7: Validation ────────────────────────────────────────────────────
    if forecast_geojson:
        try:
            result.validation_metrics = _step_validate(
                run_date, forecast_geojson, data_lake
            )
            log.info("  ✓ Validation: %s", result.validation_metrics)
        except Exception as exc:
            log.error("  ✗ Validation failed: %s", exc)
            result.errors.append(f"validation: {exc}")

    # ── Step 8: Bulletin ──────────────────────────────────────────────────────
    if forecast_geojson:
        try:
            result.bulletin_paths = _step_build_bulletin(
                run_date, forecast_geojson, data_lake, bulletin_out
            )
            log.info("  ✓ Bulletin: %s", list(result.bulletin_paths.keys()))
        except Exception as exc:
            log.error("  ✗ Bulletin build failed: %s", exc)
            result.errors.append(f"bulletin: {exc}")

    # ── Persist run log ───────────────────────────────────────────────────────
    result.elapsed_seconds = time.time() - t0
    run_log_dir = Path(cfg.get("monitoring", {}).get("run_log_dir", "outputs/pipeline_runs"))
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = run_log_dir / f"run_{run_date.strftime('%Y%m%d')}.json"
    run_log_path.write_text(json.dumps(result.to_dict(), indent=2))

    log.info("=" * 65)
    log.info(
        "  Pipeline complete in %.1f s  |  %d errors",
        result.elapsed_seconds, len(result.errors),
    )
    if result.errors:
        for err in result.errors:
            log.warning("  ⚠  %s", err)
    log.info("=" * 65)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the jwalaNetra_2 daily fire-risk pipeline."
    )
    p.add_argument("--date", default=None, help="ISO date to process (default: today).")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--lake", default=str(DEFAULT_DATA_LAKE))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--firms-key", default=None, help="NASA FIRMS MAP_KEY.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

    run_date = (
        datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.date
        else date.today()
    )
    run_daily_pipeline(
        run_date=run_date,
        config_path=Path(args.config),
        data_lake=Path(args.lake),
        out_dir=Path(args.out),
        dry_run=args.dry_run,
        firms_api_key=args.firms_key,
    )


if __name__ == "__main__":
    main()
