"""
Spatial Validation and GeoJSON Risk Delineation
================================================
Validates Prithvi-EO burn scar / risk predictions against real-time VIIRS
fire detections using GeoPandas spatial joins, and vectorises raster
probability maps to daily GeoJSON output layers.

Three daily bulletin layers (PDF §"Daily Inference and GeoJSON Generation"):

  1. Forecast Layer  — risk polygons for the "tomorrow" 24-72 h window.
  2. Active Layer    — real-time VIIRS detections + FEDS perimeters.
  3. Context Layer   — at-risk critical infrastructure / population centres.

Usage::

    from src.analysis.spatial_validation import (
        validate_predictions,
        raster_to_geojson,
        build_daily_bulletin,
    )

    metrics = validate_predictions(
        burn_polygons_path="outputs/prithvi_burn_scars.geojson",
        viirs_points_path="data_lake/fire_detections/firms_viirs/2025/03/01/viirs_nrt_20250301.geojson",
    )
    print(metrics)   # {"precision": 0.82, "recall": 0.74, "f1": 0.78, ...}
"""

from __future__ import annotations

import src.utils.proj_fix  # noqa: F401  (PROJ/GDAL env fix)

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction validation (spatial join)
# ─────────────────────────────────────────────────────────────────────────────

def validate_predictions(
    burn_polygons_path: Union[str, Path],
    viirs_points_path: Union[str, Path],
    confidence_filter: Optional[List[str]] = None,
) -> Dict:
    """
    Spatial join VIIRS detections against Prithvi-derived burn scar polygons
    to compute segmentation precision, recall, and F1.

    Method (PDF §"Spatial Joins for Spread Validation"):
        inner join: keep only VIIRS points that fall *within* a predicted polygon.
        TP = matched points  (predicted burn scar contains real fire)
        FP = predicted polygon has no VIIRS points (over-prediction)
        FN = VIIRS points outside any polygon (missed detections)

    Parameters
    ----------
    burn_polygons_path : Path to Prithvi-derived burn scar / risk GeoJSON.
    viirs_points_path  : Path to FIRMS VIIRS point GeoJSON or GeoParquet.
    confidence_filter  : If set, keep only detections with these confidence
                         levels (e.g. ["nominal", "high"]).

    Returns
    -------
    Dict with precision, recall, f1, iou, tp, fp, fn, total_viirs,
    total_polygons, and per-fire-id detection counts.
    """
    import geopandas as gpd

    # ── Load layers ────────────────────────────────────────────────────────────
    burn_polys = gpd.read_file(burn_polygons_path)
    log.info("Loaded %d burn polygons from %s", len(burn_polys), burn_polygons_path)

    vp = Path(viirs_points_path)
    if vp.suffix == ".parquet":
        viirs_pts = gpd.read_parquet(vp)
    else:
        viirs_pts = gpd.read_file(vp)
    log.info("Loaded %d VIIRS detections from %s", len(viirs_pts), vp)

    # ── Confidence filter ──────────────────────────────────────────────────────
    if confidence_filter and "confidence" in viirs_pts.columns:
        before = len(viirs_pts)
        viirs_pts = viirs_pts[
            viirs_pts["confidence"].str.lower().isin(
                [c.lower() for c in confidence_filter]
            )
        ]
        log.info(
            "Confidence filter %s: %d → %d detections.",
            confidence_filter, before, len(viirs_pts),
        )

    # ── Align CRS ──────────────────────────────────────────────────────────────
    if burn_polys.crs != viirs_pts.crs:
        viirs_pts = viirs_pts.to_crs(burn_polys.crs)

    total_viirs    = len(viirs_pts)
    total_polygons = len(burn_polys)

    # ── Spatial join (inner): points within polygons ───────────────────────────
    matched = gpd.sjoin(
        viirs_pts,
        burn_polys,
        how="inner",
        predicate="within",
    )
    tp = len(matched)
    fn = total_viirs - tp

    # Polygons that contain at least one VIIRS detection
    hit_polygon_ids = set(matched.index_right.unique())
    fp = total_polygons - len(hit_polygon_ids)

    # ── Metrics ────────────────────────────────────────────────────────────────
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # Per-fire-id counts (if 'fireid' present)
    per_fire: Dict = {}
    if "fireid" in matched.columns:
        per_fire = matched["fireid"].value_counts().to_dict()

    metrics = {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "iou":       round(iou,       4),
        "tp": tp, "fp": fp, "fn": fn,
        "total_viirs_detections": total_viirs,
        "total_predicted_polygons": total_polygons,
        "per_fire_id_detections": per_fire,
    }

    log.info(
        "Validation  precision=%.3f  recall=%.3f  F1=%.3f  IoU=%.3f",
        precision, recall, f1, iou,
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Raster → GeoJSON vectorisation
# ─────────────────────────────────────────────────────────────────────────────

def raster_to_geojson(
    risk_raster_path: Union[str, Path],
    out_path: Union[str, Path],
    threshold: float = 0.50,
    min_area_ha: float = 0.5,
    risk_field: str = "risk_score",
) -> Path:
    """
    Vectorise a Prithvi probability raster into a GeoJSON polygon layer.

    Implements PDF §"Vectorization":
        high-risk areas are thresholded and converted via rasterio.features.shapes()

    Parameters
    ----------
    risk_raster_path : Path to a single-band GeoTIFF with float32 [0, 1] values.
    out_path         : Output GeoJSON file path.
    threshold        : Probability cut-off for "high risk" (default 0.5).
    min_area_ha      : Drop polygons smaller than this (noise filter).
    risk_field       : Property name for the mean risk score.

    Returns
    -------
    Path to the written GeoJSON file.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape

    with rasterio.open(risk_raster_path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs

    # Binary mask at threshold
    mask = (data >= threshold).astype(np.uint8)

    # Extract contiguous high-risk polygons
    polys = []
    values = []
    for geom_json, val in shapes(mask, mask=mask, transform=transform):
        polys.append(shape(geom_json))
        # Mean risk score within this polygon (approximate via raster value)
        values.append(float(val))

    if not polys:
        log.warning("No areas above threshold %.2f in %s", threshold, risk_raster_path)
        gdf = gpd.GeoDataFrame({"geometry": [], risk_field: []}, crs=crs)
    else:
        gdf = gpd.GeoDataFrame(
            {risk_field: values, "geometry": polys},
            crs=crs,
        ).to_crs("EPSG:4326")

        # Filter by minimum area
        if min_area_ha > 0:
            gdf_utm = gdf.to_crs("EPSG:32644")
            gdf["area_ha"] = gdf_utm.area / 10_000.0
            before = len(gdf)
            gdf = gdf[gdf["area_ha"] >= min_area_ha].reset_index(drop=True)
            log.info(
                "Area filter ≥%.1f ha: %d → %d polygons.", min_area_ha, before, len(gdf)
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    log.info("Risk polygons → %s  (%d features)", out_path, len(gdf))
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Daily bulletin builder
# ─────────────────────────────────────────────────────────────────────────────

def build_daily_bulletin(
    forecast_geojson: Union[str, Path],
    active_viirs_geojson: Union[str, Path],
    feds_perimeters_geojson: Optional[Union[str, Path]],
    context_geojson: Optional[Union[str, Path]],
    out_dir: Path,
    bulletin_date: Optional[date] = None,
) -> Dict[str, Path]:
    """
    Assemble the three-layer daily bulletin described in PDF §"Operational
    Workflow for Daily Risk Delineation":

      Forecast Layer  — risk polygons for tomorrow.
      Active Layer    — VIIRS detections + FEDS perimeters.
      Context Layer   — at-risk infrastructure / population centres.

    Parameters
    ----------
    forecast_geojson        : Risk polygon GeoJSON from Prithvi inference.
    active_viirs_geojson    : Real-time VIIRS detection GeoJSON.
    feds_perimeters_geojson : Estimated fire perimeters (may be None).
    context_geojson         : Critical infrastructure polygons (may be None).
    out_dir                 : Output directory for bulletin files.
    bulletin_date           : Date label (default: today).

    Returns
    -------
    Dict of {layer_name: file_path}.
    """
    import geopandas as gpd

    if bulletin_date is None:
        bulletin_date = date.today()

    tag = bulletin_date.strftime("%Y%m%d")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bulletin: Dict[str, Path] = {}

    # ── 1. Forecast Layer ──────────────────────────────────────────────────────
    fc_out = out_dir / f"forecast_{tag}.geojson"
    _copy_geojson_with_metadata(
        src=Path(forecast_geojson),
        dst=fc_out,
        extra_props={"layer": "forecast", "date": tag, "window_hours": 24},
    )
    bulletin["forecast"] = fc_out
    log.info("Forecast layer → %s", fc_out)

    # ── 2. Active Layer (VIIRS + FEDS perimeters) ─────────────────────────────
    active_gdfs = [gpd.read_file(active_viirs_geojson)]
    if feds_perimeters_geojson and Path(feds_perimeters_geojson).exists():
        active_gdfs.append(gpd.read_file(feds_perimeters_geojson))

    active_combined = gpd.pd.concat(active_gdfs, ignore_index=True)
    active_gdf = gpd.GeoDataFrame(active_combined, crs="EPSG:4326")
    active_out = out_dir / f"active_{tag}.geojson"
    active_gdf.to_file(active_out, driver="GeoJSON")
    bulletin["active"] = active_out
    log.info("Active layer   → %s", active_out)

    # ── 3. Context Layer ───────────────────────────────────────────────────────
    if context_geojson and Path(context_geojson).exists():
        ctx_gdf = gpd.read_file(context_geojson)
        # Intersect with forecast layer to find "at risk" features
        fc_gdf = gpd.read_file(fc_out)
        fc_union = fc_gdf.geometry.union_all()
        at_risk = ctx_gdf[ctx_gdf.geometry.intersects(fc_union)].copy()
        ctx_out = out_dir / f"context_{tag}.geojson"
        at_risk.to_file(ctx_out, driver="GeoJSON")
        bulletin["context"] = ctx_out
        log.info(
            "Context layer  → %s  (%d at-risk features)", ctx_out, len(at_risk)
        )
    else:
        log.debug("No context GeoJSON provided — skipping context layer.")

    # ── Bulletin index JSON ────────────────────────────────────────────────────
    index = {
        "date": tag,
        "generated_utc": datetime.utcnow().isoformat(),
        "layers": {k: str(v) for k, v in bulletin.items()},
    }
    index_path = out_dir / f"bulletin_index_{tag}.json"
    index_path.write_text(json.dumps(index, indent=2))
    log.info("Bulletin index → %s", index_path)
    bulletin["index"] = index_path

    return bulletin


def _copy_geojson_with_metadata(
    src: Path,
    dst: Path,
    extra_props: Dict,
) -> None:
    """Copy a GeoJSON file and inject extra top-level metadata properties."""
    import geopandas as gpd

    gdf = gpd.read_file(src)
    for key, val in extra_props.items():
        gdf[key] = val
    gdf.to_file(dst, driver="GeoJSON")


# ─────────────────────────────────────────────────────────────────────────────
# Department-aware ignition probability weights (PDF §"Geographic Adaptation")
# ─────────────────────────────────────────────────────────────────────────────

GUNA_DEPARTMENT_WEIGHTS: Dict[str, Dict] = {
    "Aron":         {"ignition_weight": 1.2, "road_density": "moderate"},
    "Bamori":       {"ignition_weight": 1.5, "road_density": "high"},
    "Binaganj":     {"ignition_weight": 1.0, "road_density": "low"},
    "Fatehgarh":    {"ignition_weight": 1.3, "road_density": "moderate"},
    "Maksudangarh": {"ignition_weight": 1.1, "road_density": "low"},
    "North_Guna":   {"ignition_weight": 1.4, "road_density": "high"},
    "Raghogarh":    {"ignition_weight": 1.0, "road_density": "low"},
    "South_Guna":   {"ignition_weight": 1.2, "road_density": "moderate"},
}


def apply_department_weights(
    risk_gdf: "geopandas.GeoDataFrame",
    ranges_gdf: "geopandas.GeoDataFrame",
    risk_col: str = "risk_score",
) -> "geopandas.GeoDataFrame":
    """
    Apply department-specific ignition probability weights by spatial join
    of risk polygons against forest range boundaries.

    Higher weight in ranges with dense road networks or campsite access
    (human ignition probability is higher).

    Parameters
    ----------
    risk_gdf   : Risk polygon GeoDataFrame with *risk_col* column.
    ranges_gdf : Forest range boundary GeoDataFrame with 'range_name' column.
    risk_col   : Column containing raw [0, 1] risk scores.

    Returns
    -------
    GeoDataFrame with 'weighted_risk' column clamped to [0, 1].
    """
    import geopandas as gpd

    joined = gpd.sjoin(risk_gdf, ranges_gdf[["range_name", "geometry"]], how="left")

    def _apply_weight(row):
        w = GUNA_DEPARTMENT_WEIGHTS.get(
            row.get("range_name", ""), {}
        ).get("ignition_weight", 1.0)
        return min(row[risk_col] * w, 1.0)

    joined["weighted_risk"] = joined.apply(_apply_weight, axis=1)
    return joined.drop(columns=["index_right"], errors="ignore")
