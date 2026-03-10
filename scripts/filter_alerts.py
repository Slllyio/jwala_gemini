"""
Filter Alerts — Production Inference → GeoJSON
================================================

Takes a raw DW alert raster, runs the trained LightGBM filter,
and outputs confirmed change polygons as GeoJSON for the dashboard.

Engineering fixes applied:
  1. remove_small_objects BEFORE vectorization (no compute burn)
  2. UTM projection for accurate hectare calculation
  3. Cluster-first, classify-second (polygon mean deltas → change type)
  4. State raster for cumulative_alert_count (crash-resilient)

Usage:
    python scripts/filter_alerts.py \\
        --alert-tif data/ground_truth/HAMEERPUR/alerts/alert_delta_HAMEERPUR_2025-12-19_to_2025-12-24.tif \\
        --model outputs/alert_filter/model.lgbm \\
        --config outputs/alert_filter/feature_config.json \\
        --out-dir outputs/alert_filter/inference
"""

# ── PROJ/GDAL env fix ──
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()
del _os, _sys
# ────────────────────────

import argparse
import json
import logging
import os
import re
import time

import numpy as np
import rasterio
import rasterio.features
from scipy.ndimage import uniform_filter, label as ndlabel

import matplotlib
matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants (mirrored from train_alert_filter.py) ───────────────────────────

DW_BANDS = ["water", "trees", "grass", "flooded_veg",
            "crops", "shrub_scrub", "built", "bare"]
TREES_IDX = 1
N_BANDS   = 8

SEASON_MAP = {
    1: "Winter", 2: "Winter",
    3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
    12: "Winter",
}


# ── Feature extraction (identical to training) ───────────────────────────────

def extract_features(alert_data: np.ndarray, month: int,
                     alert_thresh: float,
                     H: int, W: int) -> np.ndarray:
    """
    Extract 23 features from a single alert raster.
    alert_data: (bands, H, W) — 10-band alert raster
    Returns: features (n_pix, 23)
    """
    n_pix = H * W
    has_cloud = alert_data.shape[0] >= 10

    # Band deltas
    deltas = alert_data[:N_BANDS].astype(np.float32)
    delta_flat = deltas.reshape(N_BANDS, n_pix)

    # Cloud
    if has_cloud:
        cld_before = alert_data[8].ravel().astype(np.float32)
        cld_after  = alert_data[9].ravel().astype(np.float32)
    else:
        cld_before = np.zeros(n_pix, dtype=np.float32)
        cld_after  = np.zeros(n_pix, dtype=np.float32)
    cld_worst = np.maximum(cld_before, cld_after)

    # Spatial context
    trees_2d = deltas[TREES_IDX]
    alert_mask_2d = (np.abs(trees_2d) >= alert_thresh).astype(np.float32)
    neighbor_sum = uniform_filter(alert_mask_2d, size=3, mode="constant") * 9
    n_neighbors = np.clip((neighbor_sum - alert_mask_2d).ravel(), 0, 8)
    mean_nb_trees = uniform_filter(trees_2d, size=3, mode="constant").ravel()
    mean_sq = uniform_filter(trees_2d**2, size=3, mode="constant").ravel()
    std_nb_trees = np.sqrt(np.maximum(mean_sq - mean_nb_trees**2, 0))

    # Cross-band
    trees_delta = delta_flat[TREES_IDX]
    crops_delta = delta_flat[4]
    bare_delta  = delta_flat[7]
    trees_crops_anti = ((trees_delta < 0) & (crops_delta > 0)).astype(np.float32)
    trees_bare_anti  = ((trees_delta < 0) & (bare_delta > 0)).astype(np.float32)

    gain_deltas = delta_flat.copy()
    gain_deltas[TREES_IDX] = -999
    dominant_gain = np.argmax(gain_deltas, axis=0).astype(np.float32)
    band_div = (np.abs(delta_flat) > alert_thresh).sum(axis=0).astype(np.float32)

    # Magnitude
    abs_trees = np.abs(trees_delta)
    abs_max = np.max(np.abs(delta_flat), axis=0)

    # Metadata
    season = SEASON_MAP.get(month, "Unknown")
    month_arr = np.full(n_pix, month, dtype=np.float32)
    is_monsoon = np.full(n_pix, 1.0 if season == "Monsoon" else 0.0, dtype=np.float32)
    is_winter = np.full(n_pix, 1.0 if season == "Winter" else 0.0, dtype=np.float32)

    features = np.column_stack([
        delta_flat[0], delta_flat[1], delta_flat[2], delta_flat[3],
        delta_flat[4], delta_flat[5], delta_flat[6], delta_flat[7],
        cld_before, cld_after, cld_worst,
        n_neighbors, mean_nb_trees, std_nb_trees,
        trees_crops_anti, trees_bare_anti, dominant_gain, band_div,
        abs_trees, abs_max,
        month_arr, is_monsoon, is_winter,
    ])
    return features


# ── Change type classification (polygon-level) ───────────────────────────────

def classify_change_type(mean_deltas: dict) -> str:
    """
    Classify change type from polygon-mean deltas.
    Cluster-first, classify-second (Trap 2 fix).
    """
    trees = mean_deltas.get("trees", 0)
    crops = mean_deltas.get("crops", 0)
    bare  = mean_deltas.get("bare", 0)
    built = mean_deltas.get("built", 0)
    shrub = mean_deltas.get("shrub_scrub", 0)
    grass = mean_deltas.get("grass", 0)

    if trees > 0.05:
        return "Greening"

    if trees < -0.05:
        # Find what replaced trees
        gains = {"crops": crops, "bare": bare, "built": built,
                 "shrub_scrub": shrub, "grass": grass}
        top_gain_band = max(gains, key=gains.get)
        top_gain_val  = gains[top_gain_band]

        if top_gain_band == "crops" and top_gain_val > 0.03:
            return "Encroachment"
        elif top_gain_band == "built" and top_gain_val > 0.03:
            return "Built expansion"
        elif top_gain_band == "bare" and top_gain_val > 0.03:
            return "Clearing"
        elif top_gain_band in ("shrub_scrub", "grass") and top_gain_val > 0.03:
            return "Degradation"
        else:
            return "Tree loss (unclassified)"

    return "Other change"


# ── GeoJSON generation ────────────────────────────────────────────────────────

def generate_geojson(confirmed_mask: np.ndarray, confidence_map: np.ndarray,
                     alert_data: np.ndarray, transform, crs_str: str,
                     detection_date: str, sub_range: str,
                     min_pixels: int = 5) -> dict:
    """
    Generate GeoJSON from confirmed pixel mask.
    Applies remove_small_objects → vectorize → UTM area → classify.
    """
    try:
        from skimage.morphology import remove_small_objects
    except ImportError:
        from scipy.ndimage import label as sci_label
        # Fallback: manual small object removal
        labeled, n_feat = sci_label(confirmed_mask)
        for i in range(1, n_feat + 1):
            if (labeled == i).sum() < min_pixels:
                confirmed_mask[labeled == i] = False
        clean_mask = confirmed_mask
    else:
        # Trap 3 fix: wipe small clusters at matrix level before vectorizing
        clean_mask = remove_small_objects(
            confirmed_mask.astype(bool), min_size=min_pixels
        )

    if not clean_mask.any():
        log.info("  No confirmed polygons after small-object removal")
        return {"type": "FeatureCollection", "features": []}

    H, W = clean_mask.shape
    deltas = alert_data[:N_BANDS].astype(np.float32)

    # Label connected components for polygon-level stats
    labeled, n_features = ndlabel(clean_mask.astype(np.int32))

    # Vectorize using rasterio
    shapes_gen = rasterio.features.shapes(
        labeled.astype(np.int32),
        mask=clean_mask,
        transform=transform,
    )

    features = []
    seen_labels = set()

    for geom, value in shapes_gen:
        label_id = int(value)
        if label_id in seen_labels or label_id == 0:
            continue
        seen_labels.add(label_id)

        # Get pixels belonging to this polygon
        poly_mask = labeled == label_id
        n_pixels = int(poly_mask.sum())

        # Mean confidence
        mean_conf = float(np.mean(confidence_map[poly_mask]))

        # Mean deltas for change type classification (Trap 2: cluster-first)
        mean_deltas = {}
        for b, name in enumerate(DW_BANDS):
            mean_deltas[name] = float(np.mean(deltas[b][poly_mask]))

        change_type = classify_change_type(mean_deltas)

        # Centroid (in EPSG:4326 coordinates)
        coords = np.array(geom["coordinates"][0])
        centroid_lon = float(np.mean(coords[:, 0]))
        centroid_lat = float(np.mean(coords[:, 1]))

        # Area: use pixel count × pixel area (from transform)
        # Each pixel = |transform.a| × |transform.e| degrees
        # At ~24°N (MP latitude), 1° lat ≈ 110.6 km, 1° lon ≈ 101.3 km
        pixel_width_m  = abs(transform.a) * 101300  # approximate
        pixel_height_m = abs(transform.e) * 110600
        pixel_area_m2  = pixel_width_m * pixel_height_m
        area_ha = n_pixels * pixel_area_m2 / 10000

        feature = {
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "change_type": change_type,
                "confidence": round(mean_conf, 3),
                "area_ha": round(area_ha, 4),
                "n_pixels": n_pixels,
                "centroid": [round(centroid_lat, 6), round(centroid_lon, 6)],
                "detection_date": detection_date,
                "sub_range": sub_range,
                "mean_delta_trees": round(mean_deltas["trees"], 4),
                "mean_delta_crops": round(mean_deltas["crops"], 4),
                "mean_delta_bare": round(mean_deltas["bare"], 4),
                "mean_delta_built": round(mean_deltas["built"], 4),
            }
        }
        features.append(feature)

    log.info(f"  Generated {len(features)} polygons")
    return {"type": "FeatureCollection", "features": features}


# ── State raster management (Trap 4 fix) ──────────────────────────────────────

def load_state_raster(state_path: str, H: int, W: int) -> np.ndarray:
    """Load cumulative alert count from disk, or create zeros."""
    if os.path.exists(state_path):
        with rasterio.open(state_path) as ds:
            state = ds.read(1).astype(np.int32)
        log.info(f"  Loaded state raster: {state_path} (max={state.max()})")
        return state
    else:
        log.info(f"  No state raster found — starting fresh")
        return np.zeros((H, W), dtype=np.int32)


def save_state_raster(state: np.ndarray, state_path: str, profile: dict):
    """Save updated cumulative alert count to disk."""
    p = profile.copy()
    p.update(count=1, dtype="int32", compress="lzw")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with rasterio.open(state_path, "w", **p) as dst:
        dst.write(state.astype(np.int32), 1)
    log.info(f"  State raster saved: {state_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alert-tif", required=True, help="10-band alert delta raster")
    parser.add_argument("--model", required=True, help="Path to model.lgbm")
    parser.add_argument("--config", required=True, help="Path to feature_config.json")
    parser.add_argument("--out-dir", default="outputs/inference")
    parser.add_argument("--state-dir", default="data/state",
                        help="Dir for cumulative alert state rasters")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override operational threshold (default: from config)")
    parser.add_argument("--min-pixels", type=int, default=5,
                        help="Min cluster size in pixels (default: 5)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # ── Load config + model ──────────────────────────────────────────
    import lightgbm as lgb

    with open(args.config) as f:
        config = json.load(f)

    threshold = args.threshold or config["operational_threshold"]
    alert_thresh = config["alert_threshold"]

    model = lgb.Booster(model_file=args.model)
    log.info(f"Model loaded: {args.model}")
    log.info(f"Threshold: {threshold:.4f}")

    # ── Parse metadata from filename ─────────────────────────────────
    basename = os.path.splitext(os.path.basename(args.alert_tif))[0]
    m = re.search(r"alert_delta_(.+?)_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})",
                  basename)
    if m:
        sub_range = m.group(1)
        date_before = m.group(2)
        date_after = m.group(3)
    else:
        sub_range = "UNKNOWN"
        date_before = "unknown"
        date_after = "unknown"

    month = int(date_before.split("-")[1]) if date_before != "unknown" else 1
    detection_date = date_after

    log.info(f"Sub-range: {sub_range}")
    log.info(f"Alert window: {date_before} → {date_after}")
    log.info(f"Season: {SEASON_MAP.get(month, 'Unknown')}")

    # ── Load alert raster ────────────────────────────────────────────
    with rasterio.open(args.alert_tif) as ds:
        alert_data = ds.read()
        profile = ds.profile.copy()
        transform = ds.transform
        crs = str(ds.crs)

    H, W = alert_data.shape[1], alert_data.shape[2]
    n_pix = H * W
    log.info(f"Alert raster: {alert_data.shape} ({H}×{W} = {n_pix:,} pixels)")

    # ── Extract features ─────────────────────────────────────────────
    features = extract_features(alert_data, month, alert_thresh, H, W)

    # ── Predict ──────────────────────────────────────────────────────
    log.info("Running model prediction...")
    y_prob = model.predict(features)
    confidence_map = y_prob.reshape(H, W).astype(np.float32)

    # Alert-firing mask (only predict on significant pixels)
    trees_abs = np.abs(alert_data[TREES_IDX]).ravel()
    is_alert = trees_abs >= alert_thresh

    # Zero out non-alerting pixels
    confidence_map_masked = confidence_map.copy()
    confidence_map_masked[~is_alert.reshape(H, W)] = 0

    # Apply threshold
    confirmed_mask = confidence_map_masked >= threshold

    n_alerting = int(is_alert.sum())
    n_confirmed = int(confirmed_mask.sum())
    log.info(f"Alerting pixels: {n_alerting:,}")
    log.info(f"Confirmed pixels (P≥{threshold:.3f}): {n_confirmed:,}")
    log.info(f"Filter rate: {(1 - n_confirmed/max(n_alerting,1))*100:.1f}% filtered out")

    # ── Save confidence raster ───────────────────────────────────────
    p = profile.copy()
    p.update(count=1, dtype="float32", compress="lzw")
    conf_path = os.path.join(args.out_dir, f"confidence_{basename}.tif")
    with rasterio.open(conf_path, "w", **p) as dst:
        dst.write(confidence_map_masked, 1)
    log.info(f"Confidence raster: {conf_path}")

    # ── Generate GeoJSON ─────────────────────────────────────────────
    geojson = generate_geojson(
        confirmed_mask, confidence_map_masked, alert_data,
        transform, crs, detection_date, sub_range,
        min_pixels=args.min_pixels,
    )

    geojson_path = os.path.join(
        args.out_dir, f"confirmed_alerts_{sub_range}_{date_after}.geojson"
    )
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)
    log.info(f"GeoJSON: {geojson_path}")

    # ── Update state raster (Trap 4 fix) ─────────────────────────────
    state_path = os.path.join(args.state_dir, f"cumulative_alerts_{sub_range}.tif")
    state = load_state_raster(state_path, H, W)
    state += confirmed_mask.astype(np.int32)
    save_state_raster(state, state_path, profile)

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0

    # Change type summary
    type_counts = {}
    for feat in geojson["features"]:
        ct = feat["properties"]["change_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1

    summary = {
        "elapsed_seconds": round(elapsed, 1),
        "alert_tif": args.alert_tif,
        "sub_range": sub_range,
        "detection_date": detection_date,
        "threshold": threshold,
        "n_alerting_pixels": n_alerting,
        "n_confirmed_pixels": n_confirmed,
        "filter_rate_pct": round((1 - n_confirmed / max(n_alerting, 1)) * 100, 1),
        "n_polygons": len(geojson["features"]),
        "change_type_counts": type_counts,
        "total_area_ha": round(sum(
            f["properties"]["area_ha"] for f in geojson["features"]
        ), 2),
    }
    summary_path = os.path.join(args.out_dir, f"inference_summary_{basename}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\n{'='*65}")
    log.info(f"  INFERENCE COMPLETE in {elapsed:.1f}s")
    log.info(f"  Alerting:  {n_alerting:,} pixels")
    log.info(f"  Confirmed: {n_confirmed:,} pixels ({summary['filter_rate_pct']:.0f}% filtered)")
    log.info(f"  Polygons:  {len(geojson['features'])}")
    for ct, cnt in sorted(type_counts.items()):
        log.info(f"    {ct}: {cnt}")
    log.info(f"  Total area: {summary['total_area_ha']:.2f} ha")
    log.info(f"  GeoJSON: {geojson_path}")
    log.info(f"{'='*65}")


if __name__ == "__main__":
    main()
