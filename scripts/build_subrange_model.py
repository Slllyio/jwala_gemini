"""
Build SubRange Model — One Command, One Model
==============================================

Single entry point to train & deploy per-sub-range LightGBM alert filter.

Training: 2025 alert windows + GT (Dec 2024 → Dec 2025)
Inference: 2026+ real-time DW windows → filter noise → confirmed alerts

Phases:
  export  — Submit GEE tasks, poll & download (30-60 min)
  train   — Train LightGBM + filter + vectorize (2-5 min, all local)
  all     — Both in sequence

Usage:
    # If data already downloaded from GDrive → just train:
    python scripts/build_subrange_model.py --sub-range HAMEERPUR --phase train

    # Export data from GEE first:
    python scripts/build_subrange_model.py --sub-range HAMEERPUR --phase export

    # Train ALL sub-ranges (data must exist):
    python scripts/build_subrange_model.py --all --phase train

    # List available sub-ranges:
    python scripts/build_subrange_model.py --list

Output:
    outputs/models/{SUB_RANGE}/
    ├── model.lgbm
    ├── feature_config.json
    ├── training_report.json
    ├── filtered_{SUB_RANGE}.tif
    ├── alerts_{SUB_RANGE}.geojson
    ├── pr_curve.png
    ├── confusion_matrix.png
    └── feature_importance.png
"""

# ── PROJ/GDAL env fix ──
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()
del _os, _sys
# ────────────────────────

import argparse
import glob
import json
import logging
import os
import re
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Root directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT    = os.path.join(PROJECT_ROOT, "data", "ground_truth")
MODEL_ROOT   = os.path.join(PROJECT_ROOT, "outputs", "models")
GLOBAL_MODEL = os.path.join(PROJECT_ROOT, "outputs", "alert_filter")

# GEE config
GEE_PROJECT  = "van-suraksha-alert"
GUNA_ASSET   = "projects/van-suraksha-alert/assets/gunafinal"
DRIVE_FOLDER = "prithvi_guna_v4"

# Minimum thresholds for training viability
MIN_TP_PIXELS    = 500   # need at least 500 TP pixels for robust model
MIN_WINDOWS      = 3     # need at least 3 alert windows
MIN_TOTAL_ALERTS = 1000  # need at least 1000 alerting pixels total


# ── Utility ───────────────────────────────────────────────────────────────────

def list_subranges_local() -> list:
    """List sub-ranges that have local data in data/ground_truth/."""
    if not os.path.isdir(DATA_ROOT):
        return []
    subs = []
    for d in sorted(os.listdir(DATA_ROOT)):
        full = os.path.join(DATA_ROOT, d)
        if os.path.isdir(full):
            alerts_dir = os.path.join(full, "alerts")
            gt_files = glob.glob(os.path.join(full, "gt_delta_*.tif"))
            n_alerts = len(glob.glob(os.path.join(alerts_dir, "alert_delta_*.tif"))) if os.path.isdir(alerts_dir) else 0
            subs.append({
                "name": d,
                "n_alerts": n_alerts,
                "has_gt": len(gt_files) > 0,
                "ready": n_alerts >= MIN_WINDOWS and len(gt_files) > 0,
            })
    return subs


def list_subranges_gee() -> list:
    """List all sub-ranges from the GEE asset."""
    try:
        import ee
        ee.Initialize(project=GEE_PROJECT)
        fc = ee.FeatureCollection(GUNA_ASSET)
        all_sr = fc.aggregate_array("SUB_RANGE").getInfo()
        return sorted(set(str(x) for x in all_sr if x))
    except Exception as e:
        log.warning(f"Could not list GEE sub-ranges: {e}")
        return []


def check_data_readiness(sub_range: str) -> dict:
    """Check if a sub-range has sufficient data for training."""
    data_dir = os.path.join(DATA_ROOT, sub_range)
    alerts_dir = os.path.join(data_dir, "alerts")

    result = {
        "sub_range": sub_range,
        "data_dir": data_dir,
        "alerts_dir": alerts_dir,
        "has_data_dir": os.path.isdir(data_dir),
        "alert_tifs": [],
        "gt_tif": None,
        "ready": False,
        "reason": "",
    }

    if not result["has_data_dir"]:
        result["reason"] = f"No data directory: {data_dir}"
        return result

    # Find alert TIFs
    if os.path.isdir(alerts_dir):
        result["alert_tifs"] = sorted(glob.glob(
            os.path.join(alerts_dir, "alert_delta_*.tif")
        ))

    # Find GT TIF
    gt_files = glob.glob(os.path.join(data_dir, "gt_delta_*.tif"))
    if gt_files:
        result["gt_tif"] = gt_files[0]

    # Check readiness
    if len(result["alert_tifs"]) < MIN_WINDOWS:
        result["reason"] = f"Only {len(result['alert_tifs'])} alert windows (need {MIN_WINDOWS}+)"
    elif result["gt_tif"] is None:
        result["reason"] = "No GT delta raster found"
    else:
        result["ready"] = True
        result["reason"] = f"{len(result['alert_tifs'])} windows + GT ready"

    return result


# ── Phase A: Export ───────────────────────────────────────────────────────────

def phase_export(sub_range: str, year: int = 2025):
    """Submit GEE export tasks for alert series + GT delta."""
    import ee
    ee.Initialize(project=GEE_PROJECT)

    fc = ee.FeatureCollection(GUNA_ASSET)

    log.info(f"\n{'='*65}")
    log.info(f"  PHASE A: EXPORT  —  {sub_range}")
    log.info(f"  Training year: {year}")
    log.info(f"  GT: Dec {year-1} → Dec {year}")
    log.info(f"{'='*65}\n")

    # Get AOI
    sub_fc = fc.filter(ee.Filter.eq("SUB_RANGE", sub_range))
    count = sub_fc.size().getInfo()
    if count == 0:
        all_sr = list_subranges_gee()
        log.error(f"Sub-range '{sub_range}' not found. Available: {all_sr}")
        return False
    log.info(f"Found {count} beats in SUB_RANGE='{sub_range}'")
    aoi = sub_fc.geometry().dissolve(maxError=10)

    # ── Export alert series ──────────────────────────────────────────
    log.info(f"\nStep 1: Export alert series for {year}...")
    # Import and call the export_alert_series logic
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

    from export_alert_series import (
        list_dw_dates, dw_single_date, cloud_prob_single_date,
        check_threshold, submit_export
    )

    dates = list_dw_dates(aoi, year)
    log.info(f"  Found {len(dates)} DW dates for {year}")
    if len(dates) < 2:
        log.error(f"  Need at least 2 DW dates, got {len(dates)}")
        return False

    drive_folder = f"{DRIVE_FOLDER}/{sub_range}/alerts"
    n_exported = 0
    tree_thresh = 0.25
    other_thresh = 0.25

    for i in range(len(dates) - 1):
        d_before = dates[i]
        d_after = dates[i + 1]

        dw_before = dw_single_date(d_before, aoi)
        dw_after = dw_single_date(d_after, aoi)
        delta = dw_after.subtract(dw_before)

        result = check_threshold(delta, aoi, tree_thresh, other_thresh)
        if not result["export"]:
            log.info(f"  [{i+1:3d}] {d_before} → {d_after}  SKIP")
            continue

        log.info(f"  [{i+1:3d}] {d_before} → {d_after}  🔥 TRIGGERED")

        cld_before = cloud_prob_single_date(d_before, aoi)
        cld_after = cloud_prob_single_date(d_after, aoi)
        out = (delta
               .addBands(cld_before.rename("cloud_prob_before"))
               .addBands(cld_after.rename("cloud_prob_after")))
        name = f"alert_delta_{sub_range}_{d_before}_to_{d_after}"
        submit_export(out, name, aoi, drive_folder)
        n_exported += 1

    log.info(f"\nAlert exports submitted: {n_exported}")

    # ── Export GT delta ──────────────────────────────────────────────
    log.info(f"\nStep 2: Export GT delta (Dec {year-1} → Dec {year})...")
    from export_range_rasters import dec_median, submit as submit_gt

    img_a = dec_median(year - 1, aoi)
    img_b = dec_median(year, aoi)
    gt_delta = img_b.subtract(img_a)
    gt_name = f"gt_delta_{year-1}_{year}_{sub_range}"
    submit_gt(gt_delta, gt_name, aoi, folder=f"{DRIVE_FOLDER}/{sub_range}")
    log.info(f"  GT export submitted: {gt_name}")

    log.info(f"\n{'='*65}")
    log.info(f"  EXPORT COMPLETE for {sub_range}")
    log.info(f"  Alert rasters: {n_exported} tasks submitted")
    log.info(f"  GT raster: 1 task submitted")
    log.info(f"  Monitor: https://code.earthengine.google.com/tasks")
    log.info(f"")
    log.info(f"  After tasks finish, download from GDrive:")
    log.info(f"    {DRIVE_FOLDER}/{sub_range}/alerts/  →  data/ground_truth/{sub_range}/alerts/")
    log.info(f"    {DRIVE_FOLDER}/{sub_range}/{gt_name}.tif  →  data/ground_truth/{sub_range}/")
    log.info(f"{'='*65}")

    return True


# ── Phase B: Train + Filter + Vectorize ───────────────────────────────────────

def phase_train(sub_range: str, year: int = 2025):
    """Train model, filter alerts, vectorize — all local, no GEE needed."""
    t0 = time.time()

    # ── Check data readiness ──────────────────────────────────────────
    readiness = check_data_readiness(sub_range)
    if not readiness["ready"]:
        log.error(f"Not ready: {readiness['reason']}")
        log.error(f"Run --phase export first, then download from GDrive")
        return False

    data_dir = readiness["data_dir"]
    alerts_dir = readiness["alerts_dir"]
    alert_tifs = readiness["alert_tifs"]
    n_windows = len(alert_tifs)

    out_dir = os.path.join(MODEL_ROOT, sub_range)
    model_stem = f"model_subrange_{sub_range.lower()}_{year}"
    os.makedirs(out_dir, exist_ok=True)

    log.info(f"\n{'='*65}")
    log.info(f"  PHASE B: TRAIN + FILTER  —  {sub_range}")
    log.info(f"  Data: {data_dir}")
    log.info(f"  Windows: {n_windows}")
    log.info(f"  GT: {readiness['gt_tif']}")
    log.info(f"  Output: {out_dir}")
    log.info(f"{'='*65}\n")

    # ── Step 1: Train ─────────────────────────────────────────────────
    log.info("Step 1: TRAIN LightGBM...")

    # Import the training pipeline
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
    from train_alert_filter import (
        load_alert_stack, load_gt, label_windows,
        spatial_block_split, train_model, evaluate,
        FEATURE_NAMES, DW_BANDS, TREES_IDX, ALERT_THRESH, GT_THRESH
    )

    stack, window_meta, profile = load_alert_stack(data_dir)
    gt = load_gt(data_dir)

    X, y, blocks, windows, window_stats = label_windows(stack, gt, window_meta)

    n_tp = int(y.sum())
    n_total = len(y)
    log.info(f"  Total alert pixels: {n_total:,}")
    log.info(f"  TP (confirmed):     {n_tp:,}")
    log.info(f"  FP (noise):         {n_total - n_tp:,}")

    # ── Fallback check ────────────────────────────────────────────────
    use_global = False
    if n_tp < MIN_TP_PIXELS:
        log.warning(f"\n⚠️  Only {n_tp} TP pixels — below {MIN_TP_PIXELS} threshold!")
        global_model_path = os.path.join(GLOBAL_MODEL, "model.lgbm")
        global_config_path = os.path.join(GLOBAL_MODEL, "feature_config.json")

        if os.path.isfile(global_model_path):
            log.warning(f"   Falling back to GLOBAL model: {global_model_path}")
            use_global = True
        else:
            log.warning(f"   No global model available. Training anyway (may overfit).")

    if use_global:
        import lightgbm as lgb
        import shutil
        model = lgb.Booster(model_file=global_model_path)
        # Copy global model + config to sub-range dir
        shutil.copy2(global_model_path, os.path.join(out_dir, f"{model_stem}.lgbm"))
        shutil.copy2(global_config_path, os.path.join(out_dir, f"{model_stem}_config.json"))

        with open(os.path.join(out_dir, f"{model_stem}_config.json")) as f:
            config = json.load(f)

        report = {
            "sub_range": sub_range,
            "source": "GLOBAL_FALLBACK",
            "reason": f"Only {n_tp} TP pixels (< {MIN_TP_PIXELS})",
            "global_model": global_model_path,
            "n_windows": n_windows,
            "n_tp": n_tp,
            "n_total": n_total,
        }
        with open(os.path.join(out_dir, "training_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"  Using global model (copied to {out_dir})")

    else:
        # Train a local model
        train_mask, val_mask, test_mask = spatial_block_split(blocks)
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        log.info(f"  Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")

        model = train_model(X_train, y_train, X_val, y_val, FEATURE_NAMES)

        model_path = os.path.join(out_dir, f"{model_stem}.lgbm")
        model.save_model(model_path)
        log.info(f"  Model saved: {model_path}")

        # Evaluate
        metrics = evaluate(model, X_test, y_test, FEATURE_NAMES, out_dir)

        # Save config
        config = {
            "feature_names": FEATURE_NAMES,
            "n_features": len(FEATURE_NAMES),
            "alert_threshold": ALERT_THRESH,
            "gt_threshold": GT_THRESH,
            "operational_threshold": metrics["operational_threshold"],
            "model_file": f"{model_stem}.lgbm",
            "band_order": DW_BANDS,
            "trees_band_index": TREES_IDX,
        }
        config_path = os.path.join(out_dir, f"{model_stem}_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        report = {
            "sub_range": sub_range,
            "source": "LOCAL",
            "n_windows": n_windows,
            "n_tp": n_tp,
            "n_total": n_total,
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_test": len(y_test),
            "metrics": metrics,
            "window_stats": window_stats,
        }
        with open(os.path.join(out_dir, "training_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        log.info(f"\n  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
        log.info(f"  Average Precision:  {metrics['average_precision']:.4f}")
        log.info(f"  Op. Threshold:      {metrics['operational_threshold']:.4f}")

    # ── Step 2: Filter (stacked inference) ────────────────────────────
    log.info(f"\nStep 2: FILTER (stacked inference on {n_windows} windows)...")
    from filter_alert_raster import stack_windows, apply_hysteresis, write_filtered_raster
    import lightgbm as lgb

    model_path = os.path.join(out_dir, f"{model_stem}.lgbm")
    config_path = os.path.join(out_dir, f"{model_stem}_config.json")
    with open(config_path) as f:
        config = json.load(f)

    mdl = lgb.Booster(model_file=model_path)
    alert_thresh = config["alert_threshold"]

    score, n_fires, H, W, raster_profile, transform = stack_windows(
        alert_tifs, mdl, alert_thresh, min_fires=2
    )

    # Auto-select threshold
    active = score > 0
    if active.any():
        threshold = float(np.percentile(score[active], 80))
    else:
        threshold = config.get("operational_threshold", 0.5)
    log.info(f"  Threshold: {threshold:.3f}")

    score_2d = score.reshape(H, W)
    confirmed = apply_hysteresis(score_2d, threshold, min_pixels=5).astype(np.float32)
    n_confirmed = int(confirmed.sum())
    log.info(f"  Confirmed pixels: {n_confirmed:,}")

    filtered_tif = os.path.join(out_dir, f"filtered_{sub_range}.tif")
    write_filtered_raster(
        filtered_tif, score, n_fires.astype(np.float32).ravel(),
        confirmed.ravel(), H, W, raster_profile
    )

    # ── Step 3: Vectorize ─────────────────────────────────────────────
    log.info(f"\nStep 3: VECTORIZE...")

    if n_confirmed == 0:
        log.warning(f"  No confirmed pixels — writing empty GeoJSON")
        geojson_path = os.path.join(out_dir, f"alerts_{sub_range}.geojson")
        with open(geojson_path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": []}, f)
    else:
        import rasterio
        import rasterio.features
        from scipy.ndimage import label as ndlabel

        DW_BAND_NAMES = ["water", "trees", "grass", "flooded_veg",
                         "crops", "shrub_scrub", "built", "bare"]
        N_BANDS = 8

        # Use last alert window for change type
        last_tif = alert_tifs[-1]
        with rasterio.open(last_tif) as ds:
            alert_data = ds.read()
        alert_deltas = alert_data[:N_BANDS].astype(np.float32)

        confirmed_mask = confirmed > 0.5
        labeled, n_comp = ndlabel(confirmed_mask.astype(np.int32))

        # Read score/n_fires from filtered raster
        with rasterio.open(filtered_tif) as ds:
            score_map = ds.read(1)
            n_fires_map = ds.read(2)
            raster_transform = ds.transform

        shapes_gen = rasterio.features.shapes(
            labeled.astype(np.int32),
            mask=confirmed_mask,
            transform=raster_transform,
        )

        features_list = []
        seen = set()
        for geom, value in shapes_gen:
            lid = int(value)
            if lid in seen or lid == 0:
                continue
            seen.add(lid)

            poly_mask = labeled == lid
            n_pix = int(poly_mask.sum())
            if n_pix < 5:
                continue

            mean_score = float(np.mean(score_map[poly_mask]))
            mean_fires = float(np.mean(n_fires_map[poly_mask]))

            # Area
            pixel_w_m = abs(raster_transform.a) * 101300
            pixel_h_m = abs(raster_transform.e) * 110600
            area_ha = n_pix * pixel_w_m * pixel_h_m / 10000

            # Centroid
            coords = np.array(geom["coordinates"][0])
            cx = float(np.mean(coords[:, 0]))
            cy = float(np.mean(coords[:, 1]))

            # Change type from last window
            mean_d = {}
            for b, name in enumerate(DW_BAND_NAMES):
                mean_d[name] = float(np.mean(alert_deltas[b][poly_mask]))

            change_type = classify_change_type(mean_d)

            features_list.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "sub_range": sub_range,
                    "change_type": change_type,
                    "stacked_score": round(mean_score, 3),
                    "mean_fires": round(mean_fires, 1),
                    "area_ha": round(area_ha, 4),
                    "n_pixels": n_pix,
                    "centroid": [round(cy, 6), round(cx, 6)],
                }
            })

        geojson = {"type": "FeatureCollection", "features": features_list}
        geojson_path = os.path.join(out_dir, f"alerts_{sub_range}.geojson")
        with open(geojson_path, "w") as f:
            json.dump(geojson, f, indent=2)

        # Summary
        from collections import Counter
        type_counts = Counter(f["properties"]["change_type"] for f in features_list)
        total_area = sum(f["properties"]["area_ha"] for f in features_list)

        log.info(f"  Polygons: {len(features_list)}")
        log.info(f"  Total area: {total_area:.2f} ha")
        for ct, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
            log.info(f"    {ct}: {cnt}")

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info(f"\n{'='*65}")
    log.info(f"  ✅ {sub_range} COMPLETE in {elapsed:.1f}s")
    log.info(f"  Model:    {os.path.join(out_dir, f'{model_stem}.lgbm')}")
    log.info(f"  Raster:   {filtered_tif}")
    log.info(f"  GeoJSON:  {geojson_path}")
    log.info(f"  Report:   {os.path.join(out_dir, 'training_report.json')}")
    log.info(f"{'='*65}")

    return True


def classify_change_type(mean_deltas: dict) -> str:
    """Classify change type from polygon-mean deltas."""
    trees = mean_deltas.get("trees", 0)
    crops = mean_deltas.get("crops", 0)
    bare  = mean_deltas.get("bare", 0)
    built = mean_deltas.get("built", 0)
    shrub = mean_deltas.get("shrub_scrub", 0)
    grass = mean_deltas.get("grass", 0)

    if trees > 0.05:
        return "Greening"
    if trees < -0.05:
        gains = {"crops": crops, "bare": bare, "built": built,
                 "shrub_scrub": shrub, "grass": grass}
        top_band = max(gains, key=gains.get)
        top_val  = gains[top_band]
        if top_band == "crops" and top_val > 0.03:
            return "Encroachment"
        elif top_band == "built" and top_val > 0.03:
            return "Built expansion"
        elif top_band == "bare" and top_val > 0.03:
            return "Clearing"
        elif top_band in ("shrub_scrub", "grass") and top_val > 0.03:
            return "Degradation"
        else:
            return "Tree loss (unclassified)"
    return "Other change"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build per-sub-range ML alert filter model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train one sub-range (data must exist locally):
  python scripts/build_subrange_model.py --sub-range HAMEERPUR --phase train

  # Export data from GEE, then train:
  python scripts/build_subrange_model.py --sub-range HAMEERPUR --phase all

  # Train ALL sub-ranges that have local data:
  python scripts/build_subrange_model.py --all --phase train

  # List available sub-ranges:
  python scripts/build_subrange_model.py --list
        """
    )
    parser.add_argument("--sub-range", type=str, default=None,
                        help="Sub-range name (e.g. HAMEERPUR)")
    parser.add_argument("--all", action="store_true",
                        help="Process all sub-ranges with local data")
    parser.add_argument("--list", action="store_true",
                        help="List available sub-ranges and readiness")
    parser.add_argument("--phase", choices=["export", "train", "all"],
                        default="train",
                        help="Phase: export (GEE), train (local), all (both)")
    parser.add_argument("--year", type=int, default=2025,
                        help="Training year (GT: Dec year-1 → Dec year)")
    args = parser.parse_args()

    # ── List mode ─────────────────────────────────────────────────────
    if args.list:
        log.info(f"\n{'='*65}")
        log.info(f"  Available Sub-Ranges")
        log.info(f"{'='*65}\n")

        log.info("LOCAL (data/ground_truth/):")
        local = list_subranges_local()
        if not local:
            log.info("  (none)")
        for s in local:
            status = "✅ READY" if s["ready"] else "❌ NOT READY"
            log.info(f"  {s['name']:30s}  {s['n_alerts']:3d} windows  "
                     f"{'GT ✓' if s['has_gt'] else 'GT ✗'}  {status}")

        log.info(f"\nGEE (all sub-ranges in {GUNA_ASSET}):")
        gee_subs = list_subranges_gee()
        if gee_subs:
            for s in gee_subs:
                local_match = any(x["name"] == s for x in local)
                marker = " (local data ✓)" if local_match else ""
                log.info(f"  {s}{marker}")
        else:
            log.info("  (could not connect to GEE)")
        return

    # ── Build mode ────────────────────────────────────────────────────
    if args.all:
        # Process all sub-ranges with local data
        local = list_subranges_local()
        ready = [s for s in local if s["ready"]]
        if not ready:
            log.error("No sub-ranges with ready data. Run --list to check.")
            return

        log.info(f"\n{'='*65}")
        log.info(f"  BATCH PROCESSING {len(ready)} SUB-RANGES")
        log.info(f"{'='*65}\n")

        results = {}
        for s in ready:
            sr = s["name"]
            log.info(f"\n{'─'*65}")
            log.info(f"  Processing: {sr}")
            log.info(f"{'─'*65}")

            if args.phase in ("export", "all"):
                phase_export(sr, args.year)
            if args.phase in ("train", "all"):
                success = phase_train(sr, args.year)
                results[sr] = success

        # Summary
        log.info(f"\n\n{'='*65}")
        log.info(f"  BATCH COMPLETE")
        log.info(f"{'='*65}")
        for sr, ok in results.items():
            status = "✅" if ok else "❌"
            log.info(f"  {status} {sr}")
        log.info(f"{'='*65}")

    elif args.sub_range:
        sr = args.sub_range.strip()

        if args.phase in ("export", "all"):
            phase_export(sr, args.year)

        if args.phase in ("train", "all"):
            phase_train(sr, args.year)

    else:
        parser.print_help()
        log.error("\nSpecify --sub-range NAME, --all, or --list")


if __name__ == "__main__":
    main()
