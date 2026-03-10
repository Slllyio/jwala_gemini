"""
Batch Export — GT delta + Alert series for ALL sub-ranges
=========================================================

Runs sequentially:
  1. GT delta (8-band, Dec 2024 → Dec 2025) for each sub-range
  2. Alert series (consecutive-pair 10-band deltas, full 2025) for each sub-range

Each GEE export task is submitted to Google Drive.
The script does NOT wait for tasks to complete — it submits and moves on.

Usage:
    # Submit all sub-ranges (GT + alerts)
    python scripts/batch_export_all.py

    # GT only
    python scripts/batch_export_all.py --gt-only

    # Alerts only
    python scripts/batch_export_all.py --alerts-only

    # Skip HAMEERPUR (already done)
    python scripts/batch_export_all.py --skip HAMEERPUR

    # Only do a specific list
    python scripts/batch_export_all.py --only ARON GUNA BAMORI
"""

import ee
import argparse
import logging
import time
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GEE_PROJECT  = "van-suraksha-alert"
GUNA_ASSET   = "projects/van-suraksha-alert/assets/gunafinal"
DRIVE_FOLDER = "prithvi_guna_v4"
SCALE        = 10
MAX_PIXELS   = 1e10

DW_BANDS = [
    "water", "trees", "grass", "flooded_vegetation",
    "crops", "shrub_and_scrub", "built", "bare",
]
TREES_IDX = DW_BANDS.index("trees")


# ── Helpers (reused from export scripts) ──────────────────────────────────────

def get_sub_range_aoi(fc, sub_range):
    sub_fc = fc.filter(ee.Filter.eq("SUB_RANGE", sub_range))
    count  = sub_fc.size().getInfo()
    if count == 0:
        return None, 0
    return sub_fc.geometry().dissolve(maxError=10), count


def dec_median(year, aoi):
    return (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(f"{year}-12-01", f"{year}-12-31")
        .filterBounds(aoi)
        .select(DW_BANDS)
        .median()
        .clip(aoi)
    )


def list_dw_dates(aoi, year):
    col = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(aoi)
    )
    timestamps = col.aggregate_array("system:time_start").getInfo()
    if not timestamps:
        return []
    dates = sorted(set(
        datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        for ms in timestamps
    ))
    return dates


def dw_single_date(date_str, aoi):
    next_day = (datetime.fromisoformat(date_str) + timedelta(days=1)).strftime("%Y-%m-%d")
    return (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(date_str, next_day)
        .filterBounds(aoi)
        .select(DW_BANDS)
        .median()
        .clip(aoi)
    )


def cloud_prob_single_date(date_str, aoi):
    next_day = (datetime.fromisoformat(date_str) + timedelta(days=1)).strftime("%Y-%m-%d")
    cloud = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterDate(date_str, next_day)
        .filterBounds(aoi)
        .select("probability")
        .median()
        .clip(aoi)
    )
    return cloud.divide(100.0).rename("cloud_prob")


def check_threshold(delta, aoi, tree_thresh=0.25, other_thresh=0.25):
    trees_band = delta.select("trees")
    trees_stats = trees_band.reduceRegion(
        reducer=ee.Reducer.min(),
        geometry=aoi, scale=SCALE, maxPixels=MAX_PIXELS, bestEffort=True,
    ).getInfo()
    trees_min = trees_stats.get("trees", 0.0)
    if trees_min is None:
        trees_min = 0.0

    other_band_names = [b for b in DW_BANDS if b != "trees"]
    other_abs = delta.select(other_band_names).abs()
    other_max_per_pixel = other_abs.reduce(ee.Reducer.max())
    other_stats = other_max_per_pixel.reduceRegion(
        reducer=ee.Reducer.max(),
        geometry=aoi, scale=SCALE, maxPixels=MAX_PIXELS, bestEffort=True,
    ).getInfo()
    others_max = other_stats.get("max", 0.0)
    if others_max is None:
        others_max = 0.0

    return float(trees_min) < -tree_thresh or float(others_max) > other_thresh


def submit_export(img, name, aoi, folder):
    task = ee.batch.Export.image.toDrive(
        image=img.toFloat(),
        description=name[:100],
        folder=folder,
        fileNamePrefix=name,
        region=aoi,
        scale=SCALE,
        crs="EPSG:4326",
        maxPixels=MAX_PIXELS,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


# ── Export GT delta for one sub-range ─────────────────────────────────────────

def export_gt(sr, aoi, year_a=2024, year_b=2025):
    """Export 8-band GT delta (Dec year_a → Dec year_b)."""
    img_a = dec_median(year_a, aoi)
    img_b = dec_median(year_b, aoi)
    delta = img_b.subtract(img_a)

    name = f"gt_delta_{year_a}_{year_b}_{sr}"
    folder = f"{DRIVE_FOLDER}/{sr}"
    submit_export(delta, name, aoi, folder)
    log.info(f"    ✅ GT delta submitted: {name}")
    return 1


# ── Export alert series for one sub-range ─────────────────────────────────────

def export_alerts(sr, aoi, year=2025, tree_thresh=0.25, other_thresh=0.25):
    """Export consecutive-pair alert deltas for the full year."""
    dates = list_dw_dates(aoi, year)
    if len(dates) < 2:
        log.warning(f"    ⚠️  Only {len(dates)} DW dates — skipping alerts")
        return 0

    log.info(f"    Found {len(dates)} DW dates, checking {len(dates)-1} pairs ...")

    n_exported = 0
    n_pairs = len(dates) - 1

    for i in range(n_pairs):
        d_before = dates[i]
        d_after  = dates[i + 1]

        dw_before = dw_single_date(d_before, aoi)
        dw_after  = dw_single_date(d_after, aoi)
        delta = dw_after.subtract(dw_before)

        triggered = check_threshold(delta, aoi, tree_thresh, other_thresh)
        if not triggered:
            continue

        cld_before = cloud_prob_single_date(d_before, aoi)
        cld_after  = cloud_prob_single_date(d_after, aoi)
        out = (
            delta
            .addBands(cld_before.rename("cloud_prob_before"))
            .addBands(cld_after.rename("cloud_prob_after"))
        )

        name = f"alert_delta_{sr}_{d_before}_to_{d_after}"
        folder = f"{DRIVE_FOLDER}/{sr}/alerts"
        submit_export(out, name, aoi, folder)
        n_exported += 1

    log.info(f"    ✅ Alerts: {n_exported} exported, {n_pairs - n_exported} skipped")
    return n_exported


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-only",     action="store_true")
    parser.add_argument("--alerts-only", action="store_true")
    parser.add_argument("--skip",        nargs="*", default=[])
    parser.add_argument("--only",        nargs="*", default=[])
    parser.add_argument("--year",        type=int, default=2025)
    parser.add_argument("--year-a",      type=int, default=2024)
    parser.add_argument("--year-b",      type=int, default=2025)
    parser.add_argument("--project",     default=GEE_PROJECT)
    args = parser.parse_args()

    ee.Initialize(project=args.project)

    fc = ee.FeatureCollection(GUNA_ASSET)
    all_sr = fc.aggregate_array("SUB_RANGE").getInfo()
    unique_sr = sorted(set(str(x) for x in all_sr if x and str(x).strip()))

    skip_set = set(s.upper() for s in args.skip)
    if args.only:
        only_set = set(s.upper() for s in args.only)
        unique_sr = [s for s in unique_sr if s.upper() in only_set]

    unique_sr = [s for s in unique_sr if s.upper() not in skip_set]

    log.info(f"\n{'='*65}")
    log.info(f"  Batch Export — {len(unique_sr)} sub-ranges")
    log.info(f"  GT:     {'YES' if not args.alerts_only else 'SKIP'}")
    log.info(f"  Alerts: {'YES' if not args.gt_only else 'SKIP'}")
    log.info(f"  Skipping: {list(skip_set) if skip_set else 'none'}")
    log.info(f"{'='*65}\n")

    total_gt = 0
    total_alerts = 0

    for idx, sr in enumerate(unique_sr, 1):
        log.info(f"\n[{idx}/{len(unique_sr)}] ── {sr} ──")

        aoi, n_beats = get_sub_range_aoi(fc, sr)
        if aoi is None:
            log.warning(f"  No beats found — skipping")
            continue
        log.info(f"  {n_beats} beats")

        try:
            # GT delta
            if not args.alerts_only:
                export_gt(sr, aoi, args.year_a, args.year_b)
                total_gt += 1

            # Alert series
            if not args.gt_only:
                n = export_alerts(sr, aoi, args.year)
                total_alerts += n

        except Exception as e:
            log.error(f"  ❌ ERROR: {e}")
            continue

        # Small delay to be nice to GEE API
        time.sleep(1)

    log.info(f"\n{'='*65}")
    log.info(f"  BATCH COMPLETE")
    log.info(f"  GT delta tasks submitted:    {total_gt}")
    log.info(f"  Alert delta tasks submitted: {total_alerts}")
    log.info(f"  Monitor: https://code.earthengine.google.com/tasks")
    log.info(f"{'='*65}")


if __name__ == "__main__":
    main()
