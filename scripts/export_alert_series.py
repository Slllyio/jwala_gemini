"""
Alert Series Export — Consecutive-pair DW deltas for a full year
================================================================

For a given SUB_RANGE (e.g. HAMEERPUR) and year (e.g. 2025):

  1. Lists every unique DW acquisition date over the AOI
  2. Same-date images → median mosaic
  3. For each consecutive pair (d_i, d_{i+1}):
     - 8-band prob delta  = DW(d_{i+1}) − DW(d_i)
     - Band  9: cloud_prob_before  (S2_CLOUD_PROBABILITY for d_i,   0..1)
     - Band 10: cloud_prob_after   (S2_CLOUD_PROBABILITY for d_{i+1}, 0..1)
     - Threshold check (server-side):
         trees:       min(Δtrees)            < -tree_threshold?   → triggered
         other bands: max(|Δ_other_band|)    >  other_threshold?  → triggered
     - If triggered → export 10-band GeoTIFF
     - Else → skip

Cloud bands sourced from COPERNICUS/S2_CLOUD_PROBABILITY (0–100 → scaled 0.0–1.0).

Usage:
    # Dry-run: list dates + threshold check, no exports
    python scripts/export_alert_series.py --sub-range HAMEERPUR --year 2025 --dry-run

    # Live run: submit GEE export tasks
    python scripts/export_alert_series.py --sub-range HAMEERPUR --year 2025

    # Custom thresholds
    python scripts/export_alert_series.py --sub-range HAMEERPUR --year 2025 \\
        --tree-threshold 0.20 --other-threshold 0.30
"""

import ee
import argparse
import logging
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

# Index of the 'trees' band within DW_BANDS
TREES_IDX = DW_BANDS.index("trees")


# ── AOI helper ────────────────────────────────────────────────────────────────

def get_sub_range_aoi(fc: ee.FeatureCollection, sub_range: str) -> ee.Geometry:
    """Filter the beat asset to a single SUB_RANGE and dissolve."""
    sub_fc = fc.filter(ee.Filter.eq("SUB_RANGE", sub_range))
    count  = sub_fc.size().getInfo()
    if count == 0:
        all_sr = fc.aggregate_array("SUB_RANGE").getInfo()
        unique = sorted(set(str(x) for x in all_sr if x))
        raise RuntimeError(
            f"No features for SUB_RANGE='{sub_range}'. Available: {unique}"
        )
    log.info(f"  Found {count} beats in SUB_RANGE='{sub_range}'")
    return sub_fc.geometry().dissolve(maxError=10)


# ── Date listing ──────────────────────────────────────────────────────────────

def list_dw_dates(aoi: ee.Geometry, year: int) -> list[str]:
    """
    Returns sorted list of unique DW acquisition dates (YYYY-MM-DD strings)
    for the given AOI and year.
    """
    col = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(aoi)
    )

    timestamps = col.aggregate_array("system:time_start").getInfo()
    if not timestamps:
        raise RuntimeError(f"No DW imagery found for {year} in this AOI.")

    dates = sorted(set(
        datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        for ms in timestamps
    ))
    return dates


# ── Single-date image builders ────────────────────────────────────────────────

def dw_single_date(date_str: str, aoi: ee.Geometry) -> ee.Image:
    """
    Median of all DW images on a single date, 8 probability bands.
    Handles multiple S2 tiles covering the AOI on the same day.
    """
    next_day = (
        datetime.fromisoformat(date_str) + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    return (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(date_str, next_day)
        .filterBounds(aoi)
        .select(DW_BANDS)
        .median()
        .clip(aoi)
    )


def cloud_prob_single_date(date_str: str, aoi: ee.Geometry) -> ee.Image:
    """
    Per-pixel cloud probability from S2_CLOUD_PROBABILITY for a single date.
    Scaled from 0–100 → 0.0–1.0.
    Returns median across tiles if multiple S2 scenes on the same day.
    """
    next_day = (
        datetime.fromisoformat(date_str) + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    cloud = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterDate(date_str, next_day)
        .filterBounds(aoi)
        .select("probability")
        .median()
        .clip(aoi)
    )

    # Scale 0–100 → 0.0–1.0
    return cloud.divide(100.0).rename("cloud_prob")


def s2_spectral_single_date(date_str: str, aoi: ee.Geometry) -> ee.Image:
    """
    Compute NDVI, NBR, NDMI from raw Sentinel-2 L2A reflectance for a single date.
    Returns 3-band image: [ndvi, nbr, ndmi], each in range [-1, 1].
    """
    next_day = (
        datetime.fromisoformat(date_str) + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(date_str, next_day)
        .filterBounds(aoi)
        .select(["B4", "B8", "B11", "B12"])  # Red, NIR, SWIR1, SWIR2
        .median()
        .clip(aoi)
    )

    # Scale reflectance (0–10000) → 0.0–1.0
    s2 = s2.divide(10000.0)

    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("ndvi")
    # NBR = (NIR - SWIR2) / (NIR + SWIR2)
    nbr = s2.normalizedDifference(["B8", "B12"]).rename("nbr")
    # NDMI = (NIR - SWIR1) / (NIR + SWIR1)
    ndmi = s2.normalizedDifference(["B8", "B11"]).rename("ndmi")

    return ndvi.addBands(nbr).addBands(ndmi)


S2_RAW_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]  # Blue, Green, Red, NIR, SWIR1, SWIR2


def s2_raw_single_date(date_str: str, aoi: ee.Geometry, suffix: str = "") -> ee.Image:
    """
    Raw Sentinel-2 L2A reflectance for 6 key bands, scaled 0–1.
    Returns 6-band image: [B2, B3, B4, B8, B11, B12] (optionally with suffix).
    """
    next_day = (
        datetime.fromisoformat(date_str) + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(date_str, next_day)
        .filterBounds(aoi)
        .select(S2_RAW_BANDS)
        .median()
        .clip(aoi)
    )

    # Scale reflectance (0–10000) → 0.0–1.0
    s2 = s2.divide(10000.0)

    if suffix:
        s2 = s2.select(S2_RAW_BANDS, [b + suffix for b in S2_RAW_BANDS])

    return s2


# ── Threshold check ───────────────────────────────────────────────────────────

def check_threshold(delta: ee.Image, aoi: ee.Geometry,
                    tree_thresh: float, other_thresh: float) -> dict:
    """
    Server-side threshold check on the 8-band delta image.

    Returns dict with:
      trees_min:        float — most negative Δtrees pixel
      others_max_abs:   float — largest |Δ| across all non-trees bands
      trees_triggered:  bool  — trees_min < -tree_thresh
      others_triggered: bool  — others_max_abs > other_thresh
      export:           bool  — either triggered
    """
    # Trees band: find the minimum (most negative = most tree loss)
    trees_band = delta.select("trees")
    trees_stats = trees_band.reduceRegion(
        reducer=ee.Reducer.min(),
        geometry=aoi,
        scale=SCALE,
        maxPixels=MAX_PIXELS,
        bestEffort=True,
    ).getInfo()
    trees_min = trees_stats.get("trees", 0.0)
    if trees_min is None:
        trees_min = 0.0

    # Other bands: find the max absolute delta
    other_band_names = [b for b in DW_BANDS if b != "trees"]
    other_bands = delta.select(other_band_names)
    other_abs = other_bands.abs()
    # Reduce to single band (max across bands per pixel), then max across pixels
    other_max_per_pixel = other_abs.reduce(ee.Reducer.max())
    other_stats = other_max_per_pixel.reduceRegion(
        reducer=ee.Reducer.max(),
        geometry=aoi,
        scale=SCALE,
        maxPixels=MAX_PIXELS,
        bestEffort=True,
    ).getInfo()
    others_max = other_stats.get("max", 0.0)
    if others_max is None:
        others_max = 0.0

    trees_triggered  = float(trees_min) < -tree_thresh
    others_triggered = float(others_max) > other_thresh

    return {
        "trees_min":        round(float(trees_min), 4),
        "others_max_abs":   round(float(others_max), 4),
        "trees_triggered":  trees_triggered,
        "others_triggered": others_triggered,
        "export":           trees_triggered or others_triggered,
    }


# ── GEE export ────────────────────────────────────────────────────────────────

def submit_export(img: ee.Image, name: str, aoi: ee.Geometry,
                  folder: str) -> None:
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
    log.info(f"      ✅ Task submitted: {name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export consecutive-pair alert delta rasters for a full year"
    )
    parser.add_argument("--sub-range",       type=str, required=True,
                        help="SUB_RANGE name (e.g. HAMEERPUR)")
    parser.add_argument("--year",            type=int, default=2025)
    parser.add_argument("--tree-threshold",  type=float, default=0.25,
                        help="Export if min(Δtrees) < -threshold (default 0.25)")
    parser.add_argument("--other-threshold", type=float, default=0.25,
                        help="Export if max(|Δother|) > threshold (default 0.25)")
    parser.add_argument("--dry-run",         action="store_true",
                        help="List dates + threshold check, no exports")
    parser.add_argument("--enriched",        action="store_true",
                        help="Export enriched 32-band rasters (raw probs + S2 spectral)")
    parser.add_argument("--cancel-tasks",    action="store_true",
                        help="Cancel all pending/running GEE tasks before exporting")
    parser.add_argument("--project",         default=GEE_PROJECT)
    args = parser.parse_args()

    ee.Initialize(project=args.project)

    sr   = args.sub_range.strip()
    year = args.year

    # Cancel pending/running GEE tasks if requested
    if args.cancel_tasks:
        log.info("Cancelling pending/running GEE tasks...")
        tasks = ee.batch.Task.list()
        n_cancelled = 0
        for t in tasks:
            if t.state in ("READY", "RUNNING"):
                t.cancel()
                n_cancelled += 1
                log.info(f"  Cancelled: {t.config.get('description', t.id)} ({t.state})")
        log.info(f"  Cancelled {n_cancelled} tasks\n")

    fmt = "ENRICHED (32-band)" if args.enriched else "LEGACY (10-band)"
    log.info(f"\n{'='*65}")
    log.info(f"  Alert Series Export")
    log.info(f"  SUB_RANGE:       {sr}")
    log.info(f"  Year:            {year}")
    log.info(f"  Format:          {fmt}")
    log.info(f"  Tree threshold:  Δtrees < -{args.tree_threshold}")
    log.info(f"  Other threshold: |Δother| > {args.other_threshold}")
    log.info(f"  Mode:            {'DRY-RUN' if args.dry_run else 'LIVE'}")
    log.info(f"{'='*65}\n")

    fc  = ee.FeatureCollection(GUNA_ASSET)
    aoi = get_sub_range_aoi(fc, sr)

    # ── Step 1: list all DW dates ─────────────────────────────────────────
    log.info(f"Listing DW acquisition dates for {year} ...")
    dates = list_dw_dates(aoi, year)
    log.info(f"  Found {len(dates)} unique dates")
    log.info(f"  First: {dates[0]}  →  Last: {dates[-1]}\n")

    n_pairs    = len(dates) - 1
    n_exported = 0
    n_skipped  = 0

    drive_folder = f"{DRIVE_FOLDER}/{sr}/alerts"

    # ── Step 2: iterate consecutive pairs ─────────────────────────────────
    for i in range(n_pairs):
        d_before = dates[i]
        d_after  = dates[i + 1]
        pair_label = f"[{i+1:3d}/{n_pairs}]  {d_before} → {d_after}"

        log.info(f"  {pair_label}")

        # Build single-date medians
        dw_before = dw_single_date(d_before, aoi)
        dw_after  = dw_single_date(d_after, aoi)

        # 8-band delta
        delta = dw_after.subtract(dw_before)

        # Threshold check (requires server round-trip)
        result = check_threshold(delta, aoi,
                                 args.tree_threshold, args.other_threshold)

        flag = ""
        if result["trees_triggered"]:
            flag += f"  🌲 Δtrees_min={result['trees_min']}"
        if result["others_triggered"]:
            flag += f"  📊 |Δother|_max={result['others_max_abs']}"

        if not result["export"]:
            log.info(f"    ⏭  SKIP  (trees_min={result['trees_min']}, "
                     f"others_max={result['others_max_abs']})")
            n_skipped += 1
            continue

        log.info(f"    🔥 TRIGGERED{flag}")

        if args.dry_run:
            log.info(f"    (dry-run — would export)")
            n_exported += 1
            continue

        # Build cloud probability bands
        cld_before = cloud_prob_single_date(d_before, aoi)
        cld_after  = cloud_prob_single_date(d_after, aoi)

        if args.enriched:
            # ── Enriched 44-band raster ──────────────────────────
            # Bands 1-8:   DW raw probabilities BEFORE
            # Bands 9-16:  DW raw probabilities AFTER
            # Bands 17-24: DW delta (after - before)
            # Bands 25-26: Cloud prob (before, after)
            # Bands 27-29: S2 spectral BEFORE (ndvi, nbr, ndmi)
            # Bands 30-32: S2 spectral AFTER (ndvi, nbr, ndmi)
            # Bands 33-38: S2 raw reflectance BEFORE (B2, B3, B4, B8, B11, B12)
            # Bands 39-44: S2 raw reflectance AFTER (B2, B3, B4, B8, B11, B12)
            dw_before_renamed = dw_before.select(
                DW_BANDS,
                [b + "_before" for b in DW_BANDS]
            )
            dw_after_renamed = dw_after.select(
                DW_BANDS,
                [b + "_after" for b in DW_BANDS]
            )
            delta_renamed = delta.select(
                DW_BANDS,
                ["delta_" + b for b in DW_BANDS]
            )

            # S2 spectral indices
            spec_before = s2_spectral_single_date(d_before, aoi).select(
                ["ndvi", "nbr", "ndmi"],
                ["ndvi_before", "nbr_before", "ndmi_before"]
            )
            spec_after = s2_spectral_single_date(d_after, aoi).select(
                ["ndvi", "nbr", "ndmi"],
                ["ndvi_after", "nbr_after", "ndmi_after"]
            )

            # Raw S2 reflectance bands
            s2_raw_before = s2_raw_single_date(d_before, aoi, suffix="_before")
            s2_raw_after  = s2_raw_single_date(d_after, aoi, suffix="_after")

            out = (dw_before_renamed
                   .addBands(dw_after_renamed)
                   .addBands(delta_renamed)
                   .addBands(cld_before.rename("cloud_prob_before"))
                   .addBands(cld_after.rename("cloud_prob_after"))
                   .addBands(spec_before)
                   .addBands(spec_after)
                   .addBands(s2_raw_before)
                   .addBands(s2_raw_after))

            name = f"alert_enriched_{sr}_{d_before}_to_{d_after}"
            log.info(f"    📦 Enriched: {out.bandNames().size().getInfo()} bands")
        else:
            # ── Legacy 10-band raster ────────────────────────────
            out = (
                delta
                .addBands(cld_before.rename("cloud_prob_before"))
                .addBands(cld_after.rename("cloud_prob_after"))
            )
            name = f"alert_delta_{sr}_{d_before}_to_{d_after}"

        submit_export(out, name, aoi, drive_folder)
        n_exported += 1

    # ── Summary ───────────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info(f"  Summary")
    log.info(f"    Unique DW dates:  {len(dates)}")
    log.info(f"    Consecutive pairs checked:  {n_pairs}")
    log.info(f"    Exported (triggered):       {n_exported}")
    log.info(f"    Skipped  (below threshold): {n_skipped}")
    log.info(f"")
    if not args.dry_run and n_exported > 0:
        log.info(f"  Drive folder: {drive_folder}/")
        log.info(f"  Monitor: https://code.earthengine.google.com/tasks")
    log.info(f"{'='*65}")


if __name__ == "__main__":
    main()
