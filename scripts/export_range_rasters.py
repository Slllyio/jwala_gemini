"""
Range-split DW Raster Exporter
================================
For each Forest Range in Guna Division, exports TWO 10-band rasters to Drive:

  1. GT DELTA  (annual ground-truth labels)
     Dec year_a median  →  Dec year_b median
     Bands 1-8 : DW prob deltas  (deforestation/encroachment labels)
     Band  9   : cloud_fraction_before  (Dec year_a)
     Band 10   : cloud_fraction_after   (Dec year_b)
     File: gt_delta_{year_a}_{year_b}_{range}.tif

  2. ALERT DELTA  (near-real-time change signal)
     Second-latest available DW composite for range  →  Latest available
     Bands 1-8 : DW prob deltas  (what just changed)
     Band  9   : cloud_fraction_before  (second-latest window)
     Band 10   : cloud_fraction_after   (latest window)
     File: alert_delta_latest_{range}.tif

Why two separate rasters?
    GT delta   = what actually changed OVER THE FULL YEAR  → labelling / training
    Alert delta = what changed IN THE LAST FEW DAYS/WEEKS  → triggering alerts

Comparison:
    If alert_delta shows Δtrees < -0.5  at pixel X
    AND gt_delta also shows Δtrees < -0.5 at same pixel X
    → alert is confirming a real annual-scale change event  ✅

    If alert_delta fires but gt_delta shows no change at X
    → possibly premature alert or phenological noise  ❓

AOI: projects/van-suraksha-alert/assets/gunafinal

Drive folder: prithvi_guna_v4/{range}/

Usage:
    # Export all ranges (both GT + alert rasters)
    python scripts/export_range_rasters.py

    # Export only specific range
    python scripts/export_range_rasters.py --range North_Guna

    # Only GT (skip alert delta)
    python scripts/export_range_rasters.py --gt-only

    # Only alert delta (skip annual GT)
    python scripts/export_range_rasters.py --alert-only

    # Custom annual window
    python scripts/export_range_rasters.py --year-a 2023 --year-b 2024

    # Custom alert window size (days)
    python scripts/export_range_rasters.py --window-days 10
"""

import ee
import argparse
import logging
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GEE_PROJECT   = "van-suraksha-alert"
GUNA_ASSET    = "projects/van-suraksha-alert/assets/gunafinal"
DRIVE_FOLDER  = "prithvi_guna_v4"
SCALE         = 10          # 10m native DW resolution
MAX_PIXELS    = 1e10

DW_BANDS = [
    "water", "trees", "grass", "flooded_vegetation",
    "crops", "shrub_and_scrub", "built", "bare",
]

# ── Range extraction ──────────────────────────────────────────────────────────

def get_ranges(fc: ee.FeatureCollection) -> dict:
    """
    Dissolves beats into Forest Range geometries.
    Returns {range_name: ee.Geometry}

    Strategy:
      1. Download ONLY property strings (Beat_Name / Range) — no geometry download.
      2. Derive range name per beat client-side from the string prefix.
      3. For each range, filter the server-side FC and call .geometry().dissolve()
         — GEE handles any MultiPolygon / GeometryCollection types natively.

    Beat_Name prefix rules:
      "North_Guna_Jamner" → "North_Guna"
      "South_Guna_Jaita"  → "South_Guna"
      "Aron_Goumukh"      → "Aron"
      "Raghogarh_Narolai" → "Raghogarh"
    """
    # Detect property key names from the first feature
    # Actual gunafinal schema: BEAT, RANGE, SUB_RANGE, NEW_No_, area2015, areaOrigin
    first_props = fc.first().toDictionary().getInfo()
    prop_key_beat  = None
    prop_key_range = None
    for k in first_props:
        kl = k.lower()
        # Range property — prefer exact 'RANGE' / 'range' / 'forest_range'
        if kl in ("range", "forest_range", "range_name"):
            prop_key_range = k
        # Beat property — any key containing 'beat'
        if "beat" in kl:
            prop_key_beat = k

    # If no beat property found, fall back to whatever looks like a name
    if prop_key_beat is None:
        # Try SUB_RANGE or any string-valued key as the beat identifier
        for k, v in first_props.items():
            if isinstance(v, str) and k not in (prop_key_range,):
                prop_key_beat = k
                break

    log.info(f"  Asset properties detected: {list(first_props.keys())}")
    log.info(f"  Using beat_key='{prop_key_beat}', range_key='{prop_key_range}'")

    # Get all (Beat_Name, Range) pairs — just strings, no geometry download
    beat_names  = fc.aggregate_array(prop_key_beat).getInfo()
    range_names = (fc.aggregate_array(prop_key_range).getInfo()
                   if prop_key_range else [None] * len(beat_names))

    # Build range → beat_name mapping
    range_to_beats: dict[str, list[str]] = {}
    for beat_raw, rng_raw in zip(beat_names, range_names):
        beat_raw = str(beat_raw or "Unknown")
        rng = str(rng_raw).strip() if rng_raw is not None else None
        if not rng or rng.lower() in ("none", "null", ""):
            parts = beat_raw.split("_")
            if (len(parts) >= 2
                    and parts[0].lower() in ("north", "south", "east", "west")):
                rng = f"{parts[0]}_{parts[1]}"
            else:
                rng = parts[0]
        rng = rng.strip().replace(" ", "_")
        range_to_beats.setdefault(rng, []).append(beat_raw)

    log.info(f"  Ranges found: {sorted(range_to_beats.keys())}")

    # Server-side dissolve per range using ee.Filter — GEE handles geometry types
    ranges = {}
    for rng, beats in range_to_beats.items():
        sub_fc   = fc.filter(ee.Filter.inList(prop_key_beat, beats))
        dissolved = sub_fc.geometry().dissolve(maxError=10)
        ranges[rng] = dissolved

    return ranges


# ── DW helpers ────────────────────────────────────────────────────────────────

def dec_median(year: int, aoi: ee.Geometry) -> ee.Image:
    """December median composite, 8 DW bands, clipped to aoi."""
    return (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(f"{year}-12-01", f"{year}-12-31")
        .filterBounds(aoi)
        .select(DW_BANDS)
        .median()
        .clip(aoi)
    )


def window_median(start: str, end: str, aoi: ee.Geometry) -> ee.Image:
    """Short-window median composite for a date range string 'YYYY-MM-DD'."""
    return (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(start, end)
        .filterBounds(aoi)
        .select(DW_BANDS)
        .median()
        .clip(aoi)
    )


def cloud_fraction_for_window(start: str, end: str, aoi: ee.Geometry) -> ee.Image:
    """
    Cloud fraction per pixel = 1 - (valid_obs / total_scenes) for the window.
    Valid observation = non-null DW 'trees' value at that pixel.
    """
    col         = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                   .filterDate(start, end)
                   .filterBounds(aoi))
    total       = col.size()
    valid_count = col.select("trees").count()
    frac        = (
        ee.Image.constant(1)
        .subtract(valid_count.divide(ee.Image.constant(total)))
        .rename("cloud_fraction")
        .clip(aoi)
    )
    return frac


def cloud_fraction_dec(year: int, aoi: ee.Geometry) -> ee.Image:
    return cloud_fraction_for_window(f"{year}-12-01", f"{year}-12-31", aoi)


def find_latest_windows(aoi: ee.Geometry, window_days: int = 15) -> tuple[tuple, tuple]:
    """
    Find the two most recent DW acquisition windows for the AOI.
    Returns (latest_window, second_latest_window) as (start_str, end_str) tuples.

    Strategy:
      - Scan the DW collection for this AOI over the past 12 months
      - Bin into non-overlapping `window_days` periods
      - Return the two most recent bins that have >= 1 valid scene
    """
    today     = datetime.now(timezone.utc).date()
    scan_back = 365  # look back at most 1 year

    # Get all available dates in the DW collection for this AOI
    col = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
           .filterDate(
               (today - timedelta(days=scan_back)).isoformat(),
               today.isoformat()
           )
           .filterBounds(aoi))

    dates_info = col.aggregate_array("system:time_start").getInfo()
    if not dates_info:
        raise RuntimeError("No DW imagery found in last 365 days for this AOI.")

    # Convert to sorted dates (most-recent first)
    acq_dates = sorted(
        set(datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date()
            for ms in dates_info),
        reverse=True
    )

    log.debug(f"    Available DW dates (most recent first): {acq_dates[:5]} ...")

    # Build two non-overlapping windows backwards from the most recent date
    def make_window(anchor_date):
        end   = anchor_date
        start = anchor_date - timedelta(days=window_days)
        return start.isoformat(), end.isoformat()

    latest_anchor = acq_dates[0]
    latest_win    = make_window(latest_anchor)

    # Second window: find the most recent date BEFORE the latest_window start
    second_cutoff = datetime.fromisoformat(latest_win[0]).date()
    older_dates   = [d for d in acq_dates if d < second_cutoff]
    if not older_dates:
        raise RuntimeError(
            f"Could not find a second acquisition window before {latest_win[0]}. "
            f"Try increasing --window-days."
        )
    second_anchor = older_dates[0]
    second_win    = make_window(second_anchor)

    return latest_win, second_win


# ── 10-band builder ───────────────────────────────────────────────────────────

def make_10band(img_before: ee.Image, img_after: ee.Image,
                cld_before: ee.Image, cld_after: ee.Image) -> ee.Image:
    """
    Stack: 8-band delta + cloud_before + cloud_after = 10 bands.
    delta range: -1 (class lost) to +1 (class gained).
    cloud range:  0 (all clear)  to  1  (fully cloudy).
    """
    delta = img_after.subtract(img_before)
    return (
        delta
        .addBands(cld_before.rename("cloud_before"))
        .addBands(cld_after.rename("cloud_after"))
    )


# ── GEE export helper ─────────────────────────────────────────────────────────

def submit(img: ee.Image, name: str, aoi: ee.Geometry,
           folder: str = DRIVE_FOLDER) -> None:
    task = ee.batch.Export.image.toDrive(
        image=img.toFloat(),
        description=name[:100],   # GEE max 100 chars
        folder=folder,
        fileNamePrefix=name,
        region=aoi,
        scale=SCALE,
        crs="EPSG:4326",
        maxPixels=MAX_PIXELS,
        fileFormat="GeoTIFF",
    )
    task.start()
    log.info(f"      ✅ {name}")


# ── Per-range export ──────────────────────────────────────────────────────────

def export_gt_delta(range_name: str, aoi: ee.Geometry,
                    year_a: int, year_b: int) -> None:
    """
    GT raster: Dec year_a median → Dec year_b median.
    Pure 8-band delta (one per DW class).
    No cloud bands — December median composite (all DW scenes in the month)
    already averages out cloud noise.
    """
    log.info(f"    GT delta  (Dec {year_a} → Dec {year_b})  [8-band, no cloud] ...")

    img_a = dec_median(year_a, aoi)   # 8-band median prob
    img_b = dec_median(year_b, aoi)   # 8-band median prob
    delta = img_b.subtract(img_a)     # 8-band delta, range -1..+1

    name  = f"gt_delta_{year_a}_{year_b}_{range_name}"
    submit(delta, name, aoi, folder=f"{DRIVE_FOLDER}/{range_name}")


def export_alert_delta(range_name: str, aoi: ee.Geometry,
                       window_days: int) -> None:
    """
    Alert raster: second-latest DW window → latest DW window.
    Near-real-time change signal for alert generation.
    """
    log.info(f"    Alert delta  (latest {window_days}-day windows) ...")

    try:
        latest_win, second_win = find_latest_windows(aoi, window_days)
    except RuntimeError as e:
        log.warning(f"      ⚠️  {e}  — skipping alert delta for {range_name}")
        return

    log.info(f"      Second-latest: {second_win[0]} → {second_win[1]}")
    log.info(f"      Latest:        {latest_win[0]}  → {latest_win[1]}")

    img_before = window_median(*second_win, aoi)
    img_after  = window_median(*latest_win,  aoi)
    cld_before = cloud_fraction_for_window(*second_win, aoi)
    cld_after  = cloud_fraction_for_window(*latest_win,  aoi)

    out  = make_10band(img_before, img_after, cld_before, cld_after)
    name = f"alert_delta_latest_{range_name}"
    submit(out, name, aoi, folder=f"{DRIVE_FOLDER}/{range_name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export GT delta + alert delta rasters per Forest Range"
    )
    parser.add_argument("--year-a",      type=int, default=2024,
                        help="Reference year for annual GT delta (default: 2024)")
    parser.add_argument("--year-b",      type=int, default=2025,
                        help="Target year for annual GT delta (default: 2025)")
    parser.add_argument("--range",       type=str, default=None,
                        help="Export only this RANGE value (e.g. North_Guna)")
    parser.add_argument("--sub-range",   type=str, default=None,
                        help="Export only a single SUB_RANGE beat (e.g. hameerpur). "
                             "Overrides --range.")
    parser.add_argument("--gt-only",     action="store_true",
                        help="Skip alert delta export")
    parser.add_argument("--alert-only",  action="store_true",
                        help="Skip GT delta export")
    parser.add_argument("--window-days", type=int, default=15,
                        help="Days in each near-RT window for alert delta (default: 15)")
    parser.add_argument("--project",     default=GEE_PROJECT)
    args = parser.parse_args()

    ee.Initialize(project=args.project)

    fc  = ee.FeatureCollection(GUNA_ASSET)

    log.info(f"\n{'='*65}")
    log.info(f"  eNetra Range Raster Export")
    log.info(f"  AOI:           {GUNA_ASSET}")
    log.info(f"  Drive folder:  {DRIVE_FOLDER}/{{range}}/")
    log.info(f"  GT window:     Dec {args.year_a} → Dec {args.year_b}")
    log.info(f"  Alert window:  latest {args.window_days}-day pair per range")
    log.info(f"{'='*65}\n")

    # ── Sub-range shortcut: bypass full range extraction ───────────────────────
    sub_range_arg = getattr(args, "sub_range", None)   # argparse replaces - with _
    if sub_range_arg:
        sr = sub_range_arg.strip()
        log.info(f"SUB_RANGE mode: filtering asset to SUB_RANGE = '{sr}' ...")
        sub_fc   = fc.filter(ee.Filter.eq("SUB_RANGE", sr))
        count    = sub_fc.size().getInfo()
        if count == 0:
            # Try case-insensitive by listing all values
            all_sr = fc.aggregate_array("SUB_RANGE").getInfo()
            unique = sorted(set(str(x) for x in all_sr if x))
            log.error(f"No features found for SUB_RANGE='{sr}'. "
                      f"Available: {unique}")
            return
        log.info(f"  Found {count} beats in SUB_RANGE='{sr}'")
        aoi      = sub_fc.geometry().dissolve(maxError=10)
        safe_name = sr.replace(" ", "_")
        ranges   = {safe_name: aoi}
    else:
        log.info("Extracting Forest Ranges from beat asset ...")
        ranges = get_ranges(fc)
        log.info(f"  Found {len(ranges)} ranges: {sorted(ranges.keys())}\n")
        # Filter to single range if requested
        if args.range:
            key = args.range.strip().replace(" ", "_")
            if key not in ranges:
                log.error(f"Range '{key}' not found. Available: {sorted(ranges)}")
                return
            ranges = {key: ranges[key]}

    n_gt     = 0
    n_alert  = 0

    for rng_name in sorted(ranges.keys()):
        rng_geom = ranges[rng_name]
        log.info(f"  ── Range: {rng_name} {'─'*(40-len(rng_name))}")

        if not args.alert_only:
            export_gt_delta(rng_name, rng_geom, args.year_a, args.year_b)
            n_gt += 1

        if not args.gt_only:
            export_alert_delta(rng_name, rng_geom, args.window_days)
            n_alert += 1

        log.info("")

    total = n_gt + n_alert
    log.info(f"{'='*65}")
    log.info(f"  {total} tasks submitted")
    log.info(f"    GT delta tasks:    {n_gt}")
    log.info(f"    Alert delta tasks: {n_alert}")
    log.info(f"")
    log.info(f"  Monitor: https://code.earthengine.google.com/tasks")
    log.info(f"")
    log.info(f"  After download, organise as:")
    log.info(f"    data/ground_truth/{{range}}/gt_delta_{args.year_a}_{args.year_b}_{{range}}.tif")
    log.info(f"    data/ground_truth/{{range}}/alert_delta_latest_{{range}}.tif")
    log.info(f"{'='*65}")


if __name__ == "__main__":
    main()
