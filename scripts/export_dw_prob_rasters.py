"""
Dynamic World Probability Raster Exporter — Range-split version
================================================================
Exports Dec(year_a) → Dec(year_b) probability delta rasters, split
by Forest Range within Guna Division.

Each range gets 3 GeoTIFs:
    dw_prob_dec{year_a}_{range}.tif
    dw_prob_dec{year_b}_{range}.tif
    dw_prob_delta_{year_a}_{year_b}_{range}.tif   ← this is the viz base

Drive folder: prithvi_guna_prob_viz/

Band order in all TIFs (matches DW_BANDS constant everywhere):
    1=water  2=trees  3=grass  4=flooded_veg
    5=crops  6=shrub_and_scrub  7=built  8=bare

Why range-split?
    Full Guna = 2144 sq km → one TIF is huge + one fig is unreadable.
    Splitting by range keeps each AOI to ~200–300 sq km — fast, crisp.

Usage:
    # Export all ranges
    python scripts/export_dw_prob_rasters.py

    # Export one specific range only
    python scripts/export_dw_prob_rasters.py --range North_Guna

    # Custom year pair
    python scripts/export_dw_prob_rasters.py --year-a 2023 --year-b 2024

    # Also export full-division mosaic (large)
    python scripts/export_dw_prob_rasters.py --full-mosaic

AOI asset: projects/van-suraksha-alert/assets/gunafinal
  The asset is a FeatureCollection of beats. Each beat has a `Beat_Name`
  property like "North_Guna_Jamner". The Range is extracted as the
  portion before the last underscore-separated suffix — or from a `Range`
  property if it exists.
"""

import ee
import argparse
import logging
import re

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GEE_PROJECT  = "van-suraksha-alert"
GUNA_ASSET   = "projects/van-suraksha-alert/assets/gunafinal"
DRIVE_FOLDER = "prithvi_guna_prob_viz"

DW_BANDS = [
    "water", "trees", "grass", "flooded_vegetation",
    "crops", "shrub_and_scrub", "built", "bare",
]

# Final exported band order in all 10-band TIFs:
#   1-8  : DW probability deltas (or raw probs for the two Dec composites)
#   9    : cloud_fraction_before  (fraction of Dec year_a pixels that were cloud-masked)
#   10   : cloud_fraction_after   (fraction of Dec year_b pixels that were cloud-masked)
# cloud_fraction = 0.0 → every observation in December was cloud-free
# cloud_fraction = 1.0 → every observation was cloudy → delta is unreliable


# ── Range extraction ──────────────────────────────────────────────────────────

def extract_ranges(fc: ee.FeatureCollection) -> dict[str, ee.Geometry]:
    """
    Get the dissolved geometry for each forest range.
    Returns {range_name: ee.Geometry}

    Tries the `Range` property first; falls back to the prefix of `Beat_Name`
    (everything up to the second underscore or last word).
    Example beat names:
        "North_Guna_Jamner"  → Range "North_Guna"
        "Aron_Goumukh"       → Range "Aron"
        "Raghogarh_Narolai"  → Range "Raghogarh"
    """
    features = fc.getInfo()["features"]
    ranges: dict[str, list] = {}

    for feat in features:
        props    = feat["properties"]
        beat_raw = props.get("Beat_Name", props.get("BEAT_NAME", "Unknown"))

        # Try explicit Range property
        rng = props.get("Range", props.get("RANGE", None))
        if not rng:
            # Derive from Beat_Name: treat multi-word prefixes like North_Guna
            parts = beat_raw.split("_")
            if len(parts) >= 3 and parts[0].lower() in ("north", "south", "east", "west"):
                rng = f"{parts[0]}_{parts[1]}"
            else:
                rng = parts[0]
        rng = rng.strip().replace(" ", "_")
        ranges.setdefault(rng, []).append(
            ee.Feature(feat).geometry()
        )

    # Dissolve beats into range geometry
    dissolved = {}
    for rng, geoms in ranges.items():
        dissolved[rng] = ee.Geometry.MultiPolygon(
            [g.geometries().getInfo() if hasattr(g, 'geometries') else g.getInfo()["coordinates"]
             for g in geoms]
        ).dissolve(maxError=10)

    return dissolved


def extract_ranges_server_side(fc: ee.FeatureCollection) -> dict[str, ee.Geometry]:
    """
    Server-side range dissolve. Faster for large FCs.
    Groups by the `Range` property if present, else falls back to client side.
    """
    # Check if Range property exists
    first = fc.first().getInfo()["properties"]
    if "Range" in first or "RANGE" in first:
        prop = "Range" if "Range" in first else "RANGE"
        range_names = fc.aggregate_array(prop).distinct().getInfo()
        dissolved = {}
        for rng in range_names:
            rng_clean = str(rng).strip().replace(" ", "_")
            geom = fc.filter(ee.Filter.eq(prop, rng)).geometry().dissolve(maxError=10)
            dissolved[rng_clean] = geom
        return dissolved

    # Fall back to client-side prefix extraction
    return extract_ranges(fc)


# ── DW composites ─────────────────────────────────────────────────────────────

def cloud_fraction(year: int, aoi: ee.Geometry) -> ee.Image:
    """
    Per-pixel cloud fraction for December of `year`, range [0, 1].

    Method: count valid (non-null) DW observations in December versus
    total size of the collection.  A pixel with no valid observation
    (always cloudy/missing) gets fraction = 1.0.

    Dynamic World only contains cloud-free observations — any gap in the
    per-pixel count means a cloudy or no-data S2 acquisition at that time.
    """
    col = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(f"{year}-12-01", f"{year}-12-31")
        .filterBounds(aoi)
    )
    total_scenes = col.size()                     # integer (same for all pixels)
    valid_count  = col.select("trees").count()    # per-pixel count of non-null obs
    frac = (
        ee.Image.constant(1)
        .subtract(valid_count.divide(ee.Image.constant(total_scenes)))
        .rename("cloud_fraction")
        .clip(aoi)
    )
    return frac


def dec_composite(year: int, aoi: ee.Geometry) -> ee.Image:
    """Median of December imagery for the given year, clipped to aoi."""
    return (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(f"{year}-12-01", f"{year}-12-31")
        .filterBounds(aoi)
        .select(DW_BANDS)
        .median()
        .clip(aoi)
    )


# ── Export ────────────────────────────────────────────────────────────────────

def submit(img: ee.Image, name: str, aoi: ee.Geometry) -> None:
    task = ee.batch.Export.image.toDrive(
        image=img.toFloat(),
        description=name,
        folder=DRIVE_FOLDER,
        fileNamePrefix=name,
        region=aoi,
        scale=10,
        crs="EPSG:4326",
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()
    log.info(f"    ✅ Submitted: {name}")


def export_range(range_name: str, aoi: ee.Geometry,
                 year_a: int, year_b: int) -> None:
    safe = range_name.replace(" ", "_")
    log.info(f"  Range: {safe}")

    img_a   = dec_composite(year_a, aoi)       # 8-band prob
    img_b   = dec_composite(year_b, aoi)       # 8-band prob
    delta   = img_b.subtract(img_a)            # 8-band delta, range -1..+1

    cld_a   = cloud_fraction(year_a, aoi)     # 1-band, 0=clear 1=cloudy
    cld_b   = cloud_fraction(year_b, aoi)     # 1-band, 0=clear 1=cloudy

    # ── 10-band composite for delta TIF ──────────────────────────────────────
    # Bands 1-8: probability deltas  |  Band 9: cloud_before  |  Band 10: cloud_after
    delta_with_cloud = (
        delta
        .addBands(cld_a.rename("cloud_before"))
        .addBands(cld_b.rename("cloud_after"))
    )

    # Raw probability TIFs also get cloud bands appended for symmetry
    img_a_out = img_a.addBands(cld_a.rename("cloud"))
    img_b_out = img_b.addBands(cld_b.rename("cloud"))

    submit(img_a_out,        f"dw_prob_dec{year_a}_{safe}",            aoi)
    submit(img_b_out,        f"dw_prob_dec{year_b}_{safe}",            aoi)
    submit(delta_with_cloud, f"dw_prob_delta_{year_a}_{year_b}_{safe}", aoi)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-a",      type=int, default=2024)
    parser.add_argument("--year-b",      type=int, default=2025)
    parser.add_argument("--range",       type=str, default=None,
                        help="Export a single named range only")
    parser.add_argument("--full-mosaic", action="store_true",
                        help="Also export full-division mosaic (large file)")
    parser.add_argument("--project",     default=GEE_PROJECT)
    args = parser.parse_args()

    ee.Initialize(project=args.project)

    fc  = ee.FeatureCollection(GUNA_ASSET)
    aoi = fc.geometry()

    log.info(f"\nDynamic World Probability Raster Export — Range Split")
    log.info(f"  Dec {args.year_a}  →  Dec {args.year_b}")
    log.info(f"  Drive folder: {DRIVE_FOLDER}/")
    log.info(f"  AOI asset:    {GUNA_ASSET}\n")

    log.info("  Extracting forest ranges ...")
    ranges = extract_ranges_server_side(fc)

    if args.range:
        key = args.range.strip().replace(" ", "_")
        if key not in ranges:
            log.error(f"  Range '{key}' not found. Available: {sorted(ranges)}")
            return
        export_range(key, ranges[key], args.year_a, args.year_b)
    else:
        log.info(f"  Found {len(ranges)} ranges: {sorted(ranges)}\n")
        for rng_name, rng_geom in sorted(ranges.items()):
            export_range(rng_name, rng_geom, args.year_a, args.year_b)

    if args.full_mosaic:
        log.info(f"\n  Also exporting full-division mosaic ...")
        export_range("FullGuna", aoi, args.year_a, args.year_b)

    n_tasks = (len(ranges) if not args.range else 1) * 3
    if args.full_mosaic:
        n_tasks += 3
    log.info(f"\n  {n_tasks} tasks submitted.")
    log.info(f"  Monitor: https://code.earthengine.google.com/tasks")
    log.info(f"\nAfter downloading from Drive → data/ground_truth/{{range}}/")
    log.info(f"  python scripts/plot_alert_dynamics.py --year {args.year_b} --range North_Guna")
