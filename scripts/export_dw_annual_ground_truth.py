"""
Dynamic World Annual Ground Truth Exporter
==========================================
Exports Dec→Dec Dynamic World V1 change rasters for Guna Division.

These rasters are the annual "ground truth" that will be intersected with
daily alert GeoJSONs to evaluate:
  1. Alert accuracy  (did the alert overlap a real change?)
  2. Alert class     (deforestation vs encroachment)
  3. Area difference (alert area vs final confirmed change area)
  4. Time of change  (approximate — bounded by alert date)
  5. Missed changes  (real changes with no alert = false negatives)

Two-Class Logic:
  Class 1 — DEFORESTATION:
    trees_prob(Dec Y-1) > 0.25  AND  bare_prob(Dec Y) > 0.50
    OR
    shrub_and_scrub_prob(Dec Y-1) > 0.35  AND  bare_prob(Dec Y) > 0.50
    [captures Guna's open dry forest which DW often labels "scrub"]

  Class 2 — ENCROACHMENT (forest-to-agriculture):
    (trees_prob(Dec Y-1) > 0.25 OR shrub_and_scrub_prob(Dec Y-1) > 0.35)
    AND  crops_prob(Dec Y) > 0.50

Strategy: Export RASTERS from GEE (cheap), vectorise locally with rasterio+shapely.
GEE vectorisation at 10m for 2144 sq km would time out; local is ~30s per year.

Usage:
    python scripts/export_dw_annual_ground_truth.py
    python scripts/export_dw_annual_ground_truth.py --year 2025
    python scripts/export_dw_annual_ground_truth.py --years 2022 2023 2024 2025

Output (Google Drive → data/ground_truth/ after download):
    dw_gt_{YEAR}_deforestation.tif   -- binary raster, 1=deforestation
    dw_gt_{YEAR}_encroachment.tif    -- binary raster, 1=encroachment
"""

import ee
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

GEE_PROJECT    = "van-suraksha-alert"
GUNA_ASSET     = "projects/van-suraksha-alert/assets/gunafinal"
DRIVE_FOLDER   = "prithvi_guna_ground_truth"

# December window: post-monsoon, trees fully leafed, clear skies in MP
DEC_START      = "12-01"
DEC_END        = "12-31"

# Probability thresholds (using soft bands, not hard label)
# Lower than standard (0.5) because Guna's dry forest is sparse
THR_TREES      = 0.25   # trees_prob in DEC Y-1 to count as "was forest"
THR_SHRUB      = 0.35   # shrub_and_scrub_prob (open dry forest in DW often = scrub)
THR_BARE       = 0.50   # bare_prob in DEC Y to confirm physical removal
THR_CROPS      = 0.50   # crops_prob in DEC Y to confirm encroachment
MIN_PIXELS     = 20     # 20 × 10m² = 0.20 ha minimum polygon -- same as Hansen filter

# Years to process (DW data starts 2016; we go 2019 to cover full training window)
DEFAULT_YEARS  = [2019, 2020, 2021, 2022, 2023, 2024, 2025]


# ── GEE helpers ───────────────────────────────────────────────────────────────

def get_dec_composite(year: int, aoi: ee.Geometry) -> ee.Image:
    """
    December median composite of Dynamic World probability bands.
    Uses full December to maximise scene count (MP Dec has <5% cloud cover).
    """
    start = f"{year}-{DEC_START}"
    end   = f"{year}-{DEC_END}"
    return (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(start, end)
        .filterBounds(aoi)
        .select(["trees", "shrub_and_scrub", "crops", "bare", "built", "label"])
        .median()
        .clip(aoi)
    )


def make_change_masks(dec_prev: ee.Image, dec_curr: ee.Image) -> ee.Image:
    """
    Compute 2-band change image (deforestation, encroachment).

    "Was vegetation" = trees_prob > THR_TREES  OR  shrub_prob > THR_SHRUB
    This dual-band approach captures Guna's open dry-deciduous forest which
    Dynamic World's classifier frequently labels as shrub_and_scrub instead
    of trees due to 40-60% canopy cover (below DW's implicit "dense forest" 
    decision boundary).
    """
    # Was this pixel forest/dense-shrub in the PREVIOUS December?
    was_trees = dec_prev.select("trees").gt(THR_TREES)
    was_shrub = dec_prev.select("shrub_and_scrub").gt(THR_SHRUB)
    was_vegetation = was_trees.Or(was_shrub)

    # What does this pixel look like NOW (current December)?
    now_bare  = dec_curr.select("bare").gt(THR_BARE)
    now_crops = dec_curr.select("crops").gt(THR_CROPS)

    # Class 1: DEFORESTATION -- vegetation → bare/cleared
    # Physically: trees cut, land exposed, no regrowth yet
    deforestation = was_vegetation.And(now_bare).rename("deforestation").toByte()

    # Class 2: ENCROACHMENT -- vegetation → agriculture
    # Physically: forest cleared and ploughed for farming (kharif/rabi cycle)
    encroachment = was_vegetation.And(now_crops).rename("encroachment").toByte()

    # Minimum area filter (applied as a sieve in pixel space):
    # Morphologically erode then dilate -- removes <MIN_PIXELS isolated pixels
    # before export so vectorisation doesn't produce stamp-sized polygons.
    kernel = ee.Kernel.square(radius=2, units="pixels")  # 5×5 → 50m radius
    deforestation_clean = (deforestation.focal_min(kernel=kernel)
                                        .focal_max(kernel=kernel))
    encroachment_clean  = (encroachment.focal_min(kernel=kernel)
                                       .focal_max(kernel=kernel))

    return ee.Image.cat([deforestation_clean, encroachment_clean])


def export_year(year: int, aoi: ee.Geometry, aoi_fc: ee.FeatureCollection):
    """Export deforestation + encroachment rasters for one Dec(year-1)→Dec(year) pair."""
    prev_year = year - 1
    log.info(f"  Exporting Dec {prev_year} → Dec {year} ...")

    dec_prev = get_dec_composite(prev_year, aoi)
    dec_curr = get_dec_composite(year,      aoi)
    change   = make_change_masks(dec_prev, dec_curr)

    # Metadata band: stores the year of change for stack comparisons
    year_band = ee.Image.constant(year).rename("change_year").toByte()
    export_img = change.addBands(year_band)

    # Export full 2-band (+ metadata) raster at 10m
    desc = f"dw_gt_{year}_change"
    task = ee.batch.Export.image.toDrive(
        image=export_img,
        description=desc,
        folder=DRIVE_FOLDER,
        fileNamePrefix=desc,
        region=aoi,
        scale=10,
        crs="EPSG:4326",
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()
    log.info(f"    ✅ Task submitted: {desc}")
    log.info(f"       Bands: deforestation, encroachment, change_year")
    log.info(f"       Drive folder: {DRIVE_FOLDER}/")
    return task


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export Dynamic World annual ground truth")
    parser.add_argument("--year",  type=int, help="Single year to export")
    parser.add_argument("--years", type=int, nargs="+", help="Multiple years to export")
    parser.add_argument("--project", default=GEE_PROJECT, help="GEE project ID")
    args = parser.parse_args()

    log.info("Initialising GEE ...")
    ee.Initialize(project=args.project)
    log.info(f"GEE project: {args.project}")

    years = ([args.year] if args.year
             else args.years if args.years
             else DEFAULT_YEARS)

    log.info(f"\nDynamic World Dec→Dec Ground Truth Export")
    log.info(f"  AOI asset:    {GUNA_ASSET}")
    log.info(f"  Drive folder: {DRIVE_FOLDER}/")
    log.info(f"  Years:        {years}")
    log.info(f"  Thresholds:   trees>{THR_TREES}, shrub>{THR_SHRUB}, "
             f"bare>{THR_BARE}, crops>{THR_CROPS}")
    log.info(f"  Resolution:   10m (Sentinel-2 native)")
    log.info(f"  Classes:      1=Deforestation (→bare), 2=Encroachment (→crops)\n")

    aoi_fc  = ee.FeatureCollection(GUNA_ASSET)
    aoi     = aoi_fc.geometry()

    tasks = []
    for year in years:
        if year < 2017:
            log.warning(f"  Skipping {year}: Dynamic World starts 2016, "
                        "need Dec(year-1) = 2015 which is before DW coverage")
            continue
        tasks.append(export_year(year, aoi, aoi_fc))

    log.info(f"\n{'='*55}")
    log.info(f"  {len(tasks)} export tasks submitted to GEE.")
    log.info(f"  Monitor: https://code.earthengine.google.com/tasks")
    log.info(f"\nNext steps after download from Drive:")
    log.info(f"  1. Move TIFs → data/ground_truth/")
    log.info(f"  2. python scripts/vectorise_ground_truth.py")
    log.info(f"  3. python scripts/validate_alerts.py --year 2025")
    log.info(f"{'='*55}")


if __name__ == "__main__":
    main()
