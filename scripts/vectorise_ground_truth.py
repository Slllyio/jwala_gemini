"""
Dynamic World Ground Truth Vectoriser
======================================
Converts DW change rasters (from export_dw_annual_ground_truth.py) into
clean GeoJSON polygons locally. Much faster and cheaper than GEE vectorisation.

Input:  data/ground_truth/dw_gt_{YEAR}_change.tif  (3-band GeoTIFF)
Output: data/ground_truth/dw_gt_{YEAR}_deforestation.geojson
        data/ground_truth/dw_gt_{YEAR}_encroachment.geojson

Usage:
    python scripts/vectorise_ground_truth.py
    python scripts/vectorise_ground_truth.py --year 2025
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
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, mapping

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GT_DIR    = Path("data/ground_truth")
MIN_HA    = 0.20   # drop polygons under 0.20 ha (sub-pixel noise)

CLASS_BANDS = {
    "deforestation": 1,   # band index 1 in the GeoTIFF
    "encroachment":  2,   # band index 2
}


def vectorise_raster(tif_path: Path, band_idx: int, class_name: str,
                     year: int) -> gpd.GeoDataFrame:
    """
    Convert a binary band from the change raster into a GeoDataFrame of polygons.
    Filters out sub-pixel noise (<0.20 ha) and adds metadata attributes.
    """
    with rasterio.open(tif_path) as src:
        data      = src.read(band_idx).astype(np.uint8)
        transform = src.transform
        crs       = src.crs

    polygons = []
    for geom_dict, val in shapes(data, mask=(data == 1), transform=transform):
        if val == 1:
            geom = shape(geom_dict)
            # Area in sq metres (approximate — using degree² → m² at 24°N)
            area_ha = geom.area * (111_320 ** 2) * np.cos(np.radians(24)) / 10_000
            if area_ha >= MIN_HA:
                polygons.append({
                    "geometry":   geom,
                    "class":      class_name,
                    "change_year": year,
                    "area_ha":    round(area_ha, 4),
                    "source":     "DW_dec_dec",
                })

    if not polygons:
        log.warning(f"  {class_name} {year}: 0 polygons ≥{MIN_HA} ha")
        return gpd.GeoDataFrame(columns=["geometry","class","change_year","area_ha","source"],
                                crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(polygons, crs=crs)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    log.info(f"  {class_name} {year}: {len(gdf)} polygons "
             f"(total {gdf['area_ha'].sum():.1f} ha)")
    return gdf


def process_year(year: int):
    tif_path = GT_DIR / f"dw_gt_{year}_change.tif"
    if not tif_path.exists():
        log.warning(f"  {tif_path} not found — download from Drive first")
        return

    log.info(f"\nVectorising year {year} ...")
    for class_name, band_idx in CLASS_BANDS.items():
        gdf = vectorise_raster(tif_path, band_idx, class_name, year)
        out_path = GT_DIR / f"dw_gt_{year}_{class_name}.geojson"
        gdf.to_file(out_path, driver="GeoJSON")
        log.info(f"  → Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",  type=int, help="Single year")
    parser.add_argument("--years", type=int, nargs="+", help="Multiple years")
    args = parser.parse_args()

    years = ([args.year] if args.year
             else args.years if args.years
             else sorted([int(p.stem.split("_")[2])
                          for p in GT_DIR.glob("dw_gt_*_change.tif")]))

    if not years:
        log.error("No TIF files found in data/ground_truth/ and no years specified.")
        log.error("Run scripts/export_dw_annual_ground_truth.py first, "
                  "then download from Drive.")
        return

    for year in years:
        process_year(year)

    log.info("\nDone. Run validate_alerts.py next.")


if __name__ == "__main__":
    main()
