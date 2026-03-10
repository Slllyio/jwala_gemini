"""
Forest Boundary & Land Cover for Guna Division
===============================================
Downloads and processes two open-access datasets:

  1. GADM v4.1 India District Boundaries (Level-2)
     -> Extracts Guna district polygon -> GeoJSON + shapefile

  2. ESA WorldCover 2021 (10 m resolution, Sentinel-1/2 based)
     -> Downloads tiles covering Guna bbox
     -> Clips to district boundary
     -> Extracts forest / vegetation mask

ESA WorldCover classes used:
  10  Tree cover          -> FOREST (primary)
  20  Shrubland           -> SCRUB_FOREST (secondary, labeled separately)
  30  Grassland           -> GRASSLAND
  40  Cropland            -> AGRICULTURE (to EXCLUDE from forest fire alerts)
  50  Built-up            -> BUILT
  60  Bare/sparse veg     -> BARE
  80  Permanent water     -> WATER
  90  Herbaceous wetland  -> WETLAND

Output:
  data_lake/boundaries/guna_district.geojson         <- district polygon
  data_lake/boundaries/guna_district.gpkg            <- GeoPackage (QGIS)
  data_lake/land_cover/worldcover_guna_raw.tif       <- raw ESA WorldCover clip
  data_lake/land_cover/forest_mask_guna_10m.tif      <- binary forest mask (10m)
  data_lake/land_cover/forest_mask_guna_30m.tif      <- resampled to HLS 30m grid
  data_lake/land_cover/land_cover_stats.json         <- area breakdown by class

Usage:
  python scripts/fetch_forest_boundary.py
  python scripts/fetch_forest_boundary.py --skip-gadm    # use cached boundary
  python scripts/fetch_forest_boundary.py --skip-worldcover
  python scripts/fetch_forest_boundary.py --stats-only  # print stats and exit
"""

# ── PROJ/GDAL env fix ──
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()
del _os, _sys
# ────────────────────────

import argparse
import json
import shutil
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import requests

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
BOUND_DIR = ROOT / "data_lake" / "boundaries"
LC_DIR    = ROOT / "data_lake" / "land_cover"
CACHE_DIR = ROOT / "data_lake" / ".cache"

for d in (BOUND_DIR, LC_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Guna Division extent ──────────────────────────────────────────────────────
GUNA_BBOX   = (76.45, 23.80, 77.85, 24.95)   # (W, S, E, N)
GUNA_CENTROID = (77.15, 24.38)

# GADM state/district names for MP / Guna
GADM_STATE_VARIANTS  = ["Madhya Pradesh", "MADHYA PRADESH"]
GADM_DIST_VARIANTS   = ["Guna", "GUNA"]

# ── ESA WorldCover 2021 ───────────────────────────────────────────────────────
# Tiles are 3° × 3°, named N{lat}E{lon} (SW corner, lat/lon multiples of 3)
WORLDCOVER_BASE = (
    "https://esa-worldcover.s3.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)

# Land cover class definitions
LC_CLASSES = {
    10:  ("Tree cover",          "FOREST"),
    20:  ("Shrubland",           "SCRUB_FOREST"),
    30:  ("Grassland",           "GRASSLAND"),
    40:  ("Cropland",            "AGRICULTURE"),
    50:  ("Built-up",            "BUILT"),
    60:  ("Bare/sparse veg",     "BARE"),
    70:  ("Snow and ice",        "SNOW"),
    80:  ("Permanent water",     "WATER"),
    90:  ("Herbaceous wetland",  "WETLAND"),
    95:  ("Mangroves",           "MANGROVE"),
    100: ("Moss and lichen",     "MOSS"),
}

FOREST_CLASSES    = {10}          # strict forest
VEG_CLASSES       = {10, 20, 30}  # all vegetation
NON_FOREST_ALERT  = {40}          # agricultural — suppress forest alert


# =============================================================================
# SECTION 1: Helpers
# =============================================================================

def _download(url: str, dest: Path, desc: str = "") -> Path:
    """Download url to dest with a simple progress indicator."""
    if dest.exists():
        print(f"  [cached] {dest.name}")
        return dest
    print(f"  Downloading {desc or url} ...", end="", flush=True)
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  Downloading {desc} ... "
                      f"{downloaded/1e6:.1f}/{total/1e6:.1f} MB  ", end="")
    print(f"\r  Downloaded  {desc}: {downloaded/1e6:.1f} MB        ")
    return dest


def _worldcover_tile(lon: float, lat: float) -> str:
    """
    ESA WorldCover tile name for a given lon/lat.
    Tiles are 3°x3°, SW corner rounded down to nearest multiple of 3.
    """
    lat0 = int(lat // 3) * 3
    lon0 = int(lon // 3) * 3
    lat_dir = "N" if lat0 >= 0 else "S"
    lon_dir = "E" if lon0 >= 0 else "W"
    return f"{lat_dir}{abs(lat0):02d}{lon_dir}{abs(lon0):03d}"


def _tiles_for_bbox(bbox: tuple) -> list:
    """Return all WorldCover tile names needed to cover bbox."""
    w, s, e, n = bbox
    tiles = set()
    lat = int(s // 3) * 3
    while lat <= n:
        lon = int(w // 3) * 3
        while lon <= e:
            tiles.add(_worldcover_tile(lon + 1.5, lat + 1.5))
            lon += 3
        lat += 3
    return sorted(tiles)


# =============================================================================
# SECTION 2: GADM District Boundary
# =============================================================================

def fetch_guna_boundary() -> Path:
    """
    Download GADM v4.1 India Level-2 (districts) shapefile and extract
    the Guna district polygon.

    Returns path to guna_district.geojson
    """
    out_geojson = BOUND_DIR / "guna_district.geojson"
    if out_geojson.exists():
        print(f"  [cached] guna_district.geojson")
        return out_geojson

    print("\n[1] Fetching Guna district boundary (GADM v4.1)...")

    # GADM Level-2 India GeoJSON (the old ZIP/SHP URL is defunct; JSON is still live)
    gadm_url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_2.json"
    gadm_json = CACHE_DIR / "gadm41_IND_2.json"
    _download(gadm_url, gadm_json, "GADM India Level-2 GeoJSON (~5 MB)")

    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas not installed. Run: pip install geopandas")

    print(f"  Loading gadm41_IND_2.json...", end="", flush=True)
    gdf = gpd.read_file(gadm_json)
    print(f" {len(gdf)} districts loaded.")

    # Find Guna
    name_cols = [c for c in gdf.columns if "name" in c.lower() or "nam" in c.lower()]
    guna = None
    for col in name_cols:
        mask = gdf[col].str.upper().str.contains("GUNA", na=False)
        if mask.any():
            guna = gdf[mask]
            print(f"  Found Guna district in column '{col}': {len(guna)} feature(s)")
            break

    if guna is None or guna.empty:
        # Fallback: filter by bbox
        print("  WARNING: Guna not found by name — using bbox fallback")
        w, s, e, n = GUNA_BBOX
        guna = gdf.cx[w:e, s:n]
        print(f"  Bbox filter: {len(guna)} features in Guna area")

    if guna.empty:
        raise ValueError("Could not locate Guna district in GADM data")

    # Reproject to WGS84 if needed
    if guna.crs and guna.crs.to_epsg() != 4326:
        guna = guna.to_crs(epsg=4326)

    # Save GeoJSON
    guna.to_file(out_geojson, driver="GeoJSON")

    # Also save GeoPackage
    gpkg_path = BOUND_DIR / "guna_district.gpkg"
    guna.to_file(gpkg_path, driver="GPKG", layer="guna_district")

    print(f"  Saved: {out_geojson.name}  ({out_geojson.stat().st_size//1024} KB)")
    print(f"  Saved: {gpkg_path.name}")

    # Print basic info
    area_km2 = guna.to_crs(epsg=32643).geometry.area.sum() / 1e6
    print(f"  District area: {area_km2:,.0f} km2")

    return out_geojson


# =============================================================================
# SECTION 3: ESA WorldCover download & processing
# =============================================================================

def fetch_worldcover(bbox: tuple = GUNA_BBOX) -> list:
    """
    Download ESA WorldCover 2021 tiles for the given bounding box.
    Returns list of downloaded GeoTIFF paths.
    """
    tiles = _tiles_for_bbox(bbox)
    print(f"\n[2] Fetching ESA WorldCover 2021 — tiles: {tiles}")

    paths = []
    for tile in tiles:
        url  = WORLDCOVER_BASE.format(tile=tile)
        dest = CACHE_DIR / f"ESA_WorldCover_{tile}.tif"
        try:
            _download(url, dest, f"WorldCover tile {tile}")
            paths.append(dest)
        except Exception as e:
            print(f"  WARNING: tile {tile} failed: {e}")

    return paths


def clip_and_mask_worldcover(tile_paths: list,
                              boundary_geojson: Path,
                              bbox: tuple = GUNA_BBOX) -> dict:
    """
    Merge WorldCover tiles, clip to Guna boundary, create:
      - Raw land cover clip (worldcover_guna_raw.tif)
      - Binary forest mask at 10m (forest_mask_guna_10m.tif)
      - Binary forest mask at 30m resampled to HLS grid (forest_mask_guna_30m.tif)
      - Land cover stats JSON

    Returns dict of output paths.
    """
    try:
        import rasterio
        from rasterio.merge import merge
        from rasterio.mask import mask as rasterio_mask
        from rasterio.enums import Resampling
        from rasterio.warp import reproject, calculate_default_transform
        import geopandas as gpd
        from shapely.geometry import mapping
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}. "
                          f"Run: pip install rasterio geopandas")

    if not tile_paths:
        print("  No WorldCover tiles available — skipping land cover processing.")
        return {}

    # ── Merge tiles ──────────────────────────────────────────────────────────
    raw_path = LC_DIR / "worldcover_guna_raw.tif"
    print("\n[3] Merging and clipping WorldCover tiles...")

    srcs = [rasterio.open(p) for p in tile_paths]

    # Clip bbox first before merging (faster)
    w, s, e, n = bbox
    merged, transform = merge(srcs, bounds=(w, s, e, n))
    meta = srcs[0].meta.copy()
    meta.update({
        "driver": "GTiff",
        "height": merged.shape[1],
        "width":  merged.shape[2],
        "transform": transform,
        "compress": "lzw",
    })

    with rasterio.open(raw_path, "w", **meta) as dst:
        dst.write(merged)

    for src in srcs:
        src.close()

    print(f"  Raw WorldCover clip: {raw_path.name}  "
          f"({raw_path.stat().st_size//1024} KB)")

    # ── Mask to district boundary ────────────────────────────────────────────
    boundary_gdf = gpd.read_file(boundary_geojson)
    if boundary_gdf.crs and boundary_gdf.crs.to_epsg() != 4326:
        boundary_gdf = boundary_gdf.to_crs(epsg=4326)

    geoms = [mapping(geom) for geom in boundary_gdf.geometry]

    with rasterio.open(raw_path) as src:
        masked_data, masked_transform = rasterio_mask(src, geoms, crop=True, nodata=0)
        masked_meta = src.meta.copy()
        masked_meta.update({
            "height":    masked_data.shape[1],
            "width":     masked_data.shape[2],
            "transform": masked_transform,
            "nodata":    0,
            "compress":  "lzw",
        })

    # Overwrite raw clip with district-masked version
    with rasterio.open(raw_path, "w", **masked_meta) as dst:
        dst.write(masked_data)

    lc_arr = masked_data[0]  # 2D array of LC classes

    # ── Land cover statistics ─────────────────────────────────────────────────
    print("\n[4] Computing land cover statistics...")
    pixel_area_ha = (10 * 10) / 10000   # 10m pixel = 100m2 = 0.01 ha
    stats = {}
    total_valid = np.sum(lc_arr > 0)

    for code, (label, key) in LC_CLASSES.items():
        count = int(np.sum(lc_arr == code))
        area_ha = count * pixel_area_ha
        pct = 100.0 * count / max(total_valid, 1)
        stats[key] = {
            "class_code":  code,
            "label":       label,
            "pixels":      count,
            "area_ha":     round(area_ha, 1),
            "area_km2":    round(area_ha / 100, 2),
            "pct_of_area": round(pct, 2),
        }
        if count > 0:
            print(f"  {code:3d} {label:<25}: "
                  f"{area_ha:8,.0f} ha  ({pct:5.1f}%)")

    forest_ha  = stats.get("FOREST", {}).get("area_ha", 0)
    scrub_ha   = stats.get("SCRUB_FOREST", {}).get("area_ha", 0)
    crop_ha    = stats.get("AGRICULTURE", {}).get("area_ha", 0)
    total_ha   = total_valid * pixel_area_ha

    print(f"\n  Forest (class 10) : {forest_ha:,.0f} ha")
    print(f"  Scrub/shrub (20)  : {scrub_ha:,.0f} ha")
    print(f"  Agriculture (40)  : {crop_ha:,.0f} ha")
    print(f"  Forest cover      : {100*forest_ha/max(total_ha,1):.1f}% of district")

    stats["_summary"] = {
        "total_area_ha":   round(total_ha, 0),
        "forest_ha":       round(forest_ha, 0),
        "scrub_ha":        round(scrub_ha, 0),
        "agriculture_ha":  round(crop_ha, 0),
        "forest_pct":      round(100*forest_ha/max(total_ha,1), 1),
    }

    stats_path = LC_DIR / "land_cover_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path.name}")

    # ── Binary forest mask at 10m ────────────────────────────────────────────
    print("\n[5] Creating forest masks...")
    forest_10m_path = LC_DIR / "forest_mask_guna_10m.tif"
    forest_arr = np.zeros_like(lc_arr, dtype=np.uint8)
    for code in FOREST_CLASSES:
        forest_arr[lc_arr == code] = 1

    forest_meta = masked_meta.copy()
    forest_meta.update({"dtype": "uint8", "nodata": 255, "count": 1})

    with rasterio.open(forest_10m_path, "w", **forest_meta) as dst:
        dst.write(forest_arr[np.newaxis, :, :])

    print(f"  Saved: {forest_10m_path.name}  "
          f"({forest_10m_path.stat().st_size//1024} KB)")

    # ── Resample to 30m for HLS alignment ────────────────────────────────────
    forest_30m_path = LC_DIR / "forest_mask_guna_30m.tif"
    scale = 3    # 10m -> 30m

    with rasterio.open(forest_10m_path) as src:
        new_h = max(1, src.height // scale)
        new_w = max(1, src.width  // scale)
        data_30m = src.read(
            out_shape=(1, new_h, new_w),
            resampling=Resampling.mode   # majority vote for categorical
        )
        new_transform = src.transform * src.transform.scale(
            src.width / new_w,
            src.height / new_h
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "height":    new_h,
            "width":     new_w,
            "transform": new_transform,
            "compress":  "lzw",
        })

    with rasterio.open(forest_30m_path, "w", **out_meta) as dst:
        dst.write(data_30m)

    print(f"  Saved: {forest_30m_path.name}  "
          f"({forest_30m_path.stat().st_size//1024} KB)  "
          f"[{new_h}x{new_w} @ 30m]")

    return {
        "raw_lc":       raw_path,
        "forest_10m":   forest_10m_path,
        "forest_30m":   forest_30m_path,
        "stats":        stats_path,
        "stats_dict":   stats,
    }


# =============================================================================
# SECTION 4: Point-in-forest check (used by NRT alert pipeline)
# =============================================================================

def is_fire_in_forest(lat: float, lon: float,
                       forest_mask_path: Path = None,
                       buffer_px: int = 1) -> dict:
    """
    Check if a FIRMS fire detection point falls on or near a forest pixel.

    Parameters
    ----------
    lat, lon          : WGS84 coordinates of fire detection
    forest_mask_path  : Path to forest_mask_guna_30m.tif (auto-detected if None)
    buffer_px         : Pixel buffer radius for near-forest check (default=1 -> 30m)

    Returns
    -------
    dict with keys:
      in_forest     : bool   - fire pixel is forested
      near_forest   : bool   - within buffer_px of forest
      land_cover    : str    - land cover class at fire location
      dist_to_forest_m : float
    """
    try:
        import rasterio
        from rasterio.transform import rowcol
    except ImportError:
        return {"in_forest": None, "near_forest": None,
                "land_cover": "unknown", "dist_to_forest_m": None}

    if forest_mask_path is None:
        forest_mask_path = LC_DIR / "forest_mask_guna_30m.tif"
    raw_lc_path = LC_DIR / "worldcover_guna_raw.tif"

    if not forest_mask_path.exists():
        return {"in_forest": None, "near_forest": None,
                "land_cover": "unknown", "dist_to_forest_m": None}

    # Read forest mask at fire location
    with rasterio.open(forest_mask_path) as src:
        row, col = rowcol(src.transform, lon, lat)
        H, W = src.shape
        in_bounds = (0 <= row < H) and (0 <= col < W)
        if not in_bounds:
            return {"in_forest": False, "near_forest": False,
                    "land_cover": "out_of_bounds", "dist_to_forest_m": None}

        pixel_val = int(src.read(1)[row, col])
        in_forest = (pixel_val == 1)

        # Buffer check
        r0 = max(0, row - buffer_px)
        r1 = min(H, row + buffer_px + 1)
        c0 = max(0, col - buffer_px)
        c1 = min(W, col + buffer_px + 1)
        patch = src.read(1)[r0:r1, c0:c1]
        near_forest = bool(np.any(patch == 1))

        pixel_size = abs(src.transform.a)

    # Get land cover class label
    lc_label = "unknown"
    if raw_lc_path.exists():
        with rasterio.open(raw_lc_path) as lc_src:
            lc_row, lc_col = rowcol(lc_src.transform, lon, lat)
            lH, lW = lc_src.shape
            if (0 <= lc_row < lH) and (0 <= lc_col < lW):
                lc_code = int(lc_src.read(1)[lc_row, lc_col])
                lc_label = LC_CLASSES.get(lc_code, (str(lc_code),))[0]

    return {
        "in_forest":       in_forest,
        "near_forest":     near_forest,
        "land_cover":      lc_label,
        "dist_to_forest_m": 0.0 if in_forest else (
            pixel_size * buffer_px if near_forest else None),
    }


# =============================================================================
# SECTION 5: Main
# =============================================================================

def print_stats():
    """Print land cover statistics from cached JSON."""
    stats_path = LC_DIR / "land_cover_stats.json"
    if not stats_path.exists():
        print("No stats file found. Run fetch_forest_boundary.py first.")
        return
    with open(stats_path) as f:
        stats = json.load(f)
    s = stats.get("_summary", {})
    print("\n" + "=" * 60)
    print("LAND COVER STATS — Guna Division, MP (ESA WorldCover 2021)")
    print("=" * 60)
    print(f"  Total area  : {s.get('total_area_ha',0):>10,.0f} ha")
    print(f"  Forest      : {s.get('forest_ha',0):>10,.0f} ha  "
          f"({s.get('forest_pct',0):.1f}%)")
    print(f"  Scrubland   : {s.get('scrub_ha',0):>10,.0f} ha")
    print(f"  Agriculture : {s.get('agriculture_ha',0):>10,.0f} ha")
    print()
    for key, v in stats.items():
        if key == "_summary":
            continue
        if v.get("pixels", 0) > 0:
            print(f"  {v['label']:<25}: {v['area_ha']:>9,.0f} ha  "
                  f"({v['pct_of_area']:>5.1f}%)")


def run(skip_gadm: bool = False, skip_worldcover: bool = False):
    print("=" * 60)
    print("FOREST BOUNDARY & LAND COVER — Guna Division, MP")
    print("=" * 60)

    # Step 1: District boundary
    if skip_gadm and (BOUND_DIR / "guna_district.geojson").exists():
        boundary_path = BOUND_DIR / "guna_district.geojson"
        print("\n[1] Using cached district boundary.")
    else:
        boundary_path = fetch_guna_boundary()

    # Step 2 & 3: WorldCover
    if skip_worldcover and (LC_DIR / "worldcover_guna_raw.tif").exists():
        print("\n[2-5] Using cached WorldCover data.")
        outputs = {
            "raw_lc":     LC_DIR / "worldcover_guna_raw.tif",
            "forest_10m": LC_DIR / "forest_mask_guna_10m.tif",
            "forest_30m": LC_DIR / "forest_mask_guna_30m.tif",
        }
    else:
        tile_paths = fetch_worldcover(GUNA_BBOX)
        outputs    = clip_and_mask_worldcover(tile_paths, boundary_path, GUNA_BBOX)

    # Summary
    print_stats()

    print("\n  Output files:")
    print(f"    boundaries/ : guna_district.geojson, guna_district.gpkg")
    print(f"    land_cover/ : worldcover_guna_raw.tif")
    print(f"                  forest_mask_guna_10m.tif")
    print(f"                  forest_mask_guna_30m.tif")
    print(f"                  land_cover_stats.json")
    print(f"\n  Use is_fire_in_forest(lat, lon) in NRT alert pipeline")
    print(f"  to classify detections as FOREST vs AGRICULTURAL fires.\n")

    return outputs


def main():
    p = argparse.ArgumentParser(
        description="Fetch Guna district boundary + ESA WorldCover forest mask"
    )
    p.add_argument("--skip-gadm",       action="store_true",
                   help="Skip GADM download (use cached boundary)")
    p.add_argument("--skip-worldcover", action="store_true",
                   help="Skip WorldCover download (use cached tiles)")
    p.add_argument("--stats-only",      action="store_true",
                   help="Print land cover stats from cache and exit")
    p.add_argument("--check-point",     nargs=2, type=float, metavar=("LAT","LON"),
                   help="Check if a lat/lon point is in forest (e.g. 24.3 77.1)")
    args = p.parse_args()

    if args.stats_only:
        print_stats()
        return

    if args.check_point:
        lat, lon = args.check_point
        result = is_fire_in_forest(lat, lon)
        print(f"\nPoint ({lat}N, {lon}E):")
        for k, v in result.items():
            print(f"  {k}: {v}")
        return

    run(skip_gadm=args.skip_gadm,
        skip_worldcover=args.skip_worldcover)


if __name__ == "__main__":
    main()
