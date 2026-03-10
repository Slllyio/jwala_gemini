"""
export_guna_aoi.py
==================
Reads the MP Beats shapefile and exports Guna Division boundaries as
WGS84 GeoJSONs for:
  - GEE AOI upload (guna_division.geojson — full dissolved boundary)
  - Per-range exports for range-level monitoring
  - Per-beat file for beat-level attribution in alerts

Usage:
    python scripts/export_guna_aoi.py
"""

# ── PROJ/GDAL env fix ──
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()
del _os, _sys
# ────────────────────────

import geopandas as gpd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

SHAPEFILE = r"C:\Users\S.C.C\Downloads\MP_Merge Divsion Beat Range\MP_Merge\Mp_Beat_Merge.shp"
OUT_DIR   = Path("data/aoi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load and filter ──────────────────────────────────────────────────────────
print("Loading MP beats shapefile...")
beat = gpd.read_file(SHAPEFILE)
guna = beat[beat["Division"].str.contains("Guna", case=False, na=False)].copy()
print(f"  Loaded {len(beat)} total MP beats -> {len(guna)} Guna beats")

# Reproject to WGS84 (required by GEE)
guna_wgs = guna.to_crs("EPSG:4326")

# ── 1. Full Guna Division dissolved boundary ─────────────────────────────────
guna_div = guna_wgs.dissolve()
guna_div.to_file(OUT_DIR / "guna_division.geojson", driver="GeoJSON")
print(f"  Saved: data/aoi/guna_division.geojson")

# ── 2. Per-Range dissolved polygons ─────────────────────────────────────────
ranges = sorted(guna_wgs["Range"].unique())
for rng in ranges:
    sub  = guna_wgs[guna_wgs["Range"] == rng].dissolve()
    slug = rng.lower().replace(" ", "_")
    sub.to_file(OUT_DIR / f"guna_range_{slug}.geojson", driver="GeoJSON")
print(f"  Saved {len(ranges)} range GeoJSONs to data/aoi/")

# ── 3. All beats with attributes (for alert attribution) ────────────────────
cols = [c for c in ["Range", "Beat", "Area", "geometry"] if c in guna_wgs.columns]
guna_wgs[cols].to_file(OUT_DIR / "guna_beats.geojson", driver="GeoJSON")
print(f"  Saved: data/aoi/guna_beats.geojson ({len(guna_wgs)} beats)")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=== Guna Division Summary ===")
print(f"  {'Range':<22} {'Beats':>5}  {'Area (km²)':>10}")
print("  " + "-" * 42)
for rng, grp in guna_wgs.groupby("Range"):
    area = grp["Area"].sum() if "Area" in grp.columns else grp.geometry.area.sum() / 1e6
    print(f"  {rng:<22} {len(grp):>5}  {area:>10.1f}")

total_area = guna_wgs["Area"].sum() if "Area" in guna_wgs.columns else guna_wgs.geometry.area.sum() / 1e6
print("  " + "-" * 42)
print(f"  {'TOTAL':<22} {len(guna_wgs):>5}  {total_area:>10.1f}")

bounds = guna_wgs.total_bounds
print()
print(f"  Bounding box (WGS84):")
print(f"    West:  {bounds[0]:.4f}°")
print(f"    South: {bounds[1]:.4f}°")
print(f"    East:  {bounds[2]:.4f}°")
print(f"    North: {bounds[3]:.4f}°")
print()
print("Next step: upload data/aoi/guna_division.geojson to GEE as an asset")
print("  earthengine upload table --asset_id projects/ee-akshayr1/assets/guna_division \\")
print("    data/aoi/guna_division.geojson")
