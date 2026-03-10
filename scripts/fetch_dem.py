"""
Terrain Data — Copernicus DEM 30m Download & Analysis
======================================================
Downloads Copernicus DEM 30m tiles from the AWS Open Data Programme
(no authentication required), clips to the Guna District boundary,
then derives fire-relevant terrain layers:

  - Elevation (m)          — dem_guna_clipped.tif
  - Slope (degrees)        — slope_guna_30m.tif  (fire spreads faster up-slope)
  - Aspect (degrees)       — aspect_guna_30m.tif (south-facing = drier = higher risk)
  - Terrain fire risk mask — terrain_risk_guna.tif
      0 = flat / north-facing  (low terrain risk)
      1 = south-facing only    (moderate risk — drier micro-climate)
      2 = steep (>=30 deg)     (high risk — rapid upslope spread)
      3 = steep + south-facing (extreme terrain risk)
  - terrain_stats.json     — summary for dashboard

Data lake layout::

    data_lake/
      terrain/
        copdem30_N23E076.tif        <- raw 1-deg tile
        copdem30_N24E076.tif
        ...
        dem_guna_raw.tif            <- merged mosaic (EPSG:4326)
        dem_guna_clipped.tif        <- clipped to Guna bbox
        slope_guna_30m.tif          <- slope in degrees (float32)
        aspect_guna_30m.tif         <- aspect in degrees (float32)
        terrain_risk_guna.tif       <- uint8 risk class
        terrain_stats.json          <- stats for dashboard

Copernicus DEM tile URL pattern::

    https://copernicus-dem-30m.s3.amazonaws.com/
      Copernicus_DSM_COG_10_N{lat}_00_E{lon}_00_DEM/
        Copernicus_DSM_COG_10_N{lat}_00_E{lon}_00_DEM.tif

Reference for Horn (1981) slope/aspect derivation:
  Horn, B.K.P. (1981). "Hill Shading and the Reflectance Map".
  Proceedings of the IEEE, 69(1), 14-47.

Usage::

    python scripts/fetch_dem.py
"""

from __future__ import annotations

# ── PROJ/GDAL env fix ──
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()
del _os, _sys
# ────────────────────────

import json
import logging
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests

log = logging.getLogger(__name__)

# ── Guna Division (Madhya Pradesh) ────────────────────────────────────────────
GUNA_BBOX    = (76.45, 23.80, 77.85, 24.95)   # (min_lon, min_lat, max_lon, max_lat)
GUNA_CENTER  = (24.35, 77.15)                  # (lat, lon) — for cell-size calc

# ── Terrain risk thresholds ───────────────────────────────────────────────────
HIGH_SLOPE_DEG   = 30.0   # degrees — fire spread doubles for each 10 deg of slope
SOUTH_ASPECT_MIN = 135    # south-facing window: 135-225 deg = S/SW
SOUTH_ASPECT_MAX = 225    # (in northern hemisphere, south-facing slopes are drier)

# ── Copernicus DEM 30m — AWS Open Data (no auth required) ────────────────────
COPERNICUS_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
TERRAIN_DIR = ROOT / "data_lake" / "terrain"


# ─────────────────────────────────────────────────────────────────────────────
# Tile helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cop30_url(lat: int, lon: int) -> str:
    """Build Copernicus DEM 30m tile URL for (lat, lon) lower-left integer coords."""
    lat_s = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
    lon_s = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
    name  = f"Copernicus_DSM_COG_10_{lat_s}_00_{lon_s}_00_DEM"
    return f"{COPERNICUS_BASE}/{name}/{name}.tif"


def _tiles_for_bbox(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> List[tuple]:
    """Return list of (lat, lon) integer tile origins covering the bbox."""
    tiles = []
    for lat in range(int(math.floor(min_lat)), int(math.ceil(max_lat))):
        for lon in range(int(math.floor(min_lon)), int(math.ceil(max_lon))):
            tiles.append((lat, lon))
    return tiles


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

def download_dem_tiles(
    bbox: tuple = GUNA_BBOX,
    out_dir: Path = TERRAIN_DIR,
) -> List[Path]:
    """
    Download Copernicus DEM 30m tiles covering *bbox* from AWS Open Data.

    Returns a list of local tile paths (skips already-cached files).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    min_lon, min_lat, max_lon, max_lat = bbox
    tiles = _tiles_for_bbox(min_lon, min_lat, max_lon, max_lat)

    local: List[Path] = []
    for lat, lon in tiles:
        url  = _cop30_url(lat, lon)
        lat_s = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
        lon_s = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
        dest = out_dir / f"copdem30_{lat_s}{lon_s}.tif"

        if dest.exists():
            log.info("  Tile cached: %s", dest.name)
            local.append(dest)
            continue

        log.info("  Downloading %s ...", url)
        try:
            resp = requests.get(url, stream=True, timeout=180)
            if resp.status_code == 404:
                log.warning("  Tile not found (404): %s — skipping", url)
                continue
            resp.raise_for_status()

            size = 0
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
                    size += len(chunk)
            log.info("  Saved %s  (%.1f MB)", dest.name, size / 1e6)
            local.append(dest)

        except Exception as exc:
            log.error("  Failed to download %s: %s", url, exc)
            if dest.exists():
                dest.unlink()

    return local


# ─────────────────────────────────────────────────────────────────────────────
# Merge + clip
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_clip(
    tile_paths: List[Path],
    bbox: tuple = GUNA_BBOX,
) -> Path:
    """
    Mosaic DEM tiles and clip to bounding box.

    Returns the path of the clipped raster.
    """
    import rasterio
    from rasterio.merge import merge as rio_merge
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box, mapping

    TERRAIN_DIR.mkdir(parents=True, exist_ok=True)
    out_merged  = TERRAIN_DIR / "dem_guna_raw.tif"
    out_clipped = TERRAIN_DIR / "dem_guna_clipped.tif"

    # ── Mosaic ────────────────────────────────────────────────────────────────
    log.info("Merging %d DEM tile(s) ...", len(tile_paths))
    sources = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = rio_merge(sources)

    profile = sources[0].profile.copy()
    profile.update({
        "driver": "GTiff", "compress": "deflate", "tiled": True,
        "height": mosaic.shape[1], "width": mosaic.shape[2],
        "transform": transform,
    })
    with rasterio.open(out_merged, "w", **profile) as dst:
        dst.write(mosaic)
    for s in sources:
        s.close()
    log.info("  Merged -> %s", out_merged)

    # ── Clip ──────────────────────────────────────────────────────────────────
    min_lon, min_lat, max_lon, max_lat = bbox
    geom = [mapping(box(min_lon, min_lat, max_lon, max_lat))]

    with rasterio.open(out_merged) as src:
        clipped, clip_tfm = rio_mask(src, geom, crop=True, filled=True, nodata=-9999)
        cprofile = src.profile.copy()
        cprofile.update({
            "height": clipped.shape[1], "width": clipped.shape[2],
            "transform": clip_tfm, "nodata": -9999,
            "compress": "deflate",
        })
        with rasterio.open(out_clipped, "w", **cprofile) as dst:
            dst.write(clipped)
    log.info("  Clipped -> %s (%.1f KB)", out_clipped, out_clipped.stat().st_size / 1024)

    return out_clipped


# ─────────────────────────────────────────────────────────────────────────────
# Slope & Aspect — Horn (1981) central-difference method
# ─────────────────────────────────────────────────────────────────────────────

def compute_slope_aspect(dem_path: Path) -> tuple:
    """
    Derive slope (degrees) and aspect (degrees, N=0, E=90, S=180, W=270)
    using the Horn (1981) 3×3 window central-difference algorithm.

    Cell size is approximated in metres for the Guna centre latitude (24°N).

    Returns (slope_path, aspect_path).
    """
    import rasterio

    slope_path  = TERRAIN_DIR / "slope_guna_30m.tif"
    aspect_path = TERRAIN_DIR / "aspect_guna_30m.tif"

    lat_rad = math.radians(GUNA_CENTER[0])
    # WGS84 approximate metres per degree at 24°N
    m_per_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    m_per_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)

    with rasterio.open(dem_path) as src:
        dem      = src.read(1).astype(np.float32)
        tfm      = src.transform
        profile  = src.profile.copy()
        nodata   = src.nodata if src.nodata is not None else -9999

    dx = abs(tfm.a) * m_per_lon   # pixel width in metres
    dy = abs(tfm.e) * m_per_lat   # pixel height in metres

    # Mask nodata before processing
    invalid = dem <= (nodata + 1)
    dem[invalid] = np.nan

    # 3×3 Horn finite-difference kernels
    # (pad to retain original array shape)
    p = np.pad(dem, 1, mode="edge")
    a = p[:-2, :-2];  b = p[:-2, 1:-1];  c = p[:-2, 2:]
    d = p[1:-1, :-2];                     f = p[1:-1, 2:]
    g = p[2:, :-2];   h = p[2:, 1:-1];   i = p[2:, 2:]

    dzdx = ((c + 2*f + i) - (a + 2*d + g)) / (8.0 * dx)
    dzdy = ((g + 2*h + i) - (a + 2*b + c)) / (8.0 * dy)

    slope  = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

    # Aspect: arctan2(-dzdy, dzdx) gives math angle; convert to geographic bearing
    aspect = np.degrees(np.arctan2(-dzdy, dzdx))
    aspect = (90.0 - aspect) % 360.0    # 0=N, 90=E, 180=S, 270=W

    # Restore nodata
    nd_out = np.float32(-9999)
    slope [invalid] = nd_out
    aspect[invalid] = nd_out

    out_profile = profile.copy()
    out_profile.update({"dtype": "float32", "nodata": float(nd_out), "compress": "deflate"})

    with rasterio.open(slope_path,  "w", **out_profile) as dst:
        dst.write(slope [np.newaxis].astype(np.float32))
    with rasterio.open(aspect_path, "w", **out_profile) as dst:
        dst.write(aspect[np.newaxis].astype(np.float32))

    log.info("Slope  -> %s", slope_path)
    log.info("Aspect -> %s", aspect_path)
    return slope_path, aspect_path


# ─────────────────────────────────────────────────────────────────────────────
# Terrain fire-risk mask
# ─────────────────────────────────────────────────────────────────────────────

def compute_terrain_risk(
    slope_path: Path,
    aspect_path: Path,
    high_slope: float   = HIGH_SLOPE_DEG,
    south_min:  float   = SOUTH_ASPECT_MIN,
    south_max:  float   = SOUTH_ASPECT_MAX,
) -> Path:
    """
    Classify terrain fire risk (0-3) based on slope and aspect:

      0 — flat OR north-facing   (low terrain contribution)
      1 — south/SW-facing only   (drier micro-climate, moderate risk)
      2 — steep (>=30°) only     (rapid upslope spread, high risk)
      3 — steep AND south-facing (extreme terrain risk)
      255 — nodata

    Returns path to risk raster.
    """
    import rasterio

    risk_path = TERRAIN_DIR / "terrain_risk_guna.tif"

    with rasterio.open(slope_path)  as ss, \
         rasterio.open(aspect_path) as sa:
        slope  = ss.read(1).astype(np.float32)
        aspect = sa.read(1).astype(np.float32)
        profile = ss.profile.copy()

    nd_mask = (slope <= -9000) | (aspect <= -9000)

    is_steep = (slope >= high_slope) & ~nd_mask
    is_south = ((aspect >= south_min) & (aspect <= south_max)) & ~nd_mask

    risk = np.zeros_like(slope, dtype=np.uint8)
    risk[is_south]              = 1
    risk[is_steep]              = 2
    risk[is_steep & is_south]   = 3
    risk[nd_mask]               = 255

    profile.update({"dtype": "uint8", "nodata": 255, "compress": "deflate"})
    with rasterio.open(risk_path, "w", **profile) as dst:
        dst.write(risk[np.newaxis])

    n_total = int((~nd_mask).sum())
    log.info(
        "Terrain risk -> %s  |  extreme=%.1f%%  steep=%.1f%%  south=%.1f%%  flat=%.1f%%",
        risk_path.name,
        100 * int((risk == 3).sum()) / max(n_total, 1),
        100 * int((risk == 2).sum()) / max(n_total, 1),
        100 * int((risk == 1).sum()) / max(n_total, 1),
        100 * int((risk == 0).sum()) / max(n_total, 1),
    )
    return risk_path


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_terrain_stats(
    dem_path:    Path,
    slope_path:  Path,
    aspect_path: Path,
    risk_path:   Path,
) -> dict:
    """
    Summarise terrain statistics and write terrain_stats.json.
    """
    import rasterio

    def _valid(path, nd_thresh=-9000):
        with rasterio.open(path) as s:
            arr = s.read(1).astype(np.float32)
            nd  = s.nodata if s.nodata is not None else -9999
        return arr[arr > nd + 1]

    elev   = _valid(dem_path)
    slope  = _valid(slope_path)

    with rasterio.open(risk_path) as s:
        risk    = s.read(1)
        nd_risk = s.nodata if s.nodata is not None else 255
    risk_v = risk[risk != nd_risk]
    n = max(len(risk_v), 1)

    # Aspect octant distribution
    with rasterio.open(aspect_path) as s:
        asp = s.read(1).astype(np.float32)
        nd_asp = s.nodata if s.nodata is not None else -9999
    asp_v = asp[asp > nd_asp + 1]
    asp_n = max(len(asp_v), 1)
    octants = {
        "N  (315-360/0-45)":   float(100 * (((asp_v >= 315) | (asp_v < 45))).sum()  / asp_n),
        "NE (45-90)":          float(100 * ((asp_v >= 45)  & (asp_v < 90) ).sum()   / asp_n),
        "E  (90-135)":         float(100 * ((asp_v >= 90)  & (asp_v < 135)).sum()   / asp_n),
        "SE (135-180)":        float(100 * ((asp_v >= 135) & (asp_v < 180)).sum()   / asp_n),
        "S  (180-225)":        float(100 * ((asp_v >= 180) & (asp_v < 225)).sum()   / asp_n),
        "SW (225-270)":        float(100 * ((asp_v >= 225) & (asp_v < 270)).sum()   / asp_n),
        "W  (270-315)":        float(100 * ((asp_v >= 270) & (asp_v < 315)).sum()   / asp_n),
        "NW (315+)":           float(100 * ((asp_v >= 315) & (asp_v < 360)).sum()   / asp_n),
    }

    stats = {
        "elevation_m": {
            "min":  float(elev.min()),
            "max":  float(elev.max()),
            "mean": float(elev.mean()),
            "std":  float(elev.std()),
            "p10":  float(np.percentile(elev, 10)),
            "p90":  float(np.percentile(elev, 90)),
        },
        "slope_deg": {
            "mean":          float(slope.mean()),
            "max":           float(slope.max()),
            "pct_ge_15":     float(100 * (slope >= 15).sum() / max(len(slope), 1)),
            "pct_ge_30":     float(100 * (slope >= 30).sum() / max(len(slope), 1)),
            "pct_ge_45":     float(100 * (slope >= 45).sum() / max(len(slope), 1)),
        },
        "terrain_risk_pct": {
            "flat_low":            float(100 * (risk_v == 0).sum() / n),
            "south_facing_mod":    float(100 * (risk_v == 1).sum() / n),
            "steep_high":          float(100 * (risk_v == 2).sum() / n),
            "steep_south_extreme": float(100 * (risk_v == 3).sum() / n),
        },
        "aspect_octants_pct": octants,
    }

    out_json = TERRAIN_DIR / "terrain_stats.json"
    TERRAIN_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(stats, fh, indent=2)
    log.info("Terrain stats -> %s", out_json)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Point query (used by NRT alert pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def get_terrain_at_point(lat: float, lon: float) -> dict:
    """
    Query terrain attributes at a lat/lon point.

    Returns a dict with keys: elevation_m, slope_deg, aspect_deg, terrain_risk
    (0-3), terrain_risk_label.

    Gracefully returns empty dict if rasters are not available.
    """
    import rasterio
    from rasterio.transform import rowcol

    RISK_LABELS = {0: "Low", 1: "Moderate (south-facing)",
                   2: "High (steep)", 3: "Extreme (steep + south)"}

    rasters = {
        "elevation_m":  TERRAIN_DIR / "dem_guna_clipped.tif",
        "slope_deg":    TERRAIN_DIR / "slope_guna_30m.tif",
        "aspect_deg":   TERRAIN_DIR / "aspect_guna_30m.tif",
        "terrain_risk": TERRAIN_DIR / "terrain_risk_guna.tif",
    }

    result: dict = {}
    for key, path in rasters.items():
        if not path.exists():
            continue
        try:
            with rasterio.open(path) as src:
                row, col = rowcol(src.transform, lon, lat)
                if 0 <= row < src.height and 0 <= col < src.width:
                    val = float(src.read(1)[row, col])
                    nd  = float(src.nodata) if src.nodata is not None else -9999
                    if val != nd and val > nd:
                        result[key] = val
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")

    if "terrain_risk" in result:
        result["terrain_risk_label"] = RISK_LABELS.get(int(result["terrain_risk"]), "Unknown")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    log.info("=" * 60)
    log.info("Van Suraksha — Terrain Data Pipeline (Guna Division)")
    log.info("=" * 60)

    # 1. Download tiles
    log.info("Step 1/4 — Downloading Copernicus DEM 30m tiles ...")
    tiles = download_dem_tiles()
    if not tiles:
        log.error("No DEM tiles available. Check network connectivity.")
        return

    # 2. Merge + clip
    log.info("Step 2/4 — Merging and clipping ...")
    clipped = merge_and_clip(tiles)

    # 3. Slope + aspect
    log.info("Step 3/4 — Computing slope and aspect (Horn 1981) ...")
    slope_p, aspect_p = compute_slope_aspect(clipped)

    # 4. Risk mask + stats
    log.info("Step 4/4 — Terrain risk classification + statistics ...")
    risk_p = compute_terrain_risk(slope_p, aspect_p)
    stats  = compute_terrain_stats(clipped, slope_p, aspect_p, risk_p)

    e = stats["elevation_m"]
    s = stats["slope_deg"]
    r = stats["terrain_risk_pct"]

    log.info("-" * 60)
    log.info("Elevation  : %.0f – %.0f m  (mean %.0f m, std %.0f m)",
             e["min"], e["max"], e["mean"], e["std"])
    log.info("Slope      : mean %.1f deg | >=15 deg: %.1f%% | >=30 deg: %.1f%%",
             s["mean"], s["pct_ge_15"], s["pct_ge_30"])
    log.info("Terrain risk distribution (% of district):")
    log.info("  Extreme  (steep + south): %.1f%%", r["steep_south_extreme"])
    log.info("  High     (steep only):    %.1f%%", r["steep_high"])
    log.info("  Moderate (south-facing):  %.1f%%", r["south_facing_mod"])
    log.info("  Low      (flat/north):    %.1f%%", r["flat_low"])
    log.info("-" * 60)
    log.info("Output files: %s", TERRAIN_DIR)
    log.info("Done.")


if __name__ == "__main__":
    main()
