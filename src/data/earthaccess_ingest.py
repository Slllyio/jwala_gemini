"""
HLS Ingestion via NASA earthaccess
====================================
Downloads Harmonized Landsat Sentinel-2 (HLS) imagery directly from NASA
Earthdata as Cloud-Optimised GeoTIFFs (COGs), organised into the project
data-lake directory hierarchy.

Data-lake layout::

    data_lake/
      satellite_imagery/
        hls_s30/          <- Sentinel-2 derived (30 m, ~2-3 day revisit)
          UTM_44N/        <- geography bucket (UTM zone covering Guna Div)
            2025/01/15/
              HLS.S30.T44RKP.2025015T054329.v2.0.B02.tif
              ...
        hls_l30/          <- Landsat 8/9 derived (30 m)
          UTM_44N/
            ...

Prithvi-required bands (order matches pre-training):
  B02 Blue  |  B03 Green  |  B04 Red  |  B8A NIR-narrow  |  B11 SWIR1  |  B12 SWIR2

Usage::

    python -m src.data.earthaccess_ingest \\
        --start 2025-01-01 --end 2025-03-01 \\
        --cloud 15 --product S30
"""

from __future__ import annotations

import src.utils.proj_fix  # noqa: F401  (PROJ/GDAL env fix)

import argparse
import logging
import os
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import yaml

log = logging.getLogger(__name__)

# ── Prithvi band registry ─────────────────────────────────────────────────────
PRITHVI_BANDS_S30 = ["B02", "B03", "B04", "B8A", "B11", "B12"]  # Sentinel-2
PRITHVI_BANDS_L30 = ["B02", "B03", "B04", "B05", "B06", "B07"]  # Landsat 8/9

# ── Guna Division bounding box (WGS-84) ──────────────────────────────────────
GUNA_BBOX = (76.45, 23.80, 77.85, 24.95)   # (min_lon, min_lat, max_lon, max_lat)
GUNA_UTM_TAG = "UTM_44N"                   # UTM zone label used in directory names

# ── Default data-lake root (relative to project) ─────────────────────────────
DEFAULT_DATA_LAKE = Path("data_lake")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lake_dir(root: Path, product: str, acq_date: date) -> Path:
    """
    Return the directory path for a given product and acquisition date.

    Structure: <root>/satellite_imagery/hls_<product_lower>/<utm>/<YYYY>/<MM>/<DD>
    """
    product_key = f"hls_{product.lower()}"
    return (
        root
        / "satellite_imagery"
        / product_key
        / GUNA_UTM_TAG
        / str(acq_date.year)
        / f"{acq_date.month:02d}"
        / f"{acq_date.day:02d}"
    )


def _files_per_directory_ok(directory: Path, limit: int = 900) -> bool:
    """Guard: keep files-per-directory below *limit* to avoid FS latency."""
    if not directory.exists():
        return True
    count = sum(1 for _ in directory.iterdir())
    if count >= limit:
        log.warning(
            "Directory %s has %d files (limit %d). "
            "Consider sub-partitioning by tile.",
            directory, count, limit,
        )
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Core download
# ─────────────────────────────────────────────────────────────────────────────

def download_hls(
    start: str,
    end: str,
    cloud_cover_max: int = 15,
    product: str = "S30",
    bbox: tuple[float, float, float, float] = GUNA_BBOX,
    data_lake: Path = DEFAULT_DATA_LAKE,
    dry_run: bool = False,
) -> List[Path]:
    """
    Search and bulk-download HLS granules from NASA Earthdata.

    Parameters
    ----------
    start / end     : ISO date strings, e.g. "2025-01-01"
    cloud_cover_max : Maximum cloud cover percentage (0-100).  The PDF
                      specifies < 15 % as "non-negotiable" for high-signal
                      imagery reaching Prithvi.
    product         : "S30" (Sentinel-2) or "L30" (Landsat 8/9).
    bbox            : (min_lon, min_lat, max_lon, max_lat) in WGS-84.
    data_lake       : Root of the local data-lake.
    dry_run         : If True, search only — do not download.

    Returns
    -------
    List of downloaded file paths.
    """
    try:
        import earthaccess  # noqa: PLC0415  (lazy import keeps module loadable)
    except ImportError:
        raise ImportError(
            "earthaccess is not installed.  Run: pip install earthaccess"
        )

    short_name = f"HLS{product}"   # HLSS30 or HLSL30
    bands = PRITHVI_BANDS_S30 if product == "S30" else PRITHVI_BANDS_L30

    log.info(
        "Authenticating with NASA Earthdata …  "
        "(uses ~/.netrc or interactive prompt)"
    )
    earthaccess.login(strategy="netrc", persist=True)

    log.info(
        "Searching %s  |  %s → %s  |  cloud ≤ %d%%  |  bbox %s",
        short_name, start, end, cloud_cover_max, bbox,
    )
    results = earthaccess.search_data(
        short_name=short_name,
        cloud_hosted=True,
        temporal=(start, end),
        bounding_box=bbox,
        cloud_cover=(0, cloud_cover_max),
    )
    log.info("Found %d granules.", len(results))

    if dry_run or not results:
        return []

    downloaded: List[Path] = []
    for granule in results:
        # Derive acquisition date from granule metadata
        acq_str: str = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
        acq_date = datetime.fromisoformat(acq_str[:10]).date()

        dest_dir = _lake_dir(data_lake, product, acq_date)
        dest_dir.mkdir(parents=True, exist_ok=True)
        _files_per_directory_ok(dest_dir)

        # earthaccess.download returns a list of local paths
        paths = earthaccess.download([granule], str(dest_dir))
        downloaded.extend(Path(p) for p in paths)

    log.info("Downloaded %d files to %s", len(downloaded), data_lake)
    return downloaded


# ─────────────────────────────────────────────────────────────────────────────
# Band-selective COG reader (for preprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def load_hls_stack(
    granule_dir: Path,
    product: str = "S30",
) -> "numpy.ndarray":
    """
    Load a 6-band HLS stack from *granule_dir* as a (6, H, W) float32 array
    with DN values scaled to surface-reflectance range [0, 1].

    The band order matches Prithvi pre-training:
      [Blue, Green, Red, NIR-narrow, SWIR1, SWIR2]

    Requires: rasterio
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling

    bands = PRITHVI_BANDS_S30 if product == "S30" else PRITHVI_BANDS_L30
    tifs = sorted(granule_dir.glob("*.tif"))

    stack_layers = []
    for band_id in bands:
        matched = [t for t in tifs if f".{band_id}." in t.name]
        if not matched:
            raise FileNotFoundError(
                f"Band {band_id} not found in {granule_dir}. "
                f"Available: {[t.name for t in tifs]}"
            )
        with rasterio.open(matched[0]) as src:
            data = src.read(1, resampling=Resampling.bilinear).astype(np.float32)
            # HLS scale factor: DN / 10000 = surface reflectance
            data = np.clip(data / 10_000.0, 0.0, 1.0)
            stack_layers.append(data)

    return np.stack(stack_layers, axis=0)   # (6, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download HLS imagery from NASA Earthdata (earthaccess)."
    )
    p.add_argument("--config", default="config.yaml", help="Project YAML config.")
    p.add_argument("--start", required=True, help="Start date ISO, e.g. 2025-01-01")
    p.add_argument("--end",   required=True, help="End date ISO,   e.g. 2025-03-01")
    p.add_argument("--cloud", type=int, default=15, help="Max cloud cover %% (default 15).")
    p.add_argument(
        "--product", choices=["S30", "L30", "both"], default="S30",
        help="HLS product: S30 (Sentinel-2), L30 (Landsat), or both.",
    )
    p.add_argument("--lake",    default=None, help="Data-lake root path.")
    p.add_argument("--dry-run", action="store_true", help="Search only; do not download.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _parse_args()

    cfg = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    lake = Path(args.lake) if args.lake else DEFAULT_DATA_LAKE
    products = ["S30", "L30"] if args.product == "both" else [args.product]

    for prod in products:
        download_hls(
            start=args.start,
            end=args.end,
            cloud_cover_max=args.cloud,
            product=prod,
            data_lake=lake,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
