"""
HLS Satellite Imagery Downloader — Band-Selective, Fire-Season Focus
=====================================================================
Downloads ONLY the 6 Prithvi-EO bands (B02/03/04/B8A/B11/B12 for S30;
B02/03/04/05/06/07 for L30) from HLS granules for Guna Division, MP.

Strategy
--------
Full HLS tiles are ~50-200 MB per band. We use rasterio windowed reading
over HTTPS (COG) to crop to the Guna Division AOI, reducing each 6-band
scene from ~600 MB → ~20-30 MB.

Coverage: fire seasons (Feb–May) for 2013–2025.

Output layout:
  data_lake/satellite_imagery/
    hls_s30/T43RGH/<YYYY>/<MM>/<DD>/<granule_id>.<band>.tif
    hls_l30/T43RGH/<YYYY>/<MM>/<DD>/<granule_id>.<band>.tif

Usage
-----
  python scripts/download_hls_imagery.py
  python scripts/download_hls_imagery.py --year 2024 --product S30
  python scripts/download_hls_imagery.py --start 2023-02-01 --end 2023-05-31 --product both
  python scripts/download_hls_imagery.py --dry-run
"""

import io, os, sys, time, logging, argparse, threading
from pathlib import Path
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# ── PROJ fix: override PostgreSQL PostGIS PROJ/GDAL before any rasterio import
# The PostgreSQL installation ships an old PROJ database that conflicts.
# Must be set BEFORE rasterio/pyproj are imported.
def _fix_proj():
    try:
        import pyproj as _pp
        _proj_data = str(Path(_pp.datadir.get_data_dir()))
        os.environ["PROJ_DATA"] = _proj_data
        os.environ["PROJ_LIB"]  = _proj_data
        # Remove stale GDAL_DATA set by PostgreSQL
        os.environ.pop("GDAL_DATA", None)
    except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")  # pyproj not installed yet — will be caught later

_fix_proj()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_LAKE = ROOT / "data_lake"
LOG_DIR   = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
# Guna Division bbox (with buffer)
GUNA_BBOX = (76.75, 23.85, 77.52, 25.15)   # (west, south, east, north)

# Prithvi-EO required bands
PRITHVI_BANDS_S30 = ["B02", "B03", "B04", "B8A", "B11", "B12"]
PRITHVI_BANDS_L30 = ["B02", "B03", "B04", "B05", "B06", "B07"]

# ── Logging ────────────────────────────────────────────────────────────────────
fh = logging.FileHandler(LOG_DIR / "hls_download.log", encoding="utf-8")
ch = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[fh, ch],
)
log = logging.getLogger("hls_dl")


# ── Authentication ─────────────────────────────────────────────────────────────
def _login():
    import earthaccess
    os.environ.setdefault("EARTHDATA_USERNAME", "akshayr11@gmail.com")
    os.environ.setdefault("EARTHDATA_PASSWORD", "40983#Akki")
    earthaccess.login(strategy="netrc")
    return earthaccess


def _get_https_session():
    """Return an authenticated requests.Session for Earthdata HTTPS access."""
    import earthaccess, requests
    ea = _login()
    session = earthaccess.get_requests_https_session()
    return session


# ── URL helpers ────────────────────────────────────────────────────────────────
def _s3_to_https(s3_url: str) -> str:
    """Convert s3://lp-prod-protected/... to HTTPS Earthdata URL."""
    path = s3_url.replace("s3://lp-prod-protected/", "")
    return f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/{path}"


def _get_band_urls(granule, bands: list[str]) -> dict[str, str]:
    """Return {band_id: https_url} for the requested bands."""
    import earthaccess
    all_links = earthaccess.results.DataGranule.data_links(granule, access="direct")
    if not all_links:
        all_links = earthaccess.results.DataGranule.data_links(granule, access="on_prem")

    result = {}
    for url in all_links:
        for band in bands:
            if f".{band}." in url or url.endswith(f".{band}.tif"):
                https_url = _s3_to_https(url) if url.startswith("s3://") else url
                result[band] = https_url
                break
    return result


# ── Output path helper ─────────────────────────────────────────────────────────
def _out_dir(product: str, tile: str, acq_date: date) -> Path:
    product_key = f"hls_{product.lower()}"
    return (
        DATA_LAKE / "satellite_imagery" / product_key / tile
        / str(acq_date.year) / f"{acq_date.month:02d}" / f"{acq_date.day:02d}"
    )


# ── COG windowed crop & save ───────────────────────────────────────────────────
def _crop_and_save(session, file_obj, out_path: Path) -> int:
    """
    Open an earthaccess EarthAccessFile (streaming COG), read only the Guna
    Division window via rasterio windowed I/O, and save as compressed GeoTIFF.
    Returns file size in bytes.
    """
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import from_bounds as wfb

    try:
        with rasterio.open(file_obj) as src:
            tb = src.bounds
            # Project Guna bbox → tile CRS using pyproj (avoids EPSG lookup)
            tr = Transformer.from_crs("EPSG:4326", src.crs.to_wkt(), always_xy=True)
            x0_src, y0_src = tr.transform(GUNA_BBOX[0], GUNA_BBOX[1])
            x1_src, y1_src = tr.transform(GUNA_BBOX[2], GUNA_BBOX[3])

            # Clip to tile bounds (tile may only partially overlap AOI)
            cx0 = max(x0_src, tb.left);  cx1 = min(x1_src, tb.right)
            cy0 = max(y0_src, tb.bottom); cy1 = min(y1_src, tb.top)

            if cx1 <= cx0 or cy1 <= cy0:
                return 0   # tile does not overlap Guna bbox → skip

            win = wfb(cx0, cy0, cx1, cy1, src.transform)
            data = src.read(1, window=win)
            win_transform = src.window_transform(win)

            profile = src.profile.copy()
            profile.update({
                "height"    : data.shape[0],
                "width"     : data.shape[1],
                "transform" : win_transform,
                "compress"  : "lzw",
                "tiled"     : True,
                "blockxsize": 256,
                "blockysize": 256,
                "driver"    : "GTiff",
                "count"     : 1,
            })

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)

        return out_path.stat().st_size

    except Exception as exc:
        raise RuntimeError(f"rasterio crop failed: {exc}") from exc


# ── Alternative: direct HTTPS download of full band file ──────────────────────
def _download_band(session, band_url: str, out_path: Path) -> int:
    """
    Directly download a full band GeoTIFF (simpler but larger files).
    Uses earthaccess authenticated session.
    """
    import requests
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = session.get(band_url, stream=True, timeout=120)
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):   # 1 MB chunks
                f.write(chunk)
        return out_path.stat().st_size
    except Exception as exc:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {band_url}: {exc}") from exc


# ── Core download function ─────────────────────────────────────────────────────
def download_hls_season(
    product: str = "S30",
    start:   str = "2024-02-01",
    end:     str = "2024-05-31",
    cloud_cover_max: int = 15,
    workers: int = 3,
    crop_to_aoi: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Download/crop HLS granules (only 6 Prithvi bands) for the given period.
    Returns summary dict.
    """
    import earthaccess
    from datetime import datetime

    _login()
    bands = PRITHVI_BANDS_S30 if product == "S30" else PRITHVI_BANDS_L30
    session = _get_https_session()

    log.info(f"Searching HLS{product} {start} -> {end} cc<={cloud_cover_max}% ...")
    results = earthaccess.search_data(
        short_name=f"HLS{product}",
        cloud_hosted=True,
        temporal=(start, end),
        bounding_box=GUNA_BBOX,
        cloud_cover=(0, cloud_cover_max),
    )
    log.info(f"Found {len(results)} granules for HLS{product}")

    if dry_run:
        for g in results[:5]:
            ur = g["umm"]["GranuleUR"]
            log.info(f"  DRY-RUN: {ur}")
        return {"granules": len(results), "downloaded": 0, "skipped": 0, "errors": 0, "bytes": 0}

    summary = {"granules": len(results), "downloaded": 0, "skipped": 0, "errors": 0, "bytes": 0}
    lock = threading.Lock()

    def _process_granule(granule):
        import earthaccess as _ea
        ur  = granule["umm"]["GranuleUR"]
        acq_str = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
        acq_date = datetime.fromisoformat(acq_str[:10]).date()

        # Extract tile ID from granule UR (e.g. T43RGH from HLS.S30.T43RGH....)
        parts = ur.split(".")
        tile  = next((p for p in parts if p.startswith("T") and len(p) == 6), "UNKNOWN")
        out_dir = _out_dir(product, tile, acq_date)

        # Check if all 6 bands already exist (skip granule entirely)
        existing = [f for f in bands if any(out_dir.glob(f"*{f}*.tif"))] if out_dir.exists() else []
        if len(existing) == len(bands):
            with lock:
                summary["skipped"] += len(bands)
            return

        # Open all band files via earthaccess streaming
        try:
            file_objs = _ea.open([granule])
        except Exception as e:
            log.warning(f"  [{ur}] open failed: {e}")
            return

        granule_bytes = 0
        n_done = 0
        n_skipped = 0

        for band in bands:
            # Find file object for this band
            fo = next((f for f in file_objs if f".{band}." in str(f)), None)
            if fo is None:
                continue

            # Determine output filename from file object path
            fname = str(fo).split("/")[-1].split("\\")[-1]
            if not fname.endswith(".tif"):
                fname = f"{ur}.{band}.tif"
            out_path = out_dir / fname

            if out_path.exists():
                n_skipped += 1
                continue

            try:
                if crop_to_aoi:
                    sz = _crop_and_save(session, fo, out_path)
                else:
                    sz = _download_band(session, str(fo), out_path)
                granule_bytes += sz
                n_done += 1
            except Exception as exc:
                log.warning(f"  [{ur}] {band}: {exc}")
                with lock:
                    summary["errors"] += 1

        with lock:
            summary["downloaded"] += n_done
            summary["skipped"]    += n_skipped
            summary["bytes"]      += granule_bytes
            done_total = summary["downloaded"] + summary["skipped"]
            if n_done > 0:
                mb = granule_bytes / 1024 / 1024
                log.info(
                    f"  {acq_date} {tile} {product}: "
                    f"{n_done} bands saved, {mb:.1f} MB  "
                    f"[{done_total}/{len(results)*len(bands)} total files]"
                )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_process_granule, g) for g in results]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                log.warning(f"Granule error: {e}")
                with lock:
                    summary["errors"] += 1

    total_mb = summary["bytes"] / 1024 / 1024
    log.info(
        f"[HLS{product}] Done: {summary['downloaded']} files, "
        f"{summary['skipped']} skipped, {summary['errors']} errors, "
        f"{total_mb:.0f} MB"
    )
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Download HLS fire-season imagery")
    p.add_argument("--start",   default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end",     default=None, help="End date YYYY-MM-DD")
    p.add_argument("--year",    type=int, default=None, help="Download a single fire year (Feb-May)")
    p.add_argument("--product", choices=["S30", "L30", "both"], default="S30")
    p.add_argument("--cloud",   type=int, default=15, help="Max cloud cover %%")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--no-crop", action="store_true", help="Download full tile (no AOI crop)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.year:
        start = f"{args.year}-02-01"
        end   = f"{args.year}-05-31"
    elif args.start and args.end:
        start = args.start
        end   = args.end
    else:
        # Default: current fire season
        start = "2024-02-01"
        end   = "2025-05-31"

    products = ["S30", "L30"] if args.product == "both" else [args.product]
    grand_summary = {}

    for prod in products:
        log.info(f"=== HLS{prod} {start} -> {end} ===")
        s = download_hls_season(
            product=prod,
            start=start,
            end=end,
            cloud_cover_max=args.cloud,
            workers=args.workers,
            crop_to_aoi=not args.no_crop,
            dry_run=args.dry_run,
        )
        grand_summary[prod] = s

    log.info("=== Final Summary ===")
    total_files = 0
    total_mb    = 0
    for prod, s in grand_summary.items():
        mb = s["bytes"] / 1024 / 1024
        log.info(
            f"  HLS{prod}: {s['granules']} granules | "
            f"{s['downloaded']} files downloaded | "
            f"{s['skipped']} skipped | {s['errors']} errors | {mb:.0f} MB"
        )
        total_files += s["downloaded"]
        total_mb    += mb
    log.info(f"  TOTAL: {total_files} files, {total_mb:.0f} MB")


if __name__ == "__main__":
    main()
