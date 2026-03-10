"""
FIRMS VIIRS Fire Detection Ingestion
=====================================
Retrieves near-real-time active fire detections from NASA's Fire Information
for Resource Management System (FIRMS) via its REST API, with detections
delivered within ~3 hours of satellite overpass.

VIIRS I-Band (375 m) fields used per PDF §"High-Frequency Fire Detection":
  - bright_ti4  : I-4 channel brightness temperature (K) — intensity
  - scan / track : pixel dimensions (km) — accuracy weighting
  - frp          : Fire Radiative Power (MW) — energy release rate
  - confidence   : detection quality (low / nominal / high)
  - daynight     : overpass phase D/N — diurnal behaviour modelling
  - latitude / longitude : WGS-84 centroid

Data-lake layout::

    data_lake/
      fire_detections/
        firms_viirs/
          2025/01/15/
            viirs_nrt_20250115.parquet   <- GeoParquet (ACID-safe via Delta-style)
            viirs_nrt_20250115.geojson   <- human-readable / dashboards

Usage::

    # Fetch last 24 h of detections over Guna Division
    python -m src.data.firms_ingest --days 1

    # Fetch a custom date range
    python -m src.data.firms_ingest --start 2025-03-01 --end 2025-03-07
"""

from __future__ import annotations

import src.utils.proj_fix  # noqa: F401  (PROJ/GDAL env fix)

import argparse
import io
import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlencode

log = logging.getLogger(__name__)

# ── FIRMS API base URL ────────────────────────────────────────────────────────
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# ── Guna Division bounding box ───────────────────────────────────────────────
GUNA_BBOX = (76.45, 23.80, 77.85, 24.95)   # (min_lon, min_lat, max_lon, max_lat)

# ── VIIRS confidence levels to accept ────────────────────────────────────────
ACCEPTED_CONFIDENCE = {"nominal", "high"}   # drop "low" to reduce false positives

DEFAULT_DATA_LAKE = Path("data_lake")

# ── FIRMS API map key env-var ─────────────────────────────────────────────────
ENV_MAP_KEY = "FIRMS_MAP_KEY"   # set this env-var or pass --api-key


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _firms_dir(root: Path, acq_date: date) -> Path:
    return (
        root / "fire_detections" / "firms_viirs"
        / str(acq_date.year)
        / f"{acq_date.month:02d}"
        / f"{acq_date.day:02d}"
    )


def _bbox_to_area(bbox: tuple[float, float, float, float]) -> str:
    """FIRMS API expects "min_lon,min_lat,max_lon,max_lat" as a string."""
    return ",".join(str(v) for v in bbox)


def _parse_csv_to_gdf(csv_text: str) -> "geopandas.GeoDataFrame":
    """
    Parse FIRMS CSV text into a GeoDataFrame with geometry from lat/lon columns.
    Applies confidence filter and computes derived fields.
    """
    import io as _io

    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    if not csv_text.strip():
        log.warning("Empty FIRMS response.")
        return gpd.GeoDataFrame()

    df = pd.read_csv(_io.StringIO(csv_text))

    # Normalise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    # Filter by confidence
    if "confidence" in df.columns:
        df = df[df["confidence"].str.strip().str.lower().isin(ACCEPTED_CONFIDENCE)]

    if df.empty:
        return gpd.GeoDataFrame()

    # Build geometry
    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Parse acquisition datetime
    if "acq_date" in gdf.columns and "acq_time" in gdf.columns:
        time_str = gdf["acq_time"].astype(str).str.zfill(4)
        gdf["acq_datetime"] = pd.to_datetime(
            gdf["acq_date"].astype(str) + " "
            + time_str.str[:2] + ":" + time_str.str[2:],
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        )

    # FRP intensity category
    if "frp" in gdf.columns:
        gdf["frp_category"] = pd.cut(
            gdf["frp"],
            bins=[0, 10, 50, 200, float("inf")],
            labels=["low", "moderate", "high", "extreme"],
        )

    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# Retry + cache fallback
# ─────────────────────────────────────────────────────────────────────────────

_MAX_RETRIES = 3
_BACKOFF_BASE_S = 2.0   # 2s, 4s, 8s exponential backoff


def _fetch_with_retry(
    url: str,
    chunk_label: str = "",
) -> Optional["geopandas.GeoDataFrame"]:
    """
    Fetch FIRMS CSV with exponential backoff.

    Returns
    -------
    GeoDataFrame on success, empty GeoDataFrame if API returned no data,
    or None if all retries exhausted (signals caller to use cache).
    """
    import requests as _req

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = _req.get(url, timeout=60)
            resp.raise_for_status()
            return _parse_csv_to_gdf(resp.text)
        except _req.RequestException as exc:
            delay = _BACKOFF_BASE_S * (2 ** (attempt - 1))
            log.warning(
                "FIRMS API error (%s) attempt %d/%d: %s — retrying in %.0fs",
                chunk_label, attempt, _MAX_RETRIES, exc, delay,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(delay)

    log.error("FIRMS API failed after %d retries for %s", _MAX_RETRIES, chunk_label)
    return None   # signals "use cache fallback"


def _load_cached_range(
    start: date,
    end: date,
    data_lake: Path,
) -> Optional["geopandas.GeoDataFrame"]:
    """Load previously cached GeoParquet detections for a date range."""
    import geopandas as gpd

    gdfs: List["geopandas.GeoDataFrame"] = []
    current = start
    while current <= end:
        date_tag = current.strftime("%Y%m%d")
        parquet_path = _firms_dir(data_lake, current) / f"viirs_nrt_{date_tag}.parquet"
        if parquet_path.exists():
            try:
                gdfs.append(gpd.read_parquet(parquet_path))
            except Exception as exc:
                log.warning("Failed to read cached %s: %s", parquet_path, exc)
        current += timedelta(days=1)

    if not gdfs:
        return None
    combined = gpd.pd.concat(gdfs, ignore_index=True)
    return gpd.GeoDataFrame(combined, crs="EPSG:4326")


# ─────────────────────────────────────────────────────────────────────────────
# Core download
# ─────────────────────────────────────────────────────────────────────────────

def fetch_viirs_detections(
    api_key: str,
    start_date: date,
    end_date: Optional[date] = None,
    bbox: tuple[float, float, float, float] = GUNA_BBOX,
    data_lake: Path = DEFAULT_DATA_LAKE,
    source: str = "VIIRS_NOAA20_NRT",
) -> "geopandas.GeoDataFrame":
    """
    Fetch VIIRS active fire detections from FIRMS REST API and save to
    GeoParquet + GeoJSON in the data-lake.

    Parameters
    ----------
    api_key    : NASA FIRMS MAP_KEY (https://firms.modaps.eosdis.nasa.gov/api/area/).
    start_date : First day to retrieve.
    end_date   : Last day (inclusive). Defaults to start_date.
    bbox       : (min_lon, min_lat, max_lon, max_lat) WGS-84.
    data_lake  : Root of the local data-lake.
    source     : FIRMS data source key.  Options:
                   "VIIRS_NOAA20_NRT"   (NOAA-20, 375 m, near-real-time)
                   "VIIRS_SNPP_NRT"     (Suomi-NPP, 375 m, near-real-time)
                   "MODIS_NRT"          (MODIS, 1 km, older sensor)

    Returns
    -------
    Combined GeoDataFrame of all detections across the requested date range.
    """
    import geopandas as gpd

    if end_date is None:
        end_date = start_date

    area_str = _bbox_to_area(bbox)
    day_range = (end_date - start_date).days + 1

    # FIRMS API: one request covers up to 10 days
    max_days_per_request = 10
    all_gdfs: List["geopandas.GeoDataFrame"] = []

    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=max_days_per_request - 1), end_date)
        days = (chunk_end - current).days + 1

        url = (
            f"{FIRMS_BASE}/{api_key}/{source}/{area_str}/{days}/{current.isoformat()}"
        )
        log.info("Fetching FIRMS  %s → %s  (%d days)  …", current, chunk_end, days)

        gdf = _fetch_with_retry(url, chunk_label=f"{current}→{chunk_end}")
        if gdf is not None and not gdf.empty:
            log.info("  → %d detections (confidence: nominal/high)", len(gdf))
            all_gdfs.append(gdf)
        elif gdf is None:
            # API failed after retries — try cached data for these dates
            cached = _load_cached_range(current, chunk_end, data_lake)
            if cached is not None and not cached.empty:
                log.info("  → using %d cached detections (API unavailable)", len(cached))
                all_gdfs.append(cached)
            else:
                log.warning("  → NO data for %s→%s (API failed, no cache)", current, chunk_end)

        # Respect API rate limits
        time.sleep(1)
        current = chunk_end + timedelta(days=1)

    if not all_gdfs:
        log.warning("No VIIRS detections found for requested period.")
        return gpd.GeoDataFrame()

    combined = gpd.pd.concat(all_gdfs, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, crs="EPSG:4326")

    # Save per-day partitions to the data-lake
    if "acq_date" in combined.columns:
        for acq_date_str, grp in combined.groupby("acq_date"):
            acq_date = datetime.strptime(str(acq_date_str), "%Y-%m-%d").date()
            _save_detections(grp, acq_date, data_lake)
    else:
        _save_detections(combined, start_date, data_lake)

    return combined


def _save_detections(
    gdf: "geopandas.GeoDataFrame",
    acq_date: date,
    data_lake: Path,
) -> None:
    """Save a GeoDataFrame to GeoParquet and GeoJSON in the data-lake."""
    dest_dir = _firms_dir(data_lake, acq_date)
    dest_dir.mkdir(parents=True, exist_ok=True)

    date_tag = acq_date.strftime("%Y%m%d")

    # GeoParquet — fast spatial joins (ACID-safe for incremental updates)
    parquet_path = dest_dir / f"viirs_nrt_{date_tag}.parquet"
    gdf.to_parquet(parquet_path, index=False)
    log.info("  Saved GeoParquet → %s  (%d rows)", parquet_path, len(gdf))

    # GeoJSON — human-readable / dashboard consumption
    geojson_path = dest_dir / f"viirs_nrt_{date_tag}.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")
    log.info("  Saved GeoJSON    → %s", geojson_path)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial query utility
# ─────────────────────────────────────────────────────────────────────────────

def load_detections_for_date(acq_date: date, data_lake: Path) -> "geopandas.GeoDataFrame":
    """Load cached GeoParquet detections for a given date."""
    import geopandas as gpd

    date_tag = acq_date.strftime("%Y%m%d")
    parquet_path = _firms_dir(data_lake, acq_date) / f"viirs_nrt_{date_tag}.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No cached detections for {acq_date}. "
            f"Run fetch_viirs_detections() first.  Expected: {parquet_path}"
        )
    return gpd.read_parquet(parquet_path)


def summarise_detections(gdf: "geopandas.GeoDataFrame") -> dict:
    """
    Return a summary dictionary of VIIRS fire activity metrics:
      - total_detections
      - mean_frp_mw          : mean Fire Radiative Power
      - max_frp_mw           : peak FRP (proxy for fire intensity)
      - total_frp_mw         : aggregate energy release rate
      - day_detections       : daytime overpass count
      - night_detections     : nighttime overpass count
      - high_confidence_pct  : % of detections at 'high' confidence
    """
    if gdf.empty:
        return {}

    frp = gdf.get("frp", None)
    summary: dict = {"total_detections": len(gdf)}

    if frp is not None:
        summary["mean_frp_mw"]  = float(frp.mean())
        summary["max_frp_mw"]   = float(frp.max())
        summary["total_frp_mw"] = float(frp.sum())

    if "daynight" in gdf.columns:
        dn = gdf["daynight"].str.upper()
        summary["day_detections"]   = int((dn == "D").sum())
        summary["night_detections"] = int((dn == "N").sum())

    if "confidence" in gdf.columns:
        high_pct = (
            gdf["confidence"].str.lower().eq("high").sum() / len(gdf) * 100
        )
        summary["high_confidence_pct"] = round(high_pct, 1)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch FIRMS VIIRS active fire detections."
    )
    p.add_argument(
        "--api-key",
        default=None,
        help=(
            "NASA FIRMS MAP_KEY. "
            f"Falls back to ${ENV_MAP_KEY} environment variable. "
            "Register free at https://firms.modaps.eosdis.nasa.gov/api/area/"
        ),
    )
    p.add_argument("--start", required=True, help="Start date ISO, e.g. 2025-03-01")
    p.add_argument("--end",   default=None,  help="End date ISO (default = same as start)")
    p.add_argument(
        "--days", type=int, default=None,
        help="Alternative to --end: fetch last N days ending today.",
    )
    p.add_argument("--lake", default=None, help="Data-lake root path.")
    p.add_argument(
        "--source",
        default="VIIRS_NOAA20_NRT",
        choices=["VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT", "MODIS_NRT"],
    )
    return p.parse_args()


def main() -> None:
    import os

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _parse_args()

    api_key = args.api_key or os.environ.get(ENV_MAP_KEY)
    if not api_key:
        raise SystemExit(
            f"FIRMS API key required.  Set ${ENV_MAP_KEY} or pass --api-key."
        )

    start = datetime.strptime(args.start, "%Y-%m-%d").date()

    if args.days:
        end = date.today()
        start = end - timedelta(days=args.days - 1)
    elif args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end = start

    lake = Path(args.lake) if args.lake else DEFAULT_DATA_LAKE

    gdf = fetch_viirs_detections(
        api_key=api_key,
        start_date=start,
        end_date=end,
        data_lake=lake,
        source=args.source,
    )

    summary = summarise_detections(gdf)
    log.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
