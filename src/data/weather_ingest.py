"""
Weather Forecast Ingestion via Herbie (GFS / ECMWF)
====================================================
Downloads Numerical Weather Prediction (NWP) GRIB2 forecasts and saves
them as NetCDF4 / Zarr files in the data-lake, ready for Prithvi-WxC
atmospheric encoding and fire-weather index computation.

Fire-weather variables fetched (per PDF §"Automated Retrieval of GFS"):
  - Temperature at 2 m            (TMP:2 m above ground)
  - Relative Humidity at 2 m      (RH:2 m above ground)
  - U & V wind components at 10 m (UGRD:10 m, VGRD:10 m)
  - Total precipitation            (APCP:surface)

Data-lake layout::

    data_lake/
      weather/
        gfs_0p25/
          2025/01/15/
            gfs_f024.nc    <- 24-h lead NetCDF
            gfs_f048.nc    <- 48-h lead NetCDF
            gfs_f072.nc    <- 72-h lead NetCDF

Usage::

    python -m src.data.weather_ingest \\
        --date 2025-03-01 --leads 24 48 72
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import yaml

log = logging.getLogger(__name__)

# ── Forecast runs to try (UTC hours), in priority order ──────────────────────
GFS_RUNS = ["00:00", "06:00", "12:00", "18:00"]

# ── Fire-weather GRIB filter patterns (herbie REGEX against GRIB inventory) ──
FIRE_WEATHER_PATTERNS = [
    ":(TMP|RH):2 m above ground",          # temperature & humidity
    ":(UGRD|VGRD):10 m above ground",      # wind components
    ":APCP:surface",                        # total precipitation
]

# ── Default forecast lead times (hours) ──────────────────────────────────────
DEFAULT_LEADS: List[int] = [24, 48, 72]

DEFAULT_DATA_LAKE = Path("data_lake")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _weather_dir(root: Path, model: str, run_date: date) -> Path:
    """Return directory path: <root>/weather/<model>/<YYYY>/<MM>/<DD>/"""
    return (
        root / "weather" / model
        / str(run_date.year)
        / f"{run_date.month:02d}"
        / f"{run_date.day:02d}"
    )


def _wind_speed_and_direction(u: "xr.DataArray", v: "xr.DataArray"):
    """
    Derive wind speed (m/s) and meteorological direction (degrees from N)
    from U/V component grids.

    Returns (speed, direction) as DataArrays.
    """
    import numpy as np
    speed = (u ** 2 + v ** 2) ** 0.5
    direction = (270 - np.degrees(np.arctan2(v, u))) % 360
    return speed, direction


# ─────────────────────────────────────────────────────────────────────────────
# GFS download
# ─────────────────────────────────────────────────────────────────────────────

def download_gfs(
    run_date: str,
    lead_hours: List[int] = DEFAULT_LEADS,
    data_lake: Path = DEFAULT_DATA_LAKE,
    bbox: Optional[tuple[float, float, float, float]] = None,
    save_zarr: bool = False,
) -> List[Path]:
    """
    Download GFS 0.25-degree fire-weather variables for one *run_date*.

    Tries the 00 UTC run first; falls back to later runs if not yet published.

    Parameters
    ----------
    run_date   : ISO date string, e.g. "2025-03-01".
    lead_hours : List of forecast lead times to fetch (e.g. [24, 48, 72]).
    data_lake  : Root of the local data-lake.
    bbox       : Optional (min_lon, min_lat, max_lon, max_lat) to subset.
                 If None, saves global 0.25° grid.
    save_zarr  : If True, also save a Zarr store alongside the NetCDF.

    Returns
    -------
    List of saved file paths.
    """
    try:
        from herbie import Herbie  # noqa: PLC0415
    except ImportError:
        raise ImportError("herbie-data is not installed.  Run: pip install herbie-data")

    import xarray as xr

    dt = datetime.fromisoformat(run_date)
    dest_dir = _weather_dir(data_lake, "gfs_0p25", dt.date())
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []

    for fxx in lead_hours:
        log.info("Fetching GFS  run=%s  lead=%d h", run_date, fxx)

        # Try each synoptic run until we find an available one
        H = None
        for run_hr in ["00:00", "06:00", "12:00", "18:00"]:
            run_str = f"{run_date} {run_hr}"
            try:
                H = Herbie(run_str, model="gfs", product="pgrb2.0p25", fxx=fxx)
                # Verify the file is actually accessible
                H.inventory()
                log.info("  Using run %s", run_str)
                break
            except Exception as exc:
                log.debug("  Run %s not available: %s", run_str, exc)
                H = None

        if H is None:
            log.warning("  No GFS run found for date=%s  lead=%dh — skipping.", run_date, fxx)
            continue

        # Fetch all fire-weather variables into a single xarray Dataset
        datasets = []
        for pattern in FIRE_WEATHER_PATTERNS:
            try:
                ds = H.xarray(pattern, remove_grib=True)
                datasets.append(ds)
            except Exception as exc:
                log.warning("  Pattern '%s' failed: %s", pattern, exc)

        if not datasets:
            log.warning("  No variables retrieved for lead=%dh — skipping.", fxx)
            continue

        # Merge all variables
        combined = xr.merge(datasets, compat="override")

        # Subset to bounding box if requested
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            combined = combined.sel(
                latitude=slice(max_lat, min_lat),
                longitude=slice(min_lon, max_lon),
            )

        # Derive wind speed & direction if components are present
        if "u10" in combined and "v10" in combined:
            combined["wind_speed_10m"], combined["wind_dir_10m"] = (
                _wind_speed_and_direction(combined["u10"], combined["v10"])
            )
            combined["wind_speed_10m"].attrs["units"] = "m s-1"
            combined["wind_dir_10m"].attrs["units"] = "degrees"

        # Add metadata attributes
        combined.attrs.update({
            "source": "NOAA GFS 0.25-degree",
            "run_date": run_date,
            "lead_hours": fxx,
            "variables": "TMP2m, RH2m, UGRD10m, VGRD10m, APCP_surface",
            "created_by": "jwalaNetra_2/src/data/weather_ingest.py",
        })

        # Save as NetCDF4
        nc_path = dest_dir / f"gfs_f{fxx:03d}.nc"
        combined.to_netcdf(nc_path)
        log.info("  Saved NetCDF → %s", nc_path)
        saved.append(nc_path)

        # Optionally save Zarr (chunked, parallel access across time dimension)
        if save_zarr:
            zarr_path = dest_dir / f"gfs_f{fxx:03d}.zarr"
            combined.chunk({"latitude": 50, "longitude": 50}).to_zarr(
                zarr_path, mode="w"
            )
            log.info("  Saved Zarr   → %s", zarr_path)
            saved.append(zarr_path)

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Fire Weather Index helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_fire_weather_index(nc_path: Path) -> "xr.Dataset":
    """
    Compute a simplified Fire Weather Index (FWI) from a saved GFS NetCDF.

    Formula (Canadian FWI Fine Fuel Moisture Code proxy):
        ffmc_proxy = 100 - RH/2 + (T - 20)/5 - prec*2

    Higher values = greater fire danger.

    Returns an xarray Dataset with 'fwi_proxy' variable.
    """
    import numpy as np
    import xarray as xr

    ds = xr.open_dataset(nc_path)

    # Extract variables (GFS GRIB2 variable names from herbie)
    temp_k = ds.get("t2m", ds.get("TMP_2maboveground"))     # temperature in K
    rh     = ds.get("r2", ds.get("RH_2maboveground"))        # relative humidity %
    prec   = ds.get("tp", ds.get("APCP_surface", None))      # total precip m or kg/m2

    if temp_k is None or rh is None:
        raise KeyError("Required variables TMP or RH not found in dataset.")

    temp_c = temp_k - 273.15  # Kelvin → Celsius

    fwi = 100 - rh / 2 + (temp_c - 20) / 5
    if prec is not None:
        prec_mm = prec * 1000 if prec.attrs.get("units", "").startswith("m") else prec
        fwi = fwi - prec_mm * 2

    fwi = fwi.clip(0, 100).rename("fwi_proxy")
    fwi.attrs = {
        "long_name": "Fire Weather Index proxy (simplified Canadian FWI)",
        "units": "dimensionless [0..100]",
    }
    return fwi.to_dataset()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download GFS fire-weather forecasts via herbie."
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--date", required=True, help="Forecast run date ISO, e.g. 2025-03-01")
    p.add_argument(
        "--leads", nargs="+", type=int, default=DEFAULT_LEADS,
        help="Lead times in hours (default: 24 48 72).",
    )
    p.add_argument("--lake", default=None, help="Data-lake root path.")
    p.add_argument("--zarr", action="store_true", help="Also save Zarr stores.")
    p.add_argument(
        "--subset-bbox", action="store_true",
        help="Subset output to Guna Division bounding box.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _parse_args()

    # Guna Division bbox for subsetting
    GUNA_BBOX = (76.45, 23.80, 77.85, 24.95)

    lake = Path(args.lake) if args.lake else DEFAULT_DATA_LAKE
    bbox = GUNA_BBOX if args.subset_bbox else None

    download_gfs(
        run_date=args.date,
        lead_hours=args.leads,
        data_lake=lake,
        bbox=bbox,
        save_zarr=args.zarr,
    )


if __name__ == "__main__":
    main()
