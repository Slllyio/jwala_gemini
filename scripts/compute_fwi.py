"""
Canadian Fire Weather Index (FWI) System for Guna Division
===========================================================
Implements the full 6-component Canadian FWI system (Van Wagner 1987):

  FFMC  Fine Fuel Moisture Code      - moisture in surface litter / grass
  DMC   Duff Moisture Code           - moisture in loosely packed organic matter
  DC    Drought Code                 - moisture in deep compact organic matter
  ISI   Initial Spread Index         - rate of fire spread (FFMC + wind)
  BUI   Buildup Index                - fuel available for combustion (DMC + DC)
  FWI   Fire Weather Index           - overall fire intensity / danger (ISI + BUI)

Data sources:
  Historical (2013-present): NASA POWER API
    - Free, no account required, ERA5-based reanalysis at 0.5 deg
    - Variables: T2M, RH2M, WS2M, PRECTOTCORR (daily at Guna centroid)
  NRT (today + 24/48/72h forecast): GFS NetCDF in data_lake/weather/gfs_0p25/

Fire danger classes (Canada / NDMC scale):
  FWI 0-5      Low
  FWI 5-11.2   Moderate
  FWI 11.2-21.3 High
  FWI 21.3-38  Very High
  FWI >38      Extreme

Output:
  data_lake/fire_weather/fwi_daily.parquet   <- full historical + NRT series
  data_lake/fire_weather/fwi_daily.csv       <- human-readable

Usage:
  python scripts/compute_fwi.py                      # fetch NASA POWER + compute
  python scripts/compute_fwi.py --start 2017 --end 2025
  python scripts/compute_fwi.py --update-nrt         # append today from GFS
  python scripts/compute_fwi.py --plot               # save FWI time-series PNG

References:
  Van Wagner, C.E. 1987. Development and structure of the Canadian Forest
    Fire Weather Index System. Forestry Technical Report 35. Petawawa National
    Forestry Institute, Chalk River, Ontario. 37 pp.
"""

import argparse
import json
import math
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
OUT_DIR     = ROOT / "data_lake" / "fire_weather"
GFS_DIR     = ROOT / "data_lake" / "weather" / "gfs_0p25"
LOG_DIR     = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Guna Division centroid (WGS84) ────────────────────────────────────────────
GUNA_LAT  = 24.38
GUNA_LON  = 77.15
GUNA_BBOX = (76.45, 23.80, 77.85, 24.95)   # (W, S, E, N)

# ── FWI startup values (Van Wagner 1987 Table 1 — spring startup) ─────────────
FFMC0 = 85.0    # fine fuel moisture code start
DMC0  = 6.0     # duff moisture code start
DC0   = 15.0    # drought code start

# ── Day-length adjustment factors for ~24N latitude ──────────────────────────
# Le: DMC day-length factor (hours of sunshine proxy)
LE = {1: 6.4, 2: 7.5, 3: 9.0, 4: 10.4, 5: 11.4, 6: 11.4,
      7: 11.4, 8: 10.2, 9: 9.0, 10: 7.7, 11: 6.8, 12: 6.2}

# Lf: DC day-length factor (potential evapotranspiration correction)
LF = {1: -1.6, 2: -1.6, 3: -1.6, 4: 0.9, 5: 3.8, 6: 5.8,
      7: 6.4,  8: 5.0,  9: 2.4, 10: 0.4, 11: -1.6, 12: -1.6}

# ── Fire danger classes (FWI thresholds) ──────────────────────────────────────
def fwi_class(fwi: float) -> str:
    if fwi < 5:    return "Low"
    if fwi < 11.2: return "Moderate"
    if fwi < 21.3: return "High"
    if fwi < 38:   return "Very High"
    return "Extreme"


# =============================================================================
# SECTION 1: Canadian FWI Computation Engine (Van Wagner 1987)
# =============================================================================

def _ffmc(T: float, H: float, W: float, ro: float, Fo: float) -> float:
    """
    Fine Fuel Moisture Code (FFMC).

    Parameters
    ----------
    T  : Temperature at noon (C)
    H  : Relative humidity at noon (%)
    W  : Wind speed at noon (km/h)
    ro : 24-hour accumulated precipitation (mm)
    Fo : Previous FFMC value

    Returns
    -------
    New FFMC value
    """
    # Previous moisture content
    mo = 147.2 * (101.0 - Fo) / (59.5 + Fo)

    # Rain effect on moisture
    if ro > 0.5:
        rf = ro - 0.5
        if mo > 150.0:
            mr = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * \
                 (1.0 - math.exp(-6.93 / rf)) + 0.0015 * (mo - 150.0)**2 * rf**0.5
        else:
            mr = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * \
                 (1.0 - math.exp(-6.93 / rf))
        mo = min(mr, 250.0)

    # Equilibrium moisture for drying
    Ed = (0.942 * H**0.679
          + 11.0 * math.exp((H - 100.0) / 10.0)
          + 0.18 * (21.1 - T) * (1.0 - math.exp(-0.115 * H)))

    # Equilibrium moisture for wetting
    Ew = (0.618 * H**0.753
          + 10.0 * math.exp((H - 100.0) / 10.0)
          + 0.18 * (21.1 - T) * (1.0 - math.exp(-0.115 * H)))

    # Drying or wetting
    if mo > Ed:
        ko  = 0.424 * (1.0 - (H / 100.0)**1.7) + 0.0694 * W**0.5 * (1.0 - (H / 100.0)**8)
        kd  = ko * 0.581 * math.exp(0.0365 * T)
        m   = Ed + (mo - Ed) * 10.0**(-kd)
    elif mo < Ew:
        k1  = 0.424 * (1.0 - ((100.0 - H) / 100.0)**1.7) + \
              0.0694 * W**0.5 * (1.0 - ((100.0 - H) / 100.0)**8)
        kw  = k1 * 0.581 * math.exp(0.0365 * T)
        m   = Ew - (Ew - mo) * 10.0**(-kw)
    else:
        m = mo

    m = max(0.0, min(m, 250.0))
    return 59.5 * (250.0 - m) / (147.2 + m)


def _dmc(T: float, H: float, ro: float, Po: float, month: int) -> float:
    """
    Duff Moisture Code (DMC).

    Parameters
    ----------
    T     : Temperature at noon (C)
    H     : Relative humidity at noon (%)
    ro    : 24-hour accumulated precipitation (mm)
    Po    : Previous DMC value
    month : Calendar month (1-12)

    Returns
    -------
    New DMC value
    """
    # Rain effect
    Pr = Po
    if ro > 1.5:
        re = 0.92 * ro - 1.27
        Mo = 20.0 + math.exp(5.6348 - Po / 43.43)
        if Po <= 33.0:
            b = 100.0 / (0.5 + 0.3 * Po)
        elif Po <= 65.0:
            b = 14.0 - 1.3 * math.log(Po)
        else:
            b = 6.2 * math.log(Po) - 17.2
        Mr = Mo + 1000.0 * re / (48.77 + b * re)
        Pr = max(0.0, 244.72 - 43.43 * math.log(Mr - 20.0))

    # Drying effect
    K = 0.0
    if T >= -1.1:
        Le = LE[month]
        K  = 1.894 * (T + 1.1) * (100.0 - H) * Le * 1e-6

    return max(0.0, Pr + 100.0 * K)


def _dc(T: float, ro: float, Do: float, month: int) -> float:
    """
    Drought Code (DC).

    Parameters
    ----------
    T     : Temperature at noon (C)
    ro    : 24-hour accumulated precipitation (mm)
    Do    : Previous DC value
    month : Calendar month (1-12)

    Returns
    -------
    New DC value
    """
    # Rain effect
    Dr = Do
    if ro > 2.8:
        rd  = 0.83 * ro - 1.27
        Qo  = 800.0 * math.exp(-Do / 400.0)
        Qr  = Qo + 3.937 * rd
        Dr  = max(0.0, 400.0 * math.log(800.0 / Qr))

    # Drying effect (potential evapotranspiration)
    Lf = LF[month]
    Vo = 0.36 * (T + 2.8) + Lf if T >= -2.8 else 0.0

    return max(0.0, Dr + 0.5 * Vo)


def _isi(W: float, Fo: float) -> float:
    """Initial Spread Index (ISI)."""
    m  = 147.2 * (101.0 - Fo) / (59.5 + Fo)
    fw = math.exp(0.05039 * W)
    ff = 91.9 * math.exp(-0.1386 * m) * (1.0 + m**5.31 / 49_300_000.0)
    return 0.208 * fw * ff


def _bui(P: float, D: float) -> float:
    """Buildup Index (BUI)."""
    if P <= 0.4 * D:
        return 0.8 * P * D / (P + 0.4 * D)
    else:
        return P - (1.0 - 0.8 * D / (P + 0.4 * D)) * (0.92 + (0.0114 * P)**1.7)


def _fwi(R: float, U: float) -> float:
    """Fire Weather Index (FWI)."""
    if U > 80.0:
        fD = 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * U))
    else:
        fD = 0.626 * U**0.809 + 2.0
    B = 0.1 * R * fD
    if B > 1.0:
        return math.exp(2.72 * (0.434 * math.log(B))**0.647)
    return B


def compute_fwi_series(df: pd.DataFrame,
                       ffmc0: float = FFMC0,
                       dmc0:  float = DMC0,
                       dc0:   float = DC0) -> pd.DataFrame:
    """
    Compute the full Canadian FWI system for a weather time series.

    Input DataFrame must have columns:
      date       (date or datetime)
      temp_c     (noon temperature, Celsius)
      rh_pct     (noon relative humidity, %)
      wind_kmh   (noon wind speed, km/h)
      precip_mm  (24-h accumulated precipitation, mm)

    Returns DataFrame with all FWI components added:
      ffmc, dmc, dc, isi, bui, fwi, danger_class
    """
    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    out = []
    Fo, Po, Do = ffmc0, dmc0, dc0

    for _, row in df.iterrows():
        T   = float(row["temp_c"])
        H   = float(max(0.0, min(100.0, row["rh_pct"])))
        W   = float(max(0.0, row["wind_kmh"]))
        ro  = float(max(0.0, row["precip_mm"]))
        mon = row["date"].month

        Fo  = _ffmc(T, H, W, ro, Fo)
        Po  = _dmc(T, H, ro, Po, mon)
        Do  = _dc(T, ro, Do, mon)
        R   = _isi(W, Fo)
        U   = _bui(Po, Do)
        Fw  = _fwi(R, U)

        out.append({
            "date":         row["date"],
            "temp_c":       round(T, 1),
            "rh_pct":       round(H, 1),
            "wind_kmh":     round(W, 1),
            "precip_mm":    round(ro, 1),
            "ffmc":         round(Fo, 1),
            "dmc":          round(Po, 1),
            "dc":           round(Do, 1),
            "isi":          round(R, 1),
            "bui":          round(U, 1),
            "fwi":          round(Fw, 1),
            "danger_class": fwi_class(Fw),
        })

    return pd.DataFrame(out)


# =============================================================================
# SECTION 2: NASA POWER API — historical daily weather for Guna
# =============================================================================

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

def fetch_nasa_power(lat: float, lon: float,
                     start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily fire-weather variables from NASA POWER for a single point.

    Variables:
      T2M            - Temperature at 2m (C)
      RH2M           - Relative humidity at 2m (%)
      WS2M           - Wind speed at 2m (m/s)
      PRECTOTCORR    - Precipitation corrected (mm/day)

    Parameters
    ----------
    lat, lon : WGS84 coordinates
    start    : YYYYMMDD string
    end      : YYYYMMDD string

    Returns
    -------
    DataFrame with columns: date, temp_c, rh_pct, wind_kmh, precip_mm
    """
    params = {
        "parameters": "T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,PRECTOTCORR",
        "community":  "AG",          # Agriculture community uses noon values
        "longitude":  lon,
        "latitude":   lat,
        "start":      start,
        "end":        end,
        "format":     "JSON",
        "time-standard": "LST",      # local solar time
    }
    print(f"  Fetching NASA POWER {start} -> {end} ({lat}N, {lon}E) ...", flush=True)
    r = requests.get(NASA_POWER_URL, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()

    props = data["properties"]["parameter"]

    t2m     = props["T2M"]           # mean daily temp (C)
    rh2m    = props["RH2M"]          # mean daily RH (%)
    ws2m    = props["WS2M"]          # mean daily wind speed (m/s)
    precip  = props["PRECTOTCORR"]   # daily precip (mm)

    rows = []
    for date_str, temp in t2m.items():
        if len(date_str) != 8:
            continue
        d = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:]))
        rows.append({
            "date":      d,
            "temp_c":    float(t2m[date_str]),
            "rh_pct":    float(rh2m[date_str]),
            "wind_kmh":  float(ws2m[date_str]) * 3.6,    # m/s -> km/h
            "precip_mm": float(precip[date_str]),
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # NASA POWER uses -999 for missing — fill with previous day
    df = df.replace(-999.0, np.nan).replace(-999, np.nan)
    df["temp_c"]   = df["temp_c"].ffill().bfill()
    df["rh_pct"]   = df["rh_pct"].ffill().bfill()
    df["wind_kmh"] = df["wind_kmh"].ffill().bfill()
    df["precip_mm"]= df["precip_mm"].fillna(0.0)

    print(f"  NASA POWER: {len(df)} days loaded  "
          f"(T {df['temp_c'].min():.1f}-{df['temp_c'].max():.1f} C  "
          f"RH {df['rh_pct'].mean():.0f}% avg  "
          f"Wind {df['wind_kmh'].mean():.1f} km/h avg)", flush=True)
    return df


# =============================================================================
# SECTION 3: GFS NetCDF -> weather point extraction (NRT)
# =============================================================================

def extract_gfs_weather(gfs_dir: Path,
                        run_date: date,
                        lat: float = GUNA_LAT,
                        lon: float = GUNA_LON,
                        lead: int = 24) -> dict | None:
    """
    Extract fire-weather variables at (lat, lon) from a GFS NetCDF.

    Returns dict with keys: date, temp_c, rh_pct, wind_kmh, precip_mm
    Or None if the file doesn't exist.
    """
    try:
        import xarray as xr
    except ImportError:
        print("xarray not installed — GFS extraction skipped")
        return None

    nc_path = (gfs_dir
               / str(run_date.year)
               / f"{run_date.month:02d}"
               / f"{run_date.day:02d}"
               / f"gfs_f{lead:03d}.nc")

    if not nc_path.exists():
        return None

    ds = xr.open_dataset(nc_path)

    # Variable name variants from herbie/cfgrib
    def _get(candidates):
        for name in candidates:
            if name in ds:
                return ds[name]
        return None

    t_var   = _get(["t2m", "TMP_2maboveground", "temp_c", "T2M"])
    rh_var  = _get(["r2",  "RH_2maboveground",  "rh_pct", "RH2M"])
    u_var   = _get(["u10", "UGRD_10maboveground"])
    v_var   = _get(["v10", "VGRD_10maboveground"])
    pr_var  = _get(["tp",  "APCP_surface", "precip_mm"])

    if t_var is None or rh_var is None:
        return None

    # Select nearest grid point
    sel_kw = {}
    dim_lat = "latitude" if "latitude" in ds.dims else "lat"
    dim_lon = "longitude" if "longitude" in ds.dims else "lon"
    sel_kw[dim_lat] = lat
    sel_kw[dim_lon] = lon

    T_k  = float(t_var.sel(**sel_kw, method="nearest").values.flat[0])
    RH   = float(rh_var.sel(**sel_kw, method="nearest").values.flat[0])
    U    = float(u_var.sel(**sel_kw, method="nearest").values.flat[0]) if u_var is not None else 0.0
    V    = float(v_var.sel(**sel_kw, method="nearest").values.flat[0]) if v_var is not None else 0.0
    PR   = float(pr_var.sel(**sel_kw, method="nearest").values.flat[0]) if pr_var is not None else 0.0

    # Convert units
    T_c  = T_k - 273.15 if T_k > 100 else T_k     # K -> C (if Kelvin)
    WS   = math.sqrt(U**2 + V**2) * 3.6            # m/s -> km/h
    PR_mm = PR * 1000 if PR < 1.0 and PR >= 0 else PR  # m -> mm if needed

    # Validity date = run_date + lead hours (approximate)
    valid_date = run_date + timedelta(hours=lead)

    ds.close()
    return {
        "date":      valid_date.date() if hasattr(valid_date, "date") else valid_date,
        "temp_c":    round(T_c, 1),
        "rh_pct":    round(max(0.0, min(100.0, RH)), 1),
        "wind_kmh":  round(max(0.0, WS), 1),
        "precip_mm": round(max(0.0, PR_mm), 1),
        "source":    f"GFS f{lead:03d}",
    }


# =============================================================================
# SECTION 4: Main pipeline
# =============================================================================

def run(start_year: int = 2013,
        end_year: int   = 2025,
        update_nrt: bool = False,
        plot: bool       = False):
    """
    Full FWI pipeline:
      1. Fetch NASA POWER historical data (start_year -> end_year)
      2. Append GFS NRT forecasts (if update_nrt)
      3. Compute Canadian FWI time series
      4. Save to data_lake/fire_weather/
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_path = OUT_DIR / "fwi_daily.parquet"

    # ── Step 1: Load or fetch NASA POWER ──────────────────────────────────────
    print("=" * 65)
    print("CANADIAN FWI COMPUTATION — Guna Division, MP")
    print("=" * 65)

    start_str = f"{start_year}0101"
    end_str   = f"{end_year}1231"

    nasa_cache = OUT_DIR / f"nasa_power_{start_year}_{end_year}.parquet"

    if nasa_cache.exists():
        print(f"\n[1] Loading cached NASA POWER data from {nasa_cache.name}")
        wx = pd.read_parquet(nasa_cache)
        wx["date"] = pd.to_datetime(wx["date"]).dt.date
    else:
        print(f"\n[1] Fetching NASA POWER data ({start_year}-{end_year})...")
        # Fetch in 5-year chunks to avoid API timeouts
        chunks = []
        yr = start_year
        while yr <= end_year:
            yr_end = min(yr + 4, end_year)
            chunk  = fetch_nasa_power(
                GUNA_LAT, GUNA_LON,
                f"{yr}0101", f"{yr_end}1231"
            )
            chunks.append(chunk)
            yr += 5
        wx = pd.concat(chunks, ignore_index=True).sort_values("date")
        wx.to_parquet(nasa_cache, index=False)
        print(f"  Cached to {nasa_cache.name}")

    print(f"  Weather rows: {len(wx):,}  "
          f"({wx['date'].min()} to {wx['date'].max()})")

    # ── Step 2: Append GFS NRT ────────────────────────────────────────────────
    if update_nrt:
        print("\n[2] Appending GFS NRT forecasts...")
        nrt_rows = []
        today = date.today()
        for d_offset in range(-1, 4):    # yesterday + 3-day forecast
            run_dt = today + timedelta(days=d_offset)
            for lead in [24, 48, 72]:
                row = extract_gfs_weather(GFS_DIR, run_dt, lead=lead)
                if row:
                    nrt_rows.append(row)
        if nrt_rows:
            nrt_df = pd.DataFrame(nrt_rows)
            nrt_df["date"] = pd.to_datetime(nrt_df["date"]).dt.date
            # Deduplicate — keep NRT over NASA POWER for overlapping dates
            wx = wx[~wx["date"].isin(nrt_df["date"])]
            wx = pd.concat([wx, nrt_df[["date","temp_c","rh_pct","wind_kmh","precip_mm"]]],
                           ignore_index=True).sort_values("date")
            print(f"  Appended {len(nrt_rows)} GFS NRT rows "
                  f"(series now to {wx['date'].max()})")
        else:
            print("  No GFS NRT files found.")

    # ── Step 3: Compute Canadian FWI ─────────────────────────────────────────
    print("\n[3] Computing Canadian FWI system (Van Wagner 1987)...")

    # Reset codes at the start of each calendar year (overwintering)
    # Process year-by-year, carrying codes forward within each year
    # but resetting if there is a long gap (>90 days of missing data)
    years = sorted(wx["date"].apply(lambda d: d.year).unique())
    all_fwi = []

    Fo, Po, Do = FFMC0, DMC0, DC0
    for yr in years:
        yr_df = wx[wx["date"].apply(lambda d: d.year) == yr].copy()
        if yr_df.empty:
            continue
        # Reset codes at season start (March 1) — standard Canadian practice
        # Carry codes from previous year's last day (no reset) for continuity
        fwi_yr = compute_fwi_series(yr_df, ffmc0=Fo, dmc0=Po, dc0=Do)
        if not fwi_yr.empty:
            Fo = float(fwi_yr["ffmc"].iloc[-1])
            Po = float(fwi_yr["dmc"].iloc[-1])
            Do = float(fwi_yr["dc"].iloc[-1])
            all_fwi.append(fwi_yr)

    fwi_df = pd.concat(all_fwi, ignore_index=True).sort_values("date")

    # ── Step 4: Save ─────────────────────────────────────────────────────────
    print("\n[4] Saving outputs...")
    parquet_path = OUT_DIR / "fwi_daily.parquet"
    csv_path     = OUT_DIR / "fwi_daily.csv"
    fwi_df.to_parquet(parquet_path, index=False)
    fwi_df.to_csv(csv_path, index=False)
    print(f"  Saved: {parquet_path}  ({len(fwi_df):,} rows)")
    print(f"  Saved: {csv_path}")

    # ── Step 5: Summary ───────────────────────────────────────────────────────
    _print_summary(fwi_df)

    # ── Step 6: Optional plot ─────────────────────────────────────────────────
    if plot:
        _plot_fwi(fwi_df)

    return fwi_df


def _print_summary(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("FWI SUMMARY — Guna Division, MP")
    print("=" * 65)

    fire_season = df[df["date"].apply(lambda d: 3 <= d.month <= 6)]

    print(f"\n  Date range : {df['date'].min()} to {df['date'].max()} "
          f"({len(df):,} days)")
    print(f"\n  FWI stats (ALL days):")
    print(f"    Mean  : {df['fwi'].mean():.1f}")
    print(f"    Max   : {df['fwi'].max():.1f}  on {df.loc[df['fwi'].idxmax(), 'date']}")
    print(f"    Days >= 21.3 (Very High/Extreme): "
          f"{(df['fwi'] >= 21.3).sum()} "
          f"({100*(df['fwi']>=21.3).mean():.1f}%)")

    print(f"\n  Fire season (Mar-Jun) FWI stats:")
    if not fire_season.empty:
        print(f"    Mean  : {fire_season['fwi'].mean():.1f}")
        print(f"    Max   : {fire_season['fwi'].max():.1f}")
        print(f"    Days >= 21.3 (Very High+): "
              f"{(fire_season['fwi'] >= 21.3).sum()}")

    print(f"\n  Danger class breakdown (all days):")
    for cls, cnt in df["danger_class"].value_counts().items():
        pct = 100 * cnt / len(df)
        bar = "#" * int(pct / 2)
        print(f"    {cls:<12}: {cnt:5d} days ({pct:5.1f}%)  {bar}")

    print(f"\n  Annual peak FWI (fire season):")
    for yr in sorted(df["date"].apply(lambda d: d.year).unique()):
        yr_df = df[df["date"].apply(lambda d: d.year == yr and 3 <= d.month <= 6)]
        if yr_df.empty:
            continue
        pk = yr_df.loc[yr_df["fwi"].idxmax()]
        print(f"    {yr}: peak FWI={pk['fwi']:.1f} ({pk['danger_class']:10s}) "
              f"on {pk['date']}  "
              f"T={pk['temp_c']}C  RH={pk['rh_pct']}%  "
              f"W={pk['wind_kmh']}km/h")


def _plot_fwi(df: pd.DataFrame):
    """Save a FWI time-series PNG to data_lake/fire_weather/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Patch

        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)

        # ── Plot 1: Annual peak FWI bar chart ─────────────────────────────
        ax = axes[0]
        fire_season = df[df["date"].apply(lambda d: 3 <= d.month <= 6)].copy()
        fire_season["year"] = fire_season["date"].apply(lambda d: d.year)
        annual_peak = fire_season.groupby("year")["fwi"].max()

        colors = ["#d73027" if v >= 38 else
                  "#f46d43" if v >= 21.3 else
                  "#fdae61" if v >= 11.2 else
                  "#abd9e9" if v >= 5 else
                  "#4575b4"
                  for v in annual_peak.values]

        ax.bar(annual_peak.index, annual_peak.values, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(38,   color="#d73027", linestyle="--", linewidth=0.8, label="Extreme (38)")
        ax.axhline(21.3, color="#f46d43", linestyle="--", linewidth=0.8, label="Very High (21.3)")
        ax.axhline(11.2, color="#fdae61", linestyle="--", linewidth=0.8, label="High (11.2)")
        ax.set_title("Annual Peak FWI (Fire Season: Mar-Jun) — Guna Division, MP",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Peak FWI")
        ax.legend(fontsize=8)
        ax.set_xticks(annual_peak.index)
        ax.set_xticklabels(annual_peak.index, rotation=45)

        # ── Plot 2: Monthly FWI heatmap (year x month) ──────────────────
        ax = axes[1]
        df2 = df.copy()
        df2["year"]  = df2["date"].apply(lambda d: d.year)
        df2["month"] = df2["date"].apply(lambda d: d.month)
        pivot = df2.groupby(["year", "month"])["fwi"].mean().unstack(fill_value=0)

        years  = sorted(pivot.index)
        months = list(range(1, 13))
        data   = np.array([[pivot.loc[y, m] if m in pivot.columns else 0
                            for m in months] for y in years])

        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=50,
                       origin="upper")
        ax.set_xticks(range(12))
        ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                            "Jul","Aug","Sep","Oct","Nov","Dec"])
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)
        ax.set_title("Mean Monthly FWI Heatmap (Year x Month)", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Mean FWI")

        # ── Plot 3: Daily FWI time series (most recent 2 years) ─────────
        ax = axes[2]
        recent = df[df["date"].apply(lambda d: d.year >= df["date"].max().year - 1)].copy()
        dates  = pd.to_datetime(recent["date"])
        fwi    = recent["fwi"].values

        ax.fill_between(dates, fwi, alpha=0.4, color="#e34a33")
        ax.plot(dates, fwi, linewidth=0.7, color="#e34a33")
        ax.axhline(38,   color="#d73027", linestyle="--", linewidth=0.8)
        ax.axhline(21.3, color="#f46d43", linestyle="--", linewidth=0.8)
        ax.axhline(11.2, color="#fdae61", linestyle="--", linewidth=0.8)
        ax.set_title(f"Daily FWI — {recent['date'].min()} to {recent['date'].max()}", fontsize=11)
        ax.set_ylabel("FWI")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plot_path = OUT_DIR / "fwi_timeseries.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Plot saved: {plot_path}")

    except ImportError:
        print("  matplotlib not available — skipping plot")


# =============================================================================
# SECTION 5: CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Canadian FWI computation for Guna Division (NASA POWER + GFS)"
    )
    p.add_argument("--start",      type=int, default=2013,
                   help="Start year for NASA POWER historical data (default: 2013)")
    p.add_argument("--end",        type=int, default=2025,
                   help="End year for NASA POWER historical data (default: 2025)")
    p.add_argument("--update-nrt", action="store_true",
                   help="Append GFS NRT forecast weather to the series")
    p.add_argument("--plot",       action="store_true",
                   help="Save FWI time-series PNG to data_lake/fire_weather/")
    p.add_argument("--no-cache",   action="store_true",
                   help="Force re-download NASA POWER even if cache exists")
    args = p.parse_args()

    if args.no_cache:
        for f in OUT_DIR.glob("nasa_power_*.parquet"):
            f.unlink()
            print(f"Removed cache: {f.name}")

    run(start_year=args.start,
        end_year=args.end,
        update_nrt=args.update_nrt,
        plot=args.plot)


if __name__ == "__main__":
    main()
