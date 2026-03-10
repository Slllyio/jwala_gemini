"""
Van Suraksha NRT Alert Pipeline
================================
Pulls today's FIRMS VIIRS near-real-time fire detections, cross-references
with today's computed Fire Weather Index (FWI), and generates structured
alert bulletins for Guna Division forest fire response.

Alert severity is scored on TWO axes:
  FRP axis (fire intensity):
    >= 50 MW  -> INFERNO
    >= 20 MW  -> INTENSE
    >= 5 MW   -> ACTIVE
    >= 1 MW   -> MINOR
    <  1 MW   -> TRACE

  FWI axis (fire weather danger):
    >= 38     -> Extreme
    >= 21.3   -> Very High
    >= 11.2   -> High
    >= 5      -> Moderate
    <  5      -> Low

  Combined severity (for response priority):
    CRITICAL  : FWI >= 38  AND FRP >= 5 MW  (extreme weather + active fire)
    HIGH      : FWI >= 21.3 OR FRP >= 20 MW
    MODERATE  : FWI >= 11.2 OR FRP >= 5 MW
    LOW       : anything else with a detection
    WATCH     : no active fire but FWI >= 21.3 (fire weather warning)

Data sources:
  FIRMS NRT  : data_lake/fire_detections/firms_viirs_nrt/      (daily pull)
  FIRMS hist : data_lake/fire_detections/firms_viirs_snpp_sp/  (annual parquet)
  FWI        : data_lake/fire_weather/fwi_daily.parquet

Output:
  data_lake/alerts/<YYYY-MM-DD>_alert.json   <- structured alert bulletin
  data_lake/alerts/<YYYY-MM-DD>_alert.csv    <- tabular fire list
  data_lake/alerts/latest_alert.json         <- symlink / copy of today's alert

Usage:
  python scripts/nrt_alert_pipeline.py                  # today
  python scripts/nrt_alert_pipeline.py --date 2024-05-28
  python scripts/nrt_alert_pipeline.py --days 7         # last 7 days
  python scripts/nrt_alert_pipeline.py --date 2024-05-28 --fetch-nrt
"""

# ── PROJ/GDAL env fix ──
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()
del _os, _sys
# ────────────────────────

import argparse
import json
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parent.parent
FIRMS_NRT        = ROOT / "data_lake" / "fire_detections" / "firms_viirs_nrt"
FIRMS_SP         = ROOT / "data_lake" / "fire_detections" / "firms_viirs_snpp_sp"
FWI_PATH         = ROOT / "data_lake" / "fire_weather" / "fwi_daily.parquet"
ALERT_DIR        = ROOT / "data_lake" / "alerts"
FOREST_MASK_30M  = ROOT / "data_lake" / "land_cover" / "forest_mask_guna_30m.tif"
RAW_LC_PATH      = ROOT / "data_lake" / "land_cover" / "worldcover_guna_raw.tif"

# ESA WorldCover class names (key = pixel value)
_LC_NAMES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland",
    40: "Cropland",   50: "Built-up",  60: "Bare/sparse veg",
    80: "Water",      90: "Wetland",
}

# ── Guna Division bounding box (WGS84) ────────────────────────────────────────
GUNA_BBOX = dict(lon_min=76.45, lat_min=23.80, lon_max=77.85, lat_max=24.95)

# ── FIRMS NRT API (NASA FIRMS — requires MAP_KEY env var or hardcoded) ────────
FIRMS_NRT_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_NRT/{lon_min},{lat_min},{lon_max},{lat_max}/{days}"

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")


# =============================================================================
# SECTION 1: Fire severity scoring
# =============================================================================

def frp_category(frp_mw: float) -> str:
    if frp_mw >= 50:  return "INFERNO"
    if frp_mw >= 20:  return "INTENSE"
    if frp_mw >= 5:   return "ACTIVE"
    if frp_mw >= 1:   return "MINOR"
    return "TRACE"


def combined_severity(fwi: float, frp_mw: float) -> str:
    if fwi >= 38 and frp_mw >= 5:   return "CRITICAL"
    if fwi >= 21.3 or frp_mw >= 20: return "HIGH"
    if fwi >= 11.2 or frp_mw >= 5:  return "MODERATE"
    return "LOW"


def severity_emoji(sev: str) -> str:
    return {"CRITICAL": "[CRITICAL]", "HIGH": "[HIGH]", "MODERATE": "[MODERATE]",
            "LOW": "[LOW]", "WATCH": "[WATCH]"}.get(sev, "[?]")


def fwi_danger(fwi: float) -> str:
    if fwi >= 38:   return "Extreme"
    if fwi >= 21.3: return "Very High"
    if fwi >= 11.2: return "High"
    if fwi >= 5:    return "Moderate"
    return "Low"


# =============================================================================
# SECTION 1b: Forest / land cover check
# =============================================================================

# Lazy-loaded rasterio datasets (opened once, reused for all fires in a run)
_forest_src = None
_lc_src     = None


def _open_masks():
    """Open forest + raw land-cover rasters once per pipeline run."""
    global _forest_src, _lc_src
    try:
        import rasterio
        if FOREST_MASK_30M.exists() and _forest_src is None:
            _forest_src = rasterio.open(FOREST_MASK_30M)
        if RAW_LC_PATH.exists() and _lc_src is None:
            _lc_src = rasterio.open(RAW_LC_PATH)
    except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")


def check_fire_land_cover(lat: float, lon: float) -> dict:
    """
    Determine land cover type at a FIRMS fire detection point using
    ESA WorldCover 2021 (30 m forest mask + raw 10 m land cover).

    Returns
    -------
    dict with keys:
      in_forest       bool   fire pixel is forest (class 10)
      near_forest     bool   within 1 pixel (~30 m) of forest
      is_agricultural bool   fire pixel is cropland (class 40)
      land_cover      str    ESA class label at the fire point
      lc_code         int    ESA class code (10=forest, 40=crop, ...)
    """
    default = {"in_forest": None, "near_forest": None,
               "is_agricultural": None, "land_cover": "unknown", "lc_code": -1}

    _open_masks()
    if _forest_src is None:
        return default

    try:
        from rasterio.transform import rowcol

        # Forest mask check
        row, col = rowcol(_forest_src.transform, lon, lat)
        H, W = _forest_src.shape
        if not (0 <= row < H and 0 <= col < W):
            return {**default, "land_cover": "out_of_bounds"}

        forest_val = int(_forest_src.read(1)[row, col])
        in_forest  = (forest_val == 1)

        # Buffer check (1 pixel = ~30 m)
        r0, r1 = max(0, row - 1), min(H, row + 2)
        c0, c1 = max(0, col - 1), min(W, col + 2)
        near_forest = bool(np.any(_forest_src.read(1)[r0:r1, c0:c1] == 1))

        # Raw land cover class
        lc_code, lc_name = -1, "unknown"
        if _lc_src is not None:
            lc_row, lc_col = rowcol(_lc_src.transform, lon, lat)
            lH, lW = _lc_src.shape
            if 0 <= lc_row < lH and 0 <= lc_col < lW:
                lc_code = int(_lc_src.read(1)[lc_row, lc_col])
                lc_name = _LC_NAMES.get(lc_code, f"class_{lc_code}")

        return {
            "in_forest":       in_forest,
            "near_forest":     near_forest,
            "is_agricultural": (lc_code == 40),
            "land_cover":      lc_name,
            "lc_code":         lc_code,
        }
    except Exception:
        return default


# =============================================================================
# SECTION 2: Load fire detections
# =============================================================================

def _acq_time_to_ist(acq_time_int: int) -> str:
    """Convert FIRMS acq_time integer (HHMM UTC) -> 'HH:MM IST'."""
    h = int(acq_time_int) // 100
    m = int(acq_time_int) % 100
    tot = h * 60 + m + 330   # +5:30 for IST
    return f"{(tot // 60) % 24:02d}:{tot % 60:02d}"


def load_firms_for_date(target_date: date) -> pd.DataFrame:
    """
    Load FIRMS VIIRS detections for target_date from:
      1. NRT CSV files (if available)
      2. Annual parquet (historical fallback)

    Returns DataFrame filtered to Guna bbox.
    """
    frames = []

    # ── Try NRT CSV files first ────────────────────────────────────────────
    if FIRMS_NRT.exists():
        for csv_f in sorted(FIRMS_NRT.glob("*.csv")):
            try:
                df = pd.read_csv(csv_f)
                if df.empty:
                    continue
                df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date
                day_df = df[df["acq_date"] == target_date]
                if not day_df.empty:
                    frames.append(day_df)
            except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")

    # ── Fall back to annual parquet ────────────────────────────────────────
    if not frames and FIRMS_SP.exists():
        yr = target_date.year
        pq = FIRMS_SP / f"viirs_snpp_sp_{yr}_annual.parquet"
        if pq.exists():
            df = pd.read_parquet(pq)
            df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date
            day_df = df[df["acq_date"] == target_date]
            if not day_df.empty:
                frames.append(day_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Spatial filter: Guna bbox
    mask = (
        (combined["latitude"]  >= GUNA_BBOX["lat_min"]) &
        (combined["latitude"]  <= GUNA_BBOX["lat_max"]) &
        (combined["longitude"] >= GUNA_BBOX["lon_min"]) &
        (combined["longitude"] <= GUNA_BBOX["lon_max"])
    )
    return combined[mask].copy()


def fetch_nrt_from_api(target_date: date, api_key: str,
                       days_back: int = 1) -> pd.DataFrame:
    """
    Fetch fresh NRT data from NASA FIRMS API.
    Saves CSV to data_lake/fire_detections/firms_viirs_nrt/
    """
    FIRMS_NRT.mkdir(parents=True, exist_ok=True)
    url = FIRMS_NRT_URL.format(
        key=api_key,
        days=days_back,
        **GUNA_BBOX,
    )
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        csv_path = FIRMS_NRT / f"viirs_nrt_{target_date}.csv"
        csv_path.write_text(r.text, encoding="utf-8")
        df = pd.read_csv(csv_path)
        print(f"  Fetched {len(df)} NRT detections from FIRMS API")
        return df
    except Exception as e:
        print(f"  WARNING: FIRMS API fetch failed: {e}")
        return pd.DataFrame()


# =============================================================================
# SECTION 3: Load FWI for a date
# =============================================================================

def load_fwi(target_date: date) -> dict:
    """
    Return FWI row for target_date (or nearest available day).
    """
    default = {"fwi": np.nan, "danger_class": "Unknown",
               "ffmc": np.nan, "dmc": np.nan, "dc": np.nan,
               "isi": np.nan, "bui": np.nan,
               "temp_c": np.nan, "rh_pct": np.nan,
               "wind_kmh": np.nan, "precip_mm": np.nan}

    if not FWI_PATH.exists():
        return default

    fwi_df = pd.read_parquet(FWI_PATH)
    fwi_df["date"] = pd.to_datetime(fwi_df["date"]).dt.date

    row = fwi_df[fwi_df["date"] == target_date]
    if row.empty:
        # Use nearest date
        fwi_df["delta"] = fwi_df["date"].apply(
            lambda d: abs((d - target_date).days))
        row = fwi_df.nsmallest(1, "delta")
    if row.empty:
        return default

    r = row.iloc[0]
    return {k: (None if pd.isna(v) else v)
            for k, v in r.items() if k != "delta"}


# =============================================================================
# SECTION 4: Build alert bulletin
# =============================================================================

def build_alert(target_date: date,
                fires: pd.DataFrame,
                fwi_row: dict) -> dict:
    """
    Assemble a structured alert bulletin dict.
    """
    now_ist = datetime.now(IST).isoformat()
    fwi_val = float(fwi_row.get("fwi") or 0)
    danger  = fwi_row.get("danger_class") or fwi_danger(fwi_val)

    fire_alerts = []
    for _, row in fires.iterrows():
        frp = float(row.get("frp", 0) or 0)
        lat = float(row.get("latitude", 0))
        lon = float(row.get("longitude", 0))
        conf= str(row.get("confidence", "n"))
        dn  = str(row.get("daynight", "D"))
        t_raw = int(row.get("acq_time", 0) or 0)
        t_ist = _acq_time_to_ist(t_raw)

        # Land cover check — forest vs agriculture
        lc   = check_fire_land_cover(lat, lon)
        in_forest    = lc.get("in_forest")
        near_forest  = lc.get("near_forest")
        is_agri      = lc.get("is_agricultural")
        lc_name      = lc.get("land_cover", "unknown")

        sev = combined_severity(fwi_val, frp)
        frp_cat  = frp_category(frp)
        dn_label = "daytime" if dn == "D" else "nighttime"
        conf_label = {"h": "high", "n": "nominal", "l": "low"}.get(conf, conf)

        # Downgrade CRITICAL->HIGH if definitely NOT near any forest
        # (agricultural fires still warrant alert, but lower forest priority)
        if sev == "CRITICAL" and in_forest is False and near_forest is False:
            sev = "HIGH"

        # Land cover tag for message
        if in_forest:
            lc_tag = "[FOREST FIRE]"
        elif near_forest:
            lc_tag = "[NEAR FOREST]"
        elif is_agri:
            lc_tag = "[AGRICULTURAL]"
        else:
            lc_tag = f"[{lc_name.upper()}]" if lc_name != "unknown" else ""

        # Human-readable message
        if sev == "CRITICAL":
            msg = (f"CRITICAL {lc_tag}: {frp_cat} {dn_label} FOREST fire at "
                   f"{lat:.3f}N {lon:.3f}E — "
                   f"FRP={frp:.1f} MW, FWI={fwi_val:.1f} ({danger}). "
                   f"Immediate ground response required.")
        elif sev == "HIGH":
            if in_forest or near_forest:
                msg = (f"HIGH {lc_tag}: {frp_cat} {dn_label} fire at "
                       f"{lat:.3f}N {lon:.3f}E — "
                       f"FRP={frp:.1f} MW, FWI={fwi_val:.1f} ({danger}). "
                       f"Deploy rapid response team.")
            else:
                msg = (f"HIGH {lc_tag}: {dn_label} fire at "
                       f"{lat:.3f}N {lon:.3f}E — "
                       f"FRP={frp:.1f} MW, FWI={fwi_val:.1f} ({danger}). "
                       f"Non-forest — monitor for spread to nearby vegetation.")
        elif sev == "MODERATE":
            msg = (f"MODERATE {lc_tag}: {dn_label} fire at "
                   f"{lat:.3f}N {lon:.3f}E — "
                   f"FRP={frp:.1f} MW ({lc_name}). Monitor closely.")
        else:
            msg = (f"LOW {lc_tag}: {dn_label} fire at "
                   f"{lat:.3f}N {lon:.3f}E — "
                   f"FRP={frp:.1f} MW ({lc_name}). Routine monitoring.")

        fire_alerts.append({
            "severity":         sev,
            "fire_lat":         round(lat, 4),
            "fire_lon":         round(lon, 4),
            "fire_date":        str(row.get("acq_date", target_date)),
            "fire_time_ist":    t_ist,
            "frp_mw":           round(frp, 2),
            "frp_category":     frp_cat,
            "viirs_confidence": conf_label,
            "daynight":         dn_label,
            "fwi":              round(fwi_val, 1),
            "fwi_danger":       danger,
            "land_cover":       lc_name,
            "in_forest":        in_forest,
            "near_forest":      near_forest,
            "is_agricultural":  is_agri,
            "message":          msg,
        })

    # Sort by severity then FRP descending
    sev_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
    fire_alerts.sort(key=lambda x: (sev_order.get(x["severity"], 9),
                                    -x["frp_mw"]))

    # Overall bulletin severity
    if fire_alerts:
        bulletin_sev = fire_alerts[0]["severity"]
    elif fwi_val >= 21.3:
        bulletin_sev = "WATCH"
    else:
        bulletin_sev = "CLEAR"

    n_critical  = sum(1 for f in fire_alerts if f["severity"] == "CRITICAL")
    n_high      = sum(1 for f in fire_alerts if f["severity"] == "HIGH")
    n_forest    = sum(1 for f in fire_alerts if f.get("in_forest") or f.get("near_forest"))
    n_agri      = sum(1 for f in fire_alerts if f.get("is_agricultural") and
                      not f.get("in_forest") and not f.get("near_forest"))
    peak_frp    = max((f["frp_mw"] for f in fire_alerts), default=0.0)
    forest_mask_available = FOREST_MASK_30M.exists()

    return {
        "alert_id":        f"GUNA-{target_date}-{len(fire_alerts):03d}fires",
        "generated_at":    now_ist,
        "target_date":     str(target_date),
        "bulletin_severity": bulletin_sev,
        "fires_detected":  len(fire_alerts),
        "n_critical":      n_critical,
        "n_high":          n_high,
        "n_forest_fires":  n_forest,
        "n_agri_fires":    n_agri,
        "forest_mask_used": forest_mask_available,
        "peak_frp_mw":     round(peak_frp, 2),
        "fwi": {
            "value":       round(fwi_val, 1),
            "danger_class": danger,
            "ffmc":        fwi_row.get("ffmc"),
            "dmc":         fwi_row.get("dmc"),
            "dc":          fwi_row.get("dc"),
            "isi":         fwi_row.get("isi"),
            "bui":         fwi_row.get("bui"),
            "temp_c":      fwi_row.get("temp_c"),
            "rh_pct":      fwi_row.get("rh_pct"),
            "wind_kmh":    fwi_row.get("wind_kmh"),
        },
        "fires":           fire_alerts,
        "area": {
            "name":        "Guna Division, Madhya Pradesh",
            "bbox":        GUNA_BBOX,
        },
        "data_sources":    ["FIRMS VIIRS SNPP", "NASA POWER / GFS FWI"],
    }


# =============================================================================
# SECTION 5: Print bulletin to console
# =============================================================================

def print_bulletin(alert: dict):
    sev   = alert["bulletin_severity"]
    emoji = severity_emoji(sev)
    fwi   = alert["fwi"]

    print("\n" + "=" * 65)
    print(f"  VAN SURAKSHA FIRE ALERT  {emoji}  {sev}")
    print(f"  Guna Division, MP  |  {alert['target_date']}")
    print("=" * 65)
    print(f"\n  Fire Weather Index : {fwi['value']}  [{fwi['danger_class']}]")
    if fwi.get("temp_c") is not None:
        print(f"  Conditions         : T={fwi['temp_c']}C  "
              f"RH={fwi['rh_pct']}%  Wind={fwi['wind_kmh']} km/h")
    print(f"  FWI components     : "
          f"FFMC={fwi['ffmc']}  DMC={fwi['dmc']}  DC={fwi['dc']}  "
          f"ISI={fwi['isi']}  BUI={fwi['bui']}")

    fires = alert["fires"]
    print(f"\n  Active fires       : {len(fires)}")
    if fires:
        print(f"  CRITICAL / HIGH    : {alert['n_critical']} / {alert['n_high']}")
        if alert.get("forest_mask_used"):
            print(f"  Forest / Agri fires: {alert['n_forest_fires']} / {alert['n_agri_fires']}")
        print(f"  Peak FRP           : {alert['peak_frp_mw']} MW")
        print()
        print(f"  {'SEV':<10} {'LAT':>7} {'LON':>8} {'FRP':>7}  {'TIME IST':<10} {'LAND COVER':<16} {'MESSAGE'}")
        print("  " + "-" * 105)
        for f in fires:
            lc_disp = f.get("land_cover", "")[:14]
            print(f"  {f['severity']:<10} "
                  f"{f['fire_lat']:>7.3f} {f['fire_lon']:>8.3f} "
                  f"{f['frp_mw']:>6.1f}MW  "
                  f"{f['fire_time_ist']:<10} "
                  f"{lc_disp:<16} "
                  f"{f['message'][:55]}")
    else:
        if sev == "WATCH":
            print(f"\n  No active fires detected.")
            print(f"  Fire Weather Warning: FWI={fwi['value']} ({fwi['danger_class']}).")
            print(f"  Conditions are dangerous — maintain heightened patrol.")
        else:
            print(f"\n  No fires detected. Conditions normal.")

    print("\n" + "=" * 65)


# =============================================================================
# SECTION 6: Save alert
# =============================================================================

def save_alert(alert: dict, target_date: date):
    ALERT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = ALERT_DIR / f"{target_date}_alert.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(alert, f, indent=2, default=str)

    # CSV (fire rows only)
    if alert["fires"]:
        fire_df = pd.DataFrame(alert["fires"])
        csv_path = ALERT_DIR / f"{target_date}_alert.csv"
        fire_df.to_csv(csv_path, index=False)

    # latest_alert.json (always overwrite with most recent)
    latest_path = ALERT_DIR / "latest_alert.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(alert, f, indent=2, default=str)

    print(f"\n  Saved: {json_path.name}")
    if alert["fires"]:
        print(f"  Saved: {target_date}_alert.csv")
    print(f"  Updated: latest_alert.json")


# =============================================================================
# SECTION 7: Main
# =============================================================================

def run_for_date(target_date: date,
                 fetch_nrt: bool = False,
                 api_key: str = None) -> dict:
    """Run the NRT alert pipeline for one date."""
    print(f"\n[Van Suraksha] Processing {target_date}...")

    # Optionally refresh NRT from API
    if fetch_nrt and api_key:
        fetch_nrt_from_api(target_date, api_key)

    # Load fires
    fires = load_firms_for_date(target_date)
    print(f"  Fires in Guna bbox: {len(fires)}")

    # Load FWI
    fwi_row = load_fwi(target_date)
    fwi_val = fwi_row.get("fwi") or 0
    print(f"  FWI: {fwi_val:.1f} ({fwi_row.get('danger_class', 'Unknown')})")

    # Build + print + save
    alert = build_alert(target_date, fires, fwi_row)
    print_bulletin(alert)
    save_alert(alert, target_date)
    return alert


def run_multi_day(target_date: date, n_days: int, **kwargs) -> list:
    """Run alert pipeline for the last n_days ending at target_date."""
    alerts = []
    for i in range(n_days - 1, -1, -1):
        d = target_date - timedelta(days=i)
        alerts.append(run_for_date(d, **kwargs))
    return alerts


def _high_fwi_days_report(n: int = 20):
    """Print the N highest-FWI days in the historical record."""
    if not FWI_PATH.exists():
        print("FWI data not found — run compute_fwi.py first.")
        return
    df = pd.read_parquet(FWI_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    top = df.nlargest(n, "fwi")[["date","temp_c","rh_pct","wind_kmh","fwi","danger_class"]]
    print(f"\nTop {n} highest FWI days — Guna Division:")
    print(top.to_string(index=False))


def main():
    p = argparse.ArgumentParser(
        description="Van Suraksha NRT fire alert pipeline for Guna Division"
    )
    p.add_argument("--date", default=None,
                   help="Target date ISO (default: today)")
    p.add_argument("--days", type=int, default=1,
                   help="Number of days to process (default: 1 = today only)")
    p.add_argument("--fetch-nrt", action="store_true",
                   help="Fetch fresh NRT data from FIRMS API before processing")
    p.add_argument("--api-key", default=None,
                   help="NASA FIRMS MAP_KEY for NRT fetch (or set FIRMS_MAP_KEY env var)")
    p.add_argument("--top-fwi", action="store_true",
                   help="Print top 20 highest FWI days and exit")
    args = p.parse_args()

    if args.top_fwi:
        _high_fwi_days_report()
        return

    import os
    api_key = args.api_key or os.environ.get("FIRMS_MAP_KEY")

    target = (date.fromisoformat(args.date) if args.date else date.today())

    if args.days == 1:
        run_for_date(target,
                     fetch_nrt=args.fetch_nrt,
                     api_key=api_key)
    else:
        run_multi_day(target, args.days,
                      fetch_nrt=args.fetch_nrt,
                      api_key=api_key)


if __name__ == "__main__":
    main()
