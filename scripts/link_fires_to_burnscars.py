"""
Pixel-Level Fire -> Burn Scar Causal Linker
===========================================
For every Prithvi burn scar mask, finds the FIRMS VIIRS fire detection(s)
that SPATIALLY OVERLAP the actual burned pixels, giving true causal evidence
rather than just tile-level proximity.

How it works:
  1. Load burn scar mask (uint8 GeoTIFF, 0=unburned, 1=burned)
  2. Convert each FIRMS fire point (lat/lon WGS84) -> pixel (row, col) in
     the mask's UTM 43N grid using the affine transform
  3. If mask[row, col] == 1  ->  SPATIAL HIT (fire burned that exact pixel)
  4. If no direct hit, measure distance from fire point to nearest burned pixel
  5. Assign confidence:
       HIGH   -> spatial hit + gap <= 10 days
       MEDIUM -> spatial hit + gap <= 30 days, OR near-hit (<500m) + gap<=10d
       LOW    -> tile-level only, or long gap
       NONE   -> no FIRMS detection in window

Output:
  data_lake/burn_scars/fire_links.parquet   ← one row per (scene, best_fire)
  data_lake/burn_scars/fire_links.csv       ← same, human-readable

Columns:
  scene_date, tile, burn_area_px, burn_area_ha, burn_pct
  fire_date, fire_time_utc, fire_time_ist
  fire_lat, fire_lon, fire_frp_mw, viirs_confidence, daynight
  spatial_hit, dist_to_burn_m, days_gap
  n_firms_in_window, n_spatial_hits
  confidence  (HIGH / MEDIUM / LOW / NONE)

Usage:
  python scripts/link_fires_to_burnscars.py
  python scripts/link_fires_to_burnscars.py --window 45   # days to look back
  python scripts/link_fires_to_burnscars.py --csv-only
"""

import os, sys, warnings, argparse
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── PROJ fix (before rasterio) ────────────────────────────────────────────────
try:
    import pyproj as _pj
    _d = str(Path(_pj.datadir.get_data_dir()))
    os.environ["PROJ_DATA"] = _d
    os.environ["PROJ_LIB"]  = _d
except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
import pyproj

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
BS_BASE   = ROOT / "data_lake" / "burn_scars" / "hls_s30"
FIRMS_DIR = ROOT / "data_lake" / "fire_detections" / "firms_viirs_snpp_sp"
OUT_DIR   = ROOT / "data_lake" / "burn_scars"

# HLS pixel area: 30m × 30m = 900 m² = 0.09 ha
PIXEL_HA  = 0.09

# UTM 43N -> WGS84 converter (Proj4, no PROJ DB lookup needed)
_utm43n = pyproj.Proj("+proj=utm +zone=43 +datum=WGS84 +units=m +no_defs")
_wgs84  = pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs")
# WGS84 -> UTM 43N
def ll_to_utm(lon, lat):
    return pyproj.transform(_wgs84, _utm43n, lon, lat, errcheck=False)


# ── Load FIRMS ────────────────────────────────────────────────────────────────
def load_firms() -> pd.DataFrame:
    frames = []
    for pq in sorted(FIRMS_DIR.glob("*_annual.parquet")):
        df = pd.read_parquet(pq)
        if not df.empty:
            df["acq_date"] = pd.to_datetime(df["acq_date"])
            frames.append(df)
    if not frames:
        print("ERROR: No FIRMS parquet files found in", FIRMS_DIR)
        sys.exit(1)
    firms = pd.concat(frames, ignore_index=True)
    print(f"FIRMS loaded: {len(firms):,} detections | "
          f"{firms['acq_date'].dt.year.nunique()} years "
          f"({firms['acq_date'].dt.year.min()}–{firms['acq_date'].dt.year.max()})")
    return firms


# ── Pixel-level spatial check ─────────────────────────────────────────────────
def check_spatial_hit(firms_subset: pd.DataFrame,
                      mask: np.ndarray,
                      affine,
                      max_dist_m: float = 500.0):
    """
    For each FIRMS point in `firms_subset`:
      - Convert (lon, lat) -> UTM 43N (x, y) -> pixel (row, col)
      - If mask[row,col] == 1  -> direct spatial HIT
      - Else compute approximate distance to nearest burned pixel centroid

    Returns DataFrame with extra columns:
      row, col, in_bounds, spatial_hit, dist_to_burn_m
    """
    rows_out = []
    H, W = mask.shape

    # Pre-compute approximate burned pixel centroids (sample up to 5000)
    burned_yx = np.argwhere(mask == 1)
    if len(burned_yx) == 0:
        # No burned pixels — still record hits as False
        for _, r in firms_subset.iterrows():
            rows_out.append({"viirs_idx": r.name, "spatial_hit": False,
                             "dist_to_burn_m": np.nan, "in_bounds": False})
        return pd.DataFrame(rows_out)

    # Sample burned pixels for nearest-neighbour distance (max 5000)
    rng = np.random.default_rng(42)
    sample = burned_yx if len(burned_yx) <= 5000 else \
             burned_yx[rng.choice(len(burned_yx), 5000, replace=False)]
    # Convert to UTM x,y using affine  (col->x, row->y)
    bx = affine.c + (sample[:, 1] + 0.5) * affine.a   # x centroid
    by = affine.f + (sample[:, 0] + 0.5) * affine.e   # y centroid

    for _, r in firms_subset.iterrows():
        lon, lat = float(r["longitude"]), float(r["latitude"])
        # Convert FIRMS WGS84 -> UTM 43N
        fx, fy = ll_to_utm(lon, lat)
        # Get pixel row/col
        fc, fr = ~affine * (fx, fy)   # inverse affine: UTM -> pixel space
        fc, fr = int(fc), int(fr)     # col, row (note order!)

        in_bounds = (0 <= fr < H) and (0 <= fc < W)
        if in_bounds and mask[fr, fc] == 1:
            rows_out.append({"viirs_idx": r.name, "spatial_hit": True,
                             "dist_to_burn_m": 0.0, "in_bounds": True})
        else:
            # Nearest burned pixel distance
            dx = bx - fx
            dy = by - fy
            dist = float(np.sqrt(dx**2 + dy**2).min()) if len(bx) > 0 else np.nan
            rows_out.append({"viirs_idx": r.name, "spatial_hit": False,
                             "dist_to_burn_m": dist, "in_bounds": in_bounds})

    return pd.DataFrame(rows_out)


# ── Per-scene confidence scoring ──────────────────────────────────────────────
def score_confidence(best_hit: bool, near_hit: bool, gap: int, n_spatial: int) -> str:
    """
    Assign overall confidence that the linked FIRMS detection caused the scar.
    """
    if best_hit and gap <= 10:
        return "HIGH"
    if best_hit and gap <= 30:
        return "MEDIUM-HIGH"
    if (best_hit and gap <= 60) or (near_hit and gap <= 10):
        return "MEDIUM"
    if near_hit and gap <= 30:
        return "LOW-MEDIUM"
    if gap <= 60:
        return "LOW"
    return "NONE"


# ── Main ──────────────────────────────────────────────────────────────────────
def run(window_days: int = 60):
    firms = load_firms()

    masks = sorted(BS_BASE.rglob("*.tif"))
    if not masks:
        print(f"No burn scar masks found in {BS_BASE}")
        sys.exit(0)
    print(f"Processing {len(masks)} burn scar masks...")

    results = []

    for tif_path in masks:
        parts    = tif_path.parts
        idx      = list(parts).index("hls_s30") + 1
        tile     = parts[idx]
        yr, mo, dd = parts[idx+1], parts[idx+2], parts[idx+3]
        scene_dt = datetime(int(yr), int(mo), int(dd))

        # Load mask
        with rasterio.open(tif_path) as src:
            mask     = src.read(1)
            affine   = src.transform
            bounds   = src.bounds

        # Tile WGS84 bounds via Proj4
        lon_w, lat_s = pyproj.transform(_utm43n, _wgs84,
                                        bounds.left,  bounds.bottom,
                                        errcheck=False)
        lon_e, lat_n = pyproj.transform(_utm43n, _wgs84,
                                        bounds.right, bounds.top,
                                        errcheck=False)

        burn_px  = int(mask.sum())
        total_px = mask.size
        burn_pct = 100.0 * burn_px / total_px
        burn_ha  = burn_px * PIXEL_HA

        # Query FIRMS within tile bbox + time window
        t_start  = scene_dt - timedelta(days=window_days)
        nearby   = firms[
            (firms["latitude"]  >= lat_s) & (firms["latitude"]  <= lat_n) &
            (firms["longitude"] >= lon_w) & (firms["longitude"] <= lon_e) &
            (firms["acq_date"] >= t_start) & (firms["acq_date"] <= scene_dt)
        ].copy()

        n_in_window = len(nearby)

        if n_in_window == 0 or burn_px == 0:
            results.append({
                "scene_date": scene_dt.date(), "tile": tile,
                "burn_area_px": burn_px, "burn_area_ha": round(burn_ha, 1),
                "burn_pct": round(burn_pct, 2),
                "fire_date": None, "fire_time_utc": None, "fire_time_ist": None,
                "fire_lat": None, "fire_lon": None,
                "fire_frp_mw": None, "viirs_confidence": None, "daynight": None,
                "spatial_hit": False, "dist_to_burn_m": None,
                "days_gap": None, "n_firms_in_window": n_in_window,
                "n_spatial_hits": 0, "confidence": "NONE",
            })
            continue

        # Pixel-level spatial check for every FIRMS point in window
        hits_df = check_spatial_hit(nearby, mask, affine)
        nearby  = nearby.reset_index(drop=True)
        hits_df = hits_df.reset_index(drop=True)
        nearby  = pd.concat([nearby, hits_df[["spatial_hit","dist_to_burn_m"]]], axis=1)

        n_spatial_hits = int(nearby["spatial_hit"].sum())

        # Choose best causal fire:
        # Priority 1: spatial hit, earliest date, highest FRP
        # Priority 2: closest to burned area, earliest date
        spatial = nearby[nearby["spatial_hit"]]
        if not spatial.empty:
            # Sort by: date ASC, then FRP DESC
            best = spatial.sort_values(
                ["acq_date", "frp"], ascending=[True, False]
            ).iloc[0]
        else:
            # Fallback: closest point to burned area, then earliest date
            best = nearby.sort_values(
                ["dist_to_burn_m", "acq_date"], ascending=[True, True]
            ).iloc[0]

        # Time conversion
        t_raw = int(best["acq_time"])
        h_u, m_u = t_raw // 100, t_raw % 100
        tot_m    = h_u * 60 + m_u + 330           # IST = UTC + 5:30
        h_i, m_i = (tot_m // 60) % 24, tot_m % 60
        gap      = (scene_dt.date() - best["acq_date"].date()).days

        is_spatial_hit  = bool(best["spatial_hit"])
        dist            = float(best["dist_to_burn_m"]) if not is_spatial_hit else 0.0
        near_hit        = (not is_spatial_hit) and (dist < 500.0)
        confidence      = score_confidence(is_spatial_hit, near_hit, gap, n_spatial_hits)

        results.append({
            "scene_date":       scene_dt.date(),
            "tile":             tile,
            "burn_area_px":     burn_px,
            "burn_area_ha":     round(burn_ha, 1),
            "burn_pct":         round(burn_pct, 2),
            "fire_date":        best["acq_date"].date(),
            "fire_time_utc":    f"{h_u:02d}:{m_u:02d}",
            "fire_time_ist":    f"{h_i:02d}:{m_i:02d}",
            "fire_lat":         round(float(best["latitude"]), 5),
            "fire_lon":         round(float(best["longitude"]), 5),
            "fire_frp_mw":      round(float(best["frp"]), 2),
            "viirs_confidence": best["confidence"],
            "daynight":         best["daynight"],
            "spatial_hit":      is_spatial_hit,
            "dist_to_burn_m":   round(dist, 0) if not is_spatial_hit else 0,
            "days_gap":         gap,
            "n_firms_in_window":n_in_window,
            "n_spatial_hits":   n_spatial_hits,
            "confidence":       confidence,
        })

    # ── Save output ───────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUT_DIR / "fire_links.parquet"
    csv_path     = OUT_DIR / "fire_links.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {parquet_path}")
    print(f"Saved: {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("FIRE -> BURN SCAR CAUSAL LINKS")
    print("="*70)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)

    show_cols = [
        "scene_date", "tile", "burn_area_ha", "burn_pct",
        "fire_date", "fire_time_ist", "fire_frp_mw",
        "spatial_hit", "dist_to_burn_m", "days_gap",
        "n_firms_in_window", "confidence"
    ]
    print(df[show_cols].to_string(index=False))

    print("\n" + "-"*70)
    print("SUMMARY")
    print(f"  Total scenes processed    : {len(df)}")
    print(f"  Spatial HIT (fire on scar): {df['spatial_hit'].sum()} "
          f"({100*df['spatial_hit'].mean():.0f}%)")
    matched = df[df["fire_date"].notna()]
    print(f"  Any FIRMS match           : {len(matched)} "
          f"({100*len(matched)/max(len(df),1):.0f}%)")
    if not matched.empty:
        g = matched["days_gap"].dropna()
        print(f"  Gap (fire->scar visible)   : avg={g.mean():.0f}d  "
              f"min={g.min():.0f}d  max={g.max():.0f}d")
        print(f"  Peak FRP                  : "
              f"{matched['fire_frp_mw'].dropna().max()} MW")

    print("\n  Confidence breakdown:")
    for c, n in df["confidence"].value_counts().items():
        print(f"    {c:<15}: {n:3d} scenes")

    # High-confidence fire events
    high = df[df["confidence"].isin(["HIGH","MEDIUM-HIGH"])]
    if not high.empty:
        print(f"\n  HIGH/MEDIUM-HIGH confidence events ({len(high)}):")
        for _, r in high.iterrows():
            dn_label = "daytime" if r["daynight"] == "D" else "nighttime"
            print(f"    {r['scene_date']} | {r['tile']} | "
                  f"fire={r['fire_date']} {r['fire_time_ist']} IST ({dn_label}) | "
                  f"FRP={r['fire_frp_mw']} MW | gap={r['days_gap']}d | "
                  f"scar={r['burn_area_ha']} ha")

    return df


def main():
    p = argparse.ArgumentParser(
        description="Pixel-level FIRMS -> Burn Scar causal linker"
    )
    p.add_argument("--window", type=int, default=60,
                   help="Days to look back for FIRMS fires (default: 60)")
    p.add_argument("--csv-only", action="store_true",
                   help="Skip parquet, write CSV only")
    args = p.parse_args()
    run(window_days=args.window)


if __name__ == "__main__":
    main()
