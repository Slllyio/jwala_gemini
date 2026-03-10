#!/usr/bin/env python3
"""
scripts/build_fire_labels.py
============================
Build tiered per-pixel fire label rasters from Prithvi burn scar masks
+ VIIRS fire_links.csv for VanAgni training.

Label tiers (from fire_links.csv confidence + spatial_hit):
  GOLD       burn_scar + VIIRS spatial_hit=True + HIGH/MEDIUM-HIGH conf  → weight 3.0
  SILVER     burn_scar + VIIRS spatial_hit=True + MEDIUM conf            → weight 2.0
  BRONZE     burn_scar + VIIRS spatial_hit=False + HIGH/MEDIUM conf      → weight 1.0
  VIIRS_ONLY no burn scar found for that date                            → weight 0.5
  EXCLUDE    confidence=NONE                                             → skip

For GOLD/SILVER/BRONZE: label is taken directly from the Prithvi burn scar
mask (uint8, 0=no-burn, 1=burned) with temporal differencing to isolate
pixels newly burned in THIS event.

For VIIRS_ONLY: a 375m-radius circular raster is burned around the VIIRS
detection point as a coarse fallback.

Output structure:
  data_lake/fire_labels/{tile}/{YYYY}/{MM}/
    label_{YYYYMMDD}_{tile}.tif     uint8  {0=background, 1=fire, 255=nodata}
    weight_{YYYYMMDD}_{tile}.tif    float32 per-pixel loss weight
    meta_{YYYYMMDD}_{tile}.json     dict with tier, confidence, etc.

Usage:
  python scripts/build_fire_labels.py
  python scripts/build_fire_labels.py --dry-run      # print stats, no write
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple

# ── PROJ fix: pin to rasterio's vendored PROJ db BEFORE any geospatial imports
# PostgreSQL PostGIS ships an old PROJ.db that shadows rasterio's version.
def _fix_proj():
    try:
        import pyproj as _pp
        _proj_data = str(Path(_pp.datadir.get_data_dir()))
        os.environ["PROJ_DATA"] = _proj_data
        os.environ["PROJ_LIB"]  = _proj_data
        os.environ.pop("GDAL_DATA", None)
    except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")
_fix_proj()

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Point
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_LAKE = ROOT / "data_lake"
BURN_DIR  = DATA_LAKE / "burn_scars" / "hls_s30"
HLS_DIR   = DATA_LAKE / "satellite_imagery" / "hls_s30"
LINKS_CSV = DATA_LAKE / "burn_scars" / "fire_links.csv"
OUT_DIR   = DATA_LAKE / "fire_labels"

# ── Constants ─────────────────────────────────────────────────────────────────
TIER_WEIGHTS = {
    "GOLD":       3.0,
    "SILVER":     2.0,
    "BRONZE":     1.0,
    "VIIRS_ONLY": 0.5,
}
BACKGROUND_WEIGHT = 0.1   # weight for non-fire pixels
VIIRS_RADIUS_M    = 375   # VIIRS pixel size for fallback rasterization
LOOKBACK_DAYS     = 60    # max days to look back for previous burn scar
MIN_NEW_BURN_PX   = 10    # minimum new-burn pixels to save label

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Tier Classification
# ═════════════════════════════════════════════════════════════════════════════

def classify_tier(confidence: str, spatial_hit: bool) -> str:
    """Map (confidence, spatial_hit) → label tier string."""
    conf = str(confidence).strip().upper()
    hit  = bool(spatial_hit)

    if conf == "NONE":
        return "EXCLUDE"
    if conf in ("HIGH", "MEDIUM-HIGH") and hit:
        return "GOLD"
    if conf == "MEDIUM" and hit:
        return "SILVER"
    if conf in ("HIGH", "MEDIUM-HIGH", "MEDIUM") and not hit:
        # Burn scar confirmed via spectral analysis; VIIRS may have missed
        # due to cloud cover, night timing, or spatial displacement
        return "BRONZE"
    # LOW / LOW-MEDIUM confidence or unhandled → coarse VIIRS fallback
    return "VIIRS_ONLY"


# ═════════════════════════════════════════════════════════════════════════════
# Burn Scar File Lookup
# ═════════════════════════════════════════════════════════════════════════════

def find_burn_scar(tile: str, scene_date: str) -> Optional[Path]:
    """
    Find burn scar .tif for a given tile and scene_date (YYYY-MM-DD).
    Pattern: BURN_DIR/{tile}/{YYYY}/{MM}/{DD}/burnscar_*.tif
    """
    try:
        dt = datetime.strptime(scene_date, "%Y-%m-%d")
    except ValueError:
        return None
    pattern = f"{tile}/{dt.year}/{dt.month:02d}/{dt.day:02d}/burnscar_*.tif"
    matches = sorted(BURN_DIR.glob(pattern))
    return matches[0] if matches else None


def find_previous_burn_scar(tile: str, before_date: str,
                             lookback_days: int = LOOKBACK_DAYS) -> Optional[Path]:
    """
    Find the most recent burn scar strictly before before_date.
    Searches backwards day by day up to lookback_days.
    """
    try:
        dt = datetime.strptime(before_date, "%Y-%m-%d")
    except ValueError:
        return None
    for delta in range(1, lookback_days + 1):
        prev = dt - timedelta(days=delta)
        pattern = f"{tile}/{prev.year}/{prev.month:02d}/{prev.day:02d}/burnscar_*.tif"
        matches = sorted(BURN_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Burn Scar → Label Raster
# ═════════════════════════════════════════════════════════════════════════════

def isolate_new_burns(
    current_path: Path,
    prev_path: Optional[Path],
) -> Tuple[np.ndarray, dict]:
    """
    Extract pixels that burned NEWLY in this event by subtracting the
    previous burn scar mask.

    Returns (new_burn_array uint8 [H, W], rasterio_profile)
    new_burn_array values: 0=background, 1=new burn
    """
    with rasterio.open(current_path) as src:
        current = src.read(1).astype(np.int16)
        profile = src.profile.copy()

    if prev_path is None:
        # No prior mask available → treat all burned pixels as new
        new_burns = (current == 1).astype(np.uint8)
        return new_burns, profile

    with rasterio.open(prev_path) as src:
        previous = src.read(1).astype(np.int16)

    # Resize previous to match current if shapes differ (rare)
    if previous.shape != current.shape:
        from scipy.ndimage import zoom
        zy = current.shape[0] / previous.shape[0]
        zx = current.shape[1] / previous.shape[1]
        previous = zoom(previous, (zy, zx), order=0).astype(np.int16)

    # New burns = currently burned AND not previously burned
    new_burns = ((current == 1) & (previous != 1)).astype(np.uint8)
    return new_burns, profile


# ═════════════════════════════════════════════════════════════════════════════
# VIIRS Fallback Rasterization
# ═════════════════════════════════════════════════════════════════════════════

def get_reference_profile(tile: str) -> Optional[dict]:
    """
    Get rasterio profile from any available HLS scene for this tile.
    Used to define the grid for VIIRS-only fallback labels.
    """
    for tif in HLS_DIR.glob(f"{tile}/**/*.B02.tif"):
        with rasterio.open(tif) as src:
            return src.profile.copy()
    # Try burn scar dir as fallback
    for tif in BURN_DIR.glob(f"{tile}/**/*.tif"):
        with rasterio.open(tif) as src:
            return src.profile.copy()
    return None


def rasterize_viirs_point(
    lat: float,
    lon: float,
    tile: str,
    radius_m: float = VIIRS_RADIUS_M,
) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Rasterize a VIIRS detection point as a filled circle of radius_m metres.
    Returns (label uint8 [H, W], profile) or (None, None) if no reference grid.
    """
    profile = get_reference_profile(tile)
    if profile is None:
        log.warning(f"  No reference profile for tile {tile}, skipping VIIRS rasterization")
        return None, None

    # Build point in native CRS (VIIRS coords are WGS84 lat/lon)
    # Need to convert to tile UTM
    try:
        from pyproj import Transformer
        # Profile CRS is UTM 43N
        crs_str = str(profile.get("crs", "EPSG:32643"))
        transformer = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
        x_utm, y_utm = transformer.transform(lon, lat)
    except Exception as e:
        log.warning(f"  CRS transform failed: {e}")
        return None, None

    H = profile["height"]
    W = profile["width"]
    transform = profile["transform"]

    label = np.zeros((H, W), dtype=np.uint8)

    # Convert UTM coords to pixel coords
    col = (x_utm - transform.c) / transform.a
    row = (y_utm - transform.f) / transform.e  # e is negative (north-up)

    # Radius in pixels (30m resolution)
    r_px = int(np.ceil(radius_m / abs(transform.a)))

    # Paint circle
    rr = int(round(row))
    cc = int(round(col))
    for dr in range(-r_px, r_px + 1):
        for dc in range(-r_px, r_px + 1):
            if dr**2 + dc**2 <= r_px**2:
                rr2 = rr + dr
                cc2 = cc + dc
                if 0 <= rr2 < H and 0 <= cc2 < W:
                    label[rr2, cc2] = 1

    return label, profile


# ═════════════════════════════════════════════════════════════════════════════
# Write Label + Weight + Meta
# ═════════════════════════════════════════════════════════════════════════════

def write_label_files(
    out_dir: Path,
    tile: str,
    fire_date: str,
    label: np.ndarray,
    weight: np.ndarray,
    profile: dict,
    meta: dict,
    dry_run: bool = False,
) -> bool:
    """
    Write label.tif, weight.tif, and meta.json to out_dir.
    Returns True on success.
    """
    dt = datetime.strptime(fire_date, "%Y-%m-%d")
    tag = f"{dt.strftime('%Y%m%d')}_{tile}"

    label_path  = out_dir / f"label_{tag}.tif"
    weight_path = out_dir / f"weight_{tag}.tif"
    meta_path   = out_dir / f"meta_{tag}.json"

    if dry_run:
        fire_px = int((label == 1).sum())
        log.info(f"  [DRY-RUN] Would write {label_path.name}  fire_px={fire_px}")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    # Label raster
    lbl_profile = profile.copy()
    lbl_profile.update(dtype="uint8", count=1, nodata=255, compress="lzw")
    with rasterio.open(label_path, "w", **lbl_profile) as dst:
        dst.write(label[np.newaxis, :, :])

    # Weight raster
    wt_profile = profile.copy()
    wt_profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    with rasterio.open(weight_path, "w", **wt_profile) as dst:
        dst.write(weight[np.newaxis, :, :])

    # Meta JSON
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return True


# ═════════════════════════════════════════════════════════════════════════════
# Main Build Loop
# ═════════════════════════════════════════════════════════════════════════════

def build_labels(dry_run: bool = False) -> None:
    df = pd.read_csv(LINKS_CSV)
    df["fire_date"]  = pd.to_datetime(df["fire_date"]).dt.strftime("%Y-%m-%d")
    df["scene_date"] = pd.to_datetime(df["scene_date"]).dt.strftime("%Y-%m-%d")

    # Assign tiers
    df["tier"] = df.apply(
        lambda r: classify_tier(r["confidence"], r["spatial_hit"]), axis=1
    )

    # Sort so best tier is processed LAST per (tile, fire_date) group.
    # "Last write wins" -> GOLD always overwrites VIIRS_ONLY for the same date.
    TIER_RANK = {"EXCLUDE": 0, "VIIRS_ONLY": 1, "BRONZE": 2, "SILVER": 3, "GOLD": 4}
    df["_tier_rank"] = df["tier"].map(TIER_RANK).fillna(0)
    df = df.sort_values("_tier_rank").reset_index(drop=True)

    stats = {t: 0 for t in ["GOLD", "SILVER", "BRONZE", "VIIRS_ONLY", "EXCLUDE", "SKIP_NOBURN", "SKIP_NOREF"]}
    skipped_no_mask = 0

    log.info(f"Processing {len(df)} fire events from {LINKS_CSV.name}")
    log.info(f"Tier distribution:\n{df['tier'].value_counts().to_string()}")

    for idx, row in df.iterrows():
        tile       = str(row["tile"])
        fire_date  = str(row["fire_date"])
        scene_date = str(row["scene_date"])
        tier       = row["tier"]

        if tier == "EXCLUDE":
            stats["EXCLUDE"] += 1
            continue

        dt_fire = datetime.strptime(fire_date, "%Y-%m-%d")
        out_sub = OUT_DIR / tile / str(dt_fire.year) / f"{dt_fire.month:02d}"

        # ── Burn-scar-based label (GOLD / SILVER / BRONZE) ───────────────
        if tier in ("GOLD", "SILVER", "BRONZE"):
            mask_path = find_burn_scar(tile, scene_date)

            if mask_path is None:
                log.warning(f"[{idx}] Burn scar NOT FOUND for {tile} {scene_date} — falling back to VIIRS_ONLY")
                tier = "VIIRS_ONLY"
                skipped_no_mask += 1
            else:
                prev_path = find_previous_burn_scar(tile, scene_date)
                label, profile = isolate_new_burns(mask_path, prev_path)

                fire_px = int((label == 1).sum())
                if fire_px < MIN_NEW_BURN_PX:
                    log.debug(f"[{idx}] Only {fire_px} new-burn pixels for {tile} {scene_date} — skipping")
                    stats["SKIP_NOBURN"] += 1
                    continue

                # Per-pixel weight map
                weight = np.where(
                    label == 1,
                    TIER_WEIGHTS[tier],
                    BACKGROUND_WEIGHT
                ).astype(np.float32)

                meta = {
                    "tier":         tier,
                    "weight_value": TIER_WEIGHTS[tier],
                    "fire_date":    fire_date,
                    "scene_date":   scene_date,
                    "tile":         tile,
                    "confidence":   str(row["confidence"]),
                    "spatial_hit":  bool(row["spatial_hit"]),
                    "days_gap":     float(row.get("days_gap", -1)),
                    "burn_area_ha": float(row.get("burn_area_ha", 0)),
                    "fire_px_new":  fire_px,
                    "label_source": "burn_scar_isolated",
                    "mask_path":    str(mask_path),
                    "prev_mask":    str(prev_path) if prev_path else None,
                }

                ok = write_label_files(out_sub, tile, fire_date, label, weight, profile, meta, dry_run)
                if ok:
                    stats[tier] += 1
                    log.info(
                        f"[{idx+1:3d}/{len(df)}] {tier:10s} {tile} {fire_date} → "
                        f"fire_px={fire_px:6d}  ha={row.get('burn_area_ha',0):.0f}"
                    )
                continue  # go to next row

        # ── VIIRS-only fallback ───────────────────────────────────────────
        if tier == "VIIRS_ONLY":
            try:
                lat = float(row["fire_lat"])
                lon = float(row["fire_lon"])
            except (KeyError, ValueError):
                stats["SKIP_NOREF"] += 1
                continue

            label, profile = rasterize_viirs_point(lat, lon, tile)
            if label is None:
                stats["SKIP_NOREF"] += 1
                continue

            fire_px = int((label == 1).sum())
            weight = np.where(
                label == 1,
                TIER_WEIGHTS["VIIRS_ONLY"],
                BACKGROUND_WEIGHT
            ).astype(np.float32)

            meta = {
                "tier":         "VIIRS_ONLY",
                "weight_value": TIER_WEIGHTS["VIIRS_ONLY"],
                "fire_date":    fire_date,
                "scene_date":   scene_date,
                "tile":         tile,
                "confidence":   str(row["confidence"]),
                "spatial_hit":  bool(row["spatial_hit"]),
                "fire_lat":     lat,
                "fire_lon":     lon,
                "fire_frp_mw":  float(row.get("fire_frp_mw", 0)),
                "fire_px_new":  fire_px,
                "label_source": "viirs_rasterized_375m",
            }

            ok = write_label_files(out_sub, tile, fire_date, label, weight, profile, meta, dry_run)
            if ok:
                stats["VIIRS_ONLY"] += 1
                log.info(
                    f"[{idx+1:3d}/{len(df)}] VIIRS_ONLY  {tile} {fire_date} → "
                    f"lat={lat:.3f} lon={lon:.3f}  fire_px={fire_px}"
                )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LABEL BUILD COMPLETE" + ("  [DRY-RUN]" if dry_run else ""))
    print("=" * 60)
    usable = sum(v for k, v in stats.items() if k not in ("EXCLUDE", "SKIP_NOBURN", "SKIP_NOREF"))
    for tier, count in stats.items():
        bar = "#" * min(count // 5, 30)
        print(f"  {tier:14s}: {count:4d}  {bar}")
    print(f"\n  Total usable labels : {usable}")
    if usable > 0:
        gold_silver = stats["GOLD"] + stats["SILVER"]
        print(f"  GOLD+SILVER quality : {gold_silver/usable*100:.1f}%  (30m precise)")
    if not dry_run:
        print(f"\n  Label rasters -> {OUT_DIR}")
    print("=" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build VanAgni tiered fire label rasters")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing any files")
    parser.add_argument("--out-dir", type=str, default=None,
                        help=f"Override output directory (default: {OUT_DIR})")
    args = parser.parse_args()

    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_labels(dry_run=args.dry_run)
