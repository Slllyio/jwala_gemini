#!/usr/bin/env python3
"""
scripts/compute_dnbr_labels.py
===============================
Compute per-pixel continuous dNBR labels for VanAagni regression training.

dNBR = NBR_pre - NBR_post
where NBR = (NIR - SWIR2) / (NIR + SWIR2)   [HLS bands B8A and B12]

Output format:
  label.tif   float32  dNBR values, gated by Prithvi burn scar mask
              - 0.0    = no burn (background)
              - >0     = burned (dNBR = vegetation loss magnitude)
              - NaN    = nodata (cloud, missing imagery)
  weight.tif  float32  per-pixel loss weight (tier-based)
  meta.json   dict     with tier, dnbr_stats, scene paths

The burn scar mask gates the dNBR to prevent false severity from senescence
(tropical deciduous leaf drop Feb-May mimics NBR decrease).

Only processes burn-scar-confirmed events (GOLD/SILVER/BRONZE).
VIIRS-only events are skipped (no spectral data for dNBR).

Usage:
  python scripts/compute_dnbr_labels.py
  python scripts/compute_dnbr_labels.py --dry-run
  python scripts/compute_dnbr_labels.py --pre-window 45
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple

# PROJ fix
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
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# -- Paths ------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent
DATA_LAKE = ROOT / "data_lake"
BURN_DIR  = DATA_LAKE / "burn_scars" / "hls_s30"
HLS_DIR   = DATA_LAKE / "satellite_imagery" / "hls_s30"
LINKS_CSV = DATA_LAKE / "burn_scars" / "fire_links.csv"
OUT_DIR   = DATA_LAKE / "fire_labels"

# -- Tier weights -----------------------------------------------------------
TIER_WEIGHTS = {"GOLD": 3.0, "SILVER": 2.0, "BRONZE": 1.0}
BACKGROUND_WEIGHT = 0.1
LOOKBACK_DAYS     = 60
MIN_NEW_BURN_PX   = 10

# Pre-fire scene search
PRE_FIRE_MIN_DAYS = 30
PRE_FIRE_MAX_DAYS = 90
PRE_FIRE_SEARCH   = 12

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# Tier classification
# ===========================================================================

def classify_tier(confidence: str, spatial_hit: bool) -> str:
    conf = str(confidence).strip().upper()
    hit  = bool(spatial_hit)
    if conf == "NONE":
        return "EXCLUDE"
    if conf in ("HIGH", "MEDIUM-HIGH") and hit:
        return "GOLD"
    if conf == "MEDIUM" and hit:
        return "SILVER"
    if conf in ("HIGH", "MEDIUM-HIGH", "MEDIUM") and not hit:
        return "BRONZE"
    return "VIIRS_ONLY"


# ===========================================================================
# Burn scar helpers
# ===========================================================================

def find_burn_scar(tile: str, scene_date: str) -> Optional[Path]:
    try:
        dt = datetime.strptime(scene_date, "%Y-%m-%d")
    except ValueError:
        return None
    pattern = f"{tile}/{dt.year}/{dt.month:02d}/{dt.day:02d}/burnscar_*.tif"
    matches = sorted(BURN_DIR.glob(pattern))
    return matches[0] if matches else None


def find_previous_burn_scar(tile: str, before_date: str) -> Optional[Path]:
    try:
        dt = datetime.strptime(before_date, "%Y-%m-%d")
    except ValueError:
        return None
    for delta in range(1, LOOKBACK_DAYS + 1):
        prev = dt - timedelta(days=delta)
        pattern = f"{tile}/{prev.year}/{prev.month:02d}/{prev.day:02d}/burnscar_*.tif"
        matches = sorted(BURN_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def isolate_new_burns(current_path: Path,
                      prev_path: Optional[Path]) -> Tuple[np.ndarray, dict]:
    with rasterio.open(current_path) as src:
        current = src.read(1).astype(np.int16)
        profile = src.profile.copy()

    if prev_path is None:
        return (current == 1).astype(np.uint8), profile

    with rasterio.open(prev_path) as src:
        previous = src.read(1).astype(np.int16)

    if previous.shape != current.shape:
        from scipy.ndimage import zoom
        zy = current.shape[0] / previous.shape[0]
        zx = current.shape[1] / previous.shape[1]
        previous = zoom(previous, (zy, zx), order=0).astype(np.int16)

    return ((current == 1) & (previous != 1)).astype(np.uint8), profile


# ===========================================================================
# HLS scene lookup
# ===========================================================================

def find_hls_scene(tile: str, target: datetime,
                   search_days: int = PRE_FIRE_SEARCH) -> Optional[Path]:
    for delta in range(0, search_days + 1):
        for sign in ([0] if delta == 0 else [+delta, -delta]):
            dt = target + timedelta(days=sign)
            scene_dir = HLS_DIR / tile / f"{dt.year}" / f"{dt.month:02d}" / f"{dt.day:02d}"
            if scene_dir.exists() and any(scene_dir.glob("*.B8A.tif")):
                return scene_dir
    return None


def find_pre_fire_scene(tile: str, fire_date: datetime) -> Optional[Path]:
    """Find HLS scene 30-90 days before fire, searching at Sentinel-2 cadence."""
    for days_back in range(PRE_FIRE_MIN_DAYS, PRE_FIRE_MAX_DAYS + 1, 16):
        target = fire_date - timedelta(days=days_back)
        scene = find_hls_scene(tile, target, search_days=8)
        if scene is not None:
            return scene
    for days_back in range(PRE_FIRE_MIN_DAYS, PRE_FIRE_MAX_DAYS + 1):
        target = fire_date - timedelta(days=days_back)
        scene_dir = HLS_DIR / tile / f"{target.year}" / f"{target.month:02d}" / f"{target.day:02d}"
        if scene_dir.exists() and any(scene_dir.glob("*.B8A.tif")):
            return scene_dir
    return None


def load_nbr_bands(scene_dir: Path) -> Tuple[Optional[np.ndarray],
                                               Optional[np.ndarray],
                                               Optional[dict]]:
    """Load B8A (NIR) and B12 (SWIR2) as reflectance."""
    nir_files   = sorted(scene_dir.glob("*.B8A.tif"))
    swir2_files = sorted(scene_dir.glob("*.B12.tif"))
    if not nir_files or not swir2_files:
        return None, None, None

    with rasterio.open(nir_files[0]) as src:
        nir_raw = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    with rasterio.open(swir2_files[0]) as src:
        swir2_raw = src.read(1).astype(np.float32)

    nodata_nir   = (nir_raw <= -9990) | (nir_raw == 0)
    nodata_swir2 = (swir2_raw <= -9990) | (swir2_raw == 0)

    nir   = np.clip(nir_raw * 1e-4, 0.0, 1.0)
    swir2 = np.clip(swir2_raw * 1e-4, 0.0, 1.0)
    nir[nodata_nir]     = np.nan
    swir2[nodata_swir2] = np.nan

    return nir, swir2, profile


def compute_nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    eps = 1e-6
    return (nir - swir2) / (nir + swir2 + eps)


# ===========================================================================
# Write output files
# ===========================================================================

def write_dnbr_files(out_dir: Path, tile: str, fire_date: str,
                     dnbr_label: np.ndarray, weight: np.ndarray,
                     profile: dict, meta: dict,
                     dry_run: bool = False) -> bool:
    dt = datetime.strptime(fire_date, "%Y-%m-%d")
    tag = f"{dt.strftime('%Y%m%d')}_{tile}"

    label_path  = out_dir / f"label_{tag}.tif"
    weight_path = out_dir / f"weight_{tag}.tif"
    meta_path   = out_dir / f"meta_{tag}.json"

    burned_px = int(np.nansum(dnbr_label > 0))
    if dry_run:
        log.info(f"  [DRY-RUN] Would write {label_path.name}  burned_px={burned_px}")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    # Label raster: float32 dNBR
    lbl_profile = profile.copy()
    lbl_profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    with rasterio.open(label_path, "w", **lbl_profile) as dst:
        dst.write(dnbr_label[np.newaxis, :, :])

    # Weight raster
    wt_profile = profile.copy()
    wt_profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    with rasterio.open(weight_path, "w", **wt_profile) as dst:
        dst.write(weight[np.newaxis, :, :])

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return True


# ===========================================================================
# Main
# ===========================================================================

def build_dnbr_labels(dry_run: bool = False) -> None:
    df = pd.read_csv(LINKS_CSV)
    df["fire_date"]  = pd.to_datetime(df["fire_date"]).dt.strftime("%Y-%m-%d")
    df["scene_date"] = pd.to_datetime(df["scene_date"]).dt.strftime("%Y-%m-%d")

    df["tier"] = df.apply(
        lambda r: classify_tier(r["confidence"], r["spatial_hit"]), axis=1
    )

    # Only process burn-scar-confirmed events (skip VIIRS_ONLY)
    df_valid = df[df["tier"].isin(("GOLD", "SILVER", "BRONZE"))].copy()
    TIER_RANK = {"BRONZE": 1, "SILVER": 2, "GOLD": 3}
    df_valid["_rank"] = df_valid["tier"].map(TIER_RANK)
    df_valid = df_valid.sort_values("_rank").reset_index(drop=True)

    stats = {"processed": 0, "dnbr_success": 0, "dnbr_no_pre": 0,
             "skip_noburn": 0, "skip_no_mask": 0}
    all_dnbr = []  # collect per-fire dNBR stats

    log.info(f"Computing continuous dNBR labels (REGRESSION mode)")
    log.info(f"Burn-scar-confirmed events: {len(df_valid)}")
    log.info(f"Skipping VIIRS_ONLY ({(df['tier']=='VIIRS_ONLY').sum()}) "
             f"and EXCLUDE ({(df['tier']=='EXCLUDE').sum()})")

    for idx, row in df_valid.iterrows():
        tile       = str(row["tile"])
        fire_date  = str(row["fire_date"])
        scene_date = str(row["scene_date"])
        tier       = row["tier"]

        dt_fire  = datetime.strptime(fire_date, "%Y-%m-%d")
        dt_scene = datetime.strptime(scene_date, "%Y-%m-%d")
        out_sub  = OUT_DIR / tile / str(dt_fire.year) / f"{dt_fire.month:02d}"

        # 1. Load burn scar mask
        mask_path = find_burn_scar(tile, scene_date)
        if mask_path is None:
            stats["skip_no_mask"] += 1
            continue

        prev_path = find_previous_burn_scar(tile, scene_date)
        burn_mask, profile = isolate_new_burns(mask_path, prev_path)

        fire_px = int((burn_mask == 1).sum())
        if fire_px < MIN_NEW_BURN_PX:
            stats["skip_noburn"] += 1
            continue

        # 2. Load POST-fire HLS (scene_date = burn scar detection date)
        post_scene = find_hls_scene(tile, dt_scene, search_days=3)
        if post_scene is None:
            stats["skip_no_mask"] += 1
            continue

        nir_post, swir2_post, _ = load_nbr_bands(post_scene)
        if nir_post is None:
            stats["skip_no_mask"] += 1
            continue

        nbr_post = compute_nbr(nir_post, swir2_post)

        # 3. Find PRE-fire HLS scene
        pre_scene = find_pre_fire_scene(tile, dt_fire)

        if pre_scene is None:
            # No pre-fire scene: use a uniform low dNBR estimate for burned px
            dnbr_full = np.zeros(burn_mask.shape, dtype=np.float32)
            dnbr_full[burn_mask == 1] = 0.10  # nominal surface fire dNBR
            dnbr_mean, dnbr_std, dnbr_min, dnbr_max = 0.10, 0.0, 0.10, 0.10
            stats["dnbr_no_pre"] += 1
        else:
            nir_pre, swir2_pre, _ = load_nbr_bands(pre_scene)
            if nir_pre is None:
                dnbr_full = np.zeros(burn_mask.shape, dtype=np.float32)
                dnbr_full[burn_mask == 1] = 0.10
                dnbr_mean, dnbr_std, dnbr_min, dnbr_max = 0.10, 0.0, 0.10, 0.10
                stats["dnbr_no_pre"] += 1
            else:
                # Handle shape mismatch
                if nir_pre.shape != nir_post.shape:
                    from scipy.ndimage import zoom
                    zy = nir_post.shape[0] / nir_pre.shape[0]
                    zx = nir_post.shape[1] / nir_pre.shape[1]
                    nir_pre   = zoom(nir_pre, (zy, zx), order=1)
                    swir2_pre = zoom(swir2_pre, (zy, zx), order=1)

                nbr_pre = compute_nbr(nir_pre, swir2_pre)
                dnbr_raw = nbr_pre - nbr_post  # positive = vegetation loss

                # Resize burn_mask if needed
                if burn_mask.shape != dnbr_raw.shape:
                    from scipy.ndimage import zoom as zoom2
                    zy = dnbr_raw.shape[0] / burn_mask.shape[0]
                    zx = dnbr_raw.shape[1] / burn_mask.shape[1]
                    burn_mask = zoom2(burn_mask, (zy, zx), order=0).astype(np.uint8)

                # Build continuous label:
                # Background (burn_mask == 0) -> 0.0
                # Burned (burn_mask == 1) -> clip(dNBR, 0, max)
                # The clip at 0 removes negative dNBR (regrowth/phenology noise)
                dnbr_full = np.zeros(burn_mask.shape, dtype=np.float32)
                burned = (burn_mask == 1) & ~np.isnan(dnbr_raw)
                dnbr_full[burned] = np.clip(dnbr_raw[burned], 0.0, 2.0)

                # NaN pixels in dNBR (cloud/missing): mark as NaN in label
                nan_in_burn = (burn_mask == 1) & np.isnan(dnbr_raw)
                dnbr_full[nan_in_burn] = np.nan

                # Stats
                valid_burned = dnbr_full[burned]
                if len(valid_burned) > 0:
                    dnbr_mean = float(np.mean(valid_burned))
                    dnbr_std  = float(np.std(valid_burned))
                    dnbr_min  = float(np.min(valid_burned))
                    dnbr_max  = float(np.max(valid_burned))
                else:
                    dnbr_mean = dnbr_std = dnbr_min = dnbr_max = 0.0

                stats["dnbr_success"] += 1

        # 4. Build weight map
        weight = np.full(dnbr_full.shape, BACKGROUND_WEIGHT, dtype=np.float32)
        weight[burn_mask == 1] = TIER_WEIGHTS[tier]
        weight[np.isnan(dnbr_full)] = 0.0  # no gradient for NaN pixels

        # 5. Write output
        meta = {
            "tier":            tier,
            "weight_value":    TIER_WEIGHTS[tier],
            "fire_date":       fire_date,
            "scene_date":      scene_date,
            "tile":            tile,
            "confidence":      str(row["confidence"]),
            "spatial_hit":     bool(row["spatial_hit"]),
            "days_gap":        float(row.get("days_gap", -1)),
            "burn_area_ha":    float(row.get("burn_area_ha", 0)),
            "fire_px_new":     fire_px,
            "label_source":    "dnbr_continuous",
            "label_format":    "float32_dnbr",
            "mask_path":       str(mask_path),
            "prev_mask":       str(prev_path) if prev_path else None,
            "pre_fire_scene":  str(pre_scene) if pre_scene else None,
            "post_fire_scene": str(post_scene),
            "dnbr_stats": {
                "mean": round(dnbr_mean, 4),
                "std":  round(dnbr_std, 4),
                "min":  round(dnbr_min, 4),
                "max":  round(dnbr_max, 4),
            },
        }

        ok = write_dnbr_files(out_sub, tile, fire_date,
                              dnbr_full, weight, profile, meta, dry_run)
        if ok:
            stats["processed"] += 1
            all_dnbr.append({
                "fire_date": fire_date, "tile": tile, "tier": tier,
                "fire_px": fire_px,
                "dnbr_mean": round(dnbr_mean, 4),
                "dnbr_std": round(dnbr_std, 4),
                "dnbr_max": round(dnbr_max, 4),
            })
            log.info(
                f"[{stats['processed']:3d}] {tier:8s} {tile} {fire_date} "
                f"fire={fire_px:>7d}  "
                f"dNBR={dnbr_mean:.3f}+-{dnbr_std:.3f} "
                f"[{dnbr_min:.3f}, {dnbr_max:.3f}]"
            )

    # -- Summary ---------------------------------------------------------------
    log.info(f"\n{'='*60}")
    log.info("dNBR Label Computation Summary (REGRESSION)")
    log.info(f"{'='*60}")
    for k, v in stats.items():
        log.info(f"  {k:20s}: {v}")

    if all_dnbr:
        means = [d["dnbr_mean"] for d in all_dnbr]
        log.info(f"\ndNBR across all fires:")
        log.info(f"  Mean of means: {np.mean(means):.4f}")
        log.info(f"  Median:        {np.median(means):.4f}")
        log.info(f"  Range:         [{min(means):.4f}, {max(means):.4f}]")
        log.info(f"  Max per-pixel: {max(d['dnbr_max'] for d in all_dnbr):.4f}")

    log.info(f"{'='*60}")

    # Save summary
    if not dry_run:
        summary_path = OUT_DIR / "dnbr_summary.json"
        summary = {
            "mode": "regression",
            "stats": stats,
            "per_fire": all_dnbr,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute continuous dNBR labels for VanAagni regression training"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing files")
    args = parser.parse_args()

    build_dnbr_labels(dry_run=args.dry_run)
