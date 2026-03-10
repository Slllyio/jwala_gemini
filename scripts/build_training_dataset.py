#!/usr/bin/env python3
"""
scripts/build_training_dataset.py
==================================
Assemble VanAgni training patches: (X, y, w) triplets saved as .npz files.

Each .npz patch contains:
  hls        float32  (6, T=3, H=224, W=224)  Normalised HLS S30 bands
  indices    float32  (3, T=3, H=224, W=224)  NDVI, NBR, BSI per frame
  weather    float32  (10,)                   Current-day weather (normalised)
  weather_7d float32  (10, 7)                 7-day look-back window (normalised)
  terrain    float32  (4, H, W)               elev_norm, slope_norm, sin_asp, cos_asp
  burn_age   float32  (3, H, W)               recent / high-risk / mature channels
  landcover  float32  (11, H, W)              ESA WorldCover one-hot
  label      float32  (H, W)                  continuous dNBR (0=background, >0=burn severity)
  weight     float32  (H, W)                  per-pixel loss weight

Pipeline:
  1.  Load fire_links.csv (from burn scar pipeline)
  2.  Pre-warp terrain to UTM43N (done once, cached)
  3.  For each non-EXCLUDE fire event:
        a.  t_input = fire_date − PRED_HORIZON days  (pre-fire imagery)
        b.  Build T=3 HLS temporal stack  [t_input, t_input−16d, t_input−32d]
        c.  Get FWI weather features
        d.  Load terrain patch (warped)
        e.  Compute burn-age raster (days since last burn per pixel)
        f.  Load ESA WorldCover one-hot
        g.  Load tiered label (from build_fire_labels.py output)
        h.  Extract ≤3 patches centred on fire pixels  (224×224)
        i.  Save .npz  +  manifest row
  4.  Generate 2× negative samples per positive patch
  5.  Write manifest.csv

Usage:
  python scripts/build_training_dataset.py
  python scripts/build_training_dataset.py --pred-horizon 1   # 1-day ahead
  python scripts/build_training_dataset.py --max-events 50    # quick smoke-test
"""

import os
import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# ── PROJ fix: pin to rasterio's vendored PROJ db BEFORE any geospatial imports
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
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_LAKE  = ROOT / "data_lake"
HLS_DIR    = DATA_LAKE / "satellite_imagery" / "hls_s30"
BURN_DIR   = DATA_LAKE / "burn_scars"  / "hls_s30"
LABEL_DIR  = DATA_LAKE / "fire_labels"
TERRAIN_DIR= DATA_LAKE / "terrain"
LC_DIR     = DATA_LAKE / "land_cover"
FWI_PATH   = DATA_LAKE / "fire_weather" / "fwi_daily.parquet"
LINKS_CSV  = DATA_LAKE / "burn_scars"  / "fire_links.csv"
OUT_DIR    = DATA_LAKE / "training_patches"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
PATCH_H          = 224          # pixels  (224×224 @ 30m = 6.72 km)
PATCH_W          = 224
N_FRAMES         = 3            # temporal depth
TEMPORAL_CADENCE = 16           # days between frames
PRED_HORIZON     = 3            # days before fire_date for t_input
SEARCH_WINDOW    = 12           # ±days to search for cloud-free HLS scene
N_PATCHES_POS    = 3            # patches per positive fire event (legacy, unused in grid mode)
N_NEG_PER_POS    = 2            # negative patches per positive (legacy, unused in grid mode)
MIN_FIRE_PX      = 10           # min fire pixels to keep a fire patch
GRID_STRIDE      = 224          # grid stride (no overlap -- fixes data leakage from near-dupes)
BG_SAMPLE_RATE   = 0.15         # fraction of background grid patches to keep

# Band ordering  (Blue Green Red NIR SWIR1 SWIR2)
HLS_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]

# Prithvi-eo-v2 normalization constants (from burn_scars_config.yaml)
# Applied AFTER dividing int16 DN by 10 000
HLS_MEANS = np.array([0.033, 0.072, 0.069, 0.235, 0.172, 0.105], dtype=np.float32)
HLS_STDS  = np.array([0.012, 0.025, 0.030, 0.057, 0.069, 0.065], dtype=np.float32)

# Terrain normalization (Guna Division from terrain_stats.json)
ELEV_MIN, ELEV_MAX = 255.0, 589.0
SLOPE_MAX          = 45.0

# ESA WorldCover class value → one-hot index
LC_MAP    = {10:0, 20:1, 30:2, 40:3, 50:4, 60:5, 70:6, 80:7, 90:8, 95:9, 100:10}
N_LC      = 11

# FWI weather columns
WEATHER_COLS = ["temp_c", "rh_pct", "wind_kmh", "precip_mm",
                "ffmc", "dmc", "dc", "isi", "bui", "fwi"]

# Rough per-column normalization stats (2013-2025 Guna climatology)
WEATHER_MEANS = np.array([28.5, 45.0, 12.0,  2.0,  75.0,  55.0, 200.0,  7.0,  70.0,  15.0],
                          dtype=np.float32)
WEATHER_STDS  = np.array([ 8.0, 22.0,  8.0,  6.0,  18.0,  40.0, 150.0,  5.0,  50.0,  12.0],
                          dtype=np.float32)

# Train / Val / Test year split
# 2019 reserved for validation (63 events, HLS present Feb-May).
# 2022-2023 December fire events have no downloaded HLS so are unusable for val.
SPLIT_MAP = {**{y: "train" for y in range(2013, 2019)},
             2019: "val",
             **{y: "train" for y in range(2020, 2024)},
             **{y: "test"  for y in range(2024, 2026)}}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# HLS Scene Helpers
# ═════════════════════════════════════════════════════════════════════════════

def find_hls_scene(tile: str, target: datetime,
                   search_days: int = SEARCH_WINDOW) -> Optional[Path]:
    """Return path to HLS scene directory nearest to target date."""
    for delta in range(0, search_days + 1):
        for sign in ([0] if delta == 0 else [+delta, -delta]):
            dt = target + timedelta(days=sign)
            scene_dir = HLS_DIR / tile / f"{dt.year}" / f"{dt.month:02d}" / f"{dt.day:02d}"
            if scene_dir.exists() and any(scene_dir.glob("*.B02.tif")):
                return scene_dir
    return None


def load_hls_bands(scene_dir: Path) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Load 6-band HLS array from a scene directory.
    Returns: (float32 [6, H, W] in reflectance 0-1, rasterio profile)
             or (None, None) if any band is missing.
    """
    arrays, profile = [], None
    for band in HLS_BANDS:
        matches = sorted(scene_dir.glob(f"*.{band}.tif"))
        if not matches:
            return None, None
        with rasterio.open(matches[0]) as src:
            raw = src.read(1).astype(np.float32)
            if profile is None:
                profile = src.profile.copy()
        nodata = (raw <= -9990) | (raw == 0)
        reflectance = np.clip(raw * 1e-4, 0.0, 1.0)
        reflectance[nodata] = np.nan
        arrays.append(reflectance)
    return np.stack(arrays, axis=0), profile   # (6, H, W)


def build_temporal_stack(
    tile: str,
    t0: datetime,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
    """
    Build N_FRAMES-deep temporal HLS stack ending at t0.
    Frames: t0, t0-cadence, t0-2*cadence  (newest first in axis-1).

    Returns:
        hls_stack  float32  (6, N, H, W)  normalised bands
        idx_stack  float32  (3, N, H, W)  [NDVI, NBR, BSI]
        profile    rasterio profile dict
    """
    target_dates = [t0 - timedelta(days=i * TEMPORAL_CADENCE) for i in range(N_FRAMES)]

    frames_hls, frames_idx = [], []
    ref_profile = None
    ref_shape   = None

    for td in target_dates:
        scene_dir = find_hls_scene(tile, td)
        if scene_dir is None:
            frames_hls.append(None)
            frames_idx.append(None)
            continue

        bands, profile = load_hls_bands(scene_dir)
        if bands is None:
            frames_hls.append(None)
            frames_idx.append(None)
            continue

        H, W = bands.shape[1], bands.shape[2]
        if ref_shape is None:
            ref_shape   = (H, W)
            ref_profile = profile

        # Normalise: (x − mean) / std  (NaN → 0 before normalising)
        norm = (np.nan_to_num(bands, 0.0) - HLS_MEANS[:, None, None]) / (HLS_STDS[:, None, None] + 1e-6)
        frames_hls.append(norm.astype(np.float32))
        frames_idx.append(_spectral_indices(np.nan_to_num(bands, 0.0)))

    if all(f is None for f in frames_hls):
        return None, None, None

    # Fill missing frames with nearest valid frame
    valid_hls = next(f for f in frames_hls if f is not None)
    valid_idx = next(f for f in frames_idx if f is not None)
    frames_hls = [f if f is not None else np.zeros_like(valid_hls) for f in frames_hls]
    frames_idx = [f if f is not None else np.zeros_like(valid_idx) for f in frames_idx]

    hls_stack = np.stack(frames_hls, axis=1)   # (6, N, H, W)
    idx_stack = np.stack(frames_idx, axis=1)   # (3, N, H, W)
    return hls_stack, idx_stack, ref_profile


def _spectral_indices(bands: np.ndarray) -> np.ndarray:
    """
    Compute NDVI, NBR, BSI from 6-band float32 reflectance array.
    bands: (6, H, W)  [Blue, Green, Red, NIR, SWIR1, SWIR2]
    Returns: (3, H, W) clipped to [-1, 1]
    """
    blue, green, red, nir, swir1, swir2 = bands
    eps = 1e-6
    ndvi = (nir - red)   / (nir + red   + eps)
    nbr  = (nir - swir2) / (nir + swir2 + eps)
    bsi  = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)
    return np.clip(np.stack([ndvi, nbr, bsi], axis=0), -1.0, 1.0).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Weather Features
# ═════════════════════════════════════════════════════════════════════════════

_FWI_DF: Optional[pd.DataFrame] = None

def _fwi() -> pd.DataFrame:
    global _FWI_DF
    if _FWI_DF is None:
        try:
            df = pd.read_parquet(FWI_PATH)
        except ImportError:
            # pyarrow not installed — try reading from CSV if available
            csv_path = FWI_PATH.with_suffix(".csv")
            if csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                log.warning("pyarrow missing and no FWI CSV fallback — weather will be zeros")
                _FWI_DF = pd.DataFrame()
                return _FWI_DF
        df["date"] = pd.to_datetime(df["date"])
        _FWI_DF = df.set_index("date").sort_index()
    return _FWI_DF


def get_weather(date: datetime) -> np.ndarray:
    """Return normalised 10-dim weather vector for date."""
    df = _fwi()
    if df.empty:
        return np.zeros(len(WEATHER_COLS), dtype=np.float32)
    ts = pd.Timestamp(date.date())
    if ts in df.index:
        row = df.loc[ts]
    else:
        idx = df.index.get_indexer([ts], method="nearest")[0]
        row = df.iloc[idx]
    raw = row[WEATHER_COLS].values.astype(np.float32)
    return (raw - WEATHER_MEANS) / (WEATHER_STDS + 1e-6)


def get_weather_7d(date: datetime) -> np.ndarray:
    """Return normalised (10, 7) weather array: [date, date-1, ..., date-6]."""
    return np.stack([get_weather(date - timedelta(days=i)) for i in range(7)], axis=-1)


# ═════════════════════════════════════════════════════════════════════════════
# Terrain (warp-once cache)
# ═════════════════════════════════════════════════════════════════════════════

# In-memory cache: {tile: (4, H, W) float32 in UTM43N}
_TERRAIN_CACHE: Dict[str, np.ndarray] = {}
_TERRAIN_PROFILE: Dict[str, dict]     = {}

def _get_ref_profile(tile: str) -> Optional[dict]:
    """Get the rasterio profile of any available HLS scene for this tile."""
    for p in HLS_DIR.glob(f"{tile}/**/*.B02.tif"):
        with rasterio.open(p) as src:
            return src.profile.copy()
    return None


def _warp_terrain_to_tile(tile: str) -> Optional[np.ndarray]:
    """
    Warp elevation, slope, aspect rasters to the HLS tile CRS/grid.
    Returns (4, H, W) float32 array: [elev_norm, slope_norm, sin_asp, cos_asp]
    Caches result in _TERRAIN_CACHE.
    """
    if tile in _TERRAIN_CACHE:
        return _TERRAIN_CACHE[tile]

    ref_profile = _get_ref_profile(tile)
    if ref_profile is None:
        log.warning(f"No HLS reference profile for tile {tile}")
        return None

    dst_crs       = ref_profile["crs"]
    dst_transform = ref_profile["transform"]
    dst_H         = ref_profile["height"]
    dst_W         = ref_profile["width"]

    terrain_paths = {
        "elev":   TERRAIN_DIR / "dem_guna_clipped.tif",
        "slope":  TERRAIN_DIR / "slope_guna_30m.tif",
        "aspect": TERRAIN_DIR / "aspect_guna_30m.tif",
    }

    warped = {}
    for name, path in terrain_paths.items():
        if not path.exists():
            log.warning(f"Terrain file missing: {path}")
            warped[name] = np.zeros((dst_H, dst_W), dtype=np.float32)
            continue
        with rasterio.open(path) as src:
            dst_arr = np.zeros((dst_H, dst_W), dtype=np.float32)
            reproject(
                source      = rasterio.band(src, 1),
                destination = dst_arr,
                src_crs     = src.crs,
                dst_crs     = dst_crs,
                src_transform = src.transform,
                dst_transform = dst_transform,
                resampling  = Resampling.bilinear,
            )
        warped[name] = dst_arr

    # Normalise
    elev  = (warped["elev"]  - ELEV_MIN) / (ELEV_MAX - ELEV_MIN + 1e-6)
    slope = np.clip(warped["slope"] / SLOPE_MAX, 0.0, 1.0)
    sin_a = np.sin(np.radians(warped["aspect"])).astype(np.float32)
    cos_a = np.cos(np.radians(warped["aspect"])).astype(np.float32)

    terrain4 = np.stack(
        [elev.astype(np.float32), slope.astype(np.float32), sin_a, cos_a],
        axis=0
    )   # (4, H, W)

    _TERRAIN_CACHE[tile]   = terrain4
    _TERRAIN_PROFILE[tile] = ref_profile
    log.info(f"Warped terrain -> tile {tile}  shape={terrain4.shape}")
    return terrain4


# ═════════════════════════════════════════════════════════════════════════════
# Burn Age Feature
# ═════════════════════════════════════════════════════════════════════════════

def compute_burn_age(
    tile: str,
    query_date: datetime,
    full_H: int,
    full_W: int,
    max_years: int = 12,
) -> np.ndarray:
    """
    Compute 3-channel burn-age feature at full tile resolution.

    For each pixel: days since the most recent burn scar before query_date.
    Never-burned pixels get a large sentinel value (10 years).

    Channels:
      [0] recent_burn   exp(-age/180)              peaks just after fire
      [1] high_risk     exp(-((age-730)/365)^2)    peaks at ~2 years (regrowth)
      [2] mature        sigmoid((age-1825)/180)     rises after ~5 years

    Returns: (3, H, W) float32
    """
    age_days = np.full((full_H, full_W), np.nan, dtype=np.float32)
    cutoff   = query_date - timedelta(days=max_years * 365)

    # Collect all masks strictly before query_date, sorted newest-first
    all_masks = sorted(
        BURN_DIR.glob(f"{tile}/**/*.tif"),
        key=lambda p: str(p),
        reverse=True,
    )

    for mask_path in all_masks:
        # Parse date from path: .../tile/YYYY/MM/DD/burnscar_*.tif
        parts = mask_path.parts
        try:
            ti = next(i for i, p in enumerate(parts) if p == tile)
            mask_date = datetime(int(parts[ti+1]), int(parts[ti+2]), int(parts[ti+3]))
        except (ValueError, StopIteration, IndexError):
            continue

        if mask_date >= query_date or mask_date < cutoff:
            continue

        delta = (query_date - mask_date).days
        try:
            with rasterio.open(mask_path) as src:
                data = src.read(1).astype(np.int16)
        except Exception:
            continue

        if data.shape != (full_H, full_W):
            from scipy.ndimage import zoom
            zy = full_H / data.shape[0]
            zx = full_W / data.shape[1]
            data = zoom(data, (zy, zx), order=0).astype(np.int16)

        burned   = (data == 1)
        unfilled = np.isnan(age_days)
        age_days[unfilled & burned] = delta

        if not np.any(np.isnan(age_days)):
            break  # All pixels assigned

    # Sentinel for never-burned
    never    = np.isnan(age_days)
    age_fill = np.where(never, 3650.0, age_days)

    ch0 = np.exp(-age_fill / 180.0).astype(np.float32)
    ch1 = np.exp(-((age_fill - 730.0) ** 2) / (365.0 ** 2)).astype(np.float32)
    ch2 = (1.0 / (1.0 + np.exp(-(age_fill - 1825.0) / 180.0))).astype(np.float32)

    # Never-burned → zero across all channels
    ch0[never] = 0.0
    ch1[never] = 0.0
    ch2[never] = 0.0

    return np.stack([ch0, ch1, ch2], axis=0)   # (3, H, W)


# ═════════════════════════════════════════════════════════════════════════════
# Land Cover
# ═════════════════════════════════════════════════════════════════════════════

_LC_CACHE: Dict[str, np.ndarray] = {}

def load_landcover(tile: str, full_H: int, full_W: int) -> np.ndarray:
    """
    Load ESA WorldCover for the tile, one-hot encoded.
    Returns: (N_LC, H, W) float32
    """
    if tile in _LC_CACHE:
        return _LC_CACHE[tile]

    lc_path = LC_DIR / "worldcover_guna_raw.tif"
    if not lc_path.exists():
        # Fallback to any tif in land_cover dir
        candidates = [p for p in LC_DIR.glob("*.tif") if "forest_mask" not in p.name]
        lc_path = candidates[0] if candidates else LC_DIR / "forest_mask_guna_30m.tif"

    if not lc_path.exists():
        _LC_CACHE[tile] = np.zeros((N_LC, full_H, full_W), dtype=np.float32)
        return _LC_CACHE[tile]

    ref_profile = _get_ref_profile(tile)
    if ref_profile is None:
        _LC_CACHE[tile] = np.zeros((N_LC, full_H, full_W), dtype=np.float32)
        return _LC_CACHE[tile]

    # Warp WorldCover to HLS tile grid
    dst_arr = np.zeros((full_H, full_W), dtype=np.uint8)
    with rasterio.open(lc_path) as src:
        reproject(
            source        = rasterio.band(src, 1),
            destination   = dst_arr,
            src_crs       = src.crs,
            dst_crs       = ref_profile["crs"],
            src_transform = src.transform,
            dst_transform = ref_profile["transform"],
            resampling    = Resampling.nearest,
        )

    # One-hot encode
    one_hot = np.zeros((N_LC, full_H, full_W), dtype=np.float32)
    for lc_val, idx in LC_MAP.items():
        one_hot[idx] = (dst_arr == lc_val).astype(np.float32)

    _LC_CACHE[tile] = one_hot
    return one_hot


# ═════════════════════════════════════════════════════════════════════════════
# Label Loading
# ═════════════════════════════════════════════════════════════════════════════

def load_label(tile: str, fire_date: datetime) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load tiered label and weight rasters from LABEL_DIR.
    Returns: (label uint8 [H,W], weight float32 [H,W]) or (None, None).
    """
    dt  = fire_date
    tag = f"{dt.strftime('%Y%m%d')}_{tile}"
    lbl_path = LABEL_DIR / tile / str(dt.year) / f"{dt.month:02d}" / f"label_{tag}.tif"
    wt_path  = LABEL_DIR / tile / str(dt.year) / f"{dt.month:02d}" / f"weight_{tag}.tif"

    if not lbl_path.exists():
        return None, None

    with rasterio.open(lbl_path) as src:
        raw = src.read(1)

    # Handle mixed label types:
    #   float32 = real dNBR from compute_dnbr_labels.py (0.0=bg, >0=severity)
    #   uint8   = old binary from build_fire_labels.py (0=bg, 1=fire, 255=nodata)
    if raw.dtype == np.uint8:
        # Convert binary to proxy dNBR: fire pixels get median dNBR (0.15)
        label = np.zeros_like(raw, dtype=np.float32)
        label[raw == 1] = 0.15   # proxy severity for binary-only events
        label[raw == 255] = np.nan  # nodata
    else:
        label = raw.astype(np.float32)

    if wt_path.exists():
        with rasterio.open(wt_path) as src:
            weight = src.read(1).astype(np.float32)
    else:
        weight = np.where(label > 0, 1.0, 0.1).astype(np.float32)

    return label, weight


# ═════════════════════════════════════════════════════════════════════════════
# Patch Window Selection
# ═════════════════════════════════════════════════════════════════════════════

def grid_windows(
    label: np.ndarray,
    full_H: int,
    full_W: int,
    stride: int = GRID_STRIDE,
    bg_sample_rate: float = BG_SAMPLE_RATE,
    min_fire_px: int = MIN_FIRE_PX,
) -> List[Tuple[int, int, bool]]:
    """
    Pixel-based grid extraction: scan the ENTIRE tile and return patches.

    Returns list of (row_start, col_start, is_fire) for ALL fire patches
    (dNBR > 0 with enough fire pixels) plus a random sample of background patches.

    With stride=112 on 224px patches (50% overlap), a 3660x3660 tile yields
    ~1024 grid positions. Fire patches are all kept; background patches are
    sampled at bg_sample_rate to control dataset size.
    """
    fire_windows = []
    bg_candidates = []

    for r0 in range(0, max(1, full_H - PATCH_H + 1), stride):
        for c0 in range(0, max(1, full_W - PATCH_W + 1), stride):
            patch = label[r0 : r0 + PATCH_H, c0 : c0 + PATCH_W]

            # For regression: fire pixels have dNBR > 0
            valid = np.isfinite(patch)
            fire_px = int((patch[valid] > 0).sum()) if valid.any() else 0

            if fire_px >= min_fire_px:
                fire_windows.append((r0, c0, True))
            elif valid.sum() > PATCH_H * PATCH_W * 0.5:
                bg_candidates.append((r0, c0, False))

    # Randomly sample background patches
    np.random.shuffle(bg_candidates)
    n_bg = max(1, int(len(bg_candidates) * bg_sample_rate))
    bg_windows = bg_candidates[:n_bg]

    return fire_windows + bg_windows


# ═════════════════════════════════════════════════════════════════════════════
# Save Patch
# ═════════════════════════════════════════════════════════════════════════════

def save_patch(
    out_path: Path,
    hls: np.ndarray,
    indices: np.ndarray,
    weather: np.ndarray,
    weather_7d: np.ndarray,
    terrain: np.ndarray,
    burn_age: np.ndarray,
    landcover: np.ndarray,
    label: np.ndarray,
    weight: np.ndarray,
    meta: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        hls        = hls,          # (6, T, H, W)
        indices    = indices,      # (3, T, H, W)
        weather    = weather,      # (10,)
        weather_7d = weather_7d,   # (10, 7)
        terrain    = terrain,      # (4, H, W)
        burn_age   = burn_age,     # (3, H, W)
        landcover  = landcover,    # (11, H, W)
        label      = label,        # (H, W)
        weight     = weight,       # (H, W)
        meta       = json.dumps(meta),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Negative Sample Generator
# ═════════════════════════════════════════════════════════════════════════════

def generate_negatives(
    fire_df: pd.DataFrame,
    n_per_pos: int,
    existing_manifest: List[dict],
) -> List[dict]:
    """
    Generate negative sample specifications.
    Dates chosen from fire-season months (Feb-June) but >30 days from any fire.
    """
    tiles      = fire_df["tile"].unique().tolist()
    fire_dates = set(pd.to_datetime(fire_df["fire_date"]).dt.date.tolist())
    target     = len(existing_manifest) * n_per_pos
    negatives  = []
    attempts   = 0
    np.random.seed(42)

    while len(negatives) < target and attempts < target * 30:
        attempts += 1
        tile  = str(np.random.choice(tiles))
        year  = int(np.random.choice(range(2013, 2025)))
        month = int(np.random.choice([2, 3, 4, 5, 6]))
        day   = int(np.random.randint(1, 28))
        try:
            dt = datetime(year, month, day)
        except ValueError:
            continue

        # Reject if too close to any fire event
        too_close = any(abs((dt.date() - fd).days) < 30 for fd in fire_dates)
        if too_close:
            continue

        # Reject if no HLS scene available
        t_input = dt - timedelta(days=PRED_HORIZON)
        if find_hls_scene(tile, t_input) is None:
            continue

        negatives.append({"tile": tile, "date": dt, "neg_idx": len(negatives)})

    log.info(f"Generated {len(negatives)} negative sample specs ({attempts} attempts)")
    return negatives


# ═════════════════════════════════════════════════════════════════════════════
# Main Build
# ═════════════════════════════════════════════════════════════════════════════

def build_dataset(max_events: Optional[int] = None, pred_horizon: int = PRED_HORIZON,
                   season_filter: bool = True) -> None:
    global PRED_HORIZON
    PRED_HORIZON = pred_horizon

    # ── Load event table ─────────────────────────────────────────────────
    df = pd.read_csv(LINKS_CSV)
    df["fire_date"]  = pd.to_datetime(df["fire_date"])
    df["scene_date"] = pd.to_datetime(df["scene_date"])

    def tier(row) -> str:
        conf = str(row["confidence"]).strip().upper()
        hit  = bool(row["spatial_hit"])
        if conf == "NONE":                                          return "EXCLUDE"
        if conf in ("HIGH", "MEDIUM-HIGH") and hit:                return "GOLD"
        if conf == "MEDIUM" and hit:                               return "SILVER"
        if conf in ("HIGH", "MEDIUM-HIGH", "MEDIUM") and not hit:  return "BRONZE"
        return "VIIRS_ONLY"

    df["tier"]  = df.apply(tier, axis=1)
    df["split"] = df["fire_date"].dt.year.map(lambda y: SPLIT_MAP.get(y, "train"))
    df = df[df["tier"] != "EXCLUDE"].reset_index(drop=True)

    # ── Fire season filter: March 1 - June 15 only ──────────────────────
    # Out-of-season fires (Oct-Feb) have unreliable dNBR labels due to
    # tropical deciduous senescence and are operationally irrelevant.
    if season_filter:
        pre_filter = len(df)
        month = df["fire_date"].dt.month
        day   = df["fire_date"].dt.day
        in_season = (month >= 3) & ((month <= 5) | ((month == 6) & (day <= 15)))
        df = df[in_season].reset_index(drop=True)
        log.info(f"Fire season filter (Mar 1 - Jun 15): {pre_filter} -> {len(df)} events")

    # Deduplicate by (tile, fire_date) — multiple fire_links entries can share
    # the same tile+date (same HLS scene, same label raster). Process each
    # unique (tile, date) once; keep the highest-tier entry for metadata.
    tier_rank = {"GOLD": 4, "SILVER": 3, "BRONZE": 2, "VIIRS_ONLY": 1}
    df["_tier_rank"] = df["tier"].map(tier_rank).fillna(0)
    df = df.sort_values("_tier_rank", ascending=False).drop_duplicates(
        subset=["tile", "fire_date"], keep="first"
    ).drop(columns=["_tier_rank"]).reset_index(drop=True)

    if max_events:
        df = df.iloc[:max_events]
        log.info(f"Capped to {max_events} events (smoke-test mode)")

    log.info(f"Processing {len(df)} unique (tile, fire_date) events  (horizon={pred_horizon}d)")
    log.info(f"Tier dist:\n{df['tier'].value_counts().to_string()}")

    manifest_rows: List[dict] = []
    patch_counts = {t: 0 for t in ["GOLD", "SILVER", "BRONZE", "VIIRS_ONLY", "negative"]}
    skipped = 0

    for idx, row in df.iterrows():
        tile      = str(row["tile"])
        fire_date = row["fire_date"].to_pydatetime()
        t_input   = fire_date - timedelta(days=pred_horizon)
        split     = row["split"]
        event_tier = row["tier"]

        log.info(
            f"[{idx+1:3d}/{len(df)}] {tile}  fire={fire_date.date()}  "
            f"input={t_input.date()}  tier={event_tier}"
        )

        # ── HLS temporal stack ────────────────────────────────────────────
        hls_stack, idx_stack, profile = build_temporal_stack(tile, t_input)
        if hls_stack is None:
            log.warning(f"  No HLS scenes found near {t_input.date()} — skip")
            skipped += 1
            continue

        full_H, full_W = hls_stack.shape[2], hls_stack.shape[3]

        # ── Weather (use t_input, NOT fire_date — avoids data leak) ──────
        # At inference we only know weather up to prediction date, not the
        # actual fire date which is PRED_HORIZON days in the future.
        weather    = get_weather(t_input)
        weather_7d = get_weather_7d(t_input)

        # ── Terrain (warped, full tile) ───────────────────────────────────
        terrain_full = _warp_terrain_to_tile(tile)
        if terrain_full is None:
            terrain_full = np.zeros((4, full_H, full_W), dtype=np.float32)

        # ── Burn age (full tile) ──────────────────────────────────────────
        burn_age_full = compute_burn_age(tile, fire_date, full_H, full_W)

        # ── Land cover (full tile) ────────────────────────────────────────
        lc_full = load_landcover(tile, full_H, full_W)

        # ── Label ─────────────────────────────────────────────────────────
        label_full, weight_full = load_label(tile, fire_date)
        if label_full is None:
            # No pre-built label: fire event skipped by build_fire_labels.py
            log.warning(f"  Label not found for {tile} {fire_date.date()} — skip")
            skipped += 1
            continue

        # ── Grid-based patch extraction (pixel-level coverage) ────────────
        # Scan entire tile: keep ALL fire patches + sampled background patches.
        # This gives the model diverse spatial context from the full landscape.
        windows = grid_windows(label_full, full_H, full_W)
        n_fire_w = sum(1 for _, _, is_f in windows if is_f)
        n_bg_w   = len(windows) - n_fire_w
        log.info(f"  Grid: {n_fire_w} fire + {n_bg_w} background patches")

        for win_idx, (r0, c0, is_fire) in enumerate(windows):
            r1 = r0 + PATCH_H
            c1 = c0 + PATCH_W

            lbl_patch = label_full [r0:r1, c0:c1]
            wt_patch  = weight_full[r0:r1, c0:c1]

            # Count fire pixels for manifest
            valid_lbl = lbl_patch[np.isfinite(lbl_patch)]
            fire_px = int((valid_lbl > 0).sum()) if len(valid_lbl) > 0 else 0

            hls_patch  = hls_stack  [:, :, r0:r1, c0:c1]
            idx_patch  = idx_stack  [:, :, r0:r1, c0:c1]
            terr_patch = terrain_full[:, r0:r1, c0:c1]
            age_patch  = burn_age_full[:, r0:r1, c0:c1]
            lc_patch   = lc_full     [:, r0:r1, c0:c1]

            tag  = "f" if is_fire else "bg"
            pid  = f"{tile}_{fire_date.strftime('%Y%m%d')}_{tag}{win_idx:04d}"
            out_path = OUT_DIR / split / f"patch_{pid}.npz"

            meta = {
                "patch_id":   pid,
                "tile":       tile,
                "fire_date":  fire_date.isoformat(),
                "t_input":    t_input.isoformat(),
                "split":      split,
                "tier":       event_tier if is_fire else "NEGATIVE",
                "fire_pixels":fire_px,
                "window":     [r0, c0, PATCH_H, PATCH_W],
                "pred_horizon": pred_horizon,
            }

            save_patch(out_path, hls_patch, idx_patch, weather, weather_7d,
                       terr_patch, age_patch, lc_patch, lbl_patch, wt_patch, meta)

            manifest_rows.append({
                "path":       str(out_path.relative_to(ROOT)),
                "patch_id":   pid,
                "tile":       tile,
                "fire_date":  fire_date.date(),
                "split":      split,
                "tier":       event_tier if is_fire else "NEGATIVE",
                "fire_pixels":fire_px,
                "is_fire":    1 if is_fire else 0,
            })
            if is_fire:
                patch_counts[event_tier] = patch_counts.get(event_tier, 0) + 1
            else:
                patch_counts["negative"] += 1

    n_pos = sum(v for k, v in patch_counts.items() if k != "negative")
    n_neg = patch_counts["negative"]
    log.info(f"\nGrid extraction done: {n_pos} fire + {n_neg} background patches")
    log.info(f"Skipped events: {skipped}")

    # NOTE: Background patches are already included by grid_windows().
    # No separate negative generation pass needed.

    # ── Manifest ─────────────────────────────────────────────────────────
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = OUT_DIR / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DATASET BUILD COMPLETE")
    print("=" * 65)
    print(f"\nPatches -> {OUT_DIR}")
    print(f"Manifest -> {manifest_path}")
    print(f"\nLabel tier breakdown:")
    for t, c in patch_counts.items():
        print(f"  {t:12s}: {c:5d} patches")
    total = sum(patch_counts.values())
    print(f"  {'TOTAL':12s}: {total:5d} patches")
    print(f"\nSplit summary:")
    for split, grp in manifest.groupby("split"):
        nf = int(grp["is_fire"].sum())
        nn = len(grp) - nf
        print(f"  {split:6s}: {len(grp):5d} total  ({nf} fire + {nn} negative)")
    print("=" * 65)


# ═════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build VanAgni training patches")
    parser.add_argument("--pred-horizon", type=int, default=PRED_HORIZON,
                        help=f"Days before fire_date for input imagery (default: {PRED_HORIZON})")
    parser.add_argument("--max-events",   type=int, default=None,
                        help="Limit events processed (smoke-test)")
    parser.add_argument("--no-season-filter", action="store_true",
                        help="Include fires outside Mar 1 - Jun 15")
    parser.add_argument("--out-dir",      type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sp in ["train", "val", "test"]:
        (OUT_DIR / sp).mkdir(exist_ok=True)

    build_dataset(max_events=args.max_events, pred_horizon=args.pred_horizon,
                  season_filter=not args.no_season_filter)
