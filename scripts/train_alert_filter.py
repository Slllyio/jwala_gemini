"""
Train Alert Filter — Binary LightGBM: Real Change vs Noise
============================================================

Trains a binary classifier on alert-firing pixels.
  Positive (1) = TP  — alert confirmed by annual GT
  Negative (0) = FP + FP_contra  — noise / phenology

Supports two raster formats:
  LEGACY (10-band): 8 DW deltas + 2 cloud → 30 features
  ENRICHED (44-band): 8 raw DW before + 8 after + 8 delta + 2 cloud
                      + 6 S2 spectral + 12 S2 raw → 50 features

Usage:
    python scripts/train_alert_filter.py \\
        --data-dir data/ground_truth/HAMEERPUR \\
        --out-dir outputs/alert_filter
"""

# ── PROJ/GDAL env fix ──
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()
del _os, _sys
# ────────────────────────

import argparse
from datetime import datetime
import json
import logging
import os
import re
import time
from glob import glob
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DW_BANDS = ["water", "trees", "grass", "flooded_veg",
            "crops", "shrub_scrub", "built", "bare"]
TREES_IDX = 1
N_BANDS   = 8

ALERT_THRESH = 0.10
GT_THRESH    = 0.10
SEVERE_LOSS_THRESH = 0.30  # tree loss so severe it's alert-worthy regardless
ANTHROPO_GAIN_BANDS = [4, 6]  # crops=4, built=6 in DW band order

# Semantic mapping: DW band index -> change class
# 0=water, 1=trees, 2=grass, 3=flooded_veg, 4=crops, 5=shrub, 6=built, 7=bare
GAIN_BAND_TO_CLASS = {
    4: 1,  # Crops -> Encroachment
    6: 2,  # Built -> Built Expansion
    2: 3,  # Grass -> Degradation (phenology)
    5: 4,  # Shrub -> Shrub/Scrub (phenology)
    0: 6,  # Water -> Other
    3: 6,  # Flooded veg -> Other
    7: 6,  # Bare -> Other
}

SEASON_MAP = {
    1: "Winter", 2: "Winter",
    3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
    12: "Winter",
}

# ── Band indices for enriched (44-band) rasters ─────────────────────────────
# Bands 0-7:   DW raw probabilities BEFORE
# Bands 8-15:  DW raw probabilities AFTER
# Bands 16-23: DW delta (after - before)
# Bands 24-25: Cloud prob (before, after)
# Bands 26-28: S2 spectral BEFORE (ndvi, nbr, ndmi)
# Bands 29-31: S2 spectral AFTER (ndvi, nbr, ndmi)
# Bands 32-37: S2 raw reflectance BEFORE (B2, B3, B4, B8, B11, B12)
# Bands 38-43: S2 raw reflectance AFTER (B2, B3, B4, B8, B11, B12)
ENRICHED_DW_BEFORE = slice(0, 8)
ENRICHED_DW_AFTER  = slice(8, 16)
ENRICHED_DELTA     = slice(16, 24)
ENRICHED_CLOUD     = slice(24, 26)
ENRICHED_SPEC_BEFORE = slice(26, 29)  # ndvi, nbr, ndmi
ENRICHED_SPEC_AFTER  = slice(29, 32)
ENRICHED_S2_BEFORE = slice(32, 38)    # B2, B3, B4, B8, B11, B12
ENRICHED_S2_AFTER  = slice(38, 44)

# Feature names for LEGACY (30 features)
FEATURE_NAMES_LEGACY = [
    # Band deltas (8)
    "delta_water", "delta_trees", "delta_grass", "delta_flooded_veg",
    "delta_crops", "delta_shrub_scrub", "delta_built", "delta_bare",
    # Cloud (3)
    "cloud_before", "cloud_after", "cloud_worst",
    # Spatial (5)
    "n_neighbors_alert", "mean_neighbor_trees", "std_neighbor_trees",
    "delta_trees_5x5_mean", "delta_trees_7x7_mean",
    # Cross-band (5)
    "trees_crops_anti", "trees_bare_anti",
    "dominant_gain_band", "band_diversity",
    "tree_loss_ratio",
    # Magnitude (2)
    "abs_delta_trees", "abs_max_band",
    # Temporal (4)
    "cum_alert_count", "cum_delta_trees",
    "max_loss_so_far", "windows_since_first_alert",
    # Metadata (3)
    "month", "is_monsoon", "is_winter",
]

# Feature names for ENRICHED (84 features total)
FEATURE_NAMES_ENRICHED = [
    # Band deltas (8)
    "delta_water", "delta_trees", "delta_grass", "delta_flooded_veg",
    "delta_crops", "delta_shrub_scrub", "delta_built", "delta_bare",
    # Raw DW before (8)
    "water_before", "trees_before", "grass_before", "flooded_veg_before",
    "crops_before", "shrub_scrub_before", "built_before", "bare_before",
    # Raw DW after (8)
    "water_after", "trees_after", "grass_after", "flooded_veg_after",
    "crops_after", "shrub_scrub_after", "built_after", "bare_after",
    # S2 spectral (6)
    "ndvi_before", "nbr_before", "ndmi_before",
    "ndvi_after", "nbr_after", "ndmi_after",
    # S2 raw reflectance (4 key bands only — B4_Red, B8_NIR, B11_SWIR1, B12_SWIR2)
    "red_before", "nir_before", "swir1_before", "swir2_before",
    "red_after", "nir_after", "swir1_after", "swir2_after",
    # Derived spectral deltas (3)
    "delta_ndvi", "delta_nbr", "delta_ndmi",
    # Cloud (3)
    "cloud_before", "cloud_after", "cloud_worst",
    # Spatial (5)
    "n_neighbors_alert", "mean_neighbor_trees", "std_neighbor_trees",
    "delta_trees_5x5_mean", "delta_trees_7x7_mean",
    # Cross-band (6)
    "trees_crops_anti", "trees_bare_anti",
    "dominant_gain_band", "dominant_gain_class", "band_diversity",
    "tree_loss_ratio",
    # Magnitude (2)
    "abs_delta_trees", "abs_max_band",
    # ── v5 new features (15) ─────────────────────────────────────────
    # Confidence & relative (3)
    "relative_tree_loss", "dw_entropy_before", "dw_entropy_after",
    # Semantic transitions (4)
    "dw_before_class", "dominant_gain_value", "gain_gap", "transition_type",
    # Season × class interactions (3)
    "monsoon_x_grass", "monsoon_x_shrub", "winter_x_crops",
    # Spectral indices (3)
    "ndbi_before", "ndbi_after", "delta_ndbi",
    # Temporal window (2)
    "window_duration_days", "alert_rate",
    # ── end v5 ───────────────────────────────────────────────────────
    # Temporal DW (4)
    "cum_alert_count", "cum_delta_trees",
    "max_loss_so_far", "windows_since_first_alert",
    # Temporal NDVI trajectory (5)
    "cum_delta_ndvi", "max_ndvi_loss_so_far",
    "ndvi_at_first_alert", "cum_delta_nbr",
    "ndvi_recovery_flag",
    # Metadata (3)
    "month", "is_monsoon", "is_winter",
]

# Will be set at runtime based on detected raster format
FEATURE_NAMES = FEATURE_NAMES_LEGACY  # default, overridden if enriched


# ── Loading ───────────────────────────────────────────────────────────────────

def parse_dates(filename: str):
    m = re.search(r"(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})", filename)
    if m:
        return m.group(1), m.group(2)
    return "unknown", "unknown"


def load_alert_stack(data_dir: str):
    """Load alert rasters — prefers enriched (44-band) over legacy (10-band)."""
    global FEATURE_NAMES

    # Try enriched first
    pattern = os.path.join(data_dir, "alerts", "alert_enriched_*.tif")
    files = sorted(glob(pattern))
    enriched = len(files) > 0

    if not files:
        # Fall back to legacy
        pattern = os.path.join(data_dir, "alerts", "alert_delta_*.tif")
        files = sorted(glob(pattern))
    if not files:
        pattern = os.path.join(data_dir, "alert_delta_*.tif")
        files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No alert rasters in {data_dir}")

    arrays, meta_list = [], []
    profile = None
    for f in files:
        with rasterio.open(f) as ds:
            arrays.append(ds.read())
            if profile is None:
                profile = ds.profile.copy()
                n_bands = ds.count
        d_before, d_after = parse_dates(Path(f).stem)
        month = int(d_before.split("-")[1]) if d_before != "unknown" else 1
        meta_list.append({
            "idx": len(meta_list), "file": Path(f).stem,
            "date_before": d_before, "date_after": d_after,
            "month": month, "season": SEASON_MAP.get(month, "Unknown"),
        })

    stack = np.stack(arrays, axis=0)

    # Set feature names based on detected format
    if n_bands >= 44:
        FEATURE_NAMES = FEATURE_NAMES_ENRICHED
        log.info(f"Alert stack: {stack.shape} ({len(files)} windows) — ENRICHED ({n_bands} bands, {len(FEATURE_NAMES)} features)")
    else:
        FEATURE_NAMES = FEATURE_NAMES_LEGACY
        log.info(f"Alert stack: {stack.shape} ({len(files)} windows) — LEGACY ({n_bands} bands, {len(FEATURE_NAMES)} features)")

    return stack, meta_list, profile


def load_gt(data_dir: str):
    files = glob(os.path.join(data_dir, "gt_delta_*.tif"))
    if not files:
        raise FileNotFoundError(f"No GT raster in {data_dir}")
    with rasterio.open(files[0]) as ds:
        gt = ds.read()
    log.info(f"GT: {gt.shape}")
    return gt


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features_for_window(stack: np.ndarray, w: int,
                                cum_alert_count: np.ndarray,
                                window_meta: dict, H: int, W: int):
    """
    Extract per-window features for ALL pixels in window w.
    Enriched (44-band): 46 base features + 4 temporal = 50 total
    Legacy (10-band):   26 base features + 4 temporal = 30 total
    Temporal features are appended externally by label_windows / stack_windows.
    """
    n_pix = H * W
    n_bands_in_stack = stack.shape[1]
    is_enriched = n_bands_in_stack >= 44

    if is_enriched:
        # ── Extract from ENRICHED 44-band raster ─────────────────────
        dw_before = stack[w, ENRICHED_DW_BEFORE, :, :].astype(np.float32)  # (8, H, W)
        dw_after  = stack[w, ENRICHED_DW_AFTER, :, :].astype(np.float32)
        deltas    = stack[w, ENRICHED_DELTA, :, :].astype(np.float32)
        cld_raw   = stack[w, ENRICHED_CLOUD, :, :].astype(np.float32)  # (2, H, W)
        spec_before = stack[w, ENRICHED_SPEC_BEFORE, :, :].astype(np.float32)  # (3, H, W)
        spec_after  = stack[w, ENRICHED_SPEC_AFTER, :, :].astype(np.float32)
        s2_before = stack[w, ENRICHED_S2_BEFORE, :, :].astype(np.float32)  # (6, H, W)
        s2_after  = stack[w, ENRICHED_S2_AFTER, :, :].astype(np.float32)

        dw_before_flat = dw_before.reshape(N_BANDS, n_pix)
        dw_after_flat  = dw_after.reshape(N_BANDS, n_pix)
        delta_flat     = deltas.reshape(N_BANDS, n_pix)
        cld_before = cld_raw[0].ravel()
        cld_after  = cld_raw[1].ravel()

        spec_before_flat = spec_before.reshape(3, n_pix)  # ndvi, nbr, ndmi
        spec_after_flat  = spec_after.reshape(3, n_pix)
        s2_before_flat = s2_before.reshape(6, n_pix)  # B2, B3, B4, B8, B11, B12
        s2_after_flat  = s2_after.reshape(6, n_pix)
    else:
        # ── Extract from LEGACY 10-band raster ───────────────────────
        deltas = stack[w, :N_BANDS, :, :].astype(np.float32)
        delta_flat = deltas.reshape(N_BANDS, n_pix)
        if n_bands_in_stack >= 10:
            cld_before = stack[w, 8, :, :].ravel().astype(np.float32)
            cld_after  = stack[w, 9, :, :].ravel().astype(np.float32)
        else:
            cld_before = np.zeros(n_pix, dtype=np.float32)
            cld_after  = np.zeros(n_pix, dtype=np.float32)

    cld_worst = np.maximum(cld_before, cld_after)

    # ── Spatial context (5) — 3×3, 5×5, 7×7 neighborhood ────────────
    trees_2d = deltas[TREES_IDX]  # (H, W)
    alert_mask_2d = (trees_2d <= -ALERT_THRESH).astype(np.float32)  # loss only

    neighbor_sum = uniform_filter(alert_mask_2d, size=3, mode="constant") * 9
    n_neighbors = (neighbor_sum - alert_mask_2d).ravel()
    n_neighbors = np.clip(n_neighbors, 0, 8)

    mean_nb_trees = uniform_filter(trees_2d, size=3, mode="constant").ravel()
    mean_sq = uniform_filter(trees_2d**2, size=3, mode="constant").ravel()
    std_nb_trees = np.sqrt(np.maximum(mean_sq - mean_nb_trees**2, 0))

    delta_trees_5x5 = uniform_filter(trees_2d, size=5, mode="constant").ravel()
    delta_trees_7x7 = uniform_filter(trees_2d, size=7, mode="constant").ravel()

    # ── Cross-band interactions (5) ──────────────────────────────────
    trees_delta = delta_flat[TREES_IDX]
    crops_delta = delta_flat[4]
    bare_delta  = delta_flat[7]

    trees_crops_anti = ((trees_delta < 0) & (crops_delta > 0)).astype(np.float32)
    trees_bare_anti  = ((trees_delta < 0) & (bare_delta > 0)).astype(np.float32)

    gain_deltas = delta_flat.copy()
    gain_deltas[TREES_IDX] = -999
    dominant_gain = np.argmax(gain_deltas, axis=0).astype(np.float32)
    band_div = (np.abs(delta_flat) > ALERT_THRESH).sum(axis=0).astype(np.float32)

    # Semantic change class from dominant gain band
    band_to_class = np.array([GAIN_BAND_TO_CLASS.get(b, 6) for b in range(N_BANDS)])
    dom_gain_idx = np.argmax(gain_deltas, axis=0)
    dom_gain_val = np.max(gain_deltas, axis=0)
    dominant_gain_class = band_to_class[dom_gain_idx].astype(np.float32)
    # Pure Tree Loss: severe tree drop but dominant gain is very small
    pure_loss = (trees_delta <= -SEVERE_LOSS_THRESH) & (dom_gain_val < 0.05)
    dominant_gain_class[pure_loss] = 5.0  # Pure Tree Loss

    total_volatility = np.abs(trees_delta) + np.abs(crops_delta) + np.abs(bare_delta)
    tree_loss_ratio = np.abs(trees_delta) / (total_volatility + 1e-6)

    # ── Magnitude (2) ────────────────────────────────────────────────
    abs_trees = np.abs(trees_delta)
    abs_max = np.max(np.abs(delta_flat), axis=0)

    # ── Metadata (3) ─────────────────────────────────────────────────
    month = np.full(n_pix, window_meta["month"], dtype=np.float32)
    is_monsoon = np.full(n_pix,
                         1.0 if window_meta["season"] == "Monsoon" else 0.0,
                         dtype=np.float32)
    is_winter = np.full(n_pix,
                        1.0 if window_meta["season"] == "Winter" else 0.0,
                        dtype=np.float32)

    if is_enriched:
        # ── Enriched assembly ─────────────────────────────────────────
        # Spectral deltas
        delta_ndvi = spec_after_flat[0] - spec_before_flat[0]
        delta_nbr  = spec_after_flat[1] - spec_before_flat[1]
        delta_ndmi = spec_after_flat[2] - spec_before_flat[2]

        # ── v5: Confidence & relative features (3) ────────────────────
        trees_before_val = dw_before_flat[TREES_IDX]  # (n_pix,)
        relative_tree_loss = trees_delta / (trees_before_val + 1e-6)

        # DW entropy: -Σ p_i * log(p_i)
        eps = 1e-8
        p_before = dw_before_flat + eps  # (8, n_pix)
        p_after  = dw_after_flat  + eps
        dw_entropy_before = -(p_before * np.log(p_before)).sum(axis=0)
        dw_entropy_after  = -(p_after  * np.log(p_after)).sum(axis=0)

        # ── v5: Semantic transition features (4) ──────────────────────
        dw_before_class = np.argmax(dw_before_flat, axis=0).astype(np.float32)
        dominant_gain_value = dom_gain_val.astype(np.float32)
        # Gain gap: top gain vs runner-up
        sorted_gains = np.sort(gain_deltas, axis=0)  # ascending
        gain_gap = (sorted_gains[-1] - sorted_gains[-2]).astype(np.float32)
        # Transition type: before_class * 10 + gain_class
        transition_type = (dw_before_class * 10 + dominant_gain_class).astype(np.float32)

        # ── v5: Season × class interactions (3) ───────────────────────
        monsoon_x_grass = is_monsoon * (dominant_gain_class == 3).astype(np.float32)
        monsoon_x_shrub = is_monsoon * (dominant_gain_class == 4).astype(np.float32)
        winter_x_crops  = is_winter  * (dominant_gain_class == 1).astype(np.float32)

        # ── v5: Spectral indices NDBI (3) ─────────────────────────────
        # s2_before_flat: B2=0, B3=1, B4=2, B8=3, B11=4, B12=5
        nir_bef = s2_before_flat[3] + 1e-6
        swir1_bef = s2_before_flat[4] + 1e-6
        nir_aft = s2_after_flat[3] + 1e-6
        swir1_aft = s2_after_flat[4] + 1e-6
        ndbi_before = (swir1_bef - nir_bef) / (swir1_bef + nir_bef)
        ndbi_after  = (swir1_aft - nir_aft) / (swir1_aft + nir_aft)
        delta_ndbi  = ndbi_after - ndbi_before

        # ── v5: Window duration (1) ───────────────────────────────────
        try:
            d_bef = datetime.strptime(window_meta["date_before"], "%Y-%m-%d")
            d_aft = datetime.strptime(window_meta["date_after"], "%Y-%m-%d")
            w_dur = float((d_aft - d_bef).days)
        except (ValueError, KeyError):
            w_dur = 30.0  # fallback
        window_duration_days = np.full(n_pix, w_dur, dtype=np.float32)

        # ── v5: Alert rate (1) — cum_count / (wins_since + 1) ─────────
        wins_since_first = np.where(
            cum_alert_count > 0,
            np.maximum(cum_alert_count - 1, 0).astype(np.float32),
            np.float32(0)
        )
        alert_rate = cum_alert_count.astype(np.float32) / (wins_since_first + 1.0)

        # S2 raw: pick 4 key bands (B4=Red, B8=NIR, B11=SWIR1, B12=SWIR2)
        # indices in s2_before_flat: B2=0, B3=1, B4=2, B8=3, B11=4, B12=5
        features = np.column_stack([
            # Band deltas (8)
            delta_flat[0], delta_flat[1], delta_flat[2], delta_flat[3],
            delta_flat[4], delta_flat[5], delta_flat[6], delta_flat[7],
            # Raw DW before (8)
            dw_before_flat[0], dw_before_flat[1], dw_before_flat[2], dw_before_flat[3],
            dw_before_flat[4], dw_before_flat[5], dw_before_flat[6], dw_before_flat[7],
            # Raw DW after (8)
            dw_after_flat[0], dw_after_flat[1], dw_after_flat[2], dw_after_flat[3],
            dw_after_flat[4], dw_after_flat[5], dw_after_flat[6], dw_after_flat[7],
            # S2 spectral (6)
            spec_before_flat[0], spec_before_flat[1], spec_before_flat[2],
            spec_after_flat[0], spec_after_flat[1], spec_after_flat[2],
            # S2 raw reflectance: Red, NIR, SWIR1, SWIR2 (8)
            s2_before_flat[2], s2_before_flat[3], s2_before_flat[4], s2_before_flat[5],
            s2_after_flat[2], s2_after_flat[3], s2_after_flat[4], s2_after_flat[5],
            # Spectral deltas (3)
            delta_ndvi, delta_nbr, delta_ndmi,
            # Cloud (3)
            cld_before, cld_after, cld_worst,
            # Spatial (5)
            n_neighbors, mean_nb_trees, std_nb_trees,
            delta_trees_5x5, delta_trees_7x7,
            # Cross-band (6)
            trees_crops_anti, trees_bare_anti, dominant_gain, dominant_gain_class,
            band_div, tree_loss_ratio,
            # Magnitude (2)
            abs_trees, abs_max,
            # ── v5 new features (15) ─────────────────────────────────
            relative_tree_loss, dw_entropy_before, dw_entropy_after,
            dw_before_class, dominant_gain_value, gain_gap, transition_type,
            monsoon_x_grass, monsoon_x_shrub, winter_x_crops,
            ndbi_before, ndbi_after, delta_ndbi,
            window_duration_days, alert_rate,
            # Metadata (3)
            month, is_monsoon, is_winter,
        ])
    else:
        # ── Legacy assembly (n_pix, 26) ──────────────────────────────
        features = np.column_stack([
            # Band deltas (8)
            delta_flat[0], delta_flat[1], delta_flat[2], delta_flat[3],
            delta_flat[4], delta_flat[5], delta_flat[6], delta_flat[7],
            # Cloud (3)
            cld_before, cld_after, cld_worst,
            # Spatial (5)
            n_neighbors, mean_nb_trees, std_nb_trees,
            delta_trees_5x5, delta_trees_7x7,
            # Cross-band (5)
            trees_crops_anti, trees_bare_anti, dominant_gain, band_div,
            tree_loss_ratio,
            # Magnitude (2)
            abs_trees, abs_max,
            # Metadata (3)
            month, is_monsoon, is_winter,
        ])
    return features


# ── Labeling with temporal state tracking ─────────────────────────────────────

def label_windows(stack: np.ndarray, gt: np.ndarray, window_meta: list):
    """
    Label all pixel-window pairs with CLEAN GT and TEMPORAL features.
    Returns: features (total_alert_pixels, N_feat), labels (total_alert_pixels,),
             spatial_blocks (total_alert_pixels,), metadata per record
    """
    N, B, H, W = stack.shape
    n_pix = H * W
    is_enriched = B >= 44

    # For enriched rasters, delta_trees is at band 16+1=17; for legacy, band 1
    delta_trees_band = (ENRICHED_DELTA.start + TREES_IDX) if is_enriched else TREES_IDX

    gt_trees = gt[TREES_IDX].ravel()
    is_gt_change = np.abs(gt_trees) >= GT_THRESH
    trees_all = stack[:, delta_trees_band, :, :].reshape(N, n_pix)

    # ── Clean GT: Event-based labeling ────────────────────────────────
    # For GT-changed pixels, find the window with the strongest ALERTING
    # signal. Only that specific window gets GT=1; all others are noise.
    # Key fix: we only consider windows where the pixel ACTUALLY fires
    # an alert (|Δtrees| >= ALERT_THRESH), not all windows.
    alert_mask_all = trees_all <= -ALERT_THRESH  # (N, n_pix) loss only

    best_window_per_pixel = np.full(n_pix, -1, dtype=np.int32)
    if is_gt_change.any():
        gt_idx = np.where(is_gt_change)[0]
        gt_trees_neg = gt_trees[gt_idx] < 0  # pixels that lost trees

        window_deltas = trees_all[:, gt_idx]          # (N, n_gt)
        fires_per_gt = alert_mask_all[:, gt_idx]      # (N, n_gt) bool

        # Mask non-alerting windows to neutral so they can't win
        masked_deltas = window_deltas.copy().astype(np.float32)
        masked_deltas[~fires_per_gt] = 0  # neutral = won't win argmin/argmax

        # For loss pixels: best = window with most negative delta (among alerting)
        # For gain pixels: best = window with most positive delta (among alerting)
        best_w = np.argmin(masked_deltas, axis=0)  # steepest drop
        best_w_gain = np.argmax(masked_deltas, axis=0)
        best_w[~gt_trees_neg] = best_w_gain[~gt_trees_neg]

        # Only assign if pixel fires at least one alert
        any_fire = fires_per_gt.any(axis=0)  # (n_gt,) bool
        best_w_final = np.where(any_fire, best_w, -1)
        best_window_per_pixel[gt_idx] = best_w_final

    n_assigned = int((best_window_per_pixel >= 0).sum())
    log.info(f"  Clean GT: {int(is_gt_change.sum()):,} GT-changed pixels, "
             f"{n_assigned:,} assigned to best alert window")

    # ── Temporal state accumulators ───────────────────────────────────
    cum_alert_count = np.zeros(n_pix, dtype=np.int32)
    cum_delta_trees = np.zeros(n_pix, dtype=np.float32)
    max_loss_so_far = np.zeros(n_pix, dtype=np.float32)
    first_alert_window = np.full(n_pix, -1, dtype=np.int32)

    # ── NDVI trajectory accumulators (enriched only) ──────────────────
    if is_enriched:
        ndvi_band_idx = ENRICHED_SPEC_BEFORE.start  # band 26 = ndvi_before
        nbr_band_idx  = ENRICHED_SPEC_BEFORE.start + 1  # band 27 = nbr_before
        ndvi_after_band_idx = ENRICHED_SPEC_AFTER.start  # band 29 = ndvi_after
        nbr_after_band_idx  = ENRICHED_SPEC_AFTER.start + 1  # band 30 = nbr_after

        # Track NDVI/NBR delta per window: ndvi_after - ndvi_before
        ndvi_delta_all = (stack[:, ndvi_after_band_idx, :, :] -
                          stack[:, ndvi_band_idx, :, :]).reshape(N, n_pix).astype(np.float32)
        nbr_delta_all = (stack[:, nbr_after_band_idx, :, :] -
                         stack[:, nbr_band_idx, :, :]).reshape(N, n_pix).astype(np.float32)
        ndvi_before_all = stack[:, ndvi_band_idx, :, :].reshape(N, n_pix).astype(np.float32)

        cum_delta_ndvi = np.zeros(n_pix, dtype=np.float32)
        cum_delta_nbr  = np.zeros(n_pix, dtype=np.float32)
        max_ndvi_loss  = np.zeros(n_pix, dtype=np.float32)  # most negative cum NDVI
        ndvi_at_first_alert = np.zeros(n_pix, dtype=np.float32)

    # Spatial block assignment (500m grid -> ~50 pixel blocks at 10m)
    block_size = 50
    row_blocks = np.arange(H) // block_size
    col_blocks = np.arange(W) // block_size
    n_col_blocks = col_blocks.max() + 1
    block_grid = (row_blocks[:, None] * n_col_blocks + col_blocks[None, :]).ravel()

    all_features = []
    all_labels = []
    all_blocks = []
    all_windows = []
    window_stats = []

    for w in range(N):
        # ── Per-window features (26) ─────────────────────────────────
        feats = extract_features_for_window(
            stack, w, cum_alert_count, window_meta[w], H, W
        )

        # ── Temporal features — state BEFORE this window ──────────────
        wins_since = np.where(
            first_alert_window >= 0,
            (w - first_alert_window).astype(np.float32),
            np.float32(-1)
        )
        temporal_feats_base = np.column_stack([
            cum_alert_count.astype(np.float32),
            cum_delta_trees,
            max_loss_so_far,
            wins_since,
        ])

        if is_enriched:
            # NDVI recovery: has cumulative NDVI recovered from max loss?
            ndvi_recovery = (cum_delta_ndvi - max_ndvi_loss).clip(0)
            ndvi_recovery_flag = (ndvi_recovery > 0.05).astype(np.float32)

            temporal_feats_ndvi = np.column_stack([
                cum_delta_ndvi,
                max_ndvi_loss,
                ndvi_at_first_alert,
                cum_delta_nbr,
                ndvi_recovery_flag,
            ])
            temporal_feats = np.column_stack([temporal_feats_base, temporal_feats_ndvi])
        else:
            temporal_feats = temporal_feats_base

        # Combine: base per-window feats + temporal
        feats_full = np.column_stack([feats, temporal_feats])

        # ── Labels: Clean GT ─────────────────────────────────────────
        alert_trees = trees_all[w]
        is_alert = alert_trees <= -ALERT_THRESH  # loss only

        # Clean GT: pixel is TP only if this is its BEST window
        is_best_window = (best_window_per_pixel == w)
        tp = is_alert & is_gt_change & is_best_window

        fp        = is_alert & ~is_gt_change
        fp_contra = is_alert & is_gt_change & ~is_best_window  # alerted but wrong window

        # ── Update temporal state AFTER feature extraction ────────────
        delta_trees_w = trees_all[w].astype(np.float32)
        cum_alert_count += is_alert.astype(np.int32)
        cum_delta_trees += delta_trees_w
        max_loss_so_far = np.minimum(max_loss_so_far, delta_trees_w)
        first_alert_window_new = np.where(
            (first_alert_window < 0) & is_alert, w, first_alert_window
        )

        # Update NDVI trajectory accumulators (enriched only)
        if is_enriched:
            ndvi_delta_w = ndvi_delta_all[w]
            nbr_delta_w  = nbr_delta_all[w]
            cum_delta_ndvi += ndvi_delta_w
            cum_delta_nbr  += nbr_delta_w
            max_ndvi_loss = np.minimum(max_ndvi_loss, cum_delta_ndvi)

            # Record NDVI at first alert
            newly_alerted = (first_alert_window < 0) & is_alert
            ndvi_at_first_alert = np.where(
                newly_alerted,
                ndvi_before_all[w],
                ndvi_at_first_alert
            )

        first_alert_window = first_alert_window_new

        # ── Keep only alert-firing pixels ─────────────────────────────
        alert_idx = np.where(is_alert)[0]
        if len(alert_idx) == 0:
            continue

        labels = tp[alert_idx].astype(np.int32)

        all_features.append(feats_full[alert_idx])
        all_labels.append(labels)
        all_blocks.append(block_grid[alert_idx])
        all_windows.append(np.full(len(alert_idx), w, dtype=np.int32))

        n_tp = int(tp[alert_idx].sum())
        n_fp = int(fp[alert_idx].sum())
        n_fpc = int(fp_contra[alert_idx].sum())
        total = n_tp + n_fp + n_fpc
        prec = n_tp / total if total > 0 else 0
        window_stats.append({
            "window": w,
            "date": f"{window_meta[w]['date_before']} -> {window_meta[w]['date_after']}",
            "season": window_meta[w]["season"],
            "n_alerts": total, "n_tp": n_tp, "n_fp": n_fp,
            "n_fpc": n_fpc, "precision": round(prec, 4),
        })

        log.info(
            f"  W{w:02d} ({window_meta[w]['date_before']} -> "
            f"{window_meta[w]['date_after']} {window_meta[w]['season']:14s}) "
            f"alerts={total:>7,}  TP={n_tp:>6,}  FP={n_fp:>6,}  "
            f"FPc={n_fpc:>5,}  Prec={prec:.3f}"
        )

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    blocks = np.concatenate(all_blocks, axis=0)
    windows = np.concatenate(all_windows, axis=0)

    log.info(f"\nTotal training records: {len(y):,}")
    log.info(f"  Real (TP):  {int(y.sum()):,}  ({y.mean()*100:.2f}%)")
    log.info(f"  Noise:      {int((1-y).sum()):,}  ({(1-y.mean())*100:.2f}%)")

    return X, y, blocks, windows, window_stats


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val, feature_names):
    """Train LightGBM binary classifier with balanced undersampling."""
    import lightgbm as lgb

    # ── Balanced undersampling ───────────────────────────────────────
    # Downsample noise to 2:1 ratio (not 1:1) for TRAINING only.
    # Slight class imbalance helps prevent overconfident predictions.
    # Validation stays UNSAMPLED for honest metric evaluation.
    rng = np.random.default_rng(42)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    # Sample noise at 2:1 ratio
    target_neg = min(n_neg, n_pos * 2)
    neg_sample = rng.choice(neg_idx, size=target_neg, replace=False)
    bal_idx = np.concatenate([pos_idx, neg_sample])
    rng.shuffle(bal_idx)
    X_bal = X_train[bal_idx]
    y_bal = y_train[bal_idx]
    log.info(f"Balanced undersampling: {n_pos:,} pos + {target_neg:,} neg = {len(bal_idx):,}")

    dtrain = lgb.Dataset(X_bal, label=y_bal,
                         feature_name=feature_names, free_raw_data=False)
    dval   = lgb.Dataset(X_val, label=y_val,
                         feature_name=feature_names, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.02,      # Slower learning — more stable
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 200,   # Strong regularization
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "lambda_l1": 1.0,           # L1 regularization
        "lambda_l2": 5.0,           # L2 regularization
        "scale_pos_weight": 2.0,    # Boost minority (TP) class — with 2:1 undersampling this gives ~1:1 effective weight
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    callbacks = [
        lgb.early_stopping(200, verbose=True),
        lgb.log_evaluation(100),
    ]

    model = lgb.train(
        params, dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )
    log.info(f"Best iteration: {model.best_iteration}")
    return model


def spatial_block_split(blocks, test_frac=0.15, val_frac=0.15, seed=42):
    """
    GroupKFold-style split: entire spatial blocks go to train/val/test.
    No pixel leaks between sets.
    """
    rng = np.random.default_rng(seed)
    unique_blocks = np.unique(blocks)
    rng.shuffle(unique_blocks)

    n_test = max(1, int(len(unique_blocks) * test_frac))
    n_val  = max(1, int(len(unique_blocks) * val_frac))

    test_blocks = set(unique_blocks[:n_test])
    val_blocks  = set(unique_blocks[n_test:n_test + n_val])
    train_blocks = set(unique_blocks[n_test + n_val:])

    train_mask = np.isin(blocks, list(train_blocks))
    val_mask   = np.isin(blocks, list(val_blocks))
    test_mask  = np.isin(blocks, list(test_blocks))

    return train_mask, val_mask, test_mask


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, feature_names, out_dir):
    """Full evaluation: PR curve, confusion matrix, feature importance."""
    from sklearn.metrics import (
        precision_recall_curve, average_precision_score,
        confusion_matrix, classification_report, balanced_accuracy_score,
    )

    y_prob = model.predict(X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    # ── Metrics ──────────────────────────────────────────────────────
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_prob)
    log.info(f"Balanced accuracy: {bal_acc:.4f}")
    log.info(f"Average precision (AP): {ap:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['Noise', 'Real'])}")

    # ── PR Curve + F1-optimal threshold selection ──────────────────────
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    # F1 score for every possible threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    thresh_f1  = float(thresholds[optimal_idx])
    prec_f1    = float(precision[optimal_idx])
    rec_f1     = float(recall[optimal_idx])
    best_f1    = float(f1_scores[optimal_idx])

    log.info(f"\n{'='*50}")
    log.info(f"  🎯 OPTIMAL THRESHOLD (Max F1)")
    log.info(f"{'='*50}")
    log.info(f"  Threshold:   {thresh_f1:.4f}")
    log.info(f"  F1-Score:    {best_f1:.4f}")
    log.info(f"  Precision:   {prec_f1:.4f}")
    log.info(f"  Recall:      {rec_f1:.4f}")
    log.info(f"{'='*50}")

    # Also find high-precision threshold (prec >= 80%) as secondary
    prec80_mask = precision[:-1] >= 0.80
    if prec80_mask.any():
        idx80 = np.where(prec80_mask)[0][0]
        thresh80 = float(thresholds[idx80])
        prec80   = float(precision[idx80])
        rec80    = float(recall[idx80])
    else:
        thresh80, prec80, rec80 = thresh_f1, prec_f1, rec_f1

    log.info(f"\n  (Secondary) High-precision threshold (P≥80%): {thresh80:.4f}")
    log.info(f"    Precision: {prec80:.4f}  Recall: {rec80:.4f}")

    # Plot PR curve with both thresholds marked
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(recall, precision, linewidth=2, color="teal")
    ax.scatter([rec_f1], [prec_f1], s=150, c="red", zorder=5, marker="*",
               label=f"★ F1-Optimal: P={prec_f1:.2f}, R={rec_f1:.2f}, T={thresh_f1:.3f}, F1={best_f1:.3f}")
    ax.scatter([rec80], [prec80], s=80, c="orange", zorder=5, marker="D",
               label=f"◆ High-Prec: P={prec80:.2f}, R={rec80:.2f}, T={thresh80:.3f}")
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, label="80% Precision line")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve (AP = {ap:.3f})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Confusion Matrix at F1-optimal threshold ─────────────────────
    y_pred_f1 = (y_prob >= thresh_f1).astype(int)
    cm = confusion_matrix(y_test, y_pred_f1)

    log.info(f"\n  📊 Confusion Matrix (F1-optimal, T={thresh_f1:.4f}):")
    log.info(f"    TN (Correct No-Change): {cm[0, 0]:,}")
    log.info(f"    FP (False Alarms):      {cm[0, 1]:,}")
    log.info(f"    FN (Missed Cuts):       {cm[1, 0]:,}")
    log.info(f"    TP (Caught Cuts):       {cm[1, 1]:,}")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Noise", "Real"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion Matrix (F1-Optimal T={thresh_f1:.3f})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Feature Importance ───────────────────────────────────────────
    importance = model.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importance)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    n_show = min(23, len(feature_names))
    top_idx = sorted_idx[:n_show]
    ax.barh(range(n_show), importance[top_idx][::-1],
            color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([feature_names[i] for i in top_idx][::-1], fontsize=10)
    ax.set_xlabel("Gain", fontsize=11)
    ax.set_title("Feature Importance (LightGBM Gain)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"\nTop 10 features:")
    for i in range(min(10, len(feature_names))):
        idx = sorted_idx[i]
        log.info(f"  {i+1:2d}. {feature_names[idx]:25s}  gain={importance[idx]:.0f}")

    return {
        "balanced_accuracy": round(bal_acc, 4),
        "average_precision": round(ap, 4),
        "operational_threshold": thresh_f1,
        "f1_score": round(best_f1, 4),
        "precision_at_threshold": round(prec_f1, 4),
        "recall_at_threshold": round(rec_f1, 4),
        "high_precision_threshold": thresh80,
        "high_precision_prec": round(prec80, 4),
        "high_precision_recall": round(rec80, 4),
        "confusion_matrix": cm.tolist(),
        "top_features": [
            {"name": feature_names[sorted_idx[i]],
             "gain": float(importance[sorted_idx[i]])}
            for i in range(min(10, len(feature_names)))
        ],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="outputs/alert_filter")
    parser.add_argument("--alert-thresh", type=float, default=None,
                        help="Override ALERT_THRESH (default: 0.10)")
    args = parser.parse_args()

    # Override global threshold if specified
    global ALERT_THRESH
    if args.alert_thresh is not None:
        ALERT_THRESH = args.alert_thresh
        log.info(f"ALERT_THRESH overridden to {ALERT_THRESH}")

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # ── Phase 1: Load ────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("  Phase 1: LOAD")
    log.info("=" * 65)
    stack, window_meta, profile = load_alert_stack(args.data_dir)
    gt = load_gt(args.data_dir)

    # ── Phase 2: Features + Labels ───────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  Phase 2: FEATURES + LABELS")
    log.info("=" * 65)
    X, y, blocks, windows, window_stats = label_windows(stack, gt, window_meta)

    # ── Phase 3: Split ───────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  Phase 3: SPATIAL BLOCK SPLIT")
    log.info("=" * 65)
    train_mask, val_mask, test_mask = spatial_block_split(blocks)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    log.info(f"Train: {len(y_train):>10,}  (real: {int(y_train.sum()):>7,}  "
             f"noise: {int((1-y_train).sum()):>7,})")
    log.info(f"Val:   {len(y_val):>10,}  (real: {int(y_val.sum()):>7,}  "
             f"noise: {int((1-y_val).sum()):>7,})")
    log.info(f"Test:  {len(y_test):>10,}  (real: {int(y_test.sum()):>7,}  "
             f"noise: {int((1-y_test).sum()):>7,})")

    n_unique_blocks = len(np.unique(blocks))
    log.info(f"Spatial blocks: {n_unique_blocks} "
             f"(~{200*10/1000:.0f}km × {200*10/1000:.0f}km each)")

    # ── Phase 4: Train ───────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  Phase 4: TRAIN LightGBM")
    log.info("=" * 65)
    model = train_model(X_train, y_train, X_val, y_val, FEATURE_NAMES)

    model_path = os.path.join(args.out_dir, "model.lgbm")
    model.save_model(model_path)
    log.info(f"Model saved: {model_path}")

    # ── Phase 5: Evaluate ────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  Phase 5: EVALUATE")
    log.info("=" * 65)
    metrics = evaluate(model, X_test, y_test, FEATURE_NAMES, args.out_dir)

    # ── Phase 6: Export config ───────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  Phase 6: EXPORT")
    log.info("=" * 65)

    feature_config = {
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "alert_threshold": ALERT_THRESH,
        "gt_threshold": GT_THRESH,
        "operational_threshold": metrics["operational_threshold"],
        "model_file": "model.lgbm",
        "band_order": DW_BANDS,
        "trees_band_index": TREES_IDX,
    }
    config_path = os.path.join(args.out_dir, "feature_config.json")
    with open(config_path, "w") as f:
        json.dump(feature_config, f, indent=2)
    log.info(f"Feature config: {config_path}")

    # Training report
    elapsed = time.time() - t0
    report = {
        "elapsed_seconds": round(elapsed, 1),
        "data_dir": args.data_dir,
        "n_windows": len(window_meta),
        "n_total_records": len(y),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "n_spatial_blocks": n_unique_blocks,
        "metrics": metrics,
        "window_stats": window_stats,
        "feature_config": feature_config,
    }
    report_path = os.path.join(args.out_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"\n{'='*65}")
    log.info(f"  DONE in {elapsed:.1f}s")
    log.info(f"  Balanced accuracy:       {metrics['balanced_accuracy']:.4f}")
    log.info(f"  Average precision (AP):  {metrics['average_precision']:.4f}")
    log.info(f"  Operational threshold:   {metrics['operational_threshold']:.4f}")
    log.info(f"  Precision @ threshold:   {metrics['precision_at_threshold']:.4f}")
    log.info(f"  Recall @ threshold:      {metrics['recall_at_threshold']:.4f}")
    log.info(f"  Model: {model_path}")
    log.info(f"  Config: {config_path}")
    log.info(f"{'='*65}")


if __name__ == "__main__":
    main()
