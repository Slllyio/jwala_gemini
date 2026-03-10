"""
Filter Alert Raster — Alert TIF In, Filtered TIF Out
=====================================================

Clean modular design:
  Input:  Raw DW alert raster(s)  (10-band .tif)
  Output: Filtered raster         (scored .tif)

The filtered raster can then be vectorized by a separate tool
(vectorize_raster.py, QGIS, gdal_polygonize, etc.)

Two modes:
  1. SINGLE-WINDOW: filter one alert raster
     python scripts/filter_alert_raster.py \
         --alert-tifs data/alerts/alert_delta_HAMEERPUR_2025-12-19_to_2025-12-24.tif \
         --model outputs/alert_filter/model.lgbm \
         --config outputs/alert_filter/feature_config.json \
         --out-tif outputs/filtered/filtered_HAMEERPUR.tif

  2. MULTI-WINDOW (temporal stacking): filter N rasters, stack scores
     python scripts/filter_alert_raster.py \
         --alert-tifs data/alerts/alert_delta_HAMEERPUR_*.tif \
         --model outputs/alert_filter/model.lgbm \
         --config outputs/alert_filter/feature_config.json \
         --out-tif outputs/filtered/stacked_HAMEERPUR.tif \
         --min-fires 2

Output bands:
  Band 1: stacked_score (or single-window P if 1 file)
  Band 2: n_fires (how many windows this pixel fired)
  Band 3: confirmed mask (binary: 1=confirmed, 0=filtered)
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
import glob

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter, binary_closing

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DW_BANDS = ["water", "trees", "grass", "flooded_veg",
            "crops", "shrub_scrub", "built", "bare"]
TREES_IDX = 1
N_BANDS   = 8
SEVERE_LOSS_THRESH = 0.30  # alert-worthy regardless of replacement
ANTHROPO_GAIN_BANDS = [4, 6]  # crops=4, built=6
ALERT_THRESH = 0.10  # default alert threshold

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

# ── Post-classification: change class labels ─────────────────────────────────
# Stage 2 labels for confirmed alerts — phenology tagged, not suppressed
CHANGE_CLASS = {
    0: "No Alert",
    1: "Encroachment (Crops)",
    2: "Built Expansion",
    3: "Degradation (Phenology)",
    4: "Shrub/Scrub (Phenology)",
    5: "Pure Tree Loss",
    6: "Other",
}
# DW band -> change class mapping
GAIN_BAND_TO_CLASS = {
    0: 6,  # water -> Other
    2: 3,  # grass -> Degradation
    3: 6,  # flooded_veg -> Other
    4: 1,  # crops -> Encroachment
    5: 4,  # shrub_scrub -> Shrub/Scrub
    6: 2,  # built -> Built
    7: 3,  # bare -> Degradation (bare soil exposure)
}

SEASON_MAP = {
    1: "Winter", 2: "Winter",
    3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
    12: "Winter",
}


def parse_dates(stem: str):
    """Extract (date_before, date_after, sub_range) from filename stem."""
    m = re.search(r"alert_(?:delta|enriched)_(.+?)_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})", stem)
    if m:
        return m.group(2), m.group(3), m.group(1)
    return "unknown", "unknown", "UNKNOWN"


# ── Feature extraction (identical to training) ────────────────────────────────

def extract_features(alert_data: np.ndarray, month: int,
                     alert_thresh: float, H: int, W: int,
                     cum_alert_count: np.ndarray = None,
                     date_before: str = "unknown",
                     date_after: str = "unknown") -> np.ndarray:
    """Extract per-window features from a single alert raster.
    Enriched (44-band): 66 base features (51 legacy + 15 v5)
    Legacy (10-band):   26 base features
    Temporal features are appended externally by stack_windows or filter_single.
    """
    n_pix = H * W
    n_bands_raster = alert_data.shape[0]
    is_enriched = n_bands_raster >= 44

    if is_enriched:
        # ── Extract from ENRICHED 44-band raster ─────────────────────
        dw_before = alert_data[ENRICHED_DW_BEFORE].astype(np.float32)  # (8, H, W)
        dw_after  = alert_data[ENRICHED_DW_AFTER].astype(np.float32)
        deltas    = alert_data[ENRICHED_DELTA].astype(np.float32)
        cld_raw   = alert_data[ENRICHED_CLOUD].astype(np.float32)  # (2, H, W)
        spec_before = alert_data[ENRICHED_SPEC_BEFORE].astype(np.float32)  # (3, H, W)
        spec_after  = alert_data[ENRICHED_SPEC_AFTER].astype(np.float32)
        s2_before = alert_data[ENRICHED_S2_BEFORE].astype(np.float32)  # (6, H, W)
        s2_after  = alert_data[ENRICHED_S2_AFTER].astype(np.float32)

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
        deltas = alert_data[:N_BANDS].astype(np.float32)
        delta_flat = deltas.reshape(N_BANDS, n_pix)
        has_cloud = n_bands_raster >= 10
        if has_cloud:
            cld_before = alert_data[8].ravel().astype(np.float32)
            cld_after  = alert_data[9].ravel().astype(np.float32)
        else:
            cld_before = np.zeros(n_pix, dtype=np.float32)
            cld_after  = np.zeros(n_pix, dtype=np.float32)

    cld_worst = np.maximum(cld_before, cld_after)

    # ── Spatial context (5) ──────────────────────────────────────────
    trees_2d = deltas[TREES_IDX]  # (H, W)
    alert_mask_2d = (trees_2d <= -alert_thresh).astype(np.float32)
    neighbor_sum = uniform_filter(alert_mask_2d, size=3, mode="constant") * 9
    n_neighbors = np.clip((neighbor_sum - alert_mask_2d).ravel(), 0, 8)
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
    band_div = (np.abs(delta_flat) > alert_thresh).sum(axis=0).astype(np.float32)

    # Semantic change class from dominant gain band
    band_to_class = np.array([GAIN_BAND_TO_CLASS.get(b, 6) for b in range(N_BANDS)])
    dom_gain_idx = np.argmax(gain_deltas, axis=0)
    dom_gain_val = np.max(gain_deltas, axis=0)
    dominant_gain_class = band_to_class[dom_gain_idx].astype(np.float32)
    pure_loss = (trees_delta <= -SEVERE_LOSS_THRESH) & (dom_gain_val < 0.05)
    dominant_gain_class[pure_loss] = 5.0  # Pure Tree Loss

    total_volatility = np.abs(trees_delta) + np.abs(crops_delta) + np.abs(bare_delta)
    tree_loss_ratio = np.abs(trees_delta) / (total_volatility + 1e-6)

    # ── Magnitude (2) ────────────────────────────────────────────────
    abs_trees = np.abs(trees_delta)
    abs_max = np.max(np.abs(delta_flat), axis=0)

    # ── Metadata (3) ─────────────────────────────────────────────────
    season = SEASON_MAP.get(month, "Unknown")
    month_arr = np.full(n_pix, month, dtype=np.float32)
    is_monsoon = np.full(n_pix, 1.0 if season == "Monsoon" else 0.0, dtype=np.float32)
    is_winter = np.full(n_pix, 1.0 if season == "Winter" else 0.0, dtype=np.float32)

    if is_enriched:
        # ── Enriched assembly ─────────────────────────────────────────
        delta_ndvi = spec_after_flat[0] - spec_before_flat[0]
        delta_nbr  = spec_after_flat[1] - spec_before_flat[1]
        delta_ndmi = spec_after_flat[2] - spec_before_flat[2]

        # ── v5: Confidence & relative features (3) ────────────────────
        trees_before_val = dw_before_flat[TREES_IDX]
        relative_tree_loss = trees_delta / (trees_before_val + 1e-6)

        eps = 1e-8
        p_before = dw_before_flat + eps
        p_after  = dw_after_flat  + eps
        dw_entropy_before = -(p_before * np.log(p_before)).sum(axis=0)
        dw_entropy_after  = -(p_after  * np.log(p_after)).sum(axis=0)

        # ── v5: Semantic transition features (4) ──────────────────────
        dw_before_class = np.argmax(dw_before_flat, axis=0).astype(np.float32)
        dominant_gain_value = dom_gain_val.astype(np.float32)
        sorted_gains = np.sort(gain_deltas, axis=0)
        gain_gap = (sorted_gains[-1] - sorted_gains[-2]).astype(np.float32)
        transition_type = (dw_before_class * 10 + dominant_gain_class).astype(np.float32)

        # ── v5: Season × class interactions (3) ───────────────────────
        monsoon_x_grass = is_monsoon * (dominant_gain_class == 3).astype(np.float32)
        monsoon_x_shrub = is_monsoon * (dominant_gain_class == 4).astype(np.float32)
        winter_x_crops  = is_winter  * (dominant_gain_class == 1).astype(np.float32)

        # ── v5: Spectral indices NDBI (3) ─────────────────────────────
        nir_bef = s2_before_flat[3] + 1e-6
        swir1_bef = s2_before_flat[4] + 1e-6
        nir_aft = s2_after_flat[3] + 1e-6
        swir1_aft = s2_after_flat[4] + 1e-6
        ndbi_before = (swir1_bef - nir_bef) / (swir1_bef + nir_bef)
        ndbi_after  = (swir1_aft - nir_aft) / (swir1_aft + nir_aft)
        delta_ndbi_val  = ndbi_after - ndbi_before

        # ── v5: Window duration (1) ───────────────────────────────────
        try:
            d_bef = datetime.strptime(date_before, "%Y-%m-%d")
            d_aft = datetime.strptime(date_after, "%Y-%m-%d")
            w_dur = float((d_aft - d_bef).days)
        except (ValueError, KeyError):
            w_dur = 30.0
        window_duration_days = np.full(n_pix, w_dur, dtype=np.float32)

        # ── v5: Alert rate (1) ────────────────────────────────────────
        if cum_alert_count is not None:
            wins_since_first = np.where(
                cum_alert_count > 0,
                np.maximum(cum_alert_count - 1, 0).astype(np.float32),
                np.float32(0)
            )
            alert_rate = cum_alert_count.astype(np.float32) / (wins_since_first + 1.0)
        else:
            alert_rate = np.zeros(n_pix, dtype=np.float32)

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
            ndbi_before, ndbi_after, delta_ndbi_val,
            window_duration_days, alert_rate,
            # Metadata (3) — month, is_monsoon, is_winter omitted here,
            # appended with temporal features to match training order
        ])
    else:
        # ── Legacy assembly (n_pix, 26) ──────────────────────────────
        features = np.column_stack([
            delta_flat[0], delta_flat[1], delta_flat[2], delta_flat[3],
            delta_flat[4], delta_flat[5], delta_flat[6], delta_flat[7],
            cld_before, cld_after, cld_worst,
            n_neighbors, mean_nb_trees, std_nb_trees,
            delta_trees_5x5, delta_trees_7x7,
            trees_crops_anti, trees_bare_anti, dominant_gain, band_div,
            tree_loss_ratio,
            abs_trees, abs_max,
            month_arr, is_monsoon, is_winter,
        ])

    return features, is_enriched


# ── Post-classification: classify change type per pixel ──────────────────────

def classify_change_class(data: np.ndarray, is_alert: np.ndarray) -> np.ndarray:
    """
    Stage 2: Classify each alerting pixel by its dominant land cover change.
    
    Args:
        data: (B, H, W) raster — enriched (44-band) or legacy (10-band)
        is_alert: (n_pix,) boolean mask of alerting pixels
    Returns:
        change_class: (n_pix,) int array with class labels
    """
    n_pix = is_alert.shape[0]
    change_class = np.zeros(n_pix, dtype=np.int32)
    
    if not is_alert.any():
        return change_class
    
    # Extract delta bands — correct slice for enriched vs legacy
    is_enriched = data.shape[0] >= 44
    if is_enriched:
        deltas = data[ENRICHED_DELTA].reshape(N_BANDS, -1).astype(np.float32)
    else:
        deltas = data[:N_BANDS].reshape(N_BANDS, -1).astype(np.float32)
    
    # For alerting pixels, find dominant gain band (excluding trees)
    gains = deltas.copy()
    gains[TREES_IDX] = -999  # exclude trees from gain competition
    dom_gain_band = np.argmax(gains, axis=0)  # (n_pix,)
    dom_gain_val = np.max(gains, axis=0)
    
    # Map dominant gain band to change class
    band_to_class = np.array([GAIN_BAND_TO_CLASS.get(b, 6) for b in range(N_BANDS)])
    pixel_class = band_to_class[dom_gain_band]
    
    # Pure Tree Loss: severe tree drop but dominant gain is very small
    trees_delta = deltas[TREES_IDX]
    pure_loss = (trees_delta <= -SEVERE_LOSS_THRESH) & (dom_gain_val < 0.05)
    pixel_class[pure_loss] = 5  # Pure Tree Loss
    
    # Apply only to alerting pixels
    change_class[is_alert] = pixel_class[is_alert]
    
    return change_class


# ── Single-window filtering ──────────────────────────────────────────────────

def filter_single(tif_path: str, model, alert_thresh: float,
                  threshold: float) -> tuple:
    """
    Filter one alert raster (single-window mode with zero temporal history).
    Returns: (y_prob, is_alert, H, W, profile, transform)
    """
    with rasterio.open(tif_path) as ds:
        data = ds.read()
        profile = ds.profile.copy()
        transform = ds.transform

    H, W = data.shape[1], data.shape[2]
    n_pix = H * W

    stem = os.path.splitext(os.path.basename(tif_path))[0]
    d_before, d_after, sub_range = parse_dates(stem)
    month = int(d_before.split("-")[1]) if d_before != "unknown" else 1

    # Per-window features (66 enriched / 26 legacy)
    features, is_enriched = extract_features(
        data, month, alert_thresh, H, W,
        date_before=d_before, date_after=d_after,
    )

    # 4 base temporal features — all zeros for single-window (no history)
    temporal_feats_base = np.column_stack([
        np.zeros(n_pix, dtype=np.float32),  # cum_alert_count
        np.zeros(n_pix, dtype=np.float32),  # cum_delta_trees
        np.zeros(n_pix, dtype=np.float32),  # max_loss_so_far
        np.full(n_pix, -1, dtype=np.float32),  # windows_since_first_alert
    ])

    if is_enriched:
        # 5 NDVI trajectory features + 3 metadata (= 8 extra)
        temporal_feats_ndvi = np.column_stack([
            np.zeros(n_pix, dtype=np.float32),  # cum_delta_ndvi
            np.zeros(n_pix, dtype=np.float32),  # max_ndvi_loss
            np.zeros(n_pix, dtype=np.float32),  # ndvi_at_first_alert
            np.zeros(n_pix, dtype=np.float32),  # cum_delta_nbr
            np.zeros(n_pix, dtype=np.float32),  # ndvi_recovery_flag
        ])
        season = SEASON_MAP.get(month, "Unknown")
        month_arr = np.full(n_pix, month, dtype=np.float32)
        is_monsoon = np.full(n_pix, 1.0 if season == "Monsoon" else 0.0, dtype=np.float32)
        is_winter = np.full(n_pix, 1.0 if season == "Winter" else 0.0, dtype=np.float32)
        features_full = np.column_stack([
            features, temporal_feats_base, temporal_feats_ndvi,
            month_arr, is_monsoon, is_winter,
        ])
    else:
        features_full = np.column_stack([features, temporal_feats_base])

    y_prob = model.predict(features_full).astype(np.float32)

    # Alert-firing mask (using correct delta bands)
    if is_enriched:
        trees_delta = data[ENRICHED_DELTA.start + TREES_IDX].ravel().astype(np.float32)
    else:
        trees_delta = data[TREES_IDX].ravel().astype(np.float32)
    is_alert = trees_delta <= -alert_thresh

    # Zero out non-alerting pixels
    y_prob[~is_alert] = 0

    # ── Post-Classification: change class per pixel ──────────────────
    change_class = classify_change_class(data, is_alert)

    return y_prob, is_alert, data, H, W, profile, transform, change_class


# ── Multi-window temporal stacking ───────────────────────────────────────────

def stack_windows(tif_paths: list, model, alert_thresh: float,
                  min_fires: int = 2) -> tuple:
    """
    Run model on all windows with temporal feature tracking.
    Returns: (stacked_score, n_fires, H, W, profile, transform)
    """
    # Sort by date
    def sort_key(p):
        _, d_after, _ = parse_dates(os.path.splitext(os.path.basename(p))[0])
        return d_after
    tif_paths = sorted(tif_paths, key=sort_key)

    # Read first to get dimensions
    with rasterio.open(tif_paths[0]) as ds:
        ref_data = ds.read()
        ref_shape = ref_data.shape
        profile = ds.profile.copy()
        transform = ds.transform
    H, W = ref_shape[1], ref_shape[2]
    n_pix = H * W
    is_enriched = ref_shape[0] >= 44

    # Accumulators (stacking)
    n_fires      = np.zeros(n_pix, dtype=np.int32)
    sum_P        = np.zeros(n_pix, dtype=np.float32)
    sum_P_sq     = np.zeros(n_pix, dtype=np.float32)
    max_P        = np.zeros(n_pix, dtype=np.float32)
    cur_streak   = np.zeros(n_pix, dtype=np.int32)
    max_streak   = np.zeros(n_pix, dtype=np.int32)
    prev_fired   = np.zeros(n_pix, dtype=bool)

    # Temporal state (matching training)
    cum_alert_count = np.zeros(n_pix, dtype=np.int32)
    cum_delta_trees = np.zeros(n_pix, dtype=np.float32)
    max_loss_so_far = np.zeros(n_pix, dtype=np.float32)
    first_alert_window = np.full(n_pix, -1, dtype=np.int32)

    # Magnitude Fast-Track
    fast_track   = np.zeros(n_pix, dtype=bool)
    fast_track_P = np.zeros(n_pix, dtype=np.float32)

    # Post-classification: track change class from best (highest P) window
    best_change_class = np.zeros(n_pix, dtype=np.int32)
    best_P_for_class = np.zeros(n_pix, dtype=np.float32)

    # NDVI trajectory accumulators (enriched only)
    cum_delta_ndvi = np.zeros(n_pix, dtype=np.float32)
    max_ndvi_loss = np.zeros(n_pix, dtype=np.float32)
    ndvi_at_first_alert = np.zeros(n_pix, dtype=np.float32)
    cum_delta_nbr = np.zeros(n_pix, dtype=np.float32)

    log.info(f"Stacking {len(tif_paths)} alert windows (enriched={is_enriched})...")

    for w, tif in enumerate(tif_paths):
        stem = os.path.splitext(os.path.basename(tif))[0]
        d_before, d_after, _ = parse_dates(stem)
        month = int(d_before.split("-")[1]) if d_before != "unknown" else 1

        with rasterio.open(tif) as ds:
            data = ds.read()

        # Per-window features (66 enriched / 26 legacy)
        features, _ = extract_features(
            data, month, alert_thresh, H, W,
            cum_alert_count=cum_alert_count,
            date_before=d_before, date_after=d_after,
        )

        # 4 temporal features (state BEFORE this window)
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
            # NDVI recovery flag
            ndvi_recovery = (cum_delta_ndvi - max_ndvi_loss).clip(0)
            ndvi_recovery_flag = (ndvi_recovery > 0.05).astype(np.float32)

            temporal_feats_ndvi = np.column_stack([
                cum_delta_ndvi,
                max_ndvi_loss,
                ndvi_at_first_alert,
                cum_delta_nbr,
                ndvi_recovery_flag,
            ])
            season = SEASON_MAP.get(month, "Unknown")
            month_arr = np.full(n_pix, month, dtype=np.float32)
            is_monsoon = np.full(n_pix, 1.0 if season == "Monsoon" else 0.0, dtype=np.float32)
            is_winter = np.full(n_pix, 1.0 if season == "Winter" else 0.0, dtype=np.float32)
            features_full = np.column_stack([
                features, temporal_feats_base, temporal_feats_ndvi,
                month_arr, is_monsoon, is_winter,
            ])
        else:
            features_full = np.column_stack([features, temporal_feats_base])

        y_prob = model.predict(features_full).astype(np.float32)

        # Get delta_trees from correct bands
        if is_enriched:
            trees_delta = data[ENRICHED_DELTA.start + TREES_IDX].ravel().astype(np.float32)
        else:
            trees_delta = data[TREES_IDX].ravel().astype(np.float32)
        is_alert = trees_delta <= -alert_thresh  # loss only
        y_prob[~is_alert] = 0

        fired = is_alert
        n_fires += fired.astype(np.int32)
        sum_P   += y_prob
        sum_P_sq += y_prob ** 2
        max_P    = np.maximum(max_P, y_prob)

        # Streak
        continuing = fired & prev_fired
        starting   = fired & ~prev_fired
        cur_streak[continuing] += 1
        cur_streak[starting]    = 1
        cur_streak[~fired]      = 0
        max_streak = np.maximum(max_streak, cur_streak)
        prev_fired = fired.copy()

        # Fast-Track: P>0.90 AND delta_bare>0.20 AND delta_trees<-0.15
        if is_enriched:
            deltas = data[ENRICHED_DELTA].reshape(N_BANDS, n_pix).astype(np.float32)
        else:
            deltas = data[:N_BANDS].reshape(N_BANDS, n_pix).astype(np.float32)
        ft_mask = (
            fired &
            (y_prob > 0.90) &
            (deltas[7] > 0.20) &
            (deltas[TREES_IDX] < -0.15)
        )
        fast_track |= ft_mask
        fast_track_P = np.maximum(fast_track_P, y_prob * ft_mask.astype(np.float32))

        # Update temporal state AFTER prediction
        delta_trees_w = deltas[TREES_IDX]
        cum_alert_count += fired.astype(np.int32)
        cum_delta_trees += delta_trees_w
        max_loss_so_far = np.minimum(max_loss_so_far, delta_trees_w)
        first_alert_window_new = np.where(
            (first_alert_window < 0) & fired, w, first_alert_window
        )

        # Update NDVI trajectory accumulators (enriched only)
        if is_enriched:
            ndvi_delta_w = data[ENRICHED_SPEC_AFTER.start].ravel().astype(np.float32) - \
                           data[ENRICHED_SPEC_BEFORE.start].ravel().astype(np.float32)
            nbr_delta_w = data[ENRICHED_SPEC_AFTER.start + 1].ravel().astype(np.float32) - \
                          data[ENRICHED_SPEC_BEFORE.start + 1].ravel().astype(np.float32)
            cum_delta_ndvi += ndvi_delta_w
            cum_delta_nbr += nbr_delta_w
            max_ndvi_loss = np.minimum(max_ndvi_loss, cum_delta_ndvi)

            # Record NDVI at first alert
            newly_alerted = (first_alert_window < 0) & fired
            ndvi_at_first_alert = np.where(
                newly_alerted,
                data[ENRICHED_SPEC_BEFORE.start].ravel().astype(np.float32),
                ndvi_at_first_alert
            )

        first_alert_window = first_alert_window_new

        n_alert = int(fired.sum())
        mean_p  = float(y_prob[fired].mean()) if n_alert > 0 else 0
        log.info(f"  W{w:02d} ({d_before} -> {d_after}) alerts={n_alert:>7,}  mean_P={mean_p:.3f}")

        # Post-classification: update change class from best-P window
        window_change = classify_change_class(data, fired)
        better = y_prob > best_P_for_class
        best_change_class[better] = window_change[better]
        best_P_for_class[better] = y_prob[better]

    # Compute stacked score
    mean_P = np.where(n_fires > 0, sum_P / n_fires, 0)
    var    = np.where(n_fires > 1, (sum_P_sq / n_fires) - mean_P**2, 0)
    std_P  = np.sqrt(np.maximum(var, 0))
    consistency = np.where(mean_P > 0, 1 - std_P / mean_P, 0)
    consistency = np.clip(consistency, 0, 1)

    stacked_score = sum_P * (consistency ** 0.5)

    # min_fires filter (fast-track bypass)
    needs_min_fires = (n_fires < min_fires) & ~fast_track
    stacked_score[needs_min_fires] = 0

    ft_only = fast_track & (n_fires < min_fires)
    stacked_score[ft_only] = np.maximum(stacked_score[ft_only], fast_track_P[ft_only])

    n_ft = int(fast_track.sum())
    if n_ft > 0:
        log.info(f"Fast-Track: {n_ft:,} pixels bypassed min_fires")

    return stacked_score, n_fires, H, W, profile, transform, best_change_class


# ── Apply hysteresis + morphological cleaning ─────────────────────────────────

def apply_hysteresis(score_map_2d: np.ndarray, threshold: float,
                     min_pixels: int = 5) -> np.ndarray:
    """
    Object-Based Hysteresis:
      1. Seed at T/2 (permissive perimeter)
      2. Morphological closing (3x3) to glue fragments
      3. Remove small objects
      4. Keep polygon only if max_score >= threshold
    Returns: clean binary mask (H, W)
    """
    H, W = score_map_2d.shape

    permissive_mask = score_map_2d >= (threshold * 0.5)
    struct = np.ones((3, 3), dtype=bool)
    closed_mask = binary_closing(permissive_mask, structure=struct, iterations=1)

    # Remove small objects
    from scipy.ndimage import label as ndlabel
    labeled, n_feat = ndlabel(closed_mask.astype(np.int32))
    for i in range(1, n_feat + 1):
        if (labeled == i).sum() < min_pixels:
            closed_mask[labeled == i] = False

    # Hysteresis: keep only if max score >= threshold
    labeled2, n_feat2 = ndlabel(closed_mask.astype(np.int32))
    clean_mask = np.zeros_like(closed_mask)
    for i in range(1, n_feat2 + 1):
        component = labeled2 == i
        if score_map_2d[component].max() >= threshold:
            clean_mask[component] = True

    return clean_mask


# ── Write output raster ──────────────────────────────────────────────────────

def write_filtered_raster(out_path: str, score: np.ndarray,
                          n_fires: np.ndarray, confirmed: np.ndarray,
                          change_class: np.ndarray,
                          H: int, W: int, profile: dict):
    """
    Write 4-band filtered raster:
      Band 1: score (float32)
      Band 2: n_fires (int32 stored as float32)
      Band 3: confirmed mask (0/1 float32)
      Band 4: change_class (int: 0=none, 1=crops, 2=built, 3=degradation,
              4=shrub, 5=pure_loss, 6=other)
    """
    p = profile.copy()
    p.update(count=4, dtype="float32", compress="lzw",
             nodata=0)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(score.reshape(H, W).astype(np.float32), 1)
        dst.write(n_fires.reshape(H, W).astype(np.float32), 2)
        dst.write(confirmed.reshape(H, W).astype(np.float32), 3)
        dst.write(change_class.reshape(H, W).astype(np.float32), 4)
        dst.set_band_description(1, "stacked_score")
        dst.set_band_description(2, "n_fires")
        dst.set_band_description(3, "confirmed_mask")
        dst.set_band_description(4, "change_class")

    log.info(f"Filtered raster: {out_path}")
    log.info(f"  Band 1: stacked_score (float32)")
    log.info(f"  Band 2: n_fires (count)")
    log.info(f"  Band 3: confirmed_mask (binary 0/1)")
    log.info(f"  Band 4: change_class (0=none, 1=crops, 2=built, 3=degradation, 4=shrub, 5=pure_loss, 6=other)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Filter alert rasters -> filtered raster (TIF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single window
  python scripts/filter_alert_raster.py \\
      --alert-tifs data/alerts/alert_delta_*.tif \\
      --model outputs/alert_filter/model.lgbm \\
      --config outputs/alert_filter/feature_config.json \\
      --out-tif outputs/filtered/filtered_HAMEERPUR.tif

  # Then vectorize separately:
  python scripts/vectorize_raster.py \\
      --input-tif outputs/filtered/filtered_HAMEERPUR.tif \\
      --out-geojson outputs/filtered/alerts.geojson
        """
    )
    parser.add_argument("--alert-tifs", required=True, nargs="+",
                        help="One or more 10-band alert delta TIF files. "
                             "Use glob patterns or list multiple files.")
    parser.add_argument("--model", required=True, help="Path to model.lgbm")
    parser.add_argument("--config", required=True, help="Path to feature_config.json")
    parser.add_argument("--out-tif", required=True,
                        help="Output filtered raster path (.tif)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override threshold (default: auto from stacking)")
    parser.add_argument("--min-fires", type=int, default=2,
                        help="Min windows a pixel must fire (stacking mode)")
    parser.add_argument("--min-pixels", type=int, default=5,
                        help="Min cluster size in pixels")
    parser.add_argument("--no-hysteresis", action="store_true",
                        help="Disable object-based hysteresis (hard threshold)")
    args = parser.parse_args()

    t0 = time.time()

    # ── Expand globs ──────────────────────────────────────────────────
    tif_files = []
    for pattern in args.alert_tifs:
        expanded = glob.glob(pattern)
        if expanded:
            tif_files.extend(expanded)
        elif os.path.isfile(pattern):
            tif_files.append(pattern)
    tif_files = sorted(set(tif_files))

    if not tif_files:
        log.error("No alert TIF files found!")
        return

    log.info(f"Found {len(tif_files)} alert raster(s)")

    # ── Load config + model ───────────────────────────────────────────
    import lightgbm as lgb
    with open(args.config) as f:
        config = json.load(f)

    alert_thresh = config["alert_threshold"]
    model = lgb.Booster(model_file=args.model)
    log.info(f"Model loaded: {args.model}")

    # ── Filter ────────────────────────────────────────────────────────
    if len(tif_files) == 1:
        # Single window mode
        log.info("Mode: SINGLE WINDOW")
        y_prob, is_alert, data, H, W, profile, transform, change_class = filter_single(
            tif_files[0], model, alert_thresh,
            args.threshold or config.get("operational_threshold", 0.5)
        )
        score = y_prob
        n_fires = is_alert.astype(np.int32)

        threshold = args.threshold or config.get("operational_threshold", 0.5)
    else:
        # Multi-window stacking mode
        log.info(f"Mode: TEMPORAL STACKING ({len(tif_files)} windows)")
        score, n_fires, H, W, profile, transform, change_class = stack_windows(
            tif_files, model, alert_thresh, args.min_fires
        )

        # Auto-select threshold if not specified
        active = score > 0
        if args.threshold:
            threshold = args.threshold
        elif active.any():
            threshold = float(np.percentile(score[active], 80))
            log.info(f"Auto-selected threshold: {threshold:.3f} (80th percentile)")
        else:
            threshold = 0.5

    log.info(f"Threshold: {threshold:.3f}")

    # ── Apply hysteresis + build confirmed mask ───────────────────────
    score_2d = score.reshape(H, W)
    if args.no_hysteresis:
        confirmed = (score_2d >= threshold).astype(np.float32)
        log.info("Hysteresis: DISABLED (hard threshold)")
    else:
        confirmed = apply_hysteresis(score_2d, threshold, args.min_pixels).astype(np.float32)
        n_confirmed = int(confirmed.sum())
        log.info(f"Confirmed pixels (hysteresis): {n_confirmed:,}")

    # ── Write output ──────────────────────────────────────────────────
    # Zero out change_class for non-confirmed pixels
    change_class_out = change_class.ravel().copy()
    change_class_out[confirmed.ravel() < 0.5] = 0

    write_filtered_raster(
        args.out_tif, score, n_fires.astype(np.float32).ravel(),
        confirmed.ravel(), change_class_out, H, W, profile
    )

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    n_total = H * W
    n_confirmed = int(confirmed.sum())
    n_active = int((score > 0).sum())

    log.info(f"\n{'='*65}")
    log.info(f"  FILTERING COMPLETE in {elapsed:.1f}s")
    log.info(f"  Input:     {len(tif_files)} alert raster(s), {H}x{W} pixels")
    log.info(f"  Active:    {n_active:,} pixels with score > 0")
    log.info(f"  Confirmed: {n_confirmed:,} pixels")
    log.info(f"  Threshold: {threshold:.3f}")

    # Change class breakdown for confirmed alerts
    if n_confirmed > 0:
        log.info(f"\n  Change Class Breakdown (confirmed alerts):")
        for cls_id, cls_name in CHANGE_CLASS.items():
            if cls_id == 0:
                continue
            count = int((change_class_out == cls_id).sum())
            if count > 0:
                pct = 100 * count / n_confirmed
                is_phenology = " [PHENOLOGY]" if cls_id in (3, 4) else ""
                log.info(f"    {cls_id}: {cls_name:<30s} {count:>7,} ({pct:5.1f}%){is_phenology}")
    log.info(f"  Output:    {args.out_tif}")
    log.info(f"{'='*65}")
    log.info(f"\nNext step: vectorize with:")
    log.info(f"  python scripts/vectorize_raster.py \\")
    log.info(f"      --input-tif {args.out_tif} \\")
    log.info(f"      --out-geojson {os.path.splitext(args.out_tif)[0]}.geojson")


if __name__ == "__main__":
    main()
