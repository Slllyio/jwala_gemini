"""
Enhanced Alert Windows — GT Analysis Improvements
===================================================

Implements three improvements from the deep GT timing analysis:

1. LONGER ROLLING WINDOWS (6-month instead of 3-month)
   - Captures gradual forest loss that accumulates over >100 days
   - Uses 3 overlapping 6-month windows per year

2. SEASONAL ADJUSTMENT FACTORS
   - Normalizes DW band deltas by season to reduce false positives
   - Monsoon greening and post-monsoon senescence corrections

3. MULTI-SCALE DETECTION
   - Pixel-level + 3x3 patch-level + 5x5 neighborhood aggregation
   - Catches diffuse loss patterns missed by single-pixel detectors

Usage:
    python scripts/enhanced_alert_windows.py \\
        --input data/processed/dw_timeseries.tif \\
        --config config.yaml \\
        --output outputs/enhanced_alerts.geojson
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()

import argparse
import json
import logging
from typing import List, Optional
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# DW band names
DW_BANDS = ["water", "trees", "grass", "flooded_veg",
            "crops", "shrub_scrub", "built", "bare"]
TREES_IDX = 1  # Index of 'trees' band
N_BANDS = 8

# Seasonal normalisation factors — calibrated to Indian dry deciduous phenology
# (Rajasthan–Madhya Pradesh Teak/Sal dry forest belt, Oct–Feb leaf-fall cycle).
#
# Literature basis:
#   - Leaf fall: Oct (early) → Nov–Dec (peak) → Jan–Feb (recovery starts)
#   - Leaf flush: Feb–May (pre-monsoon), max NDVI Jul–Sep (monsoon)
#   - DW "trees" probability range in MP: 0.6–0.85 (full canopy) → 0.35–0.55 (leafless)
#   - Natural October->January trees delta: −0.15 to −0.25 (SOURCE: field & MODIS phenology)
#
# Format: {season: {band: (mean_natural_delta, std_natural_delta)}}
# Anything beyond (mean ± 2*std) is flagged as anomalous.
#
# CRITICAL FIX vs. previous values:
#   Old Winter/trees = (−0.02, 0.08)  ← WRONG: only captured 5% of real senescence
#   New dry_cool/trees = (−0.18, 0.10) ← Correct: matches Oct–Feb MP phenology data
SEASONAL_NORMS = {
    # Nov–Feb: peak leaf fall. Trees shed 15–25% DW probability naturally.
    # Bare soil rises as canopy opens. Rabi crops (wheat/chickpea) green up.
    "dry_cool": {
        "trees":        (-0.18, 0.10),  # STRONG natural drop — leaf fall peak
        "grass":        (-0.08, 0.07),  # dormancy
        "crops":        (+0.07, 0.09),  # rabi planting and growth
        "bare":         (+0.12, 0.09),  # exposed soil through open canopy
        "shrub_scrub":  (-0.06, 0.07),  # semi-deciduous understorey
        "water":        (0.00,  0.04),
        "built":        (0.00,  0.03),
        "flooded_veg":  (-0.03, 0.05),
    },
    # Mar–May: leaf flush. Trees recover before monsoon. Rabi harvest → bare.
    "dry_hot": {
        "trees":        (+0.14, 0.10),  # active flush — pre-monsoon recovery
        "grass":        (-0.06, 0.08),  # drying out
        "crops":        (-0.14, 0.11),  # rabi harvest (bare after)
        "bare":         (+0.08, 0.09),  # post-harvest + rocky outcrops
        "shrub_scrub":  (+0.05, 0.07),
        "water":        (-0.02, 0.05),
        "built":        (0.00,  0.03),
        "flooded_veg":  (-0.02, 0.04),
    },
    # Jun–Sep: monsoon. Max NDVI. All vegetation at peak. Optical gaps due to clouds.
    "monsoon": {
        "trees":        (+0.08, 0.07),  # stable high canopy (already near peak)
        "grass":        (+0.18, 0.13),  # strong greening from bare ground
        "crops":        (+0.22, 0.14),  # kharif planting peaks
        "bare":         (-0.16, 0.10),  # bare ground turns green
        "shrub_scrub":  (+0.08, 0.07),
        "water":        (+0.05, 0.08),  # river levels rise
        "built":        (0.00,  0.03),
        "flooded_veg":  (+0.08, 0.09),  # waterlogged areas expand
    },
    # Oct: transition — early leaf fall starts, kharif harvest, post-monsoon stable.
    "post_monsoon": {
        "trees":        (-0.05, 0.07),  # early senescence (not full drop yet)
        "grass":        (-0.07, 0.08),  # beginning to dry
        "crops":        (-0.12, 0.10),  # kharif harvest
        "bare":         (+0.07, 0.07),  # post-harvest exposure
        "shrub_scrub":  (-0.03, 0.06),
        "water":        (-0.03, 0.06),  # water recedes
        "built":        (0.00,  0.03),
        "flooded_veg":  (-0.05, 0.06),  # inundation ends
    },
}

SEASON_MAP = {
    1:  "dry_cool",      # January   — peak leafless
    2:  "dry_cool",      # February  — late leaf fall / early flush
    3:  "dry_hot",       # March     — flush begins
    4:  "dry_hot",       # April     — pre-monsoon heat
    5:  "dry_hot",       # May       — max pre-monsoon temps, full leaf-out
    6:  "monsoon",       # June      — SW monsoon onset
    7:  "monsoon",       # July      — peak monsoon, max NDVI
    8:  "monsoon",       # August    — monsoon active
    9:  "monsoon",       # September — monsoon withdrawal starts
    10: "post_monsoon",  # October   — early leaf fall, kharif harvest
    11: "dry_cool",      # November  — peak leaf fall begins
    12: "dry_cool",      # December  — max leafless
}


def get_season(month: int) -> str:
    """Get season name from month."""
    return SEASON_MAP.get(month, "Unknown")


def compute_rolling_deltas(timeseries, window_months: int = 6):
    """
    Compute rolling deltas over longer windows.

    Args:
        timeseries: array of shape (T, bands, H, W)
        window_months: rolling window length in months

    Returns:
        deltas: array of shape (n_windows, bands, H, W)
        windows: list of (start_idx, end_idx) tuples
    """
    T = timeseries.shape[0]

    # With monthly composites, window_months = number of steps
    step = max(1, window_months // 2)  # 50% overlap
    windows = []
    for start in range(0, T - window_months + 1, step):
        end = start + window_months
        windows.append((start, end))

    if not windows:
        windows = [(0, T)]

    deltas = np.zeros((len(windows),) + timeseries.shape[1:], dtype=np.float32)
    for i, (s, e) in enumerate(windows):
        # Delta = end mean - start mean (using first/last thirds)
        third = max(1, (e - s) // 3)
        start_mean = np.nanmean(timeseries[s:s+third], axis=0)
        end_mean = np.nanmean(timeseries[e-third:e], axis=0)
        deltas[i] = end_mean - start_mean

    log.info(f"Computed {len(windows)} rolling windows ({window_months}-month, "
             f"step={step})")
    return deltas, windows


def apply_seasonal_adjustment(deltas, month: int):
    """
    Normalise deltas using seasonal expected values.

    Adjusted delta = (raw_delta - seasonal_mean) / seasonal_std

    Values > 2.0 indicate anomalous change beyond natural variation.
    """
    season = get_season(month)
    norms = SEASONAL_NORMS.get(season, {})
    adjusted = np.copy(deltas)

    for band_idx, band_name in enumerate(DW_BANDS):
        if band_name in norms:
            mean, std = norms[band_name]
            if std > 0:
                adjusted[band_idx] = (deltas[band_idx] - mean) / std

    return adjusted


def multi_scale_detection(confidence_map, thresholds=(0.5, 0.4, 0.35)):
    """
    Multi-scale detection: pixel + 3x3 + 5x5 neighborhood.

    Combines single-pixel high-confidence detections with
    spatially-aggregated lower-confidence patterns.

    Args:
        confidence_map: (H, W) confidence scores
        thresholds: (pixel_thresh, patch3_thresh, patch5_thresh)

    Returns:
        combined_mask: boolean (H, W) — detected pixels
        combined_confidence: float (H, W) — combined confidence
    """
    from scipy.ndimage import uniform_filter

    H, W = confidence_map.shape
    pixel_thresh, patch3_thresh, patch5_thresh = thresholds

    # Scale 1: pixel-level
    pixel_mask = confidence_map >= pixel_thresh

    # Scale 2: 3x3 patch mean
    patch3_mean = uniform_filter(confidence_map, size=3, mode='constant')
    patch3_mask = patch3_mean >= patch3_thresh

    # Scale 3: 5x5 neighborhood mean
    patch5_mean = uniform_filter(confidence_map, size=5, mode='constant')
    patch5_mask = patch5_mean >= patch5_thresh

    # Combined: pixel OR (patch3 AND patch5)
    combined_mask = pixel_mask | (patch3_mask & patch5_mask)

    # Combined confidence: max of all scales
    combined_confidence = np.maximum.reduce([
        confidence_map * pixel_mask,
        patch3_mean * patch3_mask,
        patch5_mean * patch5_mask,
    ])

    n_pixel = pixel_mask.sum()
    n_patch = combined_mask.sum() - n_pixel
    log.info(f"Multi-scale detection: {n_pixel} pixel + {n_patch} patch = "
             f"{combined_mask.sum()} total")

    return combined_mask, combined_confidence


def compute_gradual_loss_score(timeseries, trees_idx: int = TREES_IDX):
    """
    Detect gradual loss by measuring cumulative tree cover decline.

    Pixels with monotonic decline over 3+ consecutive windows
    get a gradual_loss_score even if per-window deltas are small.
    """
    T = timeseries.shape[0]
    if T < 3:
        return np.zeros(timeseries.shape[2:], dtype=np.float32)

    trees = timeseries[:, trees_idx]  # (T, H, W)

    # Count consecutive declining windows
    diffs = np.diff(trees, axis=0)  # (T-1, H, W)
    declining = diffs < -0.01  # threshold for "declining"

    # Cumulative consecutive decline count
    max_consecutive = np.zeros(trees.shape[1:], dtype=np.int32)
    current_streak = np.zeros(trees.shape[1:], dtype=np.int32)

    for t in range(declining.shape[0]):
        current_streak = np.where(declining[t], current_streak + 1, 0)
        max_consecutive = np.maximum(max_consecutive, current_streak)

    # Total decline magnitude
    total_decline = trees[0] - trees[-1]

    # Score: streak * magnitude (normalised)
    gradual_score = (max_consecutive / max(T, 1)) * np.clip(total_decline, 0, 1)

    n_gradual = (gradual_score > 0.1).sum()
    log.info(f"Gradual loss: {n_gradual} pixels with score > 0.1")

    return gradual_score.astype(np.float32)


def compute_yoy_monsoon_recovery(timeseries, months: list,
                                  trees_idx: int = TREES_IDX,
                                  monsoon_months: tuple = (6, 7, 8, 9),
                                  recovery_thresh: float = 0.65) -> np.ndarray:
    """
    Year-over-year monsoon recovery check.

    KEY INSIGHT: After natural leaf fall, trees rebound to ≥0.65–0.80 DW
    probability in the following monsoon (Jun–Sep). After real deforestation,
    the cleared pixel stays below 0.50 in the next monsoon season because
    there are no trees left to flush new leaves.

    Args:
        timeseries:       (T, bands, H, W) array — monthly composites
        months:           list of calendar month integers (1–12) for each T
        trees_idx:        band index for 'trees' in DW ordering
        monsoon_months:   tuple of month indices considered monsoon peak
        recovery_thresh:  DW trees probability threshold to call it "recovered"

    Returns:
        no_recovery_mask: (H, W) float32 — 1.0 where monsoon recovery absent
                          (= likely real deforestation, not natural senescence)
    """
    T = timeseries.shape[0]
    if T < 2 or not months:
        return np.zeros(timeseries.shape[2:], dtype=np.float32)

    months_arr = np.array(months[:T])
    trees = timeseries[:, trees_idx]  # (T, H, W)

    # Find first monsoon-season timestep
    monsoon_mask = np.isin(months_arr, list(monsoon_months))
    monsoon_idxs = np.where(monsoon_mask)[0]

    if len(monsoon_idxs) == 0:
        log.debug("YoY recovery: no monsoon months in this time window")
        return np.zeros(timeseries.shape[2:], dtype=np.float32)

    # Mean tree cover during monsoon timesteps
    monsoon_trees = trees[monsoon_idxs].mean(axis=0)  # (H, W)

    # Pixels below recovery threshold during monsoon → likely real loss
    no_recovery = (monsoon_trees < recovery_thresh).astype(np.float32)

    # Also flag pixels where monsoon value is lower than overall timeseries mean
    # (indicates a persistent downward shift, not seasonal fluctuation)
    overall_mean = trees.mean(axis=0)
    below_mean   = (monsoon_trees < overall_mean * 0.85).astype(np.float32)

    combined = np.maximum(no_recovery, below_mean)

    n_no_recov = int(combined.sum())
    log.info(f"YoY monsoon recovery check: {n_no_recov} pixels show no recovery "
             f"(monsoon trees < {recovery_thresh:.2f})")

    return combined


def compute_sar_cusum_score(vh_timeseries: np.ndarray,
                             sensitivity: float = 1.5) -> np.ndarray:
    """
    CuSum (Cumulative Sum) detector on Sentinel-1 VH backscatter.

    Detects gradual VH drops that accumulate over months — exactly the
    pattern that optical 3-month windows miss in gradual deforestation.

    VH physics:
      Forest canopy → volume scattering → high VH (≈ −10 to −8 dB)
      Bare ground   → surface scattering  → low  VH (≈ −18 to −14 dB)
      Gradual loss  → 1–3 dB drop per month over 4–8 months

    Args:
        vh_timeseries: (T, H, W) float32 — VH backscatter in dB, monthly mean
        sensitivity:   CUSUM threshold multiplier (lower = more sensitive)

    Returns:
        cusum_score: (H, W) float32 — normalised cumulative anomaly score [0..1]
    """
    T = vh_timeseries.shape[0]
    if T < 3:
        return np.zeros(vh_timeseries.shape[1:], dtype=np.float32)

    # Reference: mean + std of first third ("stable forest" baseline)
    ref_len   = max(1, T // 3)
    ref_mean  = np.nanmean(vh_timeseries[:ref_len], axis=0)    # (H, W)
    ref_std   = np.nanstd(vh_timeseries[:ref_len], axis=0) + 0.5  # floor

    # CuSum: accumulate (observation - expected) when observation is low
    cusum = np.zeros_like(ref_mean)  # (H, W)
    max_cusum = np.zeros_like(cusum)

    for t in range(ref_len, T):
        # Standardised anomaly: negative values = VH drop
        anomaly = (ref_mean - vh_timeseries[t]) / ref_std  # positive = drop below ref
        cusum  = np.maximum(0, cusum + anomaly - sensitivity)  # reset at zero
        max_cusum = np.maximum(max_cusum, cusum)

    # Normalise to [0, 1]
    m = max_cusum.max()
    if m > 0:
        cusum_score = np.clip(max_cusum / m, 0, 1).astype(np.float32)
    else:
        cusum_score = max_cusum.astype(np.float32)

    n_alert = int((cusum_score > 0.5).sum())
    log.info(f"SAR CuSum: {n_alert} pixels with cusum_score > 0.5")

    return cusum_score


def process_alert_raster(
    input_path: str,
    config: dict,
    window_months: int = 6,
    output_path: Optional[str] = None,
    months: Optional[List[int]] = None,
    vh_timeseries: Optional[np.ndarray] = None,
):
    """
    Process a DW timeseries raster with enhanced detection.

    Args:
        input_path:    Path to DW timeseries GeoTIFF (bands×T stacked)
        config:        Pipeline config dict
        window_months: Rolling window length in months (default: 6)
        output_path:   Output GeoJSON path (optional)
        months:        List of calendar month integers (1-12) per timestep,
                       used for year-over-year monsoon recovery check.
                       If None, the recovery check is skipped.
        vh_timeseries: (T, H, W) Sentinel-1 VH backscatter in dB (optional).
                       If provided, CuSum score is fused into final confidence.

    Returns:
        detection_mask, confidence_map, metadata
    """
    import rasterio

    log.info(f"Loading: {input_path}")
    with rasterio.open(input_path) as src:
        data = src.read().astype(np.float32)  # (bands*T, H, W)
        profile = src.profile.copy()
        transform = src.transform
        crs_str = str(src.crs)

    # Reshape to (T, bands, H, W)
    total_bands = data.shape[0]
    n_timesteps = total_bands // N_BANDS
    if n_timesteps < 2:
        log.warning(f"Only {n_timesteps} timestep(s) — need at least 2")
        return None, None, None

    timeseries = data[:n_timesteps * N_BANDS].reshape(
        n_timesteps, N_BANDS, data.shape[1], data.shape[2])
    log.info(f"Timeseries: {timeseries.shape} (T={n_timesteps}, bands={N_BANDS})")

    # ── 1. Rolling deltas (6-month windows) ────────────────────────────────────
    deltas, windows = compute_rolling_deltas(timeseries, window_months=window_months)

    # ── 2. Seasonal adjustment on each window ──────────────────────────────────
    for i, (s, e) in enumerate(windows):
        mid_month = ((s + e) // 2) % 12 + 1
        deltas[i] = apply_seasonal_adjustment(deltas[i], mid_month)

    # ── 3. Aggregate — max anomaly across all windows ──────────────────────────
    max_anomaly = np.max(np.abs(deltas[:, TREES_IDX]), axis=0)  # (H, W)
    if max_anomaly.max() > 0:
        confidence_map = np.clip(max_anomaly / max_anomaly.max(), 0, 1)
    else:
        confidence_map = max_anomaly

    # ── 4. Gradual monotonic loss score ────────────────────────────────────────
    gradual_score = compute_gradual_loss_score(timeseries)

    # ── 5. Year-over-year monsoon recovery (key false-positive reducer) ────────
    #  Natural leaf fall → full recovery in next monsoon → weight down.
    #  Real deforestation → no recovery → weight stays.
    if months is not None:
        no_recovery_mask = compute_yoy_monsoon_recovery(timeseries, months)
        # Suppress confidence in pixels that fully recover — they are phenological
        # Boost confidence in pixels that do NOT recover — likely real loss
        confidence_map = confidence_map * np.where(
            no_recovery_mask > 0.5,
            1.20,   # boost real-loss pixels (up to max 1.0 after clip)
            0.60,   # dampen natural-phenology pixels
        )
        confidence_map = np.clip(confidence_map, 0, 1)
        n_boosted = int((no_recovery_mask > 0.5).sum())
        log.info(f"YoY recovery: boosted {n_boosted} permanent-loss pixels, "
                 f"dampened {int((no_recovery_mask <= 0.5).sum() - (no_recovery_mask==0).sum())} phenological pixels")
    else:
        no_recovery_mask = None
        log.info("YoY recovery check skipped (no months list provided)")

    # ── 6. SAR CuSum fusion ────────────────────────────────────────────────────
    if vh_timeseries is not None:
        cusum_score = compute_sar_cusum_score(vh_timeseries)
        # Fuse: any pixel flagged by EITHER optical anomaly OR SAR CuSum
        combined_confidence = np.maximum(
            np.maximum(confidence_map, gradual_score),
            cusum_score,
        )
        log.info("SAR CuSum fused into confidence map")
    else:
        combined_confidence = np.maximum(confidence_map, gradual_score)
        cusum_score = None

    # ── 7. Multi-scale detection ───────────────────────────────────────────────
    detection_mask, final_confidence = multi_scale_detection(combined_confidence)

    metadata = {
        "n_timesteps":    n_timesteps,
        "window_months":  window_months,
        "n_windows":      len(windows),
        "n_detected":     int(detection_mask.sum()),
        "n_gradual":      int((gradual_score > 0.1).sum()),
        "n_sar_cusum":    int((cusum_score > 0.5).sum()) if cusum_score is not None else 0,
        "n_no_recovery":  int((no_recovery_mask > 0.5).sum()) if no_recovery_mask is not None else 0,
        "mean_confidence": float(np.mean(final_confidence[detection_mask]))
                           if detection_mask.any() else 0.0,
        "yoy_recovery_check": months is not None,
        "sar_cusum_used":     vh_timeseries is not None,
    }

    log.info(f"Detection results: {metadata}")

    # ── 8. Save enhanced confidence raster (3 bands) ──────────────────────────
    if output_path:
        out_profile = profile.copy()
        n_out_bands = 3 if cusum_score is not None else 2
        out_profile.update(count=n_out_bands, dtype="float32")
        raster_out = output_path.replace(".geojson", "_confidence.tif")
        with rasterio.open(raster_out, "w", **out_profile) as dst:
            dst.write(final_confidence, 1)  # band 1: combined detection confidence
            dst.write(gradual_score,    2)  # band 2: monotonic loss score
            if cusum_score is not None:
                dst.write(cusum_score,  3)  # band 3: SAR CuSum score
            dst.update_tags(
                band1="combined_confidence",
                band2="gradual_loss_score",
                band3="sar_cusum_score" if cusum_score is not None else "unused",
            )
        log.info(f"Confidence raster saved: {raster_out}")

    return detection_mask, final_confidence, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Van Suraksha -- Enhanced Alert Windows")
    parser.add_argument("--input", required=True,
                        help="DW timeseries raster (bands*T, H, W)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--window-months", type=int, default=6,
                        help="Rolling window length in months (default: 6)")
    parser.add_argument("--output", default="outputs/enhanced_alerts.geojson",
                        help="Output GeoJSON path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    detection_mask, confidence, metadata = process_alert_raster(
        args.input, cfg,
        window_months=args.window_months,
        output_path=args.output,
    )

    if detection_mask is None:
        log.error("Detection failed — insufficient data")
        return

    # Use existing filter_alerts.generate_geojson if available
    try:
        from scripts.filter_alerts import generate_geojson
        import rasterio

        with rasterio.open(args.input) as src:
            alert_data = src.read().astype(np.float32)
            transform = src.transform
            crs_str = str(src.crs)

        geojson = generate_geojson(
            detection_mask, confidence, alert_data,
            transform, crs_str,
            detection_date=datetime.now().strftime("%Y-%m-%d"),
            sub_range="Enhanced Detection",
            min_pixels=3,
        )

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(geojson, f)

        n_features = len(geojson.get("features", []))
        log.info(f"[DONE] {n_features} polygons -> {output}")

    except Exception as e:
        log.warning(f"GeoJSON generation skipped: {e}")
        log.info(f"Detection complete: {metadata['n_detected']} pixels detected")


if __name__ == "__main__":
    main()
