"""
Stack Alert Scores — Temporal Accumulation Across All Windows
==============================================================

Runs the trained LightGBM model on ALL alert windows, then accumulates
per-pixel scores across time. Real deforestation fires in multiple
consecutive windows; noise is ephemeral and fires once or twice.

Per-pixel temporal features accumulated:
  - n_fires:           how many windows this pixel triggered an alert
  - sum_P:             cumulative sum of P(real) across all fires
  - mean_P:            average P(real) across fires
  - max_P:             peak confidence in any single window
  - consistency:       1 - std(P)/mean(P)  (low variance = consistent signal)
  - max_streak:        longest consecutive-window firing streak
  - first_window:      earliest window index that fired
  - last_window:       latest window index that fired

Final score:  stacked_score = sum_P * (consistency ** 0.5)

This rewards pixels that fire repeatedly with stable confidence.
A pixel with n_fires=5 and mean_P=0.4 gets a higher stacked score than
a pixel with n_fires=1 and P=0.8 — as it should for deforestation detection.

Production Hardening:
  - Magnitude Fast-Track: P>0.90 AND delta_bare>0.20 in any single window
    bypasses the min_fires requirement — timber smuggler detection.
  - Object-Based Hysteresis: morphological closing glues fragmented
    micro-polygons, then polygons are kept if max_score >= threshold.
  - Last-window deltas used for change type attribution (not washed-out
    weighted averages that dilute the signal).

Usage:
    python scripts/stack_alert_scores.py \\
        --data-dir data/ground_truth/HAMEERPUR \\
        --model outputs/alert_filter/model.lgbm \\
        --config outputs/alert_filter/feature_config.json \\
        --out-dir outputs/alert_filter/stacked
"""

# ── PROJ/GDAL env fix ──
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()
del _os, _sys
# ────────────────────────

import argparse
import json
import logging
import os
import re
import time
from glob import glob
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
from scipy.ndimage import uniform_filter, label as ndlabel, binary_closing

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants (mirrored from train_alert_filter.py) ───────────────────────────

DW_BANDS = ["water", "trees", "grass", "flooded_veg",
            "crops", "shrub_scrub", "built", "bare"]
TREES_IDX = 1
N_BANDS   = 8

SEASON_MAP = {
    1: "Winter", 2: "Winter",
    3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
    12: "Winter",
}


def parse_dates(filename: str):
    m = re.search(r"(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})", filename)
    if m:
        return m.group(1), m.group(2)
    return "unknown", "unknown"


# ── Feature extraction (identical to train_alert_filter.py) ───────────────────

def extract_features(data: np.ndarray, month: int,
                     alert_thresh: float, H: int, W: int) -> np.ndarray:
    """Extract 23 features from a single alert raster."""
    n_pix = H * W
    has_cloud = data.shape[0] >= 10

    deltas = data[:N_BANDS].astype(np.float32)
    delta_flat = deltas.reshape(N_BANDS, n_pix)

    if has_cloud:
        cld_before = data[8].ravel().astype(np.float32)
        cld_after  = data[9].ravel().astype(np.float32)
    else:
        cld_before = np.zeros(n_pix, dtype=np.float32)
        cld_after  = np.zeros(n_pix, dtype=np.float32)
    cld_worst = np.maximum(cld_before, cld_after)

    trees_2d = deltas[TREES_IDX]
    alert_mask_2d = (np.abs(trees_2d) >= alert_thresh).astype(np.float32)
    neighbor_sum = uniform_filter(alert_mask_2d, size=3, mode="constant") * 9
    n_neighbors = np.clip((neighbor_sum - alert_mask_2d).ravel(), 0, 8)
    mean_nb_trees = uniform_filter(trees_2d, size=3, mode="constant").ravel()
    mean_sq = uniform_filter(trees_2d**2, size=3, mode="constant").ravel()
    std_nb_trees = np.sqrt(np.maximum(mean_sq - mean_nb_trees**2, 0))

    trees_delta = delta_flat[TREES_IDX]
    crops_delta = delta_flat[4]
    bare_delta  = delta_flat[7]
    trees_crops_anti = ((trees_delta < 0) & (crops_delta > 0)).astype(np.float32)
    trees_bare_anti  = ((trees_delta < 0) & (bare_delta > 0)).astype(np.float32)

    gain_deltas = delta_flat.copy()
    gain_deltas[TREES_IDX] = -999
    dominant_gain = np.argmax(gain_deltas, axis=0).astype(np.float32)
    band_div = (np.abs(delta_flat) > alert_thresh).sum(axis=0).astype(np.float32)

    abs_trees = np.abs(trees_delta)
    abs_max = np.max(np.abs(delta_flat), axis=0)

    season = SEASON_MAP.get(month, "Unknown")
    month_arr = np.full(n_pix, month, dtype=np.float32)
    is_monsoon = np.full(n_pix, 1.0 if season == "Monsoon" else 0.0, dtype=np.float32)
    is_winter = np.full(n_pix, 1.0 if season == "Winter" else 0.0, dtype=np.float32)

    return np.column_stack([
        delta_flat[0], delta_flat[1], delta_flat[2], delta_flat[3],
        delta_flat[4], delta_flat[5], delta_flat[6], delta_flat[7],
        cld_before, cld_after, cld_worst,
        n_neighbors, mean_nb_trees, std_nb_trees,
        trees_crops_anti, trees_bare_anti, dominant_gain, band_div,
        abs_trees, abs_max,
        month_arr, is_monsoon, is_winter,
    ])


# ── Change type (polygon-level, same as filter_alerts.py) ─────────────────────

def classify_change_type(mean_deltas: dict) -> str:
    trees = mean_deltas.get("trees", 0)
    crops = mean_deltas.get("crops", 0)
    bare  = mean_deltas.get("bare", 0)
    built = mean_deltas.get("built", 0)
    shrub = mean_deltas.get("shrub_scrub", 0)
    grass = mean_deltas.get("grass", 0)

    if trees > 0.05:
        return "Greening"
    if trees < -0.05:
        gains = {"crops": crops, "bare": bare, "built": built,
                 "shrub_scrub": shrub, "grass": grass}
        top_band = max(gains, key=gains.get)
        top_val  = gains[top_band]
        if top_band == "crops" and top_val > 0.03:
            return "Encroachment"
        elif top_band == "built" and top_val > 0.03:
            return "Built expansion"
        elif top_band == "bare" and top_val > 0.03:
            return "Clearing"
        elif top_band in ("shrub_scrub", "grass") and top_val > 0.03:
            return "Degradation"
        else:
            return "Tree loss (unclassified)"
    return "Other change"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="outputs/alert_filter/stacked")
    parser.add_argument("--stacked-threshold", type=float, default=None,
                        help="Threshold on stacked_score. Auto-selected if omitted.")
    parser.add_argument("--min-fires", type=int, default=2,
                        help="Minimum windows a pixel must fire to be considered")
    parser.add_argument("--min-pixels", type=int, default=5,
                        help="Min cluster size")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    import lightgbm as lgb

    with open(args.config) as f:
        config = json.load(f)
    alert_thresh = config["alert_threshold"]

    model = lgb.Booster(model_file=args.model)
    log.info(f"Model loaded: {args.model}")

    # ── Load all alert rasters ───────────────────────────────────────
    pattern = os.path.join(args.data_dir, "alerts", "alert_delta_*.tif")
    files = sorted(glob(pattern))
    if not files:
        pattern = os.path.join(args.data_dir, "alert_delta_*.tif")
        files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No alert rasters in {args.data_dir}")

    log.info(f"Found {len(files)} alert rasters")

    # ── Load GT for evaluation ───────────────────────────────────────
    gt_files = glob(os.path.join(args.data_dir, "gt_delta_*.tif"))
    gt = None
    if gt_files:
        with rasterio.open(gt_files[0]) as ds:
            gt = ds.read()
        log.info(f"GT loaded: {gt.shape}")

    # ── First pass: get dimensions ───────────────────────────────────
    with rasterio.open(files[0]) as ds:
        first = ds.read()
        profile = ds.profile.copy()
        transform = ds.transform
    H, W = first.shape[1], first.shape[2]
    n_pix = H * W
    N = len(files)

    # ── Per-pixel accumulators ───────────────────────────────────────
    n_fires      = np.zeros(n_pix, dtype=np.int32)
    sum_P        = np.zeros(n_pix, dtype=np.float32)
    sum_P_sq     = np.zeros(n_pix, dtype=np.float32)
    max_P        = np.zeros(n_pix, dtype=np.float32)
    first_fire   = np.full(n_pix, -1, dtype=np.int32)
    last_fire    = np.full(n_pix, -1, dtype=np.int32)
    cur_streak   = np.zeros(n_pix, dtype=np.int32)
    max_streak   = np.zeros(n_pix, dtype=np.int32)
    prev_fired   = np.zeros(n_pix, dtype=bool)

    # ── Magnitude Fast-Track accumulator ─────────────────────────────
    # Pixels with P>0.90 AND delta_bare>0.20 in any single window
    # bypass min_fires — timber smuggler / bulldozer detection
    fast_track = np.zeros(n_pix, dtype=bool)
    fast_track_P = np.zeros(n_pix, dtype=np.float32)  # store the P value
    n_fast_tracked = 0

    # Accumulate per-window deltas (keep the last-window data for change type)
    last_window_deltas = np.zeros((N_BANDS, n_pix), dtype=np.float32)
    last_window_date = "unknown"

    log.info(f"\n{'='*65}")
    log.info(f"  TEMPORAL STACKING: {N} windows")
    log.info(f"{'='*65}")

    for w, fpath in enumerate(files):
        with rasterio.open(fpath) as ds:
            data = ds.read()

        d_before, d_after = parse_dates(Path(fpath).stem)
        month = int(d_before.split("-")[1]) if d_before != "unknown" else 1

        # Extract features & predict
        features = extract_features(data, month, alert_thresh, H, W)
        y_prob = model.predict(features).astype(np.float32)

        # Alert mask
        trees_abs = np.abs(data[TREES_IDX].ravel()).astype(np.float32)
        is_alert = trees_abs >= alert_thresh

        # Zero out non-alerting pixels
        y_prob[~is_alert] = 0.0

        # Accumulate
        fired = is_alert
        n_fires += fired.astype(np.int32)
        sum_P   += y_prob
        sum_P_sq += y_prob ** 2
        max_P    = np.maximum(max_P, y_prob)

        # First/last fire
        first_fire_mask = fired & (first_fire == -1)
        first_fire[first_fire_mask] = w
        last_fire[fired] = w

        # Streak tracking
        continuing = fired & prev_fired
        starting   = fired & ~prev_fired
        cur_streak[continuing] += 1
        cur_streak[starting]    = 1
        cur_streak[~fired]      = 0
        max_streak = np.maximum(max_streak, cur_streak)
        prev_fired = fired.copy()

        # ── Magnitude Fast-Track detection ───────────────────────────
        # A single window with P>0.90 AND delta_bare>0.20 indicates
        # violent, undeniable canopy destruction + exposed soil.
        # Don't make the forest guard wait 15 days for corroboration.
        deltas = data[:N_BANDS].reshape(N_BANDS, n_pix).astype(np.float32)
        bare_delta = deltas[7]  # bare soil band
        trees_delta = deltas[TREES_IDX]
        ft_mask = (
            fired &
            (y_prob > 0.90) &
            (bare_delta > 0.20) &
            (trees_delta < -0.15)
        )
        n_new_ft = int(ft_mask.sum()) - int((ft_mask & fast_track).sum())
        fast_track |= ft_mask
        fast_track_P = np.maximum(fast_track_P, y_prob * ft_mask.astype(np.float32))
        if n_new_ft > 0:
            n_fast_tracked += n_new_ft

        # Keep the last window's deltas for change type attribution
        # (more recent = more representative of current state)
        has_alert_this_window = fired
        for b in range(N_BANDS):
            last_window_deltas[b, has_alert_this_window] = deltas[b, has_alert_this_window]

        last_window_date = d_after if d_after != "unknown" else "unknown"

        n_alert  = int(fired.sum())
        mean_p   = float(y_prob[fired].mean()) if n_alert > 0 else 0
        n_ft     = int(ft_mask.sum())
        log.info(
            f"  W{w:02d} ({d_before} → {d_after}) "
            f"alerts={n_alert:>7,}  mean_P={mean_p:.3f}"
            f"{'  [FT=' + str(n_ft) + ']' if n_ft > 0 else ''}"
        )

    # ── Compute stacked scores ───────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info(f"  COMPUTING STACKED SCORES")
    log.info(f"{'='*65}")

    # Mean P
    has_fires = n_fires > 0
    mean_P = np.zeros(n_pix, dtype=np.float32)
    mean_P[has_fires] = sum_P[has_fires] / n_fires[has_fires]

    # Consistency: 1 - cv (coefficient of variation)
    # cv = std / mean. Low cv = consistent. We want high consistency = good.
    var_P = np.zeros(n_pix, dtype=np.float32)
    var_P[has_fires] = (sum_P_sq[has_fires] / n_fires[has_fires]) - mean_P[has_fires]**2
    var_P = np.maximum(var_P, 0)
    std_P = np.sqrt(var_P)
    consistency = np.ones(n_pix, dtype=np.float32)
    nonzero_mean = has_fires & (mean_P > 0.01)
    consistency[nonzero_mean] = 1.0 - np.clip(
        std_P[nonzero_mean] / mean_P[nonzero_mean], 0, 1
    )

    # ── Stacked score ────────────────────────────────────────────────
    # stacked_score = sum_P * consistency^0.5
    # This rewards:
    #   - Pixels that fire many times (sum_P goes up with n_fires)
    #   - With stable confidence (consistency → 1)
    #   - High individual predictions (sum_P accumulates P values)
    stacked_score = sum_P * (consistency ** 0.5)

    # Apply min_fires filter: pixels with < min_fires get score=0
    # EXCEPT for Magnitude Fast-Track pixels (bypass min_fires)
    needs_min_fires = (n_fires < args.min_fires) & ~fast_track
    stacked_score[needs_min_fires] = 0

    # Fast-tracked pixels: ensure minimum score = their single-window P
    ft_only = fast_track & (n_fires < args.min_fires)
    stacked_score[ft_only] = np.maximum(stacked_score[ft_only], fast_track_P[ft_only])

    log.info(f"Magnitude Fast-Track: {n_fast_tracked:,} pixels bypassed min_fires")

    # ── Statistics ───────────────────────────────────────────────────
    active = stacked_score > 0
    n_active = int(active.sum())

    log.info(f"Pixels with ≥{args.min_fires} fires: {n_active:,} "
             f"({n_active/n_pix*100:.2f}%)")
    if n_active > 0:
        log.info(f"  score range: [{stacked_score[active].min():.3f}, "
                 f"{stacked_score[active].max():.3f}]")
        log.info(f"  score mean:  {stacked_score[active].mean():.3f}")
        log.info(f"  score median: {np.median(stacked_score[active]):.3f}")

    # ── Auto-select threshold if not provided ────────────────────────
    if args.stacked_threshold is None:
        # Use percentile-based approach: take top 20% of active pixels
        if n_active > 0:
            percentile_80 = np.percentile(stacked_score[active], 80)
            threshold = max(percentile_80, 0.5)
        else:
            threshold = 1.0
        log.info(f"Auto-selected threshold: {threshold:.3f} (80th percentile)")
    else:
        threshold = args.stacked_threshold
        log.info(f"Using provided threshold: {threshold:.3f}")

    # ── Evaluate against GT if available ──────────────────────────────
    if gt is not None:
        gt_trees = gt[TREES_IDX].ravel().astype(np.float32)
        gt_change = np.abs(gt_trees) >= 0.10

        log.info(f"\n{'='*65}")
        log.info(f"  EVALUATION vs GT")
        log.info(f"{'='*65}")

        # Test across multiple thresholds
        thresholds_to_test = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        best_f1 = 0
        best_thresh = threshold

        results = []
        for t in thresholds_to_test:
            confirmed = stacked_score >= t
            n_conf = int(confirmed.sum())
            if n_conf == 0:
                continue

            same_dir = np.sign(
                (sum_P > 0).astype(np.float32) * (-1)  # alerts are tree-loss
            ) == np.sign(gt_trees)

            tp = int((confirmed & gt_change).sum())
            fp = int((confirmed & ~gt_change).sum())
            fn = int((~confirmed & gt_change).sum())

            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1   = 2 * prec * rec / max(prec + rec, 1e-8)

            results.append({
                "threshold": t, "confirmed": n_conf,
                "tp": tp, "fp": fp, "prec": round(prec, 4),
                "rec": round(rec, 4), "f1": round(f1, 4),
            })

            log.info(
                f"  T={t:>5.1f}  confirmed={n_conf:>7,}  "
                f"TP={tp:>6,}  FP={fp:>6,}  "
                f"P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}"
            )

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        log.info(f"\n  Best F1: {best_f1:.4f} at threshold={best_thresh:.1f}")
        threshold = best_thresh

        # ── Plot PR across thresholds ────────────────────────────────
        if results:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # PR plot
            precs = [r["prec"] for r in results]
            recs  = [r["rec"] for r in results]
            f1s   = [r["f1"] for r in results]
            ts    = [r["threshold"] for r in results]

            axes[0].plot(recs, precs, "o-", color="teal", linewidth=2, markersize=8)
            for i, t in enumerate(ts):
                axes[0].annotate(f"T={t}", (recs[i], precs[i]),
                                textcoords="offset points", xytext=(8, 5),
                                fontsize=8)
            axes[0].set_xlabel("Recall", fontsize=12)
            axes[0].set_ylabel("Precision", fontsize=12)
            axes[0].set_title("Stacked Score: Precision vs Recall", fontsize=13,
                             fontweight="bold")
            axes[0].axhline(0.8, color="gray", ls="--", alpha=0.5, label="80% Prec")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            # F1 vs threshold
            axes[1].bar(range(len(ts)), f1s, color="steelblue",
                       tick_label=[f"{t:.1f}" for t in ts])
            axes[1].set_xlabel("Stacked Score Threshold", fontsize=12)
            axes[1].set_ylabel("F1 Score", fontsize=12)
            axes[1].set_title("F1 vs Threshold", fontsize=13, fontweight="bold")
            axes[1].grid(axis="y", alpha=0.3)

            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, "stacked_pr_f1.png"),
                       dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ── Apply final threshold and generate outputs ───────────────────
    log.info(f"\n{'='*65}")
    log.info(f"  GENERATING OUTPUTS (threshold={threshold:.1f})")
    log.info(f"{'='*65}")

    score_map = stacked_score.reshape(H, W)
    n_fires_map = n_fires.reshape(H, W)

    # ── Object-Based Hysteresis ──────────────────────────────────────
    # Step 1: Create a PERMISSIVE seed mask at T/2 (captures full
    #         perimeter of clearings, including edge pixels)
    permissive_thresh = threshold * 0.5
    permissive_mask = (stacked_score >= permissive_thresh).reshape(H, W)

    # Step 2: Morphological closing to glue fragmented micro-polygons
    #         (fixes 5m wobble between Sentinel-2 overpasses)
    struct = np.ones((3, 3), dtype=bool)  # 3×3 structuring element
    closed_mask = binary_closing(permissive_mask, structure=struct, iterations=1)

    # Step 3: Cluster the closed mask
    try:
        from skimage.morphology import remove_small_objects
        closed_mask = remove_small_objects(closed_mask, min_size=args.min_pixels)
    except ImportError:
        labeled_tmp, n_feat_tmp = ndlabel(closed_mask.astype(np.int32))
        for i in range(1, n_feat_tmp + 1):
            if (labeled_tmp == i).sum() < args.min_pixels:
                closed_mask[labeled_tmp == i] = False

    # Step 4: Keep a polygon ONLY if its max_score >= threshold
    #         (hysteresis: expand at permissive T, filter at strict T)
    labeled_hysteresis, n_hysteresis = ndlabel(closed_mask.astype(np.int32))
    clean_mask = np.zeros_like(closed_mask)
    for i in range(1, n_hysteresis + 1):
        component = labeled_hysteresis == i
        if score_map[component].max() >= threshold:
            clean_mask[component] = True

    n_confirmed = int(clean_mask.sum())
    n_polys_before = n_hysteresis
    n_polys_after = ndlabel(clean_mask.astype(np.int32))[1]
    log.info(f"Hysteresis: {n_polys_before} candidates → {n_polys_after} passed")
    log.info(f"Confirmed pixels: {n_confirmed:,}")

    # ── Save stacked score raster ────────────────────────────────────
    p = profile.copy()
    p.update(count=3, dtype="float32", compress="lzw")
    score_path = os.path.join(args.out_dir, "stacked_scores.tif")
    with rasterio.open(score_path, "w", **p) as dst:
        dst.write(score_map, 1)         # Band 1: stacked_score
        dst.write(n_fires_map.astype(np.float32), 2)  # Band 2: n_fires
        dst.write(max_P.reshape(H, W), 3)  # Band 3: max single-window P
    log.info(f"Stacked score raster: {score_path}")

    # ── Vectorize to GeoJSON ─────────────────────────────────────────
    if not clean_mask.any():
        log.info("No confirmed polygons!")
        geojson = {"type": "FeatureCollection", "features": []}
    else:
        labeled, n_components = ndlabel(clean_mask.astype(np.int32))

        # Last-window deltas for change type (not washed-out weighted avg)
        lw_deltas_2d = last_window_deltas.reshape(N_BANDS, H, W)

        shapes_gen = rasterio.features.shapes(
            labeled.astype(np.int32), mask=clean_mask, transform=transform,
        )

        features_list = []
        seen = set()
        for geom, value in shapes_gen:
            lid = int(value)
            if lid in seen or lid == 0:
                continue
            seen.add(lid)

            poly_mask = labeled == lid
            n_pix_poly = int(poly_mask.sum())

            mean_score = float(np.mean(score_map[poly_mask]))
            mean_fires = float(np.mean(n_fires_map[poly_mask]))
            max_single_p = float(np.max(max_P.reshape(H, W)[poly_mask]))
            is_ft = bool(fast_track.reshape(H, W)[poly_mask].any())

            # Change type from LAST WINDOW deltas (most recent state)
            mean_d = {}
            for b, name in enumerate(DW_BANDS):
                mean_d[name] = float(np.mean(lw_deltas_2d[b][poly_mask]))

            change_type = classify_change_type(mean_d)

            # Centroid + area
            coords = np.array(geom["coordinates"][0])
            cx = float(np.mean(coords[:, 0]))
            cy = float(np.mean(coords[:, 1]))

            pixel_w_m  = abs(transform.a) * 101300
            pixel_h_m  = abs(transform.e) * 110600
            area_ha = n_pix_poly * pixel_w_m * pixel_h_m / 10000

            features_list.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "change_type": change_type,
                    "stacked_score": round(mean_score, 3),
                    "mean_fires": round(mean_fires, 1),
                    "max_single_P": round(max_single_p, 3),
                    "area_ha": round(area_ha, 4),
                    "n_pixels": n_pix_poly,
                    "fast_tracked": is_ft,
                    "centroid": [round(cy, 6), round(cx, 6)],
                    "detection_period": f"{parse_dates(Path(files[0]).stem)[0]} to "
                                       f"{parse_dates(Path(files[-1]).stem)[1]}",
                    "mean_delta_trees": round(mean_d.get("trees", 0), 4),
                    "mean_delta_crops": round(mean_d.get("crops", 0), 4),
                    "mean_delta_bare": round(mean_d.get("bare", 0), 4),
                }
            })

        geojson = {"type": "FeatureCollection", "features": features_list}
        log.info(f"Generated {len(features_list)} polygons")

    geojson_path = os.path.join(args.out_dir, "stacked_confirmed_alerts.geojson")
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)
    log.info(f"GeoJSON: {geojson_path}")

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    type_counts = {}
    for feat in geojson["features"]:
        ct = feat["properties"]["change_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1

    total_area = sum(f["properties"]["area_ha"] for f in geojson["features"])

    # ── Heatmap of stacked scores ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    im0 = axes[0].imshow(n_fires_map, cmap="YlOrRd", vmin=0,
                         vmax=min(N, n_fires_map.max()))
    axes[0].set_title("N Fires (windows triggered)", fontsize=12, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(score_map, cmap="viridis",
                         vmin=0, vmax=np.percentile(score_map[score_map > 0], 99)
                         if (score_map > 0).any() else 1)
    axes[1].set_title("Stacked Score", fontsize=12, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(clean_mask.astype(np.float32), cmap="Reds")
    axes[2].set_title(f"Confirmed (T={threshold:.1f}), {n_confirmed:,} px",
                     fontsize=12, fontweight="bold")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Temporal Stacking: {N} Windows → {len(features_list)} Polygons, "
                 f"{total_area:.1f} ha", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "stacked_heatmap.png"),
               dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "elapsed_seconds": round(elapsed, 1),
        "n_windows": N, "threshold": threshold,
        "min_fires": args.min_fires,
        "total_active_pixels": n_active,
        "confirmed_pixels": n_confirmed,
        "n_polygons": len(geojson["features"]),
        "total_area_ha": round(total_area, 2),
        "change_type_counts": type_counts,
    }
    with open(os.path.join(args.out_dir, "stacked_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\n{'='*65}")
    log.info(f"  STACKING COMPLETE in {elapsed:.1f}s")
    log.info(f"  Windows: {N}")
    log.info(f"  Threshold: {threshold}")
    log.info(f"  Active pixels (≥{args.min_fires} fires): {n_active:,}")
    log.info(f"  Confirmed: {n_confirmed:,}")
    log.info(f"  Polygons: {len(geojson['features'])}")
    for ct, cnt in sorted(type_counts.items()):
        log.info(f"    {ct}: {cnt}")
    log.info(f"  Total area: {total_area:.2f} ha")
    log.info(f"{'='*65}")


if __name__ == "__main__":
    main()
