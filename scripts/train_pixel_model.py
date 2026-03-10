"""
Pixel-wise Change Classifier — Alert Features → GT Labels
===========================================================

Extracts ~70 temporal features per pixel from 15 alert-delta rasters,
labels each pixel from the GT annual delta raster, and trains a
LightGBM multi-class classifier.

Classes:
  0 = No change
  1 = Encroachment  (trees↓ + crops↑)
  2 = Degradation   (trees↓ + bare/shrub/grass↑)
  3 = Other tree loss
  4 = Greening      (trees↑)

Usage:
    python scripts/train_pixel_model.py \\
        --data-dir data/ground_truth/HAMEERPUR \\
        --out-dir outputs/pixel_model
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
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import rasterio
import lightgbm as lgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DW_BANDS = ["water", "trees", "grass", "flooded_veg",
            "crops", "shrub_scrub", "built", "bare"]
TREES_IDX = 1
CROPS_IDX = 4
BARE_IDX  = 7
BUILT_IDX = 6
SHRUB_IDX = 5
GRASS_IDX = 2

CLASS_NAMES = {
    0: "No change",
    1: "Encroachment",
    2: "Degradation",
    3: "Other tree loss",
    4: "Greening",
}
CLASS_COLORS = {
    0: "#cccccc",
    1: "#e6550d",
    2: "#8c564b",
    3: "#d62728",
    4: "#2ca02c",
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_alert_stack(data_dir: str) -> tuple[np.ndarray, list[str]]:
    """Load all alert TIFs sorted chronologically → (N, 10, H, W)."""
    pattern = os.path.join(data_dir, "alerts", "alert_delta_*.tif")
    files = sorted(glob(pattern))
    if not files:
        pattern = os.path.join(data_dir, "alert_delta_*.tif")
        files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No alert rasters in {data_dir}")

    arrays = []
    names  = []
    meta   = None
    for f in files:
        with rasterio.open(f) as ds:
            arrays.append(ds.read())  # (bands, H, W)
            if meta is None:
                meta = ds.profile.copy()
            names.append(Path(f).stem)
    stack = np.stack(arrays, axis=0)  # (N, bands, H, W)
    log.info(f"Alert stack: {stack.shape} ({len(files)} windows)")
    return stack, names, meta


def load_gt(data_dir: str) -> np.ndarray:
    """Load GT delta → (bands, H, W)."""
    files = glob(os.path.join(data_dir, "gt_delta_*.tif"))
    if not files:
        raise FileNotFoundError(f"No GT raster in {data_dir}")
    with rasterio.open(files[0]) as ds:
        gt = ds.read()
    log.info(f"GT: {gt.shape}")
    return gt


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(stack: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Extract ~70 features per pixel from the alert stack.

    stack: (N, 10, H, W) — N alert windows, 10 bands each
    Returns: features (H*W, n_features), feature_names list
    """
    N, B, H, W = stack.shape
    n_pixels = H * W

    deltas = stack[:, :8, :, :]  # (N, 8, H, W)
    deltas = np.nan_to_num(deltas, nan=0.0)

    has_cloud = B >= 10
    if has_cloud:
        cloud_before = np.nan_to_num(stack[:, 8, :, :], nan=1.0)  # (N, H, W)
        cloud_after  = np.nan_to_num(stack[:, 9, :, :], nan=1.0)
        cloud_worst  = np.maximum(cloud_before, cloud_after)       # (N, H, W)

    features = []
    feat_names = []

    # ── Group 1: Per-band temporal stats (8 bands × 7 stats = 56) ─────
    for b_idx, b_name in enumerate(DW_BANDS):
        band_vals = deltas[:, b_idx, :, :]  # (N, H, W)
        band_flat = band_vals.reshape(N, n_pixels)  # (N, n_pix)

        # Mask: non-zero values
        nz_mask = band_flat != 0  # (N, n_pix)
        nz_count = nz_mask.sum(axis=0).astype(np.float32)  # (n_pix,)

        # Safe computation: replace unobserved with NaN for stats
        safe = np.where(nz_mask, band_flat, np.nan)

        with np.errstate(all="ignore"):
            feat_mean  = np.nanmean(safe, axis=0)
            feat_std   = np.nanstd(safe, axis=0)
            feat_min   = np.nanmin(safe, axis=0)
            feat_max   = np.nanmax(safe, axis=0)
            feat_sum   = np.nansum(safe, axis=0)

        feat_range = feat_max - feat_min

        # Replace NaN with 0 for pixels with no data
        for arr in [feat_mean, feat_std, feat_min, feat_max, feat_sum, feat_range]:
            np.nan_to_num(arr, nan=0.0, copy=False)

        features.extend([feat_mean, feat_std, feat_min, feat_max, feat_sum, nz_count, feat_range])
        feat_names.extend([
            f"{b_name}_mean", f"{b_name}_std", f"{b_name}_min", f"{b_name}_max",
            f"{b_name}_sum", f"{b_name}_count_nz", f"{b_name}_range",
        ])

    # ── Group 2: Trees-specific (5 features) ─────────────────────────
    trees_vals = deltas[:, TREES_IDX, :, :].reshape(N, n_pixels)  # (N, n_pix)

    # Count negative windows
    neg_mask = trees_vals < 0
    trees_count_neg = neg_mask.sum(axis=0).astype(np.float32)

    # Mean of negative values
    neg_safe = np.where(neg_mask, trees_vals, np.nan)
    with np.errstate(all="ignore"):
        trees_mean_neg = np.nanmean(neg_safe, axis=0)
    np.nan_to_num(trees_mean_neg, nan=0.0, copy=False)

    # Min value (worst loss)
    trees_min = np.min(trees_vals, axis=0)

    # Trend slope (linear fit index vs value)
    x = np.arange(N, dtype=np.float32)
    x_mean = x.mean()
    x_var  = np.sum((x - x_mean) ** 2)
    if x_var > 0:
        trees_trend = np.sum(
            (x[:, None] - x_mean) * trees_vals, axis=0
        ) / x_var
    else:
        trees_trend = np.zeros(n_pixels, dtype=np.float32)

    # Max consecutive negative streak
    trees_consec = np.zeros(n_pixels, dtype=np.float32)
    current_streak = np.zeros(n_pixels, dtype=np.float32)
    for t in range(N):
        is_neg = trees_vals[t] < 0
        current_streak = np.where(is_neg, current_streak + 1, 0)
        trees_consec = np.maximum(trees_consec, current_streak)

    features.extend([trees_count_neg, trees_mean_neg, trees_min, trees_trend, trees_consec])
    feat_names.extend([
        "trees_count_neg", "trees_mean_neg", "trees_min_val",
        "trees_trend_slope", "trees_max_consec_neg",
    ])

    # ── Group 3: Cloud features (5) ──────────────────────────────────
    if has_cloud:
        cw_flat = cloud_worst.reshape(N, n_pixels)

        cloud_mean = np.nanmean(cw_flat, axis=0)
        cloud_max  = np.nanmax(cw_flat, axis=0)
        cloud_min  = np.nanmin(cw_flat, axis=0)
        n_clear    = np.sum(cw_flat < 0.3, axis=0).astype(np.float32)

        # Cloud-weighted trees mean: weight = (1 - cloud)
        clarity = 1.0 - cw_flat  # (N, n_pix)
        clarity_sum = clarity.sum(axis=0)
        trees_cw = np.where(
            clarity_sum > 0,
            np.sum(trees_vals * clarity, axis=0) / clarity_sum,
            0.0,
        )

        for arr in [cloud_mean, cloud_max, cloud_min]:
            np.nan_to_num(arr, nan=0.0, copy=False)

        features.extend([cloud_mean, cloud_max, cloud_min, n_clear, trees_cw])
        feat_names.extend([
            "cloud_worst_mean", "cloud_max", "cloud_min",
            "n_clear_windows", "trees_cloud_weighted",
        ])

    # ── Group 4: Cross-band features (4) ──────────────────────────────
    crops_vals  = deltas[:, CROPS_IDX, :, :].reshape(N, n_pixels)
    bare_vals   = deltas[:, BARE_IDX, :, :].reshape(N, n_pixels)

    # Encroachment signal: fraction of windows where trees↓ AND crops↑
    encr_signal = ((trees_vals < 0) & (crops_vals > 0)).sum(axis=0).astype(np.float32)
    nz_windows  = (trees_vals != 0).sum(axis=0).astype(np.float32)
    encr_frac   = np.where(nz_windows > 0, encr_signal / nz_windows, 0.0)

    # Clearing signal: fraction where trees↓ AND bare↑
    clear_signal = ((trees_vals < 0) & (bare_vals > 0)).sum(axis=0).astype(np.float32)
    clear_frac   = np.where(nz_windows > 0, clear_signal / nz_windows, 0.0)

    # Alert frequency (threshold-based)
    any_triggered = np.zeros((N, n_pixels), dtype=bool)
    for t in range(N):
        trees_fire = trees_vals[t] < -0.25
        other_fire = np.max(np.abs(deltas[t, [i for i in range(8) if i != TREES_IDX], :, :].reshape(7, n_pixels)), axis=0) > 0.25
        any_triggered[t] = trees_fire | other_fire
    alert_freq = any_triggered.sum(axis=0).astype(np.float32)

    # Band diversity: how many different bands exceeded threshold across all windows
    band_triggered = np.zeros((8, n_pixels), dtype=bool)
    for b in range(8):
        bv = deltas[:, b, :, :].reshape(N, n_pixels)
        if b == TREES_IDX:
            band_triggered[b] = np.any(bv < -0.25, axis=0)
        else:
            band_triggered[b] = np.any(np.abs(bv) > 0.25, axis=0)
    band_diversity = band_triggered.sum(axis=0).astype(np.float32)

    features.extend([encr_frac, clear_frac, alert_freq, band_diversity])
    feat_names.extend([
        "trees_crops_anti_frac", "trees_bare_anti_frac",
        "alert_frequency", "max_band_diversity",
    ])

    # ── Stack all features ────────────────────────────────────────────
    feature_matrix = np.stack(features, axis=-1)  # (n_pixels, n_features)
    log.info(f"Feature matrix: {feature_matrix.shape}  ({len(feat_names)} features)")
    return feature_matrix, feat_names


# ── Label extraction ──────────────────────────────────────────────────────────

def extract_labels(gt: np.ndarray) -> np.ndarray:
    """
    Classify each pixel from GT bands into one of 5 classes.
    gt: (bands, H, W)  — first 8 bands are delta probs
    Returns: (H*W,) int array of class labels
    """
    n_pixels = gt.shape[1] * gt.shape[2]

    trees = gt[TREES_IDX].ravel()
    crops = gt[CROPS_IDX].ravel()
    bare  = gt[BARE_IDX].ravel()
    built = gt[BUILT_IDX].ravel()
    shrub = gt[SHRUB_IDX].ravel()
    grass = gt[GRASS_IDX].ravel()

    labels = np.zeros(n_pixels, dtype=np.int32)  # default: 0 = no change

    # Class 4: Greening (trees gained)
    greening = trees > 0.15
    labels[greening] = 4

    # Class 1: Encroachment (trees lost + crops or built gained)
    encroachment = (trees < -0.15) & ((crops > 0.10) | (built > 0.10))
    labels[encroachment] = 1

    # Class 2: Degradation (trees lost + bare/shrub/grass gained)
    degradation = (trees < -0.15) & ((bare > 0.10) | (shrub > 0.10) | (grass > 0.10))
    # Don't overwrite encroachment (priority)
    degradation = degradation & (labels != 1)
    labels[degradation] = 2

    # Class 3: Other tree loss (trees lost but no clear replacement)
    other_loss = (trees < -0.15) & (labels == 0)
    labels[other_loss] = 3

    return labels


# ── Spatial block split ───────────────────────────────────────────────────────

def spatial_block_split(H: int, W: int, grid: int = 4,
                        val_blocks: int = 2, test_blocks: int = 2,
                        seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide raster into grid × grid blocks, assign to train/val/test.
    Returns boolean masks (H*W,) for train, val, test.
    """
    rng = np.random.default_rng(seed)
    n_blocks = grid * grid
    block_ids = np.arange(n_blocks)
    rng.shuffle(block_ids)

    test_set = set(block_ids[:test_blocks])
    val_set  = set(block_ids[test_blocks:test_blocks + val_blocks])

    # Assign each pixel a block ID
    rows = np.arange(H)
    cols = np.arange(W)
    row_block = np.clip(rows * grid // H, 0, grid - 1)
    col_block = np.clip(cols * grid // W, 0, grid - 1)

    block_map = row_block[:, None] * grid + col_block[None, :]  # (H, W)
    block_flat = block_map.ravel()  # (H*W,)

    train_mask = np.ones(H * W, dtype=bool)
    val_mask   = np.zeros(H * W, dtype=bool)
    test_mask  = np.zeros(H * W, dtype=bool)

    for b in test_set:
        test_mask[block_flat == b] = True
        train_mask[block_flat == b] = False
    for b in val_set:
        val_mask[block_flat == b] = True
        train_mask[block_flat == b] = False

    return train_mask, val_mask, test_mask


# ── Training ──────────────────────────────────────────────────────────────────

def balanced_sample(X: np.ndarray, y: np.ndarray,
                    max_per_class: int = 50000, seed: int = 42) -> tuple:
    """Stratified balanced sampling: equal samples per class."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    idx_list = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        n = min(len(c_idx), max_per_class)
        idx_list.append(rng.choice(c_idx, size=n, replace=False))
    idx = np.concatenate(idx_list)
    rng.shuffle(idx)
    return X[idx], y[idx]


def train_model(X_train, y_train, X_val, y_val, feat_names):
    """Train LightGBM multi-class classifier."""
    n_classes = len(np.unique(y_train))

    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "is_unbalance": True,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feat_names, free_raw_data=False)
    dval   = lgb.Dataset(X_val, label=y_val, feature_name=feat_names, free_raw_data=False)

    log.info(f"Training LightGBM: {X_train.shape[0]:,} samples, {X_train.shape[1]} features, {n_classes} classes")

    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=30),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, feat_names, out_dir):
    """Full evaluation: confusion matrix, feature importance, metrics."""

    y_pred = model.predict(X_test).argmax(axis=1)

    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
        zero_division=0,
    )
    log.info(f"\n{report}")

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    log.info(f"Balanced accuracy: {bal_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=10, color=color)
    labels = [CLASS_NAMES[i] for i in range(cm.shape[0])]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True (GT)", fontsize=12)
    ax.set_title("Confusion Matrix (Test Set — Spatial Block)", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importance)[-25:]  # top 25

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [feat_names[i] for i in sorted_idx],
        importance[sorted_idx],
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Feature Importance (gain)", fontsize=12)
    ax.set_title("Top 25 Features — LightGBM", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path}")

    return {
        "balanced_accuracy": round(float(bal_acc), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


# ── Raster prediction ────────────────────────────────────────────────────────

def predict_raster(model, features: np.ndarray, H: int, W: int,
                   meta: dict, out_dir: str):
    """Predict on all pixels and save classified + probability rasters."""

    probs  = model.predict(features)  # (n_pixels, n_classes)
    labels = probs.argmax(axis=1)     # (n_pixels,)

    # Classified raster
    classified = labels.reshape(H, W).astype(np.uint8)
    p = meta.copy()
    p.update(count=1, dtype="uint8", compress="lzw")
    path_cls = os.path.join(out_dir, "classified_raster.tif")
    with rasterio.open(path_cls, "w", **p) as dst:
        dst.write(classified, 1)
        dst.set_band_description(1, "predicted_class")
    log.info(f"Saved: {path_cls}")

    # Probability raster (n_classes bands)
    n_classes = probs.shape[1]
    prob_map = probs.reshape(H, W, n_classes).transpose(2, 0, 1)  # (C, H, W)
    p.update(count=n_classes, dtype="float32")
    path_prob = os.path.join(out_dir, "probability_raster.tif")
    with rasterio.open(path_prob, "w", **p) as dst:
        for c in range(n_classes):
            dst.write(prob_map[c].astype(np.float32), c + 1)
            dst.set_band_description(c + 1, CLASS_NAMES.get(c, f"class_{c}"))
    log.info(f"Saved: {path_prob}")

    # Visualization
    cmap = mcolors.ListedColormap([CLASS_COLORS[i] for i in range(n_classes)])
    bounds = np.arange(n_classes + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left: classified map
    im = axes[0].imshow(classified, cmap=cmap, norm=norm, interpolation="nearest")
    axes[0].set_title("Predicted Change Class", fontsize=13, fontweight="bold")
    axes[0].axis("off")
    patches = [Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(n_classes)]
    axes[0].legend(handles=patches, loc="lower right", fontsize=9)

    # Right: encroachment probability
    encr_prob = prob_map[1]  # class 1 = encroachment
    im2 = axes[1].imshow(encr_prob, cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Encroachment Probability", fontsize=13, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = os.path.join(out_dir, "prediction_maps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path}")

    # Class distribution
    unique, counts = np.unique(classified, return_counts=True)
    log.info("Predicted class distribution:")
    for u, c in zip(unique, counts):
        log.info(f"  {CLASS_NAMES.get(int(u), '?'):20s}: {c:>10,} pixels")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="outputs/pixel_model")
    parser.add_argument("--max-per-class", type=int, default=50000)
    parser.add_argument("--grid", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # ── Phase 1: Load ─────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("  Phase 1: LOAD DATA")
    log.info("=" * 65)

    stack, alert_names, meta = load_alert_stack(args.data_dir)
    gt = load_gt(args.data_dir)
    N, B, H, W = stack.shape

    # ── Phase 2: Extract features + labels ────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  Phase 2: FEATURE EXTRACTION")
    log.info("=" * 65)

    features, feat_names = extract_features(stack)
    labels = extract_labels(gt)

    # Valid pixel mask: at least 1 non-zero window
    nz_per_pixel = (stack[:, :8, :, :].reshape(N, 8, H * W) != 0).any(axis=1).sum(axis=0)
    valid_mask = nz_per_pixel >= 1
    log.info(f"Valid pixels (≥1 non-zero window): {valid_mask.sum():,} / {H*W:,}")

    # Class distribution in GT
    log.info("\nGT class distribution (all valid pixels):")
    for c in sorted(CLASS_NAMES):
        n = int(np.sum((labels == c) & valid_mask))
        pct = n / max(valid_mask.sum(), 1) * 100
        log.info(f"  {CLASS_NAMES[c]:20s}: {n:>10,}  ({pct:.2f}%)")

    # ── Phase 3: Split + Train ────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  Phase 3: SPATIAL SPLIT + TRAINING")
    log.info("=" * 65)

    train_mask, val_mask, test_mask = spatial_block_split(H, W, grid=args.grid)

    # Intersect with valid
    train_idx = np.where(train_mask & valid_mask)[0]
    val_idx   = np.where(val_mask & valid_mask)[0]
    test_idx  = np.where(test_mask & valid_mask)[0]

    log.info(f"Train pixels: {len(train_idx):,}")
    log.info(f"Val pixels:   {len(val_idx):,}")
    log.info(f"Test pixels:  {len(test_idx):,}")

    X_train_full = features[train_idx]
    y_train_full = labels[train_idx]
    X_val   = features[val_idx]
    y_val   = labels[val_idx]
    X_test  = features[test_idx]
    y_test  = labels[test_idx]

    # Balanced sampling for training
    X_train_bal, y_train_bal = balanced_sample(
        X_train_full, y_train_full, max_per_class=args.max_per_class
    )
    log.info(f"\nBalanced training set: {X_train_bal.shape[0]:,} samples")
    for c in sorted(CLASS_NAMES):
        log.info(f"  {CLASS_NAMES[c]:20s}: {int(np.sum(y_train_bal == c)):,}")

    model = train_model(X_train_bal, y_train_bal, X_val, y_val, feat_names)

    # Save model
    model_path = os.path.join(args.out_dir, "model.lgbm")
    model.save_model(model_path)
    log.info(f"Model saved: {model_path}")

    # ── Phase 4: Evaluate ─────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  Phase 4: EVALUATION")
    log.info("=" * 65)

    metrics = evaluate(model, X_test, y_test, feat_names, args.out_dir)

    # ── Phase 5: Full raster prediction ───────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  Phase 5: FULL RASTER PREDICTION")
    log.info("=" * 65)

    predict_raster(model, features, H, W, meta, args.out_dir)

    # ── Save training log ─────────────────────────────────────────────
    elapsed = time.time() - t0
    training_log = {
        "elapsed_seconds": round(elapsed, 1),
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "n_alert_windows": N,
        "raster_shape": [H, W],
        "n_valid_pixels": int(valid_mask.sum()),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "n_train_balanced": len(y_train_bal),
        "balanced_accuracy": metrics["balanced_accuracy"],
        "best_iteration": model.best_iteration,
        "class_names": CLASS_NAMES,
    }
    log_path = os.path.join(args.out_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    log.info(f"\nTraining log: {log_path}")

    log.info(f"\n{'='*65}")
    log.info(f"  DONE in {elapsed:.1f}s")
    log.info(f"  Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    log.info(f"  Model: {model_path}")
    log.info(f"  Outputs: {args.out_dir}/")
    log.info(f"{'='*65}")


if __name__ == "__main__":
    main()
