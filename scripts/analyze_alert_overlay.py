"""
Alert & GT Raster Overlay Analysis
====================================

Reads all 15 alert-delta TIFs + the GT-delta TIF for HAMEERPUR,
overlays them, and produces:

  1. TEMPORAL ACCUMULATION — cumulative Δtrees at each pixel across the year
  2. ALERT FREQUENCY MAP  — how many of the 15 alert windows flagged each pixel
  3. GT vs ALERT MATCH    — correlation between annual GT delta and sum-of-alerts
  4. CLOUD QUALITY MAP    — average cloud probability per pixel across all alerts
  5. HOTSPOT DETECTION    — pixels with persistent negative Δtrees + low cloud
  6. CLASS CO-OCCURRENCE  — which class changes happen together (Δtrees↓ + Δcrops↑?)
  7. SEASONAL PROFILE     — per-class delta grouped by season

Outputs saved to  outputs/hameerpur_analysis/

Usage:
    python scripts/analyze_alert_overlay.py --data-dir data/ground_truth/HAMEERPUR
"""

# ── PROJ/GDAL env fix ──
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()
del _os, _sys
# ────────────────────────

import argparse
import logging
import os
import sys
from pathlib import Path
from glob import glob
from datetime import datetime

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DW_BANDS = [
    "water", "trees", "grass", "flooded_veg",
    "crops", "shrub_scrub", "built", "bare",
]
TREES_IDX        = 1
CLOUD_BEFORE_IDX = 8   # band 9  (0-indexed)
CLOUD_AFTER_IDX  = 9   # band 10 (0-indexed)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def parse_alert_dates(filename: str) -> tuple[str, str]:
    """
    Extract date pair from filename like:
    alert_delta_HAMEERPUR_2025-02-02_to_2025-02-07.tif
    """
    stem = Path(filename).stem
    parts = stem.split("_to_")
    if len(parts) != 2:
        raise ValueError(f"Cannot parse dates from: {filename}")
    date_after = parts[1]
    # before date is the last date-like token before "_to_"
    before_part = parts[0]
    tokens = before_part.split("_")
    # Date is the last 3 tokens joined by "-" (YYYY-MM-DD)
    date_before = tokens[-1]
    # Check if it's actually last two tokens (for YYYY-MM-DD format)
    for i in range(len(tokens) - 1, -1, -1):
        if len(tokens[i]) == 10 and tokens[i][4] == "-":
            date_before = tokens[i]
            break
    return date_before, date_after


def load_alert_stack(alert_dir: str) -> tuple[np.ndarray, list[tuple[str, str]], dict]:
    """
    Loads all alert_delta_*.tif files sorted chronologically.

    Returns:
      stack:   (N, bands, H, W) array — N alert rasters, 10 bands each
      dates:   list of (date_before, date_after) tuples
      meta:    rasterio profile from the first file
    """
    pattern = os.path.join(alert_dir, "alert_delta_*.tif")
    files = sorted(glob(pattern))

    if not files:
        # Try alerts subfolder
        pattern = os.path.join(alert_dir, "alerts", "alert_delta_*.tif")
        files = sorted(glob(pattern))

    if not files:
        log.error(f"No alert rasters found in {alert_dir}")
        log.error(f"  Looked for: alert_delta_*.tif")
        sys.exit(1)

    log.info(f"Found {len(files)} alert rasters")

    arrays = []
    dates  = []
    meta   = None

    for f in files:
        with rasterio.open(f) as ds:
            arr = ds.read()  # (bands, H, W)
            if meta is None:
                meta = ds.profile.copy()
            arrays.append(arr)
            d_before, d_after = parse_alert_dates(os.path.basename(f))
            dates.append((d_before, d_after))
            log.info(f"  Loaded: {os.path.basename(f)}  "
                     f"({d_before} → {d_after})  "
                     f"shape={arr.shape}")

    stack = np.stack(arrays, axis=0)  # (N, bands, H, W)
    return stack, dates, meta


def load_gt_delta(data_dir: str) -> tuple[np.ndarray | None, dict | None]:
    """
    Loads gt_delta_*.tif — the 8-band annual ground truth.
    Returns (array (8, H, W), profile) or (None, None) if not found.
    """
    pattern = os.path.join(data_dir, "gt_delta_*.tif")
    files = glob(pattern)
    if not files:
        log.warning("No GT delta raster found — skipping GT analysis")
        return None, None

    f = files[0]
    with rasterio.open(f) as ds:
        arr  = ds.read()
        meta = ds.profile.copy()
    log.info(f"  GT loaded: {os.path.basename(f)}  shape={arr.shape}")
    return arr, meta


# ── Analysis functions ────────────────────────────────────────────────────────

def temporal_accumulation(stack: np.ndarray, dates: list) -> dict:
    """
    Compute cumulative delta for each class across all alert windows.
    stack: (N, 10, H, W) — 10-band alert deltas

    Returns dict with (8, H, W) arrays for:
      cumulative_delta   — sum of deltas across all N windows
      mean_delta         — mean delta per window
      max_abs_delta      — max absolute delta seen at each pixel
    """
    deltas = stack[:, :8, :, :]  # (N, 8, H, W), ignore cloud bands

    # Replace NaN with 0 for accumulation
    deltas = np.nan_to_num(deltas, nan=0.0)

    cumulative = np.sum(deltas, axis=0)        # (8, H, W)
    mean_delta = np.mean(deltas, axis=0)        # (8, H, W)
    max_abs    = np.max(np.abs(deltas), axis=0) # (8, H, W)

    return {
        "cumulative_delta": cumulative,
        "mean_delta": mean_delta,
        "max_abs_delta": max_abs,
    }


def alert_frequency(stack: np.ndarray, tree_thresh: float = 0.25,
                    other_thresh: float = 0.25) -> np.ndarray:
    """
    Per-pixel count of how many alert windows showed significant change.
    Returns (H, W) integer array.
    """
    deltas = stack[:, :8, :, :]  # (N, 8, H, W)
    deltas = np.nan_to_num(deltas, nan=0.0)

    # Trees: delta < -threshold
    trees_fire = deltas[:, TREES_IDX, :, :] < -tree_thresh  # (N, H, W)

    # Others: |delta| > threshold (any of the 7 non-trees bands)
    other_idxs = [i for i in range(8) if i != TREES_IDX]
    others_fire = np.any(
        np.abs(deltas[:, other_idxs, :, :]) > other_thresh,
        axis=1
    )  # (N, H, W)

    # A pixel "fires" if trees OR others triggered
    fired = trees_fire | others_fire  # (N, H, W)
    return np.sum(fired, axis=0)  # (H, W)


def cloud_quality(stack: np.ndarray) -> dict:
    """
    Aggregate cloud probability across all alert windows.
    Returns dict with (H, W) arrays.
    """
    n_bands = stack.shape[1]
    if n_bands < 10:
        log.warning("No cloud bands in stack (< 10 bands)")
        return {}

    cloud_before = stack[:, CLOUD_BEFORE_IDX, :, :]  # (N, H, W)
    cloud_after  = stack[:, CLOUD_AFTER_IDX, :, :]    # (N, H, W)

    cloud_before = np.nan_to_num(cloud_before, nan=1.0)
    cloud_after  = np.nan_to_num(cloud_after, nan=1.0)

    # Worst cloud per window = max(before, after)
    worst_per_window = np.maximum(cloud_before, cloud_after)  # (N, H, W)

    return {
        "mean_cloud":  np.mean(worst_per_window, axis=0),    # (H, W)
        "max_cloud":   np.max(worst_per_window, axis=0),     # (H, W)
        "min_cloud":   np.min(worst_per_window, axis=0),     # (H, W)
        "clear_count": np.sum(worst_per_window < 0.3, axis=0),  # how many windows < 30% cloud
    }


def gt_vs_alerts(gt: np.ndarray, cumulative: np.ndarray) -> dict:
    """
    Compare GT annual delta with cumulative alert deltas.
    Both are (8, H, W) arrays.

    Returns per-band correlation and residual stats.
    """
    results = {}
    n_bands = min(gt.shape[0], cumulative.shape[0], 8)

    for i in range(n_bands):
        gt_flat   = gt[i].ravel()
        cum_flat  = cumulative[i].ravel()

        # Remove NaN/inf
        valid = np.isfinite(gt_flat) & np.isfinite(cum_flat)
        if valid.sum() < 100:
            results[DW_BANDS[i]] = {"correlation": None, "n_valid": int(valid.sum())}
            continue

        gt_v  = gt_flat[valid]
        cum_v = cum_flat[valid]

        corr = np.corrcoef(gt_v, cum_v)[0, 1]
        residual = gt_v - cum_v
        results[DW_BANDS[i]] = {
            "correlation": round(float(corr), 4),
            "residual_mean": round(float(np.mean(residual)), 4),
            "residual_std":  round(float(np.std(residual)), 4),
            "n_valid": int(valid.sum()),
        }

    return results


def hotspot_detection(cumulative: np.ndarray, cloud_mean: np.ndarray | None,
                      tree_loss_thresh: float = -0.5,
                      cloud_thresh: float = 0.3) -> np.ndarray:
    """
    Identify high-confidence deforestation hotspots:
      - Cumulative Δtrees < threshold (persistent tree loss)
      - Mean cloud prob < cloud_thresh (observations are reliable)

    Returns (H, W) boolean mask.
    """
    trees_loss = cumulative[TREES_IDX] < tree_loss_thresh  # (H, W)

    if cloud_mean is not None:
        clear = cloud_mean < cloud_thresh
        hotspot = trees_loss & clear
    else:
        hotspot = trees_loss

    return hotspot


def class_co_occurrence(cumulative: np.ndarray) -> dict:
    """
    Identify co-occurring class changes.
    E.g., pixels where trees decreased AND crops/bare/built increased.
    Returns counts.
    """
    trees_loss = cumulative[TREES_IDX] < -0.3     # significant tree loss
    crops_gain = cumulative[DW_BANDS.index("crops")] > 0.2
    bare_gain  = cumulative[DW_BANDS.index("bare")] > 0.2
    built_gain = cumulative[DW_BANDS.index("built")] > 0.2
    shrub_gain = cumulative[DW_BANDS.index("shrub_scrub")] > 0.2
    grass_gain = cumulative[DW_BANDS.index("grass")] > 0.2

    n_total = int(np.sum(np.isfinite(cumulative[0])))
    return {
        "total_valid_pixels": n_total,
        "trees_loss_pixels": int(np.sum(trees_loss)),
        "trees_loss_AND_crops_gain": int(np.sum(trees_loss & crops_gain)),
        "trees_loss_AND_bare_gain":  int(np.sum(trees_loss & bare_gain)),
        "trees_loss_AND_built_gain": int(np.sum(trees_loss & built_gain)),
        "trees_loss_AND_shrub_gain": int(np.sum(trees_loss & shrub_gain)),
        "trees_loss_AND_grass_gain": int(np.sum(trees_loss & grass_gain)),
        "encroachment": int(np.sum(trees_loss & (crops_gain | built_gain))),
        "degradation":  int(np.sum(trees_loss & (bare_gain | shrub_gain | grass_gain))),
    }


def seasonal_profile(stack: np.ndarray, dates: list) -> dict:
    """
    Group deltas by season and compute mean per-class delta.
    Seasons: Winter (Dec-Feb), Pre-monsoon (Mar-May),
             Monsoon (Jun-Sep), Post-monsoon (Oct-Nov)
    """
    def get_season(date_str: str) -> str:
        month = int(date_str.split("-")[1])
        if month in (12, 1, 2):
            return "Winter"
        elif month in (3, 4, 5):
            return "Pre-monsoon"
        elif month in (6, 7, 8, 9):
            return "Monsoon"
        else:
            return "Post-monsoon"

    season_deltas = {}
    for i, (d_before, d_after) in enumerate(dates):
        season = get_season(d_after)
        if season not in season_deltas:
            season_deltas[season] = []
        season_deltas[season].append(stack[i, :8, :, :])  # 8-band delta

    result = {}
    for season, delta_list in season_deltas.items():
        arr = np.stack(delta_list, axis=0)  # (K, 8, H, W)
        arr = np.nan_to_num(arr, nan=0.0)
        mean_per_band = {}
        for b in range(8):
            vals = arr[:, b, :, :].ravel()
            valid = vals[np.isfinite(vals) & (vals != 0)]
            if len(valid) > 0:
                mean_per_band[DW_BANDS[b]] = {
                    "mean": round(float(np.mean(valid)), 4),
                    "std":  round(float(np.std(valid)), 4),
                    "min":  round(float(np.min(valid)), 4),
                    "max":  round(float(np.max(valid)), 4),
                    "n_windows": len(delta_list),
                }
            else:
                mean_per_band[DW_BANDS[b]] = {
                    "mean": 0.0, "std": 0.0, "n_windows": len(delta_list)
                }
        result[season] = mean_per_band

    return result


# ── Visualization ─────────────────────────────────────────────────────────────

def make_diverging_cmap():
    """Red-White-Green diverging colormap for delta visualization."""
    return plt.cm.RdYlGn


def plot_cumulative_deltas(cumulative: np.ndarray, out_dir: str):
    """8-panel figure showing cumulative delta for each class."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle("Cumulative Δ Probability (Full Year 2025)", fontsize=16, fontweight="bold")

    vmax = np.nanpercentile(np.abs(cumulative), 98)
    cmap = make_diverging_cmap()

    for idx, (ax, band_name) in enumerate(zip(axes.ravel(), DW_BANDS)):
        data = cumulative[idx]
        im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Σ Δ{band_name}", fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cum. delta")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "01_cumulative_deltas.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")
    return path


def plot_alert_frequency(freq: np.ndarray, out_dir: str):
    """Heatmap: how many alert windows flagged each pixel."""
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(freq, cmap="YlOrRd", vmin=0, vmax=max(freq.max(), 1),
                   interpolation="nearest")
    ax.set_title("Alert Frequency — times each pixel triggered (2025)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="# windows triggered")

    fig.tight_layout()
    path = os.path.join(out_dir, "02_alert_frequency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")
    return path


def plot_cloud_quality(cloud_stats: dict, out_dir: str):
    """Cloud reliability map."""
    if not cloud_stats:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle("Cloud Quality Across Alert Windows", fontsize=14, fontweight="bold")

    for ax, (key, label) in zip(axes, [
        ("mean_cloud", "Mean Cloud Probability"),
        ("max_cloud", "Max Cloud Probability"),
        ("clear_count", "# Clear Windows (< 30% cloud)")
    ]):
        data = cloud_stats[key]
        if key == "clear_count":
            im = ax.imshow(data, cmap="YlGn", interpolation="nearest")
        else:
            im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(label, fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "03_cloud_quality.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")
    return path


def plot_hotspots(hotspot: np.ndarray, cumulative_trees: np.ndarray, out_dir: str):
    """Overlay hotspots on cumulative trees delta."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: cumulative trees delta
    vmax = np.nanpercentile(np.abs(cumulative_trees), 98)
    im1 = axes[0].imshow(cumulative_trees, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                          interpolation="nearest")
    axes[0].set_title("Cumulative Δtrees", fontsize=13, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Right: hotspot binary mask overlaid
    rgba = np.zeros((*hotspot.shape, 4))
    rgba[hotspot, 0] = 1.0   # Red
    rgba[hotspot, 3] = 0.8   # Alpha

    axes[1].imshow(cumulative_trees, cmap="Greys_r", interpolation="nearest")
    axes[1].imshow(rgba, interpolation="nearest")
    n_hot = int(np.sum(hotspot))
    axes[1].set_title(f"Deforestation Hotspots ({n_hot:,} pixels)",
                      fontsize=13, fontweight="bold", color="red")
    axes[1].axis("off")
    legend = [Patch(facecolor="red", alpha=0.8, label="Hotspot (Δtrees < −0.5 + clear)")]
    axes[1].legend(handles=legend, loc="lower right", fontsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "04_hotspots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")
    return path


def plot_gt_vs_cumulative(gt: np.ndarray, cumulative: np.ndarray,
                          correlation: dict, out_dir: str):
    """Scatter plots: GT vs cumulative for key bands."""
    key_bands = ["trees", "crops", "bare", "built"]
    fig, axes = plt.subplots(1, len(key_bands), figsize=(24, 6))
    fig.suptitle("GT (Annual Dec→Dec) vs Cumulative Alert Deltas", fontsize=14,
                 fontweight="bold")

    for ax, band in zip(axes, key_bands):
        idx = DW_BANDS.index(band)
        gt_flat = gt[idx].ravel()
        cum_flat = cumulative[idx].ravel()
        valid = np.isfinite(gt_flat) & np.isfinite(cum_flat) & (gt_flat != 0)

        if valid.sum() < 100:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", fontsize=12)
            ax.set_title(band)
            continue

        # Subsample for plotting
        n = min(50000, int(valid.sum()))
        idx_sample = np.random.choice(np.where(valid)[0], size=n, replace=False)

        ax.scatter(gt_flat[idx_sample], cum_flat[idx_sample], alpha=0.1, s=1,
                   color="steelblue")
        ax.plot([-1, 1], [-1, 1], "r--", lw=1, label="1:1 line")
        ax.set_xlabel("GT Δ (annual)")
        ax.set_ylabel("Cumulative alert Δ")
        corr_val = correlation.get(band, {}).get("correlation", "N/A")
        ax.set_title(f"{band}  (r={corr_val})", fontsize=12, fontweight="bold")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "05_gt_vs_cumulative.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")
    return path


def plot_co_occurrence(co_occ: dict, out_dir: str):
    """Bar chart of co-occurrence patterns."""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = [
        ("trees_loss_AND_crops_gain", "Trees↓ + Crops↑\n(Encroachment)", "#e6550d"),
        ("trees_loss_AND_built_gain", "Trees↓ + Built↑\n(Urbanization)", "#d62728"),
        ("trees_loss_AND_bare_gain",  "Trees↓ + Bare↑\n(Clearing)", "#ff7f0e"),
        ("trees_loss_AND_shrub_gain", "Trees↓ + Shrub↑\n(Degradation)", "#8c564b"),
        ("trees_loss_AND_grass_gain", "Trees↓ + Grass↑\n(Degradation)", "#2ca02c"),
    ]

    labels  = [c[1] for c in categories]
    values  = [co_occ.get(c[0], 0) for c in categories]
    colors_ = [c[2] for c in categories]

    bars = ax.barh(labels, values, color=colors_, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Pixel Count", fontsize=12)
    ax.set_title("Class Co-occurrence with Tree Loss (cumulative Δtrees < −0.3)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "06_co_occurrence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")
    return path


def plot_seasonal_profile(seasons: dict, out_dir: str):
    """Grouped bar chart of mean delta by season for key bands."""
    key_bands = ["trees", "crops", "bare", "built", "grass", "shrub_scrub"]
    season_order = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(season_order))
    width = 0.12
    colors = ["#1b9e77", "#d95f02", "#e7298a", "#7570b3", "#66a61e", "#a6761d"]

    for i, (band, color) in enumerate(zip(key_bands, colors)):
        vals = []
        for s in season_order:
            if s in seasons and band in seasons[s]:
                vals.append(seasons[s][band]["mean"])
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=band, color=color, edgecolor="black",
               linewidth=0.5)

    ax.set_xticks(x + width * len(key_bands) / 2)
    ax.set_xticklabels(season_order, fontsize=12)
    ax.set_ylabel("Mean Δ probability", fontsize=12)
    ax.set_title("Seasonal Mean Delta by Class (non-zero pixels only)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, ncol=3)
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "07_seasonal_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")
    return path


def plot_temporal_trees_timeline(stack: np.ndarray, dates: list, out_dir: str):
    """
    Timeline chart: mean/min Δtrees per alert window, showing temporal evolution.
    """
    deltas_trees = stack[:, TREES_IDX, :, :]  # (N, H, W)
    deltas_trees = np.nan_to_num(deltas_trees, nan=0.0)

    means = []
    mins  = []
    pct5  = []
    labels = []

    for i in range(len(dates)):
        d = deltas_trees[i]
        nonzero = d[d != 0]
        if len(nonzero) > 0:
            means.append(float(np.mean(nonzero)))
            mins.append(float(np.min(nonzero)))
            pct5.append(float(np.percentile(nonzero, 5)))
        else:
            means.append(0.0)
            mins.append(0.0)
            pct5.append(0.0)
        labels.append(f"{dates[i][0]}\n→\n{dates[i][1]}")

    fig, ax = plt.subplots(figsize=(20, 7))
    x = np.arange(len(dates))

    ax.fill_between(x, mins, pct5, alpha=0.2, color="red", label="Min → 5th percentile")
    ax.fill_between(x, pct5, means, alpha=0.2, color="orange", label="5th pct → Mean")
    ax.plot(x, means, "o-", color="darkred", lw=2, ms=6, label="Mean Δtrees (non-zero)")
    ax.plot(x, mins, "v--", color="red", lw=1, ms=4, alpha=0.7, label="Min Δtrees")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Δtrees", fontsize=12)
    ax.set_title("Temporal Δtrees Profile — All Alert Windows (2025)",
                 fontsize=14, fontweight="bold")
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(-0.25, color="red", lw=1, ls=":", label="Threshold (−0.25)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "08_trees_timeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")
    return path


# ── Save raster outputs ──────────────────────────────────────────────────────

def save_raster(data: np.ndarray, profile: dict, path: str,
                band_names: list[str] | None = None):
    """Save a numpy array as GeoTIFF using the reference profile."""
    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    p = profile.copy()
    p.update(count=data.shape[0], dtype="float32", compress="lzw")

    with rasterio.open(path, "w", **p) as dst:
        for i in range(data.shape[0]):
            dst.write(data[i].astype(np.float32), i + 1)
            if band_names and i < len(band_names):
                dst.set_band_description(i + 1, band_names[i])

    log.info(f"  Raster saved: {path}")


# ── Report generation ────────────────────────────────────────────────────────

def write_report(out_dir: str, n_alerts: int, dates: list,
                 co_occ: dict, seasons: dict, corr: dict | None,
                 cloud_stats: dict, hotspot_count: int):
    """Write a markdown analysis report."""
    path = os.path.join(out_dir, "analysis_report.md")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# HAMEERPUR Alert Overlay Analysis — 2025\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Alert windows analyzed**: {n_alerts}\n")
        f.write(f"**Period**: {dates[0][0]} → {dates[-1][1]}\n\n")

        f.write("---\n\n## 1. Key Findings\n\n")

        # Hotspots
        f.write(f"### Deforestation Hotspots\n")
        f.write(f"- **{hotspot_count:,}** pixels identified as persistent deforestation "
                f"(cumulative Δtrees < −0.5 under clear sky conditions)\n\n")

        # Co-occurrence
        f.write("### Change Type Breakdown\n\n")
        f.write("| Pattern | Pixel Count | Interpretation |\n")
        f.write("|---------|------------|----------------|\n")
        f.write(f"| Trees↓ (total) | {co_occ.get('trees_loss_pixels', 0):,} | "
                f"Any significant tree loss |\n")
        f.write(f"| Trees↓ + Crops↑ | {co_occ.get('trees_loss_AND_crops_gain', 0):,} | "
                f"**Encroachment** (forest → agriculture) |\n")
        f.write(f"| Trees↓ + Built↑ | {co_occ.get('trees_loss_AND_built_gain', 0):,} | "
                f"**Urbanization** (forest → settlement) |\n")
        f.write(f"| Trees↓ + Bare↑ | {co_occ.get('trees_loss_AND_bare_gain', 0):,} | "
                f"**Clearing** (forest → bare land) |\n")
        f.write(f"| Trees↓ + Shrub↑ | {co_occ.get('trees_loss_AND_shrub_gain', 0):,} | "
                f"**Degradation** |\n")
        f.write(f"| Trees↓ + Grass↑ | {co_occ.get('trees_loss_AND_grass_gain', 0):,} | "
                f"**Degradation** |\n")
        f.write(f"| **Encroachment** total | {co_occ.get('encroachment', 0):,} | "
                f"Trees→Crops OR Trees→Built |\n")
        f.write(f"| **Degradation** total | {co_occ.get('degradation', 0):,} | "
                f"Trees→Bare/Shrub/Grass |\n\n")

        # Seasonal
        f.write("### Seasonal Patterns\n\n")
        f.write("| Season | Mean Δtrees | Mean Δcrops | Mean Δbare | # Windows |\n")
        f.write("|--------|-----------|-----------|----------|----------|\n")
        for s in ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]:
            if s in seasons:
                t = seasons[s].get("trees", {}).get("mean", 0)
                c = seasons[s].get("crops", {}).get("mean", 0)
                b = seasons[s].get("bare", {}).get("mean", 0)
                n = seasons[s].get("trees", {}).get("n_windows", 0)
                f.write(f"| {s} | {t:.4f} | {c:.4f} | {b:.4f} | {n} |\n")
        f.write("\n")

        # GT correlation
        if corr:
            f.write("### GT vs Cumulative Alert Correlation\n\n")
            f.write("| Band | Pearson r | Residual Mean | Residual Std |\n")
            f.write("|------|-----------|---------------|-------------|\n")
            for band, stats in corr.items():
                r = stats.get("correlation", "N/A")
                rm = stats.get("residual_mean", "N/A")
                rs = stats.get("residual_std", "N/A")
                f.write(f"| {band} | {r} | {rm} | {rs} |\n")
            f.write("\n")

        # Cloud
        if cloud_stats:
            mc = cloud_stats.get("mean_cloud")
            if mc is not None:
                f.write("### Cloud Quality\n\n")
                f.write(f"- Mean cloud probability (worst of before/after): "
                        f"**{np.nanmean(mc):.2%}**\n")
                clear = cloud_stats.get("clear_count")
                if clear is not None:
                    f.write(f"- Pixels with ≥10 clear windows: "
                            f"**{int(np.sum(clear >= 10)):,}**\n")
                f.write("\n")

        f.write("---\n\n## 2. Figures\n\n")
        for fig_file in sorted(os.listdir(out_dir)):
            if fig_file.endswith(".png"):
                f.write(f"### {fig_file}\n")
                f.write(f"![{fig_file}]({fig_file})\n\n")

    log.info(f"  Report saved: {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Overlay analysis of alert + GT rasters")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing GT + alert TIFs for one sub-range")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: outputs/hameerpur_analysis)")
    parser.add_argument("--tree-thresh", type=float, default=0.25)
    parser.add_argument("--other-thresh", type=float, default=0.25)
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir  = args.out_dir or os.path.join("outputs", "hameerpur_analysis")
    os.makedirs(out_dir, exist_ok=True)

    log.info(f"\n{'='*65}")
    log.info(f"  HAMEERPUR Alert Overlay Analysis")
    log.info(f"  Data dir: {data_dir}")
    log.info(f"  Output:   {out_dir}")
    log.info(f"{'='*65}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────
    log.info("Loading alert rasters ...")
    stack, dates, meta = load_alert_stack(data_dir)
    N, B, H, W = stack.shape
    log.info(f"  Stack shape: {stack.shape}  ({N} windows × {B} bands × {H}×{W} pixels)")

    log.info("Loading GT raster ...")
    gt, gt_meta = load_gt_delta(data_dir)

    # ── 2. Temporal accumulation ──────────────────────────────────────────
    log.info("\n── Temporal Accumulation ──")
    accum = temporal_accumulation(stack, dates)
    plot_cumulative_deltas(accum["cumulative_delta"], out_dir)
    save_raster(accum["cumulative_delta"], meta, 
                os.path.join(out_dir, "cumulative_delta.tif"), DW_BANDS)

    # ── 3. Alert frequency ────────────────────────────────────────────────
    log.info("\n── Alert Frequency ──")
    freq = alert_frequency(stack, args.tree_thresh, args.other_thresh)
    plot_alert_frequency(freq, out_dir)
    save_raster(freq.astype(np.float32), meta, 
                os.path.join(out_dir, "alert_frequency.tif"), ["freq"])
    log.info(f"  Max frequency: {freq.max()} windows")
    log.info(f"  Pixels fired ≥1: {int(np.sum(freq >= 1)):,}")
    log.info(f"  Pixels fired ≥3: {int(np.sum(freq >= 3)):,}")
    log.info(f"  Pixels fired ≥5: {int(np.sum(freq >= 5)):,}")

    # ── 4. Cloud quality ──────────────────────────────────────────────────
    log.info("\n── Cloud Quality ──")
    cld = cloud_quality(stack)
    plot_cloud_quality(cld, out_dir)

    # ── 5. Hotspot detection ──────────────────────────────────────────────
    log.info("\n── Hotspot Detection ──")
    cloud_mean = cld.get("mean_cloud")
    hotspot = hotspot_detection(accum["cumulative_delta"], cloud_mean)
    n_hotspot = int(np.sum(hotspot))
    log.info(f"  Hotspot pixels: {n_hotspot:,}")
    plot_hotspots(hotspot, accum["cumulative_delta"][TREES_IDX], out_dir)
    save_raster(hotspot.astype(np.float32), meta,
                os.path.join(out_dir, "hotspot_mask.tif"), ["hotspot"])

    # ── 6. Class co-occurrence ────────────────────────────────────────────
    log.info("\n── Class Co-occurrence ──")
    co_occ = class_co_occurrence(accum["cumulative_delta"])
    plot_co_occurrence(co_occ, out_dir)
    for k, v in co_occ.items():
        log.info(f"  {k}: {v:,}")

    # ── 7. Seasonal profile ───────────────────────────────────────────────
    log.info("\n── Seasonal Profile ──")
    seasons = seasonal_profile(stack, dates)
    plot_seasonal_profile(seasons, out_dir)
    for season, bands in seasons.items():
        trees_mean = bands.get("trees", {}).get("mean", 0)
        log.info(f"  {season:15s}  mean Δtrees={trees_mean:.4f}  "
                 f"({bands.get('trees', {}).get('n_windows', 0)} windows)")

    # ── 8. Temporal trees timeline ────────────────────────────────────────
    log.info("\n── Temporal Trees Timeline ──")
    plot_temporal_trees_timeline(stack, dates, out_dir)

    # ── 9. GT vs Alerts ───────────────────────────────────────────────────
    corr = None
    if gt is not None:
        log.info("\n── GT vs Cumulative Alert Comparison ──")
        # Ensure same spatial extent (H, W)
        if gt.shape[1:] == accum["cumulative_delta"].shape[1:]:
            corr = gt_vs_alerts(gt, accum["cumulative_delta"])
            plot_gt_vs_cumulative(gt, accum["cumulative_delta"], corr, out_dir)
            for band, stats in corr.items():
                log.info(f"  {band:15s}  r={stats.get('correlation', 'N/A')}")
        else:
            log.warning(f"  GT shape {gt.shape} != alert shape "
                        f"{accum['cumulative_delta'].shape} — skipping comparison")
            log.warning(f"  (Different extents — run both through same AOI)")

    # ── 10. Write report ──────────────────────────────────────────────────
    log.info("\n── Writing Report ──")
    report_path = write_report(
        out_dir, N, dates, co_occ, seasons, corr, cld, n_hotspot
    )

    log.info(f"\n{'='*65}")
    log.info(f"  Analysis complete!")
    log.info(f"  Output directory: {out_dir}")
    log.info(f"  Report: {report_path}")
    log.info(f"  Figures: {len([f for f in os.listdir(out_dir) if f.endswith('.png')])} PNG")
    log.info(f"  Rasters: {len([f for f in os.listdir(out_dir) if f.endswith('.tif')])} TIF")
    log.info(f"{'='*65}")


if __name__ == "__main__":
    main()
