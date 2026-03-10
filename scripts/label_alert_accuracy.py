"""
Alert Accuracy Labeling — Validate NRT alerts against Annual GT
================================================================

For every pixel × window, labels the alert as:
    TP          — True Positive:  alert detected change confirmed by GT
    FP          — False Positive: alert fired but GT shows no annual change
    FP_contra   — Alert + GT both significant but OPPOSITE directions (phenology)
    FN          — False Negative: GT shows change, alert missed it, AND no prior
                  window already caught it (temporal state tracking)
    TN          — True Negative:  both stable

Key engineering fixes (from critical review):
  1. Temporal State Tracking — once a pixel fires TP, later quiet windows ≠ FN
  2. Feature / Label Separation — no target leakage in the parquet export
  3. Asymmetric Magnitude — only penalize if alert EXAGGERATES GT
  4. Corroboration = Lack-of-Reversal — does the next window bounce back?

Usage:
    python scripts/label_alert_accuracy.py \\
        --data-dir data/ground_truth/HAMEERPUR \\
        --out-dir outputs/alert_accuracy
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DW_BANDS = ["water", "trees", "grass", "flooded_veg",
            "crops", "shrub_scrub", "built", "bare"]
TREES_IDX = 1
N_BANDS   = 8

ALERT_THRESH = 0.10
GT_THRESH    = 0.10

LABEL_TP        = 0
LABEL_FP        = 1
LABEL_FP_CONTRA = 2
LABEL_FN        = 3
LABEL_TN        = 4
LABEL_NAMES = {
    LABEL_TP:        "TP",
    LABEL_FP:        "FP",
    LABEL_FP_CONTRA: "FP_contra",
    LABEL_FN:        "FN",
    LABEL_TN:        "TN",
}
LABEL_COLORS = {
    LABEL_TP:        "#2ca02c",
    LABEL_FP:        "#d62728",
    LABEL_FP_CONTRA: "#ff7f0e",
    LABEL_FN:        "#9467bd",
    LABEL_TN:        "#cccccc",
}

SEASON_MAP = {
    1: "Winter", 2: "Winter",
    3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
    12: "Winter",
}
SEASONS_ORDERED = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]


# ── Data loading ──────────────────────────────────────────────────────────────

def parse_dates(filename: str) -> tuple[str, str]:
    """Extract (date_before, date_after) from filename."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})", filename)
    if m:
        return m.group(1), m.group(2)
    return "unknown", "unknown"


def get_season(date_str: str) -> str:
    """Return season name from YYYY-MM-DD string."""
    try:
        month = int(date_str.split("-")[1])
        return SEASON_MAP.get(month, "Unknown")
    except Exception:
        return "Unknown"


def load_alert_stack(data_dir: str):
    """Load alerts sorted chronologically → (N, bands, H, W), metadata."""
    pattern = os.path.join(data_dir, "alerts", "alert_delta_*.tif")
    files = sorted(glob(pattern))
    if not files:
        pattern = os.path.join(data_dir, "alert_delta_*.tif")
        files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No alert rasters in {data_dir}")

    arrays = []
    window_meta = []
    profile = None
    for f in files:
        with rasterio.open(f) as ds:
            arrays.append(ds.read())
            if profile is None:
                profile = ds.profile.copy()
        name = Path(f).stem
        d_before, d_after = parse_dates(name)
        window_meta.append({
            "idx": len(window_meta),
            "file": name,
            "date_before": d_before,
            "date_after": d_after,
            "season": get_season(d_before),
        })

    stack = np.stack(arrays, axis=0)  # (N, bands, H, W)
    log.info(f"Alert stack: {stack.shape}  ({len(files)} windows)")
    return stack, window_meta, profile


def load_gt(data_dir: str):
    files = glob(os.path.join(data_dir, "gt_delta_*.tif"))
    if not files:
        raise FileNotFoundError(f"No GT raster in {data_dir}")
    with rasterio.open(files[0]) as ds:
        gt = ds.read()
    log.info(f"GT: {gt.shape}")
    return gt


# ── Phase 2: Label per-pixel per-window ───────────────────────────────────────

def label_all_windows(stack: np.ndarray, gt: np.ndarray,
                      window_meta: list) -> dict:
    """
    Process window-by-window with temporal state tracking.

    Returns dict with:
      - per_pixel aggregates (H, W)
      - per_window statistics
      - records for parquet export (filtered to active pixels)
    """
    N, B, H, W = stack.shape
    n_pix = H * W

    gt_trees = gt[TREES_IDX].ravel()                # (n_pix,)
    gt_deltas = gt[:N_BANDS].reshape(N_BANDS, n_pix)  # (8, n_pix)

    is_gt_change = np.abs(gt_trees) >= GT_THRESH      # (n_pix,)

    # ── Accumulators ──────────────────────────────────────────────────
    # Per-pixel: count of each label type (for trees band)
    counts = np.zeros((5, n_pix), dtype=np.int32)   # [TP, FP, FPc, FN, TN]

    # Per-window: aggregate counts
    window_stats = []

    # Temporal state tracking (Trap 1 fix)
    already_detected = np.zeros(n_pix, dtype=bool)

    # Pre-compute lack-of-reversal for each window (Trap 4 fix)
    # For window W, check if the NEXT window reverses the trees delta
    trees_deltas = stack[:, TREES_IDX, :, :].reshape(N, n_pix)  # (N, n_pix)

    # Records for export (only active pixels)
    # Determine active pixels: |alert trees| > 0.05 in ANY window OR GT changed
    any_active = np.any(np.abs(trees_deltas) > 0.05, axis=0) | is_gt_change
    active_idx = np.where(any_active)[0]
    n_active = len(active_idx)
    log.info(f"Active pixels (|Δtrees| > 0.05 in any window OR GT changed): "
             f"{n_active:,} / {n_pix:,}")

    # Also sample 1% of stable TN pixels (reviewer's suggestion)
    stable_idx = np.where(~any_active)[0]
    rng = np.random.default_rng(42)
    n_sample = max(1, len(stable_idx) // 100)
    tn_sample = rng.choice(stable_idx, size=n_sample, replace=False)
    export_idx = np.sort(np.concatenate([active_idx, tn_sample]))
    log.info(f"Export pixels: {len(export_idx):,} "
             f"({n_active:,} active + {n_sample:,} TN sample)")

    # Pre-allocate record arrays for export
    # Features (X): alert_val, cloud_before, cloud_after, season
    # Label (Y): TP/FP/FP_contra/FN/TN
    # Stored per window, then concatenated
    export_records = []

    has_cloud = B >= 10

    # ── Process each window ───────────────────────────────────────────
    for w in range(N):
        alert_trees = trees_deltas[w]        # (n_pix,)
        is_alert = np.abs(alert_trees) >= ALERT_THRESH
        same_dir = np.sign(alert_trees) == np.sign(gt_trees)

        # ── Label logic (trees band) ─────────────────────────────────
        tp        = is_alert & is_gt_change & same_dir
        fp        = is_alert & ~is_gt_change
        fp_contra = is_alert & is_gt_change & ~same_dir

        # FN: GT changed, alert didn't fire, AND not already caught (Trap 1)
        fn = ~is_alert & is_gt_change & ~already_detected
        tn = ~is_alert & ~is_gt_change

        # Also: pixels where alert didn't fire AND GT changed BUT already
        # caught → these are TN (the pixel is already cleared, expected 0 delta)
        already_cleared_quiet = ~is_alert & is_gt_change & already_detected
        tn = tn | already_cleared_quiet

        # Update state tracking
        already_detected = already_detected | tp

        # Accumulate per-pixel counts
        counts[LABEL_TP]        += tp.astype(np.int32)
        counts[LABEL_FP]        += fp.astype(np.int32)
        counts[LABEL_FP_CONTRA] += fp_contra.astype(np.int32)
        counts[LABEL_FN]        += fn.astype(np.int32)
        counts[LABEL_TN]        += tn.astype(np.int32)

        # Per-window stats
        w_stats = {
            "window": w,
            "date_before": window_meta[w]["date_before"],
            "date_after": window_meta[w]["date_after"],
            "season": window_meta[w]["season"],
            "n_tp": int(tp.sum()),
            "n_fp": int(fp.sum()),
            "n_fp_contra": int(fp_contra.sum()),
            "n_fn": int(fn.sum()),
            "n_tn": int(tn.sum()),
        }
        n_alerts = w_stats["n_tp"] + w_stats["n_fp"] + w_stats["n_fp_contra"]
        w_stats["precision"] = (
            round(w_stats["n_tp"] / n_alerts, 4) if n_alerts > 0 else 0.0
        )
        window_stats.append(w_stats)

        # ── Lack-of-reversal (Trap 4 fix) ────────────────────────────
        # For each pixel in this window, does the NEXT window reverse?
        if w < N - 1:
            next_delta = trees_deltas[w + 1]
            # Reversal = next window has opposite sign AND significant
            reversal = (
                (np.sign(alert_trees) != np.sign(next_delta))
                & (np.abs(next_delta) >= ALERT_THRESH)
            )
            no_reversal = ~reversal
        else:
            # Last window: no next window to check
            no_reversal = np.ones(n_pix, dtype=bool)

        # ── Magnitude: asymmetric penalty (Trap 3 fix) ───────────────
        # Only penalize if alert EXAGGERATES GT
        # exaggeration_ratio > 1 means alert is louder than GT → bad
        # exaggeration_ratio ≤ 1 means alert caught early onset → fine
        with np.errstate(divide="ignore", invalid="ignore"):
            exaggeration_ratio = np.where(
                np.abs(gt_trees) > 0.01,
                np.abs(alert_trees) / np.abs(gt_trees),
                np.where(np.abs(alert_trees) > ALERT_THRESH, 999.0, 0.0),
            )
        is_exaggerated = exaggeration_ratio > 1.5

        # ── Build export records for this window ─────────────────────
        # Only for export_idx pixels
        alert_vals_exp  = alert_trees[export_idx]
        gt_vals_exp     = gt_trees[export_idx]

        # All 8 bands for export
        alert_all_exp = stack[w, :N_BANDS, :, :].reshape(N_BANDS, n_pix)[:, export_idx]
        gt_all_exp    = gt_deltas[:, export_idx]

        # Cloud
        if has_cloud:
            cld_before_exp = stack[w, 8, :, :].ravel()[export_idx]
            cld_after_exp  = stack[w, 9, :, :].ravel()[export_idx]
        else:
            cld_before_exp = np.zeros(len(export_idx), dtype=np.float32)
            cld_after_exp  = np.zeros(len(export_idx), dtype=np.float32)

        # Label for export pixels
        label_arr = np.full(len(export_idx), LABEL_TN, dtype=np.int32)
        label_arr[tp[export_idx]]        = LABEL_TP
        label_arr[fp[export_idx]]        = LABEL_FP
        label_arr[fp_contra[export_idx]] = LABEL_FP_CONTRA
        label_arr[fn[export_idx]]        = LABEL_FN

        # Features: no-reversal and exaggeration for export pixels
        no_rev_exp = no_reversal[export_idx]
        exagg_exp  = is_exaggerated[export_idx]

        # Build record dict for this window
        rec = {
            "pixel_idx": export_idx,
            "window_idx": np.full(len(export_idx), w, dtype=np.int32),
            "season": window_meta[w]["season"],
            "date_before": window_meta[w]["date_before"],
            "date_after": window_meta[w]["date_after"],

            # ── TARGET (Y) — purely from alert vs GT comparison ──
            "label": label_arr,

            # ── FEATURES (X) — observable at alert time, NO GT info ──
            "alert_trees": alert_vals_exp.astype(np.float32),
            "alert_crops": alert_all_exp[4].astype(np.float32),
            "alert_bare": alert_all_exp[7].astype(np.float32),
            "alert_built": alert_all_exp[6].astype(np.float32),
            "alert_abs_max": np.max(np.abs(alert_all_exp), axis=0).astype(np.float32),
            "cloud_before": cld_before_exp.astype(np.float32),
            "cloud_after": cld_after_exp.astype(np.float32),
            "cloud_worst": np.maximum(cld_before_exp, cld_after_exp).astype(np.float32),
            "no_reversal": no_rev_exp.astype(np.int8),
            "is_exaggerated": exagg_exp.astype(np.int8),

            # ── GT values (for analysis only, NOT for ML features) ──
            "gt_trees": gt_vals_exp.astype(np.float32),
        }
        export_records.append(rec)

        log.info(
            f"  W{w:02d} ({window_meta[w]['date_before']} → "
            f"{window_meta[w]['date_after']}  {window_meta[w]['season']:14s})  "
            f"TP={w_stats['n_tp']:>6,}  FP={w_stats['n_fp']:>6,}  "
            f"FPc={w_stats['n_fp_contra']:>5,}  "
            f"FN={w_stats['n_fn']:>6,}  "
            f"Prec={w_stats['precision']:.3f}"
        )

    return {
        "counts": counts,           # (5, n_pix)
        "window_stats": window_stats,
        "export_records": export_records,
        "H": H, "W": W,
        "already_detected": already_detected,
    }


# ── Phase 3: Analysis figures ─────────────────────────────────────────────────

def plot_analyses(result: dict, out_dir: str, profile: dict):
    """Generate all 8 figures + accuracy rasters."""

    H, W = result["H"], result["W"]
    n_pix = H * W
    counts = result["counts"]
    ws = result["window_stats"]

    # ── Fig 1: Per-pixel accuracy raster ──────────────────────────────
    tp_count    = counts[LABEL_TP]
    alert_count = counts[LABEL_TP] + counts[LABEL_FP] + counts[LABEL_FP_CONTRA]

    with np.errstate(divide="ignore", invalid="ignore"):
        pixel_accuracy = np.where(
            alert_count > 0,
            tp_count.astype(np.float32) / alert_count.astype(np.float32),
            np.nan,
        )

    acc_map = pixel_accuracy.reshape(H, W)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(acc_map, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("Per-pixel Alert Accuracy\n(TP / Total Alerts Fired)",
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_pixel_accuracy.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: 01_pixel_accuracy.png")

    # Save as GeoTIFF
    p = profile.copy()
    p.update(count=1, dtype="float32", compress="lzw")
    with rasterio.open(os.path.join(out_dir, "pixel_accuracy.tif"), "w", **p) as dst:
        dst.write(np.nan_to_num(acc_map, nan=-1).astype(np.float32), 1)

    # ── Fig 2: Per-window accuracy bars ───────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    labels_w = [f"W{w['window']}\n{w['date_before'][:5]}" for w in ws]
    tp_vals = [w["n_tp"] for w in ws]
    fp_vals = [w["n_fp"] for w in ws]
    fpc_vals = [w["n_fp_contra"] for w in ws]
    x = np.arange(len(ws))

    bars_tp  = ax.bar(x, tp_vals, label="TP (confirmed)",
                      color=LABEL_COLORS[LABEL_TP], edgecolor="black", linewidth=0.5)
    bars_fp  = ax.bar(x, fp_vals, bottom=tp_vals, label="FP (not confirmed)",
                      color=LABEL_COLORS[LABEL_FP], edgecolor="black", linewidth=0.5)
    bottoms2 = [t + f for t, f in zip(tp_vals, fp_vals)]
    bars_fpc = ax.bar(x, fpc_vals, bottom=bottoms2, label="FP_contra (opposite direction)",
                      color=LABEL_COLORS[LABEL_FP_CONTRA], edgecolor="black", linewidth=0.5)

    # Precision annotation
    for i, w in enumerate(ws):
        ax.text(i, tp_vals[i] + fp_vals[i] + fpc_vals[i] + 200,
                f"{w['precision']:.1%}", ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_w, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Pixel count", fontsize=11)
    ax.set_title("Per-Window Alert Breakdown (with Precision %)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_per_window_accuracy.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: 02_per_window_accuracy.png")

    # ── Fig 3: Overall TP/FP/FN/TN pie ───────────────────────────────
    totals = {LABEL_NAMES[i]: int(counts[i].sum()) for i in range(5)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: all labels pie
    vals = [totals[k] for k in ["TP", "FP", "FP_contra", "FN", "TN"]]
    colors = [LABEL_COLORS[i] for i in range(5)]
    names = list(LABEL_NAMES.values())
    axes[0].pie(vals, labels=names, colors=colors, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 10})
    axes[0].set_title("All Pixel-Window Labels", fontsize=12, fontweight="bold")

    # Right: alert-only pie (exclude TN)
    alert_vals = [totals["TP"], totals["FP"], totals["FP_contra"]]
    alert_names = ["TP", "FP", "FP_contra"]
    alert_colors = [LABEL_COLORS[LABEL_TP], LABEL_COLORS[LABEL_FP],
                    LABEL_COLORS[LABEL_FP_CONTRA]]
    axes[1].pie(alert_vals, labels=alert_names, colors=alert_colors,
                autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    axes[1].set_title("Alert-Only Labels (excl. TN/FN)", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_label_breakdown.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: 03_label_breakdown.png")

    # ── Fig 4: Season × accuracy heatmap ──────────────────────────────
    season_data = {s: {"tp": 0, "fp": 0, "fpc": 0, "fn": 0} for s in SEASONS_ORDERED}
    for w in ws:
        s = w["season"]
        if s in season_data:
            season_data[s]["tp"]  += w["n_tp"]
            season_data[s]["fp"]  += w["n_fp"]
            season_data[s]["fpc"] += w["n_fp_contra"]
            season_data[s]["fn"]  += w["n_fn"]

    fig, ax = plt.subplots(figsize=(10, 5))
    season_labels = SEASONS_ORDERED
    prec_vals = []
    for s in season_labels:
        d = season_data[s]
        total = d["tp"] + d["fp"] + d["fpc"]
        prec_vals.append(d["tp"] / total if total > 0 else 0)

    colors_s = plt.cm.RdYlGn(np.array(prec_vals))
    bars = ax.bar(season_labels, prec_vals, color=colors_s,
                  edgecolor="black", linewidth=0.5)
    for i, (bar, v) in enumerate(zip(bars, prec_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Alert Precision (TP / Alerts)", fontsize=11)
    ax.set_title("Alert Precision by Season", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_season_accuracy.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: 04_season_accuracy.png")

    # ── Fig 5: Cloud vs accuracy ──────────────────────────────────────
    # Aggregate from export records
    all_cloud = []
    all_labels = []
    for rec in result["export_records"]:
        mask = (rec["label"] != LABEL_TN) & (rec["label"] != LABEL_FN)
        all_cloud.append(rec["cloud_worst"][mask])
        all_labels.append(rec["label"][mask])

    if all_cloud:
        all_cloud = np.concatenate(all_cloud)
        all_labels = np.concatenate(all_labels)

        bins = np.array([0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
        bin_idx = np.digitize(all_cloud, bins) - 1
        bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]

        prec_by_cloud = []
        counts_by_cloud = []
        for b in range(len(bins) - 1):
            mask_b = bin_idx == b
            if mask_b.sum() == 0:
                prec_by_cloud.append(0)
                counts_by_cloud.append(0)
                continue
            tp_b = (all_labels[mask_b] == LABEL_TP).sum()
            total_b = mask_b.sum()
            prec_by_cloud.append(tp_b / total_b if total_b > 0 else 0)
            counts_by_cloud.append(int(total_b))

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(bin_labels, prec_by_cloud, color="steelblue",
                edgecolor="black", linewidth=0.5, alpha=0.8)
        ax1.set_xlabel("Cloud Probability (worst of before/after)", fontsize=11)
        ax1.set_ylabel("Precision (TP / Alerts)", fontsize=11, color="steelblue")
        ax1.set_ylim(0, 1)

        ax2 = ax1.twinx()
        ax2.plot(bin_labels, counts_by_cloud, "o-", color="crimson",
                 linewidth=2, markersize=6)
        ax2.set_ylabel("Alert count", fontsize=11, color="crimson")

        ax1.set_title("Alert Precision vs Cloud Probability",
                      fontsize=13, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "05_cloud_vs_accuracy.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved: 05_cloud_vs_accuracy.png")

    # ── Fig 6: Corroboration (no-reversal) vs accuracy ────────────────
    all_rev = []
    all_lab2 = []
    for rec in result["export_records"]:
        alert_mask = rec["label"] <= LABEL_FP_CONTRA  # TP, FP, FP_contra only
        all_rev.append(rec["no_reversal"][alert_mask])
        all_lab2.append(rec["label"][alert_mask])

    if all_rev:
        all_rev = np.concatenate(all_rev)
        all_lab2 = np.concatenate(all_lab2)

        fig, ax = plt.subplots(figsize=(8, 5))
        for rev_val, label_text in [(1, "No reversal\n(corroborated)"),
                                     (0, "Reversed\n(bounced back)")]:
            m = all_rev == rev_val
            if m.sum() == 0:
                continue
            tp_n = (all_lab2[m] == LABEL_TP).sum()
            total_n = m.sum()
            prec_n = tp_n / total_n
            ax.bar(label_text, prec_n,
                   color=LABEL_COLORS[LABEL_TP] if rev_val else LABEL_COLORS[LABEL_FP],
                   edgecolor="black", linewidth=0.5)
            ax.text(rev_val if rev_val == 0 else rev_val - 1,
                    prec_n + 0.02, f"{prec_n:.1%}\n(n={total_n:,})",
                    ha="center", fontsize=11, fontweight="bold")

        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title("Alert Precision: Corroborated vs Reversed",
                     fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "06_corroboration_effect.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved: 06_corroboration_effect.png")

    # ── Fig 7: Magnitude vs accuracy scatter ──────────────────────────
    all_mag = []
    all_lab3 = []
    for rec in result["export_records"]:
        alert_mask = np.abs(rec["alert_trees"]) >= ALERT_THRESH
        all_mag.append(rec["alert_trees"][alert_mask])
        all_lab3.append(rec["label"][alert_mask])

    if all_mag:
        all_mag = np.concatenate(all_mag)
        all_lab3 = np.concatenate(all_lab3)

        # Bin by magnitude
        mag_abs = np.abs(all_mag)
        bins_m = np.array([0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.0])
        bin_idx_m = np.digitize(mag_abs, bins_m) - 1
        bin_labels_m = [f"{bins_m[i]:.2f}-{bins_m[i+1]:.2f}"
                        for i in range(len(bins_m) - 1)]

        prec_by_mag = []
        for b in range(len(bins_m) - 1):
            m = bin_idx_m == b
            if m.sum() == 0:
                prec_by_mag.append(np.nan)
                continue
            prec_by_mag.append((all_lab3[m] == LABEL_TP).sum() / m.sum())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(bin_labels_m, prec_by_mag, color="teal",
               edgecolor="black", linewidth=0.5)
        ax.set_xlabel("|Alert Δtrees|", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title("Alert Precision by Signal Magnitude",
                     fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "07_magnitude_vs_accuracy.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved: 07_magnitude_vs_accuracy.png")

    # ── Fig 8: Temporal state — cumulative TP detection curve ─────────
    cum_detected = np.zeros(len(ws))
    running = 0
    for i, w in enumerate(ws):
        running += w["n_tp"]
        cum_detected[i] = running

    # Count total GT-change pixels from the FN counts of window 0
    # (before any detection, FN = all GT-change pixels)
    total_gt_change = ws[0]["n_fn"] + ws[0]["n_tp"] + ws[0]["n_fp_contra"] if ws else 1
    # Normalize as fraction of total GT change pixels
    if total_gt_change > 0:
        cum_frac = cum_detected / total_gt_change
    else:
        cum_frac = cum_detected

    fig, ax = plt.subplots(figsize=(12, 5))
    x_labels = [f"W{w['window']}\n{w['date_before'][:5]}" for w in ws]
    ax.plot(range(len(ws)), cum_frac, "o-", color="darkgreen",
            linewidth=2, markersize=8)
    ax.fill_between(range(len(ws)), cum_frac, alpha=0.2, color="green")
    ax.set_xticks(range(len(ws)))
    ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Fraction of GT Change Detected", fontsize=11)
    ax.set_title("Cumulative Detection Rate\n(with Temporal State Tracking)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, min(1.0, cum_frac[-1] * 1.2) if len(cum_frac) > 0 else 1)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="100% GT change")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_cumulative_detection.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: 08_cumulative_detection.png")

    return totals


# ── Phase 4: Export ───────────────────────────────────────────────────────────

def export_parquet(records: list, out_dir: str):
    """Save labeled records as parquet (via numpy → simple CSV if no pyarrow)."""
    try:
        import pandas as pd

        dfs = []
        for rec in records:
            df = pd.DataFrame({
                "pixel_idx":     rec["pixel_idx"],
                "window_idx":    rec["window_idx"],
                "season":        rec["season"],
                "date_before":   rec["date_before"],
                "date_after":    rec["date_after"],
                "label":         rec["label"],
                "label_name":    [LABEL_NAMES[l] for l in rec["label"]],
                "alert_trees":   rec["alert_trees"],
                "alert_crops":   rec["alert_crops"],
                "alert_bare":    rec["alert_bare"],
                "alert_built":   rec["alert_built"],
                "alert_abs_max": rec["alert_abs_max"],
                "cloud_before":  rec["cloud_before"],
                "cloud_after":   rec["cloud_after"],
                "cloud_worst":   rec["cloud_worst"],
                "no_reversal":   rec["no_reversal"],
                "is_exaggerated": rec["is_exaggerated"],
                "gt_trees":      rec["gt_trees"],
            })
            dfs.append(df)

        full = pd.concat(dfs, ignore_index=True)

        # Try parquet first
        try:
            path = os.path.join(out_dir, "labeled_records.parquet")
            full.to_parquet(path, engine="pyarrow", index=False)
            log.info(f"  Saved: {path}  ({len(full):,} records)")
        except Exception:
            path = os.path.join(out_dir, "labeled_records.csv")
            full.to_csv(path, index=False)
            log.info(f"  Saved: {path}  ({len(full):,} records)")

        # Summary stats
        log.info(f"\n  Label distribution in export:")
        for name in ["TP", "FP", "FP_contra", "FN", "TN"]:
            n = (full["label_name"] == name).sum()
            pct = n / len(full) * 100
            log.info(f"    {name:12s}: {n:>10,}  ({pct:.2f}%)")

        return len(full)

    except ImportError:
        log.warning("  pandas not available — skipping parquet export")
        return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="outputs/alert_accuracy")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # Phase 1: Load
    log.info("=" * 65)
    log.info("  Phase 1: LOAD")
    log.info("=" * 65)
    stack, window_meta, profile = load_alert_stack(args.data_dir)
    gt = load_gt(args.data_dir)

    # Phase 2: Label
    log.info("\n" + "=" * 65)
    log.info("  Phase 2: LABEL (with temporal state tracking)")
    log.info("=" * 65)
    result = label_all_windows(stack, gt, window_meta)

    # Phase 3: Analyze
    log.info("\n" + "=" * 65)
    log.info("  Phase 3: ANALYZE")
    log.info("=" * 65)
    totals = plot_analyses(result, args.out_dir, profile)

    # Phase 4: Export
    log.info("\n" + "=" * 65)
    log.info("  Phase 4: EXPORT")
    log.info("=" * 65)
    n_records = export_parquet(result["export_records"], args.out_dir)

    # Summary report
    elapsed = time.time() - t0
    summary = {
        "elapsed_seconds": round(elapsed, 1),
        "n_windows": len(window_meta),
        "label_totals": totals,
        "n_export_records": n_records,
        "already_detected_pixels": int(result["already_detected"].sum()),
        "windows": result["window_stats"],
    }
    with open(os.path.join(args.out_dir, "accuracy_report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\n{'='*65}")
    log.info(f"  DONE in {elapsed:.1f}s")
    log.info(f"  Already-detected pixels (TP in ≥1 window): "
             f"{int(result['already_detected'].sum()):,}")
    log.info(f"  Outputs: {args.out_dir}/")
    log.info(f"{'='*65}")


if __name__ == "__main__":
    main()
