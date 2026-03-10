#!/usr/bin/env python3
"""
Classify GT Raster by Date of Occurrence
=========================================
Creates a raster where each pixel with real GT tree loss gets assigned
the date/window index of its first detection by an alert window.

Output bands:
  1. first_detection_window  — Window index (0-14) of first alert, 255=undetected, 0=no change
  2. first_detection_doy     — Day-of-year (1-365) when first detected, 0=undetected/no change
  3. gt_severity             — Encoded severity: 0=none, 1=moderate(0.1-0.3), 2=severe(>0.3)
  4. cumulative_loss         — Sum of per-window tree losses (float32)

Usage:
    python scripts/classify_gt_by_date.py
"""

# ── PROJ/GDAL env fix ──
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()
del _os, _sys
# ────────────────────────

import numpy as np
import rasterio
import os
import sys
import logging
from glob import glob
from datetime import datetime

log = logging.getLogger("classify_gt_date")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)

GT_DIR    = "data/ground_truth/HAMEERPUR"
ALERT_DIR = os.path.join(GT_DIR, "alerts")
GT_TIF    = os.path.join(GT_DIR, "gt_delta_2024_2025_HAMEERPUR.tif")
OUT_TIF   = os.path.join("outputs", "gt_classified_by_date_HAMEERPUR.tif")

# Enriched TIF band layout
DW_DELTA_START = 16
TREES_IDX = 1
ALERT_THRESH = 0.10


def parse_dates_from_filename(fname):
    """Extract start/end dates from filename like alert_enriched_HAMEERPUR_2025-02-02_to_2025-02-07.tif"""
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})", fname)
    if m:
        d1 = datetime.strptime(m.group(1), "%Y-%m-%d")
        d2 = datetime.strptime(m.group(2), "%Y-%m-%d")
        return d1, d2
    return None, None


def get_season(month):
    """Return season label from month."""
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Pre-monsoon"
    elif month in (6, 7, 8, 9):
        return "Monsoon"
    else:
        return "Post-monsoon"


def main():
    # ── Load GT ─────────────────────────────────────────────────────
    with rasterio.open(GT_TIF) as ds:
        gt = ds.read()
        gt_profile = ds.profile.copy()
    gt_trees = gt[TREES_IDX]  # (H, W) — annual tree delta
    H, W = gt_trees.shape
    log.info(f"Grid: {H}x{W}")

    # ── Load alert windows ──────────────────────────────────────────
    tifs = sorted(glob(os.path.join(ALERT_DIR, "alert_enriched_HAMEERPUR_*.tif")))
    n_windows = len(tifs)
    log.info(f"Found {n_windows} alert windows")

    # Parse dates for each window
    window_info = []
    for wi, tif in enumerate(tifs):
        d1, d2 = parse_dates_from_filename(tif)
        if d1 is None:
            # Fallback: try tags
            with rasterio.open(tif) as ds:
                tags = ds.tags()
            d1_str = tags.get("window_start", "2025-01-01")
            d2_str = tags.get("window_end", "2025-01-01")
            try:
                d1 = datetime.strptime(d1_str, "%Y-%m-%d")
                d2 = datetime.strptime(d2_str, "%Y-%m-%d")
            except ValueError:
                d1 = datetime(2025, 1, 1)
                d2 = datetime(2025, 1, 1)
        mid_date = d1 + (d2 - d1) / 2
        doy = mid_date.timetuple().tm_yday
        season = get_season(mid_date.month)
        window_info.append({
            "idx": wi,
            "start": d1,
            "end": d2,
            "mid": mid_date,
            "doy": doy,
            "season": season,
            "label": f"W{wi:02d} ({d1.strftime('%b %d')}-{d2.strftime('%b %d')} {season})"
        })
        log.info(f"  W{wi:02d}: {d1.date()} → {d2.date()}  DOY={doy}  {season}")

    # ── Initialize output bands ─────────────────────────────────────
    # Band 1: first detection window index (uint8, 255=undetected, 0=no GT change)
    first_win = np.zeros((H, W), dtype=np.uint8)       # 0 = no change
    # Band 2: first detection day-of-year (uint16, 0=undetected/no change)
    first_doy = np.zeros((H, W), dtype=np.uint16)
    # Band 3: GT severity (uint8: 0=none, 1=moderate, 2=severe)
    severity = np.zeros((H, W), dtype=np.uint8)
    # Band 4: cumulative loss from all windows (float32)
    cum_loss = np.zeros((H, W), dtype=np.float32)
    # Band 5: max single-window loss (float32)
    max_loss = np.zeros((H, W), dtype=np.float32)
    # Band 6: number of windows that fire (uint8)
    n_fires = np.zeros((H, W), dtype=np.uint8)

    # Classify GT severity
    gt_real = gt_trees < -0.10
    gt_severe = gt_trees < -0.30
    severity[gt_real & ~gt_severe] = 1   # moderate (0.10-0.30)
    severity[gt_severe] = 2              # severe (>0.30)

    # Mark undetected placeholder — pixels with GT change but no detection yet
    # Will be set to 255 after processing
    has_change = gt_trees < -0.10
    first_win[has_change] = 255  # undetected initially

    log.info(f"GT real change pixels: {has_change.sum():,}")
    log.info(f"  Moderate (0.10-0.30): {(severity == 1).sum():,}")
    log.info(f"  Severe (>0.30):       {(severity == 2).sum():,}")

    # ── Process each window chronologically ─────────────────────────
    for wi, tif in enumerate(tifs):
        with rasterio.open(tif) as ds:
            data = ds.read()

        delta_trees = data[DW_DELTA_START + TREES_IDX]  # (H, W)
        is_alert = delta_trees <= -ALERT_THRESH

        # Track loss
        w_loss = np.abs(np.minimum(delta_trees, 0))
        # Handle NaN
        w_loss = np.nan_to_num(w_loss, nan=0.0)
        cum_loss += w_loss
        better = w_loss > max_loss
        max_loss[better] = w_loss[better]
        n_fires += np.nan_to_num(is_alert, nan=0).astype(np.uint8)

        # First detection: only for pixels that haven't been detected yet (first_win == 255)
        new_detect = is_alert & (first_win == 255)
        n_new = new_detect.sum()

        if n_new > 0:
            first_win[new_detect] = wi + 1   # 1-indexed (0=no change, 255=undetected)
            first_doy[new_detect] = window_info[wi]["doy"]

        info = window_info[wi]
        log.info(f"  {info['label']}: alerts={int(is_alert.sum()):>7,}  "
                 f"new_GT_detections={int(n_new):>5,}  "
                 f"mean_loss={w_loss[has_change].mean():.4f}")

    # ── Summary ─────────────────────────────────────────────────────
    detected = (first_win > 0) & (first_win < 255)
    still_undetected = first_win == 255
    no_change = first_win == 0

    log.info(f"\n{'='*60}")
    log.info(f"  CLASSIFICATION SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  No GT change:   {no_change.sum():>8,} pixels")
    log.info(f"  Detected:       {detected.sum():>8,} pixels ({100*detected.sum()/has_change.sum():.1f}%)")
    log.info(f"  Undetected:     {still_undetected.sum():>8,} pixels ({100*still_undetected.sum()/has_change.sum():.1f}%)")

    # Per-window detection summary
    log.info(f"\n  First detection by window:")
    for wi in range(n_windows):
        info = window_info[wi]
        count = (first_win == (wi + 1)).sum()
        count_sev = ((first_win == (wi + 1)) & gt_severe).sum()
        if count > 0:
            log.info(f"    {info['label']}: {int(count):>6,} pixels  "
                     f"(severe: {int(count_sev):>4,})")

    # Season summary
    log.info(f"\n  First detection by season:")
    season_counts = {}
    for wi in range(n_windows):
        s = window_info[wi]["season"]
        c = int((first_win == (wi + 1)).sum())
        season_counts[s] = season_counts.get(s, 0) + c
    for s in ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]:
        c = season_counts.get(s, 0)
        log.info(f"    {s:15s}: {c:>6,} ({100*c/max(detected.sum(),1):.1f}%)")

    # ── Write output raster ─────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_TIF), exist_ok=True)

    out_profile = gt_profile.copy()
    out_profile.update(
        count=6,
        dtype="float32",
        compress="deflate",
        nodata=None,
    )

    with rasterio.open(OUT_TIF, "w", **out_profile) as dst:
        dst.write(first_win.astype(np.float32), 1)
        dst.write(first_doy.astype(np.float32), 2)
        dst.write(severity.astype(np.float32), 3)
        dst.write(cum_loss, 4)
        dst.write(max_loss, 5)
        dst.write(n_fires.astype(np.float32), 6)

        dst.set_band_description(1, "first_detection_window (0=no_change, 1-15=window_index, 255=undetected)")
        dst.set_band_description(2, "first_detection_doy (day-of-year, 0=none)")
        dst.set_band_description(3, "gt_severity (0=none, 1=moderate_0.1-0.3, 2=severe_>0.3)")
        dst.set_band_description(4, "cumulative_loss (sum of per-window tree losses)")
        dst.set_band_description(5, "max_single_window_loss")
        dst.set_band_description(6, "n_fires (windows that triggered)")

        # Store window metadata as tags
        dst.update_tags(
            n_windows=str(n_windows),
            alert_threshold=str(ALERT_THRESH),
            **{f"W{wi+1:02d}": f"{window_info[wi]['start'].date()}_to_{window_info[wi]['end'].date()}_{window_info[wi]['season']}"
               for wi in range(n_windows)}
        )

    log.info(f"\n  Output: {OUT_TIF}")
    log.info(f"  6 bands: first_win, first_doy, severity, cum_loss, max_loss, n_fires")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
