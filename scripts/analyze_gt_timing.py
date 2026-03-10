"""
GT Change Timing Analysis
==========================
Cross-references GT raster (pixels with tree loss > threshold) with alert
windows to determine WHEN each deforestation event was first detected.
"""

# ── PROJ/GDAL env fix ──
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()
del _os, _sys
# ────────────────────────

import numpy as np
import rasterio
from glob import glob
import os, re

# ── Config ────────────────────────────────────────────────────────────────────
GT_RASTER = "data/ground_truth/HAMEERPUR/gt_delta_2024_2025_HAMEERPUR.tif"
ALERT_DIR = "data/ground_truth/HAMEERPUR/alerts"
TREE_LOSS_THRESH = 0.3  # |delta_trees| > this
TREES_IDX = 1  # band index for trees in GT (0-indexed)
ENRICHED_DELTA = slice(16, 24)  # delta bands in enriched rasters

def parse_dates(stem):
    m = re.search(r"alert_(?:delta|enriched)_(.+?)_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})", stem)
    if m:
        return m.group(2), m.group(3), m.group(1)
    return "unknown", "unknown", "UNKNOWN"

SEASON_MAP = {
    1: "Winter", 2: "Winter",
    3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
    12: "Winter",
}

# ── Load GT ───────────────────────────────────────────────────────────────────
with rasterio.open(GT_RASTER) as ds:
    gt = ds.read()
H, W = gt.shape[1], gt.shape[2]
n_pix = H * W

gt_trees = gt[TREES_IDX].ravel()  # delta_trees from GT (annual change)
significant_loss = gt_trees < -TREE_LOSS_THRESH
n_significant = int(significant_loss.sum())
print(f"\n{'='*70}")
print(f"  GT RASTER: {H}x{W} = {n_pix:,} pixels")
print(f"  Pixels with tree loss > {TREE_LOSS_THRESH}: {n_significant:,}")
print(f"{'='*70}\n")

# ── Load alert windows (sorted by date) ───────────────────────────────────────
alert_files = sorted(glob(os.path.join(ALERT_DIR, "alert_enriched_*.tif")))
print(f"Found {len(alert_files)} alert windows\n")

# Track per-pixel: first window with ANY tree loss, first with SEVERE loss
first_alert_window = np.full(n_pix, -1, dtype=np.int32)  # any delta_trees < -0.1
first_severe_window = np.full(n_pix, -1, dtype=np.int32)  # delta_trees < -0.3
peak_loss_window = np.full(n_pix, -1, dtype=np.int32)     # window with biggest loss
peak_loss_value = np.zeros(n_pix, dtype=np.float32)
cum_tree_loss = np.zeros(n_pix, dtype=np.float32)

window_info = []

for w, f in enumerate(alert_files):
    stem = os.path.splitext(os.path.basename(f))[0]
    d_before, d_after, sub = parse_dates(stem)
    month = int(d_before.split("-")[1]) if d_before != "unknown" else 1
    season = SEASON_MAP.get(month, "Unknown")
    
    with rasterio.open(f) as ds:
        data = ds.read()
    
    is_enriched = data.shape[0] >= 44
    if is_enriched:
        delta_trees = data[ENRICHED_DELTA.start + TREES_IDX].ravel().astype(np.float32)
    else:
        delta_trees = data[TREES_IDX].ravel().astype(np.float32)
    
    # Count pixels with significant loss in this window
    alert_mask = delta_trees < -0.10
    severe_mask = delta_trees < -TREE_LOSS_THRESH
    
    # Among GT significant pixels, how many fire here?
    gt_and_alert = significant_loss & alert_mask
    gt_and_severe = significant_loss & severe_mask
    
    # First alert detection
    newly_detected = (first_alert_window < 0) & gt_and_alert & significant_loss
    first_alert_window[newly_detected] = w
    
    # First severe detection
    newly_severe = (first_severe_window < 0) & gt_and_severe & significant_loss
    first_severe_window[newly_severe] = w
    
    # Track peak loss window (among GT pixels)
    bigger_loss = (delta_trees < peak_loss_value) & significant_loss
    peak_loss_window[bigger_loss] = w
    peak_loss_value[bigger_loss] = delta_trees[bigger_loss]
    
    # Cumulative
    cum_tree_loss += delta_trees
    
    n_gt_alert = int(gt_and_alert.sum())
    n_gt_severe = int(gt_and_severe.sum())
    n_newly_det = int(newly_detected.sum())
    
    window_info.append({
        'w': w, 'd_before': d_before, 'd_after': d_after, 'season': season,
        'total_alerts': int(alert_mask.sum()),
        'gt_alerts': n_gt_alert,
        'gt_severe': n_gt_severe,
        'newly_detected': n_newly_det,
    })
    
    print(f"  W{w:02d} ({d_before} → {d_after}) {season:14s}  "
          f"alerts={int(alert_mask.sum()):>7,}  "
          f"GT∩alert={n_gt_alert:>6,}  GT∩severe={n_gt_severe:>5,}  "
          f"new_detect={n_newly_det:>5,}")

# ── Summary: When was change first detected? ─────────────────────────────────
print(f"\n{'='*70}")
print(f"  TIMING OF SIGNIFICANT TREE LOSS (GT delta_trees < -{TREE_LOSS_THRESH})")
print(f"{'='*70}\n")

detected =  first_alert_window[significant_loss] >= 0
n_detected = int(detected.sum())
n_never = n_significant - n_detected
print(f"  GT pixels with loss > {TREE_LOSS_THRESH}:  {n_significant:,}")
print(f"  Detected by ≥1 window:      {n_detected:,} ({100*n_detected/n_significant:.1f}%)")
print(f"  Never detected:              {n_never:,} ({100*n_never/n_significant:.1f}%)")

print(f"\n  Distribution of FIRST detection window:")
print(f"  {'Window':8s} {'Date Range':30s} {'Season':14s} {'First Detected':>15s} {'Cumulative %':>12s}")
print(f"  {'─'*8} {'─'*30} {'─'*14} {'─'*15} {'─'*12}")

cum = 0
for wi in window_info:
    w = wi['w']
    first_det = int((first_alert_window[significant_loss] == w).sum())
    cum += first_det
    cum_pct = 100 * cum / n_significant if n_significant > 0 else 0
    print(f"  W{w:02d}     {wi['d_before']} → {wi['d_after']}     {wi['season']:14s}  "
          f"{first_det:>12,}   {cum_pct:>9.1f}%")

# ── Peak loss timing ─────────────────────────────────────────────────────────
print(f"\n  Distribution of PEAK LOSS window (when tree loss was worst):")
print(f"  {'Window':8s} {'Date Range':30s} {'Season':14s} {'Peak Loss Here':>15s}  {'Mean Loss':>10s}")
print(f"  {'─'*8} {'─'*30} {'─'*14} {'─'*15}  {'─'*10}")

for wi in window_info:
    w = wi['w']
    peak_here = significant_loss & (peak_loss_window == w)
    n_peak = int(peak_here.sum())
    mean_loss = float(peak_loss_value[peak_here].mean()) if n_peak > 0 else 0
    print(f"  W{w:02d}     {wi['d_before']} → {wi['d_after']}     {wi['season']:14s}  "
          f"{n_peak:>12,}   {mean_loss:>9.3f}")

# ── Season summary ────────────────────────────────────────────────────────────
print(f"\n  SEASONAL SUMMARY (first detection):")
season_counts = {}
for w in range(len(window_info)):
    season = window_info[w]['season']
    first_det = int((first_alert_window[significant_loss] == w).sum())
    season_counts[season] = season_counts.get(season, 0) + first_det

for season, count in sorted(season_counts.items(), key=lambda x: -x[1]):
    pct = 100 * count / n_significant if n_significant > 0 else 0
    print(f"    {season:20s}: {count:>6,} ({pct:5.1f}%)")

undetected_pct = 100 * n_never / n_significant if n_significant > 0 else 0
print(f"    {'Never detected':20s}: {n_never:>6,} ({undetected_pct:5.1f}%)")

print(f"\n{'='*70}")
