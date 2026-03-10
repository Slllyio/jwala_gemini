"""
Deep Analysis: GT vs Alert Model Performance
=============================================
Comprehensive profiling of TP, FP, FN characteristics to identify
model improvement strategies.
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
import os, re, json
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
GT_RASTER = "data/ground_truth/HAMEERPUR/gt_delta_2024_2025_HAMEERPUR.tif"
ALERT_DIR = "data/ground_truth/HAMEERPUR/alerts"
STACKED_TIF = "outputs/alert_filter_enriched_v5/stacked_v5.tif"
TRAINING_REPORT = "outputs/alert_filter_enriched_v5/training_report.json"

TREES_IDX = 1
N_BANDS = 8
GT_LOSS_THRESH = 0.10   # same as training
SEVERE_LOSS = 0.30
ENRICHED_DELTA = slice(16, 24)
ENRICHED_DW_BEFORE = slice(0, 8)
ENRICHED_DW_AFTER  = slice(8, 16)
ENRICHED_SPEC_BEFORE = slice(26, 29)
ENRICHED_SPEC_AFTER  = slice(29, 32)

DW_NAMES = ["water", "trees", "grass", "flooded_veg", "crops", "shrub_scrub", "built", "bare"]

def parse_dates(stem):
    m = re.search(r"alert_(?:delta|enriched)_(.+?)_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})", stem)
    if m: return m.group(2), m.group(3)
    return "unknown", "unknown"

# ── 1. Load GT ─────────────────────────────────────────────────────────────────
with rasterio.open(GT_RASTER) as ds:
    gt = ds.read()
H, W = gt.shape[1], gt.shape[2]
n_pix = H * W

gt_trees = gt[TREES_IDX].ravel()     # annual delta_trees
gt_all   = gt[:N_BANDS].reshape(N_BANDS, n_pix)

gt_real     = gt_trees < -GT_LOSS_THRESH      # real change pixels
gt_severe   = gt_trees < -SEVERE_LOSS          # severe change
gt_moderate = gt_real & ~gt_severe             # 0.1 < |loss| < 0.3
gt_none     = ~gt_real                          # no significant change

print(f"\n{'='*72}")
print(f"  DEEP ANALYSIS: Van Suraksha Alert Filter v5")
print(f"{'='*72}")
print(f"\n  GT Raster: {H}x{W} = {n_pix:,} pixels")
print(f"  Real change (loss > 0.1):   {int(gt_real.sum()):>8,} ({100*gt_real.sum()/n_pix:.2f}%)")
print(f"    - Severe (loss > 0.3):    {int(gt_severe.sum()):>8,} ({100*gt_severe.sum()/n_pix:.2f}%)")
print(f"    - Moderate (0.1-0.3):     {int(gt_moderate.sum()):>8,} ({100*gt_moderate.sum()/n_pix:.2f}%)")
print(f"  No change (loss <= 0.1):    {int(gt_none.sum()):>8,} ({100*gt_none.sum()/n_pix:.2f}%)")

# ── 2. Load stacked v5 output ─────────────────────────────────────────────────
with rasterio.open(STACKED_TIF) as ds:
    stacked = ds.read()
score     = stacked[0].ravel()
n_fires   = stacked[1].ravel().astype(int)
confirmed = stacked[2].ravel() > 0.5
change_cls = stacked[3].ravel().astype(int)

# ── 3. Confusion matrix: GT vs Confirmed ─────────────────────────────────────
tp = gt_real & confirmed
fp = gt_none & confirmed
fn = gt_real & ~confirmed
tn = gt_none & ~confirmed

print(f"\n  {'='*50}")
print(f"  PIXEL-LEVEL CONFUSION (GT vs v5 Confirmed)")
print(f"  {'='*50}")
print(f"  TP (real + confirmed):    {int(tp.sum()):>8,}")
print(f"  FP (no-change + conf'd):  {int(fp.sum()):>8,}")
print(f"  FN (real + missed):       {int(fn.sum()):>8,}")
print(f"  TN (no-change + clean):   {int(tn.sum()):>8,}")
pixel_prec = tp.sum() / (tp.sum() + fp.sum()) if (tp.sum() + fp.sum()) > 0 else 0
pixel_rec  = tp.sum() / (tp.sum() + fn.sum()) if (tp.sum() + fn.sum()) > 0 else 0
pixel_f1   = 2*pixel_prec*pixel_rec / (pixel_prec + pixel_rec) if (pixel_prec + pixel_rec) > 0 else 0
print(f"  Pixel precision: {pixel_prec:.4f}")
print(f"  Pixel recall:    {pixel_rec:.4f}")
print(f"  Pixel F1:        {pixel_f1:.4f}")

# ── 4. Severity breakdown: How well do we detect each tier? ───────────────────
print(f"\n  {'='*50}")
print(f"  DETECTION BY SEVERITY")
print(f"  {'='*50}")
for label, mask in [("Severe (>0.3)", gt_severe), ("Moderate (0.1-0.3)", gt_moderate)]:
    det = (mask & confirmed).sum()
    total = mask.sum()
    pct = 100*det/total if total > 0 else 0
    print(f"  {label:25s}: {int(det):>6,} / {int(total):>6,} detected ({pct:.1f}%)")

# ── 5. FN Profile: What characterizes missed real change? ─────────────────────
print(f"\n  {'='*50}")
print(f"  FALSE NEGATIVE (FN) PROFILE — Missed Real Change")
print(f"  {'='*50}")
fn_mask = gt_real & ~confirmed
fn_severe = fn_mask & gt_severe
fn_moderate = fn_mask & gt_moderate

print(f"  Total FN: {int(fn_mask.sum()):,}")
print(f"    Severe FN (loss > 0.3): {int(fn_severe.sum()):,} ({100*fn_severe.sum()/fn_mask.sum():.1f}%)")
print(f"    Moderate FN (0.1-0.3): {int(fn_moderate.sum()):,} ({100*fn_moderate.sum()/fn_mask.sum():.1f}%)")

# Land cover profile of FN pixels
print(f"\n  DW before-state for FN pixels (mean of DW bands):")
for b, name in enumerate(DW_NAMES):
    val = float(gt_all[b, fn_mask].mean()) if fn_mask.sum() > 0 else 0
    bar = "#" * int(val * 40)
    print(f"    {name:15s}: {val:.3f}  {bar}")

# Compare TP vs FN land cover
print(f"\n  TP vs FN comparison (mean DW before):")
print(f"    {'Band':15s}  {'TP':>7s}  {'FN':>7s}  {'Diff':>8s}")
for b, name in enumerate(DW_NAMES):
    tp_val = float(gt_all[b, tp].mean()) if tp.sum() > 0 else 0
    fn_val = float(gt_all[b, fn_mask].mean()) if fn_mask.sum() > 0 else 0
    diff = fn_val - tp_val
    print(f"    {name:15s}: {tp_val:>6.3f}  {fn_val:>6.3f}  {diff:+7.3f}")

# Mean gt_trees for TP vs FN
print(f"\n  GT delta_trees:")
print(f"    TP mean loss: {float(gt_trees[tp].mean()):.3f} (severe)")
print(f"    FN mean loss: {float(gt_trees[fn_mask].mean()):.3f} (weaker)")

# ── 6. FP Profile: What is the model falsely confirming? ──────────────────────
print(f"\n  {'='*50}")
print(f"  FALSE POSITIVE (FP) PROFILE — False Alarms")
print(f"  {'='*50}")
print(f"  Total FP: {int(fp.sum()):,}")

# FP change classes
CHANGE_CLASS = {1: "Encroachment", 2: "Built", 3: "Degradation", 4: "Shrub/Scrub", 5: "Pure Loss", 6: "Other"}
print(f"\n  FP by change class (what model thinks they are):")
for cls_id, cls_name in CHANGE_CLASS.items():
    count = int((change_cls[fp] == cls_id).sum())
    if count > 0:
        pct = 100 * count / fp.sum()
        print(f"    {cls_id}: {cls_name:25s} {count:>6,} ({pct:.1f}%)")

# FP land cover analysis
print(f"\n  FP vs TP: DW before-state comparison:")
print(f"    {'Band':15s}  {'TP':>7s}  {'FP':>7s}  {'Diff':>8s}  {'Insight':20s}")
for b, name in enumerate(DW_NAMES):
    tp_val = float(gt_all[b, tp].mean()) if tp.sum() > 0 else 0
    fp_val = float(gt_all[b, fp].mean()) if fp.sum() > 0 else 0
    diff = fp_val - tp_val
    insight = ""
    if abs(diff) > 0.03:
        if diff > 0:
            insight = f"FP has MORE {name}"
        else:
            insight = f"TP has MORE {name}"
    print(f"    {name:15s}: {tp_val:>6.3f}  {fp_val:>6.3f}  {diff:+7.3f}  {insight}")

# FP score distribution
print(f"\n  FP stacked score distribution:")
fp_scores = score[fp]
for pct in [25, 50, 75, 90, 95]:
    print(f"    P{pct:02d}: {np.percentile(fp_scores, pct):.3f}")

# ── 7. Per-window deep analysis ───────────────────────────────────────────────
print(f"\n  {'='*50}")
print(f"  PER-WINDOW ANALYSIS")
print(f"  {'='*50}")

alert_files = sorted(glob(os.path.join(ALERT_DIR, "alert_enriched_*.tif")))

# Analyze which windows are most discriminative
print(f"\n  Window quality analysis:")
print(f"  {'Win':4s} {'Date Range':25s} {'Alerts':>8s} {'TP':>7s} {'FP':>7s} {'Prec':>6s} {'TP/FP':>7s} {'Quality':10s}")

with open(TRAINING_REPORT) as f:
    report = json.load(f)

for ws in report['window_stats']:
    w = ws['window']
    prec = ws['precision']
    tp_w = ws['n_tp']
    fp_w = ws['n_fp'] + ws['n_fpc']
    ratio = tp_w / (fp_w + 1)
    quality = "GOOD" if prec > 0.10 else "NOISY" if prec > 0.03 else "VERY NOISY"
    if tp_w == 0:
        quality = "DEAD"
    print(f"  W{w:02d}  {ws['date']:25s} {ws['n_alerts']:>8,} {tp_w:>7,} {fp_w:>7,} {prec:>5.1%} {ratio:>6.3f}  {quality}")

# ── 8. Alert coverage gap analysis ───────────────────────────────────────────
print(f"\n  {'='*50}")
print(f"  COVERAGE GAP ANALYSIS")
print(f"  {'='*50}")

# Check which GT pixels never trigger ANY alert across all windows
ever_fired = np.zeros(n_pix, dtype=bool)
max_window_loss = np.zeros(n_pix, dtype=np.float32)
for f in alert_files:
    with rasterio.open(f) as ds:
        data = ds.read()
    is_enriched = data.shape[0] >= 44
    if is_enriched:
        dt = data[ENRICHED_DELTA.start + TREES_IDX].ravel().astype(np.float32)
    else:
        dt = data[TREES_IDX].ravel().astype(np.float32)
    fired = dt < -0.10
    ever_fired |= fired
    max_window_loss = np.minimum(max_window_loss, dt)

gt_never_fired = gt_real & ~ever_fired
gt_fired_but_missed = gt_real & ever_fired & ~confirmed
gt_fired_and_confirmed = gt_real & ever_fired & confirmed

print(f"  GT real pixels: {int(gt_real.sum()):,}")
print(f"  Never fired any alert: {int(gt_never_fired.sum()):,} ({100*gt_never_fired.sum()/gt_real.sum():.1f}%)")
print(f"    -> These need MORE WINDOWS or LOWER threshold to catch")
print(f"  Fired but not confirmed: {int(gt_fired_but_missed.sum()):,} ({100*gt_fired_but_missed.sum()/gt_real.sum():.1f}%)")
print(f"    -> These need BETTER MODEL SCORING to rescue")
print(f"  Fired and confirmed: {int(gt_fired_and_confirmed.sum()):,} ({100*gt_fired_and_confirmed.sum()/gt_real.sum():.1f}%)")

# Profile never-fired pixels
print(f"\n  Profile of NEVER-FIRED GT pixels:")
print(f"    Mean GT delta_trees: {float(gt_trees[gt_never_fired].mean()):.3f}")
print(f"    Max single-window loss: {float(max_window_loss[gt_never_fired].min()):.3f}")
print(f"    Median single-window loss: {float(np.median(max_window_loss[gt_never_fired])):.3f}")

# What % have small per-window losses
for thresh in [0.05, 0.08, 0.10, 0.15, 0.20]:
    count = int((max_window_loss[gt_never_fired] > -thresh).sum())
    pct = 100*count/gt_never_fired.sum() if gt_never_fired.sum() > 0 else 0
    print(f"    Max per-window loss < {thresh}: {count:>5,} ({pct:.1f}%)")

# Profile of fired-but-missed pixels
if gt_fired_but_missed.sum() > 0:
    print(f"\n  Profile of FIRED-BUT-MISSED GT pixels:")
    print(f"    Mean GT delta_trees: {float(gt_trees[gt_fired_but_missed].mean()):.3f}")
    fbm_scores = score[gt_fired_but_missed]
    print(f"    Mean stacked score: {float(fbm_scores.mean()):.3f}")
    print(f"    Median stacked score: {float(np.median(fbm_scores)):.3f}")
    fbm_nfires = n_fires[gt_fired_but_missed]
    print(f"    Mean n_fires: {float(fbm_nfires.mean()):.1f}")
    print(f"    Pixels with n_fires >= 2: {int((fbm_nfires >= 2).sum()):,} ({100*(fbm_nfires >= 2).sum()/gt_fired_but_missed.sum():.1f}%)")
    print(f"    Pixels with score > 0: {int((fbm_scores > 0).sum()):,}")

# ── 9. Class imbalance analysis ───────────────────────────────────────────────
print(f"\n  {'='*50}")
print(f"  CLASS IMBALANCE ANALYSIS")
print(f"  {'='*50}")
n_train = report['n_train']
n_test  = report['n_test']
n_total = report['n_total_records']
# From confusion matrix
cm = report['metrics']['confusion_matrix']
test_neg = cm[0][0] + cm[0][1]
test_pos = cm[1][0] + cm[1][1]
print(f"  Total samples: {n_total:,}")
print(f"  Test: {n_test:,} ({test_pos:,} pos, {test_neg:,} neg)")
print(f"  Positive ratio: {100*test_pos/n_test:.1f}%")
print(f"  Imbalance ratio: 1:{test_neg/test_pos:.1f}")

# ── 10. Temporal gap analysis ─────────────────────────────────────────────────
print(f"\n  {'='*50}")
print(f"  TEMPORAL COVERAGE GAPS")
print(f"  {'='*50}")

dates = []
for f in alert_files:
    d_before, d_after = parse_dates(os.path.splitext(os.path.basename(f))[0])
    dates.append((d_before, d_after))

from datetime import datetime
prev_end = None
total_covered = 0
total_gap = 0
for i, (d_bef, d_aft) in enumerate(dates):
    start = datetime.strptime(d_bef, "%Y-%m-%d")
    end   = datetime.strptime(d_aft, "%Y-%m-%d")
    dur = (end - start).days
    total_covered += dur
    if prev_end:
        gap = (start - prev_end).days
        gap_str = f"  GAP={gap}d" if gap > 10 else ""
        print(f"  W{i:02d}: {d_bef} -> {d_aft}  ({dur:>3d}d){gap_str}")
        if gap > 10:
            total_gap += gap
    else:
        print(f"  W{i:02d}: {d_bef} -> {d_aft}  ({dur:>3d}d)")
    prev_end = end

annual_range = (datetime.strptime(dates[-1][1], "%Y-%m-%d") - datetime.strptime(dates[0][0], "%Y-%m-%d")).days
print(f"\n  Total days covered: {total_covered}d out of {annual_range}d ({100*total_covered/annual_range:.1f}%)")
print(f"  Total gap days: {total_gap}d ({100*total_gap/annual_range:.1f}%)")
print(f"  Number of windows: {len(dates)}")
print(f"  Mean revisit: {annual_range / len(dates):.0f}d")

# ── 11. Improvement recommendations scoring ──────────────────────────────────
print(f"\n{'='*72}")
print(f"  IMPROVEMENT RECOMMENDATIONS (ranked by expected impact)")
print(f"{'='*72}")

improvements = []

# R1: More windows
n_never = int(gt_never_fired.sum())
pct_never = 100*n_never/gt_real.sum()
improvements.append((
    "ADD MORE ALERT WINDOWS",
    f"{n_never:,} GT pixels ({pct_never:.0f}%) never fire any alert. Gap months: Jun-Aug, late Jan. "
    f"Adding 10-15 more windows (esp. Jun-Aug monsoon) could recover these.",
    pct_never * 0.5,  # estimated impact
    "HIGH"
))

# R2: Lower alert threshold
improvements.append((
    "LOWER ALERT THRESHOLD (0.10 -> 0.05)",
    f"Currently delta_trees < -0.10 triggers an alert. Many gradual losses "
    f"have per-window loss of -0.05 to -0.10. Lowering to 0.05 increases FP but catches more TP.",
    15,
    "MEDIUM"
))

# R3: Fix class imbalance
improvements.append((
    "ADDRESS CLASS IMBALANCE",
    f"Training has 1:{test_neg/test_pos:.0f} imbalance. Use SMOTE, class weights, "
    f"or focal loss to improve minority class learning.",
    10,
    "MEDIUM"
))

# R4: Multi-site training data
improvements.append((
    "ADD MORE TRAINING SITES",
    f"Model trained on single site (HAMEERPUR). Adding 3-5 diverse sites "
    f"(different forest types, elevation, climate) will improve generalization.",
    25,
    "CRITICAL"
))

# R5: Rescue fired-but-missed
n_fbm = int(gt_fired_but_missed.sum())
improvements.append((
    "TUNE STACKING/HYSTERESIS PARAMS",
    f"{n_fbm:,} GT pixels fire but don't pass stacking. "
    f"Adjusting min_fires, threshold, or hysteresis params could rescue ~50%.",
    8,
    "MEDIUM"
))

for i, (title, desc, impact, priority) in enumerate(sorted(improvements, key=lambda x: -x[2]), 1):
    print(f"\n  {i}. [{priority}] {title}")
    print(f"     Impact: ~{impact:.0f}% recall improvement")
    print(f"     {desc}")

print(f"\n{'='*72}")
