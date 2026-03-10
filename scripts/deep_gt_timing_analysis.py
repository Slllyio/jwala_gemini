#!/usr/bin/env python3
"""Deep GT Timing Analysis — Fixed band layout for enriched TIFs."""
import numpy as np, rasterio, glob, os, sys, logging

log = logging.getLogger("deep_gt_timing")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)

GT_DIR  = "data/ground_truth/HAMEERPUR"
ALERT_DIR = os.path.join(GT_DIR, "alerts")
GT_TIF  = os.path.join(GT_DIR, "gt_delta_2024_2025_HAMEERPUR.tif")

# Enriched TIF band layout: 0-7 DW_before, 8-15 DW_after, 16-23 DW_delta, 24-25 cloud, 26+ S2
DW_DELTA_START = 16
DW_BEFORE_START = 0
DW_AFTER_START  = 8
TREES_IDX = 1
N_DW = 8
ALERT_THRESH = 0.10

DW_NAMES = {0:"water",1:"trees",2:"grass",3:"flooded_veg",4:"crops",5:"shrub",6:"built",7:"bare"}


def main():
    with rasterio.open(GT_TIF) as ds:
        gt = ds.read()
    gt_trees = gt[TREES_IDX].ravel()
    H, W = gt[0].shape
    N = H * W
    gt_severe = gt_trees < -0.30
    gt_real = gt_trees < -0.10
    log.info(f"Grid: {H}x{W} = {N:,}")
    log.info(f"GT severe: {gt_severe.sum():,}  GT real: {gt_real.sum():,}")

    tifs = sorted(glob.glob(os.path.join(ALERT_DIR, "alert_enriched_HAMEERPUR_*.tif")))
    n_windows = len(tifs)
    log.info(f"Found {n_windows} windows")

    fires_per_pixel = np.zeros(N, dtype=np.int32)
    max_window_loss = np.zeros(N, dtype=np.float32)
    sum_window_loss = np.zeros(N, dtype=np.float32)
    first_fire_win  = np.full(N, -1, dtype=np.int32)
    per_window_delta = np.zeros((n_windows, N), dtype=np.float32)
    win_dates = []

    for wi, tif in enumerate(tifs):
        with rasterio.open(tif) as ds:
            data = ds.read()
            tags = ds.tags()
        w_start = tags.get("window_start", "?")
        w_end   = tags.get("window_end", "?")
        win_dates.append((w_start, w_end))

        delta_trees = data[DW_DELTA_START + TREES_IDX].ravel().astype(np.float32)
        per_window_delta[wi] = delta_trees

        is_alert = delta_trees <= -ALERT_THRESH
        fires_per_pixel += is_alert.astype(np.int32)
        loss = np.abs(np.minimum(delta_trees, 0))
        better = loss > max_window_loss
        max_window_loss[better] = loss[better]
        sum_window_loss += loss

        newly_fired = is_alert & (first_fire_win < 0)
        first_fire_win[newly_fired] = wi

    # ── Severe GT: Detected vs Undetected ────────────────────────────
    sev_idx = np.where(gt_severe)[0]
    det_mask = fires_per_pixel[sev_idx] > 0
    det_idx  = sev_idx[det_mask]
    undet_idx = sev_idx[~det_mask]
    n_det = len(det_idx)
    n_undet = len(undet_idx)

    print("\n" + "="*70)
    print("  1. DETECTED vs UNDETECTED SEVERE GT PIXELS")
    print("="*70)
    print(f"  Total severe:  {len(sev_idx):,}")
    print(f"  Detected:      {n_det:,} ({100*n_det/len(sev_idx):.1f}%)")
    print(f"  Undetected:    {n_undet:,} ({100*n_undet/len(sev_idx):.1f}%)")

    print(f"\n  Annual GT loss magnitude:")
    print(f"    Detected:   mean={np.abs(gt_trees[det_idx]).mean():.3f}  median={np.median(np.abs(gt_trees[det_idx])):.3f}")
    print(f"    Undetected: mean={np.abs(gt_trees[undet_idx]).mean():.3f}  median={np.median(np.abs(gt_trees[undet_idx])):.3f}")

    print(f"\n  Max single-window loss:")
    print(f"    Detected:   mean={max_window_loss[det_idx].mean():.3f}  median={np.median(max_window_loss[det_idx]):.3f}")
    print(f"    Undetected: mean={max_window_loss[undet_idx].mean():.3f}  median={np.median(max_window_loss[undet_idx]):.3f}")

    print(f"\n  N fires (windows that triggered):")
    print(f"    Detected:   mean={fires_per_pixel[det_idx].mean():.1f}  median={np.median(fires_per_pixel[det_idx]):.0f}")
    # undetected is 0 by definition
    
    # ── 2. WHY are undetected pixels undetected? ─────────────────────
    print("\n" + "="*70)
    print("  2. WHY UNDETECTED? (per-window delta profile)")
    print("="*70)

    undet_deltas = per_window_delta[:, undet_idx]   # (n_windows, n_undet)
    undet_best = undet_deltas.min(axis=0)           # most negative

    print(f"\n  Best (most negative) single-window delta for undetected:")
    print(f"    Mean:   {undet_best.mean():.4f}")
    print(f"    Median: {np.median(undet_best):.4f}")
    print(f"    P10:    {np.percentile(undet_best, 10):.4f}")
    print(f"    Min:    {undet_best.min():.4f}")

    close_05 = ((undet_best < -0.05) & (undet_best >= -0.10)).sum()
    close_08 = ((undet_best < -0.08) & (undet_best >= -0.10)).sum()
    zero     = (undet_best >= -0.01).sum()
    print(f"\n  Proximity to -0.10 threshold:")
    print(f"    [-0.10, -0.05): {close_05:,} ({100*close_05/n_undet:.1f}%) — would fire at threshold=0.05")
    print(f"    [-0.10, -0.08): {close_08:,} ({100*close_08/n_undet:.1f}%) — would fire at threshold=0.08")
    print(f"    > -0.01 (zero): {zero:,} ({100*zero/n_undet:.1f}%) — no measurable per-window loss")

    # Windows with any loss
    n_neg_windows = (undet_deltas < -0.01).sum(axis=0)
    print(f"\n  Windows with measurable loss (delta < -0.01) for undetected:")
    print(f"    Mean:   {n_neg_windows.mean():.1f}  Median: {np.median(n_neg_windows):.0f}")
    for nw in range(n_windows + 1):
        c = (n_neg_windows == nw).sum()
        if c > 0:
            print(f"      {nw:2d} windows: {c:>5,} ({100*c/n_undet:.1f}%)")

    # ── 3. PER-WINDOW PROFILE ────────────────────────────────────────
    print("\n" + "="*70)
    print("  3. PER-WINDOW PROFILE")
    print("="*70)

    print(f"\n  DETECTED severe pixels — per-window contribution:")
    for wi in range(n_windows):
        w_d = per_window_delta[wi, det_idx]
        n_fire = int((w_d <= -ALERT_THRESH).sum())
        ml = np.abs(np.minimum(w_d, 0)).mean()
        print(f"    W{wi:02d} ({win_dates[wi][0]:>10s}->{win_dates[wi][1]:>10s}): "
              f"fires={n_fire:>5,}/{n_det:,}  mean_loss={ml:.4f}")

    print(f"\n  UNDETECTED severe pixels — per-window delta profile:")
    for wi in range(n_windows):
        w_d = per_window_delta[wi, undet_idx]
        near = int(((w_d < -0.05) & (w_d >= -0.10)).sum())
        ml = np.abs(np.minimum(w_d, 0)).mean()
        mn = w_d.min()
        print(f"    W{wi:02d} ({win_dates[wi][0]:>10s}->{win_dates[wi][1]:>10s}): "
              f"mean_loss={ml:.4f}  min_delta={mn:+.4f}  near_thresh={near:>4,}")

    # ── 4. LAND COVER COMPOSITION ────────────────────────────────────
    print("\n" + "="*70)
    print("  4. LAND COVER (DW before → after)")
    print("="*70)

    # Use the last loaded TIF (W14) for DW
    with rasterio.open(tifs[-1]) as ds:
        ref_data = ds.read()

    dw_b = ref_data[DW_BEFORE_START:DW_BEFORE_START+N_DW].reshape(N_DW, -1)
    dw_a = ref_data[DW_AFTER_START:DW_AFTER_START+N_DW].reshape(N_DW, -1)

    print(f"\n  DW BEFORE:")
    print(f"  {'Class':15s} | {'Detected':>10s} | {'Undetected':>10s}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}")
    for bi in range(N_DW):
        d = dw_b[bi, det_idx].mean()
        u = dw_b[bi, undet_idx].mean()
        if d > 0.01 or u > 0.01:
            print(f"  {DW_NAMES[bi]:15s} | {d:10.3f} | {u:10.3f}")

    print(f"\n  DW AFTER:")
    print(f"  {'Class':15s} | {'Detected':>10s} | {'Undetected':>10s}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}")
    for bi in range(N_DW):
        d = dw_a[bi, det_idx].mean()
        u = dw_a[bi, undet_idx].mean()
        if d > 0.01 or u > 0.01:
            print(f"  {DW_NAMES[bi]:15s} | {d:10.3f} | {u:10.3f}")

    # Transition patterns
    print(f"\n  Dominant transition (before -> after):")
    for label, idx_set in [("Detected", det_idx), ("Undetected", undet_idx)]:
        dom_b = dw_b[:, idx_set].argmax(axis=0)
        dom_a = dw_a[:, idx_set].argmax(axis=0)
        trans = {}
        for i in range(len(idx_set)):
            key = f"{DW_NAMES[dom_b[i]]} -> {DW_NAMES[dom_a[i]]}"
            trans[key] = trans.get(key, 0) + 1
        print(f"\n    {label}:")
        for t, c in sorted(trans.items(), key=lambda x: -x[1])[:8]:
            print(f"      {t:30s}: {c:>5,} ({100*c/len(idx_set):.1f}%)")

    # ── 5. THRESHOLD SENSITIVITY ─────────────────────────────────────
    print("\n" + "="*70)
    print("  5. THRESHOLD SENSITIVITY")
    print("="*70)
    for th in [0.10, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01]:
        f = np.zeros(N, dtype=np.int32)
        for wi in range(n_windows):
            f += (per_window_delta[wi] <= -th).astype(np.int32)
        sev_d = (f[sev_idx] > 0).sum()
        real_d = (f[gt_real] > 0).sum()
        total = (f > 0).sum()
        print(f"  T={th:.2f}: severe={sev_d:>5,}/{len(sev_idx):,} ({100*sev_d/len(sev_idx):>5.1f}%)  "
              f"real={real_d:>6,}/{gt_real.sum():,} ({100*real_d/gt_real.sum():>5.1f}%)  "
              f"total_firing={total:>8,}")

    # ── 6. CUMULATIVE LOSS FOR UNDETECTED ────────────────────────────
    print("\n" + "="*70)
    print("  6. HOW LOSS ACCUMULATES FOR UNDETECTED PIXELS")
    print("="*70)
    cum = np.zeros(n_undet, dtype=np.float32)
    for wi in range(n_windows):
        w_d = per_window_delta[wi, undet_idx]
        cum += np.abs(np.minimum(w_d, 0))
        print(f"    After W{wi:02d} ({win_dates[wi][0]:>10s}): "
              f"mean_cum={cum.mean():.4f}  P50={np.median(cum):.4f}  P90={np.percentile(cum, 90):.4f}")

    gt_annual = np.abs(gt_trees[undet_idx])
    coverage = cum / np.maximum(gt_annual, 0.001)
    invisible = (coverage < 0.05).sum()
    partial   = ((coverage >= 0.05) & (coverage < 0.30)).sum()
    sig       = (coverage >= 0.30).sum()
    print(f"\n  Fraction of annual loss captured by windows:")
    print(f"    Mean: {coverage.mean():.3f}  Median: {np.median(coverage):.3f}")
    print(f"    Invisible (<5%):    {invisible:>5,} ({100*invisible/n_undet:.1f}%) — loss entirely outside windows")
    print(f"    Partial (5-30%):    {partial:>5,} ({100*partial/n_undet:.1f}%)")
    print(f"    Significant (>30%): {sig:>5,} ({100*sig/n_undet:.1f}%) — captured but too gradual per-window")

    # ── 7. SPATIAL DISTRIBUTION ──────────────────────────────────────
    print("\n" + "="*70)
    print("  7. SPATIAL DISTRIBUTION")
    print("="*70)
    det_r, det_c = det_idx // W, det_idx % W
    und_r, und_c = undet_idx // W, undet_idx % W
    mid_r, mid_c = H // 2, W // 2
    quads = {"NW": (0, mid_r, 0, mid_c), "NE": (0, mid_r, mid_c, W),
             "SW": (mid_r, H, 0, mid_c), "SE": (mid_r, H, mid_c, W)}
    print(f"  {'Quad':5s} | {'Detected':>10s} | {'Undetected':>10s} | {'Rate':>8s}")
    for q, (r0, r1, c0, c1) in quads.items():
        d = ((det_r >= r0) & (det_r < r1) & (det_c >= c0) & (det_c < c1)).sum()
        u = ((und_r >= r0) & (und_r < r1) & (und_c >= c0) & (und_c < c1)).sum()
        rate = d / max(d + u, 1)
        print(f"  {q:5s} | {int(d):>10,} | {int(u):>10,} | {rate:>8.1%}")

    print("\n" + "="*70)
    print("  DONE")
    print("="*70)

if __name__ == "__main__":
    main()
