#!/usr/bin/env python3
"""
Deep phenological analysis of alert classes.

Goal: Quantify how much of Degradation and Shrub/Scrub alerts
are phenological noise vs real change, using:
1. Seasonal distribution (winter-heavy = phenology)
2. Magnitude distribution (small delta = phenology, large = real)
3. NDVI trajectory (NDVI_before high + NDVI_after high = deciduous, just lost leaves)
4. Cross-band correlation (trees↓ + multiple bands shifting = phenology)
"""

# ── PROJ/GDAL env fix ──
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()
del _os, _sys
# ────────────────────────

import os, sys
import numpy as np
from glob import glob
import rasterio

# Enriched raster layout
DELTA_BASE = 16
DW_BEFORE = slice(0, 8)
DW_AFTER = slice(8, 16)
SPEC_BEFORE = slice(26, 29)  # ndvi_before, nbr_before, ndmi_before
SPEC_AFTER = slice(29, 32)   # ndvi_after, nbr_after, ndmi_after

DW_NAMES = ["water", "trees", "grass", "flooded_veg", "crops", "shrub_scrub", "built", "bare"]
TREES_IDX = 1

# Seasons from month
def get_season(fname):
    import re
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})_to_', fname)
    if not m:
        return "Unknown"
    month = int(m.group(2))
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Pre-monsoon"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-monsoon"


def classify_pixel(delta_gains_idx, delta_gains_val):
    if delta_gains_val < 0.05:
        return "Pure Tree Loss"
    names = ["water", "X", "grass", "flooded_veg", "crops", "shrub_scrub", "built", "bare"]
    g = names[delta_gains_idx]
    if g == "crops":
        return "Encroachment"
    elif g == "built":
        return "Built Expansion"
    elif g in ("bare", "grass"):
        return "Degradation"
    elif g == "shrub_scrub":
        return "Shrub/Scrub"
    return "Other"


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/ground_truth/HAMEERPUR"
    threshold = 0.10

    pattern = os.path.join(data_dir, "alerts", "alert_enriched_*.tif")
    files = sorted(glob(pattern))

    # Collect stats per class per season
    season_class_counts = {}
    # Collect delta_trees magnitudes per class
    class_deltas = {c: [] for c in ["Encroachment", "Built Expansion", "Degradation", "Shrub/Scrub", "Pure Tree Loss"]}
    # Collect NDVI stats per class
    class_ndvi = {c: {"before": [], "after": [], "delta": []} for c in class_deltas}
    # trees_before probability per class
    class_trees_before = {c: [] for c in class_deltas}

    for f in files:
        fname = os.path.basename(f)
        season = get_season(fname)

        with rasterio.open(f) as ds:
            data = ds.read().astype(np.float32)

        delta_trees = data[DELTA_BASE + TREES_IDX]
        valid = ~np.isnan(delta_trees)
        alert_mask = valid & (delta_trees <= -threshold)
        n_alerts = int(alert_mask.sum())
        if n_alerts == 0:
            continue

        # Extract relevant bands for alert pixels
        deltas = data[DELTA_BASE:DELTA_BASE+8, alert_mask]  # (8, N)
        dw_before = data[:8, alert_mask]  # (8, N)
        dt = delta_trees[alert_mask]

        # NDVI if available
        has_spec = data.shape[0] >= 32
        if has_spec:
            ndvi_before = data[26, alert_mask]  # NDVI before
            ndvi_after = data[29, alert_mask]   # NDVI after
        else:
            ndvi_before = np.full(n_alerts, np.nan)
            ndvi_after = np.full(n_alerts, np.nan)

        # Classify each pixel
        gains = deltas.copy()
        gains[TREES_IDX] = -999
        dom_idx = np.argmax(gains, axis=0)
        dom_val = np.max(gains, axis=0)

        for px in range(n_alerts):
            cls = classify_pixel(dom_idx[px], dom_val[px])

            key = (season, cls)
            season_class_counts[key] = season_class_counts.get(key, 0) + 1

            if cls in class_deltas:
                class_deltas[cls].append(dt[px])
                class_trees_before[cls].append(dw_before[TREES_IDX, px])
                if has_spec:
                    class_ndvi[cls]["before"].append(ndvi_before[px])
                    class_ndvi[cls]["after"].append(ndvi_after[px])
                    class_ndvi[cls]["delta"].append(ndvi_after[px] - ndvi_before[px])

    # ================================================================
    # REPORT 1: Seasonal Distribution per Class
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  1. SEASONAL DISTRIBUTION (% of each class by season)")
    print(f"{'='*80}\n")

    seasons = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
    classes = ["Encroachment", "Built Expansion", "Degradation", "Shrub/Scrub", "Pure Tree Loss"]

    hdr = f"{'Class':<20}"
    for s in seasons:
        hdr += f" | {s:>14}"
    hdr += f" | {'TOTAL':>10}"
    print(hdr)
    print("-" * len(hdr))

    for cls in classes:
        totals = [season_class_counts.get((s, cls), 0) for s in seasons]
        grand = sum(totals)
        row = f"{cls:<20}"
        for t in totals:
            pct = 100*t/grand if grand > 0 else 0
            row += f" | {t:>8,} {pct:>4.0f}%"
        row += f" | {grand:>10,}"
        print(row)

    # ================================================================
    # REPORT 2: Delta Trees Magnitude Distribution
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  2. DELTA_TREES MAGNITUDE (how severe is the tree loss?)")
    print(f"{'='*80}\n")
    print(f"  Phenology = small changes (-0.10 to -0.20)")
    print(f"  Real loss = large changes (< -0.30)\n")

    hdr = f"{'Class':<20} | {'Mean':>8} | {'Median':>8} | {'P10':>8} | {'P90':>8} | {'<-0.30':>8} | {'<-0.50':>8}"
    print(hdr)
    print("-" * len(hdr))

    for cls in classes:
        vals = np.array(class_deltas[cls])
        if len(vals) == 0:
            continue
        mean = np.mean(vals)
        med = np.median(vals)
        p10 = np.percentile(vals, 10)
        p90 = np.percentile(vals, 90)
        severe = 100 * (vals < -0.30).sum() / len(vals)
        very_severe = 100 * (vals < -0.50).sum() / len(vals)
        print(f"{cls:<20} | {mean:>+8.3f} | {med:>+8.3f} | {p10:>+8.3f} | {p90:>+8.3f} | {severe:>7.1f}% | {very_severe:>7.1f}%")

    # ================================================================
    # REPORT 3: NDVI Analysis (phenology signature)
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  3. NDVI ANALYSIS (phenology = NDVI drops but stays moderate)")
    print(f"{'='*80}\n")
    print(f"  Real deforestation: NDVI_before HIGH (>0.4), NDVI_after LOW (<0.2)")
    print(f"  Phenology (leaf drop): NDVI_before HIGH, NDVI_after MODERATE (0.2-0.5)")
    print(f"  Phenology tells: delta_NDVI is small but DW reclassifies\n")

    hdr = f"{'Class':<20} | {'NDVI_bef':>9} | {'NDVI_aft':>9} | {'dNDVI':>9} | {'Bef>0.4':>8} | {'Aft<0.2':>8} | {'Aft>0.3':>8}"
    print(hdr)
    print("-" * len(hdr))

    for cls in classes:
        bef = np.array(class_ndvi[cls]["before"])
        aft = np.array(class_ndvi[cls]["after"])
        d = np.array(class_ndvi[cls]["delta"])
        if len(bef) == 0:
            continue
        # Remove NaN
        mask = ~(np.isnan(bef) | np.isnan(aft))
        bef, aft, d = bef[mask], aft[mask], d[mask]
        if len(bef) == 0:
            continue

        pct_bef_high = 100 * (bef > 0.4).sum() / len(bef)
        pct_aft_low = 100 * (aft < 0.2).sum() / len(aft)
        pct_aft_mod = 100 * (aft > 0.3).sum() / len(aft)

        print(f"{cls:<20} | {np.nanmean(bef):>+9.3f} | {np.nanmean(aft):>+9.3f} | {np.nanmean(d):>+9.3f} | {pct_bef_high:>7.1f}% | {pct_aft_low:>7.1f}% | {pct_aft_mod:>7.1f}%")

    # ================================================================
    # REPORT 4: Trees-Before Probability
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  4. TREES_BEFORE PROBABILITY (were these really trees?)")
    print(f"{'='*80}\n")
    print(f"  High trees_before (>0.3) = likely real forest")
    print(f"  Low trees_before (<0.15) = marginal, could be misclassified\n")

    hdr = f"{'Class':<20} | {'Mean':>8} | {'Median':>8} | {'>0.30':>8} | {'>0.50':>8} | {'<0.15':>8}"
    print(hdr)
    print("-" * len(hdr))

    for cls in classes:
        vals = np.array(class_trees_before[cls])
        if len(vals) == 0:
            continue
        mask = ~np.isnan(vals)
        vals = vals[mask]
        mean = np.mean(vals)
        med = np.median(vals)
        high = 100 * (vals > 0.30).sum() / len(vals)
        very_high = 100 * (vals > 0.50).sum() / len(vals)
        low = 100 * (vals < 0.15).sum() / len(vals)
        print(f"{cls:<20} | {mean:>8.3f} | {med:>8.3f} | {high:>7.1f}% | {very_high:>7.1f}% | {low:>7.1f}%")

    # ================================================================
    # VERDICT
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  5. VERDICT: Should we suppress Degradation + Shrub/Scrub?")
    print(f"{'='*80}\n")

    # Count winter vs non-winter for these classes
    for cls in ["Degradation", "Shrub/Scrub"]:
        winter = season_class_counts.get(("Winter", cls), 0)
        total = sum(season_class_counts.get((s, cls), 0) for s in seasons)
        non_winter = total - winter
        print(f"  {cls}:")
        print(f"    Winter:     {winter:>8,} ({100*winter/total:.0f}%)")
        print(f"    Non-winter: {non_winter:>8,} ({100*non_winter/total:.0f}%)")
        print()


if __name__ == "__main__":
    main()
