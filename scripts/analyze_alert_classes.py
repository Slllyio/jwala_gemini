#!/usr/bin/env python3
"""Per-window alert class breakdown at T=0.10, 0.15, 0.20."""

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

DELTA_BASE = 16
DW_ORDER = ["water", "trees", "grass", "flooded_veg", "crops", "shrub_scrub", "built", "bare"]

CLASS_ORDER = ["Encroachment", "Built Expansion", "Degradation", "Shrub/Scrub", "Pure Tree Loss", "Other"]

def classify_alert_pixels(data, threshold):
    delta_trees = data[DELTA_BASE + 1]  # trees at idx 1
    valid = ~np.isnan(delta_trees)
    alert_mask = valid & (delta_trees <= -threshold)
    n_alerts = int(alert_mask.sum())
    if n_alerts == 0:
        return n_alerts, {}

    deltas = data[DELTA_BASE:DELTA_BASE+8]
    gains = deltas[:, alert_mask].copy()
    gains[1] = -999  # exclude trees
    dom_idx = np.argmax(gains, axis=0)
    dom_val = np.max(gains, axis=0)

    classes = np.full(n_alerts, "Other", dtype="U20")
    classes[dom_idx == 4] = "Encroachment"
    classes[dom_idx == 6] = "Built Expansion"
    classes[(dom_idx == 7) | (dom_idx == 2)] = "Degradation"
    classes[dom_idx == 5] = "Shrub/Scrub"
    classes[dom_val < 0.05] = "Pure Tree Loss"

    counts = {}
    for cls in np.unique(classes):
        counts[cls] = int((classes == cls).sum())
    return n_alerts, counts


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/ground_truth/HAMEERPUR"
    thresholds = [0.10, 0.15, 0.20]
    
    pattern = os.path.join(data_dir, "alerts", "alert_enriched_*.tif")
    files = sorted(glob(pattern))
    if not files:
        print(f"No enriched rasters found"); sys.exit(1)

    # Extract date from filename
    def parse_dates(fname):
        import re
        m = re.search(r'(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})', fname)
        return f"{m.group(1)} -> {m.group(2)}" if m else fname

    for t in thresholds:
        print(f"\n{'='*100}")
        print(f"  THRESHOLD = {t}  (delta_trees <= -{t})")
        print(f"{'='*100}")
        hdr = f"{'Window':<8} {'Dates':<28} {'Total':>8}"
        for cls in CLASS_ORDER:
            hdr += f" | {cls:>16}"
        print(hdr)
        print("-" * len(hdr))

        grand_total = 0
        grand_counts = {c: 0 for c in CLASS_ORDER}

        for i, f in enumerate(files):
            fname = os.path.basename(f)
            dates = parse_dates(fname)
            with rasterio.open(f) as ds:
                data = ds.read().astype(np.float32)
            
            n_alerts, counts = classify_alert_pixels(data, t)
            grand_total += n_alerts

            row = f"W{i:02d}      {dates:<28} {n_alerts:>8,}"
            for cls in CLASS_ORDER:
                n = counts.get(cls, 0)
                grand_counts[cls] = grand_counts.get(cls, 0) + n
                if n > 0:
                    row += f" | {n:>16,}"
                else:
                    row += f" | {'--':>16}"
            print(row)

        # Totals row
        print("-" * len(hdr))
        row = f"{'TOTAL':<8} {'':28} {grand_total:>8,}"
        for cls in CLASS_ORDER:
            n = grand_counts[cls]
            pct = 100*n/grand_total if grand_total else 0
            row += f" | {n:>10,} {pct:>4.1f}%"
        print(row)

if __name__ == "__main__":
    main()
