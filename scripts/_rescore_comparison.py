"""
_rescore_comparison.py
======================
Pull 20 random beats from the Feb-24 alerts_log, reconstruct before/after
from baseline + delta, re-score with the V5 deseasonalized scorer, and
print a side-by-side comparison table showing V4 (old) vs V5 (new) scores.

Usage:
    python scripts/_rescore_comparison.py
"""
import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import yaml, psycopg2
from datetime import datetime, timedelta

# ── DB connection ─────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")).get("database", {})
conn = psycopg2.connect(
    host=cfg.get("host", "localhost"), port=cfg.get("port", 5432),
    dbname=cfg.get("dbname", "gis_projects"),
    user=cfg.get("user", "postgres"), password=cfg.get("password", ""),
)

# ── Import V5 scorer ────────────────────────────────────────────────────────
from _simple_dw_score import (
    dw_multi_threat_score_maxx,
    _doy_baseline, _doy_std,
    DW_MONTHLY_MEANS, DW_MONTHLY_STDS,
)

# ── Fetch all beats that had alerts in the Feb-24 run ────────────────────────
cur = conn.cursor()
cur.execute("""
    SELECT DISTINCT beat_name
    FROM alerts_log
    WHERE detection_date >= '2026-02-22'
      AND source = 'ews_daemon_v4'
    ORDER BY beat_name
""")
all_beats = [r[0] for r in cur.fetchall()]
print(f"Total beats with V4 alerts: {len(all_beats)}")

# Pick 20 random beats
random.seed(42)  # reproducible
sample_beats = random.sample(all_beats, min(20, len(all_beats)))
sample_beats.sort()
print(f"Selected {len(sample_beats)} beats: {', '.join(sample_beats)}\n")

# ── Fetch all alerts for these beats ─────────────────────────────────────────
placeholders = ",".join(["%s"] * len(sample_beats))
cur.execute(f"""
    SELECT
        beat_name,
        change_type,
        stacked_score AS v4_score,
        CASE
            WHEN stacked_score >= 0.65 THEN 'HIGH'
            WHEN stacked_score >= 0.45 THEN 'MEDIUM'
            WHEN stacked_score >= 0.25 THEN 'LOW'
            ELSE 'NO ALERT'
        END AS v4_label,
        mean_delta_trees,
        mean_delta_crops,
        mean_delta_built,
        area_ha,
        detection_date::text AS t1_date,
        detection_period,
        model_version,
        source
    FROM alerts_log
    WHERE beat_name IN ({placeholders})
      AND detection_date >= '2026-02-22'
      AND source = 'ews_daemon_v4'
    ORDER BY beat_name, stacked_score DESC
""", sample_beats)

rows = cur.fetchall()
cols = [d[0] for d in cur.description]
print(f"Total alerts to re-score: {len(rows)}\n")

# ── Re-score each alert with V5 ─────────────────────────────────────────────
# Since the DB stores only mean_delta_*, we reconstruct before/after:
#   trees_before ≈ division_baseline(t0_doy)
#   trees_after  = trees_before + mean_delta_trees
# For t0_date, we assume ~30 days before detection_date (standard lookback).
LOOKBACK_DAYS = 30

hdr = (f"{'Beat':<25} {'Type':<13} {'Area':>7} {'V4 sc':>6} {'V4':>8}  "
       f"{'V5 sc':>6} {'V5':>8}  {'dTrees':>7}  {'Change':>8}")
print("=" * len(hdr))
print(hdr)
print("-" * len(hdr))

# Accumulators
beat_summary = {}
counters = {k: 0 for k in [
    "v4_high", "v5_high", "v4_med", "v5_med",
    "v4_low", "v5_low", "v4_no", "v5_no",
    "downgrades", "upgrades", "same"
]}

label_rank = {"NO ALERT": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

for row in rows:
    r = dict(zip(cols, row))

    beat     = r["beat_name"]
    threat   = r["change_type"] or "canopy_loss"
    v4_score = float(r["v4_score"]) if r["v4_score"] is not None else 0.0
    v4_label = r["v4_label"]
    area_ha  = float(r["area_ha"]) if r["area_ha"] is not None else 0.0
    delta_t  = float(r["mean_delta_trees"]) if r["mean_delta_trees"] is not None else 0.0
    delta_c  = float(r["mean_delta_crops"]) if r["mean_delta_crops"] is not None else 0.0
    delta_b  = float(r["mean_delta_built"]) if r["mean_delta_built"] is not None else 0.0
    t1_str   = r["t1_date"][:10] if r["t1_date"] else "2026-02-22"

    # Reconstruct dates
    t1_dt  = datetime.strptime(t1_str, "%Y-%m-%d")
    t0_dt  = t1_dt - timedelta(days=LOOKBACK_DAYS)
    t0_str = t0_dt.strftime("%Y-%m-%d")

    # Reconstruct absolute values from baseline + delta
    t0_doy = t0_dt.timetuple().tm_yday
    t1_doy = t1_dt.timetuple().tm_yday
    trees_before = _doy_baseline(t0_doy, DW_MONTHLY_MEANS)
    trees_after  = trees_before + delta_t

    # For crops/built: use small baseline approximation
    crops_before = 0.10   # typical Guna dry-season crops baseline
    crops_after  = crops_before + delta_c
    built_before = 0.02   # typical forest-area built baseline
    built_after  = built_before + delta_b

    cloud_frac = 0.0  # v4 alerts were all clear-sky passes

    # V5 patch_z: state anomaly (corrected)
    div_mu  = _doy_baseline(t1_doy, DW_MONTHLY_MEANS)
    div_std = max(_doy_std(t1_doy, DW_MONTHLY_STDS), 0.05)
    v5_pz   = round((trees_after - div_mu) / div_std, 3)

    # V5 score
    v5_data = dw_multi_threat_score_maxx(
        trees_after  = trees_after,
        trees_before = trees_before,
        crops_after  = crops_after,
        crops_before = crops_before,
        built_after  = built_after,
        built_before = built_before,
        date_str     = t1_str,
        cloud_frac   = cloud_frac,
        patch_z      = v5_pz,
        t0_date_str  = t0_str,
    )
    v5_score = v5_data["score"]
    v5_label = v5_data["label"]

    # Change indicator
    v4_r = label_rank.get(v4_label, 0)
    v5_r = label_rank.get(v5_label, 0)
    if v5_r < v4_r:
        change = "v DOWN"
        counters["downgrades"] += 1
    elif v5_r > v4_r:
        change = "^ UP"
        counters["upgrades"] += 1
    else:
        change = "="
        counters["same"] += 1

    # Count labels
    for ver, lbl in [("v4", v4_label), ("v5", v5_label)]:
        key = f"{ver}_{'high' if lbl == 'HIGH' else 'med' if lbl == 'MEDIUM' else 'low' if lbl == 'LOW' else 'no'}"
        counters[key] += 1

    # Per-beat tracking
    if beat not in beat_summary:
        beat_summary[beat] = {"v4_high": 0, "v4_med": 0, "v5_high": 0, "v5_med": 0, "total": 0}
    beat_summary[beat]["total"] += 1
    if v4_label == "HIGH": beat_summary[beat]["v4_high"] += 1
    if v4_label == "MEDIUM": beat_summary[beat]["v4_med"] += 1
    if v5_label == "HIGH": beat_summary[beat]["v5_high"] += 1
    if v5_label == "MEDIUM": beat_summary[beat]["v5_med"] += 1

    print(f"{beat:<25} {threat:<13} {area_ha:>7.2f} {v4_score:>6.3f} {v4_label:>8}  "
          f"{v5_score:>6.3f} {v5_label:>8}  {delta_t:>+7.3f}  {change:>8}")

# ── Summary ──────────────────────────────────────────────────────────────────
N = max(len(rows), 1)
print(f"\n{'=' * 80}")
print("SUMMARY: V4 vs V5 Re-scoring Comparison (20 Random Beats)")
print("=" * 80)

print(f"\n  Total alerts re-scored:  {len(rows)}")
print(f"  Downgrades (V5 < V4):   {counters['downgrades']:>4}  ({100*counters['downgrades']/N:.1f}%)")
print(f"  Upgrades   (V5 > V4):   {counters['upgrades']:>4}  ({100*counters['upgrades']/N:.1f}%)")
print(f"  Unchanged:              {counters['same']:>4}  ({100*counters['same']/N:.1f}%)")

print(f"\n  {'Label':<12} {'V4 Count':>10} {'V5 Count':>10} {'Delta':>8}")
print(f"  {'-'*42}")
for lbl, k in [("HIGH", "high"), ("MEDIUM", "med"), ("LOW", "low"), ("NO ALERT", "no")]:
    v4c, v5c = counters[f"v4_{k}"], counters[f"v5_{k}"]
    print(f"  {lbl:<12} {v4c:>10} {v5c:>10} {v5c - v4c:>+8}")

print(f"\n  Per-Beat Breakdown:")
print(f"  {'Beat':<25} {'V4 H':>5} {'V4 M':>5} {'V5 H':>5} {'V5 M':>5}  {'Total':>5}  {'H reduced':>10}")
print(f"  {'-'*70}")
for beat in sorted(beat_summary):
    bs = beat_summary[beat]
    h_red = bs["v4_high"] - bs["v5_high"]
    print(f"  {beat:<25} {bs['v4_high']:>5} {bs['v4_med']:>5} {bs['v5_high']:>5} {bs['v5_med']:>5}  {bs['total']:>5}  {h_red:>+10}")

total_v4_h = sum(bs["v4_high"] for bs in beat_summary.values())
total_v5_h = sum(bs["v5_high"] for bs in beat_summary.values())
total_v4_m = sum(bs["v4_med"] for bs in beat_summary.values())
total_v5_m = sum(bs["v5_med"] for bs in beat_summary.values())
print(f"  {'TOTAL':<25} {total_v4_h:>5} {total_v4_m:>5} {total_v5_h:>5} {total_v5_m:>5}  {len(rows):>5}  {total_v4_h - total_v5_h:>+10}")

pct_h_reduction = 100 * (total_v4_h - total_v5_h) / max(total_v4_h, 1)
print(f"\n  HIGH alerts reduced by {pct_h_reduction:.0f}% ({total_v4_h} -> {total_v5_h})")

conn.close()
print("\nDone.")
