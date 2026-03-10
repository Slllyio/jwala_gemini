"""
_rescore_viz.py — Visualize V4 vs V5 re-scoring comparison
"""
import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import yaml, psycopg2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict

from _simple_dw_score import (
    dw_multi_threat_score_maxx,
    _doy_baseline, _doy_std,
    DW_MONTHLY_MEANS, DW_MONTHLY_STDS,
)

# ── DB ───────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "viz"
OUT.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")).get("database", {})
conn = psycopg2.connect(
    host=cfg.get("host", "localhost"), port=cfg.get("port", 5432),
    dbname=cfg.get("dbname", "gis_projects"),
    user=cfg.get("user", "postgres"), password=cfg.get("password", ""),
)

# ── Fetch data ───────────────────────────────────────────────────────────────
cur = conn.cursor()
cur.execute("""
    SELECT DISTINCT beat_name FROM alerts_log
    WHERE detection_date >= '2026-02-22' AND source = 'ews_daemon_v4'
    ORDER BY beat_name
""")
all_beats = [r[0] for r in cur.fetchall()]
random.seed(42)
sample_beats = sorted(random.sample(all_beats, min(20, len(all_beats))))

placeholders = ",".join(["%s"] * len(sample_beats))
cur.execute(f"""
    SELECT beat_name, change_type, stacked_score,
           CASE WHEN stacked_score >= 0.65 THEN 'HIGH'
                WHEN stacked_score >= 0.45 THEN 'MEDIUM'
                WHEN stacked_score >= 0.25 THEN 'LOW'
                ELSE 'NO ALERT' END AS v4_label,
           mean_delta_trees, mean_delta_crops, mean_delta_built,
           area_ha, detection_date::text AS t1_date
    FROM alerts_log
    WHERE beat_name IN ({placeholders})
      AND detection_date >= '2026-02-22' AND source = 'ews_daemon_v4'
    ORDER BY beat_name, stacked_score DESC
""", sample_beats)
rows = cur.fetchall()
cols = [d[0] for d in cur.description]
conn.close()

# ── Re-score with V5 ────────────────────────────────────────────────────────
LOOKBACK = 30
records = []
for row in rows:
    r = dict(zip(cols, row))
    v4_score = float(r["stacked_score"] or 0)
    v4_label = r["v4_label"]
    delta_t = float(r["mean_delta_trees"] or 0)
    delta_c = float(r["mean_delta_crops"] or 0)
    delta_b = float(r["mean_delta_built"] or 0)
    t1_str = r["t1_date"][:10]
    t1_dt = datetime.strptime(t1_str, "%Y-%m-%d")
    t0_dt = t1_dt - timedelta(days=LOOKBACK)
    t0_str = t0_dt.strftime("%Y-%m-%d")

    trees_before = _doy_baseline(t0_dt.timetuple().tm_yday, DW_MONTHLY_MEANS)
    trees_after = trees_before + delta_t
    t1_doy = t1_dt.timetuple().tm_yday
    div_mu = _doy_baseline(t1_doy, DW_MONTHLY_MEANS)
    div_std = max(_doy_std(t1_doy, DW_MONTHLY_STDS), 0.05)
    v5_pz = round((trees_after - div_mu) / div_std, 3)

    v5_data = dw_multi_threat_score_maxx(
        trees_after=trees_after, trees_before=trees_before,
        crops_after=0.10 + delta_c, crops_before=0.10,
        built_after=0.02 + delta_b, built_before=0.02,
        date_str=t1_str, cloud_frac=0.0, patch_z=v5_pz, t0_date_str=t0_str,
    )
    records.append({
        "beat": r["beat_name"], "v4_score": v4_score, "v4_label": v4_label,
        "v5_score": v5_data["score"], "v5_label": v5_data["label"],
        "delta_trees": delta_t, "area_ha": float(r["area_ha"] or 0),
    })

# ── Colour palette ───────────────────────────────────────────────────────────
COLORS = {
    "HIGH": "#E53935",      # vivid red
    "MEDIUM": "#FB8C00",    # amber
    "LOW": "#FDD835",       # yellow
    "NO ALERT": "#43A047",  # green
}
BG = "#181C24"
CARD = "#1E2330"
GRID = "#2A2F3C"
TEXT = "#E0E0E0"
ACCENT = "#64B5F6"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "font.family": "sans-serif",
    "font.size": 11,
})

# ── FIGURE 1: Multi-panel comparison ────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.suptitle("V4 → V5 Phenology Fix : 20-Beat Re-scoring Comparison",
             fontsize=20, fontweight="bold", color="white", y=0.97)

# PANEL 1 — Stacked bar: label distribution
ax1 = fig.add_subplot(2, 3, 1)
labels_order = ["HIGH", "MEDIUM", "LOW", "NO ALERT"]
v4_counts = Counter(r["v4_label"] for r in records)
v5_counts = Counter(r["v5_label"] for r in records)
x = np.array([0, 1])
bottoms_v4, bottoms_v5 = np.zeros(1), np.zeros(1)
for lbl in reversed(labels_order):
    c4, c5 = v4_counts.get(lbl, 0), v5_counts.get(lbl, 0)
    ax1.bar(0, c4, bottom=bottoms_v4[0], color=COLORS[lbl], width=0.5, edgecolor=BG)
    ax1.bar(1, c5, bottom=bottoms_v5[0], color=COLORS[lbl], width=0.5, edgecolor=BG)
    bottoms_v4[0] += c4
    bottoms_v5[0] += c5
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["V4 (old)", "V5 (fixed)"], fontsize=12, fontweight="bold")
ax1.set_ylabel("Alert Count", fontsize=12)
ax1.set_title("Alert Distribution", fontsize=14, fontweight="bold", pad=10)
patches = [mpatches.Patch(color=COLORS[l], label=l) for l in labels_order]
ax1.legend(handles=patches, loc="upper right", fontsize=9, framealpha=0.3)

# PANEL 2 — Score scatter: V4 vs V5
ax2 = fig.add_subplot(2, 3, 2)
v4_sc = [r["v4_score"] for r in records]
v5_sc = [r["v5_score"] for r in records]
colors = [COLORS[r["v5_label"]] for r in records]
ax2.scatter(v4_sc, v5_sc, c=colors, alpha=0.5, s=15, edgecolors="none")
ax2.plot([0, 1], [0, 1], "--", color="#555", linewidth=1, label="No change")
ax2.axhline(0.70, color=COLORS["HIGH"], linewidth=0.8, linestyle=":", alpha=0.6)
ax2.axhline(0.45, color=COLORS["MEDIUM"], linewidth=0.8, linestyle=":", alpha=0.6)
ax2.axvline(0.65, color=COLORS["HIGH"], linewidth=0.8, linestyle=":", alpha=0.6)
ax2.set_xlabel("V4 Score", fontsize=12)
ax2.set_ylabel("V5 Score", fontsize=12)
ax2.set_title("Score Migration", fontsize=14, fontweight="bold", pad=10)
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(-0.05, 1.05)

# PANEL 3 — Delta trees histogram: what V4 called HIGH
ax3 = fig.add_subplot(2, 3, 3)
v4_highs = [r for r in records if r["v4_label"] == "HIGH"]
still_high = [r["delta_trees"] for r in v4_highs if r["v5_label"] == "HIGH"]
downgraded = [r["delta_trees"] for r in v4_highs if r["v5_label"] != "HIGH"]
bins = np.linspace(-0.40, 0, 30)
ax3.hist(downgraded, bins=bins, color=COLORS["NO ALERT"], alpha=0.8,
         label=f"Downgraded ({len(downgraded)})", edgecolor=BG)
ax3.hist(still_high, bins=bins, color=COLORS["HIGH"], alpha=0.9,
         label=f"Still HIGH ({len(still_high)})", edgecolor=BG)
ax3.set_xlabel("ΔTrees (DW probability)", fontsize=12)
ax3.set_ylabel("Count", fontsize=12)
ax3.set_title("V4 HIGHs: Where Did They Go?", fontsize=14, fontweight="bold", pad=10)
ax3.legend(fontsize=9, framealpha=0.3)

# PANEL 4 — Per-beat HIGH reduction waterfall
ax4 = fig.add_subplot(2, 1, 2)
beat_v4h = defaultdict(int)
beat_v5h = defaultdict(int)
for r in records:
    if r["v4_label"] == "HIGH": beat_v4h[r["beat"]] += 1
    if r["v5_label"] == "HIGH": beat_v5h[r["beat"]] += 1
# Sort by V4 HIGH count descending
sorted_beats = sorted(beat_v4h.keys(), key=lambda b: beat_v4h[b], reverse=True)
if not sorted_beats:
    sorted_beats = sample_beats
x_pos = np.arange(len(sorted_beats))
v4_bars = [beat_v4h.get(b, 0) for b in sorted_beats]
v5_bars = [beat_v5h.get(b, 0) for b in sorted_beats]

bar_width = 0.35
bars1 = ax4.bar(x_pos - bar_width/2, v4_bars, bar_width, color="#E53935",
                alpha=0.85, label="V4 HIGH", edgecolor=BG)
bars2 = ax4.bar(x_pos + bar_width/2, v5_bars, bar_width, color="#43A047",
                alpha=0.85, label="V5 HIGH", edgecolor=BG)

# Add reduction labels
for i, (v4, v5) in enumerate(zip(v4_bars, v5_bars)):
    if v4 > 0:
        pct = 100 * (v4 - v5) / v4
        ax4.text(i, max(v4, v5) + 1, f"−{pct:.0f}%",
                 ha="center", fontsize=8, color=ACCENT, fontweight="bold")

ax4.set_xticks(x_pos)
ax4.set_xticklabels([b[:12] for b in sorted_beats], rotation=45, ha="right", fontsize=9)
ax4.set_ylabel("HIGH Alert Count", fontsize=12)
ax4.set_title("Per-Beat HIGH Alert Reduction", fontsize=14, fontweight="bold", pad=10)
ax4.legend(fontsize=10, framealpha=0.3)

# Summary annotation
total_v4h = sum(v4_bars)
total_v5h = sum(v5_bars)
summary_text = (f"Total: {total_v4h} → {total_v5h} HIGHs  "
                f"(−{100*(total_v4h-total_v5h)/max(total_v4h,1):.0f}%)")
fig.text(0.5, 0.01, summary_text, ha="center", fontsize=14,
         fontweight="bold", color=ACCENT,
         bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD, edgecolor=ACCENT, alpha=0.9))

plt.tight_layout(rect=[0, 0.03, 1, 0.94])
out_path = OUT / "v4_vs_v5_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()

# ── FIGURE 2: Score distribution violin/box ──────────────────────────────────
fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle("Score Distribution: V4 vs V5", fontsize=16, fontweight="bold", color="white")

# Box plots
data_box = [v4_sc, v5_sc]
bp = ax5.boxplot(data_box, labels=["V4 (old)", "V5 (fixed)"],
                 patch_artist=True, widths=0.5,
                 medianprops=dict(color="white", linewidth=2))
bp["boxes"][0].set_facecolor("#E53935")
bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor("#43A047")
bp["boxes"][1].set_alpha(0.7)
for whisker in bp["whiskers"]: whisker.set_color(TEXT)
for cap in bp["caps"]: cap.set_color(TEXT)
for flier in bp["fliers"]: flier.set(marker="o", markerfacecolor=ACCENT, markersize=3, alpha=0.4)
ax5.axhline(0.70, color=COLORS["HIGH"], linewidth=1, linestyle="--", alpha=0.5, label="HIGH ≥ 0.70")
ax5.axhline(0.45, color=COLORS["MEDIUM"], linewidth=1, linestyle="--", alpha=0.5, label="MEDIUM ≥ 0.45")
ax5.set_ylabel("Score", fontsize=12)
ax5.set_title("Score Box Plot", fontsize=13, fontweight="bold")
ax5.legend(fontsize=9, framealpha=0.3)

# Histogram overlay
bins_h = np.linspace(0, 1, 40)
ax6.hist(v4_sc, bins=bins_h, alpha=0.6, color="#E53935", label="V4", edgecolor=BG)
ax6.hist(v5_sc, bins=bins_h, alpha=0.6, color="#43A047", label="V5", edgecolor=BG)
ax6.axvline(0.70, color="white", linewidth=1, linestyle="--", alpha=0.6)
ax6.axvline(0.45, color="white", linewidth=1, linestyle="--", alpha=0.4)
ax6.set_xlabel("Score", fontsize=12)
ax6.set_ylabel("Count", fontsize=12)
ax6.set_title("Score Histogram", fontsize=13, fontweight="bold")
ax6.legend(fontsize=10, framealpha=0.3)

plt.tight_layout()
out2 = OUT / "v4_vs_v5_distributions.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close()

print(f"\nTotal alerts: {len(records)}")
print(f"V4 HIGHs: {sum(1 for r in records if r['v4_label']=='HIGH')}")
print(f"V5 HIGHs: {sum(1 for r in records if r['v5_label']=='HIGH')}")
print("Done.")
