"""
_score_10yr_vsn.py  —  Pandas Local Backtester for VSN Export
=============================================================
Loads the 10-year pixel history CSV from GEE (exported by _export_10yr_vsn.py),
pairs consecutive clean passes per pixel, applies the V5 scorer, and produces:

  1. Full scored CSV: outputs/simulation/v5_10yr_backtest_all_beats.csv
  2. Summary analytics + visualisations

Usage:
    python scripts/_score_10yr_vsn.py [--input PATH_TO_CSV]
"""
import sys, pathlib, argparse, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from datetime import datetime

from _simple_dw_score import (
    dw_multi_threat_score_maxx,
    dw_pheno_score_maxx,
    _doy_baseline, _doy_std,
    DW_MONTHLY_MEANS, DW_MONTHLY_STDS,
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
SIM_DIR = ROOT / "outputs" / "simulation"
SIM_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR = ROOT / "outputs" / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ── Parse args ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Score 10yr VSN pixel history")
parser.add_argument("--input", type=str,
                    default=str(SIM_DIR / "Guna_10Yr_Pixel_History.csv"),
                    help="Path to the GEE-exported CSV")
parser.add_argument("--max-gap", type=int, default=90,
                    help="Max days between consecutive passes (default: 90)")
parser.add_argument("--cloud-max", type=float, default=30.0,
                    help="Max cloud probability to consider clean (default: 30)")
args = parser.parse_args()

# ── Load Data ────────────────────────────────────────────────────────────────
t_start = time.time()
print(f"📂 Loading pixel history from: {args.input}")
df = pd.read_csv(args.input)
print(f"   {len(df):,} raw rows loaded ({df['beat_name'].nunique()} beats)")

# ── Clean + Sort ─────────────────────────────────────────────────────────────
# Ensure numeric columns
for col in ["trees", "crops", "built"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
# Add probability column if missing (our exporter doesn't fetch cloud data)
if "probability" not in df.columns:
    df["probability"] = 0.0
else:
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce").fillna(0.0)
df = df.dropna(subset=["trees", "crops", "built"]).copy()

# Parse dates
df["date_obj"] = pd.to_datetime(df["date"])
df["year"] = df["date_obj"].dt.year
df["month"] = df["date_obj"].dt.month
df["day"] = df["date_obj"].dt.day
df["doy"] = df["date_obj"].dt.dayofyear
df["month_name"] = df["date_obj"].dt.strftime("%B")

# Create a unique pixel ID (beat_name may not be unique across rows, need point identity)
# If GEE didn't assign point_id, create one from beat+geometry hash
if "point_id" not in df.columns:
    # Group by beat_name and assign sequential point IDs within each beat
    # based on spatial position (we'll use the row order within each beat-date group)
    print("   ⚠ No point_id column — creating synthetic pixel identifiers...")
    # Since GEE randomPoints with seed=42 is deterministic, same points appear each pass.
    # We identify unique pixels by their (beat_name, trees, crops, built) fingerprint
    # across all dates. A simpler approach: assign row-order based ID per beat per date.
    df = df.sort_values(["beat_name", "date_obj"]).reset_index(drop=True)
    df["_rank"] = df.groupby(["beat_name", "date"]).cumcount()
    df["point_id"] = df["beat_name"] + "_pt" + df["_rank"].astype(str)
    df = df.drop(columns=["_rank"])

# Sort chronologically per pixel
df = df.sort_values(["point_id", "date_obj"]).reset_index(drop=True)
print(f"   {df['point_id'].nunique():,} unique pixels after cleaning")

# ── Cloud Gating ─────────────────────────────────────────────────────────────
n_before = len(df)
if df["probability"].max() > 0:
    df = df[df["probability"] < args.cloud_max].copy()
    n_after = len(df)
    print(f"☁️  Cloud gate (<{args.cloud_max}%): {n_before:,} → {n_after:,} rows "
          f"({100 * n_after / max(n_before, 1):.1f}% kept)")
else:
    n_after = len(df)
    print(f"☁️  No cloud data — skipping cloud gate ({n_after:,} rows kept)")

# ── Pandas Time-Machine: Pair Consecutive Clean Passes ───────────────────────
print("⏱  Pairing consecutive temporal passes per pixel...")
grp = df.groupby("point_id")
df["trees_before"] = grp["trees"].shift(1)
df["crops_before"] = grp["crops"].shift(1)
df["built_before"] = grp["built"].shift(1)
df["t0_date"]      = grp["date"].shift(1)
df["t0_date_obj"]  = grp["date_obj"].shift(1)

# Drop first row of every pixel (no t0 baseline)
df = df.dropna(subset=["trees_before"]).copy()

# DEDUP: remove duplicate (point_id, date) rows from overlapping S2 granules
n_pre_dedup = len(df)
df = df.drop_duplicates(subset=["point_id", "date"], keep="first").copy()
print(f"🔄 Dedup: {n_pre_dedup:,} → {len(df):,} rows (removed {n_pre_dedup - len(df):,} dupes)")

# Compute gap and filter
df["gap_days"] = (df["date_obj"] - df["t0_date_obj"]).dt.days
n_before_gap = len(df)
df = df[(df["gap_days"] > 0) & (df["gap_days"] <= args.max_gap)].copy()
print(f"📏 Gap filter (0<gap≤{args.max_gap}d): {n_before_gap:,} → {len(df):,} pairs")

# ── Compute Derived Columns ──────────────────────────────────────────────────
df["raw_delta_trees"] = df["trees"] - df["trees_before"]
df["raw_delta_crops"] = df["crops"] - df["crops_before"]
df["raw_delta_built"] = df["built"] - df["built_before"]

# Division baseline at t1 for each pair
df["baseline_mu_t1"] = df["doy"].apply(lambda d: round(_doy_baseline(d, DW_MONTHLY_MEANS), 4))
df["baseline_std_t1"] = df["doy"].apply(lambda d: round(max(_doy_std(d, DW_MONTHLY_STDS), 0.05), 4))

# ── Apply V5 Scorer ──────────────────────────────────────────────────────────
print(f"\n🧠 Applying V5 scorer to {len(df):,} pairs...")

def apply_v5(row):
    """Score one pair using V5 multi-threat scorer."""
    # Patch-level z-score (state anomaly at t1)
    patch_z = (row["trees"] - row["baseline_mu_t1"]) / max(row["baseline_std_t1"], 0.05)

    # Multi-threat score
    res = dw_multi_threat_score_maxx(
        trees_after=row["trees"],
        trees_before=row["trees_before"],
        crops_after=row["crops"],
        crops_before=row["crops_before"],
        built_after=row["built"],
        built_before=row["built_before"],
        date_str=row["date"],
        cloud_frac=row["probability"] / 100.0,
        patch_z=patch_z,
        t0_date_str=row["t0_date"],
    )

    return pd.Series({
        "v5_score": round(res["score"], 4),
        "v5_label": res["label"],
        "typology": res.get("typology", ""),
        "patch_z": round(patch_z, 3),
    })

# Apply in chunks to show progress
CHUNK = 5000
results = []
for i in range(0, len(df), CHUNK):
    chunk = df.iloc[i:i + CHUNK]
    scored = chunk.apply(apply_v5, axis=1)
    results.append(scored)
    pct = min(100, 100 * (i + CHUNK) / len(df))
    print(f"   {pct:5.1f}%  ({i + len(scored):,} / {len(df):,})")

scored_df = pd.concat(results, ignore_index=True)
df = df.reset_index(drop=True)
df[["v5_score", "v5_label", "typology", "patch_z"]] = scored_df

elapsed = time.time() - t_start
print(f"\n⏱  Scoring completed in {elapsed:.1f}s")

# ── Save Full Results ────────────────────────────────────────────────────────
out_cols = [
    "beat_name", "range_name", "point_id",
    "t0_date", "date", "day", "month", "month_name", "year", "doy", "gap_days",
    "probability",
    "trees_before", "trees", "crops_before", "crops",
    "built_before", "built",
    "raw_delta_trees", "raw_delta_crops", "raw_delta_built",
    "baseline_mu_t1", "baseline_std_t1",
    "patch_z", "v5_score", "v5_label", "typology",
]
# Only include columns that exist
out_cols = [c for c in out_cols if c in df.columns]

out_path = SIM_DIR / "v5_10yr_backtest_all_beats.csv"
df[out_cols].to_csv(out_path, index=False)
print(f"\n💾 Saved {len(df):,} rows to {out_path}")

# ═══════════════════════════════════════════════════════════════════════════════
#                        ANALYTICS & VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 70}")
print("  V5 BACKTEST ANALYTICS  —  10-YEAR VIRTUAL SENSOR NETWORK")
print(f"{'═' * 70}")

# 1. Overall label distribution
print("\n📊 Overall Label Distribution:")
label_counts = df["v5_label"].value_counts()
for lbl in ["HIGH", "MEDIUM", "LOW", "NO ALERT"]:
    n = label_counts.get(lbl, 0)
    pct = 100 * n / len(df)
    bar = "█" * int(pct / 2)
    print(f"   {lbl:>10}: {n:>8,}  ({pct:5.1f}%)  {bar}")

# 2. HIGH alerts by typology
print("\n🔥 HIGH Alerts by Typology:")
high_df = df[df["v5_label"] == "HIGH"]
if len(high_df) > 0:
    for typ, cnt in high_df["typology"].value_counts().items():
        print(f"   {typ}: {cnt:,}")
else:
    print("   (none)")

# 3. THE WINTER SILENCE TEST — Feb/Mar/Apr canopy_loss HIGHs
print("\n❄️  WINTER SILENCE TEST (Feb/Mar/Apr):")
canopy_high = df[(df["v5_label"] == "HIGH") & (df["typology"].str.upper() == "CANOPY_LOSS")]
winter_months = [2, 3, 4]
for m in winter_months:
    month_name = ["", "Jan", "Feb", "Mar", "Apr", "May"][m]
    n = len(canopy_high[canopy_high["month"] == m])
    total_pairs_m = len(df[df["month"] == m])
    rate = 100 * n / max(total_pairs_m, 1)
    print(f"   {month_name}: {n:>5} HIGHs out of {total_pairs_m:,} pairs  ({rate:.2f}%)")

# 4. HIGH canopy alerts by month (all years)
print("\n📅 HIGH Canopy Alerts by Month (all years):")
month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
for m in range(1, 13):
    n = len(canopy_high[canopy_high["month"] == m])
    bar = "█" * min(n, 50)
    print(f"   {month_names[m]:>3} ({m:02d}): {n:>5}  {bar}")

# 5. HIGH alerts by year
print("\n📆 HIGH Alerts by Year:")
by_year = df[df["v5_label"] == "HIGH"].groupby("year").size()
for yr, cnt in by_year.items():
    print(f"   {yr}: {cnt:>5}")

# 6. Per-beat alert rate
print("\n🌳 Top 15 Beats by HIGH Alert Rate:")
beat_total = df.groupby("beat_name").size().rename("total")
beat_high = high_df.groupby("beat_name").size().rename("highs")
beat_stats = pd.concat([beat_total, beat_high], axis=1).fillna(0)
beat_stats["rate"] = 100 * beat_stats["highs"] / beat_stats["total"]
beat_stats = beat_stats.sort_values("rate", ascending=False)
for _, row in beat_stats.head(15).iterrows():
    print(f"   {row.name:>20}: {int(row['highs']):>4} HIGHs / {int(row['total']):>5} "
          f"pairs  ({row['rate']:.1f}%)")

# 7. Monthly score heatmap data (year × month)
print("\n📊 Mean V5 Score by Year × Month:")
heatmap = df.groupby(["year", "month"])["v5_score"].mean().unstack(fill_value=0)
print(heatmap.round(3).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
#                          VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    BG = "#181C24"
    CARD = "#1E2330"
    GRID = "#2A2F3C"
    TEXT = "#E0E0E0"
    ACCENT = "#64B5F6"
    COLORS = {"HIGH": "#E53935", "MEDIUM": "#FB8C00", "LOW": "#FDD835", "NO ALERT": "#43A047"}

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": CARD,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "text.color": TEXT, "font.family": "sans-serif", "font.size": 11,
    })

    # ── Fig 1: Year × Month Heatmap ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle("V5 Mean Score: Year × Month Heatmap (10-Year VSN Backtest)",
                 fontsize=16, fontweight="bold", color="white")

    hm_data = df.groupby(["year", "month"])["v5_score"].mean().unstack(fill_value=0)
    im = ax.imshow(hm_data.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.5)
    ax.set_yticks(range(len(hm_data.index)))
    ax.set_yticklabels(hm_data.index)
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names[1:], rotation=45, ha="right")
    ax.set_ylabel("Year")
    ax.set_xlabel("Month")
    plt.colorbar(im, ax=ax, label="Mean V5 Score", shrink=0.8)

    # Annotate cells
    for i in range(len(hm_data.index)):
        for j in range(len(hm_data.columns)):
            val = hm_data.iloc[i, j]
            color = "white" if val > 0.25 else TEXT
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    plt.tight_layout()
    fig.savefig(VIZ_DIR / "vsn_10yr_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"\n📈 Saved: {VIZ_DIR / 'vsn_10yr_heatmap.png'}")
    plt.close()

    # ── Fig 2: Monthly HIGH Alert Count (stacked by year) ───────────────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig2.suptitle("10-Year VSN Backtest: Alert Patterns",
                  fontsize=16, fontweight="bold", color="white")

    # Panel 1: HIGH canopy alerts by month
    canopy_by_month = canopy_high.groupby("month").size().reindex(range(1, 13), fill_value=0)
    bars = ax1.bar(range(1, 13), canopy_by_month.values,
                   color=[COLORS["HIGH"] if m not in [2, 3, 4] else ACCENT for m in range(1, 13)],
                   alpha=0.85, edgecolor=BG)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(month_names[1:], rotation=45, ha="right")
    ax1.set_ylabel("HIGH Canopy Alerts")
    ax1.set_title("Winter Silence Test: Feb-Apr Should Be Near Zero", fontweight="bold")

    # Highlight Feb/Mar/Apr
    for m in winter_months:
        ax1.axvspan(m - 0.5, m + 0.5, alpha=0.1, color=ACCENT)

    # Panel 2: All labels by year
    label_year = df.groupby(["year", "v5_label"]).size().unstack(fill_value=0)
    for col in ["HIGH", "MEDIUM", "LOW", "NO ALERT"]:
        if col not in label_year.columns:
            label_year[col] = 0
    label_year = label_year[["HIGH", "MEDIUM", "LOW", "NO ALERT"]]
    label_year.plot(kind="bar", stacked=True, ax=ax2,
                    color=[COLORS[l] for l in label_year.columns],
                    edgecolor=BG, alpha=0.85)
    ax2.set_ylabel("Total Pairs")
    ax2.set_title("Alert Distribution by Year", fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.3)

    plt.tight_layout()
    fig2.savefig(VIZ_DIR / "vsn_10yr_alerts.png", dpi=150, bbox_inches="tight")
    print(f"📈 Saved: {VIZ_DIR / 'vsn_10yr_alerts.png'}")
    plt.close()

    # ── Fig 3: Score time series for one sample pixel ────────────────────────
    sample_pixel = df["point_id"].value_counts().idxmax()  # pixel with most data
    px = df[df["point_id"] == sample_pixel].copy()
    fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig3.suptitle(f"10-Year Pixel History: {sample_pixel}",
                  fontsize=14, fontweight="bold", color="white")

    ax3.plot(px["date_obj"], px["trees"], "g-", alpha=0.7, linewidth=1, label="trees_after")
    ax3.plot(px["date_obj"], px["trees_before"], "g--", alpha=0.4, linewidth=0.8, label="trees_before")
    ax3.fill_between(px["date_obj"], px["trees_before"], px["trees"],
                     alpha=0.15, color="green")
    ax3.set_ylabel("DW Trees Probability")
    ax3.set_title("Tree Cover Time Series")
    ax3.legend(fontsize=9, framealpha=0.3)

    colors_ts = [COLORS.get(l, "#666") for l in px["v5_label"]]
    ax4.scatter(px["date_obj"], px["v5_score"], c=colors_ts, s=10, alpha=0.7)
    ax4.axhline(0.70, color=COLORS["HIGH"], linewidth=0.8, linestyle="--", alpha=0.5)
    ax4.axhline(0.45, color=COLORS["MEDIUM"], linewidth=0.8, linestyle="--", alpha=0.5)
    ax4.set_ylabel("V5 Score")
    ax4.set_xlabel("Date")
    ax4.set_title("V5 Score Over Time")

    patches = [mpatches.Patch(color=COLORS[l], label=l)
               for l in ["HIGH", "MEDIUM", "LOW", "NO ALERT"]]
    ax4.legend(handles=patches, fontsize=9, framealpha=0.3)

    plt.tight_layout()
    fig3.savefig(VIZ_DIR / "vsn_pixel_timeseries.png", dpi=150, bbox_inches="tight")
    print(f"📈 Saved: {VIZ_DIR / 'vsn_pixel_timeseries.png'}")
    plt.close()

except ImportError:
    print("\n(matplotlib not available — skipping visualisations)")

print(f"\n✅ 10-Year VSN Backtest Complete  ({time.time() - t_start:.1f}s total)")
