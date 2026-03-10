"""
_historical_simulation.py
=========================
Comprehensive backtest of the V5 deseasonalized scorer from Oct 2015
through Jan 2026. For every 10-day step, simulates 8 drop/change
scenarios and records scores + labels.

Output: outputs/simulation/v5_historical_simulation.csv

Usage:
    python scripts/_historical_simulation.py
"""
import sys, pathlib, csv
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from datetime import datetime, timedelta
from _simple_dw_score import (
    dw_pheno_score_maxx, dw_multi_threat_score_maxx,
    _doy_baseline, _doy_std,
    DW_MONTHLY_MEANS, DW_MONTHLY_STDS,
)

# ── Config ───────────────────────────────────────────────────────────────────
ROOT       = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR    = ROOT / "outputs" / "simulation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH   = OUT_DIR / "v5_historical_simulation.csv"

START_DATE = datetime(2015, 10, 1)
END_DATE   = datetime(2026, 1, 31)
STEP_DAYS  = 10          # every 10 days
LOOKBACK   = 30          # t0 is 30 days before t1

# Scenarios: (name, delta_trees_from_before, delta_crops, delta_built)
# delta is applied to the baseline "before" value
SCENARIOS = [
    ("no_change",             0.00,   0.00,  0.00),
    ("natural_phenology",    "pheno", 0.00,  0.00),   # special: uses expected seasonal change
    ("mild_drop_-0.05",     -0.05,   0.00,  0.00),
    ("moderate_drop_-0.10", -0.10,   0.01,  0.00),
    ("strong_drop_-0.15",   -0.15,   0.03,  0.01),
    ("severe_drop_-0.20",   -0.20,   0.05,  0.02),
    ("deforestation_-0.35", -0.35,   0.10,  0.05),
    ("total_clearing_-0.50",-0.50,   0.15,  0.10),
]

# ── Simulation loop ─────────────────────────────────────────────────────────
print(f"Simulating V5 scorer from {START_DATE.date()} to {END_DATE.date()}")
print(f"Step: {STEP_DAYS} days | Lookback: {LOOKBACK} days | Scenarios: {len(SCENARIOS)}")

fields = [
    "date", "day", "month", "month_name", "year", "doy",
    "scenario",
    "trees_before", "trees_after", "raw_delta",
    "expected_pheno_change", "adj_delta",
    "baseline_mu_t1", "baseline_std_t1",
    "baseline_mu_t0", "baseline_std_t0",
    "v5_score", "v5_label",
    "v5_canopy_score", "v5_canopy_label",
    "patch_z",
]

rows_written = 0
month_names = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    t1 = START_DATE
    while t1 <= END_DATE:
        t0 = t1 - timedelta(days=LOOKBACK)
        t1_str = t1.strftime("%Y-%m-%d")
        t0_str = t0.strftime("%Y-%m-%d")

        t1_doy = t1.timetuple().tm_yday
        t0_doy = t0.timetuple().tm_yday

        # Division baselines at t0 and t1
        base_mu_t0 = _doy_baseline(t0_doy, DW_MONTHLY_MEANS)
        base_std_t0 = max(_doy_std(t0_doy, DW_MONTHLY_STDS), 0.05)
        base_mu_t1 = _doy_baseline(t1_doy, DW_MONTHLY_MEANS)
        base_std_t1 = max(_doy_std(t1_doy, DW_MONTHLY_STDS), 0.05)

        # Expected phenological change (clamped for green-up trap)
        expected_pheno = min(0.0, base_mu_t1 - base_mu_t0)

        # "Before" state = baseline at t0 (typical healthy forest)
        trees_before = base_mu_t0

        for scenario_name, delta_t, delta_c, delta_b in SCENARIOS:
            # Handle special "natural_phenology" scenario
            if delta_t == "pheno":
                # Natural change = baseline_t1 - baseline_t0 (unclamped)
                actual_delta = base_mu_t1 - base_mu_t0
            else:
                actual_delta = float(delta_t)

            trees_after = trees_before + actual_delta
            trees_after = max(0.0, min(1.0, trees_after))  # clamp [0,1]

            raw_delta = trees_after - trees_before
            adj_delta = raw_delta - expected_pheno

            # Crops / built
            crops_before = 0.10
            crops_after = max(0.0, min(1.0, crops_before + float(delta_c if delta_c != "pheno" else 0)))
            built_before = 0.02
            built_after = max(0.0, min(1.0, built_before + float(delta_b if delta_b != "pheno" else 0)))

            # V5 patch_z (state anomaly)
            patch_z = round((trees_after - base_mu_t1) / base_std_t1, 3)

            # V5 multi-threat score
            v5_multi = dw_multi_threat_score_maxx(
                trees_after=trees_after, trees_before=trees_before,
                crops_after=crops_after, crops_before=crops_before,
                built_after=built_after, built_before=built_before,
                date_str=t1_str, cloud_frac=0.0,
                patch_z=patch_z, t0_date_str=t0_str,
            )

            # V5 canopy-only score
            v5_canopy = dw_pheno_score_maxx(
                trees_after=trees_after, trees_before=trees_before,
                date_str=t1_str, cloud_frac=0.0, t0_date_str=t0_str,
            )

            row = {
                "date":         t1_str,
                "day":          t1.day,
                "month":        t1.month,
                "month_name":   month_names[t1.month],
                "year":         t1.year,
                "doy":          t1_doy,
                "scenario":     scenario_name,
                "trees_before": round(trees_before, 4),
                "trees_after":  round(trees_after, 4),
                "raw_delta":    round(raw_delta, 4),
                "expected_pheno_change": round(expected_pheno, 4),
                "adj_delta":    round(adj_delta, 4),
                "baseline_mu_t1": round(base_mu_t1, 4),
                "baseline_std_t1": round(base_std_t1, 4),
                "baseline_mu_t0": round(base_mu_t0, 4),
                "baseline_std_t0": round(base_std_t0, 4),
                "v5_score":     round(v5_multi["score"], 4),
                "v5_label":     v5_multi["label"],
                "v5_canopy_score": round(v5_canopy["score"], 4),
                "v5_canopy_label": v5_canopy["label"],
                "patch_z":      patch_z,
            }
            writer.writerow(row)
            rows_written += 1

        t1 += timedelta(days=STEP_DAYS)

        # Progress
        if rows_written % 800 == 0:
            pct = 100 * (t1 - START_DATE).days / (END_DATE - START_DATE).days
            print(f"  {t1.date()} ({pct:.0f}%) — {rows_written} rows written")

print(f"\n✅ Simulation complete: {rows_written} rows written to {CSV_PATH}")
print(f"   Date range: {START_DATE.date()} to {END_DATE.date()}")
print(f"   Scenarios:  {len(SCENARIOS)} per timestep")
print(f"   Step size:  {STEP_DAYS} days")

# ── Quick summary statistics ─────────────────────────────────────────────────
import pandas as pd

df = pd.read_csv(CSV_PATH)
print(f"\n{'=' * 70}")
print("SUMMARY BY SCENARIO AND LABEL")
print("=" * 70)
pivot = df.groupby(["scenario", "v5_label"]).size().unstack(fill_value=0)
# Reorder columns
for col in ["HIGH", "MEDIUM", "LOW", "NO ALERT"]:
    if col not in pivot.columns:
        pivot[col] = 0
pivot = pivot[["HIGH", "MEDIUM", "LOW", "NO ALERT"]]
print(pivot.to_string())

print(f"\n{'=' * 70}")
print("MONTHLY AVERAGE V5 SCORE BY SCENARIO")
print("=" * 70)
monthly = df.groupby(["month_name", "month", "scenario"])["v5_score"].mean().reset_index()
monthly = monthly.sort_values(["scenario", "month"])
for scenario in df["scenario"].unique():
    s = monthly[monthly["scenario"] == scenario]
    print(f"\n  {scenario}:")
    for _, row in s.iterrows():
        bar = "#" * int(row["v5_score"] * 40)
        print(f"    {row['month_name']:>12} ({int(row['month']):02d}): {row['v5_score']:.3f} {bar}")

print(f"\n{'=' * 70}")
print("HIGH ALERTS BY YEAR AND MONTH (deforestation_-0.35 scenario)")
print("=" * 70)
defor = df[(df["scenario"] == "deforestation_-0.35") & (df["v5_label"] == "HIGH")]
by_ym = defor.groupby(["year", "month_name"]).size().reset_index(name="count")
print(by_ym.to_string(index=False))

print("\nDone.")
