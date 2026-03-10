"""
Simulation: Quantify how badly the patch_z bug inflates canopy-loss scores.
"""
import sys, math
sys.path.insert(0, 'scripts')
from _simple_dw_score import (
    _range_baseline, RANGE_DW_MODELS, _doy_baseline,
    DW_MONTHLY_MEANS, dw_multi_threat_score_maxx
)
from datetime import datetime

def doy(d):
    return datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday

# ----- Range model predictions -----------------------------------------------
print("South_Guna harmonic model predictions (what model expects DW trees to be):")
for ds in ["2026-02-12","2026-02-14","2026-02-17","2026-02-22","2026-02-24","2026-03-15"]:
    mu, std = _range_baseline("South_Guna", ds)
    print(f"  {ds}: model_mu={mu:.4f}  model_std={std:.4f}")

print()
t1_date = "2026-02-22"
t0_date = "2026-02-12"
base_mu, base_std = _range_baseline("South_Guna", t1_date)
d0_mu, _  = _range_baseline("South_Guna", t0_date)

# Expected phenological delta from harmonic model
pheno_exp_harmonic = base_mu - d0_mu

d0_div = _doy_baseline(doy(t0_date), DW_MONTHLY_MEANS)
d1_div = _doy_baseline(doy(t1_date), DW_MONTHLY_MEANS)
pheno_exp_monthly = d1_div - d0_div

print(f"Expected phenological delta  {t0_date}->{t1_date}:")
print(f"  Monthly-table model:        {pheno_exp_monthly:+.4f}")
print(f"  Range harmonic model:       {pheno_exp_harmonic:+.4f}")
print()

# ----- Score simulation -------------------------------------------------------
print("=" * 80)
print("SCORE SIMULATION: Feb leaf-fall scenario (trees_before=0.65, t0=Feb12, t1=Feb22)")
print("=" * 80)

scenarios = [
    # (trees_before, trees_after, description)
    (0.65, 0.60, "Mild drop (-0.05): gentle winter leaf flush"),
    (0.65, 0.55, "Moderate drop (-0.10): typical Feb senescence"),
    (0.65, 0.52, "Stronger drop (-0.13): fast Feb senescence"),
    (0.65, 0.48, "Heavy drop (-0.17): severe but still seasonal"),
    (0.65, 0.40, "Very heavy drop (-0.25): unusual - possible real loss"),
    (0.65, 0.20, "REAL DEFORESTATION (-0.45): clear-felling"),
    (0.65, 0.10, "TOTAL LOSS (-0.55): complete clearance"),
]

print(f"{'Trees change':<28} {'delta':>7} {'patch_z':>9} {'patch_z':>9} {'Score':>7} {'Score':>7} {'Label':>8} {'Label':>8}")
print(f"{'Scenario':<28} {'actual':>7} {'BUGGY':>9} {'CORRECT':>9} {'BUGGY':>7} {'CORRECT':>7} {'BUGGY':>8} {'CORRECT':>8}")
print("-" * 90)

for trees_before, trees_after, desc in scenarios:
    delta = trees_after - trees_before
    drop  = -delta

    # Current (buggy) z-score: uses DELTA / residual_std
    z_buggy = delta / max(base_std, 1e-6)

    # Correct z-score: uses (trees_after - baseline_mu) / residual_std
    z_correct = (trees_after - base_mu) / max(base_std, 1e-6)

    s_buggy = dw_multi_threat_score_maxx(
        trees_after=trees_after, trees_before=trees_before,
        crops_after=0.05, crops_before=0.05,
        built_after=0.01, built_before=0.01,
        date_str=t1_date, cloud_frac=0.0,
        range_name="South_Guna",
        patch_z=z_buggy, cusum_score=0.0
    )

    s_correct = dw_multi_threat_score_maxx(
        trees_after=trees_after, trees_before=trees_before,
        crops_after=0.05, crops_before=0.05,
        built_after=0.01, built_before=0.01,
        date_str=t1_date, cloud_frac=0.0,
        range_name="South_Guna",
        patch_z=z_correct, cusum_score=0.0
    )

    print(f"{desc[:28]:<28} {delta:>7.3f} {z_buggy:>9.2f} {z_correct:>9.2f}"
          f" {s_buggy['score']:>7.3f} {s_correct['score']:>7.3f}"
          f" {s_buggy['label']:>8} {s_correct['label']:>8}")

print()

# Also show what the deseasonalized scorer would see
print("=" * 80)
print("DESEASONALIZED SCORING: what residual remains after removing pheno_expected?")
print(f"  Harmonic expected delta ({t0_date} -> {t1_date}): {pheno_exp_harmonic:+.4f}")
print()
print(f"{'Scenario':<28} {'raw_delta':>10} {'residual':>10} {'residual_label'}")
print("-" * 60)
for trees_before, trees_after, desc in scenarios:
    delta = trees_after - trees_before
    residual = delta - pheno_exp_harmonic  # anomalous component
    # residual > 0 means trees dropped MORE than expected = genuine loss signal
    residual_drop = max(0.0, -residual)
    if residual_drop > 0.15:
        verdict = "STRONG SIGNAL"
    elif residual_drop > 0.08:
        verdict = "MODERATE"
    elif residual_drop > 0.03:
        verdict = "WEAK"
    else:
        verdict = "PHENOLOGY (suppress)"
    print(f"{desc[:28]:<28} {delta:>10.3f} {residual:>10.3f} {verdict}")
