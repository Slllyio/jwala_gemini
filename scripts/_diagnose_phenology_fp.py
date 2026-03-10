"""
Diagnostic: Are the canopy-loss alerts phenologically driven (false positives)?

Strategy:
1. Pull actual trees_before / trees_after from DB for the Feb-24 live run
2. Compare the raw delta against the EXPECTED phenological delta for the same DOY window
3. Compute RESIDUAL delta (actual - pheno_expected)
4. Show how many alerts survive with residual-based scoring vs raw-delta scoring
5. Print per-beat verdicts
"""
import pathlib, sys, yaml, psycopg2
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")).get("database", {})
conn = psycopg2.connect(
    host=cfg.get("host", "localhost"), port=cfg.get("port", 5432),
    dbname=cfg.get("dbname", "gis_projects"), user=cfg.get("user", "postgres"),
    password=cfg.get("password", ""),
)
cur = conn.cursor()

# Pull beat-level aggregated data from live run
cur.execute("""
SELECT
    beat_name,
    change_type,
    AVG(mean_delta_trees) avg_dt,
    AVG(stacked_score) avg_score,
    AVG(area_ha) avg_ha,
    COUNT(*) n,
    detection_date::date t1,
    baseline_date::date t0
FROM alerts_log
WHERE ingested_at::date = '2026-02-24' AND change_type = 'canopy_loss'
GROUP BY beat_name, change_type, detection_date::date, baseline_date::date
ORDER BY avg_dt
""")
rows = cur.fetchall()
conn.close()

# Load scorer baseline tables
from _simple_dw_score import _doy_baseline, DW_MONTHLY_MEANS, _range_baseline, RANGE_DW_MODELS

def doy_of(date_str):
    return datetime.strptime(str(date_str)[:10], "%Y-%m-%d").timetuple().tm_yday

print()
print("=" * 110)
print("PHENOLOGICAL FALSE-POSITIVE DIAGNOSTIC  (Feb 24 2026 canopy_loss alerts)")
print("=" * 110)
print()
print("COLUMN GUIDE:")
print("  avg_dt      = actual trees delta (t1 - t0) from DB [DW probability units, 0-1]")
print("  pheno_exp   = expected phenological delta from seasonal model (baseline[t1_doy] - baseline[t0_doy])")
print("  residual    = actual_delta - expected_delta  (positive = genuine anomalous loss BEYOND phenology)")
print("  pheno_frac  = what fraction of the signal is explained by normal seasonal decline [100% = all phenology]")
print("  verdict     = REAL (residual > 0.08) | MARGINAL (0.03-0.08) | PHENO (likely normal leaf-fall)")
print()
print(f"{'Beat':<26} {'t0':>10} {'t1':>10} {'avg_dt':>8} {'pheno_exp':>10} {'residual':>10} {'pheno_frac%':>12} {'n':>4} {'ha':>7}  Verdict")
print("-" * 110)

n_real = n_marginal = n_pheno = 0

for r in rows:
    beat, ctype, avg_dt, avg_score, avg_ha, n, t1, t0 = r
    t0_s = str(t0)[:10] if t0 else "?"
    t1_s = str(t1)[:10] if t1 else "?"

    if t0 and t1:
        doy_t0 = doy_of(t0_s)
        doy_t1 = doy_of(t1_s)
        base_t0 = _doy_baseline(doy_t0, DW_MONTHLY_MEANS)
        base_t1 = _doy_baseline(doy_t1, DW_MONTHLY_MEANS)
        pheno_exp = base_t1 - base_t0
    else:
        pheno_exp = 0.0
        base_t0 = base_t1 = None

    avg_dt_f = float(avg_dt or 0)
    residual = avg_dt_f - pheno_exp             # e.g., actual_dt=-0.20 - pheno=-0.012 → residual=-0.188
    residual_drop = -residual                    # positive = genuine anomalous loss
    pheno_exp_drop = -pheno_exp                  # positive = expected normal loss
    # Phenological fraction: how much of the TOTAL drop is explained by phenology?
    total_drop = -avg_dt_f
    pheno_frac_pct = (pheno_exp_drop / max(total_drop, 1e-6)) * 100

    if residual_drop > 0.08:
        verdict = "REAL"
        n_real += 1
    elif residual_drop > 0.03:
        verdict = "MARGINAL"
        n_marginal += 1
    else:
        verdict = "PHENO"
        n_pheno += 1

    print(f"{str(beat)[:26]:<26} {t0_s:>10} {t1_s:>10} {avg_dt_f:>8.3f} {pheno_exp:>10.4f} {residual:>10.4f} {pheno_frac_pct:>11.1f}% {n:>4} {avg_ha:>7.1f}  {verdict}")

print()
print("=" * 110)
print(f"SUMMARY: {len(rows)} beat-threat combinations (canopy_loss only)")
print(f"  REAL   (residual drop > 0.08):    {n_real:>4}  beatgroups  — potentially genuine deforestation")
print(f"  MARGINAL (0.03-0.08):             {n_marginal:>4}  beatgroups  — ambiguous / needs investigation")
print(f"  PHENO  (residual ≤ 0.03):         {n_pheno:>4}  beatgroups  — likely phenological false positive")
print()
print("KEY INSIGHT: If PHENO >> REAL, the phenological deseasonalization is broken.")
print("The fix is to score the RESIDUAL delta (actual - expected_seasonal), not the raw delta.")
print()

# Also show what typical February phenological rates look like
print("FEBRUARY PHENOLOGY REFERENCE (division-wide baseline DW_MONTHLY_MEANS):")
for m in [1, 2, 3, 4]:
    from _simple_dw_score import DW_MONTHLY_MEANS, DW_MONTHLY_STDS
    mu = DW_MONTHLY_MEANS.get(m)
    sd = DW_MONTHLY_STDS.get(m)
    if mu and sd:
        print(f"  Month {m:02d}: mean={mu:.4f}  std={sd:.4f}")
print()
# Show the expected 10-day phenological change in late February
print("Expected 10-day DW trees change by date pair (from seasonal model):")
test_pairs = [
    ("2026-02-12", "2026-02-22"),
    ("2026-02-14", "2026-02-22"),
    ("2026-02-17", "2026-02-22"),
    ("2026-02-17", "2026-02-24"),
    ("2026-02-22", "2026-02-24"),
]
for t0s, t1s in test_pairs:
    d0 = doy_of(t0s)
    d1 = doy_of(t1s)
    b0 = _doy_baseline(d0, DW_MONTHLY_MEANS)
    b1 = _doy_baseline(d1, DW_MONTHLY_MEANS)
    exp_d = b1 - b0
    print(f"  {t0s} -> {t1s} (gap={d1-d0}d):  baseline changes from {b0:.4f} to {b1:.4f} = {exp_d:+.4f}  [pheno expected drop: {-exp_d:.4f}]")
