"""
Verify range phenology model loading and predictions.
Run: python scripts/_test_range_models.py
"""
import sys, os
sys.path.insert(0, "scripts")

from _simple_dw_score import RANGE_DW_MODELS, _range_baseline, dw_multi_threat_score_maxx

print(f"Models loaded: {len(RANGE_DW_MODELS)}")
assert len(RANGE_DW_MODELS) == 8, f"Expected 8 range models, got {len(RANGE_DW_MODELS)}"

print("\nRange baseline predictions for 2026-02-09:")
print(f"  {'Range':<22}  mu     std    R2")
print(f"  {'-'*50}")
for rname, model in sorted(RANGE_DW_MODELS.items()):
    mu, std = _range_baseline(rname, "2026-02-09")
    print(f"  {rname:<22}  {mu:.3f}  {std:.3f}  {model.r2:.3f}")

# Sanity checks
for rname in RANGE_DW_MODELS:
    for date in ["2026-01-15", "2026-04-15", "2026-09-15"]:
        mu, std = _range_baseline(rname, date)
        assert 0.0 <= mu <= 1.0, f"{rname} {date}: mu={mu} out of range"
        assert 0.01 <= std <= 0.5, f"{rname} {date}: std={std} out of range"

print("\nPhysics check: all mu in [0,1], all std in [0.01,0.5]: PASS")

# Test scorer with range_name
print("\nScorer with range_name='North_Guna':")
r = dw_multi_threat_score_maxx(
    trees_after=0.49, trees_before=0.72,
    crops_after=0.03, crops_before=0.03,
    built_after=0.02, built_before=0.02,
    date_str="2026-02-09",
    range_name="North_Guna",
)
print(f"  score={r['score']}  label={r['label']}  typology={r['typology']}")
assert r["label"] == "HIGH", f"Expected HIGH, got {r['label']}"

print("\nScorer without range_name (fallback to division baseline):")
r2 = dw_multi_threat_score_maxx(
    trees_after=0.49, trees_before=0.72,
    crops_after=0.03, crops_before=0.03,
    built_after=0.02, built_before=0.02,
    date_str="2026-02-09",
)
print(f"  score={r2['score']}  label={r2['label']}  typology={r2['typology']}")
assert r2["label"] == "HIGH"

# Test graceful fallback for unknown range
r3 = dw_multi_threat_score_maxx(
    trees_after=0.49, trees_before=0.72,
    crops_after=0.03, crops_before=0.03,
    built_after=0.02, built_before=0.02,
    date_str="2026-02-09",
    range_name="NONEXISTENT_RANGE",
)
assert r3["label"] == "HIGH", "Fallback for unknown range should still give HIGH"
print("  Graceful fallback for unknown range_name: PASS")

print("\n\033[92m=== ALL RANGE MODEL TESTS PASSED ===\033[0m")
