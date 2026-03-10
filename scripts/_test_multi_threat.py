"""
Quick smoke-test for dw_multi_threat_score_maxx.
Run: python scripts/_test_multi_threat.py
"""
import sys
sys.path.insert(0, "scripts")

# pylint: disable=wrong-import-position
from _simple_dw_score import dw_multi_threat_score_maxx  # noqa: E402

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
errors = 0

def check(name, result, expected_typology=None, expected_label=None, should_alert=None):
    global errors
    ok = True
    if expected_typology and result["typology"] != expected_typology:
        print(f"{FAIL} {name}: typology={result['typology']}, expected {expected_typology}")
        ok = False
    if expected_label and result["label"] != expected_label:
        print(f"{FAIL} {name}: label={result['label']}, expected {expected_label}")
        ok = False
    if should_alert is True and result["label"] in ("NO ALERT", "LOW"):
        print(f"{FAIL} {name}: expected alert but got {result['label']}")
        ok = False
    if should_alert is False and result["label"] in ("MEDIUM", "HIGH"):
        print(f"{FAIL} {name}: expected NO alert but got {result['label']}")
        ok = False
    if ok:
        print(f"{PASS} {name}: score={result['score']}  label={result['label']}  type={result['typology']}")
    else:
        errors += 1
    return result

# ── Test 1: Confirmed canopy loss — should fire ──────────────────────────────
r1 = check(
    "Confirmed canopy loss (0.72→0.49)",
    dw_multi_threat_score_maxx(0.49, 0.72, 0.05, 0.05, 0.03, 0.02, "2026-02-09"),
    expected_typology="CANOPY_LOSS",
    should_alert=True,
)

# ── Test 2: Virgin Ground Gate — existing farm seasonal flush ─────────────────
r2 = check(
    "Farm seasonal flush (crops_before=0.60 → crops_after=0.80)",
    dw_multi_threat_score_maxx(0.72, 0.72, 0.80, 0.60, 0.03, 0.02, "2026-09-15"),
    should_alert=False,
)

# ── Test 3: Bare-rock illusion — built rise but below 0.30 gate ───────────────
r3 = check(
    "Bare-rock illusion (built_after=0.22 < 0.30 gate)",
    dw_multi_threat_score_maxx(0.60, 0.62, 0.04, 0.04, 0.22, 0.05, "2026-02-15"),
    should_alert=False,
)

# ── Test 4: Real infrastructure — built_after=0.45 clears gate ───────────────
r4 = check(
    "Real built encroachment (built_after=0.45)",
    dw_multi_threat_score_maxx(0.60, 0.62, 0.04, 0.04, 0.45, 0.05, "2026-02-15"),
    expected_typology="BUILT_ENCROACHMENT",
)

# ── Test 5: 95% cloud, HIGH-evidence clear-cut ───────────────────────────────
# Adaptive exponent: cloud_weight = base_cloud_w ^ (1 - best_ev)
# When ev=1.0 → exponent=0.0 → cloud_w=1.0  (high-certainty events resist cloud)
# This is CORRECT by design — a confirmed 60pp tree drop is real regardless of cloud.
r5 = check(
    "Clear-cut under 95% cloud (ev=1.0 → cloud_w=1.0 by adaptive exponent design)",
    dw_multi_threat_score_maxx(0.20, 0.80, 0.05, 0.05, 0.03, 0.02, "2026-02-09", cloud_frac=0.95),
    expected_typology="CANOPY_LOSS",
    should_alert=True,
)
# The adaptive cloud_w=1.0 for ev=1.0 is correct. To see cloud crushing a MARGINAL case:
r5b = dw_multi_threat_score_maxx(0.62, 0.70, 0.05, 0.05, 0.03, 0.02, "2026-02-09", cloud_frac=0.95)
assert r5b["cloud_w"] < 0.5, f"Marginal ev under 95% cloud should be crushed, got cloud_w={r5b['cloud_w']}"
print(f"       Marginal ev cloud_w={r5b['cloud_w']:.3f}  score={r5b['score']}  [cloud crushes marginal]")

# ── Test 6: New crop encroachment — virgin land (crops_before=0.05) ───────────
r6 = check(
    "New crop encroachment (crops_before=0.05, crops_after=0.55)",
    dw_multi_threat_score_maxx(0.55, 0.60, 0.55, 0.05, 0.03, 0.02, "2026-09-15"),
    expected_typology="CROP_ENCROACHMENT",
)

# ── Test 7: Verbose breakdown ─────────────────────────────────────────────────
print("\n-- Verbose breakdown: confirmed canopy loss ---")
dw_multi_threat_score_maxx(0.49, 0.72, 0.05, 0.05, 0.03, 0.02, "2026-02-09", verbose=True)

print()
if errors == 0:
    print(f"\033[92m=== ALL {7} TESTS PASSED ===\033[0m")
else:
    print(f"\033[91m=== {errors} TEST(S) FAILED ===\033[0m")
    sys.exit(1)
