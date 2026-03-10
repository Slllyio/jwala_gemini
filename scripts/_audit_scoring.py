"""
Deep scoring audit for two confirmed-miss patches in Mar Ki Mahu.
Runs exact verbose traces and proposes calibration fixes.

Patch 1 (Feb 14): NO ALERT  0.172  ΔTrees=-0.60pp  0.80 ha  CuSuM=0.273
Patch 2 (Feb 09): MEDIUM    0.672  ΔTrees=-6.99pp  0.30 ha  CuSuM=0.298
Both are in the KML-confirmed deforestation zone → SHOULD be HIGH.

Run: python scripts/_audit_scoring.py
"""
import sys, math
sys.path.insert(0, "scripts")
sys.stdout.reconfigure(encoding="utf-8")
from _simple_dw_score import (
    dw_pheno_score_maxx, dw_multi_threat_score_maxx,
    RANGE_DW_MODELS, _range_baseline, _logistic
)

RANGE = "North_Guna"
SEP   = "-" * 72

def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")

# ── North_Guna baseline for both dates ──────────────────────────────────────
mu_feb09, std_feb09 = _range_baseline(RANGE, "2026-02-09")
mu_feb14, std_feb14 = _range_baseline(RANGE, "2026-02-14")

section("NORTH_GUNA BASELINE CONTEXT")
print(f"  Feb 09: μ = {mu_feb09:.4f}  σ = {std_feb09:.4f}")
print(f"  Feb 14: μ = {mu_feb14:.4f}  σ = {std_feb14:.4f}")
print(f"  Range model R²: {RANGE_DW_MODELS[RANGE].r2:.3f}")

# ============================================================================
#  PATCH 2 — Feb 09  ΔTrees = -6.99pp = -0.0699
#  Score: 0.672 MEDIUM  (should be HIGH)
# ============================================================================
section("PATCH 2  (Feb 09  ΔTrees=-6.99pp = -0.0699)")

DELTA_P2      = -0.0699
TREES_AFT_P2  = max(0.0, mu_feb09 + DELTA_P2)   # reconstruct from delta + mu
TREES_BEF_P2  = min(1.0, TREES_AFT_P2 - DELTA_P2)
CROPS_BEF_P2  = 0.05
CROPS_AFT_P2  = 0.05
BUILT_BEF_P2  = 0.02
BUILT_AFT_P2  = 0.02
CLOUD_P2      = 0.0   # Feb 09 had 0.3% cloud

print(f"\n  Reconstructed inputs:")
print(f"    trees_before={TREES_BEF_P2:.4f}  trees_after={TREES_AFT_P2:.4f}")
print(f"    delta={DELTA_P2:+.4f}  (={DELTA_P2*100:.2f} pp)")

# --- V2 verbose ---
print(f"\n  V2 canopy sub-scorer (verbose):")
r2 = dw_pheno_score_maxx(
    trees_after=TREES_AFT_P2, trees_before=TREES_BEF_P2,
    date_str="2026-02-09", cloud_frac=CLOUD_P2,
    verbose=True, range_name=RANGE
)
print(f"  → score={r2['score']}  label={r2['label']}")

# --- V3 full ---
print(f"\n  V3 multi-threat (verbose):")
r3 = dw_multi_threat_score_maxx(
    trees_after=TREES_AFT_P2, trees_before=TREES_BEF_P2,
    crops_after=CROPS_AFT_P2, crops_before=CROPS_BEF_P2,
    built_after=BUILT_AFT_P2, built_before=BUILT_BEF_P2,
    date_str="2026-02-09", cloud_frac=CLOUD_P2,
    verbose=True, range_name=RANGE
)
print(f"  → score={r3['score']}  label={r3['label']}")

# --- ROOT CAUSE DISSECTION ---
drop_p2     = abs(DELTA_P2)
dyn_mid_p2  = 0.05 + TREES_BEF_P2 * 0.10
raw_ev_p2   = _logistic(drop_p2, midpoint=dyn_mid_p2, steepness=25.0)
eff_base_p2 = max(TREES_BEF_P2, mu_feb09)
z_p2        = (TREES_AFT_P2 - eff_base_p2) / std_feb09
raw_bonus_p2 = max(0.0, -z_p2) / 4.0
gate_p2     = _logistic(drop_p2, midpoint=0.08, steepness=40.0)
gated_p2    = raw_bonus_p2 * gate_p2
evidence_p2 = min(raw_ev_p2 * (1.0 + gated_p2), 1.0)

print(f"\n  ── ROOT CAUSE DISSECTION ──")
print(f"  drop_mag      = {drop_p2:.4f} ({drop_p2*100:.2f} pp)")
print(f"  dyn_midpoint  = {dyn_mid_p2:.4f}  ← sigmoid midpoint for raw_ev")
print(f"  raw_ev        = {raw_ev_p2:.4f}  ← WHERE the score plateaus")
print(f"  eff_baseline  = {eff_base_p2:.4f}")
print(f"  z_dw          = {z_p2:+.4f} sigma")
print(f"  raw_bonus     = {raw_bonus_p2:.4f}")
print(f"  gate (bonus gate) = {gate_p2:.4f}")
print(f"  gated_bonus   = {gated_p2:.4f}")
print(f"  evidence      = {evidence_p2:.4f}")
print(f"  PROBLEM: raw_ev={raw_ev_p2:.4f} is the ceiling → evidence={evidence_p2:.4f} → just below HIGH at 0.70")
print(f"  A -6.99pp drop on a {TREES_BEF_P2:.3f} baseline = {drop_p2/TREES_BEF_P2*100:.1f}% RELATIVE loss")
print(f"  The scorer uses ABSOLUTE drop (6.99pp) — barely above the sigmoid midpoint ({dyn_mid_p2*100:.1f}pp)")

# ============================================================================
#  PATCH 1 — Feb 14  ΔTrees = -0.60pp = -0.006
#  Score: 0.172 NO ALERT  (should be HIGH — patch is inside KML zone)
# ============================================================================
section("PATCH 1  (Feb 14  ΔTrees=-0.60pp = -0.006)")

DELTA_P1     = -0.006
TREES_AFT_P1 = max(0.0, mu_feb14 + DELTA_P1)
TREES_BEF_P1 = min(1.0, TREES_AFT_P1 - DELTA_P1)
CLOUD_P1     = 0.0

print(f"\n  Reconstructed inputs:")
print(f"    trees_before={TREES_BEF_P1:.4f}  trees_after={TREES_AFT_P1:.4f}")
print(f"    delta={DELTA_P1:+.4f}  (={DELTA_P1*100:.2f} pp)")

print(f"\n  V2 canopy sub-scorer (verbose):")
r2_p1 = dw_pheno_score_maxx(
    trees_after=TREES_AFT_P1, trees_before=TREES_BEF_P1,
    date_str="2026-02-14", cloud_frac=CLOUD_P1,
    verbose=True, range_name=RANGE
)
print(f"  → score={r2_p1['score']}  label={r2_p1['label']}")

drop_p1     = abs(DELTA_P1)
dyn_mid_p1  = 0.05 + TREES_BEF_P1 * 0.10
raw_ev_p1   = _logistic(drop_p1, midpoint=dyn_mid_p1, steepness=25.0)

print(f"\n  ── ROOT CAUSE DISSECTION ──")
print(f"  drop_mag = {drop_p1:.4f} ({drop_p1*100:.2f} pp)")
print(f"  dyn_mid  = {dyn_mid_p1:.4f} ({dyn_mid_p1*100:.1f} pp)")
print(f"  raw_ev   = {raw_ev_p1:.4f} → TINY because drop < midpoint")
print(f"  FUNDAMENTAL ISSUE: -0.60pp DW Δ is genuinely sub-noise at patch level.")
print(f"  This patch should NOT score high on DW alone on Feb 14.")
print(f"  The SPATIAL OVERLAP with the confirmed KML zone is the real signal.")
print(f"  The scorer cannot see it — it only sees pixel averages, not polygon overlap.")

# ============================================================================
#  THRESHOLD ANALYSIS: where do bands sit?
# ============================================================================
section("THRESHOLD CALIBRATION ANALYSIS")

print("""
  Current thresholds (hardcoded):
    HIGH    ≥ 0.70
    MEDIUM  ≥ 0.45
    LOW     ≥ 0.25
    NO ALERT < 0.25

  Evidence arithmetic for Patch 2 (the borderline case):
    raw_ev × (1 + gated_bonus) = evidence → score
""")

# Show what happens as we vary the HIGH threshold
print("  Effect of changing the HIGH threshold on Patch 2 (score=0.672):")
for thresh in [0.70, 0.65, 0.60, 0.55]:
    lbl = "HIGH" if r3["score"] >= thresh else "MEDIUM"
    print(f"    HIGH threshold={thresh:.2f} → Patch 2 = {lbl}")

print(f"""
  Effect of adding a CUSUM boost rule:
    If CuSuM ≥ 0.25 AND evidence ≥ 0.50 → promote to HIGH
    Patch 2 CuSuM=0.298 ≥ 0.25, evidence={evidence_p2:.3f} ≥ 0.50 → PROMOTED to HIGH ✓
""")

# ============================================================================
#  PROPOSED FIXES
# ============================================================================
section("PROPOSED FIXES")
print("""
  FIX A — Relative-loss amplifier on raw_ev (addresses Patch 2)
  ─────────────────────────────────────────────────────────────
  Problem: 6.99pp absolute drop on a 13.2pp canopy baseline
           = 53% RELATIVE loss. But the scorer treats it same
           as 6.99pp on an 80pp baseline (8.7% relative).
  Fix: blend relative_drop into the midpoint calculation.
       rel_drop = drop_mag / max(trees_before, 0.05)
       hybrid_ev = 0.7*raw_ev + 0.3*logistic(rel_drop, mid=0.30, steep=8)
  This boosts Patch 2: rel_drop=0.53 → hybrid_ev ~0.76 → score HIGH ✓

  FIX B — CuSuM promotion gate (addresses Patch 2)
  ─────────────────────────────────────────────────
  Problem: CuSuM accumulates evidence across multiple passes.
           Patch 2 CuSuM=0.298 is independent confirmation of change.
  Fix: if cusum_score ≥ 0.25 AND evidence ≥ 0.55:
           final_score = max(final_score + 0.08, final_score)
       Patch 2: 0.672 + 0.08 = 0.752 → HIGH ✓

  FIX C — Temporal persistence flag (addresses Patch 1)
  ──────────────────────────────────────────────────────
  Problem: Patch 1 (Feb 14, -0.60pp) genuinely shows near-zero DW Δ.
           But it OVERLAPS the same grid position as Patch 2 (Feb 09 MEDIUM).
           DW probability is lagged — the loss was captured Feb 09,
           so Feb 14 shows plateau not fresh drop.
  Fix: carry a beat-level state dict {patch_id: last_alert_level}.
       If the SAME patch was MEDIUM+ within 14 days → don't demote to NO ALERT;
       hold at LOW minimum.
       Patch 1: previous = MEDIUM (Feb 09) → hold at LOW minimum.
       (This requires daemon state, NOT scorer change.)

  FIX D — Steepness increase for sparse canopy (addresses Patch 2 root cause)
  ────────────────────────────────────────────────────────────────────────────
  Problem: steepness=25 was calibrated for dense forest (trees_before ~0.7+).
           For sparse canopy (trees_before < 0.25), even a 53% relative loss
           only produces 6.99pp absolute → falls on the shallow part of sigmoid.
  Fix: adaptive steepness = 25 + max(0, (0.30 - trees_before) / 0.30) * 20
       → sparse forest (trees_before=0.13): steepness=45
       → dense forest (trees_before=0.80): steepness=25 (unchanged)
""")

section("RECOMMENDED IMMEDIATE ACTION (to fix Patch 2 without breaking others)")
print("""
  1. Add relative_drop signal (FIX A) — direct scorer change in V2/V3
  2. Add CuSuM micro-boost to V3 score (FIX B) — if cusum is available in props
  3. Lower HIGH threshold from 0.70 → 0.65 with adaptive steepness (FIX D)
     (this also catches more valid cases in sparse North_Guna canopy)

  Patch 1 (Feb 14, -0.60pp) is genuinely sub-threshold at the pixel level.
  The real fix for it is daemon-side temporal persistence (FIX C), NOT scorer.
""")
