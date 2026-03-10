"""
FULL BRAIN ANALYSIS — Feb 14 Patch (+2953+6861)
-----------------------------------------------
Key data:
  dw_trees_delta = -0.0060  (-0.60 pp)
  dw_trees_zscore = -2.61σ   (from pipeline)
  cusum_score = 0.273
  area_ha = 0.80

The problem: z=-2.61σ (pipeline) but scorer gives 0.113.

This script traces EXACTLY why, then implements the fix.
"""
import sys, math
sys.path.insert(0, "scripts")
sys.stdout.reconfigure(encoding="utf-8")
from _simple_dw_score import (
    dw_pheno_score_maxx, dw_multi_threat_score_maxx,
    RANGE_DW_MODELS, _range_baseline, _logistic
)

SEP = "─" * 72
RANGE = "North_Guna"

mu, std = _range_baseline(RANGE, "2026-02-14")

print("=" * 72)
print("  DEEP ANALYSIS — Feb 14 Patch (+2953+6861)")
print("=" * 72)

print(f"""
  From GeoJSON (what the pipeline recorded):
    dw_trees_delta    = -0.0060   (-0.60 pp)
    dw_trees_zscore   = -2.61 σ   ← computed by pipeline
    cusum_score       = 0.273     ← CuSuM accumulation
    area_ha           = 0.80
    cloud_frac        = 0.0 (assumed clear)

  From range model (North_Guna, DOY 45 / Feb 14):
    μ (baseline mean) = {mu:.4f}  ({mu*100:.2f} pp)
    σ (baseline std)  = {std:.4f}  ({std*100:.2f} pp)   ← RANGE-LEVEL σ
""")

# --- Reconstruct how the pipeline computed z = -2.61 ---
delta = -0.006
pipeline_z = -2.61

# Solve: z = delta / pixel_std  →  pixel_std = delta / z
# pixel_std = -0.006 / -2.61 = 0.0023
pixel_std_implied = abs(delta / pipeline_z)

print(f"  ── Z-SCORE MISMATCH ANALYSIS ──────────────────────────────────────")
print(f"""
  The pipeline z-score is:   z = ΔTrees / pixel_temporal_std
  If z = -2.61 and ΔTrees = -0.006:
  
    pixel_temporal_std = |ΔTrees| / |z| = 0.006 / 2.61 = {pixel_std_implied:.4f}

  This means the INDIVIDUAL PATCH has a pixel-level temporal std of ONLY
  {pixel_std_implied*100:.2f} pp — extremely low variance, an almost-pristine pixel.
  
  Our range model uses σ = {std:.4f} ({std*100:.2f} pp) — 8.6× LARGER than patch σ.
  
  When the scorer uses the range-level σ:
    scorer_z = -0.006 / {std:.4f} = {delta/std:.2f}  ← only -0.30σ !
    gate fires: raw_bonus = max(0, +0.30) / 4 = 0.075
    bonus_gate = logistic(0.006, mid=0.08, steep=40) = {_logistic(0.006, 0.08, 40):.3f}
    gated_bonus = 0.075 × {_logistic(0.006, 0.08, 40):.3f} = {0.075*_logistic(0.006, 0.08, 40):.3f}

  CORE PROBLEM:
  The range-level σ (0.020) drowns out the -0.60pp drop.
  But the pixel's OWN time-series says this drop is 2.61σ — very unusual for IT.
  The scorer should USE the pipeline z-score, not recompute from range σ.
""")

# --- CuSuM is also being totally ignored ---
print(f"  ── CUSUM IS COMPLETELY IGNORED ─────────────────────────────────────")
print(f"""
  cusum_score = 0.273

  CuSuM (Cumulative Sum) is an independent, multi-pass change detector.
  It accumulates anomalies over time without false-resetting on noise.
  CuSuM = 0.273 means: across MULTIPLE satellite passes, this patch
  has been consistently anomalous — ACCUMULATED evidence, not single-pass.

  The V3 scorer does NOT accept cusum_score as an input parameter.
  It is stored in GeoJSON but thrown away at scoring time.
  This is a design gap — CuSuM provides exactly the temporal persistence
  evidence that a single-date DW Δ cannot.
""")

# --- What the score SHOULD be if we use pipeline z + cusum ---
print(f"  ── WHAT THE SCORE SHOULD BE ─────────────────────────────────────────")
# Use pipeline z=-2.61 directly as the z_dw
# raw_bonus = max(0, 2.61) / 4 = 0.6525
# gate at drop=0.006 → still small but non-zero
# But the real signal is the z itself: at z=-2.61, this is a 99th percentile event FOR THIS PIXEL

# Additionally, use cusum as a multiplier
cusum_ev = _logistic(0.273, midpoint=0.22, steepness=25.0)  # CuSuM evidence
pz_ev    = _logistic(2.61,  midpoint=1.96, steepness=5.0)   # pipeline z evidence (use absolute)

print(f"""
  If we incorporate pipeline z (-2.61σ) and CuSuM (0.273):
  
    cusum_evidence = logistic(cusum=0.273, mid=0.22, steep=25) = {cusum_ev:.3f}
    pipez_evidence = logistic(|z|=2.61,  mid=1.96, steep=5)  = {pz_ev:.3f}
  
  Combined boost = max(cusum_ev, pipez_ev) = {max(cusum_ev, pz_ev):.3f}
  
  Current base score = 0.113
  Boosted score = 0.113 + 0.30 × {max(cusum_ev, pz_ev):.3f} = {0.113 + 0.30 * max(cusum_ev, pz_ev):.3f}
  
  Result: {0.113 + 0.30 * max(cusum_ev, pz_ev):.3f} → {'LOW' if 0.113 + 0.30 * max(cusum_ev, pz_ev) >= 0.25 else 'STILL NO ALERT (but higher)'}
  
  With z=-2.61σ + CuSuM=0.273 + 0.80 ha, at minimum this should be LOW.
""")

print(f"  ── THE COMPLETE FIX PLAN ───────────────────────────────────────────")
print(f"""
  FIX 1 — Pass pipeline z-score INTO the scorer (V3 only)
    dw_multi_threat_score_maxx(..., patch_z=props['dw_trees_zscore'])
    When patch_z is provided AND |patch_z| > 1.96:
      pz_boost = logistic(|patch_z|, mid=1.96, steep=5) × 0.30
      final_score += pz_boost

  FIX 2 — Pass CuSuM INTO the scorer (V3 only)
    dw_multi_threat_score_maxx(..., cusum_score=props['cusum_score'])
    cusum_boost = logistic(cusum, mid=0.22, steep=25) × 0.20
    final_score += cusum_boost

  FIX 3 — Cap combined boosts to avoid runaway
    total_boost = min(pz_boost + cusum_boost, 0.35)
    final_score = min(base_score + total_boost, 1.0)

  Applied to Feb 14 patch:
    base_score   = 0.113
    pz_boost     = logistic(2.61, 1.96, 5) × 0.30 = {pz_ev:.3f} × 0.30 = {pz_ev*0.30:.3f}
    cusum_boost  = logistic(0.273, 0.22, 25) × 0.20 = {cusum_ev:.3f} × 0.20 = {cusum_ev*0.20:.3f}
    total_boost  = min({pz_ev*0.30 + cusum_ev*0.20:.3f}, 0.35) = {min(pz_ev*0.30 + cusum_ev*0.20, 0.35):.3f}
    final_score  = min(0.113 + {min(pz_ev*0.30 + cusum_ev*0.20, 0.35):.3f}, 1.0) = {min(0.113 + min(pz_ev*0.30 + cusum_ev*0.20, 0.35), 1.0):.3f}
    label        = {'LOW' if min(0.113 + min(pz_ev*0.30 + cusum_ev*0.20, 0.35), 1.0) >= 0.25 else 'NO ALERT→boosted'}
    
  Applied to Feb 09 patch (+2943+6862):
    base_score  = 0.710 (already HIGH)
    pz_boost    = logistic(2.29, 1.96, 5) × 0.30 = {_logistic(2.29,1.96,5):.3f} × 0.30 = {_logistic(2.29,1.96,5)*0.30:.3f}
    cusum_boost = logistic(0.298, 0.22, 25) × 0.20 = {_logistic(0.298,0.22,25):.3f} × 0.20 = {_logistic(0.298,0.22,25)*0.20:.3f}
    total_boost = {min(_logistic(2.29,1.96,5)*0.30+_logistic(0.298,0.22,25)*0.20, 0.35):.3f}
    final_score = {min(0.710 + min(_logistic(2.29,1.96,5)*0.30+_logistic(0.298,0.22,25)*0.20, 0.35), 1.0):.3f} HIGH ✓
""")
