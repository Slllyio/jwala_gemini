"""
Patch V3 dw_multi_threat_score_maxx to accept:
  - patch_z    : pipeline-computed pixel z-score (from dw_trees_zscore in GeoJSON)
  - cusum_score: CuSuM accumulation score
Both boost the final_score when they exceed their signal thresholds.

This addresses: Feb 14 patch scores 0.113 despite z=-2.61σ and CuSuM=0.273
Expected result: 0.113 → 0.463 LOW
"""
path = r'scripts\_simple_dw_score.py'
txt  = open(path, encoding='utf-8').read()

# 1. Add new params to function signature
OLD_SIG = '''def dw_multi_threat_score_maxx(
    trees_after:  float,
    trees_before: float,
    crops_after:  float,
    crops_before: float,
    built_after:  float,
    built_before: float,
    date_str:     str,          # "YYYY-MM-DD"
    cloud_frac:   float = 0.0,
    verbose:      bool  = False,
    range_name:   str   = "",   # if set, V2 canopy sub-scorer uses range harmonic model
) -> dict:'''

NEW_SIG = '''def dw_multi_threat_score_maxx(
    trees_after:  float,
    trees_before: float,
    crops_after:  float,
    crops_before: float,
    built_after:  float,
    built_before: float,
    date_str:     str,          # "YYYY-MM-DD"
    cloud_frac:   float = 0.0,
    verbose:      bool  = False,
    range_name:   str   = "",   # if set, V2 canopy sub-scorer uses range harmonic model
    patch_z:      float = 0.0,  # pipeline pixel z-score (dw_trees_zscore); 0 = not supplied
    cusum_score:  float = 0.0,  # CuSuM accumulation score; 0 = not supplied
) -> dict:'''

# 2. Add patch_z + cusum boost BEFORE the final label assignment
OLD_SCORE_BLOCK = '''    final_score = best_ev * cloud_weight

    # Threshold aligned to V2: HIGH >= 0.65 (range model std ~0.020)
    label = (\"HIGH\"   if final_score >= 0.65 else
             \"MEDIUM\" if final_score >= 0.45 else
             \"LOW\"    if final_score >= 0.25 else \"NO ALERT\")

    if verbose:
        print(f\"  dominant={best_typology}  best_ev={best_ev:.3f}\"
              f\"  cloud_w={cloud_weight:.3f}  => final_score={final_score:.3f}  [{label}]\")'''

NEW_SCORE_BLOCK = '''    final_score = best_ev * cloud_weight

    # ── Fix 9: Pipeline z-score + CuSuM boost ───────────────────────────────
    # Problem: scorer computes z from RANGE σ (~0.020). Pipeline uses PIXEL σ
    # which can be 8× finer (e.g. 0.0023 for a stable pixel). A -0.60pp drop on
    # a pixel with σ=0.0023 is -2.61σ — statistically significant — but the range
    # model sees it as only -0.30σ and assigns score=0.113.
    #
    # Solution: accept the pre-computed pipeline z (patch_z) and the CuSuM
    # accumulation score (cusum_score) as supplemental evidence channels.
    # Cap total boost at 0.35 to prevent runaway from degenerate inputs.
    #
    #   pz_boost    fires when |patch_z| > 1.96 (95th percentile)
    #   cusum_boost fires when cusum_score > 0.22 (active accumulation)
    pz_boost    = 0.0
    cusum_boost = 0.0
    if patch_z != 0.0 and abs(patch_z) > 1.96:
        pz_boost = _logistic(abs(patch_z), midpoint=1.96, steepness=5.0) * 0.30
    if cusum_score > 0.0:
        cusum_boost = _logistic(cusum_score, midpoint=0.22, steepness=25.0) * 0.20
    total_boost = min(pz_boost + cusum_boost, 0.35)
    final_score = min(final_score + total_boost, 1.0)

    if verbose and (pz_boost > 0 or cusum_boost > 0):
        print(f\"  [Fix9] patch_z={patch_z:+.2f}  pz_boost={pz_boost:.3f}  \"
              f\"cusum={cusum_score:.3f}  cusum_boost={cusum_boost:.3f}  \"
              f\"total_boost={total_boost:.3f}  => boosted={final_score:.3f}\")

    # Threshold aligned to V2: HIGH >= 0.65 (range model std ~0.020)
    label = (\"HIGH\"   if final_score >= 0.65 else
             \"MEDIUM\" if final_score >= 0.45 else
             \"LOW\"    if final_score >= 0.25 else \"NO ALERT\")

    if verbose:
        print(f\"  dominant={best_typology}  best_ev={best_ev:.3f}\"
              f\"  cloud_w={cloud_weight:.3f}  => final_score={final_score:.3f}  [{label}]\")'''

# Apply patches
if OLD_SIG not in txt:
    print("ERROR: signature not found")
else:
    txt = txt.replace(OLD_SIG, NEW_SIG, 1)
    print("Signature patched OK")

if OLD_SCORE_BLOCK not in txt:
    print("ERROR: score block not found — may need CRLF normalisation")
    # Try normalising line endings
    OLD2 = OLD_SCORE_BLOCK.replace('\r\n', '\n')
    if OLD2 in txt.replace('\r\n', '\n'):
        txt_norm = txt.replace('\r\n', '\n')
        txt = txt_norm.replace(OLD2, NEW_SCORE_BLOCK.replace('\r\n', '\n'), 1)
        print("Score block patched OK (LF mode)")
    else:
        print("CRITICAL: block not found even with LF normalisation")
        # Print context around final_score
        idx = txt.find('final_score = best_ev * cloud_weight')
        print("Context:", repr(txt[idx:idx+400]))
else:
    txt = txt.replace(OLD_SCORE_BLOCK, NEW_SCORE_BLOCK, 1)
    print("Score block patched OK")

open(path, 'w', encoding='utf-8').write(txt)
print("Done.")
