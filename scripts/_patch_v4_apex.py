"""
V4 Apex Scorer Patch — Three Critical Architectural Fixes

FIX A: |patch_z| absolute value bug → strictly directional (patch_z < -1.96 only)
FIX B: pz_boost/cusum_boost injected into tree_ev BEFORE typology and cloud
FIX C: Remove boosts from the post-cloud-shield block entirely

Also: Reconstruction trap note (requires GeoJSON fix in daemon, not scorer)
"""
path = r'scripts\_simple_dw_score.py'
txt  = open(path, encoding='utf-8').read()

# =============================================================================
# FIX B + FIX A: Inject boosts into tree_ev before typology resolution
# Remove the old Fix 9 post-cloud-shield block
# =============================================================================

OLD_V3_BODY = '''    # ── Sub-scorer A: Canopy Loss (V2 SOTA, cloud applied globally below) ──
    tree_res = dw_pheno_score_maxx(
        trees_after  = trees_after,
        trees_before = trees_before,
        date_str     = date_str,
        cloud_frac   = 0.0,      # raw evidence; cloud applied uniformly below
        verbose      = verbose,
        range_name   = range_name,   # use range harmonic baseline when available
    )
    tree_ev = tree_res["evidence"]'''

NEW_V3_BODY = '''    # ── Sub-scorer A: Canopy Loss (V2 SOTA, cloud applied globally below) ──
    tree_res = dw_pheno_score_maxx(
        trees_after  = trees_after,
        trees_before = trees_before,
        date_str     = date_str,
        cloud_frac   = 0.0,      # raw evidence; cloud applied uniformly below
        verbose      = verbose,
        range_name   = range_name,   # use range harmonic baseline when available
    )
    base_tree_ev = tree_res["evidence"]

    # ── V4 Fix A+B: Pipeline z-score + CuSuM boost INTO tree_ev ─────────────
    # CRITICALLY: injected HERE — before typology resolution and before cloud
    # shielding — so that:
    #   · Boosts only raise tree_ev → cannot contaminate CROP/BUILT typologies
    #   · Cloud exponent reflects the boosted evidence → physics-correct
    #   · Directional guard: ONLY fire for loss (patch_z < -1.96), never green-up
    #
    # The absolute value bug (|patch_z|) would fire on +3.5σ green-up events,
    # generating phantom deforestation alerts. Fixed by signed threshold.
    pz_boost    = 0.0
    cusum_boost = 0.0
    if patch_z is not None and patch_z < -1.96:          # strictly loss direction
        pz_boost = _logistic(-patch_z, midpoint=1.96, steepness=5.0) * 0.30
    if cusum_score is not None and cusum_score > 0.0:
        cusum_boost = _logistic(cusum_score, midpoint=0.22, steepness=25.0) * 0.20
    total_boost = min(pz_boost + cusum_boost, 0.35)
    tree_ev = min(base_tree_ev + total_boost, 1.0)

    if verbose and (pz_boost > 0 or cusum_boost > 0):
        print(f"  [V4] patch_z={patch_z:+.2f} pz_boost={pz_boost:.3f} "
              f"cusum={cusum_score:.3f} cusum_boost={cusum_boost:.3f} "
              f"Δtree_ev={total_boost:.3f} "
              f"tree_ev: {base_tree_ev:.3f} → {tree_ev:.3f}")'''

# ── Also remove the old Fix 9 block that was post-cloud-shield
OLD_FIX9_BLOCK = '''    final_score = best_ev * cloud_weight

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
        print(f"  [Fix9] patch_z={patch_z:+.2f}  pz_boost={pz_boost:.3f}  "
              f"cusum={cusum_score:.3f}  cusum_boost={cusum_boost:.3f}  "
              f"total_boost={total_boost:.3f}  => boosted={final_score:.3f}")'''

NEW_NO_FIX9 = '''    final_score = best_ev * cloud_weight
    # NOTE: pz_boost + cusum_boost are applied into tree_ev above (before typology
    # and cloud shielding), so no post-cloud adjustment is needed or allowed.'''

# Apply patches
n_applied = 0

if OLD_V3_BODY in txt:
    txt = txt.replace(OLD_V3_BODY, NEW_V3_BODY, 1)
    print("FIX B+A: tree_ev injection patched OK")
    n_applied += 1
else:
    print("ERROR: OLD_V3_BODY not found")
    idx = txt.find('Sub-scorer A: Canopy Loss')
    print("Context:", repr(txt[max(0,idx-50):idx+400]))

if OLD_FIX9_BLOCK in txt:
    txt = txt.replace(OLD_FIX9_BLOCK, NEW_NO_FIX9, 1)
    print("FIX B: old post-cloud Fix9 block removed OK")
    n_applied += 1
else:
    # Try LF
    OLD_FIX9_LF = OLD_FIX9_BLOCK.replace('\r\n', '\n')
    NEW_FIX9_LF = NEW_NO_FIX9.replace('\r\n', '\n')
    txt_lf = txt.replace('\r\n', '\n')
    if OLD_FIX9_LF in txt_lf:
        txt = txt_lf.replace(OLD_FIX9_LF, NEW_FIX9_LF, 1)
        print("FIX B: old Fix9 removed OK (LF mode)")
        n_applied += 1
    else:
        print("ERROR: Fix9 block not found — searching context...")
        idx = txt.find('Fix 9')
        if idx < 0: idx = txt.find('pz_boost    = 0.0')
        print("Context:", repr(txt[max(0,idx-20):idx+600]))

open(path, 'w', encoding='utf-8').write(txt)
print(f"\nDone. {n_applied}/2 patches applied.")
