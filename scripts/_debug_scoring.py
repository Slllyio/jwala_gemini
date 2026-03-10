"""
Diagnose score_patch() on synthetic signal values matching real Feb 2026 patches.
Prints evidence breakdown so we can see why alerts aren't firing.
"""
import sys, math
sys.path.insert(0, ".")
from src.inference.rules_engine import score_patch, _load_rules_cfg
import yaml

cfg = yaml.safe_load(open("config.yaml"))
rc  = _load_rules_cfg(cfg)

print(f"Thresholds:")
for k, v in rc.items():
    print(f"  {k:30s} = {v}")
print()

# Signal values based on the Feb 2026 patch outputs we saw previously
# (from _test_batch_pairs terminal output and implementation_plan annotations)
patches = [
    # Name, ndvi_z, dw_z, dw_td, dw_bg, dw_crop, dw_blt, dVH, cusum
    # ── EXACT KML values from confirmed Feb 5-20 clearance ──────────────────
    ("Feb09_EXACT_KML",        +0.96, None,  -0.119, +0.025, None, None,  +0.074, None),
    ("Feb09_GEE_Patch4",       +0.83, None,  -0.226, +0.033, None, None,  -0.036, None),  # real GEE patch
    ("Feb12_partial_cloud",    +0.85, None,  -0.095, +0.018, None, None,  +0.21,  None),
    # ── Reference cases ──────────────────────────────────────────────────────
    ("IDEAL_TP",               -3.50, -3.0,  -0.200, 0.12,  0.05,  0.03,  -2.500, 0.600),
    ("NDVI_z_only (-3.5)",     -3.50, None,   None,  None,  None,  None,   None,  None),
    ("DW_instant_only (-0.18)", None, None,  -0.180, None,  None,  None,   None,  None),
    ("DW_instant_just_above",   None, None,  -0.105, None,  None,  None,   None,  None),
    ("Pheno+Instant agree",    -2.5, -2.0,  -0.150,  0.09,  None,  None,  None,  None),
    ("Feb22_P3 (SAR FP)",      -0.45, None,  -0.042, 0.00, -0.026, None,  2.598, 0.537),
]


hdr = (f"{'Test case':30s}  {'flag':4s}  {'conf':6s}  "
       f"{'raw':6s}  {'adj':6s}  {'opt':6s}  {'sar':6s}  {'pheno':6s}  {'inst':6s}")
print(hdr)
print("-" * len(hdr))

for row in patches:
    name, ndvi_z, dw_z, dw_td, dw_bg, dw_crop, dw_blt, dVH, cusum = row
    props = {
        "ndvi_zscore":     ndvi_z,
        "dw_trees_zscore": dw_z,
        "dw_trees_delta":  dw_td,
        "dw_bare_delta":   dw_bg,
        "dw_crops_delta":  dw_crop,
        "dw_built_delta":  dw_blt,
        "dVH":             dVH,
        "cusum_score":     cusum,
        "dNDVI":           None,
    }
    conf, details = score_patch(props, rc, 0.65, season="DRY")
    tier_label = "–" if conf < 0.35 else ("H" if conf >= 0.65 else "M")
    print(
        f"{name:30s}  {tier_label:1s}  {conf:6.3f}  "
        f"{details.get('raw',0):6.3f}  {details.get('adj',0):6.3f}  "
        f"{details.get('opt',0):6.3f}  {details.get('sar',0):6.3f}  "
        f"{details.get('pheno',0):6.3f}  {details.get('inst',0):6.3f}"
    )
