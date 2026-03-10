"""
Verbose evidence breakdown for score_patch debugging.
Shows per-signal evidence, weights, and tier threshold analysis.
"""
import sys, math
sys.path.insert(0, ".")
from src.inference.rules_engine import score_patch, _load_rules_cfg, _sigmoid
import yaml

cfg = yaml.safe_load(open("config.yaml"))
rc  = _load_rules_cfg(cfg)

# ── Manual evidence trace for Feb09_GEE_Patch4 ───────────────────────────────
print("=" * 70)
print("MANUAL EVIDENCE TRACE: Feb09_GEE_Patch4")
print("=" * 70)
ndvi_z = +0.83
dw_td  = -0.226
dw_bg  = +0.033
dVH    = -0.036

ndvi_z_t = abs(rc["ndvi_z_thresh"])   # 2.0
dw_td_t  = rc["dw_tree_drop"]         # should be 0.10
dw_bg_t  = rc["dw_bare_rise"]         # 0.08
vh_eff   = rc["vh_drop_db"] * 0.75    # 1.5

print(f"\n  ndvi_z_thresh = {ndvi_z_t}, dw_tree_drop = {dw_td_t}, dw_bare_rise = {dw_bg_t}")

dw_td_e = _sigmoid(-dw_td - dw_td_t, scale=15.0)
dw_bg_e = _sigmoid(dw_bg - dw_bg_t, scale=15.0)  # undamped (dw_td < -0.04)
vh_e    = _sigmoid(-dVH - vh_eff, scale=0.8)

# conv_e fires when dw_td < -0.04 but conv_raw = 0 (no crop/built data)
conv_raw = 0.0
conv_e = _sigmoid(conv_raw - 1.0, scale=2.0)  # sigmoid(-1, 2) = 0.119

signals = [
    ("ndvi_e",  None,    3.5, "pheno"),
    ("dw_z_e",  None,    3.0, "pheno"),
    ("dw_td_e", dw_td_e, 3.0, "inst"),
    ("dw_bg_e", dw_bg_e, 2.5, "inst"),
    ("conv_e",  conv_e,  1.5, "inst"),   # ← fires with ~0.12 even w/ no crop/built!
    ("vh_e",    vh_e,    1.5, "sar"),
    ("cusum_e", None,    0.8, "sar"),
]

print(f"\n  Evidence table:")
print(f"  {'Signal':10s}  {'ev':7s}  {'weight':6s}  {'ev×w':7s}")
print(f"  {'-'*40}")
wsum = wtot = 0.0
for name, ev, w, fam in signals:
    if ev is None:
        print(f"  {name:10s}  {'None':7s}  {w:6.1f}  EXCLUDED")
    else:
        print(f"  {name:10s}  {ev:7.4f}  {w:6.1f}  {ev*w:7.4f}")
        wsum += ev * w
        wtot += w

raw = wsum / wtot
print(f"\n  wsum={wsum:.4f}  wtot={wtot:.1f}  raw_score={raw:.4f}")
print(f"\n  KEY FINDING: conv_e={conv_e:.4f} with weight=1.5 is dragging score down!")
print(f"  Without conv_e: raw = {(wsum - conv_e*1.5)/(wtot - 1.5):.4f}")

# ── Score via actual function ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ACTUAL score_patch() OUTPUT")
print("=" * 70)
props = {
    "ndvi_zscore":     +0.83, "dw_trees_zscore": None,
    "dw_trees_delta":  -0.226, "dw_bare_delta":  +0.033,
    "dw_crops_delta":  None, "dw_built_delta":   None,
    "dVH": -0.036, "cusum_score": None, "dNDVI": None,
}
tier, conf, details = score_patch(props, rc, 0.65, season="DRY")
print(f"  tier={tier}, conf={conf}")
for k, v in details.items():
    print(f"    {k:12s} = {v}")

# ── All 4 patches ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ALL 4 CONFIRMED PATCHES (from GEE Feb-09 anchor)")
print("=" * 70)
patches = [
    # name,         ndvi_z, dw_td,  dw_bg
    ("Patch1 0.24ha", +1.411, -0.071, +0.025),
    ("Patch2 0.22ha", -0.766, -0.028, +0.033),
    ("Patch3 1.41ha", +1.425, -0.106, +0.018),
    ("Patch4 0.27ha", +0.832, -0.226, +0.033),
]
print(f"\n  {'Name':16s}  {'tier':4s}  {'conf':6s}  {'raw':6s}  {'dw_td_e':7s}  {'dw_bg_e':7s}  {'conv_e':6s}")
print(f"  {'-'*65}")
for pname, ndvi_z, dw_td, dw_bg in patches:
    p = {
        "ndvi_zscore": ndvi_z, "dw_trees_zscore": None,
        "dw_trees_delta": dw_td, "dw_bare_delta": dw_bg,
        "dw_crops_delta": None, "dw_built_delta": None,
        "dVH": None, "cusum_score": None, "dNDVI": None,
    }
    tier, conf, det = score_patch(p, rc, 0.65, season="DRY")
    print(f"  {pname:16s}  {tier:4d}  {conf:6.3f}  {det['raw']:6.3f}  "
          f"{det.get('dw_td_e') or 0:7.3f}  {det.get('dw_bg_e') or 0:7.3f}  "
          f"{'—':6s}")

# ── Tier threshold analysis ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TIER THRESHOLD FIRST-PRINCIPLES ANALYSIS")
print("=" * 70)
print("""
These values were SET HEURISTICALLY without a proper derivation. Here's the issue:

TIER 1 threshold = 0.78
  Natural baseline: raw=0.85+ requires NDVI z<-3 + DW instant + SAR all firing.
  With agreement bonus 1.25×: conf = 0.85 × 1.25 = 1.06 → capped at 0.95.
  Without bonus: raw=0.78 means 3 moderate signals all at 0.78 (very high).
  → Tier 1 at 0.78 is REASONABLE for "multi-sensor confirmed" events.

TIER 2 threshold = 0.55
  Natural baseline with DW instant only (-0.18pp): raw=0.552, conf=0.552.
  DW instant at -0.18pp is a STRONG felling signal (18pp drop in 2 days).
  Tier 2 currently BARELY catches this (conf=0.552 > 0.55 by 0.002).
  All 4 KML patches have DW drops of 7-22pp — strong physical evidence.
  → Tier 2 at 0.55 is TOO HIGH for DRY season single-sensor anchoring.
  → Should be 0.45-0.50 in DRY season (or a separate DRY-season threshold).

TIER 3 threshold = 0.38
  With no-signal minimum (wsum=0 → skip): properly returns tier=0.
  DW drop of -0.071 (Patch1): raw~0.35 → barely below tier-3.
  → Tier 3 at 0.38 seems appropriate as a "ground-verify when passing" alert.

ROOT PROBLEM: A single-number threshold is blind to the ecology.
  In DRY season with leaf-off: DW instant is the only reliable sensor.
  A 0.10pp drop is physically meaningful. A 0.071pp drop is borderline.
  The threshold should be assessed against the SENSOR, not the composite score.
""")
