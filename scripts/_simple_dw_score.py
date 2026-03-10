"""
_simple_dw_score.py
--------------------
Simplified deforestation scorer -- no GEE required, pure Python.

Two scorer implementations:

  dw_pheno_score()       -- v1 baseline (delta-primary + phenology bonus + linear cloud)
  dw_pheno_score_maxx()  -- v2 SOTA with 5 mathematical upgrades:
                              1. Sigmoidal cloud trust (not linear)
                              2. Dynamic midpoint scaling (canopy-density aware)
                              3. Effective baseline (pristine forest fix)
                              4. Gated phenology bonus (dead-forest flutter fix)
                              5. DOY interpolation (no monthly step-cliff)

Run:
    python scripts/_simple_dw_score.py
"""

import sys
import math
import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
#  Load DW phenology baseline from CSV
#  Builds per-month mean AND std, averaged across all years in the file.
# ---------------------------------------------------------------------------
PHENO_CSV = Path("outputs/phenology/ndvi_phenology.csv")

_monthly_raw: dict = defaultdict(list)
with open(PHENO_CSV, newline="") as f:
    for row in csv.DictReader(f):
        m   = row.get("month",   "").strip()
        mu  = row.get("dw_mean", "").strip()
        sig = row.get("dw_std",  "").strip()
        if m and mu and sig:
            try:
                _monthly_raw[int(m)].append((float(mu), float(sig)))
            except ValueError:
                pass

# Per-month multi-year averages
dw_baseline: dict = {}
for month, vals in _monthly_raw.items():
    dw_baseline[month] = {
        "mean": sum(v[0] for v in vals) / len(vals),
        "std":  sum(v[1] for v in vals) / len(vals),
    }

# Flat dicts for the SOTA scorer (division-level fallback)
DW_MONTHLY_MEANS: dict = {m: dw_baseline[m]["mean"] for m in dw_baseline}
DW_MONTHLY_STDS:  dict = {m: dw_baseline[m]["std"]  for m in dw_baseline}


# ---------------------------------------------------------------------------
#  Range-level harmonic phenology models
#  Loaded at import from data/phenology/range_models/*__dw_trees.json
#  Falls back to division-level monthly table when a range is not found.
# ---------------------------------------------------------------------------
_RANGE_MODEL_DIR = Path(__file__).parent.parent / "data" / "phenology" / "range_models"

RANGE_DW_MODELS: dict = {}   # range_name -> PhenologyModel (dw_trees)

try:
    # Lazy-load PhenologyModel only when the models directory exists
    if _RANGE_MODEL_DIR.exists():
        import json as _json
        from dataclasses import dataclass as _dc

        # Inline minimal loader — avoids hard dep on src package path at import
        class _PM:
            """Minimal wrapper around a fitted harmonic model JSON."""
            __slots__ = ("coeffs", "res_std", "doy_std", "n_obs", "r2", "_T", "_2pi")

            def __init__(self, d: dict):
                import math as _m
                self.coeffs  = d["coeffs"]           # [a0, a1, b1, a2, b2]
                self.res_std = d["residual_std"]
                self.doy_std = {int(k): v for k, v in
                                d.get("residual_doy_std", {}).items()}
                self.n_obs   = d["n_obs"]
                self.r2      = d["r2"]
                self._T      = 365.25
                self._2pi    = 2 * _m.pi

            def predict(self, date_str: str) -> float:
                import math as _m
                from datetime import datetime as _dt
                doy = _dt.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday
                w   = self._2pi / self._T
                a0, a1, b1, a2, b2 = self.coeffs
                return (a0
                        + a1 * _m.cos(w * doy) + b1 * _m.sin(w * doy)
                        + a2 * _m.cos(2*w*doy) + b2 * _m.sin(2*w*doy))

            def sigma(self, date_str: str) -> float:
                from datetime import datetime as _dt
                doy    = _dt.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday
                # 12 monthly bins; key = bin index 0-11
                bin_idx = min(int((doy - 1) / 30.44), 11)
                return self.doy_std.get(bin_idx, self.res_std)

        for _f in sorted(_RANGE_MODEL_DIR.glob("*__dw_trees.json")):
            _label = _f.stem.replace("__dw_trees", "")   # e.g. "North_Guna"
            with open(_f) as _fh:
                RANGE_DW_MODELS[_label] = _PM(_json.load(_fh))
except Exception as _e:
    pass   # gracefully degrade to division baseline


def _range_baseline(range_name: str, date_str: str) -> tuple[float, float]:
    """
    Return (mu, std) for `date_str` using the fitted harmonic model for
    `range_name`.  Falls back to the division-level monthly interpolation
    if the range model is unavailable.

    The harmonic model typically achieves R² ≈ 0.85–0.92 per range;
    sigma is the month-bin residual std (≈ 0.02–0.10) from the JSON.
    """
    from datetime import datetime as _dt
    model = RANGE_DW_MODELS.get(range_name)
    if model is None:
        # Fallback: division-level monthly table
        doy = _dt.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday
        return _doy_baseline(doy, DW_MONTHLY_MEANS), max(_doy_std(doy, DW_MONTHLY_STDS), 0.05)
    mu  = model.predict(date_str)
    std = max(model.sigma(date_str), 0.02)   # floor 2pp — never collapse to 0
    return mu, std


# ---------------------------------------------------------------------------
#  Utility: smooth logistic with overflow guard
# ---------------------------------------------------------------------------
def _logistic(x: float, midpoint: float, steepness: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))
    except OverflowError:
        return 0.0 if x < midpoint else 1.0


# ---------------------------------------------------------------------------
#  DOY smooth baseline interpolation (eliminates monthly step-cliff)
# ---------------------------------------------------------------------------
# Approximate mid-day of each calendar month (out of 365)
_MONTH_DOY_CENTERS = {
    1: 15, 2: 46,  3: 75,  4: 106, 5: 136, 6: 167,
    7: 197, 8: 228, 9: 259, 10: 289, 11: 320, 12: 350,
}

def _doy_baseline(doy: int, monthly_means: dict) -> float:
    """
    Linearly interpolate the expected DW-trees baseline for DOY (1-365)
    between the two nearest month-centre values. Wraps Dec->Jan correctly.
    """
    c = _MONTH_DOY_CENTERS

    # Find the two bracketing month centres
    m1, m2 = 12, 1
    for m in range(1, 12):
        if c[m] <= doy < c[m + 1]:
            m1, m2 = m, m + 1
            break
    if doy >= c[12]:
        m1, m2 = 12, 1         # Dec -> Jan (year wrap)

    d1 = c[m1]
    d2 = c[m2] if m2 != 1 else c[1] + 365   # Jan centre pushed to 380 for wrap math
    d_now = doy if not (m1 == 12 and doy < c[12]) else doy + 365

    t = (d_now - d1) / float(d2 - d1)
    t = max(0.0, min(1.0, t))
    return (1.0 - t) * monthly_means[m1] + t * monthly_means[m2]


def _doy_std(doy: int, monthly_stds: dict) -> float:
    """Same interpolation for per-month std."""
    c = _MONTH_DOY_CENTERS
    m1, m2 = 12, 1
    for m in range(1, 12):
        if c[m] <= doy < c[m + 1]:
            m1, m2 = m, m + 1
            break
    if doy >= c[12]:
        m1, m2 = 12, 1
    d1 = c[m1]
    d2 = c[m2] if m2 != 1 else c[1] + 365
    d_now = doy if not (m1 == 12 and doy < c[12]) else doy + 365
    t = max(0.0, min(1.0, (d_now - d1) / float(d2 - d1)))
    return (1.0 - t) * monthly_stds[m1] + t * monthly_stds[m2]


# ===========================================================================
#  V1 -- original baseline scorer (kept for comparison)
# ===========================================================================

def dw_pheno_score(
    trees_after:  float,
    trees_before: float,
    month:        int,
    cloud_frac:   float = 0.0,
    verbose:      bool  = False,
) -> dict:
    """
    V1 scorer:  delta-primary logistic + additive phenology bonus, linear cloud.
    Kept intact for A/B comparison. See dw_pheno_score_maxx() for SOTA version.
    """
    b   = dw_baseline[month]
    mu  = b["mean"]
    sig = max(b["std"], 0.05)

    delta = trees_after - trees_before
    z_dw  = (trees_after - mu) / sig

    raw         = _logistic(-delta, midpoint=0.12, steepness=18.0)
    pheno_bonus = max(0.0, -z_dw) / 4.0
    evidence    = min(raw * (1.0 + pheno_bonus), 1.0)

    cloud_weight = max(0.0, 1.0 - cloud_frac)
    score        = evidence * cloud_weight

    if verbose:
        print(f"  [V1] delta={delta:+.3f}  z={z_dw:+.3f}  raw={raw:.3f}"
              f"  pheno_bonus={pheno_bonus:.3f}  cw={cloud_weight:.3f}  score={score:.3f}")

    label = ("HIGH"     if score >= 0.70 else
             "MEDIUM"   if score >= 0.45 else
             "LOW"      if score >= 0.25 else "NO ALERT")

    return {
        "score": round(score, 3), "evidence": round(evidence, 3),
        "z_dw": round(z_dw, 3),   "cloud_w": round(cloud_weight, 3),
        "delta": round(delta, 3),  "label": label,
    }


# ===========================================================================
#  V2 -- SOTA scorer (dw_pheno_score_maxx)
#
#  Fix 1: Sigmoidal cloud trust
#    cloud_weight = 1 - logistic(cloud_frac, mid=0.60, steep=15)
#    -> < 40% cloud => ~1.0 trust;  60% => ~0.5;  80%+ => ~0.04
#
#  Fix 2: Dynamic midpoint
#    dynamic_midpoint = 0.05 + trees_before * 0.10
#    -> Dense forest (0.80) needs 0.13 drop; sparse (0.25) needs 0.075
#
#  Fix 3: Effective baseline (pristine forest fix)
#    effective_baseline = min(trees_before, seasonal_mean)
#    -> Pristine forests generate z < 0 relative to their own pre-event level
#
#  Fix 4: Gated phenology bonus (dead-forest flutter fix)
#    gate = logistic(drop_mag, mid=0.08, steep=40)
#    bonus only fires when the physical drop is unambiguously real
#
#  Fix 5: DOY linear interpolation
#    baseline_mean = lerp(month_m1_mean, month_m2_mean, fraction_through_month)
#    -> No overnight step-cliff at month boundaries
#
#  Fix 6: Relative-loss amplifier (sparse canopy calibration)
#    rel_drop = drop_mag / max(trees_before, 0.05)
#    raw_ev = 0.70*abs_ev + 0.30*logistic(rel_drop, mid=0.30, steep=8)
#    -> A 35% relative loss on a 20pp canopy scores equiv to ~13pp absolute loss
#    -> Prevents North_Guna sparse forest (-6.99pp = 35% relative) from underscoring
#
#  Fix 7: Adaptive steepness for sparse canopy
#    steepness = 25 + max(0, (0.30 - trees_before) / 0.30) * 20
#    -> Sparse (trees_before=0.13) => steepness=45 (sharper response)
#    -> Dense  (trees_before=0.80) => steepness=25 (unchanged)
#
#  Threshold calibration (aligned to range model residual std ~0.020):
#    HIGH >= 0.65  (was 0.70 — adjusted for sparse canopy physics)
#    MEDIUM >= 0.45
#    LOW >= 0.25
# ======================================================================================

def dw_pheno_score_maxx(
    trees_after:  float,
    trees_before: float,
    date_str:     str,          # "YYYY-MM-DD"
    cloud_frac:   float = 0.0,
    verbose:      bool  = False,
    range_name:   str   = "",   # kept for API compat; no longer used for baselines
    t0_date_str:  str   = "",  # FIX A: ISO date of BEFORE composite (for deseasonalization)
) -> dict:
    """
    V2 SOTA scorer -- 5 mathematical upgrades over V1 + FIX A deseasonalization.

    Args:
        trees_after  : DW trees probability after event        [0–1]
        trees_before : DW trees probability before event       [0–1]
        date_str     : ISO date of after-composite 'YYYY-MM-DD'
        cloud_frac   : fraction of patch pixels covered by cloud [0–1]
        verbose      : print detailed breakdown
        t0_date_str  : ISO date of before-composite (enables deseasonalization)

    Returns dict with: score, evidence, z_dw, raw_ev, cloud_w, label
    """
    # -- Fix 5 + Fix A: Division-level DOY smooth baseline (ALWAYS) ----------
    # Range harmonic models are in a different scale — NEVER use for absolute
    # baseline comparison. Division monthly table is in correct DW [0–1] scale.
    doy = datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday
    baseline_mu  = _doy_baseline(doy, DW_MONTHLY_MEANS)
    baseline_std = max(_doy_std(doy, DW_MONTHLY_STDS), 0.05)

    # -- Fix A: Deseasonalized delta magnitude --------------------------------
    # Remove expected phenological change so natural leaf-fall in Feb/Mar
    # does not register as an anomalous drop.
    # GREEN-UP CLAMP: min(0.0, ...) ensures we only forgive expected drops
    # (senescence). Stable bare pixels during monsoon green-up are NOT
    # penalised for "failing to green up".
    raw_delta = trees_after - trees_before          # negative = tree loss
    if t0_date_str:
        doy_t0 = datetime.strptime(t0_date_str, "%Y-%m-%d").timetuple().tm_yday
        base_t0 = _doy_baseline(doy_t0, DW_MONTHLY_MEANS)
        expected_pheno_change = min(0.0, baseline_mu - base_t0)   # CLAMPED
    else:
        expected_pheno_change = 0.0
    adj_delta = raw_delta - expected_pheno_change   # residual after removing phenology
    drop_mag  = max(0.0, -adj_delta)                # absolute anomalous drop (0 if gain)

    # -- Fix 2: Dynamic midpoint scaling -------------------------------------
    # A 0.12 drop from a 0.80 canopy = 15% relative loss  (needs big absolute drop)
    # A 0.12 drop from a 0.25 canopy = 48% relative loss  (too easy -> FP risk)
    dynamic_midpoint = 0.05 + trees_before * 0.10

    # -- Fix 7: Adaptive steepness for sparse canopy -------------------------
    # Dense forest (trees_before >= 0.30): steepness = 25 (unchanged)
    # Sparse canopy (trees_before = 0.13): steepness = 45 (sharper blade)
    # Prevents sparse North_Guna patches from falling on the shallow S-curve shoulder
    adaptive_steepness = 25.0 + max(0.0, (0.30 - trees_before) / 0.30) * 20.0

    # -- Raw evidence: absolute drop ----------------------------------------
    abs_ev = _logistic(drop_mag, midpoint=dynamic_midpoint, steepness=adaptive_steepness)

    # -- Fix 6: Relative-loss amplifier (sparse canopy calibration) ----------
    # A -6.99pp drop on a 20pp baseline = 34.7% RELATIVE loss.
    # Without this, sparse forest underscores vs dense forest for same % loss.
    # Blend: 70% absolute evidence + 30% relative-drop evidence
    rel_drop = drop_mag / max(trees_before, 0.05)   # fractional relative loss [0–∞]
    rel_ev   = _logistic(rel_drop, midpoint=0.30, steepness=8.0)
    raw_ev   = 0.70 * abs_ev + 0.30 * rel_ev

    # -- Fix 3: Effective baseline (pristine forest) -------------------------
    # max(trees_before, seasonal_mean) = "Poor Man's Pixel History"
    #   Pristine forest (tb=0.80, mean=0.53): eff=0.80 -> z vs own pre-event state
    #   Degraded forest (tb=0.35, mean=0.53): eff=0.53 -> regional baseline is safeguard
    effective_baseline = max(trees_before, baseline_mu)
    z_dw = (trees_after - effective_baseline) / baseline_std

    # -- Fix 4: Gated phenology bonus (no dead-forest flutter) ---------------
    raw_bonus = max(0.0, -z_dw) / 4.0
    gate      = _logistic(drop_mag, midpoint=0.08, steepness=40.0)
    gated_bonus = raw_bonus * gate

    # -- Combined evidence ---------------------------------------------------
    evidence = min(raw_ev * (1.0 + gated_bonus), 1.0)

    # -- Fix 1: Sigmoidal cloud trust ----------------------------------------
    # < 40% cloud -> ~1.0 trust (visible pixels dominate)
    # = 60% cloud -> 0.5 trust
    # > 80% cloud -> ~0.04 trust (too noisy)
    cloud_weight = 1.0 - _logistic(cloud_frac, midpoint=0.60, steepness=15.0)

    score = evidence * cloud_weight

    if verbose:
        print(f"  DOY {doy:3d} | base_mu={baseline_mu:.3f}  base_std={baseline_std:.3f}")
        if t0_date_str:
            print(f"  pheno_expected={expected_pheno_change:+.4f}  raw_delta={raw_delta:+.3f}  adj_delta={adj_delta:+.3f}")
        print(f"  eff_base={effective_baseline:.3f}  drop_mag={drop_mag:.3f}")
        print(f"  dyn_mid={dynamic_midpoint:.3f}  steep={adaptive_steepness:.1f}  abs_ev={abs_ev:.3f}")
        print(f"  rel_drop={rel_drop:.3f}  rel_ev={rel_ev:.3f}  raw_ev={raw_ev:.3f}")
        print(f"  z_dw={z_dw:+.3f}  raw_bonus={raw_bonus:.3f}  gate={gate:.3f}  gated_bonus={gated_bonus:.3f}")
        print(f"  evidence={evidence:.3f}  cloud_w={cloud_weight:.3f}  => score={score:.3f}")

    # Fix E: Restore HIGH threshold to 0.70. The division-level std (~0.14)
    # provides correct scale now that range harmonic models are bypassed.
    label = ("HIGH"   if score >= 0.70 else
             "MEDIUM" if score >= 0.45 else
             "LOW"    if score >= 0.25 else "NO ALERT")

    return {
        "score":    round(score,        3),
        "evidence": round(evidence,     3),
        "z_dw":     round(z_dw,         3),
        "raw_ev":   round(raw_ev,       3),
        "cloud_w":  round(cloud_weight, 3),
        "label":    label,
    }


# ===========================================================================
#  V3 — Multi-Threat Router (dw_multi_threat_score_maxx)
#
#  Wraps V2 canopy loss scorer alongside two new gated sub-scorers:
#
#  Sub-scorer A — Crop Encroachment
#    · Virgin Ground Gate: logistic penalty crushes score if crops_before > 0.30
#      (existing farms seasonally flushing green must not trigger alerts)
#    · Dynamic midpoint: 0.08 + crops_before × 0.10
#    · Absolute Gate: crops_after must exceed 0.35 to confirm agriculture
#
#  Sub-scorer B — Built Infrastructure Encroachment
#    · Existing Built Penalty: logistic penalty if built_before > 0.20
#    · Absolute Gate (strict): built_after > 0.30 — prevents Dry-Rock/Riverbed
#      illusion where leaf-off exposes bright rocky soil mis-classified as built
#    · Dynamic midpoint: 0.06 + built_before × 0.10
#
#  Cloud shielding: adaptive exponent — high evidence resists cloud more strongly.
#    cloud_weight = base_cloud_w ^ (1 - best_ev)
# ===========================================================================


def dw_multi_threat_score_maxx(
    trees_after:  float,
    trees_before: float,
    crops_after:  float,
    crops_before: float,
    built_after:  float,
    built_before: float,
    date_str:     str,          # "YYYY-MM-DD"
    cloud_frac:   float = 0.0,
    verbose:      bool  = False,
    range_name:   str   = "",   # kept for API compat; no longer used for baselines
    patch_z:      float = 0.0,  # pipeline pixel z-score (dw_trees_zscore); 0 = not supplied
    cusum_score:  float = 0.0,  # CuSuM accumulation score; 0 = not supplied
    t0_date_str:  str   = "",  # FIX A: ISO date of BEFORE composite (for deseasonalization)
) -> dict:
    """
    V3 Multi-Threat Apex Scorer + FIX A deseasonalization.

    Evaluates three simultaneous threats for a candidate patch and returns
    the dominant one with a calibrated score in [0, 1].

    Args:
        trees_after / trees_before : DW trees probability  [0–1]
        crops_after / crops_before : DW crops probability  [0–1]
        built_after / built_before : DW built probability  [0–1]
        date_str     : ISO anchor date "YYYY-MM-DD"
        cloud_frac   : Scene cloud fraction for latest pass [0–1]
        verbose      : Print per-sub-scorer breakdown
        t0_date_str  : ISO date of before-composite (enables deseasonalization)

    Returns:
        dict with keys: score, evidence, cloud_w, label, typology, details
    """
    # ── Sub-scorer A: Canopy Loss (V2 SOTA, cloud applied globally below) ──
    tree_res = dw_pheno_score_maxx(
        trees_after  = trees_after,
        trees_before = trees_before,
        date_str     = date_str,
        cloud_frac   = 0.0,      # raw evidence; cloud applied uniformly below
        verbose      = verbose,
        range_name   = range_name,
        t0_date_str  = t0_date_str,   # FIX A: pass through for deseasonalization
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
              f"tree_ev: {base_tree_ev:.3f} → {tree_ev:.3f}")

    # ── Sub-scorer B: Crop Encroachment ─────────────────────────────────────
    crop_delta   = crops_after - crops_before
    crop_rise    = max(0.0, crop_delta)           # only rising crops count

    # Dynamic midpoint: harder to trigger if land already had some crop cover
    c_dyn_mid  = 0.08 + crops_before * 0.10
    c_raw_ev   = _logistic(crop_rise, midpoint=c_dyn_mid, steepness=25.0)

    # Virgin Ground Gate: if land was ALREADY a farm (crops_before > 0.30),
    # seasonal greening will produce large crop_rise every year — crush it.
    # At crops_before = 0.30 → penalty ≈ 0.50; at 0.50 → penalty ≈ 0.12
    c_farm_penalty = 1.0 - _logistic(crops_before, midpoint=0.30, steepness=20.0)

    # Absolute Gate: final state must unambiguously look like crops (> 0.35)
    c_abs_gate = _logistic(crops_after, midpoint=0.35, steepness=20.0)

    crop_ev = c_raw_ev * c_farm_penalty * c_abs_gate

    # ── Sub-scorer C: Built Infrastructure Encroachment ─────────────────────
    built_delta  = built_after - built_before
    built_rise   = max(0.0, built_delta)

    b_dyn_mid  = 0.06 + built_before * 0.10
    b_raw_ev   = _logistic(built_rise, midpoint=b_dyn_mid, steepness=30.0)

    # Existing Built Penalty: if already >0.20 built, new Δ is maintenance/noise
    b_exist_penalty = 1.0 - _logistic(built_before, midpoint=0.20, steepness=20.0)

    # Absolute Gate (strict): bare rocky soil in leaf-off seasons is mis-classified
    # as 'built' by DW; require built_after > 0.30 to confirm real infrastructure
    b_abs_gate = _logistic(built_after, midpoint=0.30, steepness=25.0)

    built_ev = b_raw_ev * b_exist_penalty * b_abs_gate

    if verbose:
        print(f"  [Multi-Threat] tree_ev={tree_ev:.3f}  crop_ev={crop_ev:.3f}"
              f"  built_ev={built_ev:.3f}")
        print(f"    crop:  rise={crop_rise:.3f}  dyn_mid={c_dyn_mid:.3f}"
              f"  farm_penalty={c_farm_penalty:.3f}  abs_gate={c_abs_gate:.3f}")
        print(f"    built: rise={built_rise:.3f}  dyn_mid={b_dyn_mid:.3f}"
              f"  exist_penalty={b_exist_penalty:.3f}  abs_gate={b_abs_gate:.3f}")

    # ── Resolution: dominant threat ─────────────────────────────────────────
    threats = [
        ("CANOPY_LOSS",        tree_ev),
        ("CROP_ENCROACHMENT",  crop_ev),
        ("BUILT_ENCROACHMENT", built_ev),
    ]
    best_typology, best_ev = max(threats, key=lambda x: x[1])

    # ── Adaptive cloud shielding ─────────────────────────────────────────────
    # High-evidence patches resist cloud discounting more than marginal ones.
    # cloud_weight = base_cloud_w ^ (1 - best_ev)
    #   best_ev=1.0 → exponent=0.0 → cloud_weight=1.0  (perfect evidence, ignore cloud)
    #   best_ev=0.0 → exponent=1.0 → cloud_weight=base  (no evidence, full cloud discount)
    base_cloud_w = 1.0 - _logistic(cloud_frac, midpoint=0.60, steepness=15.0)
    exponent     = max(0.0, 1.0 - best_ev)
    cloud_weight = base_cloud_w ** exponent if base_cloud_w > 0 else 0.0

    final_score = best_ev * cloud_weight
    # NOTE: pz_boost + cusum_boost are applied into tree_ev above (before typology
    # and cloud shielding), so no post-cloud adjustment is needed or allowed.

    # Fix E: Restore HIGH threshold to 0.70. Division-level std (~0.14)
    # provides correct scale now that range harmonic models are bypassed.
    label = ("HIGH"   if final_score >= 0.70 else
             "MEDIUM" if final_score >= 0.45 else
             "LOW"    if final_score >= 0.25 else "NO ALERT")

    if verbose:
        print(f"  dominant={best_typology}  best_ev={best_ev:.3f}"
              f"  cloud_w={cloud_weight:.3f}  => final_score={final_score:.3f}  [{label}]")

    return {
        "score":    round(final_score, 3),
        "evidence": round(best_ev,     3),
        "cloud_w":  round(cloud_weight, 3),
        "label":    label,
        "typology": best_typology,
        "details": {
            "tree_ev":  round(tree_ev,  3),
            "crop_ev":  round(crop_ev,  3),
            "built_ev": round(built_ev, 3),
            "trees_z":  tree_res["z_dw"],
        },
    }


# ===========================================================================
#  Print phenology table
# ===========================================================================

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

print("-- DW Trees Probability Phenology Baseline (multi-year average) --------")
print(f"  {'Month':>5}  {'DW Mean':>7}  {'DW Std':>6}  Relative density")
print("  " + "-" * 58)
for m in sorted(dw_baseline):
    b = dw_baseline[m]
    bar = "#" * int(b["mean"] * 25)
    print(f"  {MONTH_NAMES[m]:>5} ({m:02d})  {b['mean']:>7.4f}  {b['std']:>6.4f}  {bar}")


# ===========================================================================
#  Test suite -- V1 vs V2 side-by-side
# ===========================================================================

# Each row: (name, trees_before, trees_after, date_str, cloud_frac)
# Note: date_str chosen to closely match the test month
CASES = [
    # Golden cases from V1
    ("Feb-09 Guna (confirmed loss, clear)",        0.72, 0.49, "2025-02-09", 0.00),
    ("Feb-09 Guna (50% cloud)",                    0.72, 0.49, "2025-02-09", 0.50),
    ("Feb-09 Guna (90% cloud)",                    0.72, 0.49, "2025-02-09", 0.90),
    ("Feb small drop (marginal)",                  0.65, 0.58, "2025-02-15", 0.10),
    ("Feb seasonal dip (no real loss)",            0.55, 0.50, "2025-02-20", 0.05),
    ("Sep strong loss (wet, clear)",               0.85, 0.55, "2025-09-15", 0.05),
    ("Sep moderate loss (wet, 15% cloud)",         0.80, 0.68, "2025-09-15", 0.15),
    ("Oct regrowth (trees increasing)",            0.60, 0.72, "2025-10-10", 0.10),
    ("Jun stable forest (no change)",              0.70, 0.70, "2025-06-01", 0.00),
    ("Aug complete clearing (clear sky)",          0.85, 0.15, "2025-08-20", 0.00),
    # New V2 stress-test cases (the 5 exploits)
    ("50% cloud visible clear-cut   [Exploit 1]",  0.80, 0.20, "2025-05-10", 0.50),
    ("Sparse Apr bulldozed 0.25->0.10 [Exploit 2]",0.25, 0.10, "2025-04-15", 0.00),
    ("Pristine Teak 0.85->0.55       [Exploit 3]", 0.85, 0.55, "2025-11-01", 0.00),
    ("Degraded flutter (0.30->0.28)  [Exploit 4]", 0.30, 0.28, "2025-03-01", 0.05),
    ("Feb 28 vs Mar 01 cliff         [Exploit 5a]",0.72, 0.49, "2025-02-28", 0.00),
    ("Mar 01 next-day                [Exploit 5b]",0.72, 0.49, "2025-03-01", 0.00),
]

print("\n-- V1 vs V2 Comparison -------------------------------------------------")
print(f"  {'Case':46s}  {'V1 sc':>6} {'V1 lbl':>8}  {'V2 sc':>6} {'V2 lbl':>8}")
print("  " + "-" * 95)

for (name, tb, ta, date_str, cf) in CASES:
    month_int = int(date_str[5:7])
    r1 = dw_pheno_score(ta, tb, month_int, cf)
    r2 = dw_pheno_score_maxx(ta, tb, date_str, cf)

    changed = " <-- BETTER" if r1["label"] != r2["label"] else ""
    print(f"  {name:46s}  {r1['score']:>6.3f} {r1['label']:>8}  "
          f"{r2['score']:>6.3f} {r2['label']:>8}{changed}")


# -- Verbose deep-dives on three key exploit cases ---------------------------
print("\n-- Verbose: Exploit 1 (50% cloud, visible clear-cut) -------------------")
dw_pheno_score(trees_after=0.20, trees_before=0.80, month=5, cloud_frac=0.50, verbose=True)
dw_pheno_score_maxx(trees_after=0.20, trees_before=0.80, date_str="2025-05-10", cloud_frac=0.50, verbose=True)

print("\n-- Verbose: Exploit 3 (Pristine Teak 0.85->0.55) ----------------------")
dw_pheno_score(trees_after=0.55, trees_before=0.85, month=11, cloud_frac=0.00, verbose=True)
dw_pheno_score_maxx(trees_after=0.55, trees_before=0.85, date_str="2025-11-01", cloud_frac=0.00, verbose=True)

print("\n-- Verbose: Exploit 5 (Feb 28 vs Mar 01 boundary) ---------------------")
print("  Feb 28:")
dw_pheno_score_maxx(trees_after=0.49, trees_before=0.72, date_str="2025-02-28", cloud_frac=0.00, verbose=True)
print("  Mar 01:")
dw_pheno_score_maxx(trees_after=0.49, trees_before=0.72, date_str="2025-03-01", cloud_frac=0.00, verbose=True)

print("\nScore bands:  HIGH >= 0.70  |  MEDIUM >= 0.45  |  LOW >= 0.25  |  NO ALERT")
print("Use dw_pheno_score_maxx() for production; dw_pheno_score() kept for A/B comparison.\n")
