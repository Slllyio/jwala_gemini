"""
GEE-SRC Rules Engine — Same-Season Rules-Cascade
=================================================
Detects forest loss in Guna Division's dry deciduous forest by fusing
five independent GEE signals, each calibrated to avoid phenological
false positives:

  Signal 1a — ΔNDVI       : same-season MODIS optical spectral drop (250m)
  Signal 1b — ΔDW_trees   : same-season Dynamic World tree probability drop (10m)
  Signal 2  — ΔNBR        : same-season vegetation structure (SWIR) drop
  Signal 3  — ΔVH         : instantaneous SAR backscatter drop
  Signal 4  — SAR-CuSum   : persistent multi-month SAR trend score

All signals use a ±1-year same-season baseline to cancel the natural
dry-season leaf-drop cycle (NDVI 0.88→0.33 in Guna, larger than the
logging signal itself).

DW locally-calibrated phenology insight
---------------------------------------
Dynamic World (GOOGLE/DYNAMICWORLD/V1) is trained globally on Sentinel-2.
However, in Guna's dry deciduous forest the DW 'trees' probability still
varies seasonally (~0.75 monsoon → ~0.50 dry season) because the S2
spectral inputs DW sees change with leaf phenology. Same-season comparison
cancels this local bias — any ΔDW_trees > threshold is attributable to
structural forest loss, not seasonal leaf-drop. This makes DW locally
calibrated for Guna without retraining the global model.

Thresholds calibrated from MODIS MOD13Q1 phenology data (2022–2025):
  - Dry-season trough (Apr): NDVI 0.33–0.39, σ≈0.025 across years
  - dNDVI threshold 0.15 ≈ 6σ above inter-annual noise → ~0% FPAR
  - dDW threshold 0.20 ≈ structural tree cover loss (not phenological)

Thresholds calibrated from MODIS MOD13Q1 phenology data (2022–2025):
  - Dry-season trough (Apr): NDVI 0.33–0.39, σ≈0.025 across years
  - dNDVI threshold 0.15 ≈ 6σ above inter-annual noise → ~0% FPAR

Usage
-----
    from src.inference.rules_engine import run_gee_rules_engine

    result = run_gee_rules_engine(
        beat_geom  = ee.Geometry,   # beat bounding geometry
        anchor_date= "2025-03-15",  # detection anchor (YYYY-MM-DD)
        cfg        = config_dict,   # parsed config.yaml
    )
    # result: dict with keys: tier, area_ha, confidence, geojson, signals

Integration
-----------
Called by generate_alerts.py when inference.detection_mode == "rules".
Replaces run_chip_inference() + fetch_live_chip() entirely.
Does NOT require a local chip download — runs fully server-side in GEE.
"""

import src.utils.proj_fix  # noqa: F401  (PROJ/GDAL env fix)

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)

# ── Default thresholds (overridden by config.yaml:inference.rules_engine) ──────
_NDVI_THRESH         = 0.15    # dNDVI fallback when no phenology model available
_NDVI_Z_THRESH       = -2.0   # phenology z-score threshold (negative = below expected)
_DW_THRESH           = 0.20    # ΔDW_trees: semantic tree cover loss (10m)
_NBR_THRESH          = 0.10    # dNBR: vegetation structure loss
_VH_DROP_DB          = 2.0     # instantaneous SAR VH drop (dB)
_CUSUM_THRESH        = 0.50    # CuSum score: fraction of max accumulation
_OPTICAL_WINDOW_DAYS = 45      # MODIS half-width of composite window (days)
_S2_WINDOW_DAYS      = 15      # Sentinel-2 half-width (faster, 5-10 day revisit)
_S1_WINDOW_DAYS      = 45      # SAR composite window
_MIN_AREA_HA         = 0.1     # minimum alert patch (ha)
_REQUIRE_TIER1       = False   # False = allow tier-2 alerts
# SAR fast-path: confidence for SAR-only (tier3) alerts
_SAR_DRY_CONF        = 0.65    # dry season (Nov–May): SAR is reliable
_SAR_WET_CONF        = 0.40    # monsoon (Jun–Oct): SAR noisier due to moisture
# RADD near-real-time alert window
_RADD_WINDOW_DAYS    = 30      # look back 30 days for RADD disturbance alerts
# DW instant delta (consecutive-pass) thresholds
_DW_INSTANT_WINDOW   = 30      # look back N days for 2 most-recent DW images
_DW_TREE_DROP_THRESH = 0.10    # trees prob drop ≥12pp between two passes = instant loss
_DW_CROP_RISE_THRESH = 0.10    # crops prob rise ≥10pp = agriculture encroachment
_DW_BUILT_RISE_THRESH= 0.05    # built prob rise ≥5pp = construction encroachment

# Default phenology model directory (relative to project root)
_DEFAULT_PHENOLOGY_MODEL_DIR = "data/phenology/range_models"


def _is_dry_season(anchor_date: str) -> bool:
    """
    Return True if anchor_date falls in dry season (November–May).
    Dry season → SAR is more reliable (no moisture-induced backscatter).
    Monsoon (June–October) → SAR noisier, lower confidence for SAR-only alerts.
    """
    month = datetime.strptime(anchor_date, "%Y-%m-%d").month
    return month not in range(6, 11)   # Jun=6 … Oct=10 are monsoon


def _load_rules_cfg(cfg: dict) -> dict:
    """
    Resolve rules engine thresholds from config.yaml with fallbacks
    to module-level defaults.
    """
    rk = cfg.get("inference", {}).get("rules_engine", {})
    return {
        "ndvi_thresh":         float(rk.get("ndvi_threshold",       _NDVI_THRESH)),
        "ndvi_z_thresh":       float(rk.get("ndvi_z_thresh",        _NDVI_Z_THRESH)),
        "dw_thresh":           float(rk.get("dw_threshold",         _DW_THRESH)),
        "nbr_thresh":          float(rk.get("nbr_threshold",        _NBR_THRESH)),
        "vh_drop_db":          float(rk.get("vh_drop_db",           _VH_DROP_DB)),
        "cusum_thresh":        float(rk.get("cusum_thresh",         _CUSUM_THRESH)),
        "opt_window":          int(  rk.get("optical_window_days",  _OPTICAL_WINDOW_DAYS)),
        "s2_window":           int(  rk.get("s2_window_days",        _S2_WINDOW_DAYS)),
        "s1_window":           int(  rk.get("s1_window_days",       _S1_WINDOW_DAYS)),
        "min_area_ha":         float(rk.get("min_area_ha",          _MIN_AREA_HA)),
        "require_tier1":       bool( rk.get("require_tier1",        _REQUIRE_TIER1)),
        "phenology_model_dir": str(  rk.get("phenology_model_dir",  _DEFAULT_PHENOLOGY_MODEL_DIR)),
        "sar_dry_conf":        float(rk.get("sar_dry_season_conf",  _SAR_DRY_CONF)),
        "sar_wet_conf":        float(rk.get("sar_wet_season_conf",  _SAR_WET_CONF)),
        "radd_window":         int(  rk.get("radd_window_days",     _RADD_WINDOW_DAYS)),
        # DW instant delta
        "dw_instant_window":   int(  rk.get("dw_instant_window_days", _DW_INSTANT_WINDOW)),
        "dw_tree_drop":        float(rk.get("dw_tree_drop_thresh",  _DW_TREE_DROP_THRESH)),
        "dw_crop_rise":        float(rk.get("dw_crop_rise_thresh",  _DW_CROP_RISE_THRESH)),
        "dw_built_rise":       float(rk.get("dw_built_rise_thresh", _DW_BUILT_RISE_THRESH)),
        "dw_bare_rise":        float(rk.get("dw_bare_rise_thresh",  0.08)),
    }


# ── Phenology model helpers ───────────────────────────────────────────────────

def _load_range_phenology_model(
    range_label: str,
    model_dir: str,
    variable: str = "ndvi",
) -> Optional[dict]:
    """
    Load a per-range harmonic phenology model from JSON.

    Parameters
    ----------
    range_label : str  e.g. "Binaganj"
    model_dir   : str  path to range_models/ directory
    variable    : str  "ndvi" or "dw_trees"

    Returns
    -------
    dict with keys: coeffs, residual_std, r2  — or None if not found.
    """
    safe = range_label.replace("/", "__").replace(" ", "_")
    path = Path(model_dir) / f"{safe}__{variable}.json"
    if not path.exists():
        log.warning(f"[phenology] Model not found: {path}")
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        log.warning(f"[phenology] Could not load {path}: {e}")
        return None


def _harmonic_predict(coeffs: list, doy: float) -> float:
    """
    Predict NDVI from 2nd-order Fourier harmonic coefficients.
    coeffs = [a0, a1, b1, a2, b2]
    """
    x = 2.0 * math.pi * doy / 365.0
    c = coeffs
    return c[0] + c[1]*math.cos(x) + c[2]*math.sin(x) + c[3]*math.cos(2*x) + c[4]*math.sin(2*x)


def build_ndvi_zscore_image(
    ndvi_current: Any,
    anchor_date:  str,
    model:        dict,
) -> Any:
    """
    Build a pixel-wise NDVI z-score image using the harmonic phenology model.

    z = (observed_NDVI - expected_NDVI_for_DOY) / sigma

    Negative z → NDVI is below seasonal expectation.
    z < -2.0 → anomaly at 97.5% confidence.

    Parameters
    ----------
    ndvi_current : ee.Image  Current NDVI image (0-1 scaled, band="NDVI")
    anchor_date  : str       "YYYY-MM-DD"
    model        : dict      Loaded phenology model JSON

    Returns
    -------
    ee.Image  z-score image (band renamed to "ndvi_zscore")
    """
    import ee

    dt  = datetime.strptime(anchor_date, "%Y-%m-%d")
    doy = dt.timetuple().tm_yday

    expected = _harmonic_predict(model["coeffs"], doy)
    sigma    = max(model["residual_std"], 1e-6)   # guard division by zero

    # Pixel-wise: (observed - expected) / sigma
    # expected and sigma are scalars → GEE constant images
    expected_img = ee.Image.constant(expected)
    zscore_img   = (ndvi_current.subtract(expected_img)
                                .divide(sigma)
                                .rename("ndvi_zscore"))

    log.info(f"  [phenology] DOY={doy}  expected_NDVI={expected:.4f}  "
             f"sigma={sigma:.4f}  z_thresh={_NDVI_Z_THRESH}")
    return zscore_img


# ── Signal 1+2: Optical change (NDVI + NBR) ───────────────────────────────────

def build_optical_composites(
    aoi: Any,
    anchor_date: str,
    window_days: int = _OPTICAL_WINDOW_DAYS,
) -> Tuple[Any, Any, Any, Any]:
    """
    Build same-season NDVI and NBR change images using MODIS MOD13Q1.

    Uses MOD13Q1 (250 m, 16-day composites, pre-computed NDVI) for the
    NDVI signal, and HLS S30 for NBR (NBR needs SWIR2 which MODIS MOD13Q1
    does not provide at 250 m; we use MODIS MOD09A1 500 m surface
    reflectance for NBR instead — same sensor family, consistent with
    the phenology data source).

    Returns
    -------
    ndvi_current, ndvi_baseline, nbr_current, nbr_baseline
        All are ee.Image objects clipped to AOI.
    """
    import ee

    centre   = datetime.strptime(anchor_date, "%Y-%m-%d")
    start_c  = (centre - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end_c    = (centre + timedelta(days=window_days)).strftime("%Y-%m-%d")

    baseline_centre = centre - timedelta(days=365)
    start_b  = (baseline_centre - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end_b    = (baseline_centre + timedelta(days=window_days)).strftime("%Y-%m-%d")

    # ── NDVI from MODIS MOD13Q1 (identical to phenology data source) ──────────
    def modis_ndvi_composite(start: str, end: str) -> Any:
        def mask_qa(img: Any) -> Any:
            qa   = img.select("SummaryQA")
            good = qa.lte(1)  # 0=good, 1=marginal
            return img.select("NDVI").multiply(0.0001).updateMask(good).rename("NDVI")

        return (ee.ImageCollection("MODIS/061/MOD13Q1")
                  .filterBounds(aoi)
                  .filterDate(start, end)
                  .map(mask_qa)
                  .median())

    ndvi_current  = modis_ndvi_composite(start_c, end_c)
    ndvi_baseline = modis_ndvi_composite(start_b, end_b)

    # ── NBR from MODIS MOD09A1 (500 m, 8-day surface reflectance) ─────────────
    # Band 2 = NIR (841-876 nm), Band 7 = SWIR2 (2105-2155 nm)
    def modis_nbr_composite(start: str, end: str) -> Any:
        def mask_and_nbr(img: Any) -> Any:
            qa         = img.select("StateQA")
            cloud_free = qa.bitwiseAnd(0b11).eq(0)    # cloud state bits 0-1
            nir        = img.select("sur_refl_b02").multiply(0.0001)
            swir2      = img.select("sur_refl_b07").multiply(0.0001)
            nbr        = nir.subtract(swir2).divide(nir.add(swir2)).rename("NBR")
            return nbr.updateMask(cloud_free)

        return (ee.ImageCollection("MODIS/061/MOD09A1")
                  .filterBounds(aoi)
                  .filterDate(start, end)
                  .map(mask_and_nbr)
                  .median())

    nbr_current  = modis_nbr_composite(start_c, end_c)
    nbr_baseline = modis_nbr_composite(start_b, end_b)

    return ndvi_current, ndvi_baseline, nbr_current, nbr_baseline


# ── Signal 1c: Sentinel-2 NDVI fast path (±15-day, 10m, primary source) ──────

def build_s2_ndvi_composite(
    aoi: Any,
    anchor_date: str,
    window_days: int = _S2_WINDOW_DAYS,
) -> Optional[Tuple[Any, Any]]:
    """
    Build same-season Sentinel-2 NDVI composites using S2_SR_HARMONIZED.

    Advantages over MODIS:
      • 5-10 day revisit (vs 16-day MODIS)
      • 10m resolution (vs 250m MODIS)
      • ±15 day window sufficient (faster lag)

    Cloud masking uses the Scene Classification Layer (SCL band):
      SCL 4 = vegetation, 5 = bare soil, 6 = water,
      SCL 8/9/10/11 = clouds/cirrus/shadow → masked out.

    Returns None if fewer than 2 valid S2 scenes exist in the window
    (so caller can fall back to MODIS).

    Returns
    -------
    (s2_ndvi_current, s2_ndvi_baseline) : Tuple[ee.Image, ee.Image]
        NDVI images, band renamed to 'NDVI', scaled 0–1.
    OR None if insufficient S2 coverage.
    """
    import ee

    centre   = datetime.strptime(anchor_date, "%Y-%m-%d")
    start_c  = (centre - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end_c    = (centre + timedelta(days=window_days)).strftime("%Y-%m-%d")

    baseline_centre = centre - timedelta(days=365)
    start_b  = (baseline_centre - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end_b    = (baseline_centre + timedelta(days=window_days)).strftime("%Y-%m-%d")

    def mask_s2_clouds(img: Any) -> Any:
        """Mask clouds/shadows using S2 SCL band."""
        scl = img.select("SCL")
        # Keep: 4=vegetation, 5=bare, 6=water, 2=dark area, 3=shadow edge
        good_mask = (scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
                       .Or(scl.eq(2)).Or(scl.eq(3)))
        # Reflectance scale: 10000 → 0-1
        nir  = img.select("B8").multiply(0.0001)
        red  = img.select("B4").multiply(0.0001)
        ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
        return ndvi.updateMask(good_mask)

    def s2_ndvi_composite(start: str, end: str) -> Any:
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                 .filterBounds(aoi)
                 .filterDate(start, end)
                 .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
                 .map(mask_s2_clouds))
        return col

    col_c = s2_ndvi_composite(start_c, end_c)
    col_b = s2_ndvi_composite(start_b, end_b)

    # Check that both windows have at least 2 valid scenes
    try:
        n_c = col_c.size().getInfo()
        n_b = col_b.size().getInfo()
    except Exception:
        return None

    if n_c < 2 or n_b < 2:
        log.info(f"    [S2 NDVI] Insufficient scenes: current={n_c}, baseline={n_b} "
                 f" — falling back to MODIS")
        return None

    log.info(f"    [S2 NDVI] {n_c} current + {n_b} baseline scenes (window=±{window_days}d)")
    return col_c.median(), col_b.median()


# ── Signal 1b: Dynamic World tree probability (same-season, locally calibrated) ─

def build_dw_trees_composite(
    aoi: Any,
    anchor_date: str,
    window_days: int = _OPTICAL_WINDOW_DAYS,
) -> Tuple[Any, Any]:
    """
    Build same-season Dynamic World tree probability change image.

    Dynamic World (GOOGLE/DYNAMICWORLD/V1) provides per-pixel 'trees'
    probability at 10m from Sentinel-2.  Although globally trained, the
    model still sees seasonal variation in Guna's dry deciduous forest
    because the S2 spectral inputs change with leaf phenology.  By
    comparing the same calendar month in baseline vs. current year we
    cancel this phenological drift entirely, making any residual drop
    attributable to structural tree loss only.

    Parameters
    ----------
    aoi         : ee.Geometry
    anchor_date : str, "YYYY-MM-DD" — detection date
    window_days : int — half-width of composite window

    Returns
    -------
    (dw_current, dw_baseline) : ee.Image objects, band = 'trees' (0..1)
    """
    import ee

    centre   = datetime.strptime(anchor_date, "%Y-%m-%d")
    start_c  = (centre - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end_c    = (centre + timedelta(days=window_days)).strftime("%Y-%m-%d")

    baseline_centre = centre - timedelta(days=365)
    start_b  = (baseline_centre - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end_b    = (baseline_centre + timedelta(days=window_days)).strftime("%Y-%m-%d")

    def dw_composite(start: str, end: str) -> Any:
        return (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                  .filterBounds(aoi)
                  .filterDate(start, end)
                  .select("trees")   # probability 0..1
                  .median())

    dw_current  = dw_composite(start_c, end_c)
    dw_baseline = dw_composite(start_b, end_b)
    return dw_current, dw_baseline


# ── Signal 1c: DW Instant Delta — consecutive-pass classes (trees/crops/built) ─

_DW_CLOUD_THRESH = 25   # max S2 CLOUDY_PIXEL_PERCENTAGE allowed per DW image

def build_dw_instant_delta(
    aoi: Any,
    anchor_date: str,
    window_days: int = _DW_INSTANT_WINDOW,
    cloud_thresh: int = _DW_CLOUD_THRESH,
) -> Optional[dict]:
    """
    Compute pixel-wise DW probability deltas between the two most-recent
    cloud-clean Sentinel-2-derived Dynamic World images within *window_days*.

    Cloud quality gating
    --------------------
    DW probabilities are unreliable on cloudy S2 images: cloud shadow and
    thin cirrus suppress the \u2018trees\u2019 class and mimic deforestation.
    We join DW images against S2_SR_HARMONIZED on system:index and keep
    only granules where CLOUDY_PIXEL_PERCENTAGE < cloud_thresh (default 25%).
    If the most-recent image is cloudy it is skipped; the next clean older
    image is used instead.

    Three change classes (latest_clean - prev_clean):
      trees_delta  — negative = canopy removed
      crops_delta  — positive = agricultural encroachment
      built_delta  — positive = construction encroachment

    Returns None if fewer than 2 cloud-clean DW images exist in window.
    """
    import ee

    centre   = datetime.strptime(anchor_date, "%Y-%m-%d")
    start_dt = centre - timedelta(days=window_days)
    start    = start_dt.strftime("%Y-%m-%d")
    end      = (centre + timedelta(days=1)).strftime("%Y-%m-%d")  # inclusive

    # ── Step 1: cloud-OK S2 images for this AOI & window ───────────────────
    s2_ok = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(aoi)
               .filterDate(start, end)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh)))

    # ── Step 2: inner join DW ↔ S2 on system:index ─────────────────────
    # DW image IDs match S2 granule IDs exactly.  Join keeps only DW
    # images that have a corresponding cloud-OK S2 granule.
    join_filter = ee.Filter.equals(
        leftField  = "system:index",
        rightField = "system:index",
    )
    joined = ee.Join.inner("dw_img", "s2_img").apply(
        primary   = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                       .filterBounds(aoi)
                       .filterDate(start, end)),
        secondary = s2_ok,
        condition = join_filter,
    )

    # Cloud-OK DW images: copy CLOUDY_PIXEL_PERCENTAGE from S2 side
    # into each DW image so it is retrievable later via toDictionary.
    def _attach_cloud_prop(feature):
        dw_image  = ee.Image(feature.get("dw_img"))
        s2_image  = ee.Image(feature.get("s2_img"))
        cloud_pct = s2_image.get("CLOUDY_PIXEL_PERCENTAGE")
        return dw_image.set("CLOUDY_PIXEL_PERCENTAGE", cloud_pct)

    cloud_ok_dw = (ee.ImageCollection(joined.map(_attach_cloud_prop))
                   .select(["trees", "crops", "built", "bare"])
                   .sort("system:time_start"))

    try:
        n = cloud_ok_dw.size().getInfo()
    except Exception:
        return None

    if n < 2:
        log.info(f"    [DW instant] Only {n} cloud-clean image(s) in last "
                 f"{window_days}d (cloud_thresh={cloud_thresh}%) — skipping")
        return None

    # Two most-recent cloud-clean images
    img_list   = cloud_ok_dw.toList(n)
    img_prev   = ee.Image(img_list.get(n - 2))
    img_latest = ee.Image(img_list.get(n - 1))

    # ── Step 3: retrieve date + cloud% metadata for audit trail ──────────
    try:
        info_prev   = img_prev.toDictionary(
            ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"]
        ).getInfo()
        info_latest = img_latest.toDictionary(
            ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"]
        ).getInfo()
        t_prev         = datetime.utcfromtimestamp(
            info_prev["system:time_start"] / 1000).strftime("%Y-%m-%d")
        t_latest       = datetime.utcfromtimestamp(
            info_latest["system:time_start"] / 1000).strftime("%Y-%m-%d")
        cloud_prev_pct   = round(info_prev.get("CLOUDY_PIXEL_PERCENTAGE", -1), 1)
        cloud_latest_pct = round(info_latest.get("CLOUDY_PIXEL_PERCENTAGE", -1), 1)
        interval = (datetime.strptime(t_latest, "%Y-%m-%d") -
                    datetime.strptime(t_prev,   "%Y-%m-%d")).days
    except Exception:
        t_prev = t_latest = "?"
        cloud_prev_pct = cloud_latest_pct = -1
        interval = -1

    log.info(f"    [DW instant] Pass N-1: {t_prev} (☁ {cloud_prev_pct}%), "
             f"Pass N: {t_latest} (☁ {cloud_latest_pct}%), Δ={interval}d")

    return {
        "trees_delta":       img_latest.select("trees").subtract(
                                 img_prev.select("trees")
                             ).rename("dw_trees_delta"),           # neg = loss
        "crops_delta":       img_latest.select("crops").subtract(
                                 img_prev.select("crops")
                             ).rename("dw_crops_delta"),           # pos = encroachment
        "built_delta":       img_latest.select("built").subtract(
                                 img_prev.select("built")
                             ).rename("dw_built_delta"),           # pos = encroachment
        "bare_delta":        img_latest.select("bare").subtract(
                                 img_prev.select("bare")
                             ).rename("dw_bare_delta"),            # pos = bare soil rise (felling proxy)
        "dw_trees_current":  img_latest.select("trees").rename("dw_trees_now"),
        "interval_days":     interval,
        "date_prev":         t_prev,
        "date_latest":       t_latest,
        "cloud_prev_pct":    cloud_prev_pct,
        "cloud_latest_pct":  cloud_latest_pct,
        "n_cloud_ok":        n,
    }


# ── Signal 3+4: SAR signals (ΔVH + CuSum) ─────────────────────────────────────

def build_s1_delta(
    aoi: Any,
    anchor_date: str,
    window_days: int = _S1_WINDOW_DAYS,
) -> Any:
    """
    Build instantaneous SAR VH change image.

    Returns an ee.Image of (baseline_VH - current_VH) in linear power
    units (already converted to dB by the caller if needed).  Positive
    values = VH drop = structural change.

    Uses a ±1-year same-season offset identical to the optical signals.
    """
    import ee

    centre   = datetime.strptime(anchor_date, "%Y-%m-%d")
    start_c  = (centre - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end_c    = (centre + timedelta(days=window_days)).strftime("%Y-%m-%d")

    baseline_centre = centre - timedelta(days=365)
    start_b  = (baseline_centre - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end_b    = (baseline_centre + timedelta(days=window_days)).strftime("%Y-%m-%d")

    def s1_composite(start: str, end: str) -> Any:
        return (ee.ImageCollection("COPERNICUS/S1_GRD")
                  .filterBounds(aoi)
                  .filterDate(start, end)
                  .filter(ee.Filter.eq("instrumentMode", "IW"))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                  .select("VH")
                  .median())

    vh_current  = s1_composite(start_c, end_c)
    vh_baseline = s1_composite(start_b, end_b)

    # Returns: baseline - current (positive → VH drop → canopy removed)
    # Both are already in dB (S1_GRD is stored as dB)
    return vh_baseline.subtract(vh_current).rename("dVH")


def build_cusum_score(
    aoi: Any,
    anchor_date: str,
    n_fortnights: int = 26,          # 26 × 14 days = 1 year lookback
    baseline_n: int = 20,            # first 20 fortnights used as baseline
    sensitivity: float = 1.5,
) -> Any:
    """
    Build a pixel-wise CuSum score image from a 1-year S1 VH time-series.

    Algorithm (RADD-inspired, calibrated for dry deciduous forest):
    1. Build a stack of 26 biweekly VH composites over 1 year
    2. Compute per-pixel mean from the first 20 fortnights (baseline)
    3. For each subsequent observation: penalise if below mean
       cusum[t] = max(0, cusum[t-1] + (mean - obs) * sensitivity)
    4. Normalise: score = max_cusum / (n_obs × sensitivity × mean_vh_magnitude)

    Score ranges [0, 1]; >0.5 indicates a persistent SAR loss trend.
    """
    import ee

    centre = datetime.strptime(anchor_date, "%Y-%m-%d")

    # Build list of fortnightly composite images (oldest first)
    composites = []
    for i in range(n_fortnights, 0, -1):
        t_centre    = centre - timedelta(days=i * 14)
        t_start     = (t_centre - timedelta(days=7)).strftime("%Y-%m-%d")
        t_end       = (t_centre + timedelta(days=7)).strftime("%Y-%m-%d")
        # Always ensure at least one valid VH-band image exists for this window.
        # If S1 has no data (empty orbit), the fallback 0-dB image is used.
        fallback_col = ee.ImageCollection([ee.Image.constant(0).rename("VH").float()])
        s1_col = (ee.ImageCollection("COPERNICUS/S1_GRD")
                    .filterBounds(aoi)
                    .filterDate(t_start, t_end)
                    .filter(ee.Filter.eq("instrumentMode", "IW"))
                    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                    .select("VH"))
        composite = fallback_col.merge(s1_col).median().rename("VH")
        composites.append(composite)

    # Historical baseline: mean of the first baseline_n composites
    baseline_col  = ee.ImageCollection(composites[:baseline_n])
    baseline_mean = baseline_col.mean().rename("baseline_mean")

    # CuSum accumulation: iterate over the remaining composites
    # We implement CuSum using ee.List iteration (server-side)
    monitoring_imgs = composites[baseline_n:]
    n_monitor       = len(monitoring_imgs)

    if n_monitor == 0:
        # No monitoring period available — return zero score
        return baseline_mean.multiply(0).rename("cusum_score")

    # Stack monitoring images into a collection and iterate
    monitoring_stack = ee.ImageCollection(monitoring_imgs).toBands()

    # Use ee.List.iterate to compute running CuSum per pixel
    # State = ee.Image of current cusum accumulation
    init_cusum = baseline_mean.multiply(0).rename("cusum")

    def accumulate(img_band: Any, state: Any) -> Any:
        """Single CuSum step executed server-side via ee.List.iterate."""
        state  = ee.Image(state)
        # Extract this observation band
        obs    = ee.Image(img_band).rename("obs")
        deficit = baseline_mean.subtract(obs).multiply(sensitivity)
        new_cusum = state.add(deficit).max(ee.Image.constant(0))
        # Track maximum achieved
        prev_max = state.select("cusum")
        new_max  = new_cusum.max(prev_max).rename("cusum")
        return new_max

    # We'll do a simpler but equivalent computation:
    # For each monitoring image, compute (mean - obs)*sensitivity and sum positives
    # This is a simplified CuSum approximation practical for GEE
    positive_deficits = []
    for obs_img in monitoring_imgs:
        deficit = baseline_mean.subtract(obs_img.rename("VH").rename("baseline_mean")).multiply(sensitivity)
        positive_deficits.append(deficit.max(ee.Image.constant(0)))

    cusum_sum = ee.ImageCollection(positive_deficits).sum()

    # Normalise: max possible = n_monitor × sensitivity × |typical VH|
    # Typical VH in forests: ~-10 dB.  In dB space, deficit can be ~3-5 dB.
    # Normalise by n_monitor × sensitivity × 5.0 (conservative magnitude)
    max_possible = n_monitor * sensitivity * 5.0
    cusum_score  = cusum_sum.divide(max_possible).clamp(0, 1).rename("cusum_score")

    return cusum_score


# ── Signal 5: RADD Near-Real-Time SAR Disturbance Alerts ──────────────────

def build_radd_flag(
    aoi: Any,
    anchor_date: str,
    window_days: int = _RADD_WINDOW_DAYS,
) -> Optional[Any]:
    """
    Check for RADD (Radar for Detecting Deforestation) alerts in the AOI.

    RADD uses Sentinel-1 SAR to detect forest disturbances near-real-time
    (6-12 day lag, all-weather). Published by Wageningen University & Research
    on GEE as 'projects/radar-wur/raddalert/v1'.

    Alert bands:
      'alert' : 2=unconfirmed, 3=confirmed disturbance
      'date'  : days since 2018-12-31 (Julian-like)

    Returns a binary ee.Image (1=alert, 0=no alert) or None if collection
    unavailable.
    """
    import ee

    try:
        centre    = datetime.strptime(anchor_date, "%Y-%m-%d")
        start_dt  = centre - timedelta(days=window_days)

        # RADD date encoding: days since 2018-12-31
        epoch     = datetime(2018, 12, 31)
        start_day = (start_dt - epoch).days
        end_day   = (centre   - epoch).days

        # Filter to alert images only (the collection also contains
        # 'forest_baseline' images without the 'alert' band, which cause
        # mosaic() to fail).
        radd = (ee.ImageCollection("projects/radar-wur/raddalert/v1")
                  .filterBounds(aoi)
                  .filter(ee.Filter.eq("Type", "Alert"))
                  .select(["alert", "date"]))

        # Guard: RADD only covers humid tropical forests; for other
        # biomes the filtered collection is empty and mosaic() returns
        # a constant image with no usable bands.
        try:
            radd_size = radd.size().getInfo()
        except Exception:
            radd_size = 0
        if radd_size == 0:
            log.info("    [RADD] No coverage in AOI (outside humid tropics) — skipping")
            return None, 0

        # Get the latest composite
        latest = radd.mosaic()
        alert_band = latest.select("alert")
        date_band  = latest.select("date")

        # Mask to: alert confirmed (3) or unconfirmed (2) in window
        in_window   = date_band.gte(start_day).And(date_band.lte(end_day))
        radd_binary = alert_band.gte(2).And(in_window).rename("radd_alert")

        # Check if any alert pixels exist in AOI
        n_alerts = (radd_binary.reduceRegion(
            reducer  = ee.Reducer.sum(),
            geometry = aoi,
            scale    = 10,
            maxPixels= 1e7,
            bestEffort=True,
        ).getInfo().get("radd_alert", 0) or 0)

        log.info(f"    [RADD] Pixels flagged in last {window_days}d: {n_alerts}")
        return radd_binary, int(n_alerts)

    except Exception as e:
        log.warning(f"    [RADD] Collection unavailable: {e}")
        return None, 0


# ── Signal fusion + post-processing ────────────────────────────────────────────

def _forest_baseline_mask(aoi: Any, min_treecover: int = 5) -> Any:
    """
    Binary mask: pixels that were confirmed forest in the baseline year.
    Uses Hansen GFC 2024 treecover2000 ≥ min_treecover%.
    Only alert pixels that were in forest in the baseline.
    """
    import ee
    hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    return hansen.select("treecover2000").gte(min_treecover)


def _morphological_clean(mask: Any, radius_px: int = 1) -> Any:
    """
    Morphological opening: remove isolated single pixels while keeping
    connected patches.  radius_px=1 removes sub-pixel noise.
    """
    import ee
    kernel  = ee.Kernel.circle(radius_px)
    eroded  = mask.focal_min(kernel=kernel)
    dilated = eroded.focal_max(kernel=kernel)
    return dilated


def fuse_signals(
    aoi: Any,
    dndvi: Any,
    dnbr:  Any,
    dvh:   Any,
    cusum: Any,
    thresholds: dict,
    forest_mask: Any,
    dw_trees: Optional[Any] = None,
    ndvi_zscore: Optional[Any] = None,    # phenology z-score (NDVI model)
    dw_zscore:   Optional[Any] = None,    # phenology z-score (DW trees model)
    dw_instant:  Optional[dict] = None,  # output of build_dw_instant_delta()
) -> Tuple[Any, Any, Any]:
    """
    Combine all signals into tiered change masks.

    Signals
    -------
    1a  NDVI z-score (phenology-corrected, preferred) OR raw dNDVI (fallback)
    1b  ΔDW_trees year-on-year (semantic tree prob drop)
    1c  DW INSTANT delta — trees/crops/built between last two S2 passes
    1d  DW trees z-score (phenology-corrected DW trees probability)
    2   ΔNBR — vegetation structure loss (SWIR)
    3   ΔVH  — instantaneous SAR backscatter drop
    4   SAR-CuSum — persistent SAR trend score

    Alert Tiers
    -----------
    Tier 1  (conf=0.90): high-confidence — multi-signal optical+SAR agreement
    Tier 2  (conf=0.65): medium-confidence — optical+SAR or class-conversion
    Tier 3  (conf=variable): SAR-only (cloud-obscured optical)

    DW Instant Delta tiers (wired into Tier 1 / Tier 2 depending on certainty)
    ──────────────────────────────────────────────────────────────────────────
    Tier A (→ Tier 1, 0.90): DW trees z<-2 AND NDVI z<-2 (both optical agree)
    Tier B (→ Tier 1, 0.90): DW trees drop AND (crops rise OR built rise) + VH confirm
    Tier C (→ Tier 2, 0.72): DW trees drop AND VH drop (SAR confirms canopy loss)
    Tier D (→ Tier 2, 0.65): DW trees z<-2 alone (phenologically anomalous)
    """
    import ee

    # ── Optical NDVI gate (z-score or raw fallback) ──────────────────────────
    if ndvi_zscore is not None:
        ndvi_flag = ndvi_zscore.lt(thresholds["ndvi_z_thresh"])
        log.info(f"  [fuse] Using phenology z-score (thresh={thresholds['ndvi_z_thresh']:.1f})")
    else:
        ndvi_flag = dndvi.gt(thresholds["ndvi_thresh"])
        log.info(f"  [fuse] Using raw dNDVI fallback (thresh={thresholds['ndvi_thresh']:.2f})")

    nbr_flag   = dnbr.gt(thresholds["nbr_thresh"])
    vh_flag    = dvh.gt(thresholds["vh_drop_db"])
    cusum_flag = cusum.gt(thresholds["cusum_thresh"])

    # ── Year-on-year DW gate (Signal 1b, legacy) ─────────────────────────────
    if dw_trees is not None:
        dw_yoy_flag  = dw_trees.gt(thresholds["dw_thresh"])
        optical_gate = ndvi_flag.Or(dw_yoy_flag)
    else:
        optical_gate = ndvi_flag

    # ── DW Instant Delta tiers (Signal 1c + 1d) ───────────────────────────────
    # These are computed pixel-by-pixel from consecutive S2 passes
    dw_tree_drop_flag  = ee.Image.constant(0)  # default: no signal
    dw_crop_rise_flag  = ee.Image.constant(0)
    dw_built_rise_flag = ee.Image.constant(0)
    dw_z_flag          = ee.Image.constant(0)  # phenology z-score

    if dw_instant is not None:
        dw_tree_drop_flag  = dw_instant["trees_delta"].lt(
            -abs(thresholds["dw_tree_drop"])    # trees_delta is neg for drop
        )
        dw_crop_rise_flag  = dw_instant["crops_delta"].gt(
            thresholds["dw_crop_rise"]
        )
        dw_built_rise_flag = dw_instant["built_delta"].gt(
            thresholds["dw_built_rise"]
        )
        log.info(f"  [fuse] DW instant delta active "
                 f"(tree_drop>{thresholds['dw_tree_drop']:.2f}, "
                 f"crop_rise>{thresholds['dw_crop_rise']:.2f}, "
                 f"built_rise>{thresholds['dw_built_rise']:.2f})")

    if dw_zscore is not None:
        dw_z_flag = dw_zscore.lt(thresholds["ndvi_z_thresh"])  # same -2σ threshold
        log.info(f"  [fuse] DW trees z-score active (thresh={thresholds['ndvi_z_thresh']:.1f})")

    # ── Tier A: NDVI z + DW trees z — both phenology-corrected optical signals agree
    tier_A = ndvi_flag.And(dw_z_flag)    # strongest: independent confirmation

    # ── Tier B: instant DROP + class conversion + SAR — all three confirm
    conversion_flag = dw_crop_rise_flag.Or(dw_built_rise_flag)
    tier_B = dw_tree_drop_flag.And(conversion_flag).And(vh_flag)

    # ── Tier C: instant tree drop + SAR VH drop (fast detection, no phenology needed)
    tier_C = dw_tree_drop_flag.And(vh_flag)

    # ── Tier D: DW trees z-score alone (anomalous, single source)
    tier_D = dw_z_flag

    # ── Roll up into legacy Tier 1 / Tier 2 / Tier 3 ─────────────────────────
    # Tier A and B → Tier 1 level confidence (0.90)
    # Tier C and D → Tier 2 level confidence (0.65-0.72)
    dw_tier1_extra = tier_A.Or(tier_B)
    dw_tier2_extra = tier_C.Or(tier_D)

    # Legacy tiers (kept for SAR/NBR path)
    tier1 = optical_gate.And(nbr_flag).And(vh_flag).And(cusum_flag)
    tier2 = optical_gate.And(nbr_flag).And(vh_flag)
    tier3 = vh_flag.And(cusum_flag)

    # Merge DW instant tiers into the legacy structure
    tier1 = tier1.Or(dw_tier1_extra)
    tier2 = tier2.Or(dw_tier2_extra.And(tier1.Not()))  # avoid double-counting

    return tier1, tier2, tier3


def apply_post_processing(mask: Any) -> Any:
    """
    Morphological cleaning at native 10m scale.
    Replaces the old 250m connectedPixelCount which destroyed sub-hectare patches.
    A radius-1 opening removes single-pixel speckle without dissolving real patches.
    """
    import ee
    kernel = ee.Kernel.circle(radius=1)
    return mask.focal_min(kernel=kernel).focal_max(kernel=kernel)



# ── Patch-level helpers ───────────────────────────────────────────────────────

def build_candidate_mask(
    dndvi, ndvi_zscore, dw_instant, dw_zscore,
    dw_trees_delta, dvh, cusum, forest_mask, aoi, thresholds,
) -> Any:
    """
    Permissive union mask at 10m: any pixel where ANY signal exceeds ~60% of
    its alert threshold.  Cast wide; patches are scored individually afterwards.
    """
    import ee

    # Optical NDVI gate
    if ndvi_zscore is not None:
        opt = ndvi_zscore.lt(thresholds["ndvi_z_thresh"] * 0.75)
    elif dndvi is not None:
        opt = dndvi.gt(thresholds["ndvi_thresh"] * 0.5)
    else:
        opt = ee.Image.constant(0)

    # DW trees instant delta (10m, most responsive)
    if dw_instant is not None and "trees_delta" in dw_instant:
        dw_tree = dw_instant["trees_delta"].lt(-abs(thresholds["dw_tree_drop"]) * 0.6)
    else:
        dw_tree = ee.Image.constant(0)

    # DW trees phenology z-score
    dw_z = (dw_zscore.lt(thresholds["ndvi_z_thresh"] * 0.75)
            if dw_zscore is not None else ee.Image.constant(0))

    # DW year-on-year delta
    dw_yoy = (dw_trees_delta.gt(thresholds["dw_thresh"] * 0.6)
              if dw_trees_delta is not None else ee.Image.constant(0))

    # SAR VH drop
    sar = (dvh.gt(thresholds["vh_drop_db"] * 0.6)
           if dvh is not None else ee.Image.constant(0))

    candidate = opt.Or(dw_tree).Or(dw_z).Or(dw_yoy).Or(sar)
    if forest_mask is not None:
        candidate = candidate.updateMask(forest_mask)
    return candidate


def vectorize_patches(
    candidate_mask: Any,
    aoi:            Any,
    min_area_ha:    float = 0.2,
    scale:          int   = 10,
) -> Any:
    """
    Convert binary 10m candidate mask → patch polygons, filtering by minimum area.
    Returns an ee.FeatureCollection.
    """
    import ee

    vectors = candidate_mask.selfMask().reduceToVectors(
        reducer        = ee.Reducer.countEvery(),
        geometry       = aoi,
        scale          = scale,
        maxPixels      = 1e8,
        bestEffort     = True,
        geometryType   = "polygon",
        eightConnected = True,
    )

    def add_area(f):
        return f.set("area_ha", f.geometry().area(maxError=10).divide(10000))

    return (vectors.map(add_area)
                   .filter(ee.Filter.gte("area_ha", min_area_ha)))


def sample_patches(patch_fc: Any, signal_images: dict, scale: int = 10) -> Any:
    """
    For every patch polygon in patch_fc, compute mean of all signal images in
    ONE batch GEE server-side map() call — no per-patch round-trips.
    """
    import ee

    valid_images = [
        img.rename(name)
        for name, img in signal_images.items()
        if img is not None
    ]
    if not valid_images:
        return patch_fc

    multi_band = ee.Image.cat(valid_images)

    def sample_fn(feature):
        stats = multi_band.reduceRegion(
            reducer    = ee.Reducer.mean(),
            geometry   = feature.geometry(),
            scale      = scale,
            maxPixels  = 1e6,
            bestEffort = True,
        )
        return feature.set(stats)

    return patch_fc.map(sample_fn)


def _sigmoid(x: float, scale: float = 1.0) -> float:
    """Smooth step: 0.5 at x=0, ->1 for large +x, ->0 for large -x."""
    try:
        return 1.0 / (1.0 + math.exp(-scale * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# ── SOTA DW trees scorer helpers ─────────────────────────────────────────────
# Mirrors dw_pheno_score_maxx() from scripts/_simple_dw_score.py.
# Five mathematical upgrades over the old _sigmoid(-dw_td - threshold) approach:
#   Fix 1: Sigmoidal cloud trust   — <40% cloud = no penalty
#   Fix 2: Dynamic midpoint        — canopy-density-aware threshold
#   Fix 3: Effective baseline      — max(trees_before, baseline) for pristine forests
#   Fix 4: Gated phenology bonus   — bonus locked unless drop is real (>0.08)
#   Fix 5: DOY interpolation       — smooth month-center linear blend; no step-cliff
#
# Monthly DW baseline loaded once at module level from ndvi_phenology.csv.
# Falls back gracefully to the old sigmoid if props lack the required fields.
_DW_PHENO_MONTHLY_MEANS: dict = {}
_DW_PHENO_MONTHLY_STDS:  dict = {}

try:
    import csv as _csv
    from collections import defaultdict as _dd
    _pheno_csv = Path("outputs/phenology/ndvi_phenology.csv")
    _raw: dict = _dd(list)
    with open(_pheno_csv, newline="") as _f:
        for _row in _csv.DictReader(_f):
            _m   = _row.get("month",   "").strip()
            _mu  = _row.get("dw_mean", "").strip()
            _sig = _row.get("dw_std",  "").strip()
            if _m and _mu and _sig:
                try:
                    _raw[int(_m)].append((float(_mu), float(_sig)))
                except ValueError:
                    pass
    for _m, _vals in _raw.items():
        _DW_PHENO_MONTHLY_MEANS[_m] = sum(v[0] for v in _vals) / len(_vals)
        _DW_PHENO_MONTHLY_STDS[_m]  = sum(v[1] for v in _vals) / len(_vals)
    log.debug(f"[rules_engine] DW phenology baseline loaded: {len(_DW_PHENO_MONTHLY_MEANS)} months")
except Exception as _e:
    log.warning(f"[rules_engine] Could not load DW phenology baseline: {_e} — SOTA scorer disabled")

_DW_DOY_CENTERS = {
    1: 15, 2: 46,  3: 75,  4: 106, 5: 136, 6: 167,
    7: 197, 8: 228, 9: 259, 10: 289, 11: 320, 12: 350,
}

def _dw_doy_interp(doy: int, monthly: dict, fallback: float = 0.14) -> float:
    """Linear DOY interpolation between month-centre values (wraps Dec->Jan)."""
    if not monthly:
        return fallback
    c = _DW_DOY_CENTERS
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
    v1 = monthly.get(m1, fallback)
    v2 = monthly.get(m2, fallback)
    return (1.0 - t) * v1 + t * v2


def _dw_maxx_evidence(
    trees_after:  float,
    trees_before: float,
    date_str:     str,
    cloud_frac:   float = 0.0,
) -> float:
    """
    SOTA DW trees deforestation evidence in [0, 1].
    Drop-in upgrade for _sigmoid(-dw_td - threshold) in score_patch Signal 3.
    Returns 0.0..1.0 evidence (cloud-weighted).
    Returns None if baseline data unavailable (triggers old-sigmoid fallback).
    """
    if not _DW_PHENO_MONTHLY_MEANS:
        return None  # baseline not loaded — caller falls back to old sigmoid

    try:
        doy = datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday
    except ValueError:
        return None

    baseline_mu  = _dw_doy_interp(doy, _DW_PHENO_MONTHLY_MEANS)
    baseline_std = max(_dw_doy_interp(doy, _DW_PHENO_MONTHLY_STDS, 0.14), 0.05)

    delta    = trees_after - trees_before   # negative = loss
    drop_mag = max(0.0, -delta)

    # Fix 2: dynamic midpoint — dense canopy needs bigger absolute drop
    dynamic_midpoint = 0.05 + trees_before * 0.10

    # Raw evidence from drop magnitude (steepness=25 sharpens noise rejection)
    raw_ev = _sigmoid(drop_mag - dynamic_midpoint, scale=25.0)

    # Fix 3: effective baseline = max(trees_before, seasonal_mean)
    #   -> pristine forests measured vs own pre-event level, not diluted regional mean
    effective_baseline = max(trees_before, baseline_mu)
    z_dw = (trees_after - effective_baseline) / baseline_std

    # Fix 4: gated phenology bonus — bonus only fires for real drops (>0.08)
    raw_bonus = max(0.0, -z_dw) / 4.0
    gate      = _sigmoid(drop_mag - 0.08, scale=40.0)
    evidence  = min(raw_ev * (1.0 + raw_bonus * gate), 1.0)

    # Fix 1: sigmoidal cloud trust — <40% cloud = ~1.0 trust, 80%+ cloud = ~0.04
    cloud_weight = 1.0 - _sigmoid(cloud_frac - 0.60, scale=15.0)

    return evidence * cloud_weight


def score_patch(
    props:       dict,
    thresholds:  dict,
    t3_conf:     float,
    season:      str = "DRY",
    anchor_date: str = "",
) -> tuple:
    """
    Continuous weighted-evidence confidence scoring for selective tree
    felling in degraded dry deciduous forest (Guna, MP).

    Physics-grounded signal priorities for this ecosystem:
      1. NDVI phenology z-score       (w=3.5)  — primary; removes seasonal cycle
      2. DW trees phenology z-score   (w=3.0)  — primary; correlated with NDVI, discounted
      3. DW instant trees delta       (w=3.0)  — fast 2-5 day transition; semi-independent
      4. DW bare ground rise          (w=2.5)  — most specific felling proxy (stump/soil)
      5. DW crops/built conversion    (w=1.5)  — agricultural encroachment signal
      6. SAR VH drop                  (w=1.5)  — supplementary; weak in leaf-off dry season
      7. SAR CuSum                    (w=0.8)  — persistent/repeated felling only
      NBR: EXCLUDED — 250m resolution, dry-season leaf-water signal is near-zero

    Returns (tier: int, confidence: float, details: dict).
    """
    dv = props.get

    ndvi_z   = dv("ndvi_zscore")      # phenology z-score (neg = anomalously low)
    dndvi_v  = dv("dNDVI")            # fallback: raw NDVI change (pos = loss)
    dw_z     = dv("dw_trees_zscore")  # DW trees phenology z-score
    dw_td    = dv("dw_trees_delta")   # DW 2-5 day instant delta (neg = loss)
    dw_bg    = dv("dw_bare_delta")    # DW bare ground rise (pos = felling proxy)
    dw_crop  = dv("dw_crops_delta")   # DW crops rise (pos = encroachment)
    dw_blt   = dv("dw_built_delta")   # DW built rise (pos = encroachment)
    sar_dvh  = dv("dVH")             # SAR VH change; neg = canopy drop
    cusum_s  = dv("cusum_score")
    # dNBR deliberately excluded: 250m resolution = ~92% signal loss for sub-ha patches

    t = thresholds
    ndvi_z_t = t.get("ndvi_z_thresh",  -2.0)
    dw_td_t  = t.get("dw_tree_drop",    0.12)
    dw_bg_t  = t.get("dw_bare_rise",    0.08)
    dw_cr_t  = t.get("dw_crop_rise",    0.10)
    dw_bl_t  = t.get("dw_built_rise",   0.05)
    # SAR VH threshold: in DRY season (Feb-Apr) leaf-fall causes <1 dB drops
    # while real felling causes >=2 dB.  Use full 2.0 dB threshold in DRY to
    # avoid leaf-fall false positives; lower to 1.5 dB outside leaf-off season.
    if season == "DRY":
        vh_eff  = t.get("vh_drop_db", 2.0)          # full threshold: 2.0 dB
        cusum_t = t.get("cusum_thresh", 0.50)        # full threshold: 0.50
    else:
        vh_eff  = t.get("vh_drop_db", 2.0) * 0.75   # relaxed: 1.5 dB
        cusum_t = t.get("cusum_thresh", 0.50) * 0.80 # relaxed: 0.40

    # ── Per-signal evidence values in [0, 1] ─────────────────────────────────
    # evidence=1.0 -> signal strongly indicates canopy loss
    # evidence=0.5 -> signal exactly at threshold (uncertain)
    # evidence=0.0 -> signal contradicts canopy loss

    # Signal 1: NDVI z-score — primary (lower z = more anomalous = more evidence)
    # scale=1.5: at threshold (z_t - z = 0) evidence=0.5; 1 σ below threshold -> evidence=0.82
    # ── DRY-season None-exclusion (The Denominator Drag fix) ─────────────────
    # Physical basis: In Feb-Apr leaf-off deciduous forest, the 'phantom canopy'
    # effect means positive NDVI z-scores are EXPECTED after felling:
    #   - Shadow removal: bare stumps remove micro-shadows that suppressed NIR
    #   - Background dominance: dry grass/soil has equal or higher NDVI than
    #     shadowed bare-limb canopy
    # DW survives because it uses SWIR — woody biomass/lignin removal is visible
    # in SWIR even when NDVI is flat.
    #
    # WRONG FIX (old): ndvi_e = 0.5  → weight-3.5 phantom sensor drags raw_score
    # from ~0.60 → 0.48 ("Denominator Drag").
    # CORRECT FIX (new): ndvi_e = None → exclude from both numerator AND denominator.
    # raw_score then floats up organically to the DW anchor value.

    if ndvi_z is not None:
        if season == "DRY" and ndvi_z > -0.5:
            # Leaf-off: NDVI is physically blind to felling. Exclude entirely.
            ndvi_e = None
        else:
            ndvi_e = _sigmoid(ndvi_z_t - ndvi_z, scale=1.5)
    elif dndvi_v is not None:
        if season == "DRY" and dndvi_v > -0.05:
            ndvi_e = None
        else:
            ndvi_e = _sigmoid((dndvi_v - t.get("ndvi_thresh", 0.15)) * 8, scale=1.0)
    else:
        ndvi_e = None

    # Signal 2: DW trees z-score — correlated with NDVI, discounted weight
    if dw_z is not None:
        if season == "DRY" and dw_z > -0.5:
            # Same physics: seasonal model z-score is uninformative in leaf-off
            dw_z_e = None
        else:
            dw_z_e = _sigmoid(ndvi_z_t - dw_z, scale=1.5)
    else:
        dw_z_e = None

    # Signal 3: DW instant trees delta — primary anchor in DRY season (uses SWIR)
    # SOTA upgrade: replaces raw _sigmoid(-dw_td - threshold) with dw_pheno_score_maxx math:
    #   Fix 1: sigmoidal cloud trust   (<40% cloud = no penalty)
    #   Fix 2: dynamic midpoint        (canopy-density-aware threshold)
    #   Fix 3: effective baseline      (max(trees_before, seasonal) — pristine forest fix)
    #   Fix 4: gated phenology bonus   (noise gate: bonus only if drop > 0.08)
    #   Fix 5: DOY interpolation       (smooth month-boundary baseline)
    #
    # Requires: props["dw_trees_now"] (trees_after from latest DW pass) and anchor_date.
    # Falls back to old _sigmoid if these are unavailable.
    dw_trees_now = props.get("dw_trees_now")  # sampled per-patch; trees_after
    dw_cloud_frac = (
        props.get("dw_cloud_frac", 0.0) or 0.0
    ) / 100.0  # stored as 0-100 pct, convert to 0-1
    if (
        dw_td is not None
        and dw_trees_now is not None
        and anchor_date
    ):
        trees_before_est = dw_trees_now - dw_td  # trees_before = trees_after - delta
        sota_ev = _dw_maxx_evidence(
            trees_after  = dw_trees_now,
            trees_before = trees_before_est,
            date_str     = anchor_date,
            cloud_frac   = dw_cloud_frac,
        )
        if sota_ev is not None:
            dw_td_e = sota_ev
        else:
            # Baseline CSV not loaded — old sigmoid fallback
            dw_td_e = _sigmoid(-dw_td - dw_td_t, scale=15.0)
    elif dw_td is not None:
        # No trees_now or date — old sigmoid fallback
        dw_td_e = _sigmoid(-dw_td - dw_td_t, scale=15.0)
    else:
        dw_td_e = None

    # Signal 4: DW bare ground rise — most specific felling signature for Guna
    #   stump / bare soil exposed after felling -> bare_ground class rises
    dw_bg_e = _sigmoid(dw_bg - dw_bg_t, scale=15.0) if dw_bg is not None else None
    # Dampen if trees class didn't also drop (bare ground alone could be drought)
    if dw_bg_e is not None and (dw_td is None or dw_td > -0.04):
        dw_bg_e = dw_bg_e * 0.4

    # Signal 5: land-class conversion (crops/built rising while trees drop)
    # ONLY fires when actual crop/built data is present AND trees dropped.
    # Bug-fix: previously conv_e = sigmoid(-1.0) ≈ 0.119 even with no data,
    # dragging raw_score down by ~0.02-0.05 on every patch with no conversion signal.
    have_conv_data = (dw_crop is not None) or (dw_blt is not None)
    if have_conv_data and dw_td is not None and dw_td < -dw_td_t * 0.4:
        conv_raw = max(
            (dw_crop / dw_cr_t) if dw_crop is not None else 0.0,
            (dw_blt  / dw_bl_t) if dw_blt  is not None else 0.0,
        )
        conv_e = _sigmoid(conv_raw - 1.0, scale=2.0)
    else:
        conv_e = None  # no data or no tree drop → exclude entirely

    # Signal 6: SAR VH drop — supplementary; leaf-off trees have low VH baseline
    # dVH < 0 means VH dropped (loss). We check -sar_dvh > vh_eff
    vh_e = _sigmoid(-sar_dvh - vh_eff, scale=0.8) if sar_dvh is not None else None
    if vh_e is not None:
        if season == "DRY":
            # Leaf-fall false positive suppression: dry deciduous forests shed
            # leaves Feb-Apr causing VH drops of 0.5-1.3 dB that mimic felling.
            # Dampen SAR evidence in DRY season — only strong drops (>2 dB,
            # enforced by vh_eff above) produce meaningful evidence.
            vh_e = vh_e * 0.6
        else:
            vh_e = vh_e * 0.5   # monsoon: soil moisture swamps canopy signal

    # Signal 7: SAR CuSum — persistent repeated felling
    cusum_e = _sigmoid(cusum_s - cusum_t, scale=4.0) if cusum_s is not None else None
    if cusum_e is not None:
        if season == "DRY":
            # CuSum accumulates gradual leaf-fall VH decline as "persistent loss".
            # Dampen in dry season to reduce false alarm rate.
            cusum_e = cusum_e * 0.5
        else:
            cusum_e = cusum_e * 0.7

    # ── Weighted combination ───────────────────────────────────────────────────
    # Subfamily column: "pheno" = phenology-based (seasonal model, may be None in DRY)
    #                   "inst"  = structural instant signal (DW SWIR-based, primary in DRY)
    #                   "sar"   = SAR (rare corroboration)
    signal_pairs = [
        (ndvi_e,  3.5, "optical", "pheno"),
        (dw_z_e,  3.0, "optical", "pheno"),
        (dw_td_e, 3.0, "optical", "inst"),
        (dw_bg_e, 2.5, "optical", "inst"),
        (conv_e,  1.5, "optical", "inst"),
        (vh_e,    1.5, "sar",     "sar"),
        (cusum_e, 0.8, "sar",     "sar"),
    ]

    opt_ev,   opt_wt   = [], []
    pheno_ev, pheno_wt = [], []
    inst_ev,  inst_wt  = [], []
    sar_ev,   sar_wt   = [], []
    wsum = wtot = 0.0

    for ev, w, fam, subfam in signal_pairs:
        if ev is None:
            continue   # None = sensor excluded from denominator (not just value zeroed)
        wsum += ev * w
        wtot += w
        if fam == "optical":
            opt_ev.append(ev); opt_wt.append(w)
            if subfam == "pheno":
                pheno_ev.append(ev); pheno_wt.append(w)
            else:
                inst_ev.append(ev); inst_wt.append(w)
        else:
            sar_ev.append(ev); sar_wt.append(w)

    if wtot == 0:
        return (0, 0.0, {})

    raw_score = wsum / wtot

    def _wavg(vs, ws):
        return sum(v*w for v,w in zip(vs,ws)) / sum(ws) if ws else 0.5

    opt_mean   = _wavg(opt_ev,   opt_wt)
    pheno_mean = _wavg(pheno_ev, pheno_wt)
    inst_mean  = _wavg(inst_ev,  inst_wt)
    sar_mean   = _wavg(sar_ev,   sar_wt)
    have_sar   = len(sar_ev)   > 0
    have_pheno = len(pheno_ev) > 0
    have_inst  = len(inst_ev)  > 0

    # ── Cross-signal agreement bonus ──────────────────────────────────────────
    # With None-exclusion: in DRY season pheno signals are absent (None → excluded),
    # so have_pheno=False, agreement=1.0 (single-family neutral). No penalty fires.
    # In WET season both families present: check convincingness as before.
    if have_pheno and have_inst:
        if pheno_mean >= 0.60 and inst_mean >= 0.60:
            agreement = 1.25   # two independent optical families agree strongly
        elif pheno_mean >= 0.55 or inst_mean >= 0.55:
            agreement = 1.00   # at least one family convincingly above threshold
        else:
            agreement = 0.85   # both families present but neither convincing
    else:
        agreement = 1.00       # only one family available → no cross-sensor penalty

    # SAR cross-sensor bonus (rare but powerful when it fires)
    if have_sar and sar_mean >= 0.58:
        agreement = min(agreement * 1.15, 1.50)

    # ── Contradiction penalties ────────────────────────────────────────────────
    penalty = 1.0

    # DW instant trees drop but NDVI is above expected -> possible DW classifier artefact
    # In DRY season: ndvi_e is already None (excluded), so this branch never fires.
    # Guard kept for WET season correctness.
    if dw_td is not None and ndvi_z is not None and season != "DRY":
        if dw_td < -0.08 and ndvi_z > 0.3:
            penalty = min(penalty, 0.55)

    # SAR strongly rose (opposite direction) while optical says loss
    # -> soil moisture surge, not felling
    if sar_dvh is not None and sar_dvh > 2.0 and opt_mean > 0.55:
        penalty = min(penalty, 0.75)

    # ── Final confidence ─────────────────────────────────────────────────────────
    confidence = min(0.95, max(0.0, raw_score * agreement * penalty))

    details = {
        "raw":     round(raw_score, 3),
        "adj":     round(agreement * penalty, 3),
        "opt":     round(opt_mean, 3),
        "sar":     round(sar_mean, 3),
        "pheno":   round(pheno_mean, 3),
        "inst":    round(inst_mean, 3),
        "ndvi_e":  round(ndvi_e,  3) if ndvi_e  is not None else None,
        "dw_td_e": round(dw_td_e, 3) if dw_td_e is not None else None,
        "dw_bg_e": round(dw_bg_e, 3) if dw_bg_e is not None else None,
    }
    return (round(confidence, 3), details)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_gee_rules_engine(
    beat_geom:   Any,
    anchor_date: str,
    cfg:         dict,
    range_label: Optional[str] = None,   # e.g. "Binaganj" — for phenology model lookup
) -> Dict[str, Any]:
    """
    Run the full GEE-SRC rules engine for a single beat AOI.

    Parameters
    ----------
    beat_geom   : ee.Geometry
        Beat bounding geometry (or dissolved polygon).
    anchor_date : str
        Detection anchor date "YYYY-MM-DD".  Same-season baseline
        is automatically set to anchor_date - 365 days.
    cfg         : dict
        Parsed config.yaml dictionary.
    range_label : str, optional
        Range name (e.g. "Binaganj") used to load the matching per-range
        harmonic phenology model.  If None or model not found, falls back
        to raw dNDVI threshold comparison.

    Returns
    -------
    dict with keys:
        tier        (int)   : 0=no alert, 1=high confidence, 2=medium, 3=SAR-only
        area_ha     (float) : flagged area in hectares
        confidence  (float) : 0.0–1.0 detection confidence
        signals     (dict)  : individual signal stats {name: value}
        geojson     (dict)  : GeoJSON FeatureCollection of alert polygons
        skipped     (bool)  : True if a GEE error occurred
        error       (str)   : error message if skipped
    """
    import ee

    rc = _load_rules_cfg(cfg)

    # Return structure when nothing detected
    null_result = {
        "tier":       0,
        "area_ha":    0.0,
        "confidence": 0.0,
        "signals":    {},
        "geojson":    {"type": "FeatureCollection", "features": []},
        "skipped":    False,
        "error":      "",
    }

    try:
        log.info(f"  [rules engine] anchor={anchor_date}, window=±{rc['opt_window']}d")

        # ── 1. Build optical composites (NDVI + NBR) ─────────────────────────
        # Primary: Sentinel-2 NDVI (±15 days, 10m, 5-10 day revisit)
        # Fallback: MODIS MOD13Q1 (±45 days, 250m, 16-day composites)
        log.info("    Building NDVI composite (S2 primary → MODIS fallback)...")
        ndvi_source = "unknown"
        s2_result = build_s2_ndvi_composite(beat_geom, anchor_date, rc["s2_window"])
        if s2_result is not None:
            ndvi_c, ndvi_b = s2_result
            ndvi_source = f"Sentinel-2 (±{rc['s2_window']}d)"
        else:
            log.info("    Falling back to MODIS NDVI (±45 days)...")
            ndvi_c, ndvi_b, _, _ = build_optical_composites(
                beat_geom, anchor_date, rc["opt_window"]
            )
            ndvi_source = f"MODIS MOD13Q1 (±{rc['opt_window']}d)"
        log.info(f"    NDVI source: {ndvi_source}")

        # NBR always from MODIS MOD09A1 (needs SWIR2 band)
        log.info("    Building MODIS NBR composite...")
        _, _, nbr_c, nbr_b = build_optical_composites(
            beat_geom, anchor_date, rc["opt_window"]
        )
        dndvi = ndvi_b.subtract(ndvi_c).rename("dNDVI")   # positive = loss
        dnbr  = nbr_b.subtract(nbr_c).rename("dNBR")      # positive = loss

        # ── 1a. Build phenology z-score (replaces raw dNDVI when model available)
        ndvi_zscore   = None
        pheno_model   = None
        pheno_z_mean  = None  # for signal logging

        if range_label:
            # Resolve model directory relative to project root
            model_dir = rc["phenology_model_dir"]
            if not Path(model_dir).is_absolute():
                # Resolve relative to the location of this module
                model_dir = str(Path(__file__).resolve().parent.parent.parent / model_dir)

            pheno_model = _load_range_phenology_model(range_label, model_dir, "ndvi")

        if pheno_model is not None:
            log.info(f"    [phenology] Loaded model for range '{range_label}' "
                     f"(R2={pheno_model['r2']:.3f}, sigma={pheno_model['residual_std']:.4f})")
            ndvi_zscore = build_ndvi_zscore_image(ndvi_c, anchor_date, pheno_model)
        else:
            log.info("    [phenology] No model loaded — using raw dNDVI fallback")

        # ── 1b. Build Dynamic World year-on-year change (same-season baseline) ─
        log.info("    Building Dynamic World tree probability composites (same-season local calibration)...")
        dw_trees_delta = None
        try:
            dw_c, dw_b = build_dw_trees_composite(beat_geom, anchor_date, rc["opt_window"])
            # baseline - current: positive = tree probability DROP = structural loss
            dw_trees_delta = dw_b.subtract(dw_c).rename("dDW_trees")
            log.info("      DW year-on-year composites built OK")
        except Exception as dw_err:
            log.warning(f"      DW composite failed: {dw_err} — falling back to NDVI-only")

        # ── 1c. DW Instant Delta (consecutive S2 passes: trees/crops/built) ────
        log.info("    Building DW instant delta (consecutive-pass class probabilities)...")
        dw_instant = build_dw_instant_delta(beat_geom, anchor_date, rc["dw_instant_window"])
        dw_zscore  = None
        if dw_instant is not None:
            log.info(f"      trees Δ={dw_instant['date_prev']}→{dw_instant['date_latest']} "
                     f"({dw_instant['interval_days']}d interval)")

            # DW trees phenology z-score: is current trees prob anomalously low?
            if range_label:
                dw_model_dir = rc["phenology_model_dir"]
                if not Path(dw_model_dir).is_absolute():
                    dw_model_dir = str(Path(__file__).resolve().parent.parent.parent / dw_model_dir)
                dw_pheno = _load_range_phenology_model(range_label, dw_model_dir, "dw_trees")
                if dw_pheno is not None:
                    log.info(f"      [DW pheno] Loaded dw_trees model "
                             f"R2={dw_pheno['r2']:.3f}, σ={dw_pheno['residual_std']:.4f}")
                    # Reuse the same harmonic z-score function — works for any 0..1 variable
                    dw_zscore = build_ndvi_zscore_image(
                        dw_instant["dw_trees_current"], anchor_date, dw_pheno
                    ).rename("dw_trees_zscore")
                else:
                    log.info("      [DW pheno] No dw_trees model — skipping z-score")
        else:
            log.info("      DW instant delta unavailable (cloud cover or no pairs)")

        # ── 2. Build SAR delta ────────────────────────────────────────────────
        log.info("    Building S1 VH delta...")
        dvh = build_s1_delta(beat_geom, anchor_date, rc["s1_window"])

        # ── 3. Build CuSum score ──────────────────────────────────────────────
        log.info("    Building SAR CuSum score...")
        cusum = build_cusum_score(beat_geom, anchor_date)

        # ── 3b. RADD near-real-time alert (fast path) ───────────────────────
        log.info("    Checking RADD near-real-time alerts...")
        radd_img, radd_px_count = build_radd_flag(beat_geom, anchor_date, rc["radd_window"])
        radd_active = radd_px_count > 0
        if radd_active:
            log.info(f"    [RADD] ACTIVE: {radd_px_count} flagged pixels — fast-path early warning")

        # ── 4. Forest baseline mask ───────────────────────────────────────────
        forest_mask = _forest_baseline_mask(beat_geom)

        # ══════════════════════════════════════════════════════════════════════
        # NEW ARCHITECTURE: Patch-level scoring at 10m native resolution
        # ══════════════════════════════════════════════════════════════════════

        # Season-aware SAR confidence (Tier 3 path)
        dry_season   = _is_dry_season(anchor_date)
        t3_conf      = rc["sar_dry_conf"] if dry_season else rc["sar_wet_conf"]
        season_label = "DRY" if dry_season else "WET/MONSOON"
        log.info(f"    Season: {season_label} → tier3 (SAR-only) conf={t3_conf:.2f}")

        # ── Step 5: Build permissive 10m candidate mask + morphological clean ─
        log.info("    Building 10m candidate change mask (permissive union)...")
        raw_candidate  = build_candidate_mask(
            dndvi          = dndvi,
            ndvi_zscore    = ndvi_zscore,
            dw_instant     = dw_instant,
            dw_zscore      = dw_zscore,
            dw_trees_delta = dw_trees_delta,
            dvh            = dvh,
            cusum          = cusum,
            forest_mask    = forest_mask,
            aoi            = beat_geom,
            thresholds     = rc,
        )
        candidate_mask = apply_post_processing(raw_candidate)

        # ── Step 6: Vectorize → patch polygons at 10m, area-filtered ──────────
        log.info("    Vectorizing candidate patches at 10m...")
        patch_fc = vectorize_patches(
            candidate_mask = candidate_mask,
            aoi            = beat_geom,
            min_area_ha    = rc["min_area_ha"],
            scale          = 10,
        )

        # ── Step 7: Count candidate patches (early-exit gate) ─────────────────
        try:
            n_patches_candidate = patch_fc.size().getInfo()
        except Exception:
            n_patches_candidate = 0

        log.info(f"    Candidate patches (>={rc['min_area_ha']} ha): {n_patches_candidate}")

        # Baseline signals dict returned even on no-alert paths
        beat_signals: dict = {
            "ndvi_source":           ndvi_source,
            "radd_alert":            radd_active,
            "radd_px":               radd_px_count,
            "dw_instant_available":  dw_instant is not None,
            "dw_interval_days":      dw_instant["interval_days"] if dw_instant else None,
            "dw_date_prev":          dw_instant["date_prev"]     if dw_instant else None,
            "dw_date_latest":        dw_instant["date_latest"]   if dw_instant else None,
            "cloud_prev_pct":        dw_instant.get("cloud_prev_pct", -1) if dw_instant else None,
            "cloud_latest_pct":      dw_instant.get("cloud_latest_pct", -1) if dw_instant else None,
            "n_cloud_ok":            dw_instant.get("n_cloud_ok", 0) if dw_instant else 0,
            "pheno_model":           range_label if pheno_model is not None else None,
            "season":                season_label,
            "n_patches_candidate":   n_patches_candidate,
            "n_patches_alert":       0,
            "best_patch_area_ha":    0.0,
            "best_patch_tier":       0,
            "best_patch_conf":       0.0,
        }

        if n_patches_candidate == 0:
            log.info("    No candidate patches — no alert.")
            return {**null_result, "signals": beat_signals}

        # ── Step 8: Sample ALL signal images per patch in ONE batch GEE call ──
        log.info("    Batch-sampling signal images per patch...")
        all_signal_images = {
            "dNDVI":           dndvi,
            "ndvi_zscore":     ndvi_zscore,
            "dw_trees_delta":  dw_instant["trees_delta"]      if dw_instant else None,
            "dw_crops_delta":  dw_instant["crops_delta"]      if dw_instant else None,
            "dw_built_delta":  dw_instant["built_delta"]      if dw_instant else None,
            "dw_bare_delta":   dw_instant["bare_delta"]       if dw_instant else None,
            # SOTA scorer inputs: trees_after (absolute) + scene cloud fraction
            "dw_trees_now":    dw_instant["dw_trees_current"] if dw_instant else None,
            "dw_trees_zscore": dw_zscore,
            "dVH":             dvh,
            "cusum_score":     cusum,
            # dNBR excluded from patch scoring — 250m >> patch size, leaf-water signal near-zero in dry season
        }
        # Per-patch cloud fraction: use latest DW image scene cloud% (uniform per scene,
        # NOT per-pixel, so it's a scalar stored as a per-patch constant via a constant image)
        if dw_instant and dw_instant.get("cloud_latest_pct", -1) >= 0:
            import ee as _ee
            cloud_pct_img = _ee.Image.constant(dw_instant["cloud_latest_pct"]).rename("dw_cloud_frac")
            all_signal_images["dw_cloud_frac"] = cloud_pct_img
        sampled_fc = sample_patches(patch_fc, all_signal_images, scale=10)

        # ── Step 9: Fetch all patch data to Python in ONE getInfo() ───────────
        try:
            sampled_info = sampled_fc.getInfo()
        except Exception as e:
            log.warning(f"    Patch sampling failed: {e}")
            return {**null_result, "signals": beat_signals}

        features = sampled_info.get("features", [])

        # ── Step 10: Score each patch locally in Python ────────────────────────
        min_alert_conf = rc.get("min_alert_conf", 0.35)   # configurable; was implicit in tier>0
        valid_patches = []
        for feat in features:
            props = feat.get("properties", {})
            conf, score_details = score_patch(
                props,
                rc,
                t3_conf,
                season      = season_label,
                anchor_date = anchor_date,   # forwarded for SOTA DW scorer DOY lookup
            )
            if score_details:
                props["_score_details"] = score_details
            if conf >= min_alert_conf:
                props["confidence"] = conf
                feat["properties"]  = props
                valid_patches.append(feat)

        beat_signals["n_patches_alert"] = len(valid_patches)
        log.info(f"    Alert patches (conf>={min_alert_conf:.2f}): {len(valid_patches)} / {len(features)}")

        # ── Step 11: Determine beat-level result from best patch ───────────────
        if not valid_patches:
            log.info("    No patches passed scoring thresholds — no alert.")
            return {**null_result, "signals": beat_signals}

        best_patch = max(
            valid_patches,
            key=lambda f: (
                f["properties"]["confidence"],
                f["properties"].get("area_ha", 0.0),
            ),
        )
        best_props  = best_patch["properties"]
        final_conf  = best_props["confidence"]
        total_area  = sum(
            f["properties"].get("area_ha", 0.0) for f in valid_patches
        )

        beat_signals.update({
            "best_patch_area_ha":   best_props.get("area_ha", 0.0),
            "best_patch_conf":      final_conf,
            # Best-patch signal values propagated for audit trail
            "dNDVI_mean":           best_props.get("dNDVI"),
            "dNBR_mean":            best_props.get("dNBR"),
            "dVH_mean_db":          best_props.get("dVH"),
            "cusum_mean":           best_props.get("cusum_score"),
            "ndvi_zscore_mean":     best_props.get("ndvi_zscore"),
            "dDW_trees_mean":       best_props.get("dw_trees_delta"),
            "dw_trees_delta_mean":  best_props.get("dw_trees_delta"),
            "dw_crops_delta_mean":  best_props.get("dw_crops_delta"),
            "dw_built_delta_mean":  best_props.get("dw_built_delta"),
            "dw_trees_zscore_mean": best_props.get("dw_trees_zscore"),
        })
        log.info(f"    Best-patch signals: {beat_signals}")

        # ── Step 12: GeoJSON — all scored patches with per-patch signal props ──
        alert_geojson = {
            "type":     "FeatureCollection",
            "features": valid_patches,
        }

        log.info(
            f"  [rules engine] DONE conf={final_conf:.2f} "
            f"total_area={total_area:.2f}ha "
            f"(best={best_props.get('area_ha', 0):.2f}ha) n_patches={len(valid_patches)}"
        )

        return {
            "area_ha":    total_area,
            "confidence": final_conf,
            "signals":    beat_signals,
            "geojson":    alert_geojson,
            "skipped":    False,
            "error":      "",
        }

    except Exception as e:
        log.error(f"  [rules engine] GEE error: {e}")
        return {
            "tier":       0,
            "area_ha":    0.0,
            "confidence": 0.0,
            "signals":    {},
            "geojson":    {"type": "FeatureCollection", "features": []},
            "skipped":    True,
            "error":      str(e),
        }


def rules_engine_result_to_chip_mask(
    result: Dict[str, Any],
    chip_profile: dict,
) -> Optional[Any]:
    """
    Rasterize the rules-engine GeoJSON alert polygons into a (H, W) numpy
    float32 array in chip pixel space.

    Used when the rules engine result needs to be compared with the
    existing neural-model change mask format (for hybrid mode).

    Parameters
    ----------
    result       : dict returned by run_gee_rules_engine()
    chip_profile : rasterio profile of the reference chip (for transform/shape)

    Returns
    -------
    np.ndarray of shape (H, W), dtype float32, values = confidence (0 or conf)
    None if result has no alert.
    """
    import numpy as np
    import rasterio.features

    if result["tier"] == 0 or not result["geojson"]["features"]:
        return None

    h = chip_profile["height"]
    w = chip_profile["width"]
    transform = chip_profile["transform"]

    shapes = [
        (feat["geometry"], result["confidence"])
        for feat in result["geojson"]["features"]
        if feat.get("geometry")
    ]
    if not shapes:
        return None

    mask = rasterio.features.rasterize(
        shapes     = shapes,
        out_shape  = (h, w),
        transform  = transform,
        fill       = 0.0,
        dtype      = np.float32,
    )
    return mask
