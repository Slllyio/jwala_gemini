"""
Patch: Patch-level alert architecture for rules_engine.py.

Changes applied:
  1. Replace apply_post_processing() with simplified 10m morphological version.
  2. Insert 4 new functions before run_gee_rules_engine():
       build_candidate_mask(), vectorize_patches(),
       sample_patches(), score_patch()
  3. Replace steps 5-14 inside run_gee_rules_engine() with patch-level scoring.
"""
from pathlib import Path
import re

TARGET = Path(r"c:\Users\S.C.C\OneDrive\Desktop\jwalaNetra_2\src\inference\rules_engine.py")
src = TARGET.read_text(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# REPLACEMENT 1 — apply_post_processing() → simplified 10m morphological clean
# ─────────────────────────────────────────────────────────────────────────────
OLD_APP = '''\
def apply_post_processing(
    raw_mask: Any,
    aoi:      Any,
    min_area_ha: float = _MIN_AREA_HA,
    forest_mask: Optional[Any] = None,
) -> Any:
    """
    Post-process a raw binary change mask:
    1. Morphological cleaning (remove isolated pixels)
    2. Forest mask validation (must have been forest in baseline)
    3. Area filtering (≥ min_area_ha)

    Returns a clean binary mask at 250m MODIS resolution.
    """
    import ee

    # Step 1: morphological opening
    cleaned = _morphological_clean(raw_mask)

    # Step 2: restrict to confirmed forest pixels
    if forest_mask is not None:
        cleaned = cleaned.updateMask(forest_mask)

    # Step 3: connected-component area filter
    # At 250m pixels, 1 pixel = 0.0625 ha
    pixel_ha = 0.0625
    min_px   = max(1, int(min_area_ha / pixel_ha))  # e.g. 0.2 ha → 4 px

    components = cleaned.connectedComponents(
        connectedness=ee.Kernel.plus(1),
        maxSize=256,
    )
    counts = components.select("labels").connectedPixelCount(maxSize=256, eightConnected=False)
    size_mask = counts.gte(min_px)
    final     = cleaned.updateMask(size_mask)

    return final'''

NEW_APP = '''\
def apply_post_processing(mask: Any) -> Any:
    """
    Morphological cleaning at native 10m scale.
    Replaces the old 250m connectedPixelCount which destroyed sub-hectare patches.
    A radius-1 opening removes single-pixel speckle without dissolving real patches.
    """
    import ee
    kernel = ee.Kernel.circle(radius=1)
    return mask.focal_min(kernel=kernel).focal_max(kernel=kernel)'''

assert OLD_APP in src, "FAILED: apply_post_processing old block not found"
src = src.replace(OLD_APP, NEW_APP, 1)
print("[OK] Replaced apply_post_processing()")

# ─────────────────────────────────────────────────────────────────────────────
# REPLACEMENT 2 — Insert 4 new functions before "# ── Main entry point ──"
# ─────────────────────────────────────────────────────────────────────────────
NEW_FUNCTIONS = '''\

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


def score_patch(props: dict, thresholds: dict, t3_conf: float):
    """
    Python-side per-patch scoring.  Mirrors fuse_signals() tier definitions
    but operates on already-fetched float values (no GEE calls).

    Returns (tier: int, confidence: float).
    Uses explicit `is not None` checks so that 0.0 (valid signal) doesn't
    short-circuit as falsy.
    """
    dv = props.get

    ndvi_z    = dv("ndvi_zscore")
    dndvi_v   = dv("dNDVI")
    dw_z      = dv("dw_trees_zscore")
    dw_td     = dv("dw_trees_delta")   # negative = canopy loss
    dw_crop   = dv("dw_crops_delta")
    dw_built  = dv("dw_built_delta")
    sar_dvh   = dv("dVH")
    cusum_s   = dv("cusum_score")
    dnbr_s    = dv("dNBR")

    def flag(v, thresh, sign="gt"):
        if v is None:
            return False
        return (v > thresh) if sign == "gt" else (v < thresh)

    ndvi_flag  = (flag(ndvi_z, thresholds["ndvi_z_thresh"], "lt")
                  if ndvi_z is not None
                  else flag(dndvi_v, thresholds["ndvi_thresh"]))
    nbr_flag   = flag(dnbr_s,  thresholds["nbr_thresh"])
    vh_flag    = flag(sar_dvh, thresholds["vh_drop_db"])
    cusum_flag = flag(cusum_s, thresholds["cusum_thresh"])

    dw_tree_drop  = (flag(dw_td,    -abs(thresholds["dw_tree_drop"]), "lt")
                     if dw_td is not None else False)
    dw_crop_rise  = flag(dw_crop,  thresholds["dw_crop_rise"])
    dw_built_rise = flag(dw_built, thresholds["dw_built_rise"])
    dw_z_flag     = (flag(dw_z, thresholds["ndvi_z_thresh"], "lt")
                     if dw_z is not None else False)

    # Tier A: both phenology z-scores agree (strongest)
    if ndvi_flag and dw_z_flag:
        return (1, 0.90)
    # Tier B: DW instant drop + land-class conversion + SAR confirm
    if dw_tree_drop and (dw_crop_rise or dw_built_rise) and vh_flag:
        return (1, 0.90)
    # Legacy Tier 1: optical + NBR + VH + CuSum
    if ndvi_flag and nbr_flag and vh_flag and cusum_flag:
        return (1, 0.90)
    # Tier C: DW instant drop + SAR (fast, no phenology needed)
    if dw_tree_drop and vh_flag:
        return (2, 0.72)
    # Tier D: DW trees z-score alone (anomalous, single source)
    if dw_z_flag:
        return (2, 0.65)
    # Legacy Tier 2: optical + VH
    if ndvi_flag and vh_flag:
        return (2, 0.65)
    # Tier 3: SAR-only persistent signal
    if vh_flag and cusum_flag:
        return (3, t3_conf)

    return (0, 0.0)

'''

ANCHOR = "# ── Main entry point ──────────────────────────────────────────────────────────"
assert ANCHOR in src, "FAILED: main entry point anchor not found"
src = src.replace(ANCHOR, NEW_FUNCTIONS + ANCHOR, 1)
print("[OK] Inserted 4 new helper functions")

# ─────────────────────────────────────────────────────────────────────────────
# REPLACEMENT 3 — Replace steps 5-14 inside run_gee_rules_engine()
# From "# ── 5. Fuse signals" through end of the try block (last return)
# ─────────────────────────────────────────────────────────────────────────────
OLD_STEPS = '''\
        # ── 5. Fuse signals ───────────────────────────────────────────────────
        log.info("    Fusing signals (NDVI z-score/dNDVI + DW_trees + DW instant + NBR + VH + CuSum)...")
        tier1, tier2, tier3 = fuse_signals(
            aoi         = beat_geom,
            dndvi       = dndvi,
            dnbr        = dnbr,
            dvh         = dvh,
            cusum       = cusum,
            thresholds  = rc,
            forest_mask = forest_mask,
            dw_trees    = dw_trees_delta,
            ndvi_zscore = ndvi_zscore,
            dw_zscore   = dw_zscore,
            dw_instant  = dw_instant,
        )

        # ── 6. Post-process ───────────────────────────────────────────────────
        # Season-aware SAR confidence: dry season = 0.65, monsoon = 0.40
        dry_season   = _is_dry_season(anchor_date)
        t3_conf      = rc["sar_dry_conf"] if dry_season else rc["sar_wet_conf"]
        season_label = "DRY" if dry_season else "WET/MONSOON"
        log.info(f"    Season: {season_label} → tier3 (SAR-only) confidence = {t3_conf:.2f}")

        if rc["require_tier1"]:
            raw_alert      = tier1
            confidence_map = tier1.multiply(0.9)
        else:
            # SAR-only tier3 is now an OPERATIONAL alert path (fast detection)
            raw_alert      = tier1.Or(tier2).Or(tier3)
            confidence_map = (tier1.multiply(0.9)
                              .add(tier2.And(tier1.Not()).multiply(0.65))
                              .add(tier3.And(tier1.Not()).And(tier2.Not()).multiply(t3_conf)))

        alert_mask = apply_post_processing(
            raw_mask    = raw_alert,
            aoi         = beat_geom,
            min_area_ha = rc["min_area_ha"],
            forest_mask = forest_mask,
        )

        # ── 7. Sample signal stats (for logging / audit) ──────────────────────
        log.info("    Sampling signal statistics...")
        def _sample(img: Any, name: str) -> Optional[float]:
            try:
                v = img.rename(name).reduceRegion(
                    reducer   = ee.Reducer.mean(),
                    geometry  = beat_geom,
                    scale     = 250,
                    maxPixels = 1e7,
                    bestEffort= True,
                ).getInfo().get(name)
                return float(v) if v is not None else None
            except Exception:
                return None

        signals = {
            "ndvi_source":           ndvi_source,
            "dNDVI_mean":            _sample(dndvi, "dNDVI"),
            "dNBR_mean":             _sample(dnbr,  "dNBR"),
            "dVH_mean_db":           _sample(dvh,   "dVH"),
            "cusum_mean":            _sample(cusum, "cusum_score"),
            "dDW_trees_mean":        _sample(dw_trees_delta, "dDW_trees") if dw_trees_delta is not None else None,
            "ndvi_zscore_mean":      _sample(ndvi_zscore, "ndvi_zscore") if ndvi_zscore is not None else None,
            "pheno_model":           range_label if pheno_model is not None else None,
            "radd_alert":            radd_active,
            "radd_px":               radd_px_count,
            # DW instant delta signals
            "dw_instant_available":  dw_instant is not None,
            "dw_interval_days":      dw_instant["interval_days"] if dw_instant else None,
            "dw_date_prev":          dw_instant["date_prev"]     if dw_instant else None,
            "dw_date_latest":        dw_instant["date_latest"]   if dw_instant else None,
            "dw_trees_delta_mean":   _sample(dw_instant["trees_delta"], "dw_trees_delta") if dw_instant else None,
            "dw_crops_delta_mean":   _sample(dw_instant["crops_delta"], "dw_crops_delta") if dw_instant else None,
            "dw_built_delta_mean":   _sample(dw_instant["built_delta"], "dw_built_delta") if dw_instant else None,
            "dw_trees_zscore_mean":  _sample(dw_zscore, "dw_trees_zscore") if dw_zscore is not None else None,
        }
        log.info(f"    Signals: {signals}")

        # ── 8. Compute alert area ─────────────────────────────────────────────
        px_count = alert_mask.unmask(0).reduceRegion(
            reducer   = ee.Reducer.sum(),
            geometry  = beat_geom,
            scale     = 250,
            maxPixels = 1e7,
            bestEffort= True,
        ).getInfo()
        n_px    = int(px_count.get("dNDVI", 0) or 0)
        area_ha = n_px * 0.0625   # 250m × 250m = 6.25 ha per pixel; actually 0.0625 at 250m
        # Correction: 250m × 250m = 62,500 m² = 6.25 ha per pixel
        area_ha = n_px * 6.25

        # ── 9. Determine tier and confidence ─────────────────────────────────
        if area_ha == 0:
            return {**null_result, "signals": signals}

        tier1_px = (alert_mask.updateMask(tier1).unmask(0).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=beat_geom,
            scale=250, maxPixels=1e7, bestEffort=True,
        ).getInfo().get("dNDVI", 0) or 0)

        if tier1_px > 0:
            final_tier  = 1
            final_conf  = 0.9
        elif area_ha > 0:
            # Check if tier3 (SAR-only) contributed
            t3_px = (tier3.updateMask(alert_mask).unmask(0).reduceRegion(
                reducer=ee.Reducer.sum(), geometry=beat_geom,
                scale=250, maxPixels=1e7, bestEffort=True,
            ).getInfo().get("VH", 0) or 0)
            if t3_px > 0 and int(tier1_px) == 0:
                # SAR-only fast-path: confidence is season-aware
                final_tier = 3
                final_conf = t3_conf   # 0.65 dry season, 0.40 monsoon
                log.info(f"    [tier3] SAR-only alert ({season_label}) conf={final_conf:.2f}")
            else:
                final_tier = 2
                final_conf = 0.65
        else:
            return {**null_result, "signals": signals}

        # ── 10. Vectorize alert mask to GeoJSON ───────────────────────────────
        log.info(f"    Vectorizing alert mask (tier={final_tier}, area={area_ha:.2f} ha)...")
        try:
            vectors = alert_mask.reduceToVectors(
                reducer     = ee.Reducer.countEvery(),
                geometry    = beat_geom,
                scale       = 250,
                maxPixels   = 1e7,
                bestEffort  = True,
                geometryType= "polygon",
            )
            geojson = vectors.getInfo()   # FeatureCollection
        except Exception as e:
            log.warning(f"    Vectorization failed: {e} — returning pixel count only")
            geojson = {"type": "FeatureCollection", "features": []}

        log.info(f"  [rules engine] DONE tier={final_tier} area={area_ha:.2f}ha "
                 f"conf={final_conf:.2f}")

        return {
            "tier":       final_tier,
            "area_ha":    area_ha,
            "confidence": final_conf,
            "signals":    signals,
            "geojson":    geojson,
            "skipped":    False,
            "error":      "",
        }'''

NEW_STEPS = '''\
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
            "dw_trees_delta":  dw_instant["trees_delta"] if dw_instant else None,
            "dw_crops_delta":  dw_instant["crops_delta"] if dw_instant else None,
            "dw_built_delta":  dw_instant["built_delta"] if dw_instant else None,
            "dw_trees_zscore": dw_zscore,
            "dVH":             dvh,
            "cusum_score":     cusum,
            "dNBR":            dnbr,
        }
        sampled_fc = sample_patches(patch_fc, all_signal_images, scale=10)

        # ── Step 9: Fetch all patch data to Python in ONE getInfo() ───────────
        try:
            sampled_info = sampled_fc.getInfo()
        except Exception as e:
            log.warning(f"    Patch sampling failed: {e}")
            return {**null_result, "signals": beat_signals}

        features = sampled_info.get("features", [])

        # ── Step 10: Score each patch locally in Python ────────────────────────
        valid_patches = []
        for feat in features:
            props = feat.get("properties", {})
            tier, conf = score_patch(props, rc, t3_conf)
            if tier > 0:
                props["tier"]       = tier
                props["confidence"] = conf
                feat["properties"]  = props
                valid_patches.append(feat)

        beat_signals["n_patches_alert"] = len(valid_patches)
        log.info(f"    Alert patches (tier>0): {len(valid_patches)} / {len(features)}")

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
        final_tier  = best_props["tier"]
        final_conf  = best_props["confidence"]
        total_area  = sum(
            f["properties"].get("area_ha", 0.0) for f in valid_patches
        )

        beat_signals.update({
            "best_patch_area_ha":   best_props.get("area_ha", 0.0),
            "best_patch_tier":      final_tier,
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
            f"  [rules engine] DONE tier={final_tier} "
            f"total_area={total_area:.2f}ha "
            f"(best={best_props.get('area_ha', 0):.2f}ha) "
            f"conf={final_conf:.2f} n_patches={len(valid_patches)}"
        )

        return {
            "tier":       final_tier,
            "area_ha":    total_area,
            "confidence": final_conf,
            "signals":    beat_signals,
            "geojson":    alert_geojson,
            "skipped":    False,
            "error":      "",
        }'''

assert OLD_STEPS in src, "FAILED: steps 5-14 old block not found — check whitespace"
src = src.replace(OLD_STEPS, NEW_STEPS, 1)
print("[OK] Replaced run_gee_rules_engine steps 5-14")

# ── Write output ──────────────────────────────────────────────────────────────
TARGET.write_text(src, encoding="utf-8")
print(f"\nAll patches applied successfully → {TARGET}")
