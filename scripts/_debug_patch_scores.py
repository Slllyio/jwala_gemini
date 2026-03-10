"""
Per-patch score debug: run the full pipeline for ONE anchor date and print
the score breakdown for EVERY candidate patch, not just alert patches.
This shows why patches score below threshold.
"""
import sys, yaml, json, xml.etree.ElementTree as ET, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
cfg = yaml.safe_load(open("config.yaml"))

from src.data.gee_fetch import init_gee
init_gee(cfg.get("gee", {}).get("gee_project", "van-suraksha-alert"))
import ee

# ── Parse KML polygon ────────────────────────────────────────────────────────
tree = ET.parse("data/ground_truth/confirmed_loss_feb2026.kml")
root = tree.getroot()
for elem in root.iter():
    if "}" in elem.tag:
        elem.tag = elem.tag.split("}", 1)[1]
ct = root.find(".//coordinates")
coords_ll = [[float(p.split(",")[0]), float(p.split(",")[1])]
             for p in ct.text.strip().split()]
gt_geom = ee.Geometry.Polygon([coords_ll]).buffer(50)

from src.inference.rules_engine import (
    _load_rules_cfg, build_dw_instant_delta, build_s2_ndvi_composite,
    build_ndvi_zscore_image, _load_range_phenology_model, build_s1_delta,
    build_candidate_mask, apply_post_processing,
    vectorize_patches, sample_patches, score_patch, _forest_baseline_mask,
    _is_dry_season
)

rc = _load_rules_cfg(cfg)
ANCHORS = ["2026-02-09", "2026-02-14", "2026-02-22"]

for ANCHOR in ANCHORS:
    print(f"\n{'='*80}")
    print(f"Anchor: {ANCHOR}")
    print(f"{'='*80}")

    # --- Build signals ---
    dw = build_dw_instant_delta(gt_geom, ANCHOR, rc["dw_instant_window"])
    s2 = build_s2_ndvi_composite(gt_geom, ANCHOR, rc["s2_window"])
    dvh = build_s1_delta(gt_geom, ANCHOR, rc["s1_window"])

    ndvi_zscore = None
    dndvi = None
    if s2:
        ndvi_c, ndvi_b = s2
        dndvi = ndvi_b.subtract(ndvi_c).rename("dNDVI")
        model_dir = rc["phenology_model_dir"]
        if not Path(model_dir).is_absolute():
            model_dir = str(Path(__file__).resolve().parent.parent / model_dir)
        pheno = _load_range_phenology_model("Binaganj", model_dir, "ndvi")
        if pheno:
            ndvi_zscore = build_ndvi_zscore_image(ndvi_c, ANCHOR, pheno)

    forest_mask = _forest_baseline_mask(gt_geom)
    season = "DRY" if _is_dry_season(ANCHOR) else "WET"
    t3_conf = rc["sar_dry_conf"] if season == "DRY" else rc["sar_wet_conf"]

    # --- Candidate mask ---
    cand = build_candidate_mask(
        dndvi=dndvi, ndvi_zscore=ndvi_zscore,
        dw_instant=dw, dw_zscore=None,
        dw_trees_delta=None, dvh=dvh, cusum=None,
        forest_mask=forest_mask, aoi=gt_geom, thresholds=rc
    )
    cand_morph = apply_post_processing(cand)

    # --- Vectorize patches ---
    patch_fc = vectorize_patches(
        candidate_mask=cand_morph, aoi=gt_geom,
        min_area_ha=rc["min_area_ha"], scale=10
    )
    n_patches = patch_fc.size().getInfo()
    print(f"Candidate patches (>={rc['min_area_ha']} ha): {n_patches}")

    if n_patches == 0:
        print("  No candidate patches found.")
        continue

    # --- Build signal images ---
    signal_images = {}
    if ndvi_zscore is not None:
        signal_images["ndvi_zscore"] = ndvi_zscore
    if dndvi is not None:
        signal_images["dNDVI"] = dndvi
    if dw is not None:
        if "trees_delta" in dw:
            signal_images["dw_trees_delta"] = dw["trees_delta"]
        if "bare_delta" in dw:
            signal_images["dw_bare_delta"] = dw["bare_delta"]
        if "crops_delta" in dw:
            signal_images["dw_crops_delta"] = dw["crops_delta"]
        if "zscore" in dw:
            signal_images["dw_trees_zscore"] = dw["zscore"]
    if dvh is not None:
        signal_images["dVH"] = dvh

    # --- Sample all patches ---
    sampled_fc = sample_patches(patch_fc, signal_images, scale=10)
    patches_info = sampled_fc.getInfo()["features"]

    print(f"\n{'Patch':6} {'Area':>8} {'Conf':>6} {'dTrees':>8} {'NDVI_z':>8} {'bare':>8} {'dVH':>8}")
    print("-" * 65)

    for i, feat in enumerate(patches_info):
        props = feat["properties"]
        area_ha = props.get("area_ha", 0)
        conf, details = score_patch(props, rc, t3_conf, season=season)
        dt_val = props.get("dw_trees_delta")
        nz_val = props.get("ndvi_zscore")
        bg_val = props.get("dw_bare_delta")
        dv_val = props.get("dVH")

        fmt = lambda v: f"{v:+8.3f}" if v is not None else "     n/a"
        mark = " <<<ALERT" if conf >= 0.35 else ""
        print(f"{i+1:5d}  {area_ha:>7.2f}ha  {conf:>6.3f} {fmt(dt_val)} {fmt(nz_val)} {fmt(bg_val)} {fmt(dv_val)}{mark}")
        print(f"       raw={details.get('raw',0):.3f}  adj={details.get('adj',0):.3f}  "
              f"opt={details.get('opt',0):.3f}  pheno={details.get('pheno',0):.3f}  inst={details.get('inst',0):.3f}")

print("\nDone.")
