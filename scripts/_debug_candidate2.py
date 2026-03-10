"""Count candidate mask pixels directly over the KML polygon."""
import sys, yaml, xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
cfg = yaml.safe_load(open("config.yaml"))
from src.data.gee_fetch import init_gee
init_gee(cfg.get("gee", {}).get("gee_project", "van-suraksha-alert"))
import ee

# Parse KML (strip namespace)
tree = ET.parse("data/ground_truth/confirmed_loss_feb2026.kml")
root = tree.getroot()
for elem in root.iter():
    if "}" in elem.tag:
        elem.tag = elem.tag.split("}", 1)[1]
ct = root.find(".//coordinates")
coords_ll = [[float(p.split(",")[0]), float(p.split(",")[1])]
             for p in ct.text.strip().split()]
gt_geom = ee.Geometry.Polygon([coords_ll]).buffer(100)

ANCHOR = "2026-02-09"

from src.inference.rules_engine import (
    build_dw_instant_delta, build_s2_ndvi_composite, build_ndvi_zscore_image,
    _load_range_phenology_model, _load_rules_cfg, build_candidate_mask,
    apply_post_processing, _forest_baseline_mask, build_s1_delta,
    build_optical_composites
)
rc = _load_rules_cfg(cfg)

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

# Build candidate mask and count pixels
cand = build_candidate_mask(
    dndvi=dndvi, ndvi_zscore=ndvi_zscore,
    dw_instant=dw, dw_zscore=None,
    dw_trees_delta=None, dvh=dvh, cusum=None,
    forest_mask=forest_mask, aoi=gt_geom, thresholds=rc
)

# Count candidate pixels
cand_int = cand.unmask(0).selfMask()
n_cand = cand_int.reduceRegion(ee.Reducer.count(), gt_geom, 10).getInfo()
n_total = cand.unmask(0).reduceRegion(ee.Reducer.count(), gt_geom, 10).getInfo()

print(f"Candidate pixels (before morphology): {n_cand}")
print(f"Total pixels in AOI: {n_total}")

# Debug: check individual components
print("\n-- Component checks --")
thresh = rc["dw_tree_drop"]
gate = abs(thresh) * 0.6
print(f"DW tree_drop threshold: {thresh}, candidate gate: {gate}")

if dw and "trees_delta" in dw:
    td = dw["trees_delta"]
    td_cand = td.lt(-gate)
    td_fire = td_cand.unmask(0).selfMask().reduceRegion(ee.Reducer.count(), gt_geom, 10).getInfo()
    td_total = td.unmask(-999).neq(-999).reduceRegion(ee.Reducer.count(), gt_geom, 10).getInfo()
    print(f"DW instant pixels (total, fires): {td_total}  |  {td_fire}")

if ndvi_zscore is not None:
    z_thresh = rc["ndvi_z_thresh"] * 0.75
    z_fire = ndvi_zscore.lt(z_thresh).unmask(0).selfMask().reduceRegion(ee.Reducer.count(), gt_geom, 10).getInfo()
    print(f"NDVI z-score pixels firing (z < {z_thresh:.2f}): {z_fire}")

if forest_mask is not None:
    fm_fire = forest_mask.unmask(0).selfMask().reduceRegion(ee.Reducer.count(), gt_geom, 10).getInfo()
    print(f"Forest mask pixels (true): {fm_fire}")

# After morphology
cand_morph = apply_post_processing(cand)
n_morph = cand_morph.unmask(0).selfMask().reduceRegion(ee.Reducer.count(), gt_geom, 10).getInfo()
print(f"\nCandidate pixels (after morphology): {n_morph}")
