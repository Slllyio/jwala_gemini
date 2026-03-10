"""
Debug script: probe the candidate mask and raw signal values over the
confirmed loss polygon for anchor 2026-02-09.

Run with: python scripts/_debug_candidate.py
"""
import sys, yaml, xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
cfg = yaml.safe_load(open("config.yaml"))

from src.data.gee_fetch import init_gee
init_gee(cfg.get("gee", {}).get("gee_project", "van-suraksha-alert"))
import ee

# ── Parse KML ────────────────────────────────────────────────
kml_path = "data/ground_truth/confirmed_loss_feb2026.kml"
tree = ET.parse(kml_path)
root = tree.getroot()
# Strip namespace
for elem in root.iter():
    if "}" in elem.tag:
        elem.tag = elem.tag.split("}", 1)[1]
ct = root.find(".//coordinates")
coords_ll = [[float(p.split(",")[0]), float(p.split(",")[1])]
             for p in ct.text.strip().split()]
gt_geom = ee.Geometry.Polygon([coords_ll]).buffer(100)

ANCHOR = "2026-02-09"

from src.inference.rules_engine import (
    build_dw_instant_delta, build_ndvi_zscore_image,
    _load_range_phenology_model, _load_rules_cfg,
    build_s2_ndvi_composite, build_optical_composites,
    build_s1_delta, _forest_baseline_mask,
)
rc = _load_rules_cfg(cfg)

print(f"=== Signal debug over KML polygon, anchor={ANCHOR} ===\n")

# ── DW Instant ───────────────────────────────────────────────
print("1. DW instant delta...")
dw = build_dw_instant_delta(gt_geom, ANCHOR, rc["dw_instant_window"])
if dw:
    print(f"   pair: {dw['date_prev']} -> {dw['date_latest']} ({dw['interval_days']}d)")
    td = dw["trees_delta"].reduceRegion(ee.Reducer.mean(), gt_geom, 10).getInfo()
    bd = dw["bare_delta"].reduceRegion(ee.Reducer.mean(), gt_geom, 10).getInfo()
    print(f"   trees_delta mean: {td}")
    print(f"   bare_delta mean:  {bd}")
    dw_now = dw["dw_trees_current"].reduceRegion(ee.Reducer.mean(), gt_geom, 10).getInfo()
    print(f"   dw_trees_now mean: {dw_now}")
else:
    print("   DW instant: None (no clean pairs in window)")

# ── S2 NDVI ──────────────────────────────────────────────────
print("\n2. S2 NDVI...")
s2 = build_s2_ndvi_composite(gt_geom, ANCHOR, rc["s2_window"])
if s2:
    ndvi_c, ndvi_b = s2
    dndvi = ndvi_b.subtract(ndvi_c).rename("dNDVI")
    dndvi_m = dndvi.reduceRegion(ee.Reducer.mean(), gt_geom, 10).getInfo()
    ndvi_c_m = ndvi_c.reduceRegion(ee.Reducer.mean(), gt_geom, 10).getInfo()
    ndvi_b_m = ndvi_b.reduceRegion(ee.Reducer.mean(), gt_geom, 10).getInfo()
    print(f"   ndvi_current mean: {ndvi_c_m}")
    print(f"   ndvi_baseline mean: {ndvi_b_m}")
    print(f"   dNDVI mean (pos=loss): {dndvi_m}")
    
    # Phenology z-score
    model_dir = rc["phenology_model_dir"]
    if not Path(model_dir).is_absolute():
        model_dir = str(Path(__file__).resolve().parent.parent / model_dir)
    pheno = _load_range_phenology_model("Binaganj", model_dir, "ndvi")
    if pheno:
        zscore = build_ndvi_zscore_image(ndvi_c, ANCHOR, pheno)
        zm = zscore.reduceRegion(ee.Reducer.mean(), gt_geom, 10).getInfo()
        print(f"   NDVI z-score mean: {zm}")
else:
    print("   S2 NDVI: None")

# ── Forest mask ───────────────────────────────────────────────
print("\n3. Forest mask coverage...")
fmask = _forest_baseline_mask(gt_geom)
if fmask is not None:
    fpx = fmask.reduceRegion(ee.Reducer.sum(), gt_geom, 30).getInfo()
    ftot = fmask.unmask(0).reduceRegion(ee.Reducer.sum(), gt_geom, 30).getInfo()
    print(f"   forest pixels (masked): {fpx}")
    print(f"   total pixels: {ftot}")
    pct = {k: round(fpx.get(k, 0) / max(ftot.get(k, 1), 1) * 100, 1) for k in ftot}
    print(f"   forest cover pct: {pct}")
else:
    print("   forest mask: None")

# ── SAR ──────────────────────────────────────────────────────
print("\n4. SAR VH delta...")
dvh = build_s1_delta(gt_geom, ANCHOR, rc["s1_window"])
if dvh is not None:
    dvh_m = dvh.reduceRegion(ee.Reducer.mean(), gt_geom, 30).getInfo()
    dvh_p90 = dvh.reduceRegion(ee.Reducer.percentile([90]), gt_geom, 30).getInfo()
    print(f"   dVH mean (dB): {dvh_m}")
    print(f"   dVH p90 (dB): {dvh_p90}")
else:
    print("   SAR dVH: None")

print("\n=== Done ===")
