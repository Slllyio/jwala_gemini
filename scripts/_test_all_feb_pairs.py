"""
All-Feb DW pairs scan over the confirmed loss polygon.
Discovers ALL consecutive cloud-clean DW scene pairs in February 2026
and runs the full pipeline on each, using the KML polygon as the AOI.

The user confirmed loss between Feb 5-20. This script finds ALL pairs
in that window and prints per-pair signal values + alert tier.
"""
import sys, yaml, json, xml.etree.ElementTree as ET
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
cfg = yaml.safe_load(open("config.yaml"))

from src.data.gee_fetch import init_gee
init_gee(cfg.get("gee", {}).get("gee_project", "van-suraksha-alert"))
import ee

# ── Parse KML polygon ────────────────────────────────────────────────────────
kml_path = Path("data/ground_truth/confirmed_loss_feb2026.kml")
tree = ET.parse(kml_path)
root = tree.getroot()
for elem in root.iter():
    if "}" in elem.tag:
        elem.tag = elem.tag.split("}", 1)[1]
ct = root.find(".//coordinates")
coords_ll = [[float(p.split(",")[0]), float(p.split(",")[1])]
             for p in ct.text.strip().split()]

# Use the exact polygon + small buffer as AOI
gt_geom = ee.Geometry.Polygon([coords_ll]).buffer(50)
print(f"KML polygon: {len(coords_ll)} vertices")

from src.inference.rules_engine import _load_rules_cfg, build_dw_instant_delta

rc = _load_rules_cfg(cfg)

# ── Discover all clean DW pairs in Feb 2026 across the polygon ───────────────
# Load ALL Dynamic World scenes over the polygon for Feb 2026
dw_col = (
    ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
    .filterBounds(gt_geom)
    .filterDate("2026-01-25", "2026-02-25")  # ±buffer around Feb
    .select(["trees"])
)

scene_list = dw_col.aggregate_array("system:time_start").getInfo()
scene_ids  = dw_col.aggregate_array("system:index").getInfo()
import datetime as dt
scenes = sorted(zip(scene_list, scene_ids))

print(f"\nAll DW scenes found over polygon ({len(scenes)} total):")
for ts, sid in scenes:
    d = dt.datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
    print(f"  {d}  ({sid})")

# ── For each pair of consecutive scenes that are both cloud-clean, run pipeline
print("\n\nRunning pipeline once per 'latest' DW scene (it auto-selects the best N-1 pair):")
print("-" * 90)

# Build anchor dates from each clean scene as the 'latest' image
from src.inference.rules_engine import run_gee_rules_engine

# Unique dates
unique_dates = sorted(set(
    dt.datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
    for ts, _ in scenes
))
# Filter: only Feb 5 onwards (when the loss started)
anchor_dates = [d for d in unique_dates if "2026-02-05" <= d <= "2026-02-25"]

TIER_LABELS = {0: "No alert", 1: "TIER1-HIGH", 2: "TIER2-MED", 3: "TIER3-SAR"}

print(f"{'Anchor':<12} {'DW Pair':<26} {'Tier':<14} {'Area':>7} {'Conf':>6} {'dTrees':>8} {'NDVI_z':>8}")
print("-" * 90)

for anchor in anchor_dates:
    try:
        r = run_gee_rules_engine(
            beat_geom   = gt_geom,
            anchor_date = anchor,
            cfg         = cfg,
            range_label = "Binaganj",
        )
        sig = r.get("signals") or {}
        tier_str = TIER_LABELS.get(r["tier"], str(r["tier"]))
        dw_d   = sig.get("dw_trees_delta_mean")
        ndvi_z = sig.get("ndvi_zscore_mean")
        pair   = f"{sig.get('dw_date_prev','?')} -> {sig.get('dw_date_latest','?')}"
        mark   = "  <-- ALERT!" if r["tier"] > 0 else ""
        dw_str   = f"{dw_d:>+8.3f}" if dw_d is not None else "       ?"
        ndvi_str = f"{ndvi_z:>+8.3f}" if ndvi_z is not None else "       ?"
        print(f"{anchor:<12} {pair:<26} {tier_str:<14} {r['area_ha']:>6.2f}ha {r['confidence']:>5.2f} "
              f"{dw_str} {ndvi_str}{mark}")
    except Exception as exc:
        print(f"{anchor:<12} ERROR: {exc}")

print("-" * 90)
print("\nGround truth: tree loss confirmed Feb 05-20 (7-10 day clearance window)")
