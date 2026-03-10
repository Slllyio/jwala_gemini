"""
Ground-truth validation: run the alert pipeline directly over the confirmed
Feb 5-20 2026 tree-loss polygon provided by the field team.

The KML polygon is at lon~77.274, lat~24.929 (Guna district).
We test all anchor dates bracketing the event (Feb 09, 12, 14, 17, 22)
and also construct a minimal AOI from the KML geometry itself so we deliberately
run sub-beat targeted detection.
"""
import sys, json, yaml, logging, xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(message)s")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.data.gee_fetch import init_gee
init_gee(cfg.get("gee", {}).get("gee_project", "van-suraksha-alert"))
import ee

# ── Parse KML polygon ────────────────────────────────────────────────────────
kml_path = ROOT / "data/ground_truth/confirmed_loss_feb2026.kml"
tree = ET.parse(kml_path)
ns = {"kml": "http://www.opengis.net/kml/2.2"}
coords_text = tree.find(".//kml:coordinates", ns)
if coords_text is None:
    # Try without namespace
    coords_text = tree.find(".//coordinates")

raw = coords_text.text.strip().split()
coords_ll = [[float(p.split(",")[0]), float(p.split(",")[1])] for p in raw]

# Build GEE geometry from KML polygon + small buffer (50m) to widen AOI
gt_geom = ee.Geometry.Polygon([coords_ll]).buffer(50)

# Also build a bounding-box buffer for faster candidate mask computation
bbox = gt_geom.bounds()

print(f"\n{'='*70}")
print(f"  GROUND TRUTH VALIDATION -- Confirmed loss Feb 05-20 2026")
print(f"  KML polygon: {len(coords_ll)} vertices, lon~77.274 lat~24.929")
print(f"{'='*70}\n")

from src.inference.rules_engine import run_gee_rules_engine

ANCHORS = [
    "2026-02-09",   # pair 07->09  (loss ongoing)
    "2026-02-12",   # pair 09->12  (loss ongoing)
    "2026-02-14",   # pair 12->14
    "2026-02-17",   # pair 14->17
    "2026-02-22",   # pair 17->22
]

TIER_LABELS = {
    0: "No alert",
    1: "TIER 1 - HIGH",
    2: "TIER 2 - MEDIUM",
    3: "TIER 3 - SAR only",
}

print(f"{'Anchor':<12} {'Tier':<20} {'Area':>7} {'Conf':>6}  {'dTrees':>8} {'NDVI_z':>7} {'Notes'}")
print("-" * 78)

for anchor in ANCHORS:
    print(f"  [Running] {anchor} ...", end="", flush=True)
    try:
        r = run_gee_rules_engine(
            beat_geom   = gt_geom,      # ← Use the actual ground-truth polygon
            anchor_date = anchor,
            cfg         = cfg,
            range_label = "Binaganj",   # Mar Ki Mahu is in Binaganj range
        )
        sig        = r.get("signals") or {}
        tier_str   = TIER_LABELS.get(r["tier"], str(r["tier"]))
        dw_d       = sig.get("dw_trees_delta_mean")
        ndvi_z     = sig.get("ndvi_zscore_mean")
        detected   = "  + DETECTED" if r["tier"] > 0 else "  x missed"
        pair_info  = f"DW {sig.get('dw_date_prev','?')}->{sig.get('dw_date_latest','?')}"

        dw_str   = f"{dw_d:>+8.3f}" if dw_d is not None else "       ?"
        ndvi_str = f"{ndvi_z:>+7.3f}" if ndvi_z is not None else "      ?"

        print(f"\r{anchor:<12} {tier_str:<20} {r['area_ha']:>6.2f}ha {r['confidence']:>5.2f}  "
              f"{dw_str} {ndvi_str}  {detected}   [{pair_info}]")
    except Exception as exc:
        print(f"\r{anchor:<12} ERROR: {exc}")

print("-" * 78)
print("\n  Expected: at least one TIER 1 or TIER 2 alert between Feb 09-17")
print("  (Loss confirmed Feb 05-20 per field team)\n")
