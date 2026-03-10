"""
Dump alert GeoJSON for all clean pairs in Feb 2026.
Saves each anchor's patches to outputs/alerts/mar_ki_mahu_<anchor>.geojson
Also prints per-patch breakdown.
"""
import sys, json, yaml, logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(message)s")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "outputs" / "alerts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.data.gee_fetch import init_gee
init_gee(cfg.get("gee", {}).get("gee_project", "van-suraksha-alert"))

with open(ROOT / "data/aoi/guna_beats.geojson") as f:
    fc = json.load(f)

beat_feat = next(
    ft for ft in fc["features"]
    if ft["properties"].get("Beat", "").lower() == "mar_ki_mahu"
)
import ee
beat_geom = ee.Geometry(beat_feat["geometry"])
rng       = beat_feat["properties"]["Range"]

from src.inference.rules_engine import run_gee_rules_engine

ANCHORS = [
    "2026-02-09",
    "2026-02-12",
    "2026-02-14",
    "2026-02-17",
    "2026-02-22",
]

TIER_LABELS = {1: "TIER-1 HIGH", 2: "TIER-2 MEDIUM", 3: "TIER-3 SAR"}

for anchor in ANCHORS:
    print(f"\n[{anchor}] Running...", flush=True)
    r = run_gee_rules_engine(
        beat_geom   = beat_geom,
        anchor_date = anchor,
        cfg         = cfg,
        range_label = rng,
    )
    patches = r["geojson"]["features"]
    n       = len(patches)
    sig     = r.get("signals", {})

    print(f"  Tier={r['tier']}  area={r['area_ha']:.2f}ha  conf={r['confidence']:.2f}"
          f"  patches={n}  candidate={sig.get('n_patches_candidate','?')}")

    if n == 0:
        print("  No alert patches.")
        continue

    # Per-patch breakdown
    print(f"  {'#':<3} {'area_ha':>8} {'tier':>6} {'conf':>6} "
          f"{'dw_td':>7} {'dw_crop':>7} {'ndvi_z':>7} {'dVH':>7} {'cusum':>7}")
    print(f"  {'-'*70}")
    for i, feat in enumerate(patches):
        p = feat["properties"]
        print(
            f"  {i+1:<3} {p.get('area_ha', 0):>8.3f} "
            f"{TIER_LABELS.get(p.get('tier', 0), '?'):>12} "
            f"{p.get('confidence', 0):>6.2f} "
            f"{p.get('dw_trees_delta') if p.get('dw_trees_delta') is not None else 'n/a':>7} "
            f"{p.get('dw_crops_delta') if p.get('dw_crops_delta') is not None else 'n/a':>7} "
            f"{p.get('ndvi_zscore') if p.get('ndvi_zscore') is not None else 'n/a':>7} "
            f"{p.get('dVH') if p.get('dVH') is not None else 'n/a':>7} "
            f"{p.get('cusum_score') if p.get('cusum_score') is not None else 'n/a':>7}"
        )

    # Save GeoJSON
    out_path = OUT_DIR / f"mar_ki_mahu_{anchor}.geojson"
    with open(out_path, "w") as f:
        json.dump(r["geojson"], f, indent=2)
    print(f"  Saved: {out_path}")

print("\nDone.")
