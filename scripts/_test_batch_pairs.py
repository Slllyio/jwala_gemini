"""
Batch alert check for all clean DW consecutive pairs over Mar Ki Mahu, Feb 2026.

Pairs:
  07->09  anchor = 2026-02-09  (already tested, included for reference)
  09->12  anchor = 2026-02-12
  12->14  anchor = 2026-02-14
  14->17  anchor = 2026-02-17
  17->22  anchor = 2026-02-22
"""
import sys, json, yaml, logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.WARNING, format="%(message)s")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.data.gee_fetch import init_gee
init_gee(cfg.get("gee", {}).get("gee_project", "van-suraksha-alert"))
import ee

with open(ROOT / "data/aoi/guna_beats.geojson") as f:
    fc = json.load(f)

beat_feat = next(
    ft for ft in fc["features"]
    if ft["properties"].get("Beat", "").lower() == "mar_ki_mahu"
)
rng       = beat_feat["properties"]["Range"]
beat_geom = ee.Geometry(beat_feat["geometry"])

from src.inference.rules_engine import run_gee_rules_engine

ANCHORS = [
    "2026-02-09",   # pair 07->09
    "2026-02-12",   # pair 09->12
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

results = []

for anchor in ANCHORS:
    print(f"\n[Running] anchor = {anchor} ...", flush=True)
    try:
        r = run_gee_rules_engine(
            beat_geom   = beat_geom,
            anchor_date = anchor,
            cfg         = cfg,
            range_label = rng,
        )
        sig = r.get("signals") or {}
        results.append({
            "anchor":        anchor,
            "tier":          r["tier"],
            "area_ha":       r["area_ha"],
            "conf":          r["confidence"],
            "pair":          f"{sig.get('dw_date_prev','?')} -> {sig.get('dw_date_latest','?')}",
            "interval":      sig.get("dw_interval_days", -1),
            "cloud_prev":    sig.get("cloud_prev_pct", -1),
            "cloud_latest":  sig.get("cloud_latest_pct", -1),
            "dw_trees_d":    sig.get("dw_trees_delta_mean"),
            "dw_crop_d":     sig.get("dw_crops_delta_mean"),
            "dw_built_d":    sig.get("dw_built_delta_mean"),
            "dw_zscore":     sig.get("dw_trees_zscore_mean"),
            "ndvi_zscore":   sig.get("ndvi_zscore_mean"),
            "dndvi":         sig.get("dNDVI_mean"),
            "dvh":           sig.get("dVH_mean_db"),
            "cusum":         sig.get("cusum_mean"),
            "radd":          sig.get("radd_alert", False),
            "ndvi_src":      sig.get("ndvi_source", "?"),
            "skipped":       r.get("skipped", False),
            "error":         r.get("error", ""),
        })
        tier_str = TIER_LABELS.get(r["tier"], str(r["tier"]))
        print(f"  Result: {tier_str}  area={r['area_ha']:.2f}ha  conf={r['confidence']:.2f}")
    except Exception as exc:
        print(f"  ERROR: {exc}")
        results.append({"anchor": anchor, "tier": -1, "error": str(exc), "skipped": True})

# ── Print comparison table ────────────────────────────────────────────────────
SEP = "-" * 90
print(f"\n\n{'='*90}")
print(f"  BATCH RESULTS -- Mar Ki Mahu -- Feb 2026 consecutive clean DW pairs")
print(f"{'='*90}")
print(f"{'Anchor':<12} {'DW Pair':<24} {'Int':>4} {'cld_p':>6} {'cld_n':>6} "
      f"{'dTrees':>8} {'dCrops':>7} {'dVH_dB':>7} {'CuSum':>6} {'NDVI_z':>7} {'Result'}")
print(SEP)

for r in results:
    if r.get("skipped"):
        print(f"  {r['anchor']}  ERROR: {r.get('error','?')[:60]}")
        continue
    def fmt(v, d=3, w=8):
        return f"{v:>{w}.{d}f}" if isinstance(v, (int, float)) and v is not None else f"{'?':>{w}}"
    def fmtc(v, d=1, w=5):
        return f"{v:>{w}.{d}f}" if isinstance(v, (int, float)) and v is not None else f"{'?':>{w}}"
    def fmti(v, w=4):
        return f"{v:>{w}d}" if isinstance(v, int) and v is not None else f"{'?':>{w}}"

    tier_str = TIER_LABELS.get(r["tier"], "?")
    alert_mark = "  <-- ALERT" if r["tier"] > 0 else ""
    pair = r.get('pair', '? -> ?')
    print(
        f"{r['anchor']:<12} {pair:<24} {fmti(r.get('interval'))}d "
        f"{fmtc(r.get('cloud_prev'))}% {fmtc(r.get('cloud_latest'))}%  "
        f"{fmt(r.get('dw_trees_d'), 3, 8)}  {fmt(r.get('dw_crop_d'), 3, 7)} "
        f"{fmt(r.get('dvh'), 2, 7)}  {fmt(r.get('cusum'), 3, 6)}  {fmt(r.get('ndvi_zscore'), 3, 7)}  "
        f"{tier_str}{alert_mark}"
    )

print(SEP)
print(f"\n  Header key: Int=interval days  cld_p=cloud% prev pass  cld_n=cloud% latest pass")
print(f"  dTrees=DW trees delta (neg=loss)  dCrops=DW crops delta  dVH_dB=SAR drop")
print(f"  CuSum=SAR CuSum score (>0.5=persistent loss)  NDVI_z=phenology z-score")
print()
