"""Quick live test of the phenology-integrated rules engine on Mar Ki Mahu beat."""
import sys, json, yaml, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

gee_project = cfg.get("gee", {}).get("gee_project", "van-suraksha-alert")
from src.data.gee_fetch import init_gee
init_gee(gee_project)
import ee

# ── Beat geometry ─────────────────────────────────────────────────────────────
with open(ROOT / "data/aoi/guna_beats.geojson") as f:
    fc = json.load(f)

beat_feat = None
for feat in fc["features"]:
    p = feat["properties"]
    if p.get("Beat", "").lower() == "mar_ki_mahu":
        beat_feat = feat
        break

if beat_feat is None:
    print("ERROR: Beat Mar_Ki_Mahu not found in guna_beats.geojson")
    sys.exit(1)

rng      = beat_feat["properties"]["Range"]
area_ha  = beat_feat["properties"].get("Beat_Ar", "?")
anchor   = "2026-02-10"

print(f"\nBeat   : Mar_Ki_Mahu")
print(f"Range  : {rng}")
print(f"Area   : {area_ha:.1f} ha")
print(f"Anchor : {anchor}")
print()

beat_geom = ee.Geometry(beat_feat["geometry"])

# ── Run rules engine ──────────────────────────────────────────────────────────
from src.inference.rules_engine import run_gee_rules_engine, _harmonic_predict, _load_range_phenology_model, _load_rules_cfg

# Show phenology expectation for this DOY
rc  = _load_rules_cfg(cfg)
mdl = _load_range_phenology_model(rng, str(ROOT / rc["phenology_model_dir"]), "ndvi")
if mdl:
    from datetime import datetime
    doy = datetime.strptime(anchor, "%Y-%m-%d").timetuple().tm_yday
    exp = _harmonic_predict(mdl["coeffs"], doy)
    sig = mdl["residual_std"]
    print(f"[phenology] DOY={doy}  expected_NDVI={exp:.4f}  sigma={sig:.4f}")
    print(f"[phenology] Alert fires if obs_NDVI < {exp + sig * rc['ndvi_z_thresh']:.4f}  (z<{rc['ndvi_z_thresh']})")
    print()

result = run_gee_rules_engine(
    beat_geom   = beat_geom,
    anchor_date = anchor,
    cfg         = cfg,
    range_label = rng,
)

# ── Print result ──────────────────────────────────────────────────────────────
print()
print("=" * 58)
tier_labels = {0: "No alert", 1: "TIER 1 - High confidence", 2: "TIER 2 - Medium", 3: "TIER 3 - SAR only"}
print(f"  RESULT     : {tier_labels.get(result['tier'], result['tier'])}")
print(f"  AREA_HA    : {result['area_ha']:.3f} ha")
print(f"  CONFIDENCE : {result['confidence']:.2f}")
if result["skipped"]:
    print(f"  ERROR      : {result['error']}")
print()
print("  SIGNALS:")
for k, v in (result.get("signals") or {}).items():
    if v is not None:
        if isinstance(v, float):
            print(f"    {k:<26s}: {v:.5f}")
        else:
            print(f"    {k:<26s}: {v}")
print("=" * 58)
