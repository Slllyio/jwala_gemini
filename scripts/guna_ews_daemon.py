"""
guna_ews_daemon.py
==================
Stateful Micro-Batch Early Warning System (EWS) Orchestrator
for the Guna Forest Division, Madhya Pradesh.

Runs daily across all 80–100+ beats. For each beat:
  1. Queries GEE for the two most recent cloud-clean DW passes
  2. Skips the beat if already processed (SQLite state tracker — idempotent)
  3. Harvests multi-class DW deltas, cleans with morphological opening, vectorises
  4. Scores every candidate polygon with dw_multi_threat_score_maxx (SOTA V3)
  5. Routes MEDIUM/HIGH alerts to a dated GeoJSON file and the PostGIS alerts_log table

Usage:
  python scripts/guna_ews_daemon.py                  # Full division run
  python scripts/guna_ews_daemon.py --dry-run        # Score + log, no DB/state writes
  python scripts/guna_ews_daemon.py --beat "Goumukh" # Single beat (for testing)
  python scripts/guna_ews_daemon.py --date 2026-01-15 # Backfill to a past date
  python scripts/guna_ews_daemon.py --dry-run --beat "Goumukh" --date 2026-01-15
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

# psycopg2 and yaml are optional at import time:
# - yaml is only needed if config.yaml exists
# - psycopg2 is only used inside push_to_postgres() which is never called in --dry-run
try:
    import psycopg2
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False
    warnings.warn("psycopg2 not installed — PostGIS push will be skipped.")

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    _yaml = None   # type: ignore[assignment]

# Defer EE import so --help/--dry-run works without credentials
try:
    import ee
    _EE_AVAILABLE = True
except ImportError:
    _EE_AVAILABLE = False
    warnings.warn("earthengine-api not installed — GEE harvest will fail.")

# dw_multi_threat_score_maxx is imported locally inside score_patches() — not at module level

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Allow: `python scripts/guna_ews_daemon.py` from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR  = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ews_daemon")

# ── Load config.yaml ──────────────────────────────────────────────────────────
_CFG_PATH = _PROJECT_ROOT / "config.yaml"
try:
    if _YAML_AVAILABLE and _yaml is not None:
        with open(_CFG_PATH) as _f:
            _CFG = _yaml.safe_load(_f)
    else:
        raise FileNotFoundError
except FileNotFoundError:
    _CFG = {}
    log.warning("config.yaml not found — using hard-coded defaults.")


_DB_CFG = _CFG.get("database", {})

# ── Constants (overridable via env vars for CI/cloud deploys) ─────────────────
BEATS_GEOJSON  = str(_PROJECT_ROOT / "data" / "aoi" / "guna_beats.geojson")
STATE_DB       = str(_PROJECT_ROOT / "outputs" / "guna_pipeline_state.db")
FEED_DIR       = str(_PROJECT_ROOT / "outputs" / "dashboard_feed")
LOG_DIR        = str(_PROJECT_ROOT / "outputs" / "logs")
MIN_AREA_HA    = 0.10       # loose at GEE stage; SOTA scorer is the strict filter
LOOKBACK_DAYS  = 45         # search window for DW passes (45d covers foggy MP Rabi season)
GEE_CLOUD_MAX  = 30         # max cloud % for a DW image to be considered clean
GEE_ENDPOINT   = "https://earthengine-highvolume.googleapis.com"
MODEL_VERSION  = "dw_multi_threat_v5_deseason"   # V5: Fix A-E deseasonalization, HIGH=0.70
SOURCE_TAG     = "ews_daemon_v5"

# PostGIS DSN — reads from config.yaml, overridable by DATABASE_URL env var
_DEFAULT_DSN = (
    f"dbname={_DB_CFG.get('dbname','gis_projects')} "
    f"user={_DB_CFG.get('user','postgres')} "
    f"password={os.environ.get('VS_DB_PASSWORD', _DB_CFG.get('password', ''))} "
    f"host={_DB_CFG.get('host','localhost')} "
    f"port={_DB_CFG.get('port',5432)}"
)
DB_DSN = os.getenv("DATABASE_URL", _DEFAULT_DSN)

# Typology → alerts_log.change_type mapping
_TYPOLOGY_MAP = {
    "CANOPY_LOSS":       "canopy_loss",
    "CROP_ENCROACHMENT": "crop_encroachment",
    "BUILT_ENCROACHMENT":"built_encroachment",
}


# ── GEE initialisation ────────────────────────────────────────────────────────

def init_gee(project: str | None = None) -> None:
    """Initialise the Earth Engine Python API with the high-volume endpoint."""
    if not _EE_AVAILABLE:
        raise RuntimeError("earthengine-api is not installed.")
    proj = project or _CFG.get("gee", {}).get("gee_project", "van-suraksha-alert")
    ee.Initialize(
        project=proj,
        opt_url=GEE_ENDPOINT,
    )
    log.info(f"GEE initialised → project={proj}  endpoint=high-volume")


# ── SQLite state tracker ─────────────────────────────────────────────────────

def init_state_db() -> sqlite3.Connection:
    """
    Create (or open) the per-beat state SQLite database.
    Schema: beat_state(beat_name PK, last_processed_date TEXT)
    """
    os.makedirs(os.path.dirname(STATE_DB), exist_ok=True)
    conn = sqlite3.connect(STATE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS beat_state (
            beat_name            TEXT PRIMARY KEY,
            range_name           TEXT,
            last_processed_date  TEXT,
            last_t0_date         TEXT,
            last_n_alerts        INTEGER DEFAULT 0,
            updated_at           TEXT
        )
    """)
    conn.commit()
    return conn


# ── GEE: Dynamic Cloud-Gated Time Compositing with YoY Fallback ───────────────
#
# SOTA design — avoids two critical traps:
#   1. Scene-level cloud metadata (useless when fog is localised to a beat)
#   2. Seasonal Gap Phenology Trap: if t0↔t1 gap > 45 days, deciduous canopy
#      in MP 5B dry-deciduous forest will show -45% ΔTrees purely from leaf-drop,
#      generating hundreds of phantom HIGH alerts.
#
# Strategy (Bifurcated Anchoring):
#   • Measure cloud % STRICTLY inside the beat polygon at 50 m from S2_CP
#   • Find t1 = most recent clear pass in last MAX_LOOKBACK_DAYS
#   • If second clear pass is ≤ GAP_SAFE_DAYS away → use Instant Delta (safe)
#   • Otherwise → pivot to Year-on-Year baseline (same phenological state)

MAX_LOOKBACK_DAYS = 90    # Maximum backward search window
GAP_SAFE_DAYS     = 45    # Max gap for Instant Delta; beyond this use YoY
YOY_WINDOW_DAYS   = 20    # ±days around the 1-year-ago target for YoY search


def get_latest_passes(
    beat_geom:     "ee.Geometry",
    override_date: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Dynamic Cloud-Gated Seeker + YoY Fallback.

    Returns (t1_date, t0_date) for bi-temporal change detection.
    t0 is either the previous clear pass (gap ≤ GAP_SAFE_DAYS)
    or the best cloud-free pass from exactly 1 year ago (YoY fallback).

    Returns (None, None) if no usable pair can be found.
    """
    anchor = override_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _clear_dates_in_window(win_start: str, win_end: str) -> list[str]:
        """
        DW-anchored cloud gate using S2_SR_HARMONIZED SCL band.
        For each DW date in the window, compute beat-level cloud % from
        Sentinel-2 SCL (classes 8=cloud-medium, 9=cloud-high, 10=cirrus).
        Uses a ±1-day margin to handle tile-edge acquisition timing differences.
        Returns dates sorted newest-first where SCL cloud% < GEE_CLOUD_MAX.
        """
        # DW anchors the valid analysis dates
        dw_in_window = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(beat_geom)
            .filterDate(win_start, win_end)
        )
        # S2 SR for SCL cloud classification — pre-filter to window (+1 day buffer)
        s2_in_window = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(beat_geom)
            .filterDate(win_start, win_end)
            .select("SCL")
        )

        def check_cloud(dw_img: "ee.Image") -> "ee.Feature":
            date_str = dw_img.date().format("YYYY-MM-dd")
            # ±1 day margin handles tile-edge timing differences between DW and S2
            s2_day = s2_in_window.filterDate(
                dw_img.date().advance(-1, "day"),
                dw_img.date().advance( 2, "day"),
            )
            s2_scl = ee.Image(ee.Algorithms.If(
                s2_day.size().gt(0),
                s2_day.mosaic().select("SCL"),
                ee.Image.constant(9).rename("SCL"),   # no S2 data → conservatively cloudy
            ))
            # Fraction of beat pixels classified as cloud (SCL 8–10)
            cloud_frac = (
                s2_scl.gte(8).And(s2_scl.lte(10))
                .reduceRegion(
                    reducer   = ee.Reducer.mean(),
                    geometry  = beat_geom,
                    scale     = 100,
                    maxPixels = 1e6,
                ).get("SCL")
            )
            return ee.Feature(None, {
                "date":      date_str,
                "cloud_pct": ee.Number(cloud_frac).multiply(100),  # 0–100 scale
            })

        clear_fc = (
            dw_in_window.map(check_cloud)
            .filter(ee.Filter.lt("cloud_pct", GEE_CLOUD_MAX))
        )
        try:
            return (
                clear_fc
                .aggregate_array("date")
                .distinct()
                .sort()
                .reverse()
                .getInfo()
            )
        except Exception:
            return []

    try:
        # ── STEP 1: find clear dates in recent window ──────────────────────
        end_dt   = ee.Date(anchor).advance(1, "day")
        start_dt = end_dt.advance(-MAX_LOOKBACK_DAYS, "day")
        # Convert back to strings for filtering
        win_end   = ee.Date(anchor).advance(1, "day").format("YYYY-MM-dd").getInfo()
        win_start = ee.Date(anchor).advance(1 - MAX_LOOKBACK_DAYS, "day").format("YYYY-MM-dd").getInfo()

        recent = _clear_dates_in_window(win_start, win_end)

        if not recent:
            log.info(f"  No clean imagery in last {MAX_LOOKBACK_DAYS} days. Skipping.")
            return None, None

        t1 = recent[0]
        log.info(f"  🛰  t1 anchored: {t1}  ({len(recent)} clear pass(es) in window)")

        # ── STEP 2: Bifurcation check ──────────────────────────────────────
        if len(recent) >= 2:
            t0_candidate = recent[1]
            d1  = datetime.strptime(t1, "%Y-%m-%d")
            d0  = datetime.strptime(t0_candidate, "%Y-%m-%d")
            gap = (d1 - d0).days

            if gap <= GAP_SAFE_DAYS:
                log.info(f"  ✅ Instant Delta: {t0_candidate} → {t1}  (gap={gap}d ≤ {GAP_SAFE_DAYS}d, phenologically safe)")
                return t1, t0_candidate
            else:
                log.info(f"  ⚠️  Gap {gap}d > {GAP_SAFE_DAYS}d — Seasonal Gap Trap avoided. Pivoting to YoY.")
        else:
            log.info("  Only 1 clear pass found recently — pivoting to YoY baseline.")

        # ── STEP 3: Year-on-Year fallback ──────────────────────────────────
        t1_obj = datetime.strptime(t1, "%Y-%m-%d")
        try:
            yoy_target = t1_obj.replace(year=t1_obj.year - 1)
        except ValueError:  # Feb 29 in non-leap year
            yoy_target = t1_obj.replace(year=t1_obj.year - 1, day=28)

        yoy_start = (yoy_target - timedelta(days=YOY_WINDOW_DAYS)).strftime("%Y-%m-%d")
        yoy_end   = (yoy_target + timedelta(days=YOY_WINDOW_DAYS)).strftime("%Y-%m-%d")

        yoy_dates = _clear_dates_in_window(yoy_start, yoy_end)

        if yoy_dates:
            # Pick date with smallest day-of-year distance to t1 (same phenological stage)
            target_doy = t1_obj.timetuple().tm_yday
            best_yoy   = min(
                yoy_dates,
                key=lambda d: abs(datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday - target_doy),
            )
            log.info(f"  📅 YoY baseline locked: {best_yoy}  (same phenophase, 1yr prior)")
            return t1, best_yoy

        log.info(f"  No clear YoY baseline in {yoy_start}–{yoy_end}. Skipping.")
        return None, None

    except Exception as exc:
        log.warning(f"      [GEE cloud-gate] {exc}")
        return None, None


# ── GEE: harvest, clean, vectorise, sample ────────────────────────────────────

def process_beat_in_gee(
    beat_geom: "ee.Geometry",
    t0_date:   str,
    t1_date:   str,
) -> list[dict]:
    """
    Server-side GEE pipeline:
      1. Build per-date DW mosaics (handles S2 swath overlaps on same day)
      2. Read cloud fraction for T1 from S2_CLOUD_PROBABILITY
      3. Compute multi-class delta: trees, crops, built
      4. Build permissive union mask (any signal above loose threshold)
      5. Morphological opening (radius=1 px) — kills 1-pixel orbital jitter
      6. Vectorise at native 10m (8-connected component labelling)
      7. Sample area + band means per polygon
      8. Filter by minimum area (MIN_AREA_HA)

    Returns list of GeoJSON feature dicts (may be empty).
    """
    BANDS = ["trees", "crops", "built"]

    dw_col = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(beat_geom)

    t0_img = (
        dw_col
        .filterDate(t0_date, ee.Date(t0_date).advance(1, "day"))
        .mosaic()
        .select(BANDS)
    )
    t1_img = (
        dw_col
        .filterDate(t1_date, ee.Date(t1_date).advance(1, "day"))
        .mosaic()
        .select(BANDS)
    )

    # Per-pixel cloud probability for T1 (0–100 → 0–1 fraction)
    cloud_frac_img = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(beat_geom)
        .filterDate(t1_date, ee.Date(t1_date).advance(1, "day"))
        .mosaic()
        .select("probability")
        .divide(100.0)
        .unmask(0.0)
        .rename("cloud_frac")
    )

    delta = t1_img.subtract(t0_img)

    # FIX C: Deseasonalized candidate mask
    # Subtract expected phenological change so natural Feb/Mar leaf-fall
    # does not vectorize the entire forest into one 1000+ ha polygon.
    # GREEN-UP CLAMP: min(0.0, ...) — only forgive expected drops (senescence).
    # Stable bare pixels during monsoon green-up are NOT penalised for
    # "failing to green up" (the Spring-Time Hallucination trap).
    from datetime import datetime as _dt  # noqa: local import to avoid circular
    try:
        from _simple_dw_score import _doy_baseline, DW_MONTHLY_MEANS
        _t0_doy = _dt.strptime(t0_date, "%Y-%m-%d").timetuple().tm_yday
        _t1_doy = _dt.strptime(t1_date, "%Y-%m-%d").timetuple().tm_yday
        _expected_change = _doy_baseline(_t1_doy, DW_MONTHLY_MEANS) - _doy_baseline(_t0_doy, DW_MONTHLY_MEANS)
        _expected_tree_delta = min(0.0, _expected_change)  # GREEN-UP CLAMP
    except Exception:
        _expected_tree_delta = 0.0   # fallback: no deseasonalization

    adj_tree_delta = delta.select("trees").subtract(ee.Number(_expected_tree_delta))

    # Permissive union mask — loose thresholds; SOTA scorer is the gatekeeper
    candidate_mask = (
        adj_tree_delta.lt(-0.06)                   # anomalous drop beyond phenology
        .Or(delta.select("crops").gt(0.08))        # crops rising  > 8pp
        .Or(delta.select("built").gt(0.06))        # built rising  > 6pp
    )

    # Morphological opening (erode then dilate, radius=1 px at 10m)
    # Removes 1-pixel speckle and noise filaments; keeps ≥ 3-pixel-wide patches
    kernel     = ee.Kernel.circle(radius=1)
    clean_mask = candidate_mask.focal_min(kernel=kernel).focal_max(kernel=kernel)

    # Vectorise at native 10m; 8-connected topology preserves diagonal edges
    vectors = clean_mask.selfMask().reduceToVectors(
        reducer       = ee.Reducer.countEvery(),
        geometry      = beat_geom,
        scale         = 10,
        maxPixels     = 1e8,
        geometryType  = "polygon",
        eightConnected= True,
        bestEffort    = True,
    )

    # Per-polygon sampling — ONE server-side map(), ONE getInfo()
    def sample_patch(feat: "ee.Feature") -> "ee.Feature":
        area_ha = feat.geometry().area(maxError=10).divide(10000)
        stack = ee.Image.cat([
            t0_img.rename(["trees_before", "crops_before", "built_before"]),
            t1_img.rename(["trees_after",  "crops_after",  "built_after"]),
            cloud_frac_img,
        ])
        stats = stack.reduceRegion(
            reducer   = ee.Reducer.mean(),
            geometry  = feat.geometry(),
            scale     = 10,
            maxPixels = 1e6,
        )
        return feat.set(stats).set("area_ha", area_ha)

    sampled = (
        vectors
        .map(sample_patch)
        .filter(ee.Filter.gte("area_ha", MIN_AREA_HA))
    )

    try:
        return sampled.getInfo().get("features", [])
    except Exception as exc:
        log.warning(f"      [GEE getInfo] {exc} — returning empty")
        return []


# ── Python scoring ────────────────────────────────────────────────────────────

def score_patches(
    raw_patches: list[dict],
    t1_date:     str,
    t0_date:     str,
    beat_name:   str,
    range_name:  str,
    division:    str,
) -> list[dict]:
    """
    Apply dw_multi_threat_score_maxx to each raw GEE polygon.
    Returns only patches with label MEDIUM or HIGH.

    V4 FIX C — Reconstruction Trap (implemented):
    -------------------------------------------------
    `sample_patch()` in `process_beat_in_gee()` already samples the ACTUAL
    DW per-pixel mean for trees/crops/built at both t0 and t1 via
    `reduceRegion(ee.Reducer.mean())` over t0_img and t1_img.  Those values
    arrive here as `trees_before`, `trees_after` etc. in the GEE properties.

    We pass them DIRECTLY into the scorer — NO μ+Δ reconstruction.
    This means trees_before reflects the true pristine stand density, not the
    range seasonal mean.  A Teak stand at 0.85 stays at 0.85.

    Additionally, we compute the pixel-level z-score from the sampled values
    using the range harmonic model's residual std as the denominator (best
    available per-date σ).  This is passed as `patch_z` for the V4 boost.
    """
    try:
        from _simple_dw_score import dw_multi_threat_score_maxx, _range_baseline
    except ImportError:
        log.error("Cannot import scorer — score_patches returning empty")
        return []

    alerts: list[dict] = []

    for patch in raw_patches:
        p = patch.get("properties", {})

        # Guard: all required bands must be present (may be null if masked)
        required = ["trees_after", "trees_before", "crops_after", "crops_before",
                    "built_after",  "built_before"]
        if any(p.get(k) is None for k in required):
            continue

        # ── V4 FIX C: use SAMPLED absolute values directly ────────────────────
        trees_before = float(p["trees_before"])
        trees_after  = float(p["trees_after"])
        crops_before = float(p["crops_before"])
        crops_after  = float(p["crops_after"])
        built_before = float(p["built_before"])
        built_after  = float(p["built_after"])
        cloud_frac   = float(p.get("cloud_frac") or 0.0)

        # Derived delta fields (for GeoJSON output & downstream consumers)
        dw_trees_delta = round(trees_after  - trees_before,  4)
        dw_crops_delta = round(crops_after  - crops_before,  4)
        dw_built_delta = round(built_after  - built_before,  4)

        # FIX B: Correct patch_z — state anomaly, not velocity.
        # z-score measures how anomalously LOW trees_after is vs the seasonal
        # expectation (division monthly table). Uses division-level std ≈ 0.14.
        # This feeds the V4 directional boost (fires only when patch_z < -1.96).
        try:
            from _simple_dw_score import _doy_baseline, _doy_std, DW_MONTHLY_MEANS, DW_MONTHLY_STDS
            from datetime import datetime as _dt
            _t1_doy = _dt.strptime(t1_date, "%Y-%m-%d").timetuple().tm_yday
            _div_mu  = _doy_baseline(_t1_doy, DW_MONTHLY_MEANS)
            _div_std = max(_doy_std(_t1_doy, DW_MONTHLY_STDS), 0.05)
            dw_trees_zscore = round((trees_after - _div_mu) / _div_std, 3)
        except Exception:
            dw_trees_zscore = 0.0

        # CuSuM score (if computed upstream and stored in GEE properties)
        cusum_score = float(p.get("cusum_score") or 0.0)

        score_data = dw_multi_threat_score_maxx(
            trees_after  = trees_after,
            trees_before = trees_before,
            crops_after  = crops_after,
            crops_before = crops_before,
            built_after  = built_after,
            built_before = built_before,
            date_str     = t1_date,
            cloud_frac   = cloud_frac,
            range_name   = range_name,
            patch_z      = dw_trees_zscore,   # pixel z for V4 directional boost
            cusum_score  = cusum_score,        # multi-pass accumulation
            t0_date_str  = t0_date,            # FIX A: deseasonalization
        )

        if score_data["label"] in ("HIGH", "MEDIUM"):
            p.update(score_data)
            # ── Write all sampled + derived fields into the alert record ──────
            # These fields allow the map builder / dashboard to use actual
            # sampled states instead of reconstructing from delta+μ.
            p.update({
                # Sampled absolute values (FIX C: primary source)
                "trees_before":    round(trees_before, 4),
                "trees_after":     round(trees_after,  4),
                "crops_before":    round(crops_before, 4),
                "crops_after":     round(crops_after,  4),
                "built_before":    round(built_before, 4),
                "built_after":     round(built_after,  4),
                # Derived fields for backward compat & V4 boost
                "dw_trees_delta":  dw_trees_delta,
                "dw_crops_delta":  dw_crops_delta,
                "dw_built_delta":  dw_built_delta,
                "dw_trees_zscore": dw_trees_zscore,
                "cusum_score":     cusum_score,
                # Provenance
                "beat_name":       beat_name,
                "range_name":      range_name,
                "division":        division,
                "detection_date":  t1_date,
                "detection_period": f"{t0_date} to {t1_date}",  # FIX D: traceability
                "baseline_date":   t0_date,
                "status":          "PENDING_VERIFICATION",
                "model_version":   MODEL_VERSION,
                "source":          SOURCE_TAG,
            })
            patch["properties"] = p
            alerts.append(patch)

    return alerts



# ── Output: GeoJSON file ──────────────────────────────────────────────────────

def push_to_geojson(alerts: list[dict]) -> str:
    """Write dated GeoJSON file to outputs/dashboard_feed/."""
    os.makedirs(FEED_DIR, exist_ok=True)
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    filename = os.path.join(FEED_DIR, f"guna_alerts_{ts}.geojson")
    payload  = {"type": "FeatureCollection", "features": alerts}
    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    log.info(f"  GeoJSON → {filename}  ({len(alerts)} features)")
    return filename


# ── Output: PostGIS alerts_log ────────────────────────────────────────────────

_INSERT_SQL = """
    INSERT INTO alerts_log (
        beat_name,
        change_type,
        confidence,
        stacked_score,
        area_ha,
        detection_date,
        source,
        model_version,
        mean_delta_trees,
        mean_delta_crops,
        mean_delta_built,
        fast_tracked,
        sar_boost_applied,
        geom
    )
    VALUES (
        %(beat_name)s,
        %(change_type)s,
        %(confidence)s,
        %(stacked_score)s,
        %(area_ha)s,
        %(detection_date)s,
        %(source)s,
        %(model_version)s,
        %(mean_delta_trees)s,
        %(mean_delta_crops)s,
        %(mean_delta_built)s,
        FALSE,
        FALSE,
        ST_SetSRID(ST_GeomFromGeoJSON(%(geom_json)s), 4326)
    )
    ON CONFLICT DO NOTHING;
"""


def push_to_postgres(alerts: list[dict], dry_run: bool = False) -> None:
    """
    Insert verified alerts into the existing PostGIS alerts_log table.
    Uses ST_GeomFromGeoJSON — no Shapely/GeoPandas required.
    Maps scorer output dict to the actual column names in alerts_log.
    """
    if dry_run:
        log.info(f"  [dry-run] Skipping PostGIS insert for {len(alerts)} alerts.")
        return
    if not alerts:
        return

    rows_inserted = 0
    try:
        with psycopg2.connect(DB_DSN) as conn:
            with conn.cursor() as cur:
                for feat in alerts:
                    p = feat["properties"]
                    # delta values: after − before (negative for loss)
                    cur.execute(_INSERT_SQL, {
                        "beat_name":        p.get("beat_name", ""),
                        "change_type":      _TYPOLOGY_MAP.get(
                                                p.get("typology",""),
                                                p.get("typology","unknown")
                                            ),
                        "confidence":       p.get("score", 0.0),
                        "stacked_score":    p.get("evidence", 0.0),
                        "area_ha":          p.get("area_ha", 0.0),
                        "detection_date":   p.get("detection_date"),
                        "source":           SOURCE_TAG,
                        "model_version":    MODEL_VERSION,
                        "mean_delta_trees": (
                            (p.get("trees_after") or 0) -
                            (p.get("trees_before") or 0)
                        ),
                        "mean_delta_crops": (
                            (p.get("crops_after") or 0) -
                            (p.get("crops_before") or 0)
                        ),
                        "mean_delta_built": (
                            (p.get("built_after") or 0) -
                            (p.get("built_before") or 0)
                        ),
                        "geom_json": json.dumps(feat["geometry"]),
                    })
                    rows_inserted += 1
            conn.commit()
        log.info(f"  PostGIS ← inserted {rows_inserted} rows into alerts_log.")
    except psycopg2.Error as exc:
        log.error(f"  PostGIS error: {exc}")


# ── Main orchestration loop ───────────────────────────────────────────────────

def run_division_automation(args: argparse.Namespace) -> None:
    """
    Main loop: iterate over all Guna beats, check state, harvest, score, push.
    """
    log.info("=" * 70)
    log.info(f"GUNA EWS DAEMON  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    if args.dry_run:
        log.info("MODE: DRY-RUN  (no state updates, no DB inserts)")
    log.info("=" * 70)

    init_gee()
    state_conn   = init_state_db()
    state_cursor = state_conn.cursor()

    # Load beats GeoJSON (local file — always available even if GEE is down)
    with open(BEATS_GEOJSON, encoding="utf-8") as fp:
        beats = json.load(fp)["features"]

    # If --beat filter is set, restrict to matching beats
    if args.beat:
        beats = [b for b in beats
                 if args.beat.lower() in b["properties"].get("Beat", "").lower()]
        if not beats:
            log.error(f"No beat found matching --beat '{args.beat}'. Exiting.")
            return
        log.info(f"Beat filter active: {len(beats)} beat(s) selected.")

    total         = len(beats)
    all_alerts:   list[dict] = []

    for idx, beat in enumerate(beats, start=1):
        props      = beat.get("properties", {})
        beat_name  = props.get("Beat",     f"Unknown_Beat_{idx}")
        range_name = props.get("Range",    "Unknown_Range")
        division   = props.get("Division", "Guna")

        log.info(f"[{idx}/{total}]  {division} / {range_name} / {beat_name}")

        # Build GEE geometry from the GeoJSON feature geometry
        geom_type = beat["geometry"]["type"]
        coords    = beat["geometry"]["coordinates"]
        if geom_type == "Polygon":
            beat_geom = ee.Geometry.Polygon(coords)
        elif geom_type == "MultiPolygon":
            beat_geom = ee.Geometry.MultiPolygon(coords)
        else:
            log.warning(f"  Unsupported geometry type {geom_type}. Skipping.")
            continue

        # 1. Find most-recent satellite passes
        t1_date, t0_date = get_latest_passes(beat_geom, args.date)
        if not t1_date:
            log.info(f"  No clean imagery in last {LOOKBACK_DAYS} days. Skipping.")
            continue

        # 2. Idempotency check (skip if already processed this pass)
        if not args.dry_run and not args.date:
            state_cursor.execute(
                "SELECT last_processed_date FROM beat_state WHERE beat_name=?",
                (beat_name,)
            )
            row = state_cursor.fetchone()
            if row and row[0] and row[0] >= t1_date:
                log.info(f"  ✓ Up to date — last processed {row[0]}. Skipping.")
                continue

        log.info(f"  🛰  New imagery: {t0_date} → {t1_date}")

        # 3. GEE harvest
        raw_patches = process_beat_in_gee(beat_geom, t0_date, t1_date)
        log.info(f"  GEE returned {len(raw_patches)} candidate polygons.")

        if not raw_patches:
            beat_alerts = []
        else:
            # 4. SOTA V3 scoring (pure Python, no GEE)
            beat_alerts = score_patches(
                raw_patches = raw_patches,
                t1_date     = t1_date,
                t0_date     = t0_date,
                beat_name   = beat_name,
                range_name  = range_name,
                division    = division,
            )

        if beat_alerts:
            top  = beat_alerts[0]["properties"]
            log.info(
                f"  🚨 {len(beat_alerts)} alert(s)  |  "
                f"top threat: {top['typology']}  score={top['score']}"
            )
            all_alerts.extend(beat_alerts)
        else:
            log.info("  ✅ Clean — no actionable alerts.")

        # 5. Update state tracker (per-beat, commit immediately)
        if not args.dry_run:
            state_cursor.execute(
                """INSERT OR REPLACE INTO beat_state
                   (beat_name, range_name, last_processed_date, last_t0_date,
                    last_n_alerts, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    beat_name,
                    range_name,
                    t1_date,
                    t0_date,
                    len(beat_alerts),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            state_conn.commit()

        # Polite GEE back-off — avoids HTTP 429 "Too Many Requests"
        time.sleep(1.5)

    # ── Final export ─────────────────────────────────────────────────────────
    log.info("=" * 70)
    if all_alerts:
        geojson_path = push_to_geojson(all_alerts)
        push_to_postgres(all_alerts, dry_run=args.dry_run)
        log.info(
            f"DONE  |  {len(all_alerts)} alert(s) across "
            f"{len(set(a['properties']['beat_name'] for a in all_alerts))} beat(s)"
        )
        log.info(f"GeoJSON: {geojson_path}")
    else:
        log.info("DONE  |  No new alerts across the division.")
    log.info("=" * 70)

    state_conn.close()


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Guna Forest Division Multi-Threat EWS Daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/guna_ews_daemon.py                        # Full division run
  python scripts/guna_ews_daemon.py --dry-run              # No writes, just score
  python scripts/guna_ews_daemon.py --beat "Goumukh"       # Single beat
  python scripts/guna_ews_daemon.py --date 2026-01-15      # Backfill
  python scripts/guna_ews_daemon.py --dry-run --beat "Goumukh" --date 2026-01-15
        """,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Score and log only; no SQLite state updates, no PostGIS inserts.",
    )
    p.add_argument(
        "--beat", type=str, default=None, metavar="NAME",
        help="Process only beats matching NAME (case-insensitive substring match).",
    )
    p.add_argument(
        "--date", type=str, default=None, metavar="YYYY-MM-DD",
        help="Override anchor date for GEE imagery search (for backfilling past events).",
    )
    return p.parse_args()


if __name__ == "__main__":
    run_division_automation(_parse_args())
