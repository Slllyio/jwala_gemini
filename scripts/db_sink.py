"""
DB Sink — Ingest Alert GeoJSONs into PostGIS
=============================================

Creates the flywheel tables (alerts_log, flywheel_labels) and
bulk-inserts confirmed alert polygons from GeoJSON files.

Usage:
    # Create tables only
    python scripts/db_sink.py --init

    # Ingest a stacked GeoJSON
    python scripts/db_sink.py --geojson outputs/alert_filter/stacked_v2/stacked_confirmed_alerts.geojson \
        --model-version v2_stacked

    # Both
    python scripts/db_sink.py --init \
        --geojson outputs/alert_filter/stacked_v2/stacked_confirmed_alerts.geojson \
        --model-version v2_stacked
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── SQL DDL ──────────────────────────────────────────────────────────────────

DDL_ALERTS_LOG = """
CREATE TABLE IF NOT EXISTS alerts_log (
    id                  SERIAL PRIMARY KEY,
    geom                GEOMETRY(Polygon, 4326),
    change_type         TEXT,
    confidence          FLOAT,
    stacked_score       FLOAT,
    area_ha             FLOAT,
    n_pixels            INT,
    centroid_lat        FLOAT,
    centroid_lon        FLOAT,
    detection_date      DATE,
    detection_period    TEXT,
    sub_range           TEXT,
    beat_name           TEXT,
    mean_delta_trees    FLOAT,
    mean_delta_crops    FLOAT,
    mean_delta_bare     FLOAT,
    mean_delta_built    FLOAT,
    mean_delta_grass    FLOAT,
    mean_delta_water    FLOAT,
    mean_fires          FLOAT,
    -- V5: absolute sampled values for reproducibility
    trees_before        FLOAT,
    trees_after         FLOAT,
    crops_before        FLOAT,
    crops_after         FLOAT,
    built_before        FLOAT,
    built_after         FLOAT,
    dw_trees_zscore     FLOAT,
    -- Phase 2: SAR CuSum fields
    cusum_zone_score    FLOAT,
    sar_boost_applied   BOOLEAN DEFAULT FALSE,
    fast_tracked        BOOLEAN DEFAULT FALSE,
    model_version       TEXT,
    source              TEXT,
    linked_alert_id     INT,
    ingested_at         TIMESTAMPTZ DEFAULT NOW()
);
"""

# Migration: adds columns that may be missing in older DB instances
DDL_MIGRATIONS = [
    "ALTER TABLE alerts_log ADD COLUMN IF NOT EXISTS beat_name TEXT;",
    "ALTER TABLE alerts_log ADD COLUMN IF NOT EXISTS mean_delta_grass FLOAT;",
    "ALTER TABLE alerts_log ADD COLUMN IF NOT EXISTS mean_delta_water FLOAT;",
    "ALTER TABLE alerts_log ADD COLUMN IF NOT EXISTS cusum_zone_score FLOAT;",
    "ALTER TABLE alerts_log ADD COLUMN IF NOT EXISTS sar_boost_applied BOOLEAN DEFAULT FALSE;",
    # V5: absolute sampled values
    "ALTER TABLE alerts_log ADD COLUMN IF NOT EXISTS trees_before FLOAT;",
    "ALTER TABLE alerts_log ADD COLUMN IF NOT EXISTS trees_after FLOAT;",
    "ALTER TABLE alerts_log ADD COLUMN IF NOT EXISTS crops_before FLOAT;",
    "ALTER TABLE alerts_log ADD COLUMN IF NOT EXISTS crops_after FLOAT;",
    "ALTER TABLE alerts_log ADD COLUMN IF NOT EXISTS built_before FLOAT;",
    "ALTER TABLE alerts_log ADD COLUMN IF NOT EXISTS built_after FLOAT;",
    "ALTER TABLE alerts_log ADD COLUMN IF NOT EXISTS dw_trees_zscore FLOAT;",
    "ALTER TABLE alerts_log ADD COLUMN IF NOT EXISTS source TEXT;",
]

DDL_FLYWHEEL_LABELS = """
CREATE TABLE IF NOT EXISTS flywheel_labels (
    id          SERIAL PRIMARY KEY,
    alert_id    INT REFERENCES alerts_log(id) ON DELETE CASCADE,
    label       TEXT CHECK (label IN ('confirmed', 'false_positive', 'unsure')),
    labeler     TEXT,
    notes       TEXT,
    labeled_at  TIMESTAMPTZ DEFAULT NOW()
);
"""

DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_alerts_log_geom ON alerts_log USING GIST (geom);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_log_change_type ON alerts_log (change_type);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_log_detection_date ON alerts_log (detection_date);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_log_sub_range ON alerts_log (sub_range);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_log_model_version ON alerts_log (model_version);",
    "CREATE INDEX IF NOT EXISTS idx_flywheel_labels_alert_id ON flywheel_labels (alert_id);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_log_linked ON alerts_log (linked_alert_id);",
]


# ── Connection helpers ───────────────────────────────────────────────────────

def get_conn_params(config_path: str = "config.yaml") -> dict:
    """Read DB connection params from config.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    db_cfg = cfg.get("database", {})
    return {
        "host":     db_cfg.get("host", "localhost"),
        "port":     db_cfg.get("port", 5432),
        "dbname":   db_cfg.get("dbname", "gis_projects"),
        "user":     db_cfg.get("user", "postgres"),
        "password": db_cfg.get("password", ""),
    }


def connect(config_path: str = "config.yaml"):
    """Return a psycopg connection."""
    import psycopg

    params = get_conn_params(config_path)
    # Remove empty password to let trust auth work
    if not params["password"]:
        del params["password"]

    conn = psycopg.connect(**params)
    return conn


# ── Init tables ──────────────────────────────────────────────────────────────

def init_tables(conn):
    """Create PostGIS extension and flywheel tables."""
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    log.info("PostGIS extension: OK")

    cur.execute(DDL_ALERTS_LOG)
    log.info("Table alerts_log: OK")

    # Phase 2 / ongoing migrations — safe on empty and existing tables
    for migration in DDL_MIGRATIONS:
        try:
            cur.execute(migration)
        except Exception as mig_err:
            log.debug(f"Migration skipped (non-fatal): {migration[:60]}… — {mig_err}")
    log.info(f"Migrations: {len(DDL_MIGRATIONS)} applied")

    cur.execute(DDL_FLYWHEEL_LABELS)
    log.info("Table flywheel_labels: OK")

    for ddl in DDL_INDEXES:
        cur.execute(ddl)
    log.info(f"Indexes: {len(DDL_INDEXES)} created/verified")

    conn.commit()
    log.info("✅ Flywheel DB initialized")


# ── GeoJSON ingestion ────────────────────────────────────────────────────────

import math


def _safe_float(v):
    """Convert NaN/Inf/None to None for Postgres compatibility."""
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _parse_date(s: str):
    """Parse a date string, returning None on failure."""
    if not s or s == "unknown":
        return None
    try:
        from datetime import date
        parts = s.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return None


def ingest_geojson(conn, geojson_path: str, model_version: str = "unknown"):
    """
    Read a GeoJSON FeatureCollection and bulk-insert into alerts_log.
    Deduplicates by (centroid_lat, centroid_lon, detection_period, model_version).
    """
    with open(geojson_path) as f:
        fc = json.load(f)

    features = fc.get("features", [])
    if not features:
        log.warning(f"No features in {geojson_path}")
        return 0

    log.info(f"Ingesting {len(features)} features from {geojson_path}")

    cur = conn.cursor()
    inserted = 0
    skipped = 0

    for feat in features:
        props = feat.get("properties", {})
        geom_json = json.dumps(feat["geometry"])

        centroid = props.get("centroid", [None, None])
        centroid_lat = centroid[0] if centroid else None
        centroid_lon = centroid[1] if centroid else None

        detection_period = props.get("detection_period", "")

        # Parse detection_date from detection_period end date
        detection_date = None
        if detection_period and " to " in detection_period:
            end_date_str = detection_period.split(" to ")[1].strip()
            detection_date = _parse_date(end_date_str)
        elif props.get("detection_date"):
            detection_date = _parse_date(props["detection_date"])

        # Dedup check
        cur.execute(
            """SELECT 1 FROM alerts_log
               WHERE centroid_lat = %s AND centroid_lon = %s
                 AND detection_period = %s AND model_version = %s
               LIMIT 1""",
            (centroid_lat, centroid_lon, detection_period, model_version)
        )
        if cur.fetchone():
            skipped += 1
            continue

        cur.execute(
            """INSERT INTO alerts_log (
                geom, change_type, confidence, stacked_score, area_ha, n_pixels,
                centroid_lat, centroid_lon, detection_date, detection_period,
                sub_range, beat_name,
                mean_delta_trees, mean_delta_crops, mean_delta_bare,
                mean_delta_built, mean_delta_grass, mean_delta_water,
                mean_fires,
                trees_before, trees_after,
                crops_before, crops_after,
                built_before, built_after,
                dw_trees_zscore,
                cusum_zone_score, sar_boost_applied,
                fast_tracked, model_version, source
            ) VALUES (
                ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326),
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s,
                %s, %s,
                %s, %s,
                %s, %s,
                %s,
                %s, %s,
                %s, %s, %s
            )""",
            (
                geom_json,
                props.get("change_type"),
                _safe_float(props.get("confidence") or props.get("max_single_P")),
                _safe_float(props.get("stacked_score")),
                _safe_float(props.get("area_ha")),
                props.get("n_pixels"),
                centroid_lat,
                centroid_lon,
                detection_date,
                detection_period,
                props.get("sub_range"),
                props.get("beat_name"),
                _safe_float(props.get("mean_delta_trees")),
                _safe_float(props.get("mean_delta_crops")),
                _safe_float(props.get("mean_delta_bare")),
                _safe_float(props.get("mean_delta_built")),
                _safe_float(props.get("mean_delta_grass")),
                _safe_float(props.get("mean_delta_water")),
                _safe_float(props.get("mean_fires")),
                _safe_float(props.get("trees_before")),
                _safe_float(props.get("trees_after")),
                _safe_float(props.get("crops_before")),
                _safe_float(props.get("crops_after")),
                _safe_float(props.get("built_before")),
                _safe_float(props.get("built_after")),
                _safe_float(props.get("dw_trees_zscore")),
                _safe_float(props.get("cusum_zone_score")),
                bool(props.get("sar_boost_applied", False)),
                props.get("fast_tracked", False),
                model_version,
                props.get("source"),
            )
        )
        inserted += 1

    conn.commit()
    log.info(f"[OK] Ingested {inserted} alerts ({skipped} duplicates skipped)")

    # Cross-run spatial dedup
    n_linked = link_spatial_overlaps(conn)
    if n_linked:
        log.info(f"[DEDUP] Linked {n_linked} spatially overlapping alerts")

    return inserted

# ── Cross-run spatial dedup ──────────────────────────────────────────────────

def link_spatial_overlaps(conn, overlap_threshold: float = 0.30):
    """
    Find alerts from DIFFERENT model versions that spatially overlap by >= threshold.
    Links them via linked_alert_id (points newer alerts to older ones).

    Returns number of newly linked alerts.
    """
    cur = conn.cursor()

    # Find overlapping pairs not yet linked
    cur.execute("""
        WITH pairs AS (
            SELECT a.id AS newer_id, b.id AS older_id,
                   ST_Area(ST_Intersection(a.geom, b.geom)) /
                   NULLIF(LEAST(ST_Area(a.geom), ST_Area(b.geom)), 0) AS overlap_frac
            FROM alerts_log a
            JOIN alerts_log b
              ON a.id > b.id
             AND a.model_version != b.model_version
             AND ST_Intersects(a.geom, b.geom)
            WHERE a.linked_alert_id IS NULL
              AND a.geom IS NOT NULL
              AND b.geom IS NOT NULL
        )
        SELECT newer_id, older_id, overlap_frac
        FROM pairs
        WHERE overlap_frac >= %s
    """, [overlap_threshold])

    rows = cur.fetchall()
    if not rows:
        return 0

    for newer_id, older_id, frac in rows:
        cur.execute(
            "UPDATE alerts_log SET linked_alert_id = %s WHERE id = %s AND linked_alert_id IS NULL",
            [older_id, newer_id]
        )

    conn.commit()
    log.info(f"[DEDUP] {len(rows)} overlap pairs found (>= {overlap_threshold:.0%} area)")
    return len(rows)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Van Suraksha — DB Sink")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--init", action="store_true",
                        help="Create/verify flywheel tables")
    parser.add_argument("--geojson", type=str, default=None,
                        help="Path to GeoJSON FeatureCollection to ingest")
    parser.add_argument("--model-version", type=str, default="unknown",
                        help="Model version tag (e.g. v2_stacked)")
    args = parser.parse_args()

    if not args.init and not args.geojson:
        parser.error("Specify at least one of --init or --geojson")

    conn = connect(args.config)
    log.info(f"Connected to DB: {get_conn_params(args.config)['dbname']}")

    try:
        if args.init:
            init_tables(conn)

        if args.geojson:
            # Ensure tables exist before ingesting
            init_tables(conn)
            ingest_geojson(conn, args.geojson, args.model_version)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
