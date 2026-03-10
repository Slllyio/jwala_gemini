"""
Van Suraksha REST API  v2.0
============================

FastAPI application for querying alerts, submitting labels,
and monitoring pipeline health.

Changelog v2.0
--------------
- AlertSummary / alert detail now expose cusum_zone_score + sar_boost_applied (Phase 2)
- /alerts: new filter params  sar_boosted, min_cusum, beat_name
- GET /beats : per-beat aggregated summary (alert counts, area, avg risk)
- GET /alerts/{id}: full detail includes all delta channels + SAR fields
- /stats: extended with SAR boost rate and per-beat breakdown

Run:
    uvicorn src.api.api:app --reload --port 8000
"""

import logging
import os
import json
from contextlib import contextmanager
from datetime import datetime, date
from typing import Generator, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg
import yaml

log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG_PATH = os.environ.get("VS_CONFIG", "config.yaml")

try:
    with open(CONFIG_PATH) as f:
        _cfg = yaml.safe_load(f)
    _db = _cfg.get("database", {})
except FileNotFoundError:
    log.warning("Config %s not found — using env vars / defaults for DB", CONFIG_PATH)
    _db = {}

_db_password = os.environ.get("VS_DB_PASSWORD", _db.get("password", ""))
if not _db_password:
    log.warning("No database password configured (VS_DB_PASSWORD env or config.yaml database.password)")

DB_DSN = (
    f"host={_db.get('host', 'localhost')} "
    f"port={_db.get('port', 5432)} "
    f"dbname={_db.get('dbname', 'gis_projects')} "
    f"user={_db.get('user', 'postgres')} "
    f"password={_db_password}"
)

# API key for write endpoints (labels). Set via VS_API_KEY env var.
_API_KEY = os.environ.get("VS_API_KEY", "")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Van Suraksha API",
    description=(
        "Deforestation Alert API for Guna Forest Division. "
        "v2.0 adds SAR CuSum fields and per-beat beat summaries."
    ),
    version="2.0.0",
)

_ALLOWED_ORIGINS = os.environ.get("VS_CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@contextmanager
def get_conn() -> Generator[psycopg.Connection, None, None]:
    """Context manager that guarantees connection cleanup."""
    conn = psycopg.connect(DB_DSN)
    try:
        yield conn
    finally:
        conn.close()


def _require_api_key(x_api_key: str = Header(default="")) -> str:
    """Dependency for write endpoints. Skipped if VS_API_KEY is not set."""
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(403, "Invalid or missing X-Api-Key header")
    return x_api_key


# ── Pydantic models ──────────────────────────────────────────────────────────

class LabelSubmission(BaseModel):
    alert_id: int
    label: str          # confirmed | false_positive | uncertain
    labeler: str = "api_user"
    notes: str = ""


class AlertSummary(BaseModel):
    id: int
    change_type: Optional[str]
    area_ha: Optional[float]
    confidence: Optional[float]
    detection_date: Optional[str]
    model_version: Optional[str]
    centroid_lat: Optional[float]
    centroid_lon: Optional[float]
    sub_range: Optional[str]
    beat_name: Optional[str]
    # Phase 2 SAR fields
    cusum_zone_score: Optional[float]
    sar_boost_applied: Optional[bool]


class BeatSummary(BaseModel):
    beat_name: str
    alert_count: int
    total_area_ha: float
    avg_confidence: float
    sar_boost_rate: float       # fraction of alerts with SAR boost
    avg_cusum_score: Optional[float]
    latest_alert_date: Optional[str]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _alert_where(
    change_type, model_version, min_confidence, min_area,
    date_from, date_to, bbox, beat_name, sar_boosted, min_cusum,
) -> tuple[str, list]:
    """Build shared WHERE clause for /alerts and /stats."""
    conditions = ["1=1"]
    params: list = []

    if change_type:
        conditions.append("change_type = %s")
        params.append(change_type)
    if model_version:
        conditions.append("model_version = %s")
        params.append(model_version)
    if beat_name:
        conditions.append("beat_name = %s")
        params.append(beat_name)
    if min_confidence > 0:
        conditions.append("COALESCE(stacked_score, confidence) >= %s")
        params.append(min_confidence)
    if min_area > 0:
        conditions.append("area_ha >= %s")
        params.append(min_area)
    if date_from:
        conditions.append("detection_date >= %s")
        params.append(str(date_from))
    if date_to:
        conditions.append("detection_date <= %s")
        params.append(str(date_to))
    if sar_boosted is True:
        conditions.append("sar_boost_applied = TRUE")
    elif sar_boosted is False:
        conditions.append("(sar_boost_applied = FALSE OR sar_boost_applied IS NULL)")
    if min_cusum is not None and min_cusum > 0:
        conditions.append("cusum_zone_score >= %s")
        params.append(min_cusum)
    if bbox:
        try:
            minlon, minlat, maxlon, maxlat = [float(x) for x in bbox.split(",")]
            conditions.append(
                "centroid_lon BETWEEN %s AND %s AND centroid_lat BETWEEN %s AND %s"
            )
            params.extend([minlon, maxlon, minlat, maxlat])
        except ValueError:
            raise HTTPException(400, "Invalid bbox format. Use: minlon,minlat,maxlon,maxlat")

    return " AND ".join(conditions), params


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """System health check."""
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM alerts_log")
            count = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(*) FROM alerts_log WHERE sar_boost_applied = TRUE"
            )
            sar_count = cur.fetchone()[0]
        return {
            "status": "healthy",
            "alerts_count": count,
            "sar_boosted_count": sar_count,
            "timestamp": datetime.utcnow().isoformat(),
            "api_version": "2.0.0",
        }
    except Exception as e:
        log.error("Health check failed: %s", e)
        return {"status": "unhealthy", "error": "Database connection failed"}


@app.get("/alerts", response_model=list[AlertSummary])
def list_alerts(
    change_type:    Optional[str]   = Query(None, description="Filter by change type"),
    model_version:  Optional[str]   = Query(None, description="Filter by model version"),
    beat_name:      Optional[str]   = Query(None, description="Filter by beat name"),
    min_confidence: float           = Query(0.0,  description="Minimum confidence score"),
    min_area:       float           = Query(0.0,  description="Minimum area in hectares"),
    date_from:      Optional[date]  = Query(None, description="Start date"),
    date_to:        Optional[date]  = Query(None, description="End date"),
    bbox:           Optional[str]   = Query(None, description="minlon,minlat,maxlon,maxlat"),
    # Phase 2 SAR filters
    sar_boosted:    Optional[bool]  = Query(None, description="Filter to SAR-boosted alerts only"),
    min_cusum:      Optional[float] = Query(None, description="Minimum CuSum zone score (0–1)"),
    limit:          int             = Query(100, le=1000),
    offset:         int             = Query(0,   ge=0),
):
    """
    List alerts with optional filtering.

    New in v2.0: `sar_boosted`, `min_cusum`, `beat_name` filters.
    """
    where, params = _alert_where(
        change_type, model_version, min_confidence, min_area,
        date_from, date_to, bbox, beat_name, sar_boosted, min_cusum,
    )

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""SELECT id, change_type, area_ha,
                       COALESCE(stacked_score, confidence) AS confidence,
                       detection_date, model_version,
                       centroid_lat, centroid_lon,
                       sub_range, beat_name,
                       cusum_zone_score, sar_boost_applied
                FROM alerts_log
                WHERE {where}
                ORDER BY COALESCE(stacked_score, confidence) DESC NULLS LAST
                LIMIT %s OFFSET %s""",
            params + [limit, offset],
        )
        rows = cur.fetchall()

    return [
        AlertSummary(
            id=r[0], change_type=r[1],
            area_ha=float(r[2]) if r[2] is not None else None,
            confidence=float(r[3]) if r[3] is not None else None,
            detection_date=str(r[4]) if r[4] else None,
            model_version=r[5],
            centroid_lat=float(r[6]) if r[6] is not None else None,
            centroid_lon=float(r[7]) if r[7] is not None else None,
            sub_range=r[8], beat_name=r[9],
            cusum_zone_score=float(r[10]) if r[10] is not None else None,
            sar_boost_applied=bool(r[11]) if r[11] is not None else None,
        )
        for r in rows
    ]


@app.get("/alerts/{alert_id}")
def get_alert(alert_id: int):
    """Get a single alert with full details, SAR fields, and GeoJSON geometry."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT id, change_type, area_ha,
                      COALESCE(stacked_score, confidence) AS confidence,
                      detection_date, model_version, sub_range, beat_name,
                      centroid_lat, centroid_lon,
                      mean_delta_trees, mean_delta_crops, mean_delta_built,
                      mean_delta_bare, mean_delta_grass, mean_delta_water,
                      cusum_zone_score, sar_boost_applied,
                      fast_tracked, n_pixels,
                      ST_AsGeoJSON(geom) AS geojson
               FROM alerts_log WHERE id = %s""",
            [alert_id],
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(404, f"Alert {alert_id} not found")

    return {
        "id":               row[0],
        "change_type":      row[1],
        "area_ha":          float(row[2]) if row[2] is not None else None,
        "confidence":       float(row[3]) if row[3] is not None else None,
        "detection_date":   str(row[4]) if row[4] else None,
        "model_version":    row[5],
        "sub_range":        row[6],
        "beat_name":        row[7],
        "centroid_lat":     float(row[8]) if row[8] is not None else None,
        "centroid_lon":     float(row[9]) if row[9] is not None else None,
        "deltas": {
            "trees": float(row[10]) if row[10] is not None else None,
            "crops": float(row[11]) if row[11] is not None else None,
            "built": float(row[12]) if row[12] is not None else None,
            "bare":  float(row[13]) if row[13] is not None else None,
            "grass": float(row[14]) if row[14] is not None else None,
            "water": float(row[15]) if row[15] is not None else None,
        },
        # Phase 2 SAR CuSum fields
        "sar": {
            "cusum_zone_score":  float(row[16]) if row[16] is not None else None,
            "sar_boost_applied": bool(row[17])  if row[17] is not None else False,
        },
        "fast_tracked": bool(row[18]) if row[18] is not None else False,
        "n_pixels":     row[19],
        "geometry":     json.loads(row[20]) if row[20] else None,
    }


@app.get("/beats", response_model=list[BeatSummary])
def beat_summaries(
    model_version: Optional[str]  = Query(None),
    date_from:     Optional[date] = Query(None),
    date_to:       Optional[date] = Query(None),
):
    """
    Per-beat aggregated summary: alert count, total area, avg confidence,
    SAR boost rate, avg CuSum score, and latest alert date.

    Returns beats sorted by total detected area descending.
    """
    conditions = ["beat_name IS NOT NULL"]
    params: list = []
    if model_version:
        conditions.append("model_version = %s")
        params.append(model_version)
    if date_from:
        conditions.append("detection_date >= %s")
        params.append(str(date_from))
    if date_to:
        conditions.append("detection_date <= %s")
        params.append(str(date_to))

    where = " AND ".join(conditions)

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""SELECT
                   beat_name,
                   COUNT(*)::INT                                           AS alert_count,
                   COALESCE(SUM(area_ha), 0)                              AS total_area_ha,
                   COALESCE(AVG(COALESCE(stacked_score, confidence)), 0)  AS avg_confidence,
                   COALESCE(
                       SUM(CASE WHEN sar_boost_applied THEN 1 ELSE 0 END)::FLOAT
                       / NULLIF(COUNT(*), 0), 0
                   )                                                       AS sar_boost_rate,
                   AVG(cusum_zone_score)                                   AS avg_cusum_score,
                   MAX(detection_date)                                     AS latest_alert_date
                FROM alerts_log
                WHERE {where}
                GROUP BY beat_name
                ORDER BY total_area_ha DESC""",
            params,
        )
        rows = cur.fetchall()

    return [
        BeatSummary(
            beat_name=r[0],
            alert_count=int(r[1]),
            total_area_ha=round(float(r[2]), 3),
            avg_confidence=round(float(r[3]), 4),
            sar_boost_rate=round(float(r[4]), 4),
            avg_cusum_score=round(float(r[5]), 4) if r[5] is not None else None,
            latest_alert_date=str(r[6]) if r[6] else None,
        )
        for r in rows
    ]


@app.post("/labels")
def submit_label(label: LabelSubmission, _key: str = Depends(_require_api_key)):
    """Submit a flywheel label for an alert. Requires X-Api-Key header."""
    if label.label not in ("confirmed", "false_positive", "uncertain"):
        raise HTTPException(400, "label must be: confirmed, false_positive, or uncertain")

    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute("SELECT id FROM alerts_log WHERE id = %s", [label.alert_id])
        if not cur.fetchone():
            raise HTTPException(404, f"Alert {label.alert_id} not found")

        cur.execute(
            """INSERT INTO flywheel_labels (alert_id, label, labeler, notes)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (alert_id) DO UPDATE
               SET label = EXCLUDED.label,
                   labeler = EXCLUDED.labeler,
                   notes = EXCLUDED.notes,
                   labeled_at = NOW()
               RETURNING alert_id, label""",
            [label.alert_id, label.label, label.labeler, label.notes],
        )
        result = cur.fetchone()
        conn.commit()

    return {"status": "ok", "alert_id": result[0], "label": result[1]}


@app.get("/stats")
def alert_stats(
    model_version: Optional[str]  = Query(None),
    date_from:     Optional[date] = Query(None),
    date_to:       Optional[date] = Query(None),
    beat_name:     Optional[str]  = Query(None, description="Restrict to a single beat"),
):
    """
    Aggregated alert statistics.

    New in v2.0: SAR boost rate and per-beat breakdown.
    """
    conditions = ["1=1"]
    params: list = []
    if model_version:
        conditions.append("model_version = %s")
        params.append(model_version)
    if date_from:
        conditions.append("detection_date >= %s")
        params.append(str(date_from))
    if date_to:
        conditions.append("detection_date <= %s")
        params.append(str(date_to))
    if beat_name:
        conditions.append("beat_name = %s")
        params.append(beat_name)

    where = " AND ".join(conditions)

    with get_conn() as conn:
        cur = conn.cursor()

        # By change type
        cur.execute(
            f"""SELECT change_type, COUNT(*), AVG(area_ha), SUM(area_ha)
                FROM alerts_log WHERE {where}
                GROUP BY change_type ORDER BY COUNT(*) DESC""",
            params,
        )
        by_type = [
            {"type": r[0], "count": r[1],
             "avg_area_ha": round(float(r[2]), 3) if r[2] else 0,
             "total_area_ha": round(float(r[3]), 2) if r[3] else 0}
            for r in cur.fetchall()
        ]

        # By model version
        cur.execute(
            f"""SELECT model_version, COUNT(*), AVG(COALESCE(stacked_score, confidence))
                FROM alerts_log WHERE {where}
                GROUP BY model_version ORDER BY COUNT(*) DESC""",
            params,
        )
        by_version = [
            {"version": r[0], "count": r[1],
             "avg_confidence": round(float(r[2]), 4) if r[2] else 0}
            for r in cur.fetchall()
        ]

        # Label progress
        cur.execute("SELECT label, COUNT(*) FROM flywheel_labels GROUP BY label")
        labels = {r[0]: r[1] for r in cur.fetchall()}

        # Phase 2: SAR CuSum stats
        cur.execute(
            f"""SELECT
                   COUNT(*) FILTER (WHERE sar_boost_applied = TRUE)    AS sar_boosted,
                   COUNT(*)                                             AS total,
                   AVG(cusum_zone_score)                               AS avg_cusum,
                   MAX(cusum_zone_score)                               AS max_cusum
                FROM alerts_log WHERE {where}""",
            params,
        )
        r = cur.fetchone()
        n_boosted, n_total, avg_cusum, max_cusum = r
        sar_stats = {
            "boosted_alerts":   int(n_boosted or 0),
            "total_alerts":     int(n_total  or 0),
            "sar_boost_rate":   round((n_boosted or 0) / max(n_total or 1, 1), 4),
            "avg_cusum_score":  round(float(avg_cusum), 4) if avg_cusum else None,
            "max_cusum_score":  round(float(max_cusum), 4) if max_cusum else None,
        }

    return {
        "by_change_type":   by_type,
        "by_model_version": by_version,
        "labels":           labels,
        "total_alerts":     sum(t["count"] for t in by_type),
        "sar":              sar_stats,
    }


@app.get("/runs")
def pipeline_runs(limit: int = Query(10, le=50)):
    """Recent pipeline runs from tracker."""
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                """SELECT run_id, run_name, status, duration_s,
                          steps_total, steps_ok, steps_failed, created_at
                   FROM pipeline_runs
                   ORDER BY created_at DESC
                   LIMIT %s""",
                [limit],
            )
            rows = cur.fetchall()
        except psycopg.Error:
            rows = []

    return [
        {
            "run_id":       r[0], "run_name": r[1], "status": r[2],
            "duration_s":   float(r[3]) if r[3] else 0,
            "steps_total":  r[4], "steps_ok": r[5], "steps_failed": r[6],
            "created_at":   r[7].isoformat() if r[7] else None,
        }
        for r in rows
    ]
