"""
tests/test_api.py
=================
Unit tests for the Van Suraksha REST API (v2.0).

Uses FastAPI TestClient — no live DB required.  All DB calls are
monkey-patched with an in-memory fake that returns controlled fixtures.

Coverage
--------
1. /health  — returns healthy status + api_version 2.0.0
2. /alerts  — list with SAR filter params (sar_boosted, min_cusum, beat_name)
3. /alerts/{id}  — detail includes sar block + all delta channels
4. /alerts/{id}  — 404 on missing id
5. /beats   — returns BeatSummary list
6. /labels  — accept and reject invalid label values
7. /stats   — includes 'sar' block with boost_rate
8. _alert_where  — unit-tests the WHERE-clause builder in isolation
"""

import sys
import os
from unittest.mock import MagicMock, patch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import pytest
from fastapi.testclient import TestClient


# ── Fake DB cursor / connection fixture ──────────────────────────────────────

def _make_fake_conn(cursor_results: list):
    """
    Returns a (conn, cursor) mock pair.  cursor_results is a list of
    return-values that fetchone() / fetchall() will cycle through in order.
    """
    mock_cursor  = MagicMock()
    mock_conn    = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    # fetchone / fetchall alternate from the queue
    _iter = iter(cursor_results)

    def _next(*_):
        try:
            return next(_iter)
        except StopIteration:
            return None

    mock_cursor.fetchone.side_effect  = _next
    mock_cursor.fetchall.side_effect  = _next
    mock_cursor.description           = []

    return mock_conn


# ── Helper: import app with DB patched ───────────────────────────────────────

def _client(fake_conn):
    """Return a TestClient with get_conn() patched to return fake_conn."""
    with patch("src.api.api.get_conn", return_value=fake_conn):
        from src.api.api import app
        return TestClient(app, raise_server_exceptions=False)


# ── 1. /health ───────────────────────────────────────────────────────────────

class TestHealth:
    def test_healthy(self):
        conn = _make_fake_conn([
            (42,),    # SELECT COUNT(*) FROM alerts_log
            (7,),     # SELECT COUNT(*) WHERE sar_boost_applied = TRUE
        ])
        with patch("src.api.api.get_conn", return_value=conn):
            from src.api.api import app
            client = TestClient(app)
            r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert body["alerts_count"] == 42
        assert body["sar_boosted_count"] == 7
        assert body["api_version"] == "2.0.0"

    def test_unhealthy_on_db_error(self):
        conn = MagicMock()
        conn.cursor.side_effect = Exception("DB down")
        with patch("src.api.api.get_conn", return_value=conn):
            from src.api.api import app
            client = TestClient(app)
            r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "unhealthy"


# ── 2. /alerts list ───────────────────────────────────────────────────────────

_ALERT_ROW = (
    1,              # id
    "deforestation",# change_type
    0.85,           # area_ha
    0.72,           # confidence
    "2024-06-15",   # detection_date
    "v1.2",         # model_version
    24.1001,        # centroid_lat
    77.2345,        # centroid_lon
    "Guna North",   # sub_range
    "Chanderi Beat",# beat_name
    0.63,           # cusum_zone_score
    True,           # sar_boost_applied
)


class TestAlertsList:
    def _get_list(self, params=""):
        conn = _make_fake_conn([[_ALERT_ROW]])
        with patch("src.api.api.get_conn", return_value=conn):
            from src.api.api import app
            client = TestClient(app)
            return client.get(f"/alerts{params}")

    def test_basic_list(self):
        r = self._get_list()
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert data[0]["id"]               == 1
        assert data[0]["cusum_zone_score"] == 0.63
        assert data[0]["sar_boost_applied"] is True
        assert data[0]["beat_name"]        == "Chanderi Beat"

    def test_sar_boosted_filter(self):
        r = self._get_list("?sar_boosted=true")
        assert r.status_code == 200

    def test_min_cusum_filter(self):
        r = self._get_list("?min_cusum=0.5")
        assert r.status_code == 200

    def test_beat_filter(self):
        r = self._get_list("?beat_name=Chanderi+Beat")
        assert r.status_code == 200

    def test_invalid_bbox_returns_400(self):
        r = self._get_list("?bbox=notabbox")
        assert r.status_code == 400


# ── 3 & 4. /alerts/{id} ──────────────────────────────────────────────────────

_DETAIL_ROW = (
    1, "deforestation", 0.85, 0.72,
    "2024-06-15", "v1.2", "Guna North", "Chanderi Beat",
    24.1001, 77.2345,
    -0.12, 0.03, 0.04, 0.02, 0.01, -0.01,  # deltas
    0.63, True,                              # cusum, sar_boost
    False, 120,                              # fast_tracked, n_pixels
    '{"type":"Polygon","coordinates":[]}',   # geojson
)


class TestAlertDetail:
    def test_found(self):
        conn = _make_fake_conn([_DETAIL_ROW])
        with patch("src.api.api.get_conn", return_value=conn):
            from src.api.api import app
            r = TestClient(app).get("/alerts/1")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == 1
        assert body["sar"]["cusum_zone_score"]  == 0.63
        assert body["sar"]["sar_boost_applied"] is True
        assert "trees" in body["deltas"]
        assert "grass" in body["deltas"]
        assert "water" in body["deltas"]

    def test_not_found(self):
        conn = _make_fake_conn([None])
        with patch("src.api.api.get_conn", return_value=conn):
            from src.api.api import app
            r = TestClient(app).get("/alerts/9999")
        assert r.status_code == 404


# ── 5. /beats ────────────────────────────────────────────────────────────────

_BEAT_ROW = ("Chanderi Beat", 8, 14.3, 0.35, 0.625, 0.58, "2024-06-15")


class TestBeats:
    def test_list(self):
        conn = _make_fake_conn([[_BEAT_ROW]])
        with patch("src.api.api.get_conn", return_value=conn):
            from src.api.api import app
            r = TestClient(app).get("/beats")
        assert r.status_code == 200
        beats = r.json()
        assert isinstance(beats, list)
        b = beats[0]
        assert b["beat_name"]       == "Chanderi Beat"
        assert b["alert_count"]     == 8
        assert b["sar_boost_rate"]  == pytest.approx(0.625, rel=1e-3)
        assert b["avg_cusum_score"] == pytest.approx(0.58,  rel=1e-3)

    def test_empty_beats(self):
        conn = _make_fake_conn([[]])
        with patch("src.api.api.get_conn", return_value=conn):
            from src.api.api import app
            r = TestClient(app).get("/beats")
        assert r.status_code == 200
        assert r.json() == []


# ── 6. /labels ───────────────────────────────────────────────────────────────

class TestLabels:
    def _post(self, payload, conn_results):
        conn = _make_fake_conn(conn_results)
        with patch("src.api.api.get_conn", return_value=conn):
            from src.api.api import app
            return TestClient(app).post("/labels", json=payload)

    def test_valid_confirmed(self):
        r = self._post(
            {"alert_id": 1, "label": "confirmed", "labeler": "ranger_01"},
            [(1,), (1, "confirmed")],
        )
        assert r.status_code == 200
        assert r.json()["label"] == "confirmed"

    def test_invalid_label_rejected(self):
        conn = _make_fake_conn([])
        with patch("src.api.api.get_conn", return_value=conn):
            from src.api.api import app
            r = TestClient(app).post("/labels",
                                     json={"alert_id": 1, "label": "WRONG"})
        assert r.status_code == 400

    def test_missing_alert_returns_404(self):
        r = self._post(
            {"alert_id": 9999, "label": "confirmed"},
            [None],   # SELECT id → not found
        )
        assert r.status_code == 404


# ── 7. /stats includes sar block ─────────────────────────────────────────────

class TestStats:
    def test_sar_block_present(self):
        conn = _make_fake_conn([
            [("deforestation", 5, 1.2, 6.0)],   # by change type
            [("v1.2", 5, 0.4)],                  # by model version
            [("confirmed", 3)],                  # labels
            (3, 5, 0.55, 0.82),                  # SAR stats row
        ])
        with patch("src.api.api.get_conn", return_value=conn):
            from src.api.api import app
            r = TestClient(app).get("/stats")
        assert r.status_code == 200
        body = r.json()
        assert "sar" in body
        sar = body["sar"]
        assert sar["boosted_alerts"]  == 3
        assert sar["total_alerts"]    == 5
        assert "sar_boost_rate"       in sar
        assert "avg_cusum_score"      in sar


# ── 8. _alert_where unit test (SQL builder) ──────────────────────────────────

class TestAlertWhere:
    def _where(self, **kwargs):
        from src.api.api import _alert_where
        defaults = dict(
            change_type=None, model_version=None, min_confidence=0.0,
            min_area=0.0, date_from=None, date_to=None,
            bbox=None, beat_name=None, sar_boosted=None, min_cusum=None,
        )
        defaults.update(kwargs)
        return _alert_where(**defaults)

    def test_empty_gives_1_equals_1(self):
        where, params = self._where()
        assert where == "1=1"
        assert params == []

    def test_sar_boosted_true(self):
        where, _ = self._where(sar_boosted=True)
        assert "sar_boost_applied = TRUE" in where

    def test_sar_boosted_false(self):
        where, _ = self._where(sar_boosted=False)
        assert "sar_boost_applied = FALSE" in where

    def test_min_cusum(self):
        where, params = self._where(min_cusum=0.5)
        assert "cusum_zone_score" in where
        assert 0.5 in params

    def test_beat_name(self):
        where, params = self._where(beat_name="TestBeat")
        assert "beat_name = %s" in where
        assert "TestBeat" in params

    def test_invalid_bbox_raises(self):
        import pytest
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            self._where(bbox="bad_input")


# ── Run directly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
