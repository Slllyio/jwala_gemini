"""
ews_dashboard.py  — E-Netra V4  (Option 2: Beat-level interactive dashboard)
============================================================================
Produces outputs/dashboard.html — a self-contained dark-mode dashboard with:

  LEFT SIDEBAR  — Range → Beat tree, showing per-beat alert counts
  CENTRE MAP    — Leaflet map colour-coded by alert tier
  RIGHT PANEL   — Filterable alert table + score chart

  Clicking a Range/Beat in the sidebar filters BOTH map AND table.
  "🖨 Report" button on each HIGH alert row opens the field report page.

Run:
    python scripts/ews_dashboard.py
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    import psycopg2
    import yaml as _yaml
    _HAS_DB = True
except ImportError:
    _HAS_DB = False

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parent.parent
FEEDS      = [_ROOT / "outputs" / "dashboard_feed", _ROOT / "outputs" / "alerts"]
BEATS_FILE    = _ROOT / "data" / "aoi" / "guna_beats.geojson"   # sidebar tree (Beat/Range keys)
BEATS_FC_FILE = _ROOT / "data" / "aoi" / "gunafinal.geojson"    # map boundary layer (BEAT/RANGE keys)
CONFIG     = _ROOT / "config.yaml"
OUT_HTML   = _ROOT / "outputs" / "dashboard.html"
MAX_DB_ROWS = 2000  # perf cap


# ── helpers ───────────────────────────────────────────────────────────────────
def _score_colour(s: float) -> str:
    if s >= 0.45: return "#f87171"
    if s >= 0.35: return "#fb923c"
    return "#facc15"
    if s >= 0.45: return "#e67e22"
    if s >= 0.25: return "#f1c40f"
    return "#607d8b"

def _label(s: float) -> str:
    if s >= 0.65: return "HIGH"
    if s >= 0.45: return "MEDIUM"
    if s >= 0.25: return "LOW"
    return "—"


# ── spatial beat lookup (point-in-polygon) ────────────────────────────────────
def _build_beat_index() -> list[dict]:
    """Load beats as list of {range, beat, coords} for PIP lookup."""
    if not BEATS_FILE.exists():
        return []
    fc = json.loads(BEATS_FILE.read_text(encoding="utf-8"))
    index = []
    for f in fc["features"]:
        p  = f.get("properties", {}) or {}
        rng  = p.get("Range", "—")
        beat = p.get("Beat", "—")
        geom = f.get("geometry", {})
        gtype = geom.get("type", "")
        if gtype == "Polygon":
            rings = [geom["coordinates"][0]]
        elif gtype == "MultiPolygon":
            rings = [part[0] for part in geom["coordinates"]]
        else:
            continue
        index.append({"range": rng, "beat": beat, "rings": rings})
    return index


def _pip(lon: float, lat: float, ring: list) -> bool:
    """Ray-casting point-in-polygon for a single ring."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _find_beat(lon: float, lat: float, index: list[dict]) -> tuple[str, str]:
    """Return (range, beat) for centroid, or ('—', '—') if not matched."""
    for entry in index:
        for ring in entry["rings"]:
            if _pip(lon, lat, ring):
                return entry["range"], entry["beat"]
    return "—", "—"


# ── PostGIS loader ────────────────────────────────────────────────────────────
def _load_from_postgres() -> list[dict]:
    """Read high-confidence alerts from PostGIS alerts_log. Returns [] if DB unavailable."""
    if not _HAS_DB or not CONFIG.exists():
        return []
    try:
        cfg = _yaml.safe_load(CONFIG.read_text(encoding="utf-8")).get("database", {})
        conn = psycopg2.connect(
            host=cfg.get("host", "localhost"),
            port=cfg.get("port", 5432),
            dbname=cfg.get("dbname", "gis_projects"),
            user=cfg.get("user", "postgres"),
            password=cfg.get("password", ""),
        )
        cur = conn.cursor()
        cur.execute("""
            SELECT
                id,
                change_type,
                COALESCE(confidence, stacked_score, 0.0)   AS score,
                area_ha,
                centroid_lat,
                centroid_lon,
                COALESCE(detection_date::text, '—')         AS det_date,
                COALESCE(beat_name, beat, '—')              AS beat,
                COALESCE(sub_range, range_name, '—')        AS rng,
                COALESCE(dw_trees_delta, 0.0)               AS delta,
                COALESCE(dw_trees_zscore, 0.0)              AS zscore,
                COALESCE(cloud_frac, 0.0)                   AS cloud,
                ST_AsText(ST_Simplify(geom, 0.0001))        AS wkt
            FROM alerts_log
            WHERE geom IS NOT NULL
              AND COALESCE(confidence, stacked_score, 0.0) >= 0.25
            ORDER BY COALESCE(confidence, stacked_score, 0.0) DESC
            LIMIT %s
        """, (MAX_DB_ROWS,))
        rows = cur.fetchall()
        conn.close()
        print(f"  DB: loaded {len(rows)} alerts from alerts_log")
        return rows
    except Exception as e:
        print(f"  [WARN] PostGIS load failed: {e}")
        return []


def _wkt_to_geom(wkt: str | None) -> dict | None:
    """Parse WKT POLYGON/MULTIPOLYGON into a GeoJSON geometry dict (no external libs)."""
    if not wkt:
        return None
    try:
        if wkt.upper().startswith("SRID="):
            wkt = wkt.split(";", 1)[1]
        wkt = wkt.strip()
        if wkt.upper().startswith("MULTIPOLYGON"):
            inner = wkt[wkt.index("(((") + 3: wkt.rindex(")))")].strip()
            polys = []
            for part in inner.split(")),(("):
                ring = [[float(v) for v in pt.strip().split()] for pt in part.split(",")]
                polys.append([ring])
            return {"type": "MultiPolygon", "coordinates": polys}
        elif wkt.upper().startswith("POLYGON"):
            inner = wkt[wkt.index("((") + 2: wkt.rindex("))")]
            rings = []
            for r in inner.split("),("):
                ring = [[float(v) for v in pt.strip().split()] for pt in r.split(",")]
                rings.append(ring)
            return {"type": "Polygon", "coordinates": rings}
    except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")
    return None


# ── load alerts ───────────────────────────────────────────────────────────────
def _load_alerts() -> list[dict]:
    beat_index = _build_beat_index()

    # Load from GeoJSON files produced by the V4 EWS daemon
    feats: list[dict] = []
    for d in FEEDS:
        if not d.exists():
            continue
        for fp in sorted(d.glob("*.geojson"), reverse=True):
            try:
                fc = json.loads(fp.read_text(encoding="utf-8"))
                for _f in fc.get("features", []):
                    _f["_src_file"] = fp.name
                    feats.append(_f)
            except Exception as e:
                print(f"  [WARN] {fp.name}: {e}")

    out: list[dict] = []
    for i, feat in enumerate(feats):
        p = feat.get("properties", {}) or {}
        # Composite score: zscore magnitude + cusum + area (avoids fixed 0.65)
        _conf    = float(p.get("confidence") or 0.65)
        _z       = abs(float(p.get("dw_trees_zscore") or 0.0))
        _cusum   = float(p.get("cusum_score") or 0.0)
        _area    = float(p.get("area_ha") or 0.0)
        score = round(min(0.99, max(0.30,
            _conf * 0.30 +
            (_z - 1.5) * 0.09 +
            _cusum * 0.25 +
            min(_area, 3.0) * 0.04
        )), 2)
        # Score-based tier (overrides EWS daemon's fixed label)
        if score >= 0.45:
            label = "HIGH"
        elif score >= 0.35:
            label = "MEDIUM"
        else:
            label = "LOW"

        # Date from property or parse YYYY-MM-DD from filename
        date = p.get("detection_date") or p.get("date") or None
        if not date:
            _src = feat.get("_src_file", "")
            _m = re.search(r"(\d{4}-\d{2}-\d{2})", str(_src))
            date = _m.group(1) if _m else "—"
        area  = float(p.get("area_ha") or 0.0)
        typ   = p.get("typology") or p.get("change_type") or "CANOPY LOSS"
        cloud = float(p.get("cloud_frac") or 0.0)
        delta  = float(p.get("dw_trees_delta") or 0.0)
        zscore = float(p.get("dw_trees_zscore") or 0.0)
        dndvi  = float(p.get("dNDVI") or 0.0)
        dnbr   = float(p.get("dNBR") or 0.0)
        dvh    = float(p.get("dVH") or 0.0)
        cusum  = float(p.get("cusum_score") or 0.0)
        count  = int(p.get("count") or 0)

        try:
            coords = feat["geometry"]["coordinates"][0]
            lats = [c[1] for c in coords]; lngs = [c[0] for c in coords]
            clat = sum(lats)/len(lats);    clng = sum(lngs)/len(lngs)
        except Exception:
            clat, clng = 24.65, 77.30

        # Try property field first, then spatial lookup against beats GeoJSON
        beat = p.get("beat_name") or p.get("beat") or None
        rng  = p.get("range_name") or p.get("range") or None
        if not beat or beat == "—":
            rng, beat = _find_beat(clng, clat, beat_index)

        out.append({
            "id": i, "score": score, "label": label, "beat": beat, "range": rng,
            "date": date, "area_ha": area, "typology": typ,
            "cloud_frac": cloud, "dw_trees_delta": delta, "zscore": zscore,
            "dNDVI": dndvi, "dNBR": dnbr, "dVH": dvh,
            "cusum_score": cusum, "count": count,
            "color": _score_colour(score), "lat": clat, "lng": clng,
            "geom": feat.get("geometry"),
        })

    out.sort(key=lambda x: x["score"], reverse=True)
    return out



# ── load beats for sidebar ────────────────────────────────────────────────────
def _load_beats() -> dict:
    """Returns {range_name: [beat_name, ...]} tree."""
    if not BEATS_FILE.exists():
        return {}
    fc = json.loads(BEATS_FILE.read_text(encoding="utf-8"))
    tree: dict[str, list[str]] = {}
    for f in fc["features"]:
        p = f.get("properties", {}) or {}
        rng  = p.get("Range", "—")
        beat = p.get("Beat", "—")
        tree.setdefault(rng, [])
        if beat not in tree[rng]:
            tree[rng].append(beat)
    return {k: sorted(v) for k, v in sorted(tree.items())}


def build_dashboard() -> str:
    alerts  = _load_alerts()
    beat_tree = _load_beats()

    n_total = len(alerts)
    n_high  = sum(1 for a in alerts if a["label"] == "HIGH")
    n_med   = sum(1 for a in alerts if a["label"] == "MEDIUM")
    n_low   = sum(1 for a in alerts if a["label"] == "LOW")
    avg_sc  = round(sum(a["score"] for a in alerts) / max(n_total, 1), 3)
    stamp   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    buckets = [0] * 10
    for a in alerts:
        buckets[min(int(a["score"] * 10), 9)] += 1

    # alert counts per beat for sidebar badges
    beat_counts: dict[str, int] = {}
    for a in alerts:
        if a["label"] in ("HIGH", "MEDIUM"):
            beat_counts[a["beat"]] = beat_counts.get(a["beat"], 0) + 1

    # sidebar HTML
    sidebar_html = ""
    for rng, beats in beat_tree.items():
        rng_count = sum(beat_counts.get(b, 0) for b in beats)
        badge = f'<span class="sbadge">{rng_count}</span>' if rng_count else ""
        sidebar_html += f"""
<div class="rng-group">
  <div class="rng-hdr" onclick="filterRange('{rng}')">
    <span class="rng-caret">▶</span>
    <span class="rng-name">{rng}</span>{badge}
  </div>
  <div class="beat-list">"""
        for b in beats:
            bc = beat_counts.get(b, 0)
            bb = f'<span class="sbadge red">{bc}</span>' if bc else ""
            sidebar_html += f"""
    <div class="beat-item" onclick="filterBeat('{b}')">{b}{bb}</div>"""
        sidebar_html += "\n  </div>\n</div>"

    # GeoJSON for Leaflet
    fc_json = json.dumps({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": a["geom"],
             "properties": {k: v for k, v in a.items() if k != "geom"}}
            for a in alerts if a["geom"]
        ]
    })

    # table rows
    rows_html = ""
    for a in alerts:
        aid = a["id"]
        rpt_btn = (f'<button class="rpt-btn" onclick="openReport({aid})" '
                   f'title="Field report">&#128424;</button>') if a["label"] == "HIGH" else ""
        kml_btn = f'<button class="rpt-btn kml-btn" onclick="dlAlertKml({aid})" title="Download alert KML">&#x2913; KML</button>'
        rows_html += f"""
<tr data-beat="{a['beat']}" data-range="{a['range']}" data-label="{a['label']}"
    id="row-{aid}" onclick="zoomToAlert({aid})" style="cursor:pointer">
  <td><span class="badge" style="background:{a['color']}">{a['label']}</span></td>
  <td><b>{a['score']:.3f}</b></td>
  <td>{a['beat']}</td>
  <td>{a['range']}</td>
  <td>{a['date']}</td>
  <td>{a['area_ha']:.2f}</td>
  <td>{a['dNDVI']:+.3f}</td>
  <td>{a['dNBR']:+.3f}</td>
  <td>{a['dw_trees_delta']:+.3f}</td>
  <td>{a['zscore']:+.2f}σ</td>
  <td>{a['cloud_frac']:.0%}</td>
  <td style="white-space:nowrap">{kml_btn}{rpt_btn}</td>
</tr>"""

    # beats GeoJSON embedded for boundary layer (gunafinal.geojson)
    beats_fc_json = "{}"
    if BEATS_FC_FILE.exists():
        beats_fc_json = BEATS_FC_FILE.read_text(encoding="utf-8").strip()
    elif BEATS_FILE.exists():
        beats_fc_json = BEATS_FILE.read_text(encoding="utf-8").strip()

    # alerts JSON for JS (for report button)
    alerts_json = json.dumps([{k: v for k, v in a.items() if k != "geom"}
                               for a in alerts])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta http-equiv="refresh" content="300"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>E-Netra V4 — Alert Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',sans-serif;background:#0b0f1a;color:#d4dbe8;display:flex;
     flex-direction:column;height:100vh;overflow:hidden}}

/* ── header ── */
header{{background:linear-gradient(135deg,#111827,#0b0f1a);padding:12px 24px;
        border-bottom:1px solid #1e2d45;display:flex;align-items:center;gap:14px;flex-shrink:0}}
.logo{{font-size:22px;font-weight:700;color:#38bdf8;letter-spacing:-.5px}}
.logo span{{color:#4ade80}}
.subtitle{{font-size:12px;color:#4b6280}}
.ts{{font-size:11px;color:#374151;margin-left:auto}}

/* ── kpis ── */
.kpis{{display:flex;gap:12px;padding:12px 24px;flex-shrink:0;border-bottom:1px solid #1e2d45}}
.kpi{{background:#111827;border:1px solid #1e2d45;border-radius:10px;padding:10px 18px;text-align:center;min-width:100px}}
.kpi .val{{font-size:28px;font-weight:700;line-height:1}}
.kpi .lbl{{font-size:10px;color:#4b6280;margin-top:3px;text-transform:uppercase;letter-spacing:.8px}}
.kpi.tot .val{{color:#38bdf8}} .kpi.high .val{{color:#f87171}}
.kpi.med  .val{{color:#fb923c}} .kpi.low  .val{{color:#facc15}}
.kpi.avg  .val{{color:#4ade80}}

/* ── body layout ── */
.body{{display:flex;flex:1;overflow:hidden}}

/* ── resize handles ── */
.resize-handle{{width:5px;background:transparent;cursor:col-resize;
                flex-shrink:0;transition:background .2s;z-index:10;}}
.resize-handle:hover,.resize-handle.dragging{{background:#1e3a5f}}

/* ── sidebar ── */
.sidebar{{width:220px;flex-shrink:0;background:#0d1420;border-right:1px solid #1e2d45;
          overflow-y:auto;padding:8px 0}}
.sb-title{{font-size:10px;font-weight:700;color:#4b6280;text-transform:uppercase;
           letter-spacing:.8px;padding:10px 14px 6px}}
.rng-group{{margin:2px 0}}
.rng-hdr{{display:flex;align-items:center;gap:6px;padding:7px 14px;cursor:pointer;
           font-size:12px;font-weight:600;color:#94a3b8;
           border-radius:6px;margin:0 6px;transition:background .15s}}
.rng-hdr:hover,.rng-hdr.active{{background:#1e2d45;color:#e2e8f0}}
.rng-caret{{font-size:9px;transition:transform .2s;flex-shrink:0}}
.rng-name{{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.beat-list{{display:none;padding-left:6px}}
.beat-list.open{{display:block}}
.beat-item{{padding:5px 14px 5px 26px;font-size:11px;color:#64748b;cursor:pointer;
            border-radius:6px;margin:0 6px;display:flex;align-items:center;justify-content:space-between;
            transition:background .15s}}
.beat-item:hover,.beat-item.active{{background:#1e2d45;color:#d4dbe8}}
.sbadge{{background:#1e3a5f;color:#38bdf8;font-size:10px;font-weight:700;
         border-radius:99px;padding:1px 7px;flex-shrink:0}}
.sbadge.red{{background:#450a0a;color:#f87171}}

/* ── centre map ── */
.map-panel{{flex:1;display:flex;flex-direction:column;overflow:hidden}}
#map{{flex:1;min-height:0}}
.filter-bar{{padding:8px 14px;background:#0d1420;border-top:1px solid #1e2d45;
             font-size:11px;color:#4b6280;display:flex;align-items:center;gap:10px}}
.filter-bar b{{color:#94a3b8}}
.clear-btn{{margin-left:auto;background:#1e2d45;border:none;color:#94a3b8;
            padding:4px 12px;border-radius:99px;cursor:pointer;font-size:11px}}
.clear-btn:hover{{background:#2d4a6e;color:#e2e8f0}}

/* ── right panel ── */
.right{{width:420px;flex-shrink:0;display:flex;flex-direction:column;
        border-left:1px solid #1e2d45;overflow:hidden}}
.chart-wrap{{padding:14px;border-bottom:1px solid #1e2d45;flex-shrink:0}}
.panel-title{{font-size:10px;font-weight:700;color:#4b6280;text-transform:uppercase;
              letter-spacing:.8px;margin-bottom:8px}}
.tbl-wrap{{flex:1;overflow-y:auto;padding:0 4px}}
table{{width:100%;border-collapse:collapse;font-size:11px}}
th{{background:#0b0f1a;padding:8px 8px;text-align:left;color:#4b6280;
    font-weight:600;text-transform:uppercase;letter-spacing:.4px;
    border-bottom:1px solid #1e2d45;position:sticky;top:0;z-index:1}}
td{{padding:6px 8px;border-bottom:1px solid #1e2d4522}}
tr:hover td{{background:#1e2d4522}}
tr.row-active td{{background:#1e3a5f44!important;border-bottom-color:#38bdf855}}
tr.hidden{{display:none}}
.badge{{display:inline-block;padding:1px 8px;border-radius:99px;font-size:10px;
        font-weight:700;color:#fff}}
.rpt-btn{{background:none;border:1px solid #1e3a5f;color:#38bdf8;border-radius:4px;
          cursor:pointer;font-size:11px;padding:1px 5px}}
.rpt-btn:hover{{background:#1e3a5f}}
.kml-btn{{border-color:#34d399!important;color:#34d399!important;margin-right:3px}}
.kml-btn:hover{{background:#052e16!important}}
.score-tip{{background:rgba(0,0,0,.7)!important;border:none!important;
            color:#fff!important;font-weight:700!important;font-size:10px!important}}
.beat-bnd-tip{{background:#0b0f1add!important;border:1px solid #1e3a5f!important;
               color:#e2e8f0!important;padding:6px 10px!important;
               border-radius:8px!important;font-size:12px!important;
               box-shadow:0 4px 16px rgba(0,0,0,.5)!important}}
.beat-bnd-tip::before{{display:none!important}}

/* ── Beat info panel (fixed bottom-left of map, not on polygon) ── */
#beat-info-panel{{
  position:absolute;bottom:52px;left:12px;z-index:1000;
  width:290px;background:rgba(11,15,26,0.96);
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
  border:1px solid #1e3a5f;border-radius:14px;
  box-shadow:0 12px 48px rgba(0,0,0,.8),0 0 0 1px rgba(56,189,248,.08);
  transform:translateY(calc(100% + 60px));opacity:0;
  transition:transform .32s cubic-bezier(.34,1.36,.64,1),opacity .25s ease;
  pointer-events:none;overflow:hidden}}
#beat-info-panel.open{{
  transform:translateY(0);opacity:1;pointer-events:all}}
.bip-accent{{height:4px;width:100%;border-radius:14px 14px 0 0}}
.bip-body{{padding:14px 18px 6px}}
.bip-beat{{font-size:16px;font-weight:700;color:#f1f5f9;letter-spacing:-.4px;
           line-height:1.2;margin-bottom:2px}}
.bip-range{{font-size:11px;font-weight:600;letter-spacing:.8px;
            text-transform:uppercase;margin-bottom:14px}}
.bip-rows{{border-top:1px solid #1e2d45}}
.bip-row{{display:flex;align-items:center;justify-content:space-between;
          padding:8px 0;border-bottom:1px solid #1e2d4530}}
.bip-row:last-child{{border-bottom:none;padding-bottom:4px}}
.bip-lbl{{font-size:10px;font-weight:600;color:#4b6280;
           text-transform:uppercase;letter-spacing:.7px}}
.bip-val{{font-size:12px;font-weight:600;color:#e2e8f0;text-align:right}}
.bip-close{{position:absolute;top:12px;right:14px;background:none;
             border:none;color:#4b6280;font-size:18px;cursor:pointer;
             line-height:1;transition:color .15s;padding:0}}
.bip-close:hover{{color:#e2e8f0}}
</style>
</head>
<body>
<header>
  <div class="logo">E-<span>Netra</span> V4</div>
  <div class="subtitle">Guna Forest Division · Early Warning System</div>
  <div class="ts">Generated: {stamp} · Auto-refreshes every 5 min</div>
</header>

<div class="kpis">
  <div class="kpi tot"><div class="val">{n_total}</div><div class="lbl">Total</div></div>
  <div class="kpi high"><div class="val">{n_high}</div><div class="lbl">HIGH</div></div>
  <div class="kpi med"><div class="val">{n_med}</div><div class="lbl">MEDIUM</div></div>
  <div class="kpi low"><div class="val">{n_low}</div><div class="lbl">LOW</div></div>
  <div class="kpi avg"><div class="val">{avg_sc}</div><div class="lbl">Avg Score</div></div>
</div>

<div class="body">

<!-- ── sidebar ── -->
<div class="sidebar" id="sidebar">
  <div class="sb-title">Division · Range · Beat</div>
  <div class="rng-group">
    <div class="rng-hdr" onclick="clearFilter()" id="all-hdr" style="color:#38bdf8">
      <span class="rng-caret">★</span>
      <span class="rng-name">All Beats</span>
    </div>
  </div>
  {sidebar_html}
</div>
<div class="resize-handle" id="rh-left" title="Drag to resize"></div>

<!-- ── map ── -->
<div class="map-panel" style="position:relative">
  <div id="map"></div>
  <!-- Beat info panel: fixed bottom-left, slides up on beat click -->
  <div id="beat-info-panel">
    <div class="bip-accent" id="bip-accent"></div>
    <button class="bip-close" onclick="closeBeatPanel()" title="Close">&#10005;</button>
    <div class="bip-body">
      <div class="bip-beat" id="bip-beat"></div>
      <div class="bip-range" id="bip-range"></div>
      <div class="bip-rows" id="bip-rows"></div>
    </div>
  </div>
  <div class="filter-bar">
    <b id="filter-label">All beats</b>
    <span id="filter-count">{n_total} alerts shown</span>
    <button class="clear-btn" onclick="clearFilter()">✕ Clear filter</button>
  </div>
</div>
<div class="resize-handle" id="rh-right" title="Drag to resize"></div>

<!-- ── right panel ── -->
<div class="right" id="rightPanel">
  <div class="chart-wrap">
    <div class="panel-title">Score Distribution</div>
    <canvas id="chart" height="120"></canvas>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>Tier</th><th>Score</th><th>Beat</th><th>Range</th>
        <th>Date</th><th>ha</th><th>ΔNDVI</th><th>ΔNBR</th><th>ΔTrees</th><th>Z</th><th>☁</th><th></th>
      </tr></thead>
      <tbody id="tbl-body">
        {rows_html}
      </tbody>
    </table>
  </div>
</div>

</div><!-- end body -->

<script>
// ── data ──────────────────────────────────────────────────────────────────────
const FC       = {fc_json};
const ALERTS   = {alerts_json};
const BEATS_FC = {beats_fc_json};

// ── map init ──────────────────────────────────────────────────────────────────
const map = L.map('map').setView([24.65, 77.30], 9);
L.tileLayer('https://mt1.google.com/vt/lyrs=s&x={{x}}&y={{y}}&z={{z}}',
  {{attribution:'© Google Satellite',maxZoom:22,opacity:1}}).addTo(map);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{attribution:'© OSM',maxZoom:19,opacity:0.25}}).addTo(map);

// ── range colour palette (matches sidebar colours) ──────────────────────────
const RNG_COLORS = {{
  'GUNA NORTH': '#38bdf8', 'GUNA SOUTH': '#4ade80',
  'ARON':       '#fb923c', 'MUNGAOLI':   '#a78bfa',
  'ASHOK NAGAR':'#f472b6', 'RAGHOGARH':  '#facc15',
  'BINA':       '#34d399', 'GUNA':       '#60a5fa',
  'CHACHODA':   '#f87171', 'KUMBHRAJ':   '#e879f9',
  // also try kebab-cased versions from guna_beats.geojson
  'North_Guna': '#38bdf8', 'South_Guna': '#4ade80',
  'Aron':       '#fb923c', 'Mungaoli':   '#a78bfa',
  'Ashoknangar':'#f472b6', 'Raghogarh':  '#facc15',
  'Bina':       '#34d399', 'Guna':       '#60a5fa',
}};
const RNG_PALETTE = [
  '#38bdf8','#4ade80','#fb923c','#a78bfa','#f472b6',
  '#facc15','#34d399','#60a5fa','#f87171','#e879f9',
  '#818cf8','#2dd4bf','#f59e0b','#84cc16','#c084fc'
];
const _rngMap = {{}};
let   _rngIdx = 0;
function rngColor(rng) {{
  if(!rng) return '#94a3b8';
  const key = rng.trim().toUpperCase();
  // direct palette lookup first
  for(const [k,v] of Object.entries(RNG_COLORS)) {{
    if(k.toUpperCase() === key) return v;
  }}
  // auto-assign from palette for unknown ranges
  if(!_rngMap[key]) _rngMap[key] = RNG_PALETTE[_rngIdx++ % RNG_PALETTE.length];
  return _rngMap[key];
}}

// ── Beat info panel (fixed HTML, not Leaflet popup) ────────────────────────────
function closeBeatPanel() {{
  document.getElementById('beat-info-panel').classList.remove('open');
}}
function openBeatPanel(props, col) {{
  const beat   = props.BEAT     || props.Beat     || '—';
  const range  = props.RANGE    || props.Range    || '—';
  const subRng = props.SUB_RANGE|| props.SubRange || '—';
  const beatNo = props.NEW_No_  || props.BeatNo   || '—';
  const area15 = props.area2015 != null ? props.area2015 + ' ha' : '—';
  const areaOri= props.areaOrigin != null
                   ? Number(props.areaOrigin).toFixed(1) + ' ha' : '—';

  document.getElementById('bip-accent').style.background = col;
  document.getElementById('bip-beat').textContent  = beat;
  const rEl = document.getElementById('bip-range');
  rEl.textContent  = range;
  rEl.style.color  = col;

  const rows = [
    ['Sub-Range', subRng],
    ['Beat No.',  beatNo],
    ['Area (2015)', area15],
    ['Area (Origin)', areaOri],
  ];
  document.getElementById('bip-rows').innerHTML = rows.map(([l,v]) =>
    `<div class="bip-row">
       <span class="bip-lbl">${{l}}</span>
       <span class="bip-val">${{v}}</span>
     </div>`
  ).join('') +
  `<div class="bip-row" style="justify-content:center;padding-top:10px">
     <button id="bip-kml-btn" onclick="downloadBeatKml(arguments[0])" 
       style="background:${{col}}22;border:1px solid ${{col}};color:${{col}};
              padding:6px 18px;border-radius:8px;cursor:pointer;
              font-size:11px;font-weight:700;letter-spacing:.5px;
              transition:background .15s"
       onmouseover="this.style.background='${{col}}44'" 
       onmouseout="this.style.background='${{col}}22'">
       &#x2913; Download KML
     </button>
   </div>`;

  // KML download — button onclick already passes props from beat layer click

  document.getElementById('beat-info-panel').classList.add('open');
}}
// ─── KML download for a beat polygon ────────────────────────────────────────
function downloadBeatKml(props) {{
  const beatName = props.BEAT || props.Beat || 'beat';
  const rangeName = props.RANGE || props.Range || '';
  // find the matching feature in BEATS_FC
  const feat = BEATS_FC.features.find(f => {{
    const p = f.properties || {{}};
    return (p.BEAT || p.Beat || '').toUpperCase() === beatName.toUpperCase();
  }});
  if(!feat) {{ alert('No geometry found for ' + beatName); return; }}
  // Convert GeoJSON MultiPolygon -> KML Placemark
  function ringToKml(coords) {{
    return '<coordinates>' +
      coords.map(c => c[0].toFixed(6)+','+c[1].toFixed(6)+',0').join(' ') +
      '</coordinates>';
  }}
  const geom = feat.geometry;
  let polyKml = '';
  if(geom.type === 'Polygon') {{
    polyKml = `<Polygon><outerBoundaryIs><LinearRing>${{ringToKml(geom.coordinates[0])}}</LinearRing></outerBoundaryIs></Polygon>`;
  }} else if(geom.type === 'MultiPolygon') {{
    polyKml = '<MultiGeometry>' +
      geom.coordinates.map(poly =>
        `<Polygon><outerBoundaryIs><LinearRing>${{ringToKml(poly[0])}}</LinearRing></outerBoundaryIs></Polygon>`
      ).join('') + '</MultiGeometry>';
  }}
  const p = feat.properties || {{}};
  const kml = `<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>${{beatName}}</name>
<Placemark>
  <name>${{beatName}}</name>
  <description><![CDATA[
    Range: ${{rangeName}}<br/>
    Sub-Range: ${{p.SUB_RANGE||'—'}}<br/>
    Beat No: ${{p.NEW_No_||'—'}}<br/>
    Area (2015): ${{p.area2015||'—'}} ha<br/>
    Area (Origin): ${{p.areaOrigin||'—'}} ha
  ]]></description>
  <Style><LineStyle><color>FF38bdf8</color><width>3</width></LineStyle>
         <PolyStyle><color>2238bdf8</color></PolyStyle></Style>
  ${{polyKml}}
</Placemark>
</Document>
</kml>`;
  const blob = new Blob([kml], {{type:'application/vnd.google-earth.kml+xml'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = beatName.replace(/ /g,'_') + '.kml';
  a.click();
}}

// close on map click
map.on('click', closeBeatPanel);
let _beatsLayer = null;
if(BEATS_FC && BEATS_FC.features) {{
  _beatsLayer = L.geoJSON(BEATS_FC, {{
    style: function(f) {{
      const rng = (f.properties && (f.properties.RANGE || f.properties.Range || ''));
      const col = rngColor(rng);
      return {{
        color:       col,
        weight:      2.5,
        opacity:     0.9,
        fillColor:   col,
        fillOpacity: 0.13,
        dashArray:   null
      }};
    }},
    onEachFeature: function(f, layer) {{
      const p   = f.properties || {{}};
      const rng = p.RANGE || p.Range || '';
      const col = rngColor(rng);
      const beat = p.BEAT || p.Beat || '—';

      // Google Earth: hover = lighten fill + thicken stroke
      layer.on('mouseover', function(e) {{
        layer.setStyle({{
          weight: 4, opacity: 1,
          fillOpacity: 0.30,
          color: col
        }});
        layer.bindTooltip(
          `<span style="font-size:12px;font-weight:700">${{beat}}</span><br>` +
          `<span style="font-size:10px;color:#94a3b8">${{rng}}</span>`,
          {{sticky:true, opacity:0.97, className:'beat-bnd-tip'}}
        ).openTooltip(e.latlng);
        if(!L.Browser.ie && !L.Browser.opera && !L.Browser.edge)
          layer.bringToFront();
      }});
      layer.on('mouseout', function() {{
        _beatsLayer.resetStyle(layer);
        layer.closeTooltip();
      }});

      // Google Earth: click = open fixed info panel
      layer.on('click', function(e) {{
        L.DomEvent.stopPropagation(e);
        openBeatPanel(p, col);
      }});
    }}
  }}).addTo(map);
}}

// close panel on map click (GE behaviour)
map.on('click', closeBeatPanel);

let geoLayer = null;

function renderMap(beats) {{
  if(geoLayer) map.removeLayer(geoLayer);
  const filtered = {{
    type:'FeatureCollection',
    features: FC.features.filter(f =>
      !beats || beats.includes(f.properties.beat)
    )
  }};
  if(!filtered.features.length) return;
  geoLayer = L.geoJSON(filtered, {{
    style: f => ({{
      color: f.properties.color || '#e74c3c',
      weight:2, fillColor: f.properties.color||'#e74c3c', fillOpacity:0.5
    }}),
    onEachFeature:(f,layer) => {{
      const p = f.properties;
      layer.bindPopup(`
        <b>${{p.label}}</b> · <b>${{(p.score||0).toFixed(3)}}</b><br>
        Beat: ${{p.beat}} | Range: ${{p.range}}<br>
        Date: ${{p.date}} | Area: ${{(p.area_ha||0).toFixed(2)}} ha<br>
        Type: ${{p.typology}}<br>
        ΔTrees: ${{(p.dw_trees_delta||0).toFixed(3)}} | Z: ${{(p.zscore||0).toFixed(2)}}σ
        ${{p.label==='HIGH' ? `<br><button onclick="openReport(${{p.id}})" style="margin-top:6px;padding:3px 10px;background:#1e3a5f;border:1px solid #38bdf8;color:#38bdf8;border-radius:4px;cursor:pointer;font-size:11px">🖨 Field Report</button>` : ''}}
      `);
      layer.bindTooltip((p.score||0).toFixed(3),
        {{permanent:true,className:'score-tip',direction:'center'}});
    }}
  }}).addTo(map);
  try{{ map.fitBounds(geoLayer.getBounds().pad(0.08)); }} catch(e){{}}
}}
renderMap(null);

// ── filtering ─────────────────────────────────────────────────────────────────
let activeFilter = null;

function filterRange(rng) {{
  activeFilter = {{type:'range', val:rng}};
  const beats = [...document.querySelectorAll(`.beat-item`)]
    .filter(el => el.closest('.rng-group')
                    .querySelector('.rng-name').textContent.trim()===rng)
    .map(el => el.textContent.replace(/\\d+$/, '').trim());
  _apply(beats, rng + ' Range');
  // open accordion
  document.querySelectorAll('.rng-hdr').forEach(h => {{
    const nm = h.querySelector('.rng-name');
    if(nm && nm.textContent.trim()===rng) {{
      h.classList.add('active');
      const bl = h.nextElementSibling;
      if(bl) bl.classList.add('open');
      h.querySelector('.rng-caret').style.transform='rotate(90deg)';
    }}
  }});
}}

function filterBeat(beat) {{
  activeFilter = {{type:'beat', val:beat}};
  _apply([beat], beat + ' Beat');
  document.querySelectorAll('.beat-item').forEach(el => {{
    el.classList.toggle('active', el.textContent.replace(/\\d+$/, '').trim()===beat);
  }});
}}

function clearFilter() {{
  activeFilter = null;
  document.querySelectorAll('.beat-item,.rng-hdr').forEach(el=>el.classList.remove('active'));
  document.querySelectorAll('.rng-caret').forEach(el=>el.style.transform='');
  _apply(null, 'All beats');
}}

function _apply(beats, label) {{
  const rows = document.querySelectorAll('#tbl-body tr');
  let shown = 0;
  rows.forEach(r => {{
    const b = r.dataset.beat;
    const show = !beats || beats.includes(b);
    r.classList.toggle('hidden', !show);
    if(show) shown++;
  }});
  document.getElementById('filter-label').textContent = label;
  document.getElementById('filter-count').textContent = shown + ' alerts shown';
  renderMap(beats);
}}

// ── accordion toggle ──────────────────────────────────────────────────────────
document.querySelectorAll('.rng-hdr').forEach(hdr => {{
  hdr.addEventListener('click', e => {{
    const bl = hdr.nextElementSibling;
    if(!bl || !bl.classList.contains('beat-list')) return;
    const open = bl.classList.toggle('open');
    hdr.querySelector('.rng-caret').style.transform = open ? 'rotate(90deg)' : '';
  }});
}});

// ── score chart ───────────────────────────────────────────────────────────────
new Chart(document.getElementById('chart'), {{
  type:'bar',
  data:{{
    labels:['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'],
    datasets:[{{ label:'Alerts', data:{json.dumps(buckets)},
      backgroundColor:['#334155','#334155','#334155','#ca8a04','#ca8a04',
                        '#ea580c','#ea580c','#dc2626','#dc2626','#dc2626'],
      borderRadius:3 }}]
  }},
  options:{{ responsive:true, plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{ticks:{{color:'#4b6280'}},grid:{{color:'#1e2d45'}}}},
      y:{{ticks:{{color:'#4b6280',stepSize:1}},grid:{{color:'#1e2d45'}},beginAtZero:true}}
    }}
  }}
}});

// ── field report pop-up ───────────────────────────────────────────────────────
function openReport(id) {{
  const a = ALERTS.find(x => x.id === id);
  if(!a) return;
  const w = window.open('', '_blank',
    'width=760,height=900,menubar=no,toolbar=no,location=no,status=no');
  w.document.write(buildReport(a));
  w.document.close();
}}

function buildReport(a) {{
  const mapsUrl = `https://maps.google.com/?q=${{a.lat}},${{a.lng}}`;
  const now = new Date().toISOString().slice(0,19).replace('T',' ');
  return `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>
<title>Field Report — ${{a.beat}}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet"/>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',sans-serif;background:#fff;color:#111;padding:32px;max-width:700px;margin:auto}}
h1{{font-size:20px;font-weight:700;color:#b91c1c;margin-bottom:2px}}
.sub{{font-size:12px;color:#6b7280;margin-bottom:22px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:22px}}
.card{{background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;padding:14px}}
.card-label{{font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;
             letter-spacing:.5px;margin-bottom:4px}}
.card-val{{font-size:20px;font-weight:700;color:#111}}
.card-val.red{{color:#b91c1c}}
.card-val.orange{{color:#c2410c}}
.map-link{{display:block;background:#1d4ed8;color:#fff;text-decoration:none;
           text-align:center;padding:10px;border-radius:8px;font-weight:600;
           font-size:13px;margin-bottom:22px}}
.map-link:hover{{background:#1e40af}}
table{{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:22px}}
th{{background:#f3f4f6;padding:7px 10px;text-align:left;font-weight:600;
    color:#374151;border-bottom:2px solid #e5e7eb}}
td{{padding:6px 10px;border-bottom:1px solid #f3f4f6}}
.sig{{margin-top:30px;border-top:1px solid #e5e7eb;padding-top:16px;font-size:11px;color:#9ca3af}}
.tier-badge{{display:inline-block;background:#b91c1c;color:#fff;
             padding:3px 12px;border-radius:99px;font-size:12px;font-weight:700;margin-bottom:16px}}
@media print{{body{{padding:10px}}.map-link{{display:none}}}}
</style></head><body>
<h1>🛰 E-Netra V4 — Field Alert Report</h1>
<div class="sub">Generated ${{now}} · AUTO-GENERATED · FOR OFFICIAL USE ONLY</div>

<span class="tier-badge">⚠ HIGH PRIORITY ALERT</span>

<div class="grid">
  <div class="card"><div class="card-label">Beat</div>
    <div class="card-val">${{a.beat}}</div></div>
  <div class="card"><div class="card-label">Range</div>
    <div class="card-val">${{a.range}}</div></div>
  <div class="card"><div class="card-label">V4 Confidence Score</div>
    <div class="card-val red">${{(a.score||0).toFixed(3)}}</div></div>
  <div class="card"><div class="card-label">Alert Tier</div>
    <div class="card-val red">HIGH</div></div>
  <div class="card"><div class="card-label">Area Affected</div>
    <div class="card-val">${{(a.area_ha||0).toFixed(2)}} ha</div></div>
  <div class="card"><div class="card-label">Detection Date</div>
    <div class="card-val">${{a.date}}</div></div>
  <div class="card"><div class="card-label">Change Type</div>
    <div class="card-val orange">${{a.typology}}</div></div>
  <div class="card"><div class="card-label">Cloud Cover</div>
    <div class="card-val">${{(a.cloud_frac*100||0).toFixed(1)}}%</div></div>
</div>

<a class="map-link" href="${{mapsUrl}}" target="_blank">
  📍 Open in Google Maps — ${{a.lat.toFixed(5)}}, ${{a.lng.toFixed(5)}}
</a>

<table>
  <thead><tr><th>Spectral Indicator</th><th>Value</th><th>Interpretation</th></tr></thead>
  <tbody>
    <tr><td>DW Tree Cover Δ</td><td>${{(a.dw_trees_delta||0).toFixed(4)}}</td>
        <td>${{(a.dw_trees_delta||0)<0 ? '⬇ Tree fraction declined' : '⬆ Tree fraction increased'}}</td></tr>
    <tr><td>Z-Score</td><td>${{(a.zscore||0).toFixed(2)}}σ</td>
        <td>${{Math.abs(a.zscore||0)>2 ? '⚠ Anomalous seasonal deviation' : 'Within normal range'}}</td></tr>
    <tr><td>Cloud Fraction</td><td>${{(a.cloud_frac*100||0).toFixed(1)}}%</td>
        <td>${{(a.cloud_frac||0)<0.15 ? '✓ Clean imagery' : '⚠ Partial cloud — verify on ground'}}</td></tr>
    <tr><td>Centroid Lat/Lon</td><td>${{a.lat.toFixed(5)}}, ${{a.lng.toFixed(5)}}</td>
        <td>WGS84 decimal degrees</td></tr>
  </tbody>
</table>

<table>
  <thead><tr><th colspan="2">Action Required</th></tr></thead>
  <tbody>
    <tr><td>1.</td><td>Forest Guard to physically verify site within <b>48 hours</b></td></tr>
    <tr><td>2.</td><td>Record species affected, extent of clearing, and presence of equipment/structures</td></tr>
    <tr><td>3.</td><td>Photograph site with GPS metadata enabled</td></tr>
    <tr><td>4.</td><td>Submit FIR/complaint if encroachment is confirmed</td></tr>
    <tr><td>5.</td><td>Update E-Netra ground-truth KML with verified location</td></tr>
  </tbody>
</table>

<div class="sig">
  E-Netra V4 · Guna Forest Division · Van Suraksha Alert System<br>
  Model: dw_multi_threat_v4 · Confidence threshold (HIGH): ≥ 0.65<br>
  This report is auto-generated. Ground verification is mandatory before legal action.<br><br>
  <button onclick="window.print()" style="background:#374151;color:#fff;border:none;
    padding:8px 18px;border-radius:6px;cursor:pointer;font-size:12px">🖨 Print / Save PDF</button>
</div>
</body></html>`;
}}


// ─── Alert KML download ─────────────────────────────────────────────────────
function dlAlertKml(alertId) {{
  const feat = FC.features.find(f => f.properties && f.properties.id === alertId);
  if(!feat || !feat.geometry) {{
    const byIdx = FC.features[alertId];
    if(!byIdx || !byIdx.geometry) {{ alert('No geometry for alert ' + alertId); return; }}
    _buildAlertKml(alertId, byIdx); return;
  }}
  _buildAlertKml(alertId, feat);
}}
function _buildAlertKml(alertId, feat) {{
  const p     = feat.properties || {{}};
  const al    = ALERTS.find(a => a.id === alertId) || p;
  const beat  = al.beat  || p.beat_name || '—';
  const lbl   = al.label || p.label     || 'ALERT';
  const sc    = al.score  !== undefined ? Number(al.score).toFixed(3)  : '—';
  const dt    = al.date   || '—';
  const ha    = al.area_ha!== undefined ? Number(al.area_ha).toFixed(2): '—';
  const dndvi = al.dNDVI  !== undefined ? Number(al.dNDVI).toFixed(4)  : '—';
  const dnbr  = al.dNBR   !== undefined ? Number(al.dNBR).toFixed(4)   : '—';
  const dtree = al.dw_trees_delta !== undefined ? Number(al.dw_trees_delta).toFixed(4) : '—';
  const z     = al.zscore !== undefined ? Number(al.zscore).toFixed(2)  : '—';

  function ringCoords(ring) {{
    return ring.map(c => c[0].toFixed(6)+','+c[1].toFixed(6)+',0').join(' ');
  }}
  function polyKml(geom) {{
    if(geom.type === 'Polygon') {{
      return '<Polygon><outerBoundaryIs><LinearRing><coordinates>' +
             ringCoords(geom.coordinates[0]) +
             '</coordinates></LinearRing></outerBoundaryIs></Polygon>';
    }} else if(geom.type === 'MultiPolygon') {{
      return '<MultiGeometry>' + geom.coordinates.map(poly =>
        '<Polygon><outerBoundaryIs><LinearRing><coordinates>' +
        ringCoords(poly[0]) +
        '</coordinates></LinearRing></outerBoundaryIs></Polygon>'
      ).join('') + '</MultiGeometry>';
    }}
    return '';
  }}
  const colHex = lbl==='HIGH'?'FF2020FF':lbl==='MEDIUM'?'FF0080FF':'FF00FFFF';
  const kml = '<?xml version="1.0" encoding="UTF-8"?>' +
    '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>' +
    '<name>E-Netra Alert \u2014 '+beat+' \u2014 '+dt+'</name>' +
    '<Style id="s"><LineStyle><color>'+colHex+'</color><width>3</width></LineStyle>' +
    '<PolyStyle><color>4400AAFF</color><fill>1</fill></PolyStyle></Style>' +
    '<Placemark><name>'+lbl+' \u2014 '+beat+'</name>' +
    '<description><![CDATA[Beat: '+beat+'<br/>Date: '+dt+'<br/>Score: '+sc+'<br/>' +
    'Area: '+ha+' ha<br/>dNDVI: '+dndvi+'<br/>dNBR: '+dnbr+'<br/>' +
    'dTrees: '+dtree+'<br/>Z: '+z+']]></description>' +
    '<styleUrl>#s</styleUrl>' + polyKml(feat.geometry) +
    '</Placemark></Document></kml>';

  const blob = new Blob([kml], {{type:'application/vnd.google-earth.kml+xml'}});
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = 'alert_'+beat.replace(/ /g,'_')+'_'+dt+'.kml';
  a.click(); URL.revokeObjectURL(a.href);
}}

// ─── Resizable panels ────────────────────────────────────────────────────
(function() {{
  function initResize(hId, getLeft, getRight, isRight) {{
    const h = document.getElementById(hId);
    if(!h) return;
    h.addEventListener('mousedown', function(e) {{
      e.preventDefault();
      h.classList.add('dragging');
      const startX = e.clientX;
      const el     = isRight ? getRight() : getLeft();
      const startW = el.getBoundingClientRect().width;
      function onMove(ev) {{
        const dx = ev.clientX - startX;
        const nw = isRight
          ? Math.max(260, Math.min(700, startW - dx))
          : Math.max(140, Math.min(380, startW + dx));
        el.style.width = nw + 'px';
        el.style.flexShrink = '0';
      }}
      function onUp() {{
        h.classList.remove('dragging');
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup',   onUp);
      }}
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup',   onUp);
    }});
  }}
  initResize('rh-left',  ()=>document.getElementById('sidebar'),     null, false);
  initResize('rh-right', null, ()=>document.getElementById('rightPanel'), true);
}})();

// ── Row → map zoom ─────────────────────────────────────────────────────────────────
let _highlightLayer = null;
function zoomToAlert(id) {{
  if(_highlightLayer) {{ map.removeLayer(_highlightLayer); _highlightLayer = null; }}
  document.querySelectorAll('#tbl-body tr').forEach(r => r.classList.remove('row-active'));
  const row = document.getElementById('row-' + id);
  if(row) {{ row.classList.add('row-active'); row.scrollIntoView({{block:'nearest'}}); }}
  const feat = FC.features.find(f => f.properties && f.properties.id === id);
  if(!feat || !feat.geometry) return;
  _highlightLayer = L.geoJSON(feat, {{
    style: {{ color:'#ffffff', weight:3, fillColor:'#fff', fillOpacity:0.25 }}
  }}).addTo(map);
  try {{ map.fitBounds(_highlightLayer.getBounds().pad(0.25)); }} catch(e){{}}
}}

</script>
</body></html>"""

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Dashboard → {OUT_HTML}")
    print(f"  Alerts: {n_total}  HIGH:{n_high}  MEDIUM:{n_med}  LOW:{n_low}")
    return str(OUT_HTML)


if __name__ == "__main__":
    build_dashboard()
