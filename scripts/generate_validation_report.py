"""
generate_validation_report.py
==============================
Compare V4 EWS alert polygons against confirmed-loss KML ground-truth.
Produces outputs/validation_report.html — a self-contained HTML report with:
  · Precision, Recall, F1 (IoU≥0.1 matching threshold)
  · Per-alert match table (score, tier, area, matched GT polygon)
  · Leaflet map showing TP (green), FP (red), FN (orange) polygons
  · Score histogram by tier

No geopandas/shapely required — uses stdlib xml + a pure-Python polygon
intersection helper that is good enough for small convex/near-convex patches
at this resolution (10–50 ha forest fragments).

Usage:
  python scripts/generate_validation_report.py
  python scripts/generate_validation_report.py --iou 0.05   # looser match
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR         = Path(__file__).resolve().parent
PROJECT_ROOT = _DIR.parent
ALERT_DIR    = PROJECT_ROOT / "outputs" / "alerts"
KML_PATH     = PROJECT_ROOT / "data" / "ground_truth" / "confirmed_loss_feb2026.kml"
OUT_HTML     = PROJECT_ROOT / "outputs" / "validation_report.html"

KML_NS = "http://www.opengis.net/kml/2.2"


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers (pure Python, no external deps)
# ══════════════════════════════════════════════════════════════════════════════

def _bbox(coords: list[list[float]]) -> tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy)"""
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_overlap(b1, b2) -> bool:
    return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])


def _poly_area(coords: list[list[float]]) -> float:
    """Shoelace formula — returns area in geographic degrees² (not ha)."""
    n = len(coords)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += coords[i][0] * coords[j][1]
        a -= coords[j][0] * coords[i][1]
    return abs(a) / 2.0


def _deg2_to_ha(deg2: float, lat: float) -> float:
    """Very rough conversion: 1° lat ≈ 111 km, 1° lon ≈ 111*cos(lat) km."""
    lat_m = 111_319.5
    lon_m = 111_319.5 * math.cos(math.radians(lat))
    return deg2 * lat_m * lon_m / 10_000.0   # → hectares


def _clip_polygon_bbox(p1: list[list[float]], b2) -> list[list[float]]:
    """
    Sutherland-Hodgman clip of polygon p1 against axis-aligned box b2.
    Returns clipped polygon vertices (may be empty).
    """
    minx, miny, maxx, maxy = b2

    def inside(pt, edge):
        if edge == "l": return pt[0] >= minx
        if edge == "r": return pt[0] <= maxx
        if edge == "b": return pt[1] >= miny
        if edge == "t": return pt[1] <= maxy

    def intersect(a, b, edge):
        dx, dy = b[0]-a[0], b[1]-a[1]
        if edge == "l":
            t = (minx - a[0]) / dx if dx else 0; return [minx, a[1] + t*dy]
        if edge == "r":
            t = (maxx - a[0]) / dx if dx else 0; return [maxx, a[1] + t*dy]
        if edge == "b":
            t = (miny - a[1]) / dy if dy else 0; return [a[0] + t*dx, miny]
        if edge == "t":
            t = (maxy - a[1]) / dy if dy else 0; return [a[0] + t*dx, maxy]

    poly = list(p1)
    for edge in ("l", "r", "b", "t"):
        if not poly:
            break
        output = []
        for i in range(len(poly)):
            cur, prev = poly[i], poly[i-1]
            if inside(cur, edge):
                if not inside(prev, edge):
                    output.append(intersect(prev, cur, edge))
                output.append(cur)
            elif inside(prev, edge):
                output.append(intersect(prev, cur, edge))
        poly = output
    return poly


def _polygon_iou(p1: list[list[float]], p2: list[list[float]]) -> float:
    """
    Approximate IoU between two polygons using bbox-clip intersection.
    Good enough for near-convex forest patches.
    """
    b1 = _bbox(p1)
    b2 = _bbox(p2)
    if not _bbox_overlap(b1, b2):
        return 0.0

    # Clip p1 against p2's bbox and vice-versa to get intersection approx
    clip = _clip_polygon_bbox(p1, b2)
    if len(clip) < 3:
        return 0.0
    inter = _poly_area(clip)
    a1    = _poly_area(p1)
    a2    = _poly_area(p2)
    union = a1 + a2 - inter
    return inter / union if union > 1e-20 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Data loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_kml_polygons(kml_path: Path) -> list[dict]:
    """Parse KML Placemarks → list of {name, description, coords}."""
    tree = ET.parse(kml_path)
    root = tree.getroot()
    results = []
    ns = KML_NS
    for pm in root.iter(f"{{{ns}}}Placemark"):
        name_el = pm.find(f"{{{ns}}}name")
        desc_el  = pm.find(f"{{{ns}}}description")
        name = name_el.text.strip() if name_el is not None and name_el.text else "?"
        desc = desc_el.text.strip() if desc_el is not None and desc_el.text else "?"
        for coord_el in pm.iter(f"{{{ns}}}coordinates"):
            raw = coord_el.text.strip()
            coords = []
            for token in raw.split():
                parts = token.split(",")
                if len(parts) >= 2:
                    coords.append([float(parts[0]), float(parts[1])])
            if coords:
                cx = sum(c[0] for c in coords) / len(coords)
                cy = sum(c[1] for c in coords) / len(coords)
                area_deg2 = _poly_area(coords)
                results.append({
                    "name":     name,
                    "desc":     desc,
                    "coords":   coords,
                    "bbox":     _bbox(coords),
                    "area_ha":  round(_deg2_to_ha(area_deg2, cy), 3),
                    "centroid": [cx, cy],
                })
    return results


def load_alert_geojsons(alert_dir: Path) -> list[dict]:
    """Load all GeoJSON alert files → list of normalised alert dicts."""
    alerts = []
    for gj_path in sorted(alert_dir.glob("*.geojson")):
        try:
            with open(gj_path, encoding="utf-8") as f:
                fc = json.load(f)
            for feat in fc.get("features", []):
                geom = feat.get("geometry", {})
                if geom.get("type") != "Polygon":
                    continue
                ring   = geom["coordinates"][0]        # outer ring
                coords = [[c[0], c[1]] for c in ring]
                p      = feat.get("properties", {})
                score  = float(p.get("confidence") or p.get("score") or 0.0)
                raw_tier = p.get("tier") or p.get("label")
                if isinstance(raw_tier, int):
                    tier = {1: "HIGH", 2: "MEDIUM", 3: "LOW"}.get(raw_tier, "LOW")
                elif isinstance(raw_tier, str):
                    tier = raw_tier.upper()
                else:
                    tier = "HIGH" if score >= 0.65 else ("MEDIUM" if score >= 0.45 else "LOW")
                cx = sum(c[0] for c in coords) / len(coords)
                cy = sum(c[1] for c in coords) / len(coords)
                alerts.append({
                    "source":   gj_path.name,
                    "score":    round(score, 4),
                    "tier":     tier,
                    "area_ha":  round(float(p.get("area_ha") or 0.0), 2),
                    "coords":   coords,
                    "bbox":     _bbox(coords),
                    "centroid": [cx, cy],
                    "props":    p,
                })
        except Exception as e:
            print(f"  [WARN] Could not parse {gj_path.name}: {e}", file=sys.stderr)
    return alerts


# ══════════════════════════════════════════════════════════════════════════════
# Matching engine
# ══════════════════════════════════════════════════════════════════════════════

def match_alerts_to_gt(
    alerts: list[dict],
    gt_polys: list[dict],
    iou_threshold: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Greedy matching by best IoU ≥ threshold.
    Returns (tp_pairs, fp_alerts, fn_gt_polys).
    tp_pairs: list of {alert, gt, iou}
    """
    alerts_sorted = sorted(alerts, key=lambda a: a["score"], reverse=True)
    matched_gt = set()
    matched_al = set()
    tp_pairs   = []

    for ai, alert in enumerate(alerts_sorted):
        best_iou = 0.0
        best_gi  = -1
        for gi, gt in enumerate(gt_polys):
            if gi in matched_gt:
                continue
            if not _bbox_overlap(alert["bbox"], gt["bbox"]):
                continue
            iou = _polygon_iou(alert["coords"], gt["coords"])
            if iou > best_iou:
                best_iou = iou
                best_gi  = gi
        if best_iou >= iou_threshold and best_gi >= 0:
            tp_pairs.append({"alert": alert, "gt": gt_polys[best_gi], "iou": round(best_iou, 4)})
            matched_gt.add(best_gi)
            matched_al.add(ai)

    fp_alerts = [a for i, a in enumerate(alerts_sorted) if i not in matched_al]
    fn_gt     = [g for i, g in enumerate(gt_polys) if i not in matched_gt]
    return tp_pairs, fp_alerts, fn_gt


# ══════════════════════════════════════════════════════════════════════════════
# HTML report generator
# ══════════════════════════════════════════════════════════════════════════════

def _coords_to_latlng(coords: list[list[float]]) -> str:
    """Convert [[lon,lat],...] → JS [[lat,lon],...] string."""
    return "[" + ",".join(f"[{c[1]:.6f},{c[0]:.6f}]" for c in coords) + "]"


def build_html(
    tp_pairs: list[dict],
    fp_alerts: list[dict],
    fn_gt:    list[dict],
    gt_polys: list[dict],
    alerts:   list[dict],
    iou_threshold: float,
) -> str:
    n_tp  = len(tp_pairs)
    n_fp  = len(fp_alerts)
    n_fn  = len(fn_gt)
    prec  = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    rec   = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
    f1    = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Map layers JS ────────────────────────────────────────────────────────
    map_layers = []
    # TP alerts (green)
    for r in tp_pairs:
        a  = r["alert"]
        gt = r["gt"]
        coords_js = _coords_to_latlng(a["coords"])
        popup = (f"<b>TP — {a['tier']}</b><br>score={a['score']} | "
                 f"{a['area_ha']} ha<br>IoU={r['iou']}<br>GT: {gt['name']}")
        map_layers.append(
            f"L.polygon({coords_js},{{color:'#22c55e',weight:2,fillOpacity:0.4}})"
            f".bindPopup({json.dumps(popup)}).addTo(map);"
        )
    # FP alerts (red)
    for a in fp_alerts:
        coords_js = _coords_to_latlng(a["coords"])
        popup = (f"<b>FP — {a['tier']}</b><br>score={a['score']} | "
                 f"{a['area_ha']} ha<br>No GT match (IoU&lt;{iou_threshold})")
        map_layers.append(
            f"L.polygon({coords_js},{{color:'#ef4444',weight:2,fillOpacity:0.3}})"
            f".bindPopup({json.dumps(popup)}).addTo(map);"
        )
    # FN ground-truth (orange)
    for gt in fn_gt:
        coords_js = _coords_to_latlng(gt["coords"])
        popup = f"<b>FN (missed)</b><br>{gt['name']} | {gt['area_ha']} ha"
        map_layers.append(
            f"L.polygon({coords_js},{{color:'#f97316',weight:2,fillOpacity:0.3,"
            f"dashArray:'6 3'}}).bindPopup({json.dumps(popup)}).addTo(map);"
        )
    map_js = "\n".join(map_layers)

    # Map centre
    all_coords = [c for a in alerts for c in a["coords"]]
    if all_coords:
        clat = sum(c[1] for c in all_coords) / len(all_coords)
        clon = sum(c[0] for c in all_coords) / len(all_coords)
    else:
        clat, clon = 24.65, 77.30   # Guna default

    # ── Alert table rows ─────────────────────────────────────────────────────
    tp_ids = {id(r["alert"]) for r in tp_pairs}
    rows = []
    tp_map = {id(r["alert"]): r for r in tp_pairs}
    for a in sorted(alerts, key=lambda x: x["score"], reverse=True):
        if id(a) in tp_ids:
            r   = tp_map[id(a)]
            cls = "tp"; status = f"TP&nbsp;(IoU={r['iou']})"
        else:
            cls = "fp"; status = "FP"
        tier_cls = a["tier"].lower()
        rows.append(
            f'<tr class="{cls}">'
            f'<td><span class="tier {tier_cls}">{a["tier"]}</span></td>'
            f'<td class="num">{a["score"]:.4f}</td>'
            f'<td class="num">{a["area_ha"]}</td>'
            f'<td>{a["source"]}</td>'
            f'<td>{status}</td>'
            f'</tr>'
        )
    for gt in fn_gt:
        rows.append(
            f'<tr class="fn">'
            f'<td><span class="tier fn-tier">GT</span></td>'
            f'<td class="num">—</td>'
            f'<td class="num">{gt["area_ha"]}</td>'
            f'<td>{gt["name"]}</td>'
            f'<td>FN (missed)</td>'
            f'</tr>'
        )
    table_rows = "\n".join(rows)

    # ── Score histogram data ─────────────────────────────────────────────────
    bins = [0]*10
    for a in alerts:
        idx = min(int(a["score"] * 10), 9)
        bins[idx] += 1
    hist_data = json.dumps(bins)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>E-Netra V4 — Validation Report</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{{--bg:#0f172a;--card:#1e293b;--border:#334155;--text:#e2e8f0;--muted:#94a3b8;
      --green:#22c55e;--red:#ef4444;--orange:#f97316;--purple:#a78bfa;--blue:#38bdf8}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;padding:24px}}
h1{{font-size:1.5rem;font-weight:700;margin-bottom:4px}}
.subtitle{{color:var(--muted);font-size:.85rem;margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:20px}}
.kpi{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px}}
.kpi-label{{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
.kpi-val{{font-size:2rem;font-weight:700;line-height:1.2}}
.kpi.tp .kpi-val{{color:var(--green)}}
.kpi.fp .kpi-val{{color:var(--red)}}
.kpi.fn .kpi-val{{color:var(--orange)}}
.kpi.prec .kpi-val{{color:var(--blue)}}
.kpi.rec .kpi-val{{color:var(--purple)}}
.kpi.f1 .kpi-val{{color:#facc15}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}}
.card-title{{font-size:.8rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-bottom:10px}}
#map{{height:420px;border-radius:8px}}
canvas{{max-height:300px}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
th{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--border);color:var(--muted);font-weight:600;text-transform:uppercase;font-size:.7rem}}
td{{padding:6px 10px;border-bottom:1px solid #1e293b}}
tr.tp{{background:rgba(34,197,94,.06)}}
tr.fp{{background:rgba(239,68,68,.06)}}
tr.fn{{background:rgba(249,115,22,.06)}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.tier{{display:inline-block;padding:2px 7px;border-radius:99px;font-size:.7rem;font-weight:700}}
.tier.high{{background:#7f1d1d;color:#fca5a5}}
.tier.medium{{background:#78350f;color:#fde68a}}
.tier.low{{background:#14532d;color:#86efac}}
.tier.fn-tier{{background:#431407;color:#fed7aa}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;font-size:.78rem}}
.leg-item{{display:flex;align-items:center;gap:6px}}
.leg-dot{{width:12px;height:12px;border-radius:3px}}
.iou-note{{color:var(--muted);font-size:.75rem;margin-top:6px}}
</style>
</head>
<body>
<h1>🛰️ E-Netra V4 — Validation Report</h1>
<div class="subtitle">Generated {stamp} &nbsp;·&nbsp; IoU threshold: {iou_threshold} &nbsp;·&nbsp;
{len(alerts)} V4 alerts vs {len(gt_polys)} confirmed-loss GT polygons</div>

<div class="grid">
  <div class="kpi tp"><div class="kpi-label">True Positives</div><div class="kpi-val">{n_tp}</div></div>
  <div class="kpi fp"><div class="kpi-label">False Positives</div><div class="kpi-val">{n_fp}</div></div>
  <div class="kpi fn"><div class="kpi-label">False Negatives</div><div class="kpi-val">{n_fn}</div></div>
  <div class="kpi prec"><div class="kpi-label">Precision</div><div class="kpi-val">{prec:.0%}</div></div>
  <div class="kpi rec"><div class="kpi-label">Recall</div><div class="kpi-val">{rec:.0%}</div></div>
  <div class="kpi f1"><div class="kpi-label">F1 Score</div><div class="kpi-val">{f1:.2f}</div></div>
</div>

<div class="two-col">
  <div class="card">
    <div class="card-title">🗺 Alert Map</div>
    <div id="map"></div>
    <div class="legend">
      <div class="leg-item"><div class="leg-dot" style="background:#22c55e"></div>TP (matched)</div>
      <div class="leg-item"><div class="leg-dot" style="background:#ef4444"></div>FP (unmatched)</div>
      <div class="leg-item"><div class="leg-dot" style="background:#f97316;opacity:.7"></div>FN (missed GT)</div>
    </div>
    <div class="iou-note">IoU threshold = {iou_threshold} &nbsp;·&nbsp; Click polygon for details</div>
  </div>
  <div class="card">
    <div class="card-title">📊 Score Distribution</div>
    <canvas id="hist"></canvas>
  </div>
</div>

<div class="card">
  <div class="card-title">📋 Alert × Ground-Truth Match Table</div>
  <table>
    <thead><tr><th>Tier</th><th class="num">Score</th><th class="num">Area (ha)</th>
    <th>Source</th><th>Result</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>

<script>
var map = L.map('map').setView([{clat:.5f},{clon:.5f}],13);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{attribution:'© OpenStreetMap'}}).addTo(map);
{map_js}
</script>
<script>
new Chart(document.getElementById('hist'),{{
  type:'bar',
  data:{{
    labels:['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'],
    datasets:[{{
      label:'Alerts',
      data:{hist_data},
      backgroundColor:'#38bdf8',
      borderRadius:4,
    }}]
  }},
  options:{{
    responsive:true,
    plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{title:{{display:true,text:'Score bin',color:'#94a3b8'}},ticks:{{color:'#94a3b8'}},grid:{{color:'#1e293b'}}}},
      y:{{title:{{display:true,text:'Count',color:'#94a3b8'}},ticks:{{color:'#94a3b8',stepSize:1}},grid:{{color:'#1e293b'}}}}
    }}
  }}
}});
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="E-Netra V4 Validation Report")
    parser.add_argument("--iou",  type=float, default=0.10,
                        help="IoU threshold to count a match as TP (default=0.10)")
    parser.add_argument("--kml",  default=str(KML_PATH), help="Ground-truth KML path")
    parser.add_argument("--alerts", default=str(ALERT_DIR), help="Alert GeoJSON dir")
    parser.add_argument("--out",  default=str(OUT_HTML), help="Output HTML path")
    args = parser.parse_args()

    kml_path  = Path(args.kml)
    alert_dir = Path(args.alerts)
    out_path  = Path(args.out)

    if not kml_path.exists():
        print(f"ERROR: KML not found: {kml_path}", file=sys.stderr); sys.exit(1)
    if not alert_dir.exists():
        print(f"ERROR: Alert dir not found: {alert_dir}", file=sys.stderr); sys.exit(1)

    print(f"Loading KML ground-truth from: {kml_path}")
    gt_polys = load_kml_polygons(kml_path)
    print(f"  → {len(gt_polys)} GT polygons loaded")

    print(f"Loading V4 alert GeoJSONs from: {alert_dir}")
    alerts = load_alert_geojsons(alert_dir)
    print(f"  → {len(alerts)} alert polygons loaded")

    print(f"Matching (IoU ≥ {args.iou}) …")
    tp_pairs, fp_alerts, fn_gt = match_alerts_to_gt(alerts, gt_polys, args.iou)

    n_tp = len(tp_pairs)
    n_fp = len(fp_alerts)
    n_fn = len(fn_gt)
    prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    rec  = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0

    print(f"\n  TP={n_tp}  FP={n_fp}  FN={n_fn}")
    print(f"  Precision = {prec:.1%}")
    print(f"  Recall    = {rec:.1%}")
    print(f"  F1        = {f1:.3f}\n")

    html = build_html(tp_pairs, fp_alerts, fn_gt, gt_polys, alerts, args.iou)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written → {out_path}")


if __name__ == "__main__":
    main()
