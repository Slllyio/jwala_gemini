"""
enetra_overlay.py — E-Netra Alert Overlay Map Generator
=========================================================
Reusable module that generates the canonical E-Netra satellite overlay HTML
(red/yellow alert polygons + green KML ground-truth, Esri satellite basemap,
info panel, legend, layer switcher).

USAGE (command-line)
--------------------
    python scripts/enetra_overlay.py \\
        --alerts   outputs/alerts/mar_ki_mahu_2026-02-09.geojson \\
        --kml      "C:/path/to/Export.kml" \\
        --beat     "Mar Ki Mahu" \\
        --range    North_Guna \\
        --out      outputs/my_overlay.html

    # Multiple alert GeoJSONs (all dates):
    python scripts/enetra_overlay.py \\
        --alerts   outputs/alerts/mar_ki_mahu_*.geojson \\
        --kml      "C:/path/to/Export.kml" \\
        --beat     "Mar Ki Mahu" \\
        --range    North_Guna \\
        --out      outputs/my_overlay.html

USAGE (Python API)
------------------
    from scripts.enetra_overlay import build_overlay_map

    build_overlay_map(
        alert_geojsons = ["outputs/alerts/mar_ki_mahu_2026-02-09.geojson"],
        kml_path       = r"C:/Users/.../Export.kml",
        beat_name      = "Mar Ki Mahu",
        range_name     = "North_Guna",
        out_path       = "outputs/my_overlay.html",
    )

OUTPUT
------
A self-contained HTML file you open directly in any browser.
No server required.

PATCH COLOURING
---------------
V3 multi-threat scorer (dw_multi_threat_score_maxx) runs on every patch
using the range-harmonic baseline when range_name is provided:
  RED  (solid)   — HIGH  (score ≥ 0.70) or MEDIUM (score ≥ 0.45)
  GOLD (dashed)  — LOW / NO ALERT  (candidate, below threshold)
  GREEN (dashed) — Ground-truth KML polygon
"""

from __future__ import annotations
import argparse, glob, json, re, sys
from pathlib import Path
from typing import Union, List

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_overlay_map(
    alert_geojsons: List[Union[str, Path]],
    out_path:       Union[str, Path],
    kml_path:       Union[str, Path, None] = None,
    beat_name:      str = "",
    range_name:     str = "",
    center_lat:     float | None = None,
    center_lon:     float | None = None,
    zoom:           int = 15,
) -> Path:
    """
    Generate an E-Netra overlay HTML file.

    Parameters
    ----------
    alert_geojsons : list of paths to GeoJSON FeatureCollections.
        Each file's features are scored and rendered as patches.
        Expected feature properties: dw_trees_delta, dw_crops_delta,
        dw_built_delta, area_ha, cusum_score, dw_trees_zscore, cloud_frac.
    out_path : destination HTML file path.
    kml_path : optional path to a KML file containing ONE Placemark Polygon.
        Rendered as the green dashed ground-truth overlay.
    beat_name : displayed in header / info panel.
    range_name : if set and the range model exists in RANGE_DW_MODELS,
        the V3 scorer uses the harmonic baseline instead of division averages.
    center_lat, center_lon : optional map centre. Defaults to KML centroid or
        first patch centroid.
    zoom : initial map zoom level.

    Returns
    -------
    Path to the written HTML file.
    """
    # -- imports kept local so the module works even without scorer installed --
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from _simple_dw_score import (
            dw_multi_threat_score_maxx, RANGE_DW_MODELS, _range_baseline,
        )
        SCORER_OK = True
    except ImportError:
        SCORER_OK = False

    # ── 1. Parse KML ─────────────────────────────────────────────────────────
    kml_ring: list[list[float]] = []
    kml_name = ""
    kml_desc = ""
    if kml_path and Path(kml_path).exists():
        raw = Path(kml_path).read_text(encoding="utf-8")
        m = re.search(r"<coordinates>(.*?)</coordinates>", raw, re.DOTALL)
        if m:
            for tok in m.group(1).split():
                tok = tok.strip()
                if tok:
                    parts = tok.split(",")
                    try:
                        kml_ring.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        pass
        nm = re.search(r"<name>(.*?)</name>", raw, re.DOTALL)
        dm = re.search(r"<description>(.*?)</description>", raw, re.DOTALL)
        if nm: kml_name = nm.group(1).strip()
        if dm: kml_desc = dm.group(1).strip()

    # ── 2. Compute KML area & centroid ───────────────────────────────────────
    import math

    def _centroid(ring):
        n = len(ring)
        return (sum(p[0] for p in ring)/n, sum(p[1] for p in ring)/n) if n else (0.0, 0.0)

    def _bbox(ring):
        lons = [p[0] for p in ring]; lats = [p[1] for p in ring]
        return min(lons), min(lats), max(lons), max(lats)

    kml_cx, kml_cy = _centroid(kml_ring) if kml_ring else (0.0, 0.0)
    kml_area_ha: float = 0.0
    if kml_ring:
        bb = _bbox(kml_ring)
        span_m  = (bb[2]-bb[0]) * 111_320 * math.cos(math.radians(kml_cy))
        span_lat= (bb[3]-bb[1]) * 111_320
        kml_area_ha = span_m * span_lat / 10_000

    # ── 3. Load & score patches ───────────────────────────────────────────────
    def _reconstruct_and_score(props: dict, date_str: str):
        if not SCORER_OK:
            return {"score": 0.0, "label": "UNKNOWN", "typology": "UNKNOWN"}
        mu_r, std_r = _range_baseline(range_name, date_str) if range_name else (0.4, 0.1)
        delta  = props.get("dw_trees_delta", 0.0)
        cr_d   = props.get("dw_crops_delta", 0.0)
        bu_d   = props.get("dw_built_delta", 0.0)
        cloud  = props.get("cloud_frac", 0.0)
        ta     = max(0.0, min(1.0, mu_r + delta))
        tb     = max(0.0, min(1.0, ta - delta))
        cpb    = max(0.0, -cr_d) if cr_d < 0 else 0.05
        cpa    = max(0.0, cpb + cr_d)
        bub    = max(0.0, -bu_d) if bu_d < 0 else 0.02
        bua    = max(0.0, bub + bu_d)
        return dw_multi_threat_score_maxx(
            trees_after=ta, trees_before=tb,
            crops_after=cpa, crops_before=cpb,
            built_after=bua, built_before=bub,
            date_str=date_str, cloud_frac=cloud,
            range_name=range_name,
        )

    alert_feats:     list[dict] = []
    candidate_feats: list[dict] = []
    all_latlons: list[tuple[float,float]] = []

    for gpath in [Path(p) for p in alert_geojsons]:
        if not gpath.exists():
            continue
        date_str = gpath.stem   # e.g. "mar_ki_mahu_2026-02-09"
        # extract YYYY-MM-DD from anywhere in the stem
        dm = re.search(r"\d{4}-\d{2}-\d{2}", gpath.stem)
        if dm: date_str = dm.group(0)

        fc = json.loads(gpath.read_text(encoding="utf-8"))
        for feat in fc.get("features", []):
            props = feat.get("properties", {})
            r = _reconstruct_and_score(props, date_str)
            out_feat = json.loads(json.dumps(feat))
            out_feat["properties"].update({
                "_date":     date_str,
                "_score":    round(r["score"], 3),
                "_label":    r["label"],
                "_typology": r["typology"],
                "_delta":    props.get("dw_trees_delta", 0.0),
                "_area":     props.get("area_ha", 0.0),
                "_cusum":    props.get("cusum_score", 0.0),
                "_z":        props.get("dw_trees_zscore", 0.0),
            })
            # collect coords for centroid fallback
            try:
                geom = feat["geometry"]
                ring0 = (geom["coordinates"][0] if geom["type"] == "Polygon"
                         else geom["coordinates"][0][0])
                for c in ring0:
                    all_latlons.append((c[1], c[0]))
            except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")

            if r["label"] in ("HIGH", "MEDIUM"):
                alert_feats.append(out_feat)
            else:
                candidate_feats.append(out_feat)

    n_high = sum(1 for f in alert_feats if f["properties"]["_label"] == "HIGH")

    # ── 4. Map centre ─────────────────────────────────────────────────────────
    if center_lat is None or center_lon is None:
        if kml_ring:
            center_lon, center_lat = kml_cx, kml_cy
        elif all_latlons:
            center_lat = sum(p[0] for p in all_latlons) / len(all_latlons)
            center_lon = sum(p[1] for p in all_latlons) / len(all_latlons)
        else:
            center_lat, center_lon = 24.93, 77.27  # Guna fallback

    # ── 5. Range model stats for info panel ───────────────────────────────────
    r2_str   = "n/a"
    mu_str   = "n/a"
    std_str  = "n/a"
    if SCORER_OK and range_name and range_name in RANGE_DW_MODELS:
        model = RANGE_DW_MODELS[range_name]
        r2_str  = f"{model.r2:.3f}"
        try:
            from _simple_dw_score import _range_baseline as _rb
            mu_r, std_r = _rb(range_name, f"{date_str}")
            mu_str  = f"{mu_r:.3f}"
            std_str = f"{std_r:.3f}"
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")

    # ── 6. Serialise to JS ────────────────────────────────────────────────────
    kml_js  = json.dumps(kml_ring,       ensure_ascii=False)
    alrt_js = json.dumps(alert_feats,    ensure_ascii=False)
    cand_js = json.dumps(candidate_feats,ensure_ascii=False)

    kml_popup = (f'<b style="color:#3fb950">✓ Ground Truth KML</b><br>'
                 f'{kml_name}<br>{kml_desc}<br>Area: ~{kml_area_ha:.2f} ha'
                 if kml_ring else "")

    badge_txt   = f"⚠ {n_high} HIGH ALERT{'S' if n_high!=1 else ''} DETECTED" if n_high else "ℹ CANDIDATES ONLY"
    badge_color = "#f85149" if n_high else "#d29922"
    header_beat = beat_name or "Unknown Beat"
    kml_match_html = (f'<div class="kml-match">✓ Field KML loaded<br>&nbsp;{kml_name} ({kml_area_ha:.2f} ha)</div>'
                      if kml_ring else "")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>E-Netra — {header_beat}</title>
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',sans-serif;background:#0d1117;color:#e6edf3;height:100vh;display:flex;flex-direction:column}}
  #header{{background:linear-gradient(135deg,#1a2332 0%,#0d1117 100%);border-bottom:1px solid #21262d;
           padding:12px 20px;display:flex;align-items:center;justify-content:space-between}}
  #header h1{{font-size:1.05rem;font-weight:600;color:#58a6ff;letter-spacing:.5px}}
  #header .badge{{color:#fff;font-size:.75rem;padding:3px 10px;border-radius:12px;font-weight:600;
                  background:{badge_color}}}
  #map{{flex:1}}
  .legend{{background:rgba(13,17,23,.92);border:1px solid #21262d;border-radius:8px;
           padding:12px 16px;font-size:.82rem;line-height:2;backdrop-filter:blur(8px)}}
  .legend-item{{display:flex;align-items:center;gap:8px}}
  .swatch{{width:18px;height:14px;border-radius:3px;display:inline-block}}
  .info-box{{background:rgba(13,17,23,.92);border:1px solid #21262d;border-radius:8px;
             padding:12px 16px;font-size:.8rem;line-height:1.7;max-width:250px;backdrop-filter:blur(8px)}}
  .info-box b{{color:#58a6ff}}
  .info-box .kml-match{{color:#3fb950;font-weight:600;margin-top:6px;font-size:.78rem}}
</style>
</head>
<body>
<div id="header">
  <h1>🌳 E-Netra — Alert Overlay · {header_beat}{(' · Range: '+range_name) if range_name else ''}</h1>
  <span class="badge">{badge_txt}</span>
</div>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map=L.map('map',{{center:[{center_lat},{center_lon}],zoom:{zoom},zoomControl:true}});
const satellite=L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{attribution:'Esri World Imagery',maxZoom:20}});
const osm=L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{attribution:'© OpenStreetMap',maxZoom:19}});
satellite.addTo(map);

// Ground-truth KML
const kmlRing={kml_js};
let kmlLayer=null;
if(kmlRing.length){{
  kmlLayer=L.polygon(kmlRing.map(c=>[c[1],c[0]]),
    {{color:'#3fb950',weight:2.5,dashArray:'8,5',fillColor:'#3fb950',fillOpacity:.12}})
    .addTo(map).bindPopup('{kml_popup}');
}}

// Alert patches (HIGH / MEDIUM)
const alertFeatures={alrt_js};
alertFeatures.forEach(f=>{{
  const p=f.properties,g=f.geometry;
  const rings=g.type==='Polygon'?[g.coordinates[0]]:g.coordinates.map(r=>r[0]);
  rings.forEach(ring=>{{
    const ll=ring.map(c=>[c[1],c[0]]);
    const color=p._label==='HIGH'?'#f85149':'#d29922';
    L.polygon(ll,{{color,weight:2.5,fillColor:color,fillOpacity:p._label==='HIGH'?.50:.35}})
     .addTo(map).bindPopup(`
      <b style="color:${{color}}">⚠ ${{p._label}} — V3 Score: ${{p._score}}</b><br>
      <b>Date:</b> ${{p._date}}<br>
      <b>Patch ID:</b> ${{f.id||'—'}}<br>
      <b>ΔTrees:</b> ${{(p._delta*100).toFixed(2)}} pp<br>
      <b>Area:</b> ${{p._area.toFixed(2)}} ha<br>
      <b>DW z-score:</b> ${{p._z.toFixed(2)}}<br>
      <b>CuSuM:</b> ${{p._cusum.toFixed(3)}}<br>
      <b>Typology:</b> ${{p._typology}}`);
  }});
}});

// Candidate patches (LOW / NO ALERT)
const candidateGroup=L.layerGroup();
const candidateFeatures={cand_js};
candidateFeatures.forEach(f=>{{
  const p=f.properties,g=f.geometry;
  const rings=g.type==='Polygon'?[g.coordinates[0]]:g.coordinates.map(r=>r[0]);
  rings.forEach(ring=>{{
    const ll=ring.map(c=>[c[1],c[0]]);
    L.polygon(ll,{{color:'#e3b341',weight:1.5,dashArray:'4,3',fillColor:'#e3b341',fillOpacity:.18}})
     .addTo(candidateGroup).bindPopup(`
      <b style="color:#e3b341">⬡ ${{p._label}} — Score: ${{p._score}}</b><br>
      <b>Date:</b> ${{p._date}}<br>
      <b>Patch ID:</b> ${{f.id||'—'}}<br>
      <b>ΔTrees:</b> ${{(p._delta*100).toFixed(2)}} pp<br>
      <b>Area:</b> ${{p._area.toFixed(2)}} ha<br>
      <b>CuSuM:</b> ${{p._cusum.toFixed(3)}}`);
  }});
}});
candidateGroup.addTo(map);

// Legend
const legend=L.control({{position:'bottomright'}});
legend.onAdd=()=>{{
  const d=L.DomUtil.create('div','legend');
  d.innerHTML=`
    <b style="font-size:.9rem;color:#58a6ff">Map Legend</b><br>
    <div class="legend-item"><span class="swatch" style="background:#f85149;opacity:.8"></span>HIGH alert (V3 ≥ 0.70)</div>
    <div class="legend-item"><span class="swatch" style="background:#d29922;opacity:.8"></span>MEDIUM alert (V3 ≥ 0.45)</div>
    <div class="legend-item"><span class="swatch" style="background:#e3b341;opacity:.5;border:1px dashed #e3b341"></span>Candidate (LOW/NO ALERT)</div>
    ${{kmlRing.length?'<div class="legend-item"><span class="swatch" style="background:#3fb950;opacity:.5;border:2px dashed #3fb950"></span>Ground truth KML</div>':''}}
  `;
  return d;
}};
legend.addTo(map);

// Info panel
const info=L.control({{position:'topleft'}});
info.onAdd=()=>{{
  const d=L.DomUtil.create('div','info-box');
  d.innerHTML=`
    ${{'{beat_name}'?'<b>Beat:</b> {beat_name}<br>':''}}
    ${{'{range_name}'?'<b>Range:</b> {range_name}<br>':''}}
    ${{'{r2_str}'!=='n/a'?'<b>Range model R²:</b> {r2_str}<br>':''}}
    <b>HIGH patches:</b> {n_high}<br>
    <b>Total patches:</b> {len(alert_feats)+len(candidate_feats)}<br>
    <hr style="border-color:#21262d;margin:6px 0">
    Click any polygon for details
    {kml_match_html}
  `;
  return d;
}};
info.addTo(map);

// Layer switcher
L.control.layers(
  {{'Satellite':satellite,'OpenStreetMap':osm}},
  {{'Candidate patches':candidateGroup}},
  {{position:'topright'}}
).addTo(map);

if(kmlLayer) map.fitBounds(kmlLayer.getBounds().pad(1.2));
else if({len(alert_feats)+len(candidate_feats)}>0) map.setZoom({zoom});
</script>
</body>
</html>"""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import glob as _glob

    p = argparse.ArgumentParser(
        description="E-Netra Alert Overlay Map Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--alerts",  nargs="+", required=True, help="Alert GeoJSON file(s) or glob pattern")
    p.add_argument("--kml",     default=None,             help="Ground-truth KML polygon file")
    p.add_argument("--beat",    default="",               help="Beat name for title")
    p.add_argument("--range",   default="",               help="Range name (e.g. North_Guna)")
    p.add_argument("--out",     required=True,            help="Output HTML file path")
    p.add_argument("--zoom",    type=int, default=15,     help="Initial map zoom (default 15)")
    args = p.parse_args()

    # Expand globs
    geojsons = []
    for pat in args.alerts:
        expanded = _glob.glob(pat)
        geojsons.extend(expanded if expanded else [pat])

    out = build_overlay_map(
        alert_geojsons = geojsons,
        kml_path       = args.kml,
        beat_name      = args.beat,
        range_name     = args.range,
        out_path       = args.out,
        zoom           = args.zoom,
    )
    print(f"✅  Written: {out}")
