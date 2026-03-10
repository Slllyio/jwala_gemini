"""
Generate E-Netra overlay map for Mar Ki Mahu — all dates
Output: outputs/marki_mahu_overlay.html

Design: score VALUE is the primary visual signal.
  - Polygon fill colour = continuous score ramp  (0→white-blue→red 1.0)
  - Score number shown as permanent polygon tooltip
  - Popup shows full breakdown (score, delta, z, cusum, area)

Run: python scripts/_build_marki_map.py
"""
import json, sys, math, re
from pathlib import Path

sys.path.insert(0, "scripts")
from _simple_dw_score import dw_multi_threat_score_maxx, RANGE_DW_MODELS, _range_baseline

RANGE     = "North_Guna"
KML       = Path(r"C:\Users\S.C.C\Downloads\Export 2026-02-23 1656.kml")
ALERT_DIR = Path("outputs/alerts")
OUT       = Path("outputs/marki_mahu_overlay.html")

# ── 1. Load KML coords ───────────────────────────────────────────────────────
raw = KML.read_text(encoding="utf-8")
coords_str = re.search(r"<coordinates>(.*?)</coordinates>", raw, re.DOTALL).group(1)
kml_ring = []
for tok in coords_str.split():
    tok = tok.strip()
    if tok:
        parts = tok.split(",")
        kml_ring.append([float(parts[0]), float(parts[1])])

# ── 2. Score all patches ─────────────────────────────────────────────────────
def reconstruct_and_score(props, date_str):
    """
    V4 FIX C — Reconstruction Trap guard.

    PREFERRED path: daemon exports actual sampled t0/t1 DW probabilities as
    `trees_before` and `trees_after` in the GeoJSON.  When present, these are
    used directly — they reflect the real pristine state of each pixel, not
    the range-level seasonal mean.

    FALLBACK path: if the daemon only exports `dw_trees_delta`, we reconstruct
    via μ+Δ (range seasonal mean as trees_before). This is the physics violation
    audited by the user: a dense Teak stand (0.85) gets hallucinated as sparse
    scrubland (0.20) if μ=0.20.  Flag this with a warning so ops knows to
    upgrade the daemon GeoJSON export (guna_ews_daemon.py outputs t0_img/t1_img
    samplers natively — just add the export fields).
    """
    mu_r, std_r = _range_baseline(RANGE, date_str)
    delta   = props.get("dw_trees_delta", 0.0)
    cr_d    = props.get("dw_crops_delta", 0.0)
    bu_d    = props.get("dw_built_delta", 0.0)
    cloud   = props.get("cloud_frac", 0.0)
    patch_z = props.get("dw_trees_zscore", 0.0)    # pipeline pixel z-score
    cusum   = props.get("cusum_score", 0.0)         # CuSuM accumulation

    # ── Trees: prefer direct sampled values over reconstruction ───────────────
    if "trees_before" in props and "trees_after" in props:
        trees_before = max(0.0, min(1.0, float(props["trees_before"])))
        trees_after  = max(0.0, min(1.0, float(props["trees_after"])))
    else:
        # FALLBACK (physics degraded): forces trees_before ≈ μ which may be
        # 4–8× lower than the actual pristine stand density.
        # TODO: upgrade daemon to export dw_trees_before / dw_trees_after.
        trees_after  = max(0.0, min(1.0, mu_r + delta))
        trees_before = max(0.0, min(1.0, mu_r))       # keep at μ, not trees_after−delta
        # Note: trees_before = trees_after − delta causes circular physics;
        # using μ directly is the correct no-information prior.

    # ── Crops / Built: same preference pattern ────────────────────────────────
    if "crops_before" in props and "crops_after" in props:
        crops_before = max(0.0, min(1.0, float(props["crops_before"])))
        crops_after  = max(0.0, min(1.0, float(props["crops_after"])))
    else:
        crops_before = max(0.0, -cr_d) if cr_d < 0 else 0.05
        crops_after  = max(0.0, crops_before + cr_d)

    if "built_before" in props and "built_after" in props:
        built_before = max(0.0, min(1.0, float(props["built_before"])))
        built_after  = max(0.0, min(1.0, float(props["built_after"])))
    else:
        built_before = max(0.0, -bu_d) if bu_d < 0 else 0.02
        built_after  = max(0.0, built_before + bu_d)

    return dw_multi_threat_score_maxx(
        trees_after=trees_after, trees_before=trees_before,
        crops_after=crops_after, crops_before=crops_before,
        built_after=built_after, built_before=built_before,
        date_str=date_str, cloud_frac=cloud, range_name=RANGE,
        patch_z=patch_z, cusum_score=cusum)

all_patches = []
for fpath in sorted(ALERT_DIR.glob("mar_ki_mahu_*.geojson")):
    date_str = fpath.stem.replace("mar_ki_mahu_", "")
    fc = json.loads(fpath.read_text())
    for feat in fc["features"]:
        props = feat["properties"]
        r = reconstruct_and_score(props, date_str)
        feat_out = json.loads(json.dumps(feat))
        feat_out["properties"]["_date"]     = date_str
        feat_out["properties"]["_score"]    = round(r["score"], 3)
        feat_out["properties"]["_label"]    = r["label"]
        feat_out["properties"]["_typology"] = r["typology"]
        feat_out["properties"]["_delta"]    = props.get("dw_trees_delta", 0.0)
        feat_out["properties"]["_area"]     = props.get("area_ha", 0.0)
        feat_out["properties"]["_cusum"]    = props.get("cusum_score", 0.0)
        feat_out["properties"]["_z"]        = props.get("dw_trees_zscore", 0.0)
        all_patches.append(feat_out)

high_count = sum(1 for f in all_patches if f["properties"]["_label"] == "HIGH")
mu_r, std_r = _range_baseline(RANGE, "2026-02-09")
r2 = RANGE_DW_MODELS[RANGE].r2 if RANGE in RANGE_DW_MODELS else 0.0

patches_js = json.dumps(all_patches, ensure_ascii=False)
kml_ring_js = json.dumps(kml_ring, ensure_ascii=False)

print(f"All patches: {len(all_patches)}  HIGH: {high_count}")

# ── 3. HTML ──────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>E-Netra — Mar Ki Mahu · Score Map</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',sans-serif; background:#0d1117; color:#e6edf3; height:100vh; display:flex; flex-direction:column; }}
  #header {{
    background:linear-gradient(135deg,#1a2332 0%,#0d1117 100%);
    border-bottom:1px solid #21262d;
    padding:10px 20px;
    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;
  }}
  #header h1 {{ font-size:1.0rem; font-weight:600; color:#58a6ff; letter-spacing:0.5px; }}
  #header .badge {{ background:#f85149; color:white; font-size:0.75rem; padding:3px 10px; border-radius:12px; font-weight:600; }}
  #map {{ flex:1; }}
  .legend {{
    background:rgba(13,17,23,0.93); border:1px solid #21262d; border-radius:8px;
    padding:12px 16px; font-size:0.8rem; line-height:1.8; backdrop-filter:blur(8px);
  }}
  .ramp-bar {{
    width:160px; height:12px; border-radius:4px;
    background:linear-gradient(to right,#1e3a5f,#2ea04380,#d2992260,#f85149);
    margin:4px 0 2px;
  }}
  .ramp-labels {{ display:flex; justify-content:space-between; font-size:0.72rem; color:#8b949e; }}
  .info-box {{
    background:rgba(13,17,23,0.93); border:1px solid #21262d; border-radius:8px;
    padding:12px 16px; font-size:0.8rem; line-height:1.8; max-width:230px;
    backdrop-filter:blur(8px);
  }}
  .info-box b {{ color:#58a6ff; }}
  .score-badge {{
    display:inline-block; font-size:1.2rem; font-weight:700;
    padding:2px 8px; border-radius:6px; font-variant-numeric:tabular-nums;
  }}
  .leaflet-tooltip {{
    background:rgba(13,17,23,0.88)!important; border:1px solid #444!important;
    color:#fff!important; font-size:0.75rem!important; font-weight:600!important;
    padding:2px 6px!important; border-radius:4px!important;
    white-space:nowrap!important; box-shadow:none!important;
  }}
  .leaflet-popup-content-wrapper {{
    background:#161b22!important; border:1px solid #30363d!important;
    color:#e6edf3!important; border-radius:10px!important;
    box-shadow:0 8px 24px rgba(0,0,0,0.6)!important;
  }}
  .leaflet-popup-tip {{ background:#161b22!important; }}
  .popup-score {{ font-size:2rem; font-weight:800; letter-spacing:-0.5px; }}
  .popup-row {{ margin:2px 0; font-size:0.82rem; }}
  .popup-row span {{ color:#8b949e; }}
</style>
</head>
<body>
<div id="header">
  <h1>🌳 E-Netra · Mar Ki Mahu · Feb 09–22, 2026 &nbsp;|&nbsp; Range: North_Guna &nbsp;|&nbsp; V3 Score Map</h1>
  <span class="badge">⚠ {high_count} HIGH · {len(all_patches)} total patches</span>
</div>
<div id="map"></div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map = L.map('map', {{ center:[24.940, 77.262], zoom:14 }});

const satellite = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{ attribution:'Esri World Imagery', maxZoom:20 }});
const osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{ attribution:'© OpenStreetMap', maxZoom:19 }});
satellite.addTo(map);

// ── Colour ramp: 0 → dark blue, 0.45 → amber, 0.65 → red, 1.0 → bright red
function scoreColor(s) {{
  if (s < 0.25) return `hsl(210,60%,${{20 + s*60}}%)`;       // blue-grey (NO ALERT)
  if (s < 0.45) return `hsl(${{155 - s*200}},60%,38%)`;       // green -> amber (LOW)
  if (s < 0.65) return `hsl(${{55 - (s-0.45)*220}},80%,45%)`;// amber -> orange (MEDIUM)
  return `hsl(${{5 + (1-s)*10}},90%,${{40 + s*15}}%)`;        // orange-red -> crimson (HIGH)
}}
function scoreOpacity(s) {{ return 0.25 + s * 0.55; }}

// ── Score badge colour in popups
function badgeStyle(s) {{
  const c = scoreColor(s);
  return `background:${{c}}22;color:${{c}};border:1.5px solid ${{c}};`;
}}

// ── KML ground-truth polygon
const kmlRing = {kml_ring_js};
const kmlLayer = L.polygon(
  kmlRing.map(c => [c[1], c[0]]),
  {{ color:'#3fb950', weight:2.5, dashArray:'8,5', fillColor:'#3fb950', fillOpacity:0.10 }}
).addTo(map).bindPopup(
  '<b style="color:#3fb950">✓ Confirmed Loss (KML)</b><br>' +
  'Ground truth: अवैध कटाई P472<br>Field recorded: Feb 23, 2026<br>Area: ~3.63 ha'
);

// ── All patches — colour by score, label by score number
const patches = {patches_js};
patches.forEach(f => {{
  const p = f.properties;
  const s = p._score;
  const geom = f.geometry;
  const ringsRaw = geom.type === 'Polygon' ? [geom.coordinates[0]] : geom.coordinates.map(r => r[0]);

  const col = scoreColor(s);
  const scoreStr = s.toFixed(3);
  const deltaStr = (p._delta * 100).toFixed(2);
  const isKml = (p._label === 'HIGH' || p._label === 'MEDIUM');

  ringsRaw.forEach(ring => {{
    const latlngs = ring.map(c => [c[1], c[0]]);
    const poly = L.polygon(latlngs, {{
      color: col,
      weight: isKml ? 2.5 : 1.5,
      fillColor: col,
      fillOpacity: scoreOpacity(s),
      dashArray: isKml ? null : '4,3'
    }}).addTo(map);

    // Permanent tooltip showing score number
    poly.bindTooltip(scoreStr, {{
      permanent: true, direction: 'center', className: 'leaflet-tooltip'
    }});

    // Click popup — score is the hero number
    poly.bindPopup(`
      <div style="font-family:'Segoe UI',sans-serif;min-width:200px">
        <div style="margin-bottom:8px">
          <span class="popup-score" style="${{badgeStyle(s)}};padding:4px 10px;border-radius:8px;display:inline-block">
            ${{scoreStr}}
          </span>
          <span style="margin-left:8px;font-size:0.85rem;color:#8b949e">${{p._label}}</span>
        </div>
        <div class="popup-row"><span>Date</span> &nbsp;${{p._date}}</div>
        <div class="popup-row"><span>ΔTrees</span> &nbsp;<b>${{deltaStr}} pp</b></div>
        <div class="popup-row"><span>z-score</span> &nbsp;<b>${{p._z.toFixed(2)}} σ</b></div>
        <div class="popup-row"><span>CuSuM</span> &nbsp;<b>${{p._cusum.toFixed(3)}}</b></div>
        <div class="popup-row"><span>Area</span> &nbsp;${{p._area.toFixed(2)}} ha</div>
        <div class="popup-row"><span>Typology</span> &nbsp;${{p._typology}}</div>
      </div>
    `);
  }});
}});

// ── Legend — score ramp
const legend = L.control({{ position:'bottomright' }});
legend.onAdd = () => {{
  const d = L.DomUtil.create('div','legend');
  d.innerHTML = `
    <b style="color:#58a6ff">V3 Score</b>
    <div class="ramp-bar"></div>
    <div class="ramp-labels"><span>0.0</span><span>0.25</span><span>0.45</span><span>0.65</span><span>1.0</span></div>
    <div style="margin-top:8px;font-size:0.75rem;color:#8b949e;line-height:1.6">
      &lt; 0.25 &nbsp;NO ALERT<br>
      0.25–0.45 &nbsp;LOW<br>
      0.45–0.65 &nbsp;MEDIUM<br>
      ≥ 0.65 &nbsp;HIGH
    </div>
    <div style="margin-top:8px;display:flex;align-items:center;gap:6px">
      <span style="width:18px;height:12px;background:#3fb950;opacity:0.6;border:2px dashed #3fb950;display:inline-block;border-radius:2px"></span>
      <span style="font-size:0.75rem">KML Ground Truth</span>
    </div>
  `;
  return d;
}};
legend.addTo(map);

// ── Info box
const info = L.control({{ position:'topleft' }});
info.onAdd = () => {{
  const d = L.DomUtil.create('div','info-box');
  d.innerHTML = `
    <b>Beat:</b> Mar Ki Mahu<br>
    <b>Range:</b> North_Guna<br>
    <b>Model R²:</b> {r2:.3f}<br>
    <b>μ (Feb 09):</b> {mu_r:.3f} &nbsp; <b>σ:</b> {std_r:.3f}<br>
    <b>HIGH patches:</b> {high_count}<br>
    <b>Total patches:</b> {len(all_patches)}<br>
    <hr style="border-color:#21262d;margin:6px 0">
    <span style="color:#8b949e;font-size:0.75rem">Click patch → full breakdown<br>Numbers = V3 score (0–1)</span>
  `;
  return d;
}};
info.addTo(map);

// ── Layer switcher
L.control.layers(
  {{ 'Satellite':satellite, 'OpenStreetMap':osm }},
  {{}},
  {{ position:'topright' }}
).addTo(map);

map.fitBounds(kmlLayer.getBounds().pad(1.8));
</script>
</body>
</html>"""

OUT.write_text(html, encoding="utf-8")
print(f"Written: {OUT}")
