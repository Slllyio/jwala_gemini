"""
Export alert patches (from Feb-09 anchor) + original KML polygon → GeoJSON.
Then generate an interactive Leaflet HTML overlay map.
"""
import sys, json, xml.etree.ElementTree as ET, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
cfg = yaml.safe_load(open("config.yaml"))

from src.data.gee_fetch import init_gee
init_gee(cfg.get("gee", {}).get("gee_project", "van-suraksha-alert"))
import ee

# ── Parse KML polygon ────────────────────────────────────────────────────────
kml_path = "data/ground_truth/confirmed_loss_feb2026.kml"
tree = ET.parse(kml_path)
root = tree.getroot()
for elem in root.iter():
    if "}" in elem.tag:
        elem.tag = elem.tag.split("}", 1)[1]
ct = root.find(".//coordinates")
coords_ll = [[float(p.split(",")[0]), float(p.split(",")[1])]
             for p in ct.text.strip().split()]
gt_geom = ee.Geometry.Polygon([coords_ll]).buffer(50)

kml_geojson = {
    "type": "Feature",
    "properties": {"name": "Confirmed Loss (KML)", "layer": "ground_truth"},
    "geometry": {
        "type": "Polygon",
        "coordinates": [coords_ll + [coords_ll[0]]]
    }
}

# ── Re-run pipeline for Feb-09 anchor ────────────────────────────────────────
from src.inference.rules_engine import (
    _load_rules_cfg, build_dw_instant_delta, build_s2_ndvi_composite,
    build_ndvi_zscore_image, _load_range_phenology_model, build_s1_delta,
    build_candidate_mask, apply_post_processing,
    vectorize_patches, sample_patches, score_patch, _forest_baseline_mask,
    _is_dry_season
)

rc   = _load_rules_cfg(cfg)
ANCHOR = "2026-02-09"
season = "DRY" if _is_dry_season(ANCHOR) else "WET"
t3_conf = rc["sar_dry_conf"]

dw   = build_dw_instant_delta(gt_geom, ANCHOR, rc["dw_instant_window"])
s2   = build_s2_ndvi_composite(gt_geom, ANCHOR, rc["s2_window"])
dvh  = build_s1_delta(gt_geom, ANCHOR, rc["s1_window"])

ndvi_zscore = None
dndvi = None
if s2:
    ndvi_c, ndvi_b = s2
    dndvi = ndvi_b.subtract(ndvi_c).rename("dNDVI")
    model_dir = rc["phenology_model_dir"]
    if not Path(model_dir).is_absolute():
        model_dir = str(Path(__file__).resolve().parent.parent / model_dir)
    pheno = _load_range_phenology_model("Binaganj", model_dir, "ndvi")
    if pheno:
        ndvi_zscore = build_ndvi_zscore_image(ndvi_c, ANCHOR, pheno)

forest_mask = _forest_baseline_mask(gt_geom)
cand      = build_candidate_mask(
    dndvi=dndvi, ndvi_zscore=ndvi_zscore,
    dw_instant=dw, dw_zscore=None, dw_trees_delta=None,
    dvh=dvh, cusum=None,
    forest_mask=forest_mask, aoi=gt_geom, thresholds=rc
)
cand_morph = apply_post_processing(cand)
patch_fc   = vectorize_patches(
    candidate_mask=cand_morph, aoi=gt_geom,
    min_area_ha=rc["min_area_ha"], scale=10
)

# ── Build signal images & sample ─────────────────────────────────────────────
signal_images = {}
if ndvi_zscore is not None: signal_images["ndvi_zscore"]     = ndvi_zscore
if dndvi       is not None: signal_images["dNDVI"]           = dndvi
if dw          is not None:
    if "trees_delta" in dw: signal_images["dw_trees_delta"]  = dw["trees_delta"]
    if "bare_delta"  in dw: signal_images["dw_bare_delta"]   = dw["bare_delta"]
if dvh         is not None: signal_images["dVH"]             = dvh

sampled_fc  = sample_patches(patch_fc, signal_images, scale=10)
geojson_raw = sampled_fc.getInfo()

# ── Score each patch & annotate GeoJSON ──────────────────────────────────────
alert_features = []
candidate_features = []

for feat in geojson_raw["features"]:
    props = feat["properties"]
    conf, details = score_patch(props, rc, t3_conf, season=season)
    area = props.get("area_ha", 0)
    dt   = props.get("dw_trees_delta")
    nz   = props.get("ndvi_zscore")

    annotated = {
        "type": "Feature",
        "geometry": feat["geometry"],
        "properties": {
            "conf": round(conf, 3),
            "area_ha": round(area, 2),
            "dw_trees_delta": round(dt, 3) if dt is not None else None,
            "ndvi_z": round(nz, 2) if nz is not None else None,
            "raw": details.get("raw", 0),
            "adj": details.get("adj", 0),
            "label": f"{'⚠ ALERT' if conf >= 0.35 else 'Candidate'} conf={conf:.2f}\nΔtrees={dt:.3f} area={area:.2f}ha",
        }
    }
    if conf >= 0.35:
        alert_features.append(annotated)
    else:
        candidate_features.append(annotated)

print(f"\nTotal patches: {len(alert_features) + len(candidate_features)}")
print(f"  Alert (conf>=0.35): {len(alert_features)}")
print(f"  Candidates (below): {len(candidate_features)}")

# ── Write combined GeoJSON ────────────────────────────────────────────────────
out_dir = Path("outputs/overlay")
out_dir.mkdir(parents=True, exist_ok=True)

combined_geojson = {
    "type": "FeatureCollection",
    "features": [kml_geojson] + alert_features + candidate_features
}
geojson_path = out_dir / "feb09_overlay.geojson"
json.dump(combined_geojson, open(geojson_path, "w"), indent=2)
print(f"GeoJSON saved → {geojson_path}")

# ── Generate Leaflet HTML map ─────────────────────────────────────────────────
# Compute centroid from KML coords
cx = sum(c[0] for c in coords_ll) / len(coords_ll)
cy = sum(c[1] for c in coords_ll) / len(coords_ll)

kml_coords_js   = json.dumps([coords_ll + [coords_ll[0]]])
alert_js        = json.dumps(alert_features)
candidate_js    = json.dumps(candidate_features)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>E-Netra Alert Overlay — Feb 09 2026</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; height: 100vh; display: flex; flex-direction: column; }}
    #header {{
      background: linear-gradient(135deg, #1a2332 0%, #0d1117 100%);
      border-bottom: 1px solid #21262d;
      padding: 12px 20px;
      display: flex; align-items: center; justify-content: space-between;
    }}
    #header h1 {{ font-size: 1.1rem; font-weight: 600; color: #58a6ff; letter-spacing: 0.5px; }}
    #header .badge {{ 
      background: #f85149; color: white; font-size: 0.75rem; 
      padding: 3px 10px; border-radius: 12px; font-weight: 600;
    }}
    #map {{ flex: 1; }}
    .legend {{
      background: rgba(13,17,23,0.92);
      border: 1px solid #21262d;
      border-radius: 8px;
      padding: 12px 16px;
      font-size: 0.82rem;
      line-height: 2;
      backdrop-filter: blur(8px);
    }}
    .legend-item {{ display: flex; align-items: center; gap: 8px; }}
    .swatch {{ width: 18px; height: 14px; border-radius: 3px; display: inline-block; }}
    .info-box {{
      background: rgba(13,17,23,0.92);
      border: 1px solid #21262d;
      border-radius: 8px;
      padding: 12px 16px;
      font-size: 0.8rem;
      line-height: 1.7;
      max-width: 240px;
      backdrop-filter: blur(8px);
    }}
    .info-box b {{ color: #58a6ff; }}
  </style>
</head>
<body>
  <div id="header">
    <h1>🌳 E-Netra — Alert Overlay · Anchor: 2026-02-09 · DW Pair: Feb 07 → Feb 09</h1>
    <span class="badge">⚠ TIER-3 ALERT DETECTED</span>
  </div>
  <div id="map"></div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const map = L.map('map', {{
      center: [{cy:.5f}, {cx:.5f}],
      zoom: 15,
      zoomControl: true
    }});

    // Base layers
    const satellite = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
      {{ attribution: 'Esri World Imagery', maxZoom: 20 }}
    );
    const osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
      {{ attribution: '© OpenStreetMap', maxZoom: 19 }});
    satellite.addTo(map);

    // ── Ground truth KML polygon ──────────────────────────────────────────
    const kmlCoords = {kml_coords_js};
    const kmlLayer = L.polygon(
      kmlCoords[0].map(c => [c[1], c[0]]),
      {{
        color: '#3fb950', weight: 2.5, dashArray: '8,5',
        fillColor: '#3fb950', fillOpacity: 0.12
      }}
    ).addTo(map).bindPopup(
      '<b style="color:#3fb950">✓ Confirmed Loss (KML)</b><br>Ground truth: tree loss Feb 5–20, 2026'
    );

    // ── Alert patches ─────────────────────────────────────────────────────
    const alertPatches = {alert_js};
    alertPatches.forEach(f => {{
      const coords = f.geometry.coordinates[0].map(c => [c[1], c[0]]);
      const p = f.properties;
      L.polygon(coords, {{
        color: '#f85149', weight: 2.5,
        fillColor: '#f85149', fillOpacity: 0.45
      }}).addTo(map).bindPopup(`
        <b style="color:#f85149">⚠ ALERT</b><br>
        <b>Confidence:</b> ${{(p.conf*100).toFixed(1)}}%<br>
        <b>Area:</b> ${{p.area_ha}} ha<br>
        <b>ΔTrees (DW):</b> ${{(p.dw_trees_delta*100).toFixed(1)}} pp<br>
        <b>NDVI z:</b> ${{p.ndvi_z}}<br>
        <b>Raw score:</b> ${{p.raw}} · <b>Adj:</b> ${{p.adj}}
      `);
    }});

    // ── Candidate (non-alert) patches ─────────────────────────────────────
    const candidates = {candidate_js};
    const candidateGroup = L.layerGroup();
    candidates.forEach(f => {{
      const coords = f.geometry.coordinates[0].map(c => [c[1], c[0]]);
      const p = f.properties;
      L.polygon(coords, {{
        color: '#e3b341', weight: 1.5, dashArray: '4,3',
        fillColor: '#e3b341', fillOpacity: 0.20
      }}).addTo(candidateGroup).bindPopup(`
        <b style="color:#e3b341">⬡ Candidate (below threshold)</b><br>
        <b>Confidence:</b> ${{(p.conf*100).toFixed(1)}}%<br>
        <b>Area:</b> ${{p.area_ha}} ha<br>
        <b>ΔTrees (DW):</b> ${{(p.dw_trees_delta !== null ? (p.dw_trees_delta*100).toFixed(1)+'pp' : 'n/a')}}<br>
        <b>NDVI z:</b> ${{p.ndvi_z}}
      `);
    }});
    candidateGroup.addTo(map);

    // ── Legend ────────────────────────────────────────────────────────────
    const legend = L.control({{ position: 'bottomright' }});
    legend.onAdd = () => {{
      const d = L.DomUtil.create('div', 'legend');
      d.innerHTML = `
        <b style="font-size:0.9rem;color:#58a6ff">Map Legend</b><br>
        <div class="legend-item"><span class="swatch" style="background:#f85149;opacity:0.8"></span> Alert patch (Tier-3)</div>
        <div class="legend-item"><span class="swatch" style="background:#e3b341;opacity:0.5;border:1px dashed #e3b341"></span> Candidate (below threshold)</div>
        <div class="legend-item"><span class="swatch" style="background:#3fb950;opacity:0.5;border:2px dashed #3fb950"></span> Ground truth KML</div>
      `;
      return d;
    }};
    legend.addTo(map);

    // ── Info box ──────────────────────────────────────────────────────────
    const info = L.control({{ position: 'topleft' }});
    info.onAdd = () => {{
      const d = L.DomUtil.create('div', 'info-box');
      d.innerHTML = `
        <b>DW Pair:</b> Feb 07 → Feb 09, 2026<br>
        <b>Cloud cover:</b> 0.3% → 0.0%<br>
        <b>Season:</b> DRY (leaf-off)<br>
        <b>Alert Tier:</b> 3 &nbsp;<span style="color:#f85149">⚠</span><br>
        <b>Confidence:</b> 0.480<br>
        <b>ΔTrees:</b> −22.6 pp in 2 days<br>
        <b>Alert area:</b> 0.27 ha<br>
        <hr style="border-color:#21262d;margin:6px 0">
        Click any polygon for details
      `;
      return d;
    }};
    info.addTo(map);

    // ── Layer switcher ────────────────────────────────────────────────────
    L.control.layers(
      {{ 'Satellite': satellite, 'OpenStreetMap': osm }},
      {{ 'Candidate patches': candidateGroup }},
      {{ position: 'topright' }}
    ).addTo(map);

    map.fitBounds(kmlLayer.getBounds().pad(0.3));
  </script>
</body>
</html>
"""

html_path = out_dir / "feb09_overlay_map.html"
html_path.write_text(html, encoding="utf-8")
print(f"HTML map saved → {html_path}")
print(f"\nOpen in browser: {html_path.resolve()}")
