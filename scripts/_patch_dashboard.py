"""
One-shot patch for ews_dashboard.py:
  1. Date extracted from filename (YYYY-MM-DD)
  2. Additional change metrics: dNDVI, dNBR, dVH, cusum_score, count
  3. Beat info panel HTML injected into map-panel div
  4. Table header updated to show new columns
  5. KML download button in openBeatPanel JS
  6. Table row updated to show new columns
"""
import re
from pathlib import Path

SRC = Path(__file__).parent / "ews_dashboard.py"
src = SRC.read_text(encoding="utf-8")
patches_applied = []

# ─── PATCH 1: stamp _src_file on features during load ─────────────────────────
p1_old = '                fc = json.loads(fp.read_text(encoding="utf-8"))\n                feats.extend(fc.get("features", []))'
p1_new = ('                fc = json.loads(fp.read_text(encoding="utf-8"))\n'
          '                for _f in fc.get("features", []):\n'
          '                    _f["_src_file"] = fp.name\n'
          '                    feats.append(_f)')
if p1_old in src:
    src = src.replace(p1_old, p1_new)
    patches_applied.append("1-stamp-filename")
else:
    print("MISS 1: stamp filename — already patched?")

# ─── PATCH 2: extend metric variables in loop ─────────────────────────────────
p2_old = ('        delta = float(p.get("dw_trees_delta") or 0.0)\n'
          '        zscore= float(p.get("dw_trees_zscore") or 0.0)')
p2_new = ('        delta  = float(p.get("dw_trees_delta") or 0.0)\n'
          '        zscore = float(p.get("dw_trees_zscore") or 0.0)\n'
          '        dndvi  = float(p.get("dNDVI") or 0.0)\n'
          '        dnbr   = float(p.get("dNBR") or 0.0)\n'
          '        dvh    = float(p.get("dVH") or 0.0)\n'
          '        cusum  = float(p.get("cusum_score") or 0.0)\n'
          '        count  = int(p.get("count") or 0)')
if p2_old in src:
    src = src.replace(p2_old, p2_new)
    patches_applied.append("2-metrics")
else:
    print("MISS 2: metrics")

# ─── PATCH 3: date from filename ──────────────────────────────────────────────
p3_old = '        date  = p.get("detection_date") or p.get("date") or "—"'
p3_new = ('        # Date from property or parse YYYY-MM-DD from filename\n'
          '        date = p.get("detection_date") or p.get("date") or None\n'
          '        if not date:\n'
          '            _src = feat.get("_src_file", "")\n'
          '            _m = re.search(r"(\\d{4}-\\d{2}-\\d{2})", str(_src))\n'
          '            date = _m.group(1) if _m else "—"')
if p3_old in src:
    src = src.replace(p3_old, p3_new)
    patches_applied.append("3-date")
else:
    print("MISS 3: date")

# ─── PATCH 4: add new fields to out.append ────────────────────────────────────
p4_old = ('            "cloud_frac": cloud, "dw_trees_delta": delta, "zscore": zscore,\n'
          '            "color": _score_colour(score), "lat": clat, "lng": clng,')
p4_new = ('            "cloud_frac": cloud, "dw_trees_delta": delta, "zscore": zscore,\n'
          '            "dNDVI": dndvi, "dNBR": dnbr, "dVH": dvh,\n'
          '            "cusum_score": cusum, "count": count,\n'
          '            "color": _score_colour(score), "lat": clat, "lng": clng,')
if p4_old in src:
    src = src.replace(p4_old, p4_new)
    patches_applied.append("4-append")
else:
    print("MISS 4: append")

# ─── PATCH 5: map-panel get position:relative ─────────────────────────────────
p5_old = '<div class="map-panel">\n  <div id="map"></div>'
p5_new = ('<div class="map-panel" style="position:relative">\n'
          '  <div id="map"></div>\n'
          '  <!-- Beat info panel: fixed bottom-left, slides up on beat click -->\n'
          '  <div id="beat-info-panel">\n'
          '    <div class="bip-accent" id="bip-accent"></div>\n'
          '    <button class="bip-close" onclick="closeBeatPanel()" title="Close">&#10005;</button>\n'
          '    <div class="bip-body">\n'
          '      <div class="bip-beat" id="bip-beat"></div>\n'
          '      <div class="bip-range" id="bip-range"></div>\n'
          '      <div class="bip-rows" id="bip-rows"></div>\n'
          '    </div>\n'
          '  </div>')
if p5_old in src:
    src = src.replace(p5_old, p5_new, 1)
    patches_applied.append("5-panel-html")
else:
    print("MISS 5: panel html")

# ─── PATCH 6: table header — replace columns ──────────────────────────────────
# Find the existing th row and replace
p6_pat = re.compile(
    r'<th>Tier</th><th>Score</th><th>Beat</th><th>Range</th>\s*'
    r'<th>Date</th><th>ha</th><th>[^<]*</th><th>[^<]*</th><th>[^<]*</th><th>[^<]*</th><th>[^<]*</th><th></th>'
)
p6_new = ('<th>Tier</th><th>Score</th><th>Beat</th><th>Range</th>\n'
          '        <th>Date</th><th>ha</th><th>\u0394NDVI</th><th>\u0394NBR</th>'
          '<th>\u0394Trees</th><th>Z</th><th>\u2601</th><th></th>')
m6 = p6_pat.search(src)
if m6:
    src = src[:m6.start()] + p6_new + src[m6.end():]
    patches_applied.append("6-th-header")
else:
    print("MISS 6: table header")

# ─── PATCH 7: table row — add dNDVI, dNBR, KML column ───────────────────────
p7_old = ('  <td>{a[\'typology\']}</td>\n'
          '  <td>{a[\'dw_trees_delta\']:+.3f}</td>\n'
          '  <td>{a[\'zscore\']:+.2f}\u03c3</td>\n'
          '  <td>{a[\'cloud_frac\']:.0%}</td>\n'
          '  <td>{rpt_btn}</td>')
p7_new = ('  <td>{a[\'typology\']}</td>\n'
          "  <td>{a['dNDVI']:+.3f}</td>\n"
          "  <td>{a['dNBR']:+.3f}</td>\n"
          "  <td>{a['dw_trees_delta']:+.3f}</td>\n"
          "  <td>{a['zscore']:+.2f}\u03c3</td>\n"
          '  <td>{a[\'cloud_frac\']:.0%}</td>\n'
          '  <td>{rpt_btn}</td>')
if p7_old in src:
    src = src.replace(p7_old, p7_new)
    patches_applied.append("7-trow")
else:
    print("MISS 7: table row")

# ─── PATCH 8: openBeatPanel JS — add KML download ────────────────────────────
p8_old = "  document.getElementById('beat-info-panel').classList.add('open');\n}}\n// close on map click\nmap.on('click', closeBeatPanel);"
p8_new = ("  // KML download button\n"
          "  const geom = FC.features.find(f => {\n"
          "    const fb = f.properties && (f.properties.beat || f.properties.BEAT || '');\n"
          "    return fb === beat;\n"
          "  });\n"
          "  const kmlBtn = document.getElementById('bip-kml-btn');\n"
          "  if(kmlBtn) kmlBtn.onclick = function() {{ downloadBeatKml(props); }};\n"
          "\n"
          "  document.getElementById('beat-info-panel').classList.add('open');\n"
          "}}\n"
          "// close on map click\n"
          "map.on('click', closeBeatPanel);")
if p8_old in src:
    src = src.replace(p8_old, p8_new)
    patches_applied.append("8-kml-btn-click")
else:
    print("MISS 8: kml btn click")

# ─── PATCH 9: add KML download function + KML button HTML in bip-rows ─────────
p9_old = ("  document.getElementById('bip-rows').innerHTML = rows.map(([l,v]) =>\n"
          "    `<div class=\"bip-row\">\n"
          "       <span class=\"bip-lbl\">${{l}}</span>\n"
          "       <span class=\"bip-val\">${{v}}</span>\n"
          "     </div>`\n"
          "  ).join('');")
p9_new = ("  document.getElementById('bip-rows').innerHTML = rows.map(([l,v]) =>\n"
          "    `<div class=\"bip-row\">\n"
          "       <span class=\"bip-lbl\">${{l}}</span>\n"
          "       <span class=\"bip-val\">${{v}}</span>\n"
          "     </div>`\n"
          "  ).join('') +\n"
          "  `<div class=\"bip-row\" style=\"justify-content:center;padding-top:10px\">\n"
          "     <button id=\"bip-kml-btn\" onclick=\"downloadBeatKml(arguments[0])\" \n"
          "       style=\"background:${{col}}22;border:1px solid ${{col}};color:${{col}};\n"
          "              padding:6px 18px;border-radius:8px;cursor:pointer;\n"
          "              font-size:11px;font-weight:700;letter-spacing:.5px;\n"
          "              transition:background .15s\"\n"
          "       onmouseover=\"this.style.background='${{col}}44'\" \n"
          "       onmouseout=\"this.style.background='${{col}}22'\">\n"
          "       &#x2913; Download KML\n"
          "     </button>\n"
          "   </div>`;")
if p9_old in src:
    src = src.replace(p9_old, p9_new)
    patches_applied.append("9-kml-html")
else:
    print("MISS 9: kml html")

# ─── PATCH 10: add downloadBeatKml function ───────────────────────────────────
p10_marker = "// close on map click\nmap.on('click', closeBeatPanel);"
p10_insert = (
    "// ─── KML download for a beat polygon ────────────────────────────────────────\n"
    "function downloadBeatKml(props) {{\n"
    "  const beatName = props.BEAT || props.Beat || 'beat';\n"
    "  const rangeName = props.RANGE || props.Range || '';\n"
    "  // find the matching feature in BEATS_FC\n"
    "  const feat = BEATS_FC.features.find(f => {{\n"
    "    const p = f.properties || {{}};\n"
    "    return (p.BEAT || p.Beat || '').toUpperCase() === beatName.toUpperCase();\n"
    "  }});\n"
    "  if(!feat) {{ alert('No geometry found for ' + beatName); return; }}\n"
    "  // Convert GeoJSON MultiPolygon -> KML Placemark\n"
    "  function ringToKml(coords) {{\n"
    "    return '<coordinates>' +\n"
    "      coords.map(c => c[0].toFixed(6)+','+c[1].toFixed(6)+',0').join(' ') +\n"
    "      '</coordinates>';\n"
    "  }}\n"
    "  const geom = feat.geometry;\n"
    "  let polyKml = '';\n"
    "  if(geom.type === 'Polygon') {{\n"
    "    polyKml = `<Polygon><outerBoundaryIs><LinearRing>${{ringToKml(geom.coordinates[0])}}</LinearRing></outerBoundaryIs></Polygon>`;\n"
    "  }} else if(geom.type === 'MultiPolygon') {{\n"
    "    polyKml = '<MultiGeometry>' +\n"
    "      geom.coordinates.map(poly =>\n"
    "        `<Polygon><outerBoundaryIs><LinearRing>${{ringToKml(poly[0])}}</LinearRing></outerBoundaryIs></Polygon>`\n"
    "      ).join('') + '</MultiGeometry>';\n"
    "  }}\n"
    "  const p = feat.properties || {{}};\n"
    "  const kml = `<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n"
    "<Document>\n"
    "<name>${{beatName}}</name>\n"
    "<Placemark>\n"
    "  <name>${{beatName}}</name>\n"
    "  <description><![CDATA[\n"
    "    Range: ${{rangeName}}<br/>\n"
    "    Sub-Range: ${{p.SUB_RANGE||'—'}}<br/>\n"
    "    Beat No: ${{p.NEW_No_||'—'}}<br/>\n"
    "    Area (2015): ${{p.area2015||'—'}} ha<br/>\n"
    "    Area (Origin): ${{p.areaOrigin||'—'}} ha\n"
    "  ]]></description>\n"
    "  <Style><LineStyle><color>FF38bdf8</color><width>3</width></LineStyle>\n"
    "         <PolyStyle><color>2238bdf8</color></PolyStyle></Style>\n"
    "  ${{polyKml}}\n"
    "</Placemark>\n"
    "</Document>\n"
    "</kml>`;\n"
    "  const blob = new Blob([kml], {{type:'application/vnd.google-earth.kml+xml'}});\n"
    "  const a = document.createElement('a');\n"
    "  a.href = URL.createObjectURL(blob);\n"
    "  a.download = beatName.replace(/\\s+/g,'_') + '.kml';\n"
    "  a.click();\n"
    "}}\n\n"
    + p10_marker
)
if p10_marker in src:
    src = src.replace(p10_marker, p10_insert, 1)
    patches_applied.append("10-downloadKml-fn")
else:
    print("MISS 10: downloadKml fn")

SRC.write_text(src, encoding="utf-8")
print("\nPatches applied:", patches_applied)
print("DONE")
