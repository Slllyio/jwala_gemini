"""
All-in-one patch for ews_dashboard.py:
  1. Composite score (not fixed 0.65)
  2. KML download of ALERT POLYGON from table rows
  3. Resizable panels (drag dividers between sidebar/map/right)
"""
from pathlib import Path

SRC = Path(__file__).parent / "ews_dashboard.py"
txt = SRC.read_text(encoding="utf-8")
ok = []

# ─── PATCH A: Composite score in _load_alerts ──────────────────────────────
# Replace the line that reads confidence as score with a computed composite
old_score = '        score = float(p.get("score") or p.get("confidence") or p.get("adj") or 0.0)'
new_score = (
    '        # Composite score: zscore magnitude + cusum + area (avoids fixed 0.65)\n'
    '        _conf    = float(p.get("confidence") or 0.65)\n'
    '        _z       = abs(float(p.get("dw_trees_zscore") or 0.0))\n'
    '        _cusum   = float(p.get("cusum_score") or 0.0)\n'
    '        _area    = float(p.get("area_ha") or 0.0)\n'
    '        score = round(min(0.99, max(0.30,\n'
    '            _conf * 0.30 +\n'
    '            (_z - 1.5) * 0.09 +\n'
    '            _cusum * 0.25 +\n'
    '            min(_area, 3.0) * 0.04\n'
    '        )), 2)'
)
if old_score in txt:
    txt = txt.replace(old_score, new_score)
    ok.append("A-score")
else:
    print("MISS A: score")

# ─── PATCH B: Add KML download button to table rows ────────────────────────
old_rpt = (
    '        rpt_btn = (f\'<button class="rpt-btn" onclick="openReport({aid})" \'\n'
    '                   f\'title="Field report">&#128424;</button>\') if a["label"] == "HIGH" else ""\n'
    '        rows_html += f"""\n'
    '<tr data-beat="{a[\'beat\']}" data-range="{a[\'range\']}" data-label="{a[\'label\']}">\n'
    '  <td><span class="badge" style="background:{a[\'color\']}">{a[\'label\']}</span></td>\n'
    '  <td><b>{a[\'score\']:.3f}</b></td>\n'
    '  <td>{a[\'beat\']}</td>\n'
    '  <td>{a[\'range\']}</td>\n'
    '  <td>{a[\'date\']}</td>\n'
    '  <td>{a[\'area_ha\']:.2f}</td>\n'
    '  <td>{a[\'typology\']}</td>\n'
    '  <td>{a[\'dNDVI\']:+.3f}</td>\n'
    '  <td>{a[\'dNBR\']:+.3f}</td>\n'
    '  <td>{a[\'dw_trees_delta\']:+.3f}</td>\n'
    '  <td>{a[\'zscore\']:+.2f}\u03c3</td>\n'
    '  <td>{a[\'cloud_frac\']:.0%}</td>\n'
    '  <td>{rpt_btn}</td>\n'
    '</tr>"""'
)
new_rpt = (
    '        rpt_btn = (f\'<button class="rpt-btn" onclick="openReport({aid})" \'\n'
    '                   f\'title="Field report">&#128424;</button>\') if a["label"] == "HIGH" else ""\n'
    '        kml_btn = f\'<button class="rpt-btn kml-btn" onclick="dlAlertKml({aid})" title="Download alert KML">&#x2913; KML</button>\'\n'
    '        rows_html += f"""\n'
    '<tr data-beat="{a[\'beat\']}" data-range="{a[\'range\']}" data-label="{a[\'label\']}">\n'
    '  <td><span class="badge" style="background:{a[\'color\']}">{a[\'label\']}</span></td>\n'
    '  <td><b>{a[\'score\']:.3f}</b></td>\n'
    '  <td>{a[\'beat\']}</td>\n'
    '  <td>{a[\'range\']}</td>\n'
    '  <td>{a[\'date\']}</td>\n'
    '  <td>{a[\'area_ha\']:.2f}</td>\n'
    '  <td>{a[\'dNDVI\']:+.3f}</td>\n'
    '  <td>{a[\'dNBR\']:+.3f}</td>\n'
    '  <td>{a[\'dw_trees_delta\']:+.3f}</td>\n'
    '  <td>{a[\'zscore\']:+.2f}\u03c3</td>\n'
    '  <td>{a[\'cloud_frac\']:.0%}</td>\n'
    '  <td style="white-space:nowrap">{kml_btn}{rpt_btn}</td>\n'
    '</tr>"""'
)
if old_rpt in txt:
    txt = txt.replace(old_rpt, new_rpt)
    ok.append("B-kml-btn")
else:
    print("MISS B: kml row btn")

# ─── PATCH C: kml-btn CSS ──────────────────────────────────────────────────
old_css = ".rpt-btn:hover{{background:#1e3a5f}}"
new_css = (
    ".rpt-btn:hover{{background:#1e3a5f}}\n"
    ".kml-btn{{border-color:#34d399!important;color:#34d399!important;margin-right:3px}}\n"
    ".kml-btn:hover{{background:#052e16!important}}"
)
if old_css in txt:
    txt = txt.replace(old_css, new_css)
    ok.append("C-kml-css")
else:
    print("MISS C: kml css")

# ─── PATCH D: Resize handle CSS ────────────────────────────────────────────
old_body_css = "/* ── sidebar ── */\n.sidebar{{width:220px;flex-shrink:0;"
new_body_css = (
    "/* ── resize handles ── */\n"
    ".resize-handle{{width:5px;background:transparent;cursor:col-resize;\n"
    "                flex-shrink:0;transition:background .2s;z-index:10;}}\n"
    ".resize-handle:hover,.resize-handle.dragging{{background:#1e3a5f}}\n"
    "\n"
    "/* ── sidebar ── */\n"
    ".sidebar{{width:220px;flex-shrink:0;"
)
if old_body_css in txt:
    txt = txt.replace(old_body_css, new_body_css)
    ok.append("D-resize-css")
else:
    print("MISS D: resize css")

# ─── PATCH E: Add resize handles to body HTML ──────────────────────────────
old_body_html = "<!-- ── sidebar ── -->\n<div class=\"sidebar\">"
new_body_html = (
    "<!-- ── sidebar ── -->\n"
    "<div class=\"sidebar\" id=\"sidebar\">"
)
if old_body_html in txt:
    txt = txt.replace(old_body_html, new_body_html)
    ok.append("E-sidebar-id")
else:
    print("MISS E: sidebar id")

old_close_sb = "</div>\n\n<!-- ── map ── -->\n<div class=\"map-panel\" style=\"position:relative\">"
new_close_sb = (
    "</div>\n"
    '<div class="resize-handle" id="rh-left" title="Drag to resize"></div>\n'
    "\n<!-- ── map ── -->\n<div class=\"map-panel\" style=\"position:relative\">"
)
if old_close_sb in txt:
    txt = txt.replace(old_close_sb, new_close_sb)
    ok.append("E2-rh-left")
else:
    print("MISS E2: rh-left")

old_map_right = "</div>\n\n<!-- ── right panel ── -->\n<div class=\"right\">"
new_map_right = (
    "</div>\n"
    '<div class="resize-handle" id="rh-right" title="Drag to resize"></div>\n'
    "\n<!-- ── right panel ── -->\n<div class=\"right\" id=\"rightPanel\">"
)
if old_map_right in txt:
    txt = txt.replace(old_map_right, new_map_right)
    ok.append("E3-rh-right")
else:
    print("MISS E3: rh-right")

# ─── PATCH F: Add dlAlertKml JS + resizer JS (inject before </script>) ────
kml_js = r"""
// ─── Alert KML download ───────────────────────────────────────────────────
function dlAlertKml(alertId) {
  // Find feature in FC whose properties.id === alertId
  const feat = FC.features.find(f => f.properties && f.properties.id === alertId);
  if(!feat || !feat.geometry) {
    // fallback: try by index
    const byIdx = FC.features[alertId];
    if(!byIdx || !byIdx.geometry) {
      alert('No geometry for alert ' + alertId);
      return;
    }
    return _buildAlertKml(alertId, byIdx);
  }
  _buildAlertKml(alertId, feat);
}
function _buildAlertKml(alertId, feat) {
  const p   = feat.properties || {};
  const al  = ALERTS.find(a => a.id === alertId) || p;
  const beat = al.beat || p.beat_name || '—';
  const lbl  = al.label || p.label || 'ALERT';
  const sc   = al.score !== undefined ? al.score.toFixed(3) : '—';
  const dt   = al.date  || '—';
  const ha   = al.area_ha !== undefined ? Number(al.area_ha).toFixed(2) : '—';
  const dndvi = al.dNDVI !== undefined ? Number(al.dNDVI).toFixed(4) : '—';
  const dnbr  = al.dNBR  !== undefined ? Number(al.dNBR).toFixed(4)  : '—';
  const dtree = al.dw_trees_delta !== undefined ? Number(al.dw_trees_delta).toFixed(4) : '—';
  const z     = al.zscore !== undefined ? Number(al.zscore).toFixed(2) : '—';

  function ringCoords(ring) {
    return ring.map(c => c[0].toFixed(6)+','+c[1].toFixed(6)+',0').join('\n          ');
  }
  function polyKml(geom) {
    if(geom.type === 'Polygon') {
      return `<Polygon>
        <outerBoundaryIs><LinearRing><coordinates>
          ${ringCoords(geom.coordinates[0])}
        </coordinates></LinearRing></outerBoundaryIs>
      </Polygon>`;
    } else if(geom.type === 'MultiPolygon') {
      return '<MultiGeometry>' + geom.coordinates.map(poly =>
        `<Polygon><outerBoundaryIs><LinearRing><coordinates>
          ${ringCoords(poly[0])}
        </coordinates></LinearRing></outerBoundaryIs></Polygon>`
      ).join('') + '</MultiGeometry>';
    }
    return '';
  }

  const colorHex = lbl==='HIGH' ? 'FF2020FF'
                  : lbl==='MEDIUM' ? 'FF0080FF' : 'FF00FFFF';
  const kml = `<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>E-Netra Alert — ${beat} — ${dt}</name>
  <Style id="alertStyle">
    <LineStyle><color>${colorHex}</color><width>3</width></LineStyle>
    <PolyStyle><color>4400AAFF</color><fill>1</fill></PolyStyle>
  </Style>
  <Placemark>
    <name>${lbl} Alert — ${beat}</name>
    <description><![CDATA[
      <b>Beat:</b> ${beat}<br/>
      <b>Date:</b> ${dt}<br/>
      <b>Score:</b> ${sc}<br/>
      <b>Area:</b> ${ha} ha<br/>
      <b>&Delta;NDVI:</b> ${dndvi}<br/>
      <b>&Delta;NBR:</b>  ${dnbr}<br/>
      <b>&Delta;Trees:</b>${dtree}<br/>
      <b>Z-score:</b> ${z}&sigma;
    ]]></description>
    <styleUrl>#alertStyle</styleUrl>
    ${polyKml(feat.geometry)}
  </Placemark>
</Document>
</kml>`;

  const blob = new Blob([kml], {type:'application/vnd.google-earth.kml+xml'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `alert_${beat.replace(/ /g,'_')}_${dt}.kml`;
  a.click();
}

// ─── Resizable panels ─────────────────────────────────────────────────────
(function() {
  function initResize(handleId, leftEl, rightEl, isRight) {
    const handle = document.getElementById(handleId);
    if(!handle) return;
    handle.addEventListener('mousedown', function(e) {
      e.preventDefault();
      handle.classList.add('dragging');
      const startX  = e.clientX;
      const startW  = isRight
        ? rightEl.getBoundingClientRect().width
        : leftEl.getBoundingClientRect().width;
      function onMove(ev) {
        const dx = ev.clientX - startX;
        if(isRight) {
          const nw = Math.max(260, Math.min(700, startW - dx));
          rightEl.style.width = nw + 'px';
          rightEl.style.flexShrink = '0';
        } else {
          const nw = Math.max(140, Math.min(380, startW + dx));
          leftEl.style.width = nw + 'px';
          leftEl.style.flexShrink = '0';
        }
      }
      function onUp() {
        handle.classList.remove('dragging');
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });
  }
  const sidebar   = document.getElementById('sidebar');
  const rightPanel= document.getElementById('rightPanel');
  initResize('rh-left',  sidebar, null,       false);
  initResize('rh-right', null,    rightPanel,  true);
})();
"""

old_close_script = "</script>\n</body></html>"
new_close_script = kml_js + "\n</script>\n</body></html>"
if old_close_script in txt:
    txt = txt.replace(old_close_script, new_close_script)
    ok.append("F-js")
else:
    print("MISS F: js inject")

SRC.write_text(txt, encoding="utf-8")
print("\nPatches:", ok)
print("DONE")
