"""Fix: escape all JS braces for f-string in the injected alert KML + resize JS"""
from pathlib import Path
import re

SRC = Path(__file__).parent / "ews_dashboard.py"
txt = SRC.read_text(encoding="utf-8")

# Find the injected block and remove it (everything from the dlAlertKml comment to end of resizer IIFE)
# Then replace with properly escaped version

# Locate the start marker we inserted
START = "\n// \u2500\u2500\u2500 Alert KML download"
END   = "})();\n"

idx_start = txt.find(START, txt.find("function dlAlertKml"))
# Actually search from near end of file
# The block starts with the kml comment and ends with the resizer IIFE close
idx_start = txt.rfind("// \u2500\u2500\u2500 Alert KML download")
idx_end   = txt.rfind("})();\n") + len("})();\n")

if idx_start == -1 or idx_end <= idx_start:
    print("Could not locate injection block!", idx_start, idx_end)
    exit(1)

print("Replacing block lines", txt[:idx_start].count('\n')+1, "to", txt[:idx_end].count('\n')+1)

# The properly f-string-escaped version ({{ and }} for literal JS braces)
escaped_js = """
// \u2500\u2500\u2500 Alert KML download \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
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
  const beat  = al.beat  || p.beat_name || '\u2014';
  const lbl   = al.label || p.label     || 'ALERT';
  const sc    = al.score  !== undefined ? Number(al.score).toFixed(3)  : '\u2014';
  const dt    = al.date   || '\u2014';
  const ha    = al.area_ha!== undefined ? Number(al.area_ha).toFixed(2): '\u2014';
  const dndvi = al.dNDVI  !== undefined ? Number(al.dNDVI).toFixed(4)  : '\u2014';
  const dnbr  = al.dNBR   !== undefined ? Number(al.dNBR).toFixed(4)   : '\u2014';
  const dtree = al.dw_trees_delta !== undefined ? Number(al.dw_trees_delta).toFixed(4) : '\u2014';
  const z     = al.zscore !== undefined ? Number(al.zscore).toFixed(2)  : '\u2014';

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
  const kml = '<?xml version="1.0" encoding="UTF-8"?>\\n' +
    '<kml xmlns="http://www.opengis.net/kml/2.2">\\n<Document>\\n' +
    '<name>E-Netra Alert \u2014 '+beat+' \u2014 '+dt+'</name>\\n' +
    '<Style id="s"><LineStyle><color>'+colHex+'</color><width>3</width></LineStyle>' +
    '<PolyStyle><color>4400AAFF</color><fill>1</fill></PolyStyle></Style>\\n' +
    '<Placemark><name>'+lbl+' \u2014 '+beat+'</name>\\n' +
    '<description><![CDATA[Beat: '+beat+'<br/>Date: '+dt+'<br/>Score: '+sc+'<br/>' +
    'Area: '+ha+' ha<br/>&Delta;NDVI: '+dndvi+'<br/>&Delta;NBR: '+dnbr+'<br/>' +
    '&Delta;Trees: '+dtree+'<br/>Z: '+z+'&sigma;]]></description>\\n' +
    '<styleUrl>#s</styleUrl>\\n' + polyKml(feat.geometry) + '\\n' +
    '</Placemark>\\n</Document>\\n</kml>';

  const blob = new Blob([kml], {{type:'application/vnd.google-earth.kml+xml'}});
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = 'alert_'+beat.replace(/ /g,'_')+'_'+dt+'.kml';
  a.click(); URL.revokeObjectURL(a.href);
}}

// \u2500\u2500\u2500 Resizable panels \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
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
"""

txt = txt[:idx_start] + escaped_js + txt[idx_end:]
SRC.write_text(txt, encoding="utf-8")
print("Done — block replaced, f-string braces escaped")
