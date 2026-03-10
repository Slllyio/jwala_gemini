"""
Patch: 
  1. Tier differentiation: score-based HIGH/MEDIUM/LOW (override EWS daemon fixed label)
  2. Table row click → zoom map + highlight alert polygon
"""
from pathlib import Path

SRC = Path(__file__).parent / "ews_dashboard.py"
txt = SRC.read_text(encoding="utf-8")
ok = []

# ── PATCH 1: Tier differentiation in _load_alerts ────────────────────────────
# After score is computed, derive label from score thresholds
old_label_end = '''        date  = p.get("detection_date") or p.get("date") or "—"'''
new_label_end = '''        # Score-based tier: override daemon label for better differentiation
        if score >= 0.45:
            label = "HIGH"
        elif score >= 0.35:
            label = "MEDIUM"
        else:
            label = "LOW"

        date  = p.get("detection_date") or p.get("date") or "—"'''

if old_label_end in txt:
    txt = txt.replace(old_label_end, new_label_end)
    ok.append("1-tier")
else:
    print("MISS 1: tier label")

# ── PATCH 2: _score_colour threshold update  ─────────────────────────────────
# The colour function uses score thresholds — align with new tiers
old_col = 'def _score_colour(s: float) -> str:\n    if s >= 0.65: return "#f87171"\n    if s >= 0.45: return "#fb923c"\n    if s >= 0.25: return "#facc15"\n    return "#4ade80"'
new_col = 'def _score_colour(s: float) -> str:\n    if s >= 0.45: return "#f87171"   # HIGH — red\n    if s >= 0.35: return "#fb923c"   # MEDIUM — orange\n    return "#facc15"                 # LOW — yellow'

if old_col in txt:
    txt = txt.replace(old_col, new_col)
    ok.append("2-colour")
else:
    # try alternate spacing
    old_col2 = 'def _score_colour(s: float) -> str:\n    if s >= 0.65: return "#f87171"\n    if s >= 0.40: return "#fb923c"\n    if s >= 0.20: return "#facc15"\n    return "#4ade80"'
    if old_col2 in txt:
        txt = txt.replace(old_col2, new_col)
        ok.append("2-colour-alt")
    else:
        # just find the function and replace it
        import re
        txt = re.sub(
            r'def _score_colour\(s: float\) -> str:.*?return "[^"]*"',
            'def _score_colour(s: float) -> str:\n    if s >= 0.45: return "#f87171"\n    if s >= 0.35: return "#fb923c"\n    return "#facc15"',
            txt, flags=re.S
        )
        ok.append("2-colour-regex")

# ── PATCH 3: KPI colour thresholds ───────────────────────────────────────────
# n_high / n_med / n_low now uses score-based tiers (already sorted by label)
old_kpi = '    n_high = sum(1 for a in alerts if a["label"] == "HIGH")\n    n_med  = sum(1 for a in alerts if a["label"] == "MEDIUM")\n    n_low  = sum(1 for a in alerts if a["label"] == "LOW")'
new_kpi = '    n_high = sum(1 for a in alerts if a["score"] >= 0.45)\n    n_med  = sum(1 for a in alerts if 0.35 <= a["score"] < 0.45)\n    n_low  = sum(1 for a in alerts if a["score"] < 0.35)'
if old_kpi in txt:
    txt = txt.replace(old_kpi, new_kpi)
    ok.append("3-kpi")
else:
    print("MISS 3: kpi counts")

# ── PATCH 4: Table rows — add onclick + cursor style ─────────────────────────
old_tr = '<tr data-beat="{a[\'beat\']}" data-range="{a[\'range\']}" data-label="{a[\'label\']}">'
new_tr = '<tr data-beat="{a[\'beat\']}" data-range="{a[\'range\']}" data-label="{a[\'label\']}"\n    id="row-{aid}" onclick="zoomToAlert({aid})" style="cursor:pointer">'
if old_tr in txt:
    txt = txt.replace(old_tr, new_tr)
    ok.append("4-row-click")
else:
    print("MISS 4: tr onclick")

# ── PATCH 5: Add zoomToAlert JS function (inject before closing </script>) ────
zoom_js = """
// ── Row → map zoom ────────────────────────────────────────────────────────────
let _highlightLayer = null;
function zoomToAlert(id) {
  // clear previous highlight
  if(_highlightLayer) { map.removeLayer(_highlightLayer); _highlightLayer = null; }
  // highlight row
  document.querySelectorAll('#tbl-body tr').forEach(r => r.classList.remove('row-active'));
  const row = document.getElementById('row-' + id);
  if(row) { row.classList.add('row-active'); row.scrollIntoView({block:'nearest'}); }
  // find feature
  const feat = FC.features.find(f => f.properties && f.properties.id === id);
  if(!feat || !feat.geometry) return;
  _highlightLayer = L.geoJSON(feat, {
    style: { color:'#ffffff', weight:3, fillColor:'#fff', fillOpacity:0.25 }
  }).addTo(map);
  try { map.fitBounds(_highlightLayer.getBounds().pad(0.25)); } catch(e){}
}
"""

old_close = "\n</script>\n</body></html>\"\"\""
new_close = zoom_js + "\n</script>\n</body></html>\"\"\""
if old_close in txt:
    txt = txt.replace(old_close, new_close)
    ok.append("5-zoom-js")
else:
    print("MISS 5: zoom js")

# ── PATCH 6: Active row CSS ───────────────────────────────────────────────────
old_tr_css = "tr:hover td{{background:#1e2d4522}}"
new_tr_css = "tr:hover td{{background:#1e2d4522}}\ntr.row-active td{{background:#1e3a5f44!important;border-bottom-color:#38bdf855}}"
if old_tr_css in txt:
    txt = txt.replace(old_tr_css, new_tr_css)
    ok.append("6-css")
else:
    print("MISS 6: active css")

SRC.write_text(txt, encoding="utf-8")
print("Patches:", ok)
