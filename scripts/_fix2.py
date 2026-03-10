"""Quick fixup: table header + regex warning"""
from pathlib import Path
src = Path(__file__).parent / "ews_dashboard.py"
txt = src.read_text(encoding="utf-8")

# Fix table header
old = "<th>Tier</th><th>Score</th><th>Beat</th><th>Range</th>\n        <th>Date</th><th>ha</th><th>Type</th><th>\u0394Tree</th><th>Z</th><th>\u2601</th><th></th>"
new = "<th>Tier</th><th>Score</th><th>Beat</th><th>Range</th>\n        <th>Date</th><th>ha</th><th>\u0394NDVI</th><th>\u0394NBR</th><th>\u0394Trees</th><th>Z</th><th>\u2601</th><th></th>"
if old in txt:
    txt = txt.replace(old, new)
    print("TH ok")
else:
    print("TH miss")

# Fix regex warning: /\s+/ -> replace with / /
# Find the line with a.download and sanitise it avoiding \s inside string literal
idx = txt.find("a.download = beatName.replace(")
if idx != -1:
    line_end = txt.find("\n", idx)
    old_line = txt[idx:line_end]
    new_line = old_line.replace(r"/\s+/g", "/ /g")
    txt = txt[:idx] + new_line + txt[line_end:]
    print("Regex ok:", repr(new_line[:60]))
else:
    print("Regex line not found")

src.write_text(txt, encoding="utf-8")
print("Saved")
