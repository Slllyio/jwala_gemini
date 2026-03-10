"""Analyze dashboard.html JS section for problems"""
from pathlib import Path

html = Path(__file__).parent.parent / "outputs" / "dashboard.html"
txt = html.read_text(encoding="utf-8")

start = txt.find("<script>")
end   = txt.rfind("</script>")
js = txt[start+8:end]

lines = js.split("\n")
print(f"JS section: {len(lines)} lines ({len(js)} chars)")

# Check 1: backtick balance
bt = js.count("`")
print(f"Backticks: {bt} -> {'EVEN OK' if bt % 2 == 0 else 'ODD - UNCLOSED TEMPLATE!'}")

# Check 2: look for lines with { not doubled (raw braces that shouldn't be there)
# Find first line with a bare { that might cause issues
problem_lines = []
for i, line in enumerate(lines, 1):
    # skip comment lines
    stripped = line.strip()
    if stripped.startswith("//") or stripped.startswith("*"):
        continue
    # look for =>{ without doubling (this would be raw in f-string = syntax error)
    # In generated HTML these should already be single braces (f-string resolved them)
    # So just look for Python f-string artifacts: {{ or }} that leaked
    if "{{" in line or "}}" in line:
        problem_lines.append((i, line[:100]))

if problem_lines:
    print(f"\nF-STRING ARTIFACTS ({{{{ or }}}}) LEAKED into HTML:")
    for n, l in problem_lines[:20]:
        print(f"  Line {n}: {l}")
else:
    print("No f-string artifacts found in HTML output")

# Check 3: find the openReport function
idx = js.find("openReport")
if idx != -1:
    chunk = js[max(0,idx-50):idx+500]
    print("\n--- openReport area ---")
    print(chunk[:400])

# Check 4: look for the renderMap function and map init
print("\n--- map init ---")
idx2 = js.find("L.map(")
if idx2 != -1:
    print(js[idx2:idx2+200])
else:
    print("L.map NOT FOUND!")

# Check 5: look for first JS error - any obvious syntax issues
# Find anything after the closing }} of a function that looks wrong
print("\n--- BEATS_FC const ---")
idx3 = js.find("const BEATS_FC")
print(js[idx3:idx3+50] if idx3!=-1 else "NOT FOUND")

# Check 6: check if dlAlertKml and resize JS have unresolved {{ }}
print("\n--- dlAlertKml fragment ---")
idx4 = js.find("dlAlertKml")
if idx4 != -1:
    print(js[idx4:idx4+300])
