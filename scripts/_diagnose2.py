"""Detailed JS analysis - look for actual first syntax error"""
from pathlib import Path
import re

html = (Path(__file__).parent.parent / "outputs" / "dashboard.html").read_text(encoding="utf-8")

start = html.find("<script>")
end   = html.rfind("</script>")
js = html[start+8:end]

lines = js.split("\n")
print(f"Total JS lines: {len(lines)}")
print(f"Total JS chars: {len(js)}")

# Find lines that have Python-style {{ or }} that should NOT be in output HTML
artifacts = [(i+1, l) for i, l in enumerate(lines) if "{{" in l or "}}" in l]
print(f"\nLines with {{ or }} (should be ZERO in final HTML):")
for n, l in artifacts[:15]:
    print(f"  L{n}: {l[:120]}")

# Find renderMap
idx = js.find("function renderMap")
print(f"\n--- renderMap function (first 600 chars) ---")
print(js[idx:idx+600] if idx!=-1 else "NOT FOUND")

# Find openBeatPanel  
idx2 = js.find("function openBeatPanel")
print(f"\n--- openBeatPanel (first 400 chars) ---")
print(js[idx2:idx2+400] if idx2!=-1 else "NOT FOUND")

# Find the resizer IIFE
idx3 = js.find("function initResize")
print(f"\n--- initResize (first 300 chars) ---")
print(js[idx3:idx3+300] if idx3!=-1 else "NOT FOUND")

# Find closeBeatPanel - make sure it's defined
idx4 = js.find("function closeBeatPanel")
print(f"\n--- closeBeatPanel ---")
print(js[idx4:idx4+100] if idx4!=-1 else "NOT FOUND")

# Find the bip-kml-btn (removed earlier?) 
idx5 = js.find("bip-kml-btn")
print(f"\n--- bip-kml-btn in JS? ---")
print("FOUND" if idx5!=-1 else "NOT FOUND (good)")

# check the ${{ template literal pattern usage 
# In generated HTML these should be ${ not ${{
dollar_braces = [(i+1,l) for i,l in enumerate(lines) if "${" in l]
print(f"\nLines with template literals ${{ (should exist):")
for n,l in dollar_braces[:5]:
    print(f"  L{n}:", l[:100])
