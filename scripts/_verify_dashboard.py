"""Quick verify: check beat assignments embedded in dashboard.html"""
import json, re
from pathlib import Path

html = Path("outputs/dashboard.html").read_text(encoding="utf-8")

# check FC variable
has_fc = "const FC" in html
print(f"FC variable present: {has_fc}")
print(f"ALERTS variable present: {'const ALERTS' in html}")

# extract ALERTS JSON – it's on one line: const ALERTS = [...];
idx = html.find("const ALERTS =")
if idx < 0:
    print("ERROR: ALERTS not found")
else:
    chunk = html[idx + len("const ALERTS ="):].lstrip()
    # find the closing ]; 
    depth = 0
    end = 0
    for k, ch in enumerate(chunk):
        if ch == "[": depth += 1
        elif ch == "]": 
            depth -= 1
            if depth == 0:
                end = k + 1
                break
    data = json.loads(chunk[:end])
    print(f"\nTotal alerts in JS: {len(data)}")
    beats_found = sum(1 for a in data if a["beat"] != "—")
    print(f"Alerts with beat assigned: {beats_found}/{len(data)}")
    print()
    for a in data:
        print(f"  id={a['id']:2d} score={a['score']:.3f}  beat={a['beat']:<25} range={a['range']}")
