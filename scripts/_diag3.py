"""Find JS errors in generated dashboard"""
from pathlib import Path, json

html = Path(__file__).parent.parent / "outputs" / "dashboard.html"
js_txt = html.read_text(encoding="utf-8")
start = js_txt.find("<script>")
end   = js_txt.rfind("</script>")
js = js_txt[start+8:end]
lines = js.split("\n")
print(f"JS lines: {len(lines)}, chars: {len(js)}")

# find FC
fc_idx = js.find("const FC")
print("FC line:", js[fc_idx:fc_idx+60])

# Count FC features
import json as J
fc_semi = js.find(";\nconst ALERTS")
try:
    fc_obj = J.loads(js[js.find("{", fc_idx):fc_semi])
    print(f"FC features: {len(fc_obj['features'])}")
except Exception as e:
    print("FC parse FAIL:", e)
