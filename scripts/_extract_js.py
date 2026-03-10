"""Extract JS from dashboard.html and save for node syntax check"""
from pathlib import Path

html = (Path(__file__).parent.parent / "outputs" / "dashboard.html").read_text(encoding="utf-8")
start = html.find("<script>")
end   = html.rfind("</script>")
js = html[start+8:end]

out = Path(__file__).parent.parent / "outputs" / "_check.js"
out.write_text(js, encoding="utf-8")
print(f"Saved {len(js)} chars to {out}")
print(f"Last 200 chars:\n{js[-200:]}")
