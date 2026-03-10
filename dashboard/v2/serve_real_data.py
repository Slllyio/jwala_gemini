"""
JwalaNetra Command Horizon v2 - Real Data Server
Serves actual pipeline outputs to the dashboard.

Usage:
    python serve_real_data.py
    Then open http://localhost:8790/v2/index.html
"""

import json
import os
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime

# ─── Paths ───
BASE = Path(r"C:\Users\S.C.C\OneDrive\Desktop\JwalaNetra_Vanaagni_Operational_2026")
JVALA_OUTPUTS = BASE / "jwalaNetra" / "outputs" / "jvala"
FUSED_LATEST = BASE / "outputs" / "fused" / "fused_latest.json"
ALERT_LATEST = BASE / "jwalaNetra_2" / "data_lake" / "alerts" / "latest_alert.json"
DASHBOARD_DIR = BASE / "jwalaNetra_2" / "dashboard"
PHENOLOGY_DIR = BASE / "jwalaNetra_2" / "data" / "phenology" / "range_models"

PORT = 8790
TODAY = datetime.now().strftime("%Y-%m-%d")


def build_real_dashboard_data():
    """Assemble all real data into a single JSON for the dashboard."""
    data = {
        "date": TODAY,
        "generated_at": datetime.now().isoformat(),
        "source": "REAL_PIPELINE",
    }

    # 1. Latest alert (FWI, fires, severity)
    if ALERT_LATEST.exists():
        alert = json.loads(ALERT_LATEST.read_text(encoding="utf-8"))
        data["alert"] = alert
        data["fwi"] = alert.get("fwi", {})
        data["bulletin_severity"] = alert.get("bulletin_severity", "UNKNOWN")
        data["fires_detected"] = alert.get("fires_detected", 0)
        data["fires"] = alert.get("fires", [])
    else:
        data["alert"] = None
        data["fwi"] = {}

    # 2. Fused per-beat risk data (the real risk scores)
    if FUSED_LATEST.exists():
        fused = json.loads(FUSED_LATEST.read_text(encoding="utf-8"))
        beats_raw = fused.get("beats", [])

        beats = []
        for b in beats_raw:
            src = b.get("sources", {}).get("jwalaNetra", {})
            firms = b.get("sources", {}).get("firms_nrt", {})
            beats.append({
                "beat_id": b.get("beat_id"),
                "fused_tier": b.get("fused_tier"),
                "risk_score": src.get("risk_score", 0),
                "p_ignition": src.get("p_ignition", 0),
                "muhurta_stage": src.get("muhurta_stage"),
                "lat": b.get("centroid_lat"),
                "lon": b.get("centroid_lon"),
                "fire_count": firms.get("fire_count", 0),
                "peak_frp": firms.get("peak_frp_mw", 0),
            })

        data["beats"] = beats
        data["total_beats"] = len(beats)

        # Tier summary
        from collections import Counter
        tier_counts = Counter(b["fused_tier"] for b in beats)
        data["tier_summary"] = dict(tier_counts)

        # Beats with active fires or elevated risk
        data["active_beats"] = [b for b in beats if b["fused_tier"] != "CLEAR" or b["fire_count"] > 0]
    else:
        data["beats"] = []

    # 3. Per-beat SOP data (real tactical output)
    sop_files = list(JVALA_OUTPUTS.glob(f"*{TODAY}_sop.json"))
    sops = []
    for sf in sop_files[:747]:  # cap at total beats
        try:
            sop = json.loads(sf.read_text(encoding="utf-8"))
            sops.append({
                "beat_id": sop.get("beat_id"),
                "tier": sop.get("tier"),
                "risk_score": sop.get("risk_score", 0),
                "p_ignition": sop.get("p_ignition", 0),
                "ema_score": sop.get("ema_score", 0),
                "confidence": sop.get("confidence", 0),
                "lat": sop.get("centroid_lat"),
                "lon": sop.get("centroid_lon"),
                "spread_tier": sop.get("spread_tier"),
                "spread_area_ha": sop.get("spread_area_ha"),
                "muhurta": sop.get("muhurta", {}),
                "cfl_rank": sop.get("cfl_rank"),
                "shap_reasons": sop.get("shap_reasons", []),
            })
        except Exception:
            pass

    data["sops"] = sops
    data["sop_count"] = len(sops)

    # 3b. Compute per-beat vulnerability index from muhurta physics
    # When p_ignition is uniform (e.g., all CLEAR day), this gives
    # a physically-meaningful gradient for heatmap visualization.
    for sop in sops:
        m = sop.get("muhurta", {})
        ros = m.get("ros_m_min", 0) or 0
        slope = m.get("slope_pct", 0) or 0
        wind = m.get("wind_speed_kmh", 0) or 0
        haines = m.get("haines_index", 0) or 0
        # Normalize each: ROS 0-1 (range ~0.43-0.52), slope 0-1 (1.6-12.3), wind 0-1 (20-21)
        ros_n = min(1.0, max(0, (ros - 0.42) / 0.1))
        slope_n = min(1.0, max(0, (slope - 1.5) / 11.0))
        wind_n = min(1.0, max(0, (wind - 19.5) / 2.0))
        haines_n = min(1.0, max(0, (haines - 3) / 3.0))
        # Composite vulnerability (weighted)
        vuln = 0.35 * ros_n + 0.30 * slope_n + 0.20 * wind_n + 0.15 * haines_n
        sop["vulnerability"] = round(vuln, 4)

    # Also add vulnerability to beats array
    sop_map = {s["beat_id"]: s for s in sops}
    for b in data.get("beats", []):
        s = sop_map.get(b.get("beat_id"))
        if s:
            b["vulnerability"] = s.get("vulnerability", 0)
            b["ros"] = s.get("muhurta", {}).get("ros_m_min", 0)
            b["slope"] = s.get("muhurta", {}).get("slope_pct", 0)

    # 4. Phenology models (real harmonic coefficients)
    phenology = {}
    if PHENOLOGY_DIR.exists():
        for pf in PHENOLOGY_DIR.glob("*.json"):
            try:
                phenology[pf.stem] = json.loads(pf.read_text(encoding="utf-8"))
            except Exception:
                pass
    data["phenology_models"] = phenology

    return data


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves dashboard files + /api/real-data endpoint."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/real-data":
            self.send_json_response()
        elif self.path == "/api/beat-sops":
            self.send_sops_response()
        else:
            super().do_GET()

    def send_json_response(self):
        data = build_real_dashboard_data()
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def send_sops_response(self):
        """Return all SOP data indexed by beat_id for fast lookup."""
        sop_index = {}
        sop_files = list(JVALA_OUTPUTS.glob(f"*{TODAY}_sop.json"))
        for sf in sop_files:
            try:
                sop = json.loads(sf.read_text(encoding="utf-8"))
                sop_index[sop["beat_id"]] = sop
            except Exception:
                pass
        body = json.dumps(sop_index, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        if "/api/" in str(args[0]):
            print(f"[API] {args[0]}")


if __name__ == "__main__":
    print(f"=== JwalaNetra Command Horizon v2 - Real Data Server ===")
    print(f"Date:       {TODAY}")
    print(f"Alert:      {ALERT_LATEST}")
    print(f"Fused:      {FUSED_LATEST}")
    print(f"SOP dir:    {JVALA_OUTPUTS}")
    print(f"Dashboard:  {DASHBOARD_DIR}")
    print(f"")

    # Pre-check data
    data = build_real_dashboard_data()
    print(f"Beats loaded:      {data['total_beats']}")
    print(f"SOPs loaded:       {data['sop_count']}")
    print(f"Bulletin severity: {data['bulletin_severity']}")
    print(f"FWI value:         {data['fwi'].get('value', 'N/A')}")
    print(f"Fires detected:    {data['fires_detected']}")
    print(f"Phenology models:  {len(data['phenology_models'])}")
    print(f"")
    print(f"Server: http://localhost:{PORT}/v2/index.html")
    print(f"API:    http://localhost:{PORT}/api/real-data")

    server = HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
