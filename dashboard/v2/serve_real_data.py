"""
JwalaNetra Command Horizon v2 - Real Data Server
Serves actual VANAAGNI_TACTICAL_2026 pipeline outputs to the dashboard.
No fabricated fields. Every value served comes directly from a pipeline file.

Usage:
    python serve_real_data.py
    Then open http://localhost:8790/v2/index.html

Endpoints:
    GET /api/real-data[?date=YYYY-MM-DD]    - Main dashboard payload
    GET /api/beat-sops[?date=YYYY-MM-DD]    - Full SOP index by beat_id
    GET /api/beat-rakshak[?date=YYYY-MM-DD] - Patrol Rakshak data by beat_id
    GET /api/available-dates                - Dates that have SOP output files
"""

import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime
from collections import Counter

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE          = Path(r"C:\Users\S.C.C\OneDrive\Desktop\JwalaNetra_Vanaagni_Operational_2026")
JVALA_OUTPUTS = BASE / "jwalaNetra" / "outputs" / "jvala"
FUSED_LATEST  = BASE / "outputs" / "fused" / "fused_latest.json"
ALERT_LATEST  = BASE / "jwalaNetra_2" / "data_lake" / "alerts" / "latest_alert.json"
DASHBOARD_DIR = BASE / "jwalaNetra_2" / "dashboard"

PORT  = 8790
TODAY = datetime.now().strftime("%Y-%m-%d")


# ─── Helper: available dates ──────────────────────────────────────────────────

def get_available_dates():
    """Scan JVALA_OUTPUTS for dates that have SOP output files."""
    dates = set()
    for sf in JVALA_OUTPUTS.glob("*_sop.json"):
        name = sf.stem  # e.g. AGARA_-_P_245_2026-03-10_sop
        for part in name.split("_"):
            if len(part) == 10 and part[4:5] == "-" and part[7:8] == "-":
                try:
                    datetime.strptime(part, "%Y-%m-%d")
                    dates.add(part)
                except ValueError:
                    pass
                break
    return sorted(dates)


# ─── Helper: satellite pass info ──────────────────────────────────────────────

def get_sat_pass_info():
    """Extract satellite pass info from latest_alert.json."""
    if not ALERT_LATEST.exists():
        return {"last_pass": "--", "sources": []}
    try:
        alert     = json.loads(ALERT_LATEST.read_text(encoding="utf-8"))
        sources   = alert.get("data_sources", [])
        generated = alert.get("generated_at", "")
        try:
            dt       = datetime.fromisoformat(generated)
            time_str = dt.strftime("%H:%M IST")
        except (ValueError, TypeError):
            time_str = "--"
        return {
            "last_pass":    time_str,
            "sources":      sources,
            "generated_at": generated,
        }
    except Exception as exc:
        print(f"[WARN] Could not load sat pass info: {exc}")
        return {"last_pass": "--", "sources": []}


# ─── Main data assembly ───────────────────────────────────────────────────────

def build_real_dashboard_data(target_date=None):
    """Assemble pipeline outputs into a single JSON payload for the dashboard.

    All fields come directly from pipeline files:
      - fused_latest.json  -> division_fwi, bulletin_severity, total_firms_fires, beats
      - latest_alert.json  -> sat_pass info, generated_at, fires, data_sources
      - *_sop.json files   -> per-beat SOP detail (loaded separately via /api/beat-sops)
    No computed or fabricated composite scores are added here.
    """
    date_str = target_date or TODAY

    data = {
        "date":            date_str,
        "generated_at":    datetime.now().isoformat(),
        "source":          "REAL_PIPELINE",
        "available_dates": get_available_dates(),
        "sat_pass":        get_sat_pass_info(),
    }

    # 1. Latest alert (fires list, data_sources, generated_at)
    if ALERT_LATEST.exists():
        try:
            alert = json.loads(ALERT_LATEST.read_text(encoding="utf-8"))
            data["alert"]             = alert
            data["fires"]             = alert.get("fires", [])
            data["total_firms_fires"] = len(data["fires"])
        except Exception as exc:
            print(f"[WARN] Could not load alert: {exc}")
            data["alert"]             = None
            data["fires"]             = []
            data["total_firms_fires"] = 0
    else:
        data["alert"]             = None
        data["fires"]             = []
        data["total_firms_fires"] = 0

    # 2. Fused per-beat risk data from fused_latest.json
    beats = []
    bulletin_severity  = "UNKNOWN"
    division_fwi       = {}

    if FUSED_LATEST.exists():
        try:
            fused = json.loads(FUSED_LATEST.read_text(encoding="utf-8"))

            bulletin_severity = fused.get("bulletin_severity", "UNKNOWN")
            division_fwi      = fused.get("division_fwi", {})

            for b in fused.get("beats", []):
                src   = b.get("sources", {}).get("jwalaNetra", {})
                firms = b.get("sources", {}).get("firms_nrt", {})

                beats.append({
                    "beat_id":          b.get("beat_id"),
                    "fused_tier":       b.get("fused_tier"),
                    "fused_tier_rank":  b.get("fused_tier_rank"),
                    "centroid_lat":     b.get("centroid_lat"),
                    "centroid_lon":     b.get("centroid_lon"),
                    # Flatten key fields for frontend convenience
                    "risk_score":       src.get("risk_score"),
                    "p_ignition":       src.get("p_ignition"),
                    "muhurta_stage":    src.get("muhurta_stage"),
                    "peak_frp":         firms.get("peak_frp_mw", 0),
                    "fire_count":       firms.get("fire_count", 0),
                    "lat":              b.get("centroid_lat"),
                    "lon":              b.get("centroid_lon"),
                    "sources": {
                        "jwalaNetra": {
                            "tier":          src.get("tier"),
                            "risk_score":    src.get("risk_score"),
                            "p_ignition":    src.get("p_ignition"),
                            "muhurta_stage": src.get("muhurta_stage"),
                        },
                        "firms_nrt": {
                            "fire_count":    firms.get("fire_count", 0),
                            "peak_frp_mw":   firms.get("peak_frp_mw", 0),
                            "fires":         firms.get("fires", []),
                        },
                        "vanaagni_severity": b.get("sources", {}).get("vanaagni_severity"),
                    },
                })
        except Exception as exc:
            print(f"[WARN] Could not load fused data: {exc}")

    data["beats"]             = beats
    data["total_beats"]       = len(beats)
    data["bulletin_severity"] = bulletin_severity
    data["division_fwi"]      = division_fwi
    data["tier_summary"]      = dict(
        Counter(b["fused_tier"] for b in beats if b["fused_tier"])
    )
    data["active_beats"] = [
        b for b in beats
        if b["fused_tier"] not in (None, "CLEAR")
        or b["sources"]["firms_nrt"]["fire_count"] > 0
    ]

    return data


# ─── SOP loader ───────────────────────────────────────────────────────────────

def _load_sops(date_str):
    """Load SOP JSON files for a given date. Returns list of dicts.

    Only fields that exist in the pipeline SOP file are included.
    No fabricated or derived fields are added.
    """
    sops = []
    for sf in list(JVALA_OUTPUTS.glob(f"*{date_str}_sop.json"))[:747]:
        try:
            sop = json.loads(sf.read_text(encoding="utf-8"))
            sops.append({
                "beat_id":         sop.get("beat_id"),
                "tier":            sop.get("tier"),
                "risk_score":      sop.get("risk_score"),
                "p_ignition":      sop.get("p_ignition"),
                "ema_score":       sop.get("ema_score"),
                "confidence":      sop.get("confidence"),
                "muhurta_stage":   sop.get("muhurta_stage"),
                "muhurta_safe":    sop.get("muhurta_safe"),
                "muhurta":         sop.get("muhurta", {}),
                "cfl_rank":        sop.get("cfl_rank"),
                "spread_tier":     sop.get("spread_tier"),
                "spread_area_ha":  sop.get("spread_area_ha"),
                "shap_reasons":    sop.get("shap_reasons", []),
            })
        except Exception as exc:
            print(f"[WARN] SOP load failed for {sf.name}: {exc}")
    return sops


# ─── Rakshak loader ───────────────────────────────────────────────────────────

def _load_rakshak(date_str):
    """Load Rakshak patrol JSON files for a given date, indexed by beat_id."""
    index = {}
    for rf in JVALA_OUTPUTS.glob(f"*{date_str}_rakshak.json"):
        try:
            rec = json.loads(rf.read_text(encoding="utf-8"))
            bid = rec.get("beat_id")
            if bid:
                index[bid] = rec
        except Exception as exc:
            print(f"[WARN] Rakshak load failed for {rf.name}: {exc}")
    return index


# ─── HTTP Handler ─────────────────────────────────────────────────────────────

class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves dashboard static files plus /api/* endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    # ------------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        qs     = parse_qs(parsed.query)

        if path == "/api/real-data":
            self._handle_real_data(qs)
        elif path == "/api/beat-sops":
            self._handle_beat_sops(qs)
        elif path == "/api/beat-rakshak":
            self._handle_beat_rakshak(qs)
        elif path == "/api/available-dates":
            self._handle_available_dates()
        else:
            super().do_GET()

    # ------------------------------------------------------------------
    def _handle_real_data(self, qs):
        target_date = qs.get("date", [None])[0]
        data = build_real_dashboard_data(target_date)
        self._send_json(data)

    def _handle_beat_sops(self, qs):
        """Return SOP records indexed by beat_id for a given date."""
        date_str = qs.get("date", [TODAY])[0]
        sops     = _load_sops(date_str)
        index    = {s["beat_id"]: s for s in sops if s.get("beat_id")}
        self._send_json(index)

    def _handle_beat_rakshak(self, qs):
        """Return Rakshak patrol data indexed by beat_id for a given date."""
        date_str = qs.get("date", [TODAY])[0]
        index    = _load_rakshak(date_str)
        self._send_json({"date": date_str, "count": len(index), "beats": index})

    def _handle_available_dates(self):
        dates = get_available_dates()
        self._send_json({"dates": dates, "today": TODAY})

    # ------------------------------------------------------------------
    def _send_json(self, payload):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _fmt, *args):
        if "/api/" in str(args[0]):
            print(f"[API] {args[0]}")


# ─── Startup ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== JwalaNetra Command Horizon v2 - Real Data Server ===")
    print(f"Date:          {TODAY}")
    print(f"Alert:         {ALERT_LATEST}")
    print(f"Fused:         {FUSED_LATEST}")
    print(f"SOP/Rakshak:   {JVALA_OUTPUTS}")
    print(f"Dashboard:     {DASHBOARD_DIR}")
    print()

    # Pre-flight data check
    data = build_real_dashboard_data()

    print(f"Beats loaded:      {data['total_beats']}")
    print(f"Tier summary:      {data['tier_summary']}")
    print(f"SOPs available:    {len(list(JVALA_OUTPUTS.glob(f'*{TODAY}_sop.json')))}")
    print(f"Rakshak files:     {len(list(JVALA_OUTPUTS.glob(f'*{TODAY}_rakshak.json')))}")
    print(f"Bulletin severity: {data.get('bulletin_severity', 'N/A')}")
    fwi = data.get("division_fwi", {})
    print(f"FWI value:         {fwi.get('value', 'N/A')}  "
          f"danger class: {fwi.get('danger_class', 'N/A')}")
    print(f"Fires detected:    {data.get('total_firms_fires', 0)}")
    print(f"Active beats:      {len(data.get('active_beats', []))}")
    print()
    print(f"Server:  http://localhost:{PORT}/v2/index.html")
    print(f"API:     http://localhost:{PORT}/api/real-data")
    print(f"SOPs:    http://localhost:{PORT}/api/beat-sops")
    print(f"Rakshak: http://localhost:{PORT}/api/beat-rakshak")
    print(f"Dates:   http://localhost:{PORT}/api/available-dates")
    print()

    server = HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
