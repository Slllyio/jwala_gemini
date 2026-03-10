"""
JwalaNetra Command Horizon v2 - Real Data Server
Serves actual pipeline outputs to the dashboard.

Usage:
    python serve_real_data.py
    Then open http://localhost:8790/v2/index.html

Endpoints:
    GET /api/real-data[?date=YYYY-MM-DD]  - Main dashboard payload
    GET /api/beat-sops[?date=YYYY-MM-DD]  - Full SOP index by beat_id
    GET /api/fsi-compartments             - Per-compartment FSI scores
    GET /api/beat-rakshak[?date=YYYY-MM-DD] - Patrol Rakshak data by beat_id
    GET /api/available-dates              - Dates that have SOP output files
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
PHENOLOGY_DIR = BASE / "jwalaNetra_2" / "data" / "phenology" / "range_models"

# New data paths
FSI_COMPARTMENTS = (
    BASE / "jwalaNetra_2" / "outputs" / "checkpoints" / "outputs"
    / "fsi_viz" / "fsi_compartments.json"
)
PIPELINE_RUNS = (
    BASE / "jwalaNetra_2" / "outputs" / "checkpoints" / "outputs"
    / "jvala" / "pipeline_runs"
)

PORT  = 8790
TODAY = datetime.now().strftime("%Y-%m-%d")


# ─── Helper: FSI data ─────────────────────────────────────────────────────────

def load_fsi_compartments():
    """Load FSI compartments list; return (list, name->entry dict)."""
    if not FSI_COMPARTMENTS.exists():
        return [], {}
    try:
        entries = json.loads(FSI_COMPARTMENTS.read_text(encoding="utf-8"))
        # Build index on 'name' field (e.g. "P 296") which matches NEW_No_
        index = {e["name"]: e for e in entries if "name" in e}
        return entries, index
    except Exception as exc:
        print(f"[WARN] Could not load FSI compartments: {exc}")
        return [], {}


def fsi_max_score(entries):
    """Return the maximum mean_fsi across all compartments (for normalisation)."""
    scores = [e.get("mean_fsi", 0) for e in entries if e.get("mean_fsi")]
    return max(scores) if scores else 1.0


# ─── Helper: pipeline run metadata ───────────────────────────────────────────

def load_pipeline_metadata():
    """Read the latest pipeline run file and return summary metadata."""
    if not PIPELINE_RUNS.exists():
        return {}
    run_files = sorted(PIPELINE_RUNS.glob("run_*.json"))
    if not run_files:
        return {}
    latest = run_files[-1]
    try:
        run = json.loads(latest.read_text(encoding="utf-8"))
        return {
            "run_file":        latest.name,
            "anchor_date":     run.get("anchor_date"),
            "beats_processed": run.get("beats_processed", 0),
            "beats_escalated": run.get("beats_escalated", 0),
            "errors":          run.get("errors", 0),
        }
    except Exception as exc:
        print(f"[WARN] Could not load pipeline run metadata: {exc}")
        return {}


# ─── Helper: available dates ──────────────────────────────────────────────────

def get_available_dates():
    """Scan for dates that have SOP output files."""
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


# ─── Helper: satellite pass ───────────────────────────────────────────────────

def get_sat_pass_info():
    """Extract satellite pass info from alert data."""
    if not ALERT_LATEST.exists():
        return {"last_pass": "--", "sources": []}
    try:
        alert = json.loads(ALERT_LATEST.read_text(encoding="utf-8"))
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


# ─── Vulnerability computation ────────────────────────────────────────────────

def compute_vulnerability(sop, fsi_normalized):
    """
    Composite vulnerability index incorporating FSI when available.

    Formula (with FSI):
        vuln = 0.40 * fsi_n + 0.25 * ros_n + 0.20 * slope_n
               + 0.10 * wind_n + 0.05 * haines_n

    Formula (without FSI, legacy fallback):
        vuln = 0.35 * ros_n + 0.30 * slope_n + 0.20 * wind_n + 0.15 * haines_n
    """
    m         = sop.get("muhurta", {})
    ros       = m.get("ros_m_min", 0) or 0
    slope     = m.get("slope_pct", 0) or 0
    wind      = m.get("wind_speed_kmh", 0) or 0
    haines    = m.get("haines_index", 0) or 0

    ros_n    = min(1.0, max(0.0, (ros - 0.42) / 0.1))
    slope_n  = min(1.0, max(0.0, (slope - 1.5) / 11.0))
    wind_n   = min(1.0, max(0.0, (wind - 19.5) / 2.0))
    haines_n = min(1.0, max(0.0, (haines - 3) / 3.0))

    if fsi_normalized is not None:
        vuln = (
            0.40 * fsi_normalized
            + 0.25 * ros_n
            + 0.20 * slope_n
            + 0.10 * wind_n
            + 0.05 * haines_n
        )
    else:
        vuln = (
            0.35 * ros_n
            + 0.30 * slope_n
            + 0.20 * wind_n
            + 0.15 * haines_n
        )

    return round(vuln, 4)


# ─── Main data assembly ───────────────────────────────────────────────────────

def build_real_dashboard_data(target_date=None):
    """Assemble all real data into a single JSON payload for the dashboard."""
    date_str = target_date or TODAY

    # Pre-load FSI so it can be merged into beats
    fsi_entries, fsi_index = load_fsi_compartments()
    fsi_max = fsi_max_score(fsi_entries)

    data = {
        "date":             date_str,
        "generated_at":     datetime.now().isoformat(),
        "source":           "REAL_PIPELINE",
        "available_dates":  get_available_dates(),
        "sat_pass":         get_sat_pass_info(),
        "pipeline_meta":    load_pipeline_metadata(),
        "phenology_models": _load_phenology(),
        "fsi_loaded":       len(fsi_entries),
    }

    # 1. Latest alert (FWI, fires, severity)
    if ALERT_LATEST.exists():
        try:
            alert = json.loads(ALERT_LATEST.read_text(encoding="utf-8"))
            data["alert"]            = alert
            data["fwi"]              = alert.get("fwi", {})
            data["bulletin_severity"] = alert.get("bulletin_severity", "UNKNOWN")
            data["fires_detected"]   = alert.get("fires_detected", 0)
            data["fires"]            = alert.get("fires", [])
        except Exception as exc:
            print(f"[WARN] Could not load alert: {exc}")
            data["alert"] = None
            data["fwi"]   = {}
    else:
        data["alert"]            = None
        data["fwi"]              = {}
        data["bulletin_severity"] = "UNKNOWN"
        data["fires_detected"]   = 0
        data["fires"]            = []

    # 2. Fused per-beat risk data
    beats = []
    if FUSED_LATEST.exists():
        try:
            fused     = json.loads(FUSED_LATEST.read_text(encoding="utf-8"))
            beats_raw = fused.get("beats", [])

            for b in beats_raw:
                src   = b.get("sources", {}).get("jwalaNetra", {})
                firms = b.get("sources", {}).get("firms_nrt", {})
                beat_id = b.get("beat_id", "")

                # FSI lookup: beat_id often encodes compartment number like "P 245"
                fsi_entry = _resolve_fsi(beat_id, fsi_index)
                mean_fsi  = fsi_entry.get("mean_fsi") if fsi_entry else None
                fsi_n     = (mean_fsi / fsi_max) if (mean_fsi is not None and fsi_max > 0) else None

                beats.append({
                    "beat_id":       beat_id,
                    "fused_tier":    b.get("fused_tier"),
                    "risk_score":    src.get("risk_score", 0),
                    "p_ignition":    src.get("p_ignition", 0),
                    "muhurta_stage": src.get("muhurta_stage"),
                    "lat":           b.get("centroid_lat"),
                    "lon":           b.get("centroid_lon"),
                    "fire_count":    firms.get("fire_count", 0),
                    "peak_frp":      firms.get("peak_frp_mw", 0),
                    "mean_fsi":      mean_fsi,
                    "max_fsi":       fsi_entry.get("max_fsi") if fsi_entry else None,
                    "fsi_waypoints": fsi_entry.get("waypoints", []) if fsi_entry else [],
                })
        except Exception as exc:
            print(f"[WARN] Could not load fused data: {exc}")

    data["beats"]       = beats
    data["total_beats"] = len(beats)
    data["tier_summary"] = dict(Counter(b["fused_tier"] for b in beats if b["fused_tier"]))
    data["active_beats"] = [
        b for b in beats
        if b["fused_tier"] not in (None, "CLEAR") or b["fire_count"] > 0
    ]

    # 3. Per-beat SOP data
    sops    = _load_sops(date_str)
    sop_map = {s["beat_id"]: s for s in sops}

    # Compute vulnerability with FSI
    for sop in sops:
        bid       = sop.get("beat_id", "")
        fsi_entry = _resolve_fsi(bid, fsi_index)
        mean_fsi  = fsi_entry.get("mean_fsi") if fsi_entry else None
        fsi_n     = (mean_fsi / fsi_max) if (mean_fsi is not None and fsi_max > 0) else None
        sop["mean_fsi"]      = mean_fsi
        sop["vulnerability"] = compute_vulnerability(sop, fsi_n)

    data["sops"]      = sops
    data["sop_count"] = len(sops)

    # Merge SOP vulnerability + physics back into beats array
    for b in beats:
        s = sop_map.get(b["beat_id"])
        if s:
            b["vulnerability"] = s.get("vulnerability", 0)
            m = s.get("muhurta", {})
            b["ros"]   = m.get("ros_m_min", 0)
            b["slope"] = m.get("slope_pct", 0)

    return data


def _load_phenology():
    """Load all range phenology model JSON files."""
    models = {}
    if not PHENOLOGY_DIR.exists():
        return models
    for pf in PHENOLOGY_DIR.glob("*.json"):
        try:
            models[pf.stem] = json.loads(pf.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Phenology load failed for {pf.name}: {exc}")
    return models


def _load_sops(date_str):
    """Load SOP JSON files for a given date, return list of dicts."""
    sops = []
    for sf in list(JVALA_OUTPUTS.glob(f"*{date_str}_sop.json"))[:747]:
        try:
            sop = json.loads(sf.read_text(encoding="utf-8"))
            sops.append({
                "beat_id":       sop.get("beat_id"),
                "tier":          sop.get("tier"),
                "risk_score":    sop.get("risk_score", 0),
                "p_ignition":    sop.get("p_ignition", 0),
                "ema_score":     sop.get("ema_score", 0),
                "confidence":    sop.get("confidence", 0),
                "lat":           sop.get("centroid_lat"),
                "lon":           sop.get("centroid_lon"),
                "spread_tier":   sop.get("spread_tier"),
                "spread_area_ha": sop.get("spread_area_ha"),
                "muhurta":       sop.get("muhurta", {}),
                "cfl_rank":      sop.get("cfl_rank"),
                "shap_reasons":  sop.get("shap_reasons", []),
            })
        except Exception as exc:
            print(f"[WARN] SOP load failed for {sf.name}: {exc}")
    return sops


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


def _resolve_fsi(beat_id, fsi_index):
    """
    Try to match a beat_id string to a compartment name in fsi_index.

    Beat IDs can look like:
      "AGARA / P 245"   -> key "P 245"
      "AGARA - P_245"   -> normalise to "P 245"
    FSI compartment names look like "P 296", "P 245", etc.
    """
    if not fsi_index or not beat_id:
        return None

    # Direct match first
    if beat_id in fsi_index:
        return fsi_index[beat_id]

    # Try to extract trailing compartment code, e.g. "P 245" or "P245"
    import re
    # Pattern: optional letter(s), space or underscore, digits
    m = re.search(r'\b([A-Z])\s*[-_]?\s*(\d{2,4})\b', beat_id.upper())
    if m:
        candidate = f"{m.group(1)} {m.group(2)}"
        if candidate in fsi_index:
            return fsi_index[candidate]

    # Last fallback: look for any slash-separated part matching an FSI name
    for part in re.split(r'[/\-_,]', beat_id):
        part = part.strip()
        if part in fsi_index:
            return fsi_index[part]

    return None


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
        elif path == "/api/fsi-compartments":
            self._handle_fsi_compartments()
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
        """Return full SOP records indexed by beat_id."""
        date_str = qs.get("date", [TODAY])[0]
        sops     = _load_sops(date_str)
        index    = {s["beat_id"]: s for s in sops if s.get("beat_id")}
        self._send_json(index)

    def _handle_fsi_compartments(self):
        """Return the raw FSI compartments list (strips heavy cells array)."""
        entries, _ = load_fsi_compartments()
        # Strip 'cells' to keep response lightweight; dashboard only needs summary fields
        slim = [
            {k: v for k, v in e.items() if k != "cells"}
            for e in entries
        ]
        self._send_json({"compartments": slim, "count": len(slim)})

    def _handle_beat_rakshak(self, qs):
        """Return Rakshak patrol data indexed by beat_id."""
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
    print(f"Date:            {TODAY}")
    print(f"Alert:           {ALERT_LATEST}")
    print(f"Fused:           {FUSED_LATEST}")
    print(f"SOP / Rakshak:   {JVALA_OUTPUTS}")
    print(f"FSI compartments:{FSI_COMPARTMENTS}")
    print(f"Pipeline runs:   {PIPELINE_RUNS}")
    print(f"Dashboard:       {DASHBOARD_DIR}")
    print()

    # Pre-flight data check
    data = build_real_dashboard_data()
    fsi_entries, _ = load_fsi_compartments()

    print(f"Beats loaded:      {data['total_beats']}")
    print(f"SOPs loaded:       {data['sop_count']}")
    print(f"Rakshak files:     {len(list(JVALA_OUTPUTS.glob(f'*{TODAY}_rakshak.json')))}")
    print(f"FSI compartments:  {len(fsi_entries)}")
    print(f"Phenology models:  {len(data['phenology_models'])}")
    meta = data.get("pipeline_meta", {})
    if meta:
        print(f"Pipeline run:      {meta.get('run_file')} "
              f"({meta.get('beats_processed')} beats, "
              f"{meta.get('beats_escalated')} escalated, "
              f"{meta.get('errors')} errors)")
    print(f"Bulletin severity: {data.get('bulletin_severity', 'N/A')}")
    print(f"FWI value:         {data['fwi'].get('value', 'N/A')}")
    print(f"Fires detected:    {data.get('fires_detected', 0)}")
    print()
    print(f"Server:  http://localhost:{PORT}/v2/index.html")
    print(f"API:     http://localhost:{PORT}/api/real-data")
    print(f"FSI:     http://localhost:{PORT}/api/fsi-compartments")
    print(f"Rakshak: http://localhost:{PORT}/api/beat-rakshak")
    print()

    server = HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
