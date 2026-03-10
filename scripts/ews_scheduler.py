"""
ews_scheduler.py
================
Windows Task Scheduler wrapper for guna_ews_daemon.py.

Configure Task Scheduler to run this script daily at 03:00 AM:
  Program/script : C:\\path\\to\\venv\\Scripts\\python.exe
  Add arguments  : scripts\\ews_scheduler.py
  Start in       : C:\\Users\\S.C.C\\OneDrive\\Desktop\\jwalaNetra_2

The wrapper:
  · Sets PYTHONPATH so daemon can import _simple_dw_score
  · Appends stdout + stderr to a daily rotating log file
  · Rebuilds outputs/dashboard.html after every successful daemon run   [D]
  · Sends a Slack webhook summary when HIGH alerts are detected          [E]
  · Exits with code 0 on success, 1 on failure (so Task Scheduler can retry)
  · Prints a one-line status suitable for Windows Event Log
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_THIS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent
PYTHON_EXE   = sys.executable                          # same venv this script runs in
DAEMON       = str(_THIS_DIR / "guna_ews_daemon.py")
DASHBOARD_SC = str(_THIS_DIR / "ews_dashboard.py")
FEED_DIR     = PROJECT_ROOT / "outputs" / "dashboard_feed"
LOG_DIR      = PROJECT_ROOT / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"ews_run_{datetime.now().strftime('%Y%m%d')}.log"


# ── Notification helpers ───────────────────────────────────────────────────────
def _notify_failure(message: str) -> None:
    """Send a failure notification (Slack webhook or stderr fallback)."""
    webhook = os.getenv("SLACK_WEBHOOK")
    if webhook:
        try:
            import urllib.request
            payload = json.dumps({"text": f":x: *E-Netra EWS failed*\n{message}"}).encode()
            req = urllib.request.Request(
                webhook, data=payload,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[SLACK] Could not send failure notification: {e}", file=sys.stderr)
    print(f"[FAILURE NOTIFICATION] {message}", file=sys.stderr)


def _collect_high_alerts() -> list[dict]:
    """
    Read the most-recently-written feed GeoJSON and collect HIGH alerts.
    Returns a list of dicts: {beat, range, score, area_ha, date, typology}
    """
    if not FEED_DIR.exists():
        return []
    files = sorted(FEED_DIR.glob("*.geojson"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return []
    try:
        with open(files[0], encoding="utf-8") as f:
            fc = json.load(f)
        highs = []
        for feat in fc.get("features", []):
            p = feat.get("properties", {})
            label = p.get("label", "")
            score = float(p.get("score") or p.get("confidence") or 0.0)
            if label == "HIGH" or score >= 0.65:
                highs.append({
                    "beat":     p.get("beat_name") or p.get("beat") or "?",
                    "range":    p.get("range_name") or p.get("range") or "?",
                    "score":    round(score, 3),
                    "area_ha":  round(float(p.get("area_ha") or 0.0), 2),
                    "date":     p.get("detection_date") or p.get("date") or "?",
                    "typology": p.get("typology") or "CANOPY_LOSS",
                })
        return highs
    except Exception:
        return []


def _notify_high_alerts(highs: list[dict], log_fp) -> None:
    """Post HIGH alert digest to Slack webhook and log. [Task E]"""
    if not highs:
        log_fp.write("[SLACK] No HIGH alerts — digest skipped.\n")
        return

    webhook = os.getenv("SLACK_WEBHOOK")
    lines = [f":fire: *E-Netra V4 — {len(highs)} HIGH alert(s) detected*"]
    for h in highs:
        lines.append(
            f"  • *{h['beat']}* ({h['range']}) | score={h['score']} | "
            f"{h['area_ha']} ha | {h['typology']} | {h['date']}"
        )
    summary = "\n".join(lines)
    log_fp.write(f"\n[HIGH ALERTS]\n{summary}\n")

    if webhook:
        try:
            import urllib.request
            payload = json.dumps({"text": summary}).encode()
            req = urllib.request.Request(
                webhook, data=payload,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            urllib.request.urlopen(req, timeout=5)
            log_fp.write("[SLACK] HIGH alert digest sent OK.\n")
        except Exception as e:
            log_fp.write(f"[SLACK] Failed to send HIGH alert: {e}\n")
    else:
        log_fp.write("[SLACK] SLACK_WEBHOOK env var not set — skipping push notification.\n"
                     "        Set SLACK_WEBHOOK=https://hooks.slack.com/services/... to enable.\n")


def _rebuild_dashboard(log_fp) -> None:
    """Rebuild outputs/dashboard.html after a successful daemon run. [Task D]"""
    log_fp.write("\n[DASHBOARD] Rebuilding dashboard.html …\n")
    try:
        result = subprocess.run(
            [PYTHON_EXE, DASHBOARD_SC],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        log_fp.write(result.stdout or "")
        if result.returncode == 0:
            log_fp.write("[DASHBOARD] Rebuild OK.\n")
        else:
            log_fp.write(f"[DASHBOARD] Rebuild FAILED (rc={result.returncode}): {result.stderr}\n")
    except Exception as e:
        log_fp.write(f"[DASHBOARD] Exception during rebuild: {e}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_task() -> int:
    started = datetime.now()
    banner  = f"\n{'='*60}\nRUN START: {started.isoformat()}\n{'='*60}\n"

    with open(LOG_FILE, "a", encoding="utf-8") as log_fp:
        log_fp.write(banner)
        log_fp.flush()

        # Inherit environment, add scripts/ to PYTHONPATH so daemon can import
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(_THIS_DIR) + os.pathsep + existing_pp
            if existing_pp else str(_THIS_DIR)
        )

        try:
            subprocess.run(
                [PYTHON_EXE, DAEMON],
                cwd     = str(PROJECT_ROOT),
                env     = env,
                stdout  = log_fp,
                stderr  = subprocess.STDOUT,
                text    = True,
                check   = True,
            )
            elapsed = (datetime.now() - started).seconds
            log_fp.write(f"\nRUN SUCCESS — elapsed {elapsed}s\n")

            # ── Task D: rebuild dashboard ──────────────────────────────────────
            _rebuild_dashboard(log_fp)

            # ── Task E: notify HIGH alerts via Slack ──────────────────────────
            highs = _collect_high_alerts()
            _notify_high_alerts(highs, log_fp)

            print(
                f"[{datetime.now()}] EWS completed in {elapsed}s | "
                f"HIGH alerts: {len(highs)} | Log: {LOG_FILE}"
            )
            return 0

        except subprocess.CalledProcessError as exc:
            msg = (f"EWS Daemon FAILED with exit code {exc.returncode}. "
                   f"Log: {LOG_FILE}")
            print(f"[{datetime.now()}] {msg}", file=sys.stderr)
            log_fp.write(f"\n{msg}\n")
            _notify_failure(msg)
            return 1

        except FileNotFoundError:
            msg = f"Daemon script not found: {DAEMON}"
            print(f"[{datetime.now()}] {msg}", file=sys.stderr)
            log_fp.write(f"\n{msg}\n")
            _notify_failure(msg)
            return 1


if __name__ == "__main__":
    sys.exit(run_task())
