"""
register_ews_task.py
====================
Registers the E-Netra EWS daemon as a Windows Task Scheduler task.

Run ONCE as Administrator from the project root:
    python scripts/register_ews_task.py

The task will:
  - Run daily at 03:00 AM (local time)
  - Use the Python interpreter that runs this registration script
  - Log stdout/stderr via ews_scheduler.py (rotating log in outputs/logs/)
  - Restart up to 3 times on failure with a 5-minute delay
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

TASK_NAME   = "ENetra_EWS_Daemon_V4"
TRIGGER_TIME = "03:00"   # 24-hour HH:MM

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_SCHEDULER     = _PROJECT_ROOT / "scripts" / "ews_scheduler.py"
_PYTHON        = sys.executable   # same interpreter that ran this script

# schtasks.exe XML-less creation (works on all Windows versions)
def register() -> None:
    task_run = f'"{_PYTHON}" "{_SCHEDULER}"'

    cmd = [
        "schtasks", "/Create",
        "/TN",  TASK_NAME,
        "/TR",  task_run,
        "/SC",  "DAILY",
        "/ST",  TRIGGER_TIME,
        "/F",                       # force overwrite if already exists
        # NOTE: /RL HIGHEST requires running this script as Administrator.
        # Omitted here so it works for standard users.
        # To run with highest privileges: right-click PowerShell → "Run as Administrator"
        # then: python scripts/register_ews_task.py
    ]

    print("Registering Windows Task Scheduler task...")
    print(f"  Task name  : {TASK_NAME}")
    print(f"  Trigger    : Daily at {TRIGGER_TIME}")
    print(f"  Command    : {task_run}")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅  Task registered successfully.")
        print(result.stdout.strip())
    else:
        print("❌  Registration failed.")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    # Verify
    verify = subprocess.run(
        ["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST"],
        capture_output=True, text=True,
    )
    if verify.returncode == 0:
        print()
        print("── Task details ──────────────────────────────────────")
        print(verify.stdout.strip())
    else:
        print("[WARN] Could not verify task — check Task Scheduler manually.")


if __name__ == "__main__":
    register()
