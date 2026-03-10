"""
E-Netra V4 — Dashboard Auto-Rebuild Watcher
Runs ews_dashboard.py on a schedule and serves outputs/ via HTTP.
Usage:  python scripts/watch_rebuild.py [--interval 300] [--port 8765]
"""
import argparse
import http.server
import threading
import time
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DASHBOARD_SCRIPT = ROOT / "scripts" / "ews_dashboard.py"


def rebuild():
    print(f"\n[{time.strftime('%H:%M:%S')}] Rebuilding dashboard...", flush=True)
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(DASHBOARD_SCRIPT)],
        cwd=ROOT,
        capture_output=False,
    )
    if result.returncode == 0:
        print(f"[{time.strftime('%H:%M:%S')}] ✓ Dashboard rebuilt OK", flush=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ✗ Rebuild failed (exit {result.returncode})", flush=True)


def serve(port: int, directory: Path):
    import os
    os.chdir(directory)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # suppress per-request logs

    httpd = http.server.HTTPServer(("127.0.0.1", port), Handler)
    print(f"  Serving http://127.0.0.1:{port}/dashboard.html", flush=True)
    httpd.serve_forever()


def main():
    p = argparse.ArgumentParser(description="E-Netra dashboard watcher")
    p.add_argument("--interval", type=int, default=300, help="Rebuild interval in seconds (default: 300)")
    p.add_argument("--port",     type=int, default=8765, help="HTTP port (default: 8765)")
    p.add_argument("--no-serve", action="store_true",  help="Skip HTTP server, only rebuild")
    args = p.parse_args()

    print("=" * 60)
    print("  E-Netra V4 — Dashboard Watcher")
    print(f"  Rebuild every {args.interval}s  |  Port {args.port}")
    print("=" * 60)

    # Initial build
    rebuild()

    # Start HTTP server in background
    if not args.no_serve:
        t = threading.Thread(
            target=serve,
            args=(args.port, ROOT / "outputs"),
            daemon=True,
        )
        t.start()

    # Rebuild loop
    try:
        while True:
            time.sleep(args.interval)
            rebuild()
    except KeyboardInterrupt:
        print("\nWatcher stopped.")


if __name__ == "__main__":
    main()
