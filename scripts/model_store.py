"""
Model Store — Version, Register, and Archive Model Artifacts
=============================================================

CLI tool for managing model experiment artifacts:
  register : snapshot model + config + report → versioned archive
  list     : show all registered versions with key metrics
  promote  : mark a version as production
  export   : pack a version into a .tar.gz

Usage:
    python scripts/model_store.py register \\
        --source outputs/alert_filter_enriched_v2 \\
        --version enriched_v2 \\
        --notes "Best F1 so far"

    python scripts/model_store.py list
    python scripts/model_store.py promote --version enriched_v2
    python scripts/model_store.py export --version enriched_v2
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REGISTRY_DIR = Path("outputs/model_registry")
MANIFEST_FILE = "manifest.json"


# ── Register ─────────────────────────────────────────────────

def register(
    source_dir: str,
    version: str,
    notes: str = "",
    config_path: str = "config.yaml",
):
    """Snapshot a model experiment into the registry."""
    src = Path(source_dir)
    if not src.is_dir():
        log.error(f"Source directory not found: {src}")
        return

    dest = REGISTRY_DIR / version
    if dest.exists():
        log.error(f"Version '{version}' already registered at {dest}")
        return

    dest.mkdir(parents=True, exist_ok=True)

    # Copy model artifacts
    artifacts_copied = []
    for pattern in ["*.lgbm", "*.pt", "*.pth", "*.json", "*.png"]:
        for f in src.glob(pattern):
            shutil.copy2(f, dest / f.name)
            artifacts_copied.append(f.name)

    # Copy subdirectory GeoJSONs (stacked results)
    for sub in src.iterdir():
        if sub.is_dir():
            for gj in sub.glob("*.geojson"):
                shutil.copy2(gj, dest / gj.name)
                artifacts_copied.append(gj.name)

    # Extract metrics from training report
    metrics = {}
    report_path = dest / "training_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        m = report.get("metrics", {})
        metrics = {
            "balanced_accuracy": m.get("balanced_accuracy"),
            "average_precision": m.get("average_precision"),
            "f1_score": m.get("f1_score"),
            "precision_at_threshold": m.get("precision_at_threshold"),
            "recall_at_threshold": m.get("recall_at_threshold"),
            "n_features": report.get("feature_config", {}).get("n_features"),
        }

    # Write manifest
    manifest = {
        "version": version,
        "source_dir": str(src),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "artifacts": artifacts_copied,
        "metrics": metrics,
        "is_production": False,
    }
    with open(dest / MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"[OK] Registered '{version}' ({len(artifacts_copied)} files)")
    log.info(f"   Location: {dest}")

    # Try to notify
    try:
        from src.monitoring import Notifier
        notifier = Notifier.from_config(config_path)
        notifier.send_model_registered(version, metrics)
    except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")

    return manifest


# -- List -----------------------------------------------------

def list_versions():
    """List all registered model versions."""
    if not REGISTRY_DIR.exists():
        log.info("No model registry found.")
        return []

    versions = []
    for d in sorted(REGISTRY_DIR.iterdir()):
        manifest_path = d / MANIFEST_FILE
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            m = json.load(f)
        versions.append(m)

    if not versions:
        log.info("Registry is empty.")
        return []

    # Print summary table
    print(f"\n{'Version':<25} {'Prod':>4} {'Bal Acc':>8} {'Avg P':>8} "
          f"{'F1':>6} {'Files':>5}  {'Registered'}")
    print("-" * 90)
    for v in versions:
        met = v.get("metrics", {})
        prod = "*" if v.get("is_production") else ""
        ba = f"{met.get('balanced_accuracy', 0):.3f}" if met.get("balanced_accuracy") else "-"
        ap = f"{met.get('average_precision', 0):.3f}" if met.get("average_precision") else "-"
        f1 = f"{met.get('f1_score', 0):.3f}" if met.get("f1_score") else "-"
        n = len(v.get("artifacts", []))
        dt = v.get("registered_at", "")[:19]
        print(f"{v['version']:<25} {prod:>4} {ba:>8} {ap:>8} {f1:>6} {n:>5}  {dt}")

    print()
    return versions


# -- Promote --------------------------------------------------

def promote(version: str):
    """Mark a version as production (demotes any current production)."""
    target = REGISTRY_DIR / version / MANIFEST_FILE
    if not target.exists():
        log.error(f"Version '{version}' not found in registry")
        return

    # Demote current production
    for d in REGISTRY_DIR.iterdir():
        mp = d / MANIFEST_FILE
        if not mp.exists():
            continue
        with open(mp) as f:
            m = json.load(f)
        if m.get("is_production"):
            m["is_production"] = False
            with open(mp, "w") as f:
                json.dump(m, f, indent=2)
            log.info(f"  Demoted: {m['version']}")

    # Promote target
    with open(target) as f:
        m = json.load(f)
    m["is_production"] = True
    m["promoted_at"] = datetime.now(timezone.utc).isoformat()
    with open(target, "w") as f:
        json.dump(m, f, indent=2)

    log.info(f"[OK] Promoted '{version}' to production")


# -- Export ---------------------------------------------------

def export(version: str, output_path: str = None):
    """Pack a version into a .tar.gz archive."""
    src = REGISTRY_DIR / version
    if not src.exists():
        log.error(f"Version '{version}' not found")
        return

    if not output_path:
        output_path = f"outputs/{version}.tar.gz"

    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(src, arcname=version)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    log.info(f"[OK] Exported '{version}' -> {output_path} ({size_mb:.1f} MB)")
    return output_path


# -- CLI ------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Van Suraksha — Model Store")
    sub = parser.add_subparsers(dest="command", required=True)

    # register
    p_reg = sub.add_parser("register", help="Register a model version")
    p_reg.add_argument("--source", required=True, help="Source experiment dir")
    p_reg.add_argument("--version", required=True, help="Version name")
    p_reg.add_argument("--notes", default="", help="Optional notes")
    p_reg.add_argument("--config", default="config.yaml")

    # list
    sub.add_parser("list", help="List registered versions")

    # promote
    p_prom = sub.add_parser("promote", help="Promote version to production")
    p_prom.add_argument("--version", required=True)

    # export
    p_exp = sub.add_parser("export", help="Export version as tar.gz")
    p_exp.add_argument("--version", required=True)
    p_exp.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.command == "register":
        register(args.source, args.version, args.notes, args.config)
    elif args.command == "list":
        list_versions()
    elif args.command == "promote":
        promote(args.version)
    elif args.command == "export":
        export(args.version, args.output)


if __name__ == "__main__":
    main()
