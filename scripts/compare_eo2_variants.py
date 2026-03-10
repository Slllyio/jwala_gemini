"""
EO-2.0 Variant A/B Comparison
==============================

Runs the validation set through two (or more) checkpoints and prints a
side-by-side table of segmentation metrics + inference throughput.

Typical use:

    python scripts/compare_eo2_variants.py \\
        --config  config.yaml \\
        --ckpts   outputs/checkpoints/best_detect.pth \\
                  outputs/checkpoints/eo2_600M_best.pth \\
        --labels  "Prithvi-100M" "EO-2.0-600M"

The script writes a markdown + CSV summary to  outputs/reports/eo2_compare_<date>.md
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import torch
import yaml
import numpy as np

from torch.utils.data import DataLoader

from src.model.full_model import build_model
from src.train.dataset import ForestChangeDataset
from src.train.metrics import SegmentationMetrics

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device(cfg: dict):
    """Mirrors the device-selection logic in train.py."""
    try:
        import torch_directml  # type: ignore[import]
        return torch_directml.device(), "DirectML"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA/ROCm"
    return torch.device("cpu"), "CPU"


def load_checkpoint(ckpt_path: str, cfg: dict, device) -> torch.nn.Module:
    """Build model and load weights from *ckpt_path*."""
    model = build_model(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt.get("model_state_dict"))
    if state is None:
        raise KeyError(f"No model state found in {ckpt_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    log.info(
        f"  Loaded {ckpt_path}  "
        f"epoch={ckpt.get('epoch','?')}  "
        f"missing={len(missing)}  unexpected={len(unexpected)}"
    )
    return model


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device,
    metrics: SegmentationMetrics,
    threshold: float = 0.5,
) -> tuple[dict, float]:
    """
    Run full validation pass.  Returns (metric_dict, throughput_imgs_per_sec).
    """
    model.eval()
    metrics.reset()
    n_images    = 0
    t_start     = time.perf_counter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images, mode="detect")
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)
        metrics.update(preds, labels, probs=probs, threshold=threshold)
        n_images += images.size(0)

    elapsed   = time.perf_counter() - t_start
    throughput = n_images / max(elapsed, 1e-6)
    return metrics.compute(), throughput


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--ckpts",     nargs="+", required=True,
                        help="One or more checkpoint paths to compare")
    parser.add_argument("--labels",    nargs="+",
                        help="Short labels for each checkpoint (positional match)")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Foreground probability threshold (matches eval_threshold)")
    parser.add_argument("--workers",   type=int, default=0)
    parser.add_argument("--out-dir",   default="outputs/reports")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # EO-2.0 overrides when use_eo2=True — mirrors load_config in train.py
    if cfg.get("model", {}).get("use_eo2", False):
        eo2_overrides = cfg.get("training", {}).get("eo2", {})
        if eo2_overrides:
            cfg["training"].update(eo2_overrides)

    device, backend = get_device(cfg)
    log.info(f"Device: {device}  ({backend})")

    labels = args.labels or [f"run_{i}" for i in range(len(args.ckpts))]
    if len(labels) < len(args.ckpts):
        labels += [f"run_{i}" for i in range(len(labels), len(args.ckpts))]

    # ── Validation dataset (same as train.py) ─────────────────────────────────
    t_cfg       = cfg["training"]
    processed   = cfg["paths"]["processed_dir"]
    val_ds      = ForestChangeDataset(os.path.join(processed, "val.csv"), augment=False)
    val_loader  = DataLoader(
        val_ds, batch_size=t_cfg["batch_size"],
        shuffle=False, num_workers=args.workers,
    )
    log.info(f"Validation set: {len(val_ds)} samples  (batch={t_cfg['batch_size']})")

    # ── Per-checkpoint evaluation ──────────────────────────────────────────────
    results: list[dict] = []
    met  = SegmentationMetrics(num_classes=cfg["model"]["num_classes"])

    for ckpt_path, label in zip(args.ckpts, labels):
        log.info(f"\n── Evaluating: {label}  ({ckpt_path}) ──")
        try:
            model = load_checkpoint(ckpt_path, cfg, device)
        except Exception as exc:
            log.error(f"  Failed to load {ckpt_path}: {exc}")
            continue

        scores, tput = run_inference(model, val_loader, device, met, args.threshold)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        row = {
            "label":        label,
            "checkpoint":   str(Path(ckpt_path).name),
            "iou_fg":       round(scores.get("iou_fg",   0.0), 4),
            "f1_fg":        round(scores.get("f1_fg",    0.0), 4),
            "precision_fg": round(scores.get("precision_fg", 0.0), 4),
            "recall_fg":    round(scores.get("recall_fg",    0.0), 4),
            "throughput":   round(tput, 1),
        }
        results.append(row)
        log.info(
            f"  IoU={row['iou_fg']:.4f}  F1={row['f1_fg']:.4f}  "
            f"Prec={row['precision_fg']:.4f}  Rec={row['recall_fg']:.4f}  "
            f"Speed={row['throughput']:.1f} img/s"
        )

    if not results:
        log.error("No successful runs — exiting.")
        return

    # ── Print table ───────────────────────────────────────────────────────────
    cols   = ["label", "iou_fg", "f1_fg", "precision_fg", "recall_fg", "throughput"]
    header = ["Model", "IoU ↑", "F1 ↑", "Precision ↑", "Recall ↑", "Img/s ↑"]
    widths = [max(len(h), max(len(str(r[c])) for r in results))
              for h, c in zip(header, cols)]

    def row_str(values):
        return " | ".join(str(v).ljust(w) for v, w in zip(values, widths))

    sep   = "-+-".join("-" * w for w in widths)
    lines = [
        "",
        "## EO-2.0 Variant Comparison",
        "",
        row_str(header),
        sep,
    ] + [row_str([r[c] for c in cols]) for r in results] + [""]

    print("\n".join(lines))

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import date
    stem = f"eo2_compare_{date.today().isoformat()}"

    md_path = out_dir / f"{stem}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Markdown saved: {md_path}")

    csv_path = out_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    log.info(f"CSV saved:      {csv_path}")

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info(f"JSON saved:     {json_path}")


if __name__ == "__main__":
    main()
