"""
scripts/run_ablations.py
========================
Ablation study runner for VanAagni fire severity model.

Runs 6 experiments (baseline + 5 ablations) to quantify each architectural
component's contribution.  Results are saved per-ablation and as a summary.

Ablations:
  0. baseline       - Re-train with identical config (reproducibility check)
  1. no_film        - FiLM conditioning disabled (GroupNorm-only passthrough)
  2. single_frame   - T=1 temporal frame instead of T=3
  3. decoder_only   - Backbone fully frozen, no LLRD (decoder + FiLM only)
  4. no_spatial_aux - Spatial auxiliary encoder zeroed out
  5. binary         - 2-class fire/no-fire instead of 5-class severity

Usage:
    # Run all ablations sequentially
    python scripts/run_ablations.py

    # Run a specific ablation
    python scripts/run_ablations.py --ablation no_film

    # Smoke-test (2 batches per epoch, 2 epochs)
    python scripts/run_ablations.py --smoke-test

    # Resume from a specific ablation (skip completed ones)
    python scripts/run_ablations.py --resume-from decoder_only
"""

import sys, os, math, time, json, logging, argparse, copy, gc
from pathlib import Path

# -- Project root setup -------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

import yaml
import numpy as np
import torch
import torch.nn as nn

# -- Imports from the training codebase ----------------------------------------
from src.train.train_vanaagni import (
    load_config,
    setup_device,
    build_optimizer,
    build_scheduler,
    build_loss,
    train_one_epoch,
    validate,
    save_checkpoint,
    load_checkpoint,
    freeze_backbone,
    amp_context,
    make_scaler,
    _NoOpScaler,
)
from src.model.vanaagni import build_vanaagni
from src.model.change_head import FocalDiceLoss
from src.data.vanaagni_dataset import get_dataloaders, inspect_batch
from src.train.metrics import SegmentationMetrics, SEVERITY_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ablation")

# ==============================================================================
# Ablation Configuration
# ==============================================================================

# Reduced training schedule for ablations (vs 80 epochs for full training)
ABLATION_EPOCHS   = 30
ABLATION_PATIENCE = 12
BASE_CONFIG_PATH  = "models/vanaagni_v1.0/vanaagni_config.yaml"
OUTPUT_BASE       = Path("outputs/vanaagni/ablations")

ABLATION_ORDER = [
    "decoder_only",     # fastest (backbone frozen, ~1.2h)
    "no_film",          # ~3.7h
    "baseline",         # ~3.7h
    "single_frame",     # ~2.0h
    "no_spatial_aux",   # ~3.3h
    "binary",           # ~3.7h
]


# ==============================================================================
# Config Modifiers (per-ablation)
# ==============================================================================

def cfg_baseline(cfg):
    """Baseline: identical config, verify reproducibility."""
    cfg["training"]["epochs"] = ABLATION_EPOCHS
    cfg["training"]["early_stop_patience"] = ABLATION_PATIENCE
    return cfg


def cfg_no_film(cfg):
    """No FiLM: same config (patching happens at model level)."""
    cfg["training"]["epochs"] = ABLATION_EPOCHS
    cfg["training"]["early_stop_patience"] = ABLATION_PATIENCE
    return cfg


def cfg_single_frame(cfg):
    """Single Frame: T=1 instead of T=3."""
    cfg["model"]["num_frames"] = 1
    cfg["training"]["epochs"] = ABLATION_EPOCHS
    cfg["training"]["early_stop_patience"] = ABLATION_PATIENCE
    return cfg


def cfg_decoder_only(cfg):
    """Decoder Only: freeze entire backbone, no LLRD."""
    cfg["llrd"]["enabled"] = False
    cfg["llrd"]["detach_prefix"] = 0
    # Freeze backbone for all epochs (Phase 1 only)
    cfg["training"]["freeze_backbone_epochs"] = ABLATION_EPOCHS + 10
    cfg["training"]["epochs"] = ABLATION_EPOCHS
    cfg["training"]["early_stop_patience"] = ABLATION_PATIENCE
    return cfg


def cfg_no_spatial_aux(cfg):
    """No Spatial Aux: same config (patching happens at model level)."""
    cfg["training"]["epochs"] = ABLATION_EPOCHS
    cfg["training"]["early_stop_patience"] = ABLATION_PATIENCE
    return cfg


def cfg_binary(cfg):
    """Binary: 2-class fire/no-fire."""
    cfg["model"]["num_classes"] = 2
    cfg["training"]["loss_weights"] = [1.0, 50.0]
    cfg["training"]["epochs"] = ABLATION_EPOCHS
    cfg["training"]["early_stop_patience"] = ABLATION_PATIENCE
    return cfg


CONFIG_MODIFIERS = {
    "baseline":       cfg_baseline,
    "no_film":        cfg_no_film,
    "single_frame":   cfg_single_frame,
    "decoder_only":   cfg_decoder_only,
    "no_spatial_aux": cfg_no_spatial_aux,
    "binary":         cfg_binary,
}


# ==============================================================================
# Model Monkey-Patches
# ==============================================================================

def patch_no_film(model):
    """Replace FiLM conditioning with identity (GroupNorm only).

    Each FiLMDecoderBlock has a `film` attribute (FiLMBlock).
    We replace its forward to just apply GroupNorm, ignoring the
    conditioning vector entirely.  gamma/beta weights are unused.
    """
    for block in model.decoder_blocks:
        film = block.film
        # Replace forward: just apply GroupNorm, skip gamma/beta modulation
        original_norm = film.norm

        def _identity_forward(x, cond, _norm=original_norm):
            return _norm(x)

        film.forward = _identity_forward

    # Also freeze the conditioning modules (not needed without FiLM)
    for p in model.weather_enc.parameters():
        p.requires_grad = False
    for p in model.forecast_enc.parameters():
        p.requires_grad = False
    for p in model.cond_fuse.parameters():
        p.requires_grad = False

    log.info("[PATCH] No FiLM: all FiLMBlock.forward -> GroupNorm-only identity")


def patch_no_spatial_aux(model):
    """Replace SpatialAuxEncoder output with zero tensors.

    The SpatialAuxEncoder produces multi-scale features that are
    concatenated with skip connections in the decoder.  By zeroing them,
    we measure the contribution of terrain + landcover + burn-age.
    """
    out_ch = model.spatial_aux.scale_projs[0][0].out_channels  # 32

    def _zero_forward(terrain, landcover, burn_age, target_sizes):
        B = terrain.shape[0]
        device = terrain.device
        return [
            torch.zeros(B, out_ch, h, w, device=device)
            for (h, w) in target_sizes
        ]

    model.spatial_aux.forward = _zero_forward

    # Freeze spatial aux params (no grad needed)
    for p in model.spatial_aux.parameters():
        p.requires_grad = False

    log.info("[PATCH] No Spatial Aux: SpatialAuxEncoder -> zero tensors")


MODEL_PATCHES = {
    "baseline":       None,
    "no_film":        patch_no_film,
    "single_frame":   None,
    "decoder_only":   None,
    "no_spatial_aux": patch_no_spatial_aux,
    "binary":         None,
}


# ==============================================================================
# Batch Pre-processing (for single-frame ablation)
# ==============================================================================

def preprocess_batch_default(batch):
    """Default: no modification."""
    return batch


def preprocess_batch_single_frame(batch):
    """Slice HLS from T=3 to T=1 (keep only the first temporal frame)."""
    batch["hls"] = batch["hls"][:, :, :1, :, :]  # (B, 6, 1, H, W)
    return batch


BATCH_PREPROCESSORS = {
    "baseline":       preprocess_batch_default,
    "no_film":        preprocess_batch_default,
    "single_frame":   preprocess_batch_single_frame,
    "decoder_only":   preprocess_batch_default,
    "no_spatial_aux": preprocess_batch_default,
    "binary":         preprocess_batch_default,
}


# ==============================================================================
# Modified Training / Validation Loops (with batch preprocessing)
# ==============================================================================

def train_one_epoch_ablation(
    model, loader, optimizer, scheduler, scaler, loss_fn,
    device, metrics, epoch, cfg, use_amp=False,
    overfit_batches=0, grad_accum=1, batch_preprocess=None,
):
    """Training epoch with optional batch preprocessing."""
    model.train()
    metrics.reset()
    total_loss = 0.0
    n_batches = 0
    log_interval = cfg["training"].get("log_interval", 20)

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        if overfit_batches > 0 and step >= overfit_batches:
            break

        # Apply batch preprocessing (e.g., single-frame slicing)
        if batch_preprocess is not None:
            batch = batch_preprocess(batch)

        hls        = batch["hls"].to(device, non_blocking=True)
        weather    = batch["weather"].to(device, non_blocking=True)
        weather_7d = batch["weather_7d"].to(device, non_blocking=True)
        terrain    = batch["terrain"].to(device, non_blocking=True)
        burn_age   = batch["burn_age"].to(device, non_blocking=True)
        landcover  = batch["landcover"].to(device, non_blocking=True)
        label      = batch["label"].to(device, non_blocking=True)
        weight     = batch["weight"].to(device, non_blocking=True)

        with amp_context(device.type, use_amp):
            logits = model(hls, weather, weather_7d, terrain, burn_age, landcover)
            if logits.shape[-2:] != label.shape[-2:]:
                logits = nn.functional.interpolate(
                    logits, size=label.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            loss_raw = loss_fn(logits, label)
            valid_mask = (label >= 0)
            if valid_mask.any():
                mean_w = weight[valid_mask].mean()
                loss = loss_raw * mean_w
            else:
                loss = loss_raw
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * grad_accum
        n_batches += 1

        preds = logits.detach().argmax(dim=1)
        metrics.update(preds, label)

        if step % log_interval == 0:
            scores = metrics.compute()
            lr = optimizer.param_groups[-1]["lr"]
            miou = scores.get("mean_iou", 0)
            ff1 = scores.get("fire_f1", scores.get("f1_fg", 0))
            log.info(
                f"  E{epoch} [{step:4d}/{len(loader)}]  "
                f"loss={loss.item() * grad_accum:.4f}  "
                f"mIoU={miou:.4f}  fireF1={ff1:.4f}  "
                f"lr={lr:.2e}"
            )

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, metrics.compute()


@torch.no_grad()
def validate_ablation(
    model, loader, loss_fn, device, metrics,
    use_amp=False, threshold=0.15, batch_preprocess=None,
    max_batches=0,
):
    """Validation loop with optional batch preprocessing."""
    model.eval()
    metrics.reset()
    total_loss = 0.0
    n_batches = 0

    for step, batch in enumerate(loader):
        if max_batches > 0 and step >= max_batches:
            break
        if batch_preprocess is not None:
            batch = batch_preprocess(batch)

        hls        = batch["hls"].to(device, non_blocking=True)
        weather    = batch["weather"].to(device, non_blocking=True)
        weather_7d = batch["weather_7d"].to(device, non_blocking=True)
        terrain    = batch["terrain"].to(device, non_blocking=True)
        burn_age   = batch["burn_age"].to(device, non_blocking=True)
        landcover  = batch["landcover"].to(device, non_blocking=True)
        label      = batch["label"].to(device, non_blocking=True)

        with amp_context(device.type, use_amp):
            logits = model(hls, weather, weather_7d, terrain, burn_age, landcover)
            if logits.shape[-2:] != label.shape[-2:]:
                logits = nn.functional.interpolate(
                    logits, size=label.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            loss = loss_fn(logits, label)

        total_loss += loss.item()
        n_batches += 1

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        metrics.update(preds, label, probs=probs, threshold=threshold)

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, metrics.compute()


# ==============================================================================
# Main Ablation Runner
# ==============================================================================

def run_ablation(name, cfg, device, is_cuda, is_dml, use_amp, overfit_batches=0):
    """Run a single ablation experiment. Returns test results dict."""

    log.info(f"\n{'='*70}")
    log.info(f"  ABLATION: {name}")
    log.info(f"{'='*70}\n")

    t_cfg = cfg["training"]
    d_cfg = cfg["data"]
    num_classes = cfg["model"].get("num_classes", 5)
    binary_mode = (num_classes == 2)

    # -- Output directory ------------------------------------------------------
    out_dir = OUTPUT_BASE / name
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -- DataLoaders -----------------------------------------------------------
    num_workers = 0  # DirectML: no multiprocessing workers
    train_dl, val_dl, test_dl = get_dataloaders(
        manifest_path   = d_cfg["manifest_path"],
        batch_size      = int(t_cfg["batch_size"]),
        num_workers     = num_workers,
        fire_oversample = float(d_cfg.get("fire_oversample", 5.0)),
        exclude_tiers   = d_cfg.get("exclude_tiers", None),
        min_fire_pixels = int(d_cfg.get("min_fire_pixels", 25)),
        pin_memory      = is_cuda,
        binary_mode     = binary_mode,
    )
    log.info(f"Train: {len(train_dl)} batches | Val: {len(val_dl)} | Test: {len(test_dl)}")

    # -- Model -----------------------------------------------------------------
    model = build_vanaagni(cfg)

    # Apply model monkey-patches
    patch_fn = MODEL_PATCHES.get(name)
    if patch_fn is not None:
        patch_fn(model)

    model = model.to(device)

    # -- LLRD / Detach Prefix --------------------------------------------------
    llrd_cfg = cfg.get("llrd", {})
    use_llrd = llrd_cfg.get("enabled", False)
    llrd_decay = float(llrd_cfg.get("decay", 0.65)) if use_llrd else 0.0
    single_phase = use_llrd

    detach_prefix = int(llrd_cfg.get("detach_prefix", 0))
    if detach_prefix > 0 and use_llrd:
        model.set_detach_prefix(detach_prefix)

    # -- Loss ------------------------------------------------------------------
    loss_fn = build_loss(cfg, device)

    # -- Metrics ---------------------------------------------------------------
    eval_threshold = float(t_cfg.get("eval_threshold", 0.15))
    train_metrics = SegmentationMetrics(num_classes=num_classes)
    val_metrics = SegmentationMetrics(num_classes=num_classes)

    # -- Training params -------------------------------------------------------
    total_epochs = int(t_cfg["epochs"])
    freeze_epochs = int(t_cfg.get("freeze_backbone_epochs", 10))
    grad_accum = int(t_cfg.get("grad_accum_steps", 1))
    patience = int(t_cfg.get("early_stop_patience", 30))
    batch_preprocess = BATCH_PREPROCESSORS.get(name, preprocess_batch_default)

    best_iou = 0.0
    best_epoch = 0
    no_improve_count = 0

    start_time = time.time()

    # ==========================================================================
    # LLRD Training Path (single phase) -- used by baseline, no_film,
    # single_frame, no_spatial_aux
    # ==========================================================================
    if single_phase:
        log.info(f"[{name}] LLRD Training (epochs 1-{total_epochs})")

        optimizer = build_optimizer(model, cfg, phase=2, llrd_decay=llrd_decay)
        sched_steps = max(1, len(train_dl) // grad_accum)
        scheduler = build_scheduler(
            optimizer, cfg, sched_steps,
            total_training_epochs=total_epochs,
        )
        scaler = make_scaler(device.type, use_amp)

        for epoch in range(1, total_epochs + 1):
            t0 = time.time()

            train_loss, train_scores = train_one_epoch_ablation(
                model, train_dl, optimizer, scheduler, scaler, loss_fn,
                device, train_metrics, epoch, cfg, use_amp,
                overfit_batches=overfit_batches, grad_accum=grad_accum,
                batch_preprocess=batch_preprocess,
            )

            val_loss, val_scores = validate_ablation(
                model, val_dl, loss_fn, device, val_metrics,
                use_amp=use_amp, threshold=eval_threshold,
                batch_preprocess=batch_preprocess,
                max_batches=overfit_batches,
            )

            elapsed = time.time() - t0
            val_miou = val_scores.get("mean_iou", 0)
            val_ff1 = val_scores.get("fire_f1", val_scores.get("f1_fg", 0))

            log.info(
                f"[{name}] E{epoch:3d}/{total_epochs}  [{elapsed:.0f}s]  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"mIoU={val_miou:.4f}  fireF1={val_ff1:.4f}"
            )

            if val_miou > best_iou:
                best_iou = val_miou
                best_epoch = epoch
                no_improve_count = 0
                # Save with lora_mode=True for lightweight checkpoint
                save_checkpoint(
                    model, optimizer, epoch, best_iou, cfg,
                    ckpt_dir / "best.pth", tag="BEST",
                    lora_mode=(detach_prefix > 0),
                )
            else:
                no_improve_count += 1

            if patience > 0 and no_improve_count >= patience:
                log.info(f"[{name}] Early stopping at epoch {epoch}")
                break

    # ==========================================================================
    # Decoder-Only Training Path (Phase 1: frozen backbone, no LLRD)
    # ==========================================================================
    else:
        phase1_end = min(freeze_epochs, total_epochs)
        log.info(f"[{name}] Phase 1: Frozen Backbone (epochs 1-{phase1_end})")

        freeze_backbone(model)
        optimizer = build_optimizer(model, cfg, phase=1)
        scheduler = build_scheduler(optimizer, cfg, len(train_dl), phase=1)
        scaler = make_scaler(device.type, use_amp)

        for epoch in range(1, phase1_end + 1):
            t0 = time.time()

            train_loss, train_scores = train_one_epoch_ablation(
                model, train_dl, optimizer, scheduler, scaler, loss_fn,
                device, train_metrics, epoch, cfg, use_amp,
                overfit_batches=overfit_batches, grad_accum=grad_accum,
                batch_preprocess=batch_preprocess,
            )

            val_loss, val_scores = validate_ablation(
                model, val_dl, loss_fn, device, val_metrics,
                use_amp=use_amp, threshold=eval_threshold,
                batch_preprocess=batch_preprocess,
                max_batches=overfit_batches,
            )

            elapsed = time.time() - t0
            val_miou = val_scores.get("mean_iou", 0)
            val_ff1 = val_scores.get("fire_f1", val_scores.get("f1_fg", 0))

            log.info(
                f"[{name}] E{epoch:3d}/{total_epochs}  [{elapsed:.0f}s]  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"mIoU={val_miou:.4f}  fireF1={val_ff1:.4f}  (frozen)"
            )

            if val_miou > best_iou:
                best_iou = val_miou
                best_epoch = epoch
                no_improve_count = 0
                save_checkpoint(
                    model, optimizer, epoch, best_iou, cfg,
                    ckpt_dir / "best.pth", tag="BEST",
                    lora_mode=True,  # lightweight ckpt to avoid MemoryError
                )
            else:
                no_improve_count += 1

            if patience > 0 and no_improve_count >= patience:
                log.info(f"[{name}] Early stopping at epoch {epoch}")
                break

    # ==========================================================================
    # Test Evaluation
    # ==========================================================================
    log.info(f"\n[{name}] Test Evaluation ...")

    # Reload best checkpoint
    best_ckpt = ckpt_dir / "best.pth"
    if best_ckpt.exists():
        load_checkpoint(model, None, str(best_ckpt), device)

    test_metrics = SegmentationMetrics(num_classes=num_classes)
    test_loss, test_scores = validate_ablation(
        model, test_dl, loss_fn, device, test_metrics,
        use_amp=use_amp, threshold=eval_threshold,
        batch_preprocess=batch_preprocess,
        max_batches=overfit_batches,
    )

    total_time = time.time() - start_time

    # Normalize metric names for comparison
    fire_f1 = test_scores.get("fire_f1", test_scores.get("f1_fg", 0))
    fire_iou = test_scores.get("fire_iou", test_scores.get("iou_fg", 0))
    fire_prec = test_scores.get("fire_precision", test_scores.get("precision_fg", 0))
    fire_rec = test_scores.get("fire_recall", test_scores.get("recall_fg", 0))

    log.info(f"\n[{name}] TEST RESULTS:")
    log.info(f"  Loss:          {test_loss:.4f}")
    log.info(f"  Mean IoU:      {test_scores.get('mean_iou', 0):.4f}")
    log.info(f"  Pixel Acc:     {test_scores.get('pixel_accuracy', 0):.4f}")
    log.info(f"  Fire F1:       {fire_f1:.4f}")
    log.info(f"  Fire IoU:      {fire_iou:.4f}")
    log.info(f"  Fire Prec:     {fire_prec:.4f}")
    log.info(f"  Fire Recall:   {fire_rec:.4f}")
    log.info(f"  Best Epoch:    {best_epoch}")
    log.info(f"  Training Time: {total_time/3600:.1f}h")

    # Save results
    results = {
        "ablation":        name,
        "best_epoch":      best_epoch,
        "best_val_iou":    best_iou,
        "test_loss":       test_loss,
        "test_scores":     test_scores,
        "training_time_s": total_time,
        "num_epochs_run":  best_epoch + no_improve_count,
        # Normalized metrics for easy comparison
        "fire_f1":         fire_f1,
        "fire_iou":        fire_iou,
        "fire_precision":  fire_prec,
        "fire_recall":     fire_rec,
        "mean_iou":        test_scores.get("mean_iou", 0),
        "pixel_accuracy":  test_scores.get("pixel_accuracy", 0),
        "config":          cfg,
    }

    results_path = out_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"  Results saved: {results_path}")

    # Cleanup GPU memory
    del model, optimizer, scaler, loss_fn
    del train_dl, val_dl, test_dl
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    return results


def kill_stale_processes():
    """Kill stale python processes that may hog DirectML VRAM.

    On Windows with DirectML, zombie python processes keep GPU memory
    allocated even after crashing.  We kill all python.exe EXCEPT
    our own PID to reclaim VRAM before each ablation run.
    """
    import subprocess
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.strip().split("\n"):
            if "python.exe" in line.lower():
                parts = line.strip('"').split('","')
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1].strip('"'))
                        if pid != my_pid:
                            subprocess.run(
                                ["taskkill", "/F", "/PID", str(pid)],
                                capture_output=True, timeout=5,
                            )
                            log.info(f"  Killed stale python PID {pid}")
                    except (ValueError, subprocess.TimeoutExpired):
                        pass
    except Exception as e:
        log.warning(f"  Could not clean stale processes: {e}")


def collect_summary(ablation_names):
    """Collect all ablation results into a summary dict."""
    summary = {"ablations": {}, "comparison_table": []}

    for name in ablation_names:
        results_path = OUTPUT_BASE / name / "test_results.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)

            summary["ablations"][name] = {
                "fire_f1":        results.get("fire_f1", 0),
                "fire_iou":       results.get("fire_iou", 0),
                "fire_precision": results.get("fire_precision", 0),
                "fire_recall":    results.get("fire_recall", 0),
                "mean_iou":       results.get("mean_iou", 0),
                "pixel_accuracy": results.get("pixel_accuracy", 0),
                "best_epoch":     results.get("best_epoch", 0),
                "training_time_h": results.get("training_time_s", 0) / 3600,
            }

    # Build comparison table (sorted by fire_f1 descending)
    rows = []
    for name, metrics in summary["ablations"].items():
        rows.append({"name": name, **metrics})
    rows.sort(key=lambda r: r["fire_f1"], reverse=True)
    summary["comparison_table"] = rows

    # Add delta from baseline
    baseline_f1 = summary["ablations"].get("baseline", {}).get("fire_f1", 0)
    for row in summary["comparison_table"]:
        row["delta_fire_f1"] = row["fire_f1"] - baseline_f1

    return summary


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="VanAagni Ablation Study")
    parser.add_argument(
        "--ablation", default=None,
        help="Run a specific ablation (e.g., 'no_film'). Default: run all.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick smoke-test: 2 batches/epoch, 2 epochs per ablation.",
    )
    parser.add_argument(
        "--resume-from", default=None,
        help="Resume from a specific ablation (skip completed ones).",
    )
    parser.add_argument(
        "--config", default=BASE_CONFIG_PATH,
        help="Path to base config YAML.",
    )
    args = parser.parse_args()

    # Determine which ablations to run
    if args.ablation:
        if args.ablation not in CONFIG_MODIFIERS:
            log.error(f"Unknown ablation: {args.ablation}")
            log.error(f"Choose from: {list(CONFIG_MODIFIERS.keys())}")
            sys.exit(1)
        ablation_names = [args.ablation]
    else:
        ablation_names = list(ABLATION_ORDER)

    # Resume: skip completed ablations
    if args.resume_from:
        if args.resume_from not in ablation_names:
            log.error(f"Resume target '{args.resume_from}' not in ablation list")
            sys.exit(1)
        idx = ablation_names.index(args.resume_from)
        ablation_names = ablation_names[idx:]

    # Smoke-test overrides
    overfit_batches = 2 if args.smoke_test else 0
    global ABLATION_EPOCHS, ABLATION_PATIENCE
    if args.smoke_test:
        ABLATION_EPOCHS = 2
        ABLATION_PATIENCE = 2

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    log.info(f"VanAagni Ablation Study")
    log.info(f"  Config:     {args.config}")
    log.info(f"  Ablations:  {ablation_names}")
    log.info(f"  Epochs:     {ABLATION_EPOCHS}")
    log.info(f"  Patience:   {ABLATION_PATIENCE}")
    log.info(f"  Smoke-test: {args.smoke_test}")
    log.info(f"  Output:     {OUTPUT_BASE}")

    # Setup device once (shared across all ablations)
    base_cfg = load_config(args.config)
    device, is_cuda, is_dml, use_amp = setup_device(base_cfg)

    # Run ablations sequentially
    all_results = {}
    for i, name in enumerate(ablation_names):
        log.info(f"\n{'#'*70}")
        log.info(f"# Ablation {i+1}/{len(ablation_names)}: {name}")
        log.info(f"{'#'*70}")

        # Skip if already completed
        existing = OUTPUT_BASE / name / "test_results.json"
        if existing.exists() and not args.smoke_test:
            log.info(f"  Skipping '{name}' -- results already exist at {existing}")
            log.info(f"  Delete to re-run.")
            with open(existing) as f:
                all_results[name] = json.load(f)
            continue

        # Kill stale processes (DirectML VRAM cleanup)
        if is_dml:
            log.info("  Cleaning stale processes ...")
            kill_stale_processes()
            time.sleep(2)

        # Deep copy config and apply modifications
        cfg = copy.deepcopy(base_cfg)
        cfg = CONFIG_MODIFIERS[name](cfg)

        # Update output paths for this ablation
        cfg["paths"]["output_dir"] = str(OUTPUT_BASE / name)
        cfg["paths"]["checkpoint_dir"] = str(OUTPUT_BASE / name / "checkpoints")

        try:
            results = run_ablation(
                name, cfg, device, is_cuda, is_dml, use_amp,
                overfit_batches=overfit_batches,
            )
            all_results[name] = results
        except Exception as e:
            log.error(f"  FAILED: {name} -- {e}")
            import traceback
            log.error(traceback.format_exc())
            all_results[name] = {"error": str(e), "ablation": name}

        # Force garbage collection between runs
        gc.collect()

    # Generate summary
    log.info(f"\n{'='*70}")
    log.info(f"  ABLATION STUDY COMPLETE")
    log.info(f"{'='*70}")

    summary = collect_summary(ABLATION_ORDER)
    summary_path = OUTPUT_BASE / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Summary saved: {summary_path}")

    # Print comparison table
    log.info(f"\n{'Ablation':<20s} {'Fire F1':>8s} {'Fire IoU':>9s} "
             f"{'Prec':>6s} {'Rec':>6s} {'mIoU':>7s} {'Epoch':>6s} {'Time':>6s}")
    log.info("-" * 75)
    for row in summary.get("comparison_table", []):
        delta = row.get("delta_fire_f1", 0)
        delta_str = f"({delta:+.3f})" if row["name"] != "baseline" else ""
        log.info(
            f"{row['name']:<20s} {row['fire_f1']:>8.4f} {row['fire_iou']:>9.4f} "
            f"{row['fire_precision']:>6.3f} {row['fire_recall']:>6.3f} "
            f"{row['mean_iou']:>7.4f} {row['best_epoch']:>6d} "
            f"{row['training_time_h']:>5.1f}h {delta_str}"
        )

    # Also include the original E15 results for reference
    orig_results_path = Path("outputs/vanaagni/test_results.json")
    if orig_results_path.exists():
        with open(orig_results_path) as f:
            orig = json.load(f)
        orig_scores = orig.get("test_scores", {})
        log.info(f"\n{'original_E15':<20s} "
                 f"{orig_scores.get('fire_f1', 0):>8.4f} "
                 f"{orig_scores.get('fire_iou', 0):>9.4f} "
                 f"{orig_scores.get('fire_precision', 0):>6.3f} "
                 f"{orig_scores.get('fire_recall', 0):>6.3f} "
                 f"{orig_scores.get('mean_iou', 0):>7.4f} "
                 f"{orig.get('best_epoch', 0):>6d} "
                 f"{'N/A':>6s}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log.error(f"FATAL: {traceback.format_exc()}")
        raise
