"""
Training Script
===============
Fine-tunes Prithvi-100M for forest change detection on the Fatehgarh AOI.

Usage:
    # Change detection training
    python src/train/train.py --config config.yaml --task detect

    # Temporal prediction training
    python src/train/train.py --config config.yaml --task predict

    # Resume from checkpoint
    python src/train/train.py --config config.yaml --task detect --resume outputs/checkpoints/best.pth
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import yaml
import time
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# ROCm on Windows presents as torch.cuda — no special import needed

from src.model.full_model import build_model
from src.model.change_head import CombinedLoss
from src.train.dataset import ForestChangeDataset, TemporalPredictionDataset
from src.train.metrics import SegmentationMetrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # ── Phase 3: merge EO-2.0 training overrides when use_eo2=true ──────────
    if cfg.get("model", {}).get("use_eo2", False):
        eo2_overrides = cfg.get("training", {}).get("eo2", {})
        if eo2_overrides:
            cfg["training"].update(eo2_overrides)
            log.info("[EO2] Applied EO-2.0 training overrides:")
            for k, v in eo2_overrides.items():
                log.info(f"   training.{k} = {v}")

    return cfg


def get_optimizer(model: nn.Module, cfg: dict):
    t_cfg = cfg["training"]
    opt_type = t_cfg.get("optimizer", "adamw").lower()

    # Separate encoder and head parameters with differential LR
    encoder_params = list(model.encoder.parameters())
    head_params = (list(model.change_head.parameters()) +
                   list(model.prediction_head.parameters()))

    param_groups = [
        {"params": encoder_params, "lr": t_cfg["lr"] * 0.1},  # encoder: 10× lower LR
        {"params": head_params,    "lr": t_cfg["lr"]},         # head: full LR
    ]

    if opt_type == "sgd":
        # SGD runs fully natively on DirectML — no CPU fallbacks (unlike AdamW lerp)
        return torch.optim.SGD(
            param_groups,
            momentum=t_cfg.get("momentum", 0.9),
            nesterov=t_cfg.get("nesterov", True),
            weight_decay=t_cfg.get("weight_decay", 1e-4),
        )
    else:
        # AdamW — best for fine-tuning transformers; fully native on ROCm (no CPU fallback)
        return torch.optim.AdamW(param_groups, weight_decay=t_cfg.get("weight_decay", 0.05))



def get_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    t_cfg = cfg["training"]
    warmup_steps = t_cfg["warmup_epochs"] * steps_per_epoch
    total_steps = t_cfg["epochs"] * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, scaler, loss_fn,
                device, task, metrics, epoch, cfg, use_amp=False):
    model.train()
    total_loss = 0.0
    metrics.reset()
    t_cfg = cfg["training"]
    log_interval = t_cfg["log_interval"]
    mode = "detect" if task == "detect" else "predict"

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device.type, enabled=use_amp):
            if mode == "detect":
                logits = model(images, mode="detect")           # (B, C, H, W)
                loss = loss_fn(logits, labels)
                preds = logits.argmax(dim=1)
            else:
                # prediction task: images is a sequence
                T = images.shape[1]
                frame_list = [images[:, t] for t in range(T)]  # list of (B, C, H, W)
                risk_logits = model(frame_list, mode="predict")        # (B, 1, H, W) raw logits
                risk_logits = risk_logits.squeeze(1)
                loss = nn.BCEWithLogitsLoss()(risk_logits, labels.float())
                preds = (risk_logits > 0.0).long()  # threshold on logits (0 ≡ 0.5 sigmoid)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        metrics.update(preds, labels)

        if step % log_interval == 0:
            scores = metrics.compute()
            lr = optimizer.param_groups[1]["lr"]
            log.info(
                f"  Epoch {epoch} [{step}/{len(loader)}] "
                f"loss={loss.item():.4f} "
                f"IoU={scores.get('iou_fg', 0):.4f} "
                f"F1={scores.get('f1_fg', 0):.4f} "
                f"lr={lr:.2e}"
            )

    return total_loss / len(loader), metrics.compute()


@torch.no_grad()
def validate(model, loader, loss_fn, device, task, metrics, threshold=0.5):
    model.eval()
    total_loss = 0.0
    metrics.reset()
    mode = "detect" if task == "detect" else "predict"

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if mode == "detect":
            logits = model(images, mode="detect")
            loss = loss_fn(logits, labels)
            probs = torch.softmax(logits, dim=1)   # (B, C, H, W)
            preds = logits.argmax(dim=1)            # argmax fallback
        else:
            T = images.shape[1]
            frame_list = [images[:, t] for t in range(T)]
            risk_logits = model(frame_list, mode="predict").squeeze(1)  # raw logits
            loss = nn.BCEWithLogitsLoss()(risk_logits, labels.float())
            probs = None
            preds = (risk_logits > 0.0).long()  # logit 0 ≡ sigmoid 0.5

        total_loss += loss.item()
        # Pass probs + threshold so fg class is detected at lower confidence
        metrics.update(preds, labels,
                       probs=probs if mode == "detect" else None,
                       threshold=threshold)

    return total_loss / len(loader), metrics.compute()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--task", choices=["detect", "predict"], default="detect")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--tag",    default=None, help="Optional run label (logged only)")
    # ── DDP args (set automatically by torchrun; ignored for single-GPU) ──────
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1,
                        help="DDP local rank (set by torchrun; -1 = single-GPU / CPU)")
    args = parser.parse_args()

    if args.tag:
        log.info(f"Run tag: {args.tag}")

    cfg = load_config(args.config)
    t_cfg = cfg["training"]
    processed_dir = cfg["paths"]["processed_dir"]
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Device selection: DirectML > ROCm/CUDA (+ optional DDP) > CPU ─────────
    # DDP is activated when torchrun sets LOCAL_RANK env-var or --local-rank flag.
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    ddp_enabled = local_rank >= 0 and torch.cuda.is_available()

    if ddp_enabled:
        import torch.distributed as dist
        # nccl is NVIDIA-only; ROCm/HIP on Windows requires gloo
        ddp_backend = "gloo" if (torch.version.hip is not None) else "nccl"
        dist.init_process_group(backend=ddp_backend)
        torch.cuda.set_device(local_rank)
        device  = torch.device("cuda", local_rank)
        is_cuda = True
        is_dml  = False
        is_ddp  = True
        is_main = (local_rank == 0)   # only rank-0 writes checkpoints / logs
        log.info(f"[DDP] rank={local_rank} / world={dist.get_world_size()}  device={device}  backend={ddp_backend}")
    else:
        is_main = True
        is_ddp  = False
        try:
            import torch_directml  # type: ignore[import]
            device = torch_directml.device()
            is_dml  = True
            is_cuda = False
            gpu_name = torch_directml.device_name(0)
            log.info(f"Device: {device}")
            log.info(f"   GPU: {gpu_name} (via DirectML)")
        except ImportError:
            is_dml = False
            if torch.cuda.is_available():
                device  = torch.device("cuda")
                is_cuda = True
                log.info(f"Device: {device}")
                log.info(f"   GPU:     {torch.cuda.get_device_name(0)}")
                log.info(f"   VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                log.info(f"   Backend: {'ROCm/HIP' if torch.version.hip else 'CUDA'}")
            else:
                device  = torch.device("cpu")
                is_cuda = False
                log.warning("⚠️  No GPU detected — running on CPU (will be slow)")

    use_amp = t_cfg.get("amp", False) and is_cuda  # AMP not supported on DirectML
    log.info(f"   AMP: {'enabled (fp16)' if use_amp else 'disabled'}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    aug_cfg = cfg.get("augmentation", {})
    oversample = aug_cfg.get("oversample_positives", True)

    if args.task == "detect":
        DatasetClass = ForestChangeDataset
        train_ds = DatasetClass(
            os.path.join(processed_dir, "train.csv"),
            augment=True,
            aug_cfg=aug_cfg,
            oversample_positives=oversample,
        )
        val_ds = DatasetClass(
            os.path.join(processed_dir, "val.csv"),
            augment=False,
        )
    else:
        DatasetClass = TemporalPredictionDataset
        train_ds = DatasetClass(
            os.path.join(processed_dir, "train.csv"),
            num_time_steps=cfg["prediction"]["num_time_steps"],
            augment=True,
            aug_cfg=aug_cfg,
        )
        val_ds = DatasetClass(
            os.path.join(processed_dir, "val.csv"),
            num_time_steps=cfg["prediction"]["num_time_steps"],
        )

    num_workers = t_cfg.get("num_workers", 0) if is_cuda else 0
    prefetch_factor = t_cfg.get("prefetch_factor", 2) if num_workers > 0 else None
    pin_mem = t_cfg.get("pin_memory", False) and is_cuda
    train_loader = DataLoader(
        train_ds, batch_size=t_cfg["batch_size"],
        shuffle=True, num_workers=num_workers,
        pin_memory=pin_mem, drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_ds, batch_size=t_cfg["batch_size"],
        shuffle=False, num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
    )

    log.info(f"📊 Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)

    if args.resume:
        log.info(f"🔄 Resuming from: {args.resume}")
        # weights_only=True prevents arbitrary pickle code execution —
        # never load checkpoints from untrusted sources with weights_only=False.
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        # Support both old key name ('model_state_dict') and new ('model_state')
        state = ckpt.get("model_state", ckpt.get("model_state_dict"))
        if state is None:
            raise KeyError("Checkpoint has neither 'model_state' nor 'model_state_dict' key")
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info(f"   Resumed from epoch {ckpt.get('epoch', '?')}  "
                 f"missing={len(missing)} unexpected={len(unexpected)}")

    # ── Distributed Data Parallel wrap (skipped for single-GPU / DirectML) ────
    if ddp_enabled:
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data.distributed import DistributedSampler
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler   = DistributedSampler(val_ds,   shuffle=False)
        # Rebuild loaders with samplers for DDP
        train_loader = DataLoader(
            train_ds, batch_size=t_cfg["batch_size"],
            sampler=train_sampler, num_workers=num_workers,
            pin_memory=pin_mem, drop_last=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=prefetch_factor,
        )
        val_loader = DataLoader(
            val_ds, batch_size=t_cfg["batch_size"],
            sampler=val_sampler, num_workers=num_workers,
            pin_memory=pin_mem,
            persistent_workers=(num_workers > 0),
            prefetch_factor=prefetch_factor,
        )
        log.info(f"[DDP] Model wrapped. Effective batch = {t_cfg['batch_size']} × {dist.get_world_size()}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    if args.task == "detect":
        # Use config loss_weights for strong positive-class emphasis.
        # Dataset-computed weights are near [1.0, 1.0] when labels are ~50/50,
        # which causes class collapse — the model learns to predict all-background.
        lw = t_cfg.get("loss_weights", [1.0, 50.0])
        gamma = t_cfg.get("focal_gamma", 2.0)
        class_weights = torch.tensor(lw, dtype=torch.float32).to(device)
        log.info(f"Class weights (from config): {class_weights.tolist()} | Focal gamma: {gamma}")
        loss_fn = CombinedLoss(class_weights=class_weights, gamma=gamma, dice_weight=0.5)
    else:
        loss_fn = nn.BCEWithLogitsLoss()  # numerically stable: fuses sigmoid+BCE (log-sum-exp)

    eval_threshold = t_cfg.get("eval_threshold", 0.15)
    log.info(f"Eval threshold: {eval_threshold} (fg probability cutoff)")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = get_optimizer(model, cfg)
    scaler = GradScaler(device.type if is_cuda else "cpu", enabled=use_amp)
    scheduler = get_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    # ── EO-2.0 special settings ───────────────────────────────────────────────
    use_eo2           = cfg.get("model", {}).get("use_eo2", False)
    grad_accum_steps  = int(t_cfg.get("grad_accum_steps", 1))  # >1 only for EO2
    freeze_enc_epochs = int(t_cfg.get("freeze_encoder_epochs", 0))
    if use_eo2:
        log.info(f"[EO2] Gradient accumulation steps : {grad_accum_steps}")
        log.info(f"[EO2] Encoder frozen for first    : {freeze_enc_epochs} epoch(s)")
        log.info(f"[EO2] Effective batch size        : "
                 f"{t_cfg['batch_size'] * grad_accum_steps}")

    # ── TensorBoard ───────────────────────────────────────────────────────────
    tb_dir = os.path.join(cfg["paths"]["output_dir"], "tensorboard", args.task)
    writer = SummaryWriter(log_dir=tb_dir)
    log.info(f"📈 TensorBoard: tensorboard --logdir {tb_dir}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    train_metrics = SegmentationMetrics(num_classes=cfg["model"]["num_classes"])
    val_metrics = SegmentationMetrics(num_classes=cfg["model"]["num_classes"])

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_iou = 0.0
    best_epoch = 0
    log.info(f"\n{'='*60}")
    log.info(f"  Task: {args.task.upper()} | Epochs: {t_cfg['epochs']}")
    log.info(f"{'='*60}\n")

    for epoch in range(1, t_cfg["epochs"] + 1):
        t0 = time.time()

        # ── EO-2.0: phase-in encoder unfreezing ──────────────────────────────────
        if use_eo2 and freeze_enc_epochs > 0:
            if epoch == 1:
                # Freeze encoder: only heads train
                for p in model.encoder.parameters():
                    p.requires_grad = False
                log.info(f"[EO2] Encoder frozen for epochs 1–{freeze_enc_epochs}")
            elif epoch == freeze_enc_epochs + 1:
                # Unfreeze encoder: full fine-tune
                for p in model.encoder.parameters():
                    p.requires_grad = True
                log.info(f"[EO2] Encoder unfrozen at epoch {epoch} — full fine-tune engaged")

        # Train
        train_loss, train_scores = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, loss_fn,
            device, args.task, train_metrics, epoch, cfg, use_amp
        )

        # Validate
        val_loss, val_scores = validate(
            model, val_loader, loss_fn, device, args.task, val_metrics,
            threshold=eval_threshold
        )

        elapsed = time.time() - t0
        log.info(
            f"Epoch {epoch:3d}/{t_cfg['epochs']} "
            f"[{elapsed:.1f}s] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_IoU={val_scores.get('iou_fg', 0):.4f} "
            f"val_F1={val_scores.get('f1_fg', 0):.4f}"
        )

        # TensorBoard logging (rank-0 only)
        if is_main:
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            for k, v in val_scores.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        # Save best checkpoint (rank-0 only — prevents file corruption from concurrent writes)
        current_iou = val_scores.get("iou_fg", 0)
        if is_main and current_iou > best_iou:
            best_iou = current_iou
            best_epoch = epoch
            ckpt_path = ckpt_dir / f"best_{args.task}.pth"
            # In DDP the real weights live inside model.module
            state = model.module.state_dict() if ddp_enabled else model.state_dict()
            torch.save({
                "epoch": epoch,
                "model_state": state,
                "optimizer_state": optimizer.state_dict(),
                "best_iou": best_iou,
                "config": cfg,
            }, ckpt_path)
            log.info(f"  💾 Best checkpoint saved: {ckpt_path} (IoU={best_iou:.4f})")

        # Periodic checkpoint every 10 epochs (rank-0 only)
        if is_main and epoch % 10 == 0:
            state = model.module.state_dict() if ddp_enabled else model.state_dict()
            torch.save({
                "epoch": epoch,
                "model_state": state,
            }, ckpt_dir / f"epoch_{epoch:03d}_{args.task}.pth")

    if ddp_enabled:
        import torch.distributed as dist
        dist.destroy_process_group()

    if is_main:
        writer.close()
        log.info(f"\n🏆 Best epoch: {best_epoch} | Best val IoU: {best_iou:.4f}")
        log.info(f"✅ Training complete. Checkpoints in: {ckpt_dir}")


if __name__ == "__main__":
    main()
