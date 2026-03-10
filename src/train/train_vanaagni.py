"""
src/train/train_vanaagni.py
============================
Training loop for VanAgni multi-modal fire prediction model.

Two-phase training schedule:
  Phase 1 (warm-up): Backbone frozen, only decoder + FiLM + weather heads train.
  Phase 2 (fine-tune): Backbone unfrozen with differential learning rates.

Usage:
    # Full training run
    python src/train/train_vanaagni.py --config models/vanaagni/vanaagni_config.yaml

    # Resume from checkpoint
    python src/train/train_vanaagni.py --config models/vanaagni/vanaagni_config.yaml \
                                       --resume outputs/vanaagni/checkpoints/best.pth

    # Quick overfit test on 2 batches
    python src/train/train_vanaagni.py --config models/vanaagni/vanaagni_config.yaml --overfit-batches 2
"""

import sys, os, math, time, json, logging, argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext

# AMP imports -- guarded for DirectML / CPU where autocast device is invalid
try:
    from torch.amp import GradScaler, autocast as _autocast
    _HAS_AMP = True
except ImportError:
    _HAS_AMP = False


def amp_context(device_type: str, enabled: bool):
    """Return autocast context manager that is safe on any device."""
    if enabled and _HAS_AMP and device_type in ("cuda", "cpu"):
        return _autocast(device_type, enabled=True)
    return nullcontext()


def make_scaler(device_type: str, enabled: bool):
    """Create GradScaler only when AMP is actually usable."""
    if enabled and _HAS_AMP and device_type in ("cuda",):
        return GradScaler(device_type, enabled=True)
    # Return a dummy scaler that just passes through
    return _NoOpScaler()


class _NoOpScaler:
    """Drop-in GradScaler replacement for non-CUDA devices."""
    def scale(self, loss):    return loss
    def unscale_(self, opt):  pass
    def step(self, opt):      opt.step()
    def update(self):         pass

from src.model.vanaagni import build_vanaagni
from src.model.change_head import FocalDiceLoss
from src.data.vanaagni_dataset import get_dataloaders, inspect_batch
from src.train.metrics import SegmentationMetrics, RegressionMetrics, SEVERITY_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

def load_config(path: str) -> dict:
    """Load YAML config and validate required keys."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    required_sections = ["model", "training", "data", "paths"]
    for s in required_sections:
        if s not in cfg:
            raise KeyError(f"Config missing required section: '{s}'")
    return cfg


# =============================================================================
# Device Setup
# =============================================================================

def _kill_stale_gpu_holders() -> None:
    """Kill orphaned Python processes still holding the DirectML GPU.

    On Windows, a crashed training run may leave a zombie python.exe that
    locks the GPU device.  The next run then gets a "device removed" error.
    This checks for *other* python processes with high VRAM use and warns.
    """
    try:
        import subprocess, json as _json
        # tasklist doesn't expose VRAM; just check for other python.exe
        # processes that loaded torch_directml (heuristic: >500 MB RSS).
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        my_pid = os.getpid()
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        other_pids = []
        for line in lines:
            parts = line.strip('"').split('","')
            if len(parts) >= 2:
                pid = int(parts[1])
                if pid != my_pid:
                    # Check RSS (field 4, in KB)
                    if len(parts) >= 5:
                        mem_str = parts[4].replace(",", "").replace(" K", "").strip('"')
                        try:
                            mem_kb = int(mem_str)
                            if mem_kb > 500_000:  # > 500 MB
                                other_pids.append((pid, mem_kb // 1024))
                        except ValueError:
                            pass
        if other_pids:
            pids_str = ", ".join(f"PID {p} ({m} MB)" for p, m in other_pids)
            log.warning(f"[GPU] Found other large python.exe processes: {pids_str}")
            log.warning("  These may hold the GPU device. If training fails, "
                        "kill them with: taskkill /F /PID <pid>")
    except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ignored exception: {e}")  # Non-critical; don't block startup


def setup_device(cfg: dict) -> tuple:
    """
    Detect best available device: DirectML -> ROCm/CUDA -> CPU.
    Returns (device, is_cuda, is_dml, use_amp).
    """
    t_cfg = cfg["training"]

    try:
        import torch_directml  # type: ignore[import]
        _kill_stale_gpu_holders()
        device = torch_directml.device()
        is_dml  = True
        is_cuda = False
        gpu_name = torch_directml.device_name(0)
        log.info(f"Device: {device}  ({gpu_name} via DirectML)")
    except ImportError:
        is_dml = False
        if torch.cuda.is_available():
            device  = torch.device("cuda")
            is_cuda = True
            log.info(f"Device: cuda  ({torch.cuda.get_device_name(0)})")
            log.info(f"   VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            log.info(f"   Backend: {'ROCm/HIP' if torch.version.hip else 'CUDA'}")
        else:
            device  = torch.device("cpu")
            is_cuda = False
            log.warning("No GPU detected -- running on CPU (will be very slow)")

    # AMP only works on CUDA / ROCm, not DirectML
    use_amp = t_cfg.get("amp", False) and is_cuda
    log.info(f"   AMP: {'enabled (bf16)' if use_amp else 'disabled'}")

    return device, is_cuda, is_dml, use_amp


# =============================================================================
# Optimizer / Scheduler
# =============================================================================

def build_optimizer(model, cfg: dict, phase: int = 1, llrd_decay: float = 0.0):
    """
    Build AdamW optimizer with differential learning rates.

    Phase 1:   only non-backbone params (backbone is frozen).
    Phase 2:   full model with LR groups from model.get_param_groups().
    LLRD mode: per-layer LR decay for backbone (all params trainable).
    """
    t_cfg = cfg["training"]
    base_lr      = float(t_cfg["lr"])
    weight_decay = float(t_cfg.get("weight_decay", 0.05))

    if phase == 1:
        # Only trainable (non-backbone) params
        params = [p for p in model.parameters() if p.requires_grad]
        param_groups = [{"params": params, "lr": base_lr}]
        log.info(f"[Phase 1] Optimizer: {len(params)} trainable params, lr={base_lr:.1e}")
    else:
        # Differential LR for backbone + decoder + FiLM
        param_groups = model.get_param_groups(base_lr, llrd_decay=llrd_decay)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int,
                    phase: int = 1, total_training_epochs: int = 0):
    """
    Linear warm-up -> cosine annealing to min_lr.

    Phase 1:    short warm-up (2 epochs), schedule over freeze_epochs.
    Phase 2:    warm-up + cosine over remaining epochs after freeze.
    LLRD/LoRA:  pass total_training_epochs to schedule over the full run.
    """
    t_cfg = cfg["training"]

    if total_training_epochs > 0:
        # Single-phase modes (LLRD / LoRA): schedule over full run
        warmup_epochs = t_cfg.get("warmup_epochs", 3)
        total_epochs  = total_training_epochs
    elif phase == 1:
        warmup_epochs = min(2, t_cfg.get("freeze_backbone_epochs", 10))
        total_epochs  = t_cfg.get("freeze_backbone_epochs", 10)
    else:
        warmup_epochs = t_cfg.get("warmup_epochs", 3)
        total_epochs  = t_cfg["epochs"] - t_cfg.get("freeze_backbone_epochs", 10)

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = max(total_epochs * steps_per_epoch, warmup_steps + 1)
    min_lr_frac  = float(t_cfg.get("min_lr_fraction", 0.01))

    def lr_lambda(step):
        if step < warmup_steps:
            return max(0.01, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_frac, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Loss
# =============================================================================

class MaskedSmoothL1Loss(nn.Module):
    """SmoothL1 loss that masks out NaN pixels in regression targets.

    Also applies fire-weighted loss: pixels with dNBR > 0 get higher weight
    to counteract the overwhelming number of background (dNBR=0) pixels.
    """

    def __init__(self, fire_weight: float = 10.0, beta: float = 0.1):
        super().__init__()
        self.fire_weight = fire_weight
        self.beta = beta  # SmoothL1 transition point

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, H, W) float predicted dNBR
            target: (B, H, W) float ground truth dNBR (NaN = ignore)
        """
        # Mask out NaN pixels
        valid = torch.isfinite(target)
        if not valid.any():
            return pred.sum() * 0.0  # no valid pixels, return zero loss

        p = pred[valid]
        t = target[valid]

        # SmoothL1 per-pixel
        loss = nn.functional.smooth_l1_loss(p, t, reduction="none", beta=self.beta)

        # Fire-weighted: upweight pixels where target > 0 (actual burn)
        fire_mask = t > 0
        weight = torch.ones_like(loss)
        weight[fire_mask] = self.fire_weight
        loss = (loss * weight).sum() / weight.sum()

        return loss


class MultiTaskLoss(nn.Module):
    """Multi-task loss: binary fire detection (FocalBCE) + dNBR regression (SmoothL1).

    Decomposes fire severity into two sub-problems:
      1. Binary fire/no-fire: FocalBCE on ALL valid pixels (handles 142:1 imbalance)
      2. dNBR regression: SmoothL1 ONLY on fire pixels (no background dominance)

    Total loss = bce_weight * FocalBCE + reg_weight * MaskedSmoothL1
    """

    def __init__(self, bce_weight: float = 1.0, reg_weight: float = 1.0,
                 focal_gamma: float = 2.0, beta: float = 0.1,
                 fire_threshold: float = 0.01):
        super().__init__()
        self.bce_weight = bce_weight
        self.reg_weight = reg_weight
        self.focal_gamma = focal_gamma
        self.beta = beta
        self.fire_threshold = fire_threshold

    def forward(self, pred_dict: dict, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_dict: {"dnbr": (B,H,W) float, "fire_logit": (B,H,W) float}
            target:    (B,H,W) float ground truth dNBR (NaN=ignore)
        """
        dnbr_pred = pred_dict["dnbr"]
        fire_logit = pred_dict["fire_logit"]

        # Mask NaN pixels
        valid = torch.isfinite(target)
        if not valid.any():
            return dnbr_pred.sum() * 0.0

        # Binary fire target: dNBR > threshold
        fire_target = (target[valid] > self.fire_threshold).float()

        # 1. Focal BCE on ALL valid pixels (learns fire/no-fire boundary)
        bce = nn.functional.binary_cross_entropy_with_logits(
            fire_logit[valid], fire_target, reduction="none"
        )
        pt = torch.exp(-bce)  # probability of correct class
        focal_bce = ((1 - pt) ** self.focal_gamma * bce).mean()

        # 2. SmoothL1 ONLY on fire pixels (learns severity within burns)
        fire_mask = valid & (target > self.fire_threshold)
        if fire_mask.any():
            reg_loss = nn.functional.smooth_l1_loss(
                dnbr_pred[fire_mask], target[fire_mask],
                reduction="mean", beta=self.beta
            )
        else:
            reg_loss = dnbr_pred.sum() * 0.0

        return self.bce_weight * focal_bce + self.reg_weight * reg_loss


def build_loss(cfg: dict, device: torch.device) -> nn.Module:
    """Build loss function: multi-task regression or classification (FocalDice)."""
    t_cfg = cfg["training"]
    num_classes = cfg["model"].get("num_classes", 5)

    if num_classes == 1:
        # Multi-task loss: FocalBCE (fire detection) + SmoothL1 (dNBR regression)
        mt_cfg = cfg.get("multitask", {})
        bce_w  = float(mt_cfg.get("bce_weight", 1.0))
        reg_w  = float(mt_cfg.get("reg_weight", 1.0))
        gamma  = float(mt_cfg.get("focal_gamma", 2.0))
        beta   = float(t_cfg.get("smooth_l1_beta", 0.1))
        thresh = float(t_cfg.get("fire_dnbr_threshold", 0.01))
        log.info(f"Loss: MultiTask  bce_w={bce_w}  reg_w={reg_w}  "
                 f"focal_gamma={gamma}  beta={beta}  threshold={thresh}")
        return MultiTaskLoss(bce_weight=bce_w, reg_weight=reg_w,
                            focal_gamma=gamma, beta=beta,
                            fire_threshold=thresh)

    # Classification mode
    loss_weights    = t_cfg.get("loss_weights", [1.0, 8.0])
    gamma           = t_cfg.get("focal_gamma", 2.0)
    dice_weight     = t_cfg.get("dice_weight", 0.5)
    label_smoothing = float(t_cfg.get("label_smoothing", 0.0))

    class_w = torch.tensor(loss_weights, dtype=torch.float32).to(device)
    log.info(
        f"Loss: FocalDice  class_w={class_w.tolist()}  gamma={gamma}  "
        f"dice_w={dice_weight}  label_smoothing={label_smoothing}"
    )

    return FocalDiceLoss(class_weights=class_w, gamma=gamma, dice_weight=dice_weight,
                         label_smoothing=label_smoothing)


# =============================================================================
# Training Epoch
# =============================================================================

def train_one_epoch(
    model, loader, optimizer, scheduler, scaler, loss_fn,
    device, metrics, epoch, cfg, use_amp=False,
    overfit_batches: int = 0, grad_accum: int = 1,
    regression: bool = False,
):
    """
    One training epoch.

    Handles:
      - Multi-input dict batches from VanAgniDataset
      - Per-pixel weight multiplication on loss
      - Gradient accumulation
      - AMP mixed precision
      - Regression mode (continuous dNBR) or classification mode
    """
    model.train()
    metrics.reset()
    total_loss = 0.0
    n_batches  = 0
    log_interval = cfg["training"].get("log_interval", 20)

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        if overfit_batches > 0 and step >= overfit_batches:
            break

        # -- Move tensors to device ----------------------------------------
        hls        = batch["hls"].to(device, non_blocking=True)
        weather    = batch["weather"].to(device, non_blocking=True)
        weather_7d = batch["weather_7d"].to(device, non_blocking=True)
        terrain    = batch["terrain"].to(device, non_blocking=True)
        burn_age   = batch["burn_age"].to(device, non_blocking=True)
        landcover  = batch["landcover"].to(device, non_blocking=True)
        label      = batch["label"].to(device, non_blocking=True)
        weight     = batch["weight"].to(device, non_blocking=True)

        use_cached = "cached_feat" in batch

        # -- Forward pass --------------------------------------------------
        with amp_context(device.type, use_amp):
            if use_cached:
                cached_feat = batch["cached_feat"].to(device, non_blocking=True)
                output = model.forward_cached(
                    cached_feat, weather, weather_7d, terrain, burn_age, landcover,
                )
            else:
                output = model(hls, weather, weather_7d, terrain, burn_age, landcover)

            if regression:
                # Multi-task output: dict with "dnbr" and "fire_logit"
                dnbr = output["dnbr"]             # (B, H, W)
                fire_logit = output["fire_logit"] # (B, H, W)
                if dnbr.shape[-2:] != label.shape[-2:]:
                    sz = label.shape[-2:]
                    dnbr = nn.functional.interpolate(
                        dnbr.unsqueeze(1), size=sz,
                        mode="bilinear", align_corners=False,
                    ).squeeze(1)
                    fire_logit = nn.functional.interpolate(
                        fire_logit.unsqueeze(1), size=sz,
                        mode="bilinear", align_corners=False,
                    ).squeeze(1)

                # MultiTaskLoss: FocalBCE + masked SmoothL1
                loss_raw = loss_fn({"dnbr": dnbr, "fire_logit": fire_logit}, label)

                # Per-pixel tier weight: multiply by mean weight of valid pixels
                valid_mask = torch.isfinite(label)
                if valid_mask.any():
                    mean_w = weight[valid_mask].mean()
                    loss = loss_raw * mean_w
                else:
                    loss = loss_raw
            else:
                # output is (B, C, H, W) -- class logits
                if output.shape[-2:] != label.shape[-2:]:
                    output = nn.functional.interpolate(
                        output, size=label.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )

                loss_raw = loss_fn(output, label)
                valid_mask = (label >= 0)
                if valid_mask.any():
                    mean_w = weight[valid_mask].mean()
                    loss = loss_raw * mean_w
                else:
                    loss = loss_raw

            loss = loss / grad_accum

        # -- Backward + step -----------------------------------------------
        # TDR recovery: DirectML may throw RuntimeError when the GPU kernel
        # exceeds the Windows TDR timeout (~2 s).  Rather than crashing the
        # entire run we skip the batch, clear the grad graph, and continue.
        try:
            scaler.scale(loss).backward()
        except RuntimeError as exc:
            if "device" in str(exc).lower() or "tdr" in str(exc).lower():
                log.warning(f"[TDR] GPU timeout on step {step} — skipping batch "
                            f"({exc!s:.120})")
                optimizer.zero_grad(set_to_none=True)
                continue
            raise  # re-raise non-TDR errors

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * grad_accum
        n_batches  += 1

        # -- Metrics -------------------------------------------------------
        if regression:
            metrics.update(dnbr.detach(), label)
        else:
            preds = output.detach().argmax(dim=1)
            metrics.update(preds, label)

        # -- Logging -------------------------------------------------------
        if step % log_interval == 0:
            scores = metrics.compute()
            lr = optimizer.param_groups[-1]["lr"]
            if regression:
                mae  = scores.get("mae", 0)
                ff1  = scores.get("fire_f1", 0)
                pred_max = float(dnbr.detach().max())
                pred_mean = float(dnbr.detach().mean())
                log.info(
                    f"  E{epoch} [{step:4d}/{len(loader)}]  "
                    f"loss={loss.item() * grad_accum:.4f}  "
                    f"MAE={mae:.4f}  fireF1={ff1:.4f}  "
                    f"pred_max={pred_max:.4f}  pred_mean={pred_mean:.5f}  "
                    f"lr={lr:.2e}"
                )
            else:
                miou = scores.get("mean_iou", 0)
                ff1  = scores.get("fire_f1", scores.get("f1_fg", 0))
                log.info(
                    f"  E{epoch} [{step:4d}/{len(loader)}]  "
                    f"loss={loss.item() * grad_accum:.4f}  "
                    f"mIoU={miou:.4f}  fireF1={ff1:.4f}  "
                    f"lr={lr:.2e}"
                )

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, metrics.compute()


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def validate(
    model, loader, loss_fn, device, metrics,
    use_amp=False, threshold=0.15, regression=False,
):
    """
    Validation loop.

    Classification mode: uses probability threshold for fire class.
    Regression mode: computes MAE/RMSE/R2 and threshold-based fire detection.
    """
    model.eval()
    metrics.reset()
    total_loss = 0.0
    n_batches  = 0

    # Diagnostic tracking for regression
    pred_vals = [] if regression else None

    for batch in loader:
        hls        = batch["hls"].to(device, non_blocking=True)
        weather    = batch["weather"].to(device, non_blocking=True)
        weather_7d = batch["weather_7d"].to(device, non_blocking=True)
        terrain    = batch["terrain"].to(device, non_blocking=True)
        burn_age   = batch["burn_age"].to(device, non_blocking=True)
        landcover  = batch["landcover"].to(device, non_blocking=True)
        label      = batch["label"].to(device, non_blocking=True)

        use_cached = "cached_feat" in batch

        with amp_context(device.type, use_amp):
            if use_cached:
                cached_feat = batch["cached_feat"].to(device, non_blocking=True)
                output = model.forward_cached(
                    cached_feat, weather, weather_7d, terrain, burn_age, landcover,
                )
            else:
                output = model(hls, weather, weather_7d, terrain, burn_age, landcover)

            if regression:
                dnbr = output["dnbr"]
                fire_logit = output["fire_logit"]
                if dnbr.shape[-2:] != label.shape[-2:]:
                    sz = label.shape[-2:]
                    dnbr = nn.functional.interpolate(
                        dnbr.unsqueeze(1), size=sz,
                        mode="bilinear", align_corners=False,
                    ).squeeze(1)
                    fire_logit = nn.functional.interpolate(
                        fire_logit.unsqueeze(1), size=sz,
                        mode="bilinear", align_corners=False,
                    ).squeeze(1)
                loss = loss_fn({"dnbr": dnbr, "fire_logit": fire_logit}, label)
            else:
                if output.shape[-2:] != label.shape[-2:]:
                    output = nn.functional.interpolate(
                        output, size=label.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                loss = loss_fn(output, label)

        total_loss += loss.item()
        n_batches  += 1

        if regression:
            metrics.update(dnbr, label)
            # Sample prediction stats (every 10th batch to save memory)
            if n_batches % 10 == 1:
                with torch.no_grad():
                    p = dnbr.detach().cpu().numpy().flatten()
                    pred_vals.append(p[::100])  # subsample 1/100
        else:
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            metrics.update(preds, label, probs=probs, threshold=threshold)

    avg_loss = total_loss / max(n_batches, 1)
    scores = metrics.compute()

    # Log prediction distribution diagnostics for regression
    if regression and pred_vals:
        import numpy as _np
        all_p = _np.concatenate(pred_vals)
        log.info(
            f"  [VAL DIAG] pred dNBR: mean={all_p.mean():.5f} "
            f"max={all_p.max():.5f} p99={_np.percentile(all_p, 99):.5f} "
            f"p95={_np.percentile(all_p, 95):.5f} "
            f">0.01={int((all_p > 0.01).sum())}/{len(all_p)} "
            f">0.05={int((all_p > 0.05).sum())}/{len(all_p)}"
        )

    return avg_loss, scores


# =============================================================================
# Backbone Freeze / Unfreeze
# =============================================================================

def freeze_backbone(model):
    """Freeze all backbone parameters."""
    for p in model.backbone.parameters():
        p.requires_grad = False
    n_frozen = sum(p.numel() for p in model.backbone.parameters())
    log.info(f"[FROZEN] Backbone: {n_frozen:,} params frozen")


def unfreeze_backbone(model):
    """Unfreeze all backbone parameters for Phase 2 fine-tuning."""
    for p in model.backbone.parameters():
        p.requires_grad = True
    n_unfrozen = sum(p.numel() for p in model.backbone.parameters())
    log.info(f"[UNFROZEN] Backbone: {n_unfrozen:,} params now trainable")


# =============================================================================
# Checkpoint Save / Load
# =============================================================================

def save_checkpoint(model, optimizer, epoch, best_iou, cfg, path: Path,
                    tag: str = "", weights_only: bool = False,
                    lora_mode: bool = False):
    """Save training checkpoint.

    lora_mode=True saves only trainable (LoRA + decoder + FiLM) params,
    reducing checkpoint from ~11 GB to ~100 MB.

    weights_only=True skips optimizer/scheduler state.
    """
    import gc
    if lora_mode:
        # Save only trainable params directly from named_parameters()
        # IMPORTANT: avoid model.state_dict() which copies ALL 646M params
        # and causes MemoryError on RAM-constrained systems
        trainable_state = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                trainable_state[n] = p.detach().cpu()
        state = {
            "epoch":           epoch,
            "model_state":     trainable_state,
            "best_iou":        best_iou,
            "config":          cfg,
            "lora_mode":       True,
        }
    else:
        state = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": {} if weights_only else optimizer.state_dict(),
            "best_iou":        best_iou,
            "config":          cfg,
        }
    gc.collect()
    n_keys = len(state["model_state"])
    torch.save(state, path)
    del state; gc.collect()
    size_tag = f" [lora:{n_keys}keys]" if lora_mode else (
        " [weights-only]" if weights_only else "")
    log.info(f"  Checkpoint saved: {path}  (IoU={best_iou:.4f}) {tag}{size_tag}")


def load_checkpoint(model, optimizer, path: str, device):
    """Resume training from checkpoint.

    Handles num_classes mismatch gracefully: if the checkpoint's classifier
    layer has a different number of output channels (e.g. 5-class -> 2-class),
    those keys are filtered out and the new classifier is randomly initialized.
    """
    log.info(f"Resuming from: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    state = ckpt.get("model_state", ckpt.get("model_state_dict"))
    if state is None:
        raise KeyError("Checkpoint has neither 'model_state' nor 'model_state_dict'")

    # Filter out classifier keys with shape mismatch (num_classes change)
    model_state = model.state_dict()
    filtered_state = {}
    skipped = []
    for k, v in state.items():
        if k in model_state and v.shape != model_state[k].shape:
            skipped.append(f"{k}: ckpt={list(v.shape)} vs model={list(model_state[k].shape)}")
        else:
            filtered_state[k] = v

    if skipped:
        log.info(f"  Skipped {len(skipped)} shape-mismatched keys (new head):")
        for s in skipped:
            log.info(f"    {s}")

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    log.info(f"  Loaded: epoch={ckpt.get('epoch', '?')}  "
             f"best_iou={ckpt.get('best_iou', '?')}  "
             f"missing={len(missing)}  unexpected={len(unexpected)}")

    if optimizer and "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            log.info("  Optimizer state restored")
        except Exception as e:
            log.warning(f"  Could not restore optimizer state: {e}")

    return ckpt.get("epoch", 0), ckpt.get("best_iou", 0.0)


def _load_decoder_warmstart(model, ckpt_path: str, device) -> tuple:
    """Load only decoder/FiLM/skip_proj weights from a pre-LoRA checkpoint.

    Used to warm-start the LoRA training with the decoder weights learned
    during Phase 1 (frozen backbone).  Backbone state from the checkpoint
    is ignored since LoRA starts from fresh pretrained + adapters.
    """
    log.info(f"[LoRA warm-start] Loading decoder weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", {}))

    # Filter: keep only non-backbone keys
    decoder_state = {k: v for k, v in state.items()
                     if not k.startswith("backbone.")}

    if decoder_state:
        missing, unexpected = model.load_state_dict(decoder_state, strict=False)
        n_loaded = len(decoder_state) - len(unexpected)
        log.info(f"  Loaded {n_loaded} decoder/FiLM keys "
                 f"(skipped {len([k for k in state if k.startswith('backbone.')])} "
                 f"backbone keys)")
    else:
        log.warning("  No decoder keys found in checkpoint")

    return ckpt.get("epoch", 0), 0.0  # Reset best_iou for LoRA run


# =============================================================================
# Main Training
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train VanAgni fire prediction model")
    parser.add_argument("--config",  default="models/vanaagni/vanaagni_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--resume",  default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--conus-encoder", default=None,
                        help="CONUS pre-trained encoder.pth (backbone weights only)")
    parser.add_argument("--tag",     default=None, help="Optional run label")
    parser.add_argument("--overfit-batches", type=int, default=0,
                        help="If >0, overfit on N batches (debugging)")
    args = parser.parse_args()

    if args.tag:
        log.info(f"Run tag: {args.tag}")

    # -- Config -------------------------------------------------------------
    cfg   = load_config(args.config)
    t_cfg = cfg["training"]
    d_cfg = cfg["data"]
    p_cfg = cfg["paths"]

    ckpt_dir = Path(p_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -- File logging (persistent, survives pipe failures) ------------------
    out_dir = Path(p_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_name = f"train_{args.tag}.log" if args.tag else "train.log"
    fh = logging.FileHandler(out_dir / log_name, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    log.info(f"Logging to file: {out_dir / log_name}")

    # -- Device -------------------------------------------------------------
    device, is_cuda, is_dml, use_amp = setup_device(cfg)

    # -- DataLoaders --------------------------------------------------------
    num_workers = int(d_cfg.get("num_workers", 4)) if is_cuda else 0
    num_classes = cfg["model"].get("num_classes", 5)
    binary_mode = (num_classes == 2)
    regression  = (num_classes == 1)

    cache_dir = d_cfg.get("cache_dir", None)
    if cache_dir:
        log.info(f"[Cache] Backbone feature cache: {cache_dir}")

    train_dl, val_dl, test_dl = get_dataloaders(
        manifest_path   = d_cfg["manifest_path"],
        batch_size      = int(t_cfg["batch_size"]),
        num_workers     = num_workers,
        fire_oversample = float(d_cfg.get("fire_oversample", 5.0)),
        exclude_tiers   = d_cfg.get("exclude_tiers", None),
        min_fire_pixels = int(d_cfg.get("min_fire_pixels", 25)),
        pin_memory      = is_cuda,
        binary_mode     = binary_mode,
        regression      = regression,
        cache_dir       = cache_dir,
    )

    log.info(f"Train batches: {len(train_dl)}  |  Val batches: {len(val_dl)}")

    # Quick batch inspection
    sample_batch = next(iter(val_dl))
    inspect_batch(sample_batch)

    # -- Model --------------------------------------------------------------
    model = build_vanaagni(cfg)

    # -- CONUS encoder transfer (if provided) --------------------------------
    if args.conus_encoder:
        log.info(f"[CONUS] Loading pre-trained encoder: {args.conus_encoder}")
        conus_state = torch.load(args.conus_encoder, map_location="cpu",
                                 weights_only=False)
        # Match keys and load (strict=False since decoder/FiLM won't match)
        model_sd = model.state_dict()
        loaded, skipped = 0, 0
        for k, v in conus_state.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                model_sd[k] = v
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(model_sd, strict=False)
        del conus_state, model_sd
        log.info(f"[CONUS] Loaded {loaded} backbone keys, skipped {skipped}")

    model = model.to(device)

    # -- LoRA injection (if configured) -------------------------------------
    lora_cfg = cfg.get("lora", {})
    use_lora = lora_cfg.get("enabled", False)
    if use_lora:
        n_lora = model.inject_lora(
            rank           = int(lora_cfg.get("rank", 16)),
            alpha          = int(lora_cfg.get("alpha", 32)),
            dropout        = float(lora_cfg.get("dropout", 0.05)),
            target_modules = lora_cfg.get("target_modules", ["attn.qkv", "attn.proj"]),
        )
        log.info(f"[LoRA] {n_lora:,} adapter parameters injected")

    # -- LLRD config --------------------------------------------------------
    llrd_cfg  = cfg.get("llrd", {})
    use_llrd  = llrd_cfg.get("enabled", False)
    llrd_decay = float(llrd_cfg.get("decay", 0.65)) if use_llrd else 0.0

    # Single-phase mode: LLRD or LoRA bypass Phase 1/2 split
    single_phase = use_llrd or use_lora

    # -- Loss ---------------------------------------------------------------
    loss_fn = build_loss(cfg, device)

    # -- Metrics ------------------------------------------------------------
    eval_threshold = float(t_cfg.get("eval_threshold", 0.15))
    if regression:
        fire_thresh   = float(t_cfg.get("fire_dnbr_threshold", 0.05))
        train_metrics = RegressionMetrics(fire_threshold=fire_thresh)
        val_metrics   = RegressionMetrics(fire_threshold=fire_thresh)
    else:
        train_metrics = SegmentationMetrics(num_classes=num_classes)
        val_metrics   = SegmentationMetrics(num_classes=num_classes)

    # -- TensorBoard --------------------------------------------------------
    tb_dir = Path(p_cfg.get("output_dir", "outputs/vanaagni")) / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    log.info(f"TensorBoard: tensorboard --logdir {tb_dir}")

    # -- Training phases ----------------------------------------------------
    total_epochs         = int(t_cfg["epochs"])
    freeze_epochs        = int(t_cfg.get("freeze_backbone_epochs", 10))
    grad_accum           = int(t_cfg.get("grad_accum_steps", 1))
    save_every           = int(t_cfg.get("save_every_epochs", 10))
    patience             = int(t_cfg.get("early_stop_patience", 30))

    best_iou   = -1.0  # -1 so even fire_f1=0 triggers first save
    best_epoch = 0
    start_epoch = 1
    no_improve_count = 0

    # -- Resume / warm-start ------------------------------------------------
    if args.resume:
        if use_lora:
            # LoRA mode: load decoder/FiLM weights only
            start_epoch, best_iou = _load_decoder_warmstart(
                model, args.resume, device
            )
            start_epoch += 1
            log.info(f"Resuming (LoRA warm-start) from epoch {start_epoch}")
        elif single_phase:
            # LLRD: load model weights only (optimizer state won't match)
            start_epoch, best_iou = load_checkpoint(
                model, None, args.resume, device
            )
            # If classifier head was re-initialized (num_classes changed),
            # reset epoch and best_iou since metrics are incomparable
            ckpt_cfg = torch.load(args.resume, map_location="cpu",
                                  weights_only=False).get("config", {})
            ckpt_nc = ckpt_cfg.get("model", {}).get("num_classes", num_classes)
            if ckpt_nc != num_classes:
                log.info(f"  num_classes changed ({ckpt_nc} -> {num_classes}): "
                         f"resetting epoch=1, best_iou=0")
                start_epoch = 0
                best_iou = 0.0
            start_epoch += 1
            log.info(f"Resuming from epoch {start_epoch}")
        else:
            # Legacy: full state restore
            optimizer_tmp = build_optimizer(model, cfg, phase=2)
            start_epoch, best_iou = load_checkpoint(
                model, optimizer_tmp, args.resume, device
            )
            start_epoch += 1
            del optimizer_tmp
            log.info(f"Resuming from epoch {start_epoch}")

    # -- Detach prefix (freeze early blocks for DML stability) -------------
    detach_prefix = int(llrd_cfg.get("detach_prefix", 0))
    if detach_prefix > 0 and (use_llrd or use_lora):
        n_frozen = model.set_detach_prefix(detach_prefix)
        log.info(f"[DetachPrefix] {n_frozen} blocks in no_grad prefix, "
                 f"backward only through blocks {detach_prefix}-31 + decoder")

    # -- Strip frozen blocks in cached mode (saves ~2+ GB VRAM) -------------
    if cache_dir and detach_prefix > 0:
        inner = model.backbone._backbone
        if hasattr(inner, "blocks") and len(inner.blocks) > detach_prefix:
            n_stripped = 0
            for i in range(detach_prefix):
                inner.blocks[i] = nn.Identity()
                n_stripped += 1
            if hasattr(inner, "patch_embed"):
                inner.patch_embed = nn.Identity()
            log.info(f"[Cache] Stripped {n_stripped} frozen blocks from GPU "
                     f"(freed ~{n_stripped * 80} MB VRAM)")

    # ========================================================================
    # LLRD / LoRA TRAINING PATH (single phase, all params trainable)
    # ========================================================================
    if single_phase:
        mode_name = "LLRD" if use_llrd else "LoRA"
        log.info(f"\n{'='*65}")
        log.info(f"  {mode_name} Training  (epochs {start_epoch}-{total_epochs})")
        log.info(f"{'='*65}\n")

        run_epochs = total_epochs - start_epoch + 1
        optimizer = build_optimizer(model, cfg, phase=2, llrd_decay=llrd_decay)
        # scheduler steps once per grad_accum, not per batch
        sched_steps_per_epoch = max(1, len(train_dl) // grad_accum)
        scheduler = build_scheduler(optimizer, cfg, sched_steps_per_epoch,
                                     total_training_epochs=run_epochs)
        scaler    = make_scaler(device.type, use_amp)

        for epoch in range(start_epoch, total_epochs + 1):
            t0 = time.time()

            train_loss, train_scores = train_one_epoch(
                model, train_dl, optimizer, scheduler, scaler, loss_fn,
                device, train_metrics, epoch, cfg, use_amp,
                overfit_batches=args.overfit_batches, grad_accum=grad_accum,
                regression=regression,
            )
            # Flush GPU/CPU caches between train and val to avoid OOM
            import gc as _gc
            _gc.collect()

            val_loss, val_scores = validate(
                model, val_dl, loss_fn, device, val_metrics,
                use_amp=use_amp, threshold=eval_threshold,
                regression=regression,
            )
            _gc.collect()

            elapsed = time.time() - t0
            # mean_iou: for classification = actual mIoU, for regression = -MAE
            # Both conventions: higher = better (used for checkpoint selection)
            val_miou = val_scores.get("mean_iou", 0)
            val_ff1  = val_scores.get("fire_f1", val_scores.get("f1_fg", 0))

            # Log with selected LR info (first/last backbone + decoder)
            last_lrs = scheduler.get_last_lr()
            lrs = {}
            for i, g in enumerate(optimizer.param_groups):
                name = g.get("name", f"g{i}")
                if len(g["params"]) > 0:
                    lrs[name] = last_lrs[i]
            # Show first backbone, last backbone, decoder LRs
            lr_keys = list(lrs.keys())
            show_keys = []
            for k in lr_keys:
                if "bb_block_00" in k or "bb_block_31" in k or k == "decoder":
                    show_keys.append(k)
            if not show_keys:
                show_keys = lr_keys[:3]
            lr_str = "  ".join(f"{k}={lrs[k]:.1e}" for k in show_keys)

            if regression:
                val_mae  = val_scores.get("mae", 0)
                val_rmse = val_scores.get("rmse", 0)
                val_r2   = val_scores.get("r2", 0)
                log.info(
                    f"Epoch {epoch:3d}/{total_epochs}  [{elapsed:.0f}s]  "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"val_MAE={val_mae:.4f}  val_RMSE={val_rmse:.4f}  "
                    f"val_R2={val_r2:.4f}  val_fireF1={val_ff1:.4f}  "
                    f"({mode_name})  {lr_str}"
                )
            else:
                log.info(
                    f"Epoch {epoch:3d}/{total_epochs}  [{elapsed:.0f}s]  "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"val_mIoU={val_miou:.4f}  val_fireF1={val_ff1:.4f}  "
                    f"({mode_name})  {lr_str}"
                )

            # TensorBoard
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            for k, v in val_scores.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            # Log selected LRs (not all 35 groups)
            for k in show_keys:
                writer.add_scalar(f"lr/{k}", lrs[k], epoch)

            # Best checkpoint
            # Lightweight save: only trainable params (LoRA or LLRD+detach)
            lora_save = use_lora or (detach_prefix > 0)
            if val_miou > best_iou:
                best_iou   = val_miou
                best_epoch = epoch
                no_improve_count = 0
                save_checkpoint(model, optimizer, epoch, best_iou, cfg,
                                ckpt_dir / "best_vanaagni.pth", tag="BEST",
                                lora_mode=lora_save)
            else:
                no_improve_count += 1

            # Periodic checkpoint (weights-only to avoid MemoryError)
            if epoch % save_every == 0:
                save_checkpoint(model, optimizer, epoch, best_iou, cfg,
                                ckpt_dir / f"vanaagni_epoch_{epoch:03d}.pth",
                                weights_only=True, lora_mode=lora_save)

            # Early stopping
            if patience > 0 and no_improve_count >= patience:
                log.info(f"Early stopping at epoch {epoch} "
                         f"(no improvement for {patience} epochs)")
                break

    # ========================================================================
    # LEGACY PHASE 1:  Frozen backbone  (decoder + FiLM + weather heads only)
    # ========================================================================
    phase1_start = start_epoch if not single_phase else total_epochs + 1
    phase1_end   = min(freeze_epochs, total_epochs)

    if phase1_start <= phase1_end:
        log.info(f"\n{'='*65}")
        log.info(f"  PHASE 1: Frozen Backbone  (epochs {phase1_start}-{phase1_end})")
        log.info(f"{'='*65}\n")

        freeze_backbone(model)
        optimizer = build_optimizer(model, cfg, phase=1)
        scheduler = build_scheduler(optimizer, cfg, len(train_dl), phase=1)
        scaler    = make_scaler(device.type, use_amp)

        for epoch in range(phase1_start, phase1_end + 1):
            t0 = time.time()

            train_loss, train_scores = train_one_epoch(
                model, train_dl, optimizer, scheduler, scaler, loss_fn,
                device, train_metrics, epoch, cfg, use_amp,
                overfit_batches=args.overfit_batches, grad_accum=grad_accum,
                regression=regression,
            )

            val_loss, val_scores = validate(
                model, val_dl, loss_fn, device, val_metrics,
                use_amp=use_amp, threshold=eval_threshold,
                regression=regression,
            )

            elapsed = time.time() - t0
            val_miou = val_scores.get("mean_iou", 0)
            val_ff1  = val_scores.get("fire_f1", val_scores.get("f1_fg", 0))

            log.info(
                f"Epoch {epoch:3d}/{total_epochs}  [{elapsed:.0f}s]  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_mIoU={val_miou:.4f}  val_fireF1={val_ff1:.4f}  "
                f"(Phase 1 / frozen)"
            )

            # TensorBoard
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            for k, v in val_scores.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            writer.add_scalar("lr/decoder", optimizer.param_groups[0]["lr"], epoch)

            # Best checkpoint (track mean_iou across foreground classes)
            if val_miou > best_iou:
                best_iou   = val_miou
                best_epoch = epoch
                no_improve_count = 0
                save_checkpoint(model, optimizer, epoch, best_iou, cfg,
                                ckpt_dir / "best_vanaagni.pth", tag="BEST")
            else:
                no_improve_count += 1

            # Periodic checkpoint (weights-only to avoid RAM MemoryError on large models)
            if epoch % save_every == 0:
                save_checkpoint(model, optimizer, epoch, best_iou, cfg,
                                ckpt_dir / f"vanaagni_epoch_{epoch:03d}.pth",
                                weights_only=True)

    # ========================================================================
    # LEGACY PHASE 2:  Full fine-tuning with differential LR
    # ========================================================================
    phase2_start = total_epochs + 1 if single_phase else max(start_epoch, freeze_epochs + 1)
    phase2_end   = total_epochs

    if phase2_start <= phase2_end:
        log.info(f"\n{'='*65}")
        log.info(f"  PHASE 2: Full Fine-Tuning  (epochs {phase2_start}-{phase2_end})")
        log.info(f"{'='*65}\n")

        unfreeze_backbone(model)
        optimizer = build_optimizer(model, cfg, phase=2)
        scheduler = build_scheduler(optimizer, cfg, len(train_dl), phase=2)
        scaler    = make_scaler(device.type, use_amp)
        no_improve_count = 0  # reset patience for Phase 2

        for epoch in range(phase2_start, phase2_end + 1):
            t0 = time.time()

            train_loss, train_scores = train_one_epoch(
                model, train_dl, optimizer, scheduler, scaler, loss_fn,
                device, train_metrics, epoch, cfg, use_amp,
                overfit_batches=args.overfit_batches, grad_accum=grad_accum,
                regression=regression,
            )

            val_loss, val_scores = validate(
                model, val_dl, loss_fn, device, val_metrics,
                use_amp=use_amp, threshold=eval_threshold,
                regression=regression,
            )

            elapsed = time.time() - t0
            val_miou = val_scores.get("mean_iou", 0)
            val_ff1  = val_scores.get("fire_f1", val_scores.get("f1_fg", 0))

            # Log with per-group LR info
            # scheduler.get_last_lr() returns one value per param group
            last_lrs = scheduler.get_last_lr()
            lrs = {g.get("name", f"g{i}"): last_lrs[i]
                   for i, g in enumerate(optimizer.param_groups)
                   if len(g["params"]) > 0}
            lr_str = "  ".join(f"{k}={v:.1e}" for k, v in list(lrs.items())[:3])

            log.info(
                f"Epoch {epoch:3d}/{total_epochs}  [{elapsed:.0f}s]  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_mIoU={val_miou:.4f}  val_fireF1={val_ff1:.4f}  "
                f"(Phase 2)  {lr_str}"
            )

            # TensorBoard
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            for k, v in val_scores.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            for i, pg in enumerate(optimizer.param_groups):
                name = pg.get("name", f"group_{i}")
                writer.add_scalar(f"lr/{name}", pg["lr"], epoch)

            # Best checkpoint (track mean_iou across foreground classes)
            if val_miou > best_iou:
                best_iou   = val_miou
                best_epoch = epoch
                no_improve_count = 0
                save_checkpoint(model, optimizer, epoch, best_iou, cfg,
                                ckpt_dir / "best_vanaagni.pth", tag="BEST")
            else:
                no_improve_count += 1

            # Periodic checkpoint (weights-only to avoid RAM MemoryError on large models)
            if epoch % save_every == 0:
                save_checkpoint(model, optimizer, epoch, best_iou, cfg,
                                ckpt_dir / f"vanaagni_epoch_{epoch:03d}.pth",
                                weights_only=True)

            # Early stopping
            if patience > 0 and no_improve_count >= patience:
                log.info(f"Early stopping at epoch {epoch} "
                         f"(no improvement for {patience} epochs)")
                break

    # ========================================================================
    # Final Evaluation on Test Set
    # ========================================================================
    log.info(f"\n{'='*65}")
    log.info(f"  FINAL EVALUATION  (test split)")
    log.info(f"{'='*65}")

    # Load best checkpoint
    best_ckpt = ckpt_dir / "best_vanaagni.pth"
    if best_ckpt.exists():
        load_checkpoint(model, None, str(best_ckpt), device)
        log.info(f"Loaded best checkpoint (epoch {best_epoch})")

    if regression:
        fire_thresh  = float(t_cfg.get("fire_dnbr_threshold", 0.05))
        test_metrics = RegressionMetrics(fire_threshold=fire_thresh)
    else:
        test_metrics = SegmentationMetrics(num_classes=num_classes)
    test_loss, test_scores = validate(
        model, test_dl, loss_fn, device, test_metrics,
        use_amp=use_amp, threshold=eval_threshold,
        regression=regression,
    )

    log.info(f"\nTest Results:")
    log.info(f"  Loss:       {test_loss:.4f}")

    if regression:
        log.info(f"  MAE:        {test_scores.get('mae', 0):.4f}")
        log.info(f"  RMSE:       {test_scores.get('rmse', 0):.4f}")
        log.info(f"  R-squared:  {test_scores.get('r2', 0):.4f}")
        log.info(f"\n  Binary fire detection (dNBR > {fire_thresh}):")
        log.info(f"    fire_IoU:  {test_scores.get('fire_iou', 0):.4f}")
        log.info(f"    fire_F1:   {test_scores.get('fire_f1', 0):.4f}")
        log.info(f"    fire_Prec: {test_scores.get('fire_precision', 0):.4f}")
        log.info(f"    fire_Rec:  {test_scores.get('fire_recall', 0):.4f}")
    elif num_classes > 2:
        log.info(f"  Mean IoU:   {test_scores.get('mean_iou', 0):.4f}")
        log.info(f"  Pixel Acc:  {test_scores.get('pixel_accuracy', 0):.4f}")
        # Multi-class severity: per-class + aggregated fire metrics
        log.info(f"\n  Per-class IoU:")
        for name in SEVERITY_NAMES:
            iou_k = f"iou_{name}"
            f1_k  = f"f1_{name}"
            log.info(f"    {name:12s}  IoU={test_scores.get(iou_k, 0):.4f}  "
                     f"F1={test_scores.get(f1_k, 0):.4f}")
        log.info(f"\n  Aggregated fire (severity 1-4 vs no_burn):")
        log.info(f"    fire_IoU:  {test_scores.get('fire_iou', 0):.4f}")
        log.info(f"    fire_F1:   {test_scores.get('fire_f1', 0):.4f}")
        log.info(f"    fire_Prec: {test_scores.get('fire_precision', 0):.4f}")
        log.info(f"    fire_Rec:  {test_scores.get('fire_recall', 0):.4f}")
    else:
        log.info(f"  Mean IoU:   {test_scores.get('mean_iou', 0):.4f}")
        log.info(f"  Pixel Acc:  {test_scores.get('pixel_accuracy', 0):.4f}")
        # Binary mode
        log.info(f"  IoU (fg):   {test_scores.get('iou_fg', 0):.4f}")
        log.info(f"  F1 (fg):    {test_scores.get('f1_fg', 0):.4f}")
        log.info(f"  Prec (fg):  {test_scores.get('precision_fg', 0):.4f}")
        log.info(f"  Rec (fg):   {test_scores.get('recall_fg', 0):.4f}")

    # Save test results
    results = {
        "best_epoch":     best_epoch,
        "best_val_iou":   best_iou,
        "test_loss":      test_loss,
        "test_scores":    test_scores,
        "config":         cfg,
    }
    results_path = Path(p_cfg.get("output_dir", "outputs/vanaagni")) / "test_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Test results saved to: {results_path}")

    writer.close()
    log.info(f"\nBest epoch: {best_epoch}  |  Best val mIoU: {best_iou:.4f}")
    log.info(f"Training complete. Checkpoints in: {ckpt_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log.error(f"FATAL: {traceback.format_exc()}")
        raise
