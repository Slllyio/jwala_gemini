"""
Full Model: PrithviForestChange
================================
Combines:
  - PrithviEO2Backbone  (Prithvi-EO-2.0 600M via TerraTorch)  [preferred]
    or PrithviMAE       (Prithvi-100M hand-rolled)              [fallback]
  - ChangeDetectionHead (UperNet/FPN)
  - TemporalPredictionHead (ConvLSTM)

Modes:
  "detect"  — bi/multi-temporal change detection → change mask
  "predict" — temporal sequence → future risk map

Backbone selection (config.yaml):
  model.use_eo2: true            # enable EO-2.0 backbone
  model.eo2_variant: "600M"      # "300M" or "600M"
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'reconfigure') else None  # type: ignore[union-attr]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import logging
import yaml

from src.model.prithvi_mae import PrithviMAE
from src.model.change_head import ChangeDetectionHead, CombinedLoss
from src.model.prediction_head import TemporalPredictionHead
from src.model.prithvi_eo2_backbone import PrithviEO2Backbone, build_prithvi_eo2

log = logging.getLogger(__name__)


class PrithviForestChange(nn.Module):
    """
    Unified model for forest change detection and prediction.

    Two operating modes:
      - detect: Takes (B, T, C, H, W) → returns (B, num_classes, H, W) logits
      - predict: Takes List[T] of (B, C, H, W) → returns (B, 1, H, W) risk map
    """

    def __init__(
        self,
        pretrained:               str  = "ibm-nasa-geospatial/Prithvi-100M",
        num_frames:               int  = 3,
        in_chans:                 int  = 6,
        img_size:                 int  = 224,
        embed_dim:                int  = 768,
        depth:                    int  = 12,
        num_heads:                int  = 12,
        num_classes:              int  = 2,
        freeze_encoder:           bool = False,
        prediction_num_frames:    int  = 6,
        prediction_hidden_dims:   Optional[List[int]] = None,
        # Prithvi-EO-2.0 options (Phase 3)
        use_eo2:                  bool = False,
        eo2_variant:              str  = "600M",
    ):
        super().__init__()

        if prediction_hidden_dims is None:
            prediction_hidden_dims = [256, 128]

        self.num_frames = num_frames
        self.img_size   = img_size

        # ── Encoder: try Prithvi-EO-2.0 first if requested ───────────────────
        eo2_loaded = False
        if use_eo2:
            log.info(f"Phase 3: attempting Prithvi-EO-2.0 ({eo2_variant}) backbone...")
            eo2 = build_prithvi_eo2(
                variant    = eo2_variant,
                num_frames = num_frames,
                in_chans   = in_chans,
                img_size   = img_size,
            )
            if eo2 is not None:
                self.encoder = eo2
                # Actual embed_dim may differ from config placeholder
                actual_embed_dim = self.encoder.embed_dim
                eo2_loaded = True
                log.info(
                    f"[EO2] Prithvi-EO-2.0 ({eo2_variant}) loaded. "
                    f"embed_dim={actual_embed_dim}"
                )
            else:
                log.warning(
                    "[EO2] Prithvi-EO-2.0 unavailable — falling back to Prithvi-100M."
                )

        if not eo2_loaded:
            # ── Fallback: Prithvi-100M (original path) ────────────────────────
            log.info("Initializing Prithvi-100M encoder...")
            self.encoder = PrithviMAE(
                img_size   = img_size,
                num_frames = num_frames,
                in_chans   = in_chans,
                embed_dim  = embed_dim,
                depth      = depth,
                num_heads  = num_heads,
            )
            actual_embed_dim = embed_dim
            self._load_pretrained(pretrained)

        self.embed_dim = actual_embed_dim

        if freeze_encoder:
            log.info("[FROZEN] Encoder frozen — only heads will be trained")
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            log.info("[UNFROZEN] Encoder unfrozen — full fine-tuning enabled")

        # Grid size depends on patch size: EO2 uses 14px patches, 100M uses 16px
        patch_size = getattr(self.encoder, "patch_size", 16)
        grid_size  = img_size // patch_size

        # ── Change Detection Head ─────────────────────────────────────────────
        # Both 100M and EO2 feed last-4 block features; head adapts to embed_dim
        self.change_head = ChangeDetectionHead(
            embed_dim   = actual_embed_dim,
            fpn_out_dim = 256,
            num_classes = num_classes,
            img_size    = img_size,
            grid_size   = grid_size,
            num_frames  = num_frames,
            num_scales  = 4,
        )

        # ── Temporal Prediction Head ──────────────────────────────────────────
        self.prediction_head = TemporalPredictionHead(
            embed_dim   = actual_embed_dim,
            grid_size   = grid_size,
            num_frames  = prediction_num_frames,
            hidden_dims = prediction_hidden_dims,
            img_size    = img_size,
        )

    def _load_pretrained(self, pretrained: str):
        """Load pre-trained Prithvi-100M weights from HuggingFace hub."""
        try:
            from huggingface_hub import hf_hub_download
            log.info(f"Downloading Prithvi weights from: {pretrained}")
            weights_path = hf_hub_download(
                repo_id  = pretrained,
                filename = "Prithvi_100M.pt",
            )
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
            if "model" in state_dict:
                state_dict = state_dict["model"]

            missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
            log.info("[OK] Prithvi pre-trained weights loaded!")
            log.info(f"   Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
        except Exception as e:
            log.warning(f"[WARN] Could not load pre-trained weights: {e}")
            log.warning(
                "   Training from scratch (random init). "
                "Set HF_HUB_OFFLINE=0 and ensure internet access."
            )

    def forward_detect(self, x: torch.Tensor) -> torch.Tensor:
        """
        Change detection forward pass.

        Args:
            x: (B, T, C, H, W) multi-temporal imagery

        Returns:
            (B, num_classes, H, W) change logits
        """
        B, T, C, H, W = x.shape
        # Prithvi expects (B, C, T, H, W)
        x_enc = x.permute(0, 2, 1, 3, 4)

        # Get intermediate features from last 4 transformer layers
        features = self.encoder.get_intermediate_layers(x_enc, n_last=4)

        # Decode with change head
        logits = self.change_head(features, target_size=H)
        return logits

    def forward_predict(self, x_sequence: List[torch.Tensor]) -> torch.Tensor:
        """
        Future change prediction forward pass.

        Args:
            x_sequence: List of T tensors, each (B, C, H, W)

        Returns:
            (B, 1, H, W) risk probability in [0, 1]
        """
        # Extract features for each frame independently
        frame_features = []
        for frame in x_sequence:
            # Add temporal dim: (B, C, H, W) → (B, C, 1, H, W)
            frame_enc = frame.unsqueeze(2)
            # Temporarily expand num_frames
            frame_enc = frame_enc.expand(-1, -1, self.num_frames, -1, -1)
            features = self.encoder.forward_features(frame_enc)
            frame_features.append(features)

        risk_map = self.prediction_head(frame_features)
        return risk_map

    def forward(self, x, mode: str = "detect"):
        if mode == "detect":
            return self.forward_detect(x)
        elif mode == "predict":
            return self.forward_predict(x)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'detect' or 'predict'.")


def build_model(cfg: dict) -> PrithviForestChange:
    """Build model from config dict."""
    model_cfg = cfg["model"]
    pred_cfg  = cfg.get("prediction", {})

    model = PrithviForestChange(
        pretrained             = model_cfg["pretrained"],
        num_frames             = model_cfg["num_frames"],
        in_chans               = model_cfg["in_chans"],
        img_size               = model_cfg["img_size"],
        embed_dim              = model_cfg["embed_dim"],
        depth                  = model_cfg["depth"],
        num_heads              = model_cfg["num_heads"],
        num_classes            = model_cfg["num_classes"],
        freeze_encoder         = model_cfg.get("freeze_encoder", False),
        prediction_num_frames  = pred_cfg.get("num_time_steps", 6),
        prediction_hidden_dims = [pred_cfg.get("hidden_size", 256), 128],
        # Prithvi-EO-2.0 options (Phase 3)
        use_eo2                = model_cfg.get("use_eo2", False),
        eo2_variant            = model_cfg.get("eo2_variant", "600M"),
    )
    return model


if __name__ == "__main__":
    """Quick forward pass test."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)
    print("\n" + "="*50)
    print("[TEST] Prithvi Forest Change -- Model Test")
    print("="*50)

    model = build_model(cfg)
    model.eval()

    B, T, C, H, W = 1, cfg["model"]["num_frames"], cfg["model"]["in_chans"], \
                    cfg["model"]["img_size"], cfg["model"]["img_size"]

    # --- Test detection ---
    print(f"\n[DETECT] Input: ({B}, {T}, {C}, {H}, {W})")
    x = torch.randn(B, T, C, H, W)
    with torch.no_grad():
        out = model(x, mode="detect")
    print(f"[DETECT] Output: {out.shape}  Expected: ({B}, {cfg['model']['num_classes']}, {H}, {W})")
    assert out.shape == (B, cfg["model"]["num_classes"], H, W), "Shape mismatch!"

    # --- Test prediction ---
    N = cfg.get("prediction", {}).get("num_time_steps", 6)
    print(f"\n[PREDICT] Input: {N} frames of ({B}, {C}, {H}, {W})")
    frames = [torch.randn(B, C, H, W) for _ in range(N)]
    with torch.no_grad():
        risk = model(frames, mode="predict")
    print(f"[PREDICT] Output: {risk.shape}  Expected: ({B}, 1, {H}, {W})")
    print(f"[PREDICT] Risk range: [{risk.min():.3f}, {risk.max():.3f}]")
    assert risk.shape == (B, 1, H, W), "Shape mismatch!"

    n_params  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[PASS] All tests passed!")
    print(f"   Total parameters:     {n_params:,}")
    print(f"   Trainable parameters: {trainable:,}")
    print(f"   Backbone:             {'Prithvi-EO-2.0' if cfg['model'].get('use_eo2') else 'Prithvi-100M'}")
