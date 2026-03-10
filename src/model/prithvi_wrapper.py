"""
Prithvi Model Wrapper
=====================
Wraps the IBM/NASA Prithvi-100M ViT encoder for fine-tuning.

The Prithvi-100M model is a Masked AutoEncoder (MAE) trained on HLS
multi-temporal satellite imagery. We use its encoder as a feature extractor
and attach task-specific heads for change detection and prediction.

Reference: https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
import logging

log = logging.getLogger(__name__)


class PrithviEncoder(nn.Module):
    """
    Loads Prithvi-100M encoder from HuggingFace and extracts
    multi-scale spatial features for dense prediction tasks.

    Input:  (B, T, C, H, W)  — T=3 temporal frames, C=6 bands
    Output: list of feature maps at different scales
    """

    def __init__(
        self,
        pretrained: str = "ibm-nasa-geospatial/Prithvi-100M",
        num_frames: int = 3,
        in_chans: int = 6,
        img_size: int = 224,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.in_chans = in_chans
        self.img_size = img_size
        self.embed_dim = embed_dim

        self._load_prithvi(pretrained, num_frames, in_chans, img_size,
                           embed_dim, depth, num_heads)

        if freeze_encoder:
            log.info("Freezing Prithvi encoder weights")
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _load_prithvi(self, pretrained, num_frames, in_chans, img_size,
                       embed_dim, depth, num_heads):
        """Download and load Prithvi-100M weights from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
            import torch

            log.info(f"Loading Prithvi-100M from: {pretrained}")

            # Try to import from the NASA-IMPACT mmsegmentation style
            # Fall back to direct weight loading
            try:
                from src.model.prithvi_mae import PrithviMAE
                self.encoder = PrithviMAE(
                    img_size=img_size,
                    num_frames=num_frames,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    depth=depth,
                    num_heads=num_heads,
                )
                # Load pretrained weights
                weights_path = hf_hub_download(
                    repo_id=pretrained,
                    filename="Prithvi_100M.pt"
                )
                state_dict = torch.load(weights_path, map_location="cpu")
                # Handle different checkpoint formats
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                missing, unexpected = self.encoder.load_state_dict(
                    state_dict, strict=False
                )
                log.info(f"✅ Prithvi weights loaded.")
                if missing:
                    log.info(f"   Missing keys (expected for new heads): {len(missing)}")
                if unexpected:
                    log.warning(f"   Unexpected keys: {len(unexpected)}")

            except ImportError:
                log.warning("PrithviMAE module not found, building architecture from config")
                self.encoder = self._build_vit_encoder(
                    img_size, in_chans, num_frames, embed_dim, depth, num_heads
                )

        except Exception as e:
            log.warning(f"Could not load pretrained Prithvi weights: {e}")
            log.info("Building encoder from scratch (random initialization)")
            self.encoder = self._build_vit_encoder(
                img_size, in_chans, num_frames, embed_dim, depth, num_heads
            )

    def _build_vit_encoder(self, img_size, in_chans, num_frames,
                            embed_dim, depth, num_heads):
        """Build a ViT encoder compatible with Prithvi architecture."""
        import timm
        # Use timm's ViT as backbone with custom patch embedding for multi-spectral
        encoder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            in_chans=in_chans * num_frames,  # flatten temporal dim into channels
            num_classes=0,  # remove classification head
            global_pool="",
        )
        return encoder

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through Prithvi encoder.

        Args:
            x: (B, T, C, H, W) — multi-temporal satellite imagery

        Returns:
            List of feature tensors at multiple scales for decoder heads
        """
        B, T, C, H, W = x.shape

        # Prithvi's native input: (B, C, T, H, W)
        # Rearrange from (B, T, C, H, W) → (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        try:
            # Try Prithvi native forward
            features = self.encoder.forward_features(x)
        except Exception:
            # Fallback: flatten temporal as channels → (B, T*C, H, W)
            x_flat = x.reshape(B, T * C, H, W)
            features = self.encoder.forward_features(x_flat)

        return features  # (B, N_tokens, embed_dim) or list


class PatchEmbedMultiTemporal(nn.Module):
    """
    Custom 3D patch embedding for multi-temporal multi-spectral input.
    Projects (B, C, T, H, W) → (B, N_tokens, embed_dim)
    """

    def __init__(self, img_size=224, patch_size=16, temporal_size=3,
                 in_chans=6, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches_h = img_size // patch_size
        self.n_patches_w = img_size // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w * temporal_size

        # 3D conv to embed temporal + spatial patches
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, embed_dim, T, H/P, W/P)
        B, E, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, E)
        return x
