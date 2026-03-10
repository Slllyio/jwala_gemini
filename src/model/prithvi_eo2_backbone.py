"""
Prithvi-EO-2.0 Backbone (HuggingFace AutoModel)
==============================================
Loads the IBM/NASA Prithvi-EO-2.0 foundation model directly from HuggingFace
to avoid internal dispatch issues with TerraTorch Registry.
"""

import logging
from typing import List, Optional
import traceback

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# TerraTorch HuggingFace model IDs
_EO2_300M_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
_EO2_600M_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-600M"

# Canonical architecture params for each variant
_EO2_ARCH = {
    "300M": dict(embed_dim=1024, depth=24, num_heads=16, patch_size=14),
    "600M": dict(embed_dim=1280, depth=32, num_heads=16, patch_size=14),
}


class PrithviEO2Backbone(nn.Module):
    """
    Drop-in replacement for ``PrithviMAE`` using HuggingFace AutoModel directly.
    """

    def __init__(
        self,
        pretrained:   str  = _EO2_600M_ID,
        num_frames:   int  = 3,
        in_chans:     int  = 6,
        img_size:     int  = 224,
        embed_dim:    int  = 1280,
        depth:        int  = 32,
        num_heads:    int  = 16,
        patch_size:   int  = 14,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.in_chans   = in_chans
        self.img_size   = img_size
        self.embed_dim  = embed_dim
        self.depth      = depth
        self.patch_size = patch_size
        self.grid_size  = img_size // patch_size

        self._backbone: nn.Module = self._load(pretrained, num_frames, in_chans,
                                               img_size, embed_dim, depth,
                                               num_heads, patch_size)

    def _load(self, pretrained, num_frames, in_chans, img_size,
              embed_dim, depth, num_heads, patch_size) -> nn.Module:
        # TIER 1: HuggingFace AutoModel (more robust for direct inference/fine-tuning)
        try:
            from transformers import AutoModel
            log.info(f"Loading Prithvi-EO-2.0 via HuggingFace AutoModel: {pretrained}")
            backbone = AutoModel.from_pretrained(
                pretrained,
                trust_remote_code=True,
                num_frames   = num_frames,
                in_chans     = in_chans,
                img_size     = img_size,
            )
            # Prithvi HF models usually have a .model attribute or similar
            if hasattr(backbone, "config") and hasattr(backbone.config, "hidden_size"):
                self.embed_dim = backbone.config.hidden_size
            log.info(f"[OK] Prithvi-EO-2.0 loaded via AutoModel.")
            return backbone
        except Exception as exc:
            log.warning(f"HuggingFace AutoModel load failed: {exc}")

        # TIER 2: TerraTorch Registry (Fallback)
        try:
            from terratorch.registry import BACKBONE_REGISTRY
            log.info(f"Fallback: Loading via TerraTorch Registry: {pretrained}")
            registry_key = "terratorch_prithvi_eo_v2_600" if "600m" in pretrained.lower() else "terratorch_prithvi_eo_v2_300"
            backbone = BACKBONE_REGISTRY.build(
                registry_key,
                pretrained   = True,
                num_frames   = num_frames,
                in_chans     = in_chans,
                img_size     = img_size,
                patch_size   = (1, patch_size, patch_size),
            )
            return backbone
        except Exception as exc:
            log.warning(f"TerraTorch load failed: {exc}")

        return _build_vit_fallback(embed_dim, depth, num_heads,
                                   in_chans, num_frames, img_size, patch_size)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Standardize input to 5D
        if x.ndim == 4:
            B, TC, H, W = x.shape
            C = self.in_chans
            T = TC // C
            x = x.reshape(B, C, T, H, W)
        
        # Most HF models return a class with last_hidden_state
        out = self._backbone(x)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        if isinstance(out, (list, tuple)):
            return out[-1]
        return out

    def get_intermediate_layers(self, x: torch.Tensor, n_last: int = 4) -> List[torch.Tensor]:
        # Simple intermediate extraction for HF models
        # For now, we'll return n_last copies of the final features if we can't easily get intermediates
        feats = self.forward_features(x)
        if isinstance(feats, (list, tuple)):
            return [f if f.ndim == 3 else f.unsqueeze(0) for f in feats[-n_last:]]
        return [feats] * n_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def _build_vit_fallback(embed_dim, depth, num_heads, in_chans, num_frames, img_size, patch_size):
    try:
        import timm
        return timm.create_model(
            f"vit_large_patch{patch_size}_{img_size}",
            pretrained     = False,
            in_chans       = in_chans * num_frames,
            embed_dim      = embed_dim,
            depth          = depth,
            num_heads      = num_heads,
            num_classes    = 0,
            global_pool    = "",
            img_size       = img_size,
        )
    except Exception:
        from src.model.prithvi_mae import PrithviMAE
        return PrithviMAE(img_size=img_size, num_frames=num_frames, in_chans=in_chans,
                          embed_dim=embed_dim, depth=depth, num_heads=num_heads)

def build_prithvi_eo2(variant="600M", num_frames=3, in_chans=6, img_size=224):
    if variant not in _EO2_ARCH:
        raise ValueError(f"Unknown variant '{variant}'. Choose '300M' or '600M'.")
    hf_id = _EO2_300M_ID if variant == "300M" else _EO2_600M_ID
    arch = _EO2_ARCH[variant]
    try:
        return PrithviEO2Backbone(pretrained=hf_id, num_frames=num_frames, in_chans=in_chans,
                                  img_size=img_size, **arch)
    except Exception as exc:
        log.error(f"PrithviEO2Backbone construction failed: {exc}")
        return None

