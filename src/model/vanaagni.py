print(f'DEBUG: Loading vanaagni from {__file__}')
"""
src/model/vanaagni.py
=====================
VanAgni -- Multi-Modal Forest Fire Prediction Model

Architecture:
  Prithvi-EO-2.0-600M-TL backbone  (frozen Phase 1 -> differential LR Phase 2)
       |  4 multi-scale ViT features  [layers 7, 15, 23, 31]
       v
  UNet Decoder with FiLM Conditioning
       +-- FiLM(weather_current + weather_7d)  -> per-channel affine at each scale
       +-- SpatialAux(terrain + landcover + burn_age)  -> additive at each scale
       +-- Progressive upsample: 16^2 -> 32^2 -> 64^2 -> 128^2 -> 224^2
       |
       v
  Severity Classifier  Conv(64 -> 5)  -> (B, 5, H, W) logits

Input tensor contract:
  hls         (B, 6, T=3, 224, 224)   normalised HLS S30 bands
  indices     (B, 3, T=3, 224, 224)   NDVI, NBR, BSI  [optional, not used by backbone]
  weather     (B, 10)                 current-day FWI weather (normalised)
  weather_7d  (B, 10, 7)             7-day lookback window
  terrain     (B, 4, 224, 224)       elev, slope, sin_asp, cos_asp
  burn_age    (B, 3, 224, 224)       recent / high-risk / mature
  landcover   (B, 11, 224, 224)      ESA WorldCover one-hot

Output:
  logits      (B, C, 224, 224)       C=5 burn severity classes:
                                       0=no_burn, 1=very_low, 2=low,
                                       3=moderate, 4=high
                                     (or C=2 for legacy binary mode)

Usage:
    model = build_vanaagni(cfg)
    logits = model(hls, weather, weather_7d, terrain, burn_age, landcover)
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.prithvi_eo2_backbone import (
    PrithviEO2Backbone, build_prithvi_eo2, _patch_detach_prefix,
)
from src.model.film_layers import (
    FiLMBlock,
    WeatherEncoder,
    ForecastWeatherEncoder,
    SpatialAuxEncoder,
)
from src.model.change_head import FocalDiceLoss

log = logging.getLogger(__name__)

# -- Architecture lookup per variant ------------------------------------------
# TL variants share dimensions with their base; only pretrained weights differ.
_EO2_ARCH = {
    "300M":    dict(embed_dim=1024, depth=24, num_heads=16, patch_size=14),
    "300M-TL": dict(embed_dim=1024, depth=24, num_heads=16, patch_size=14),
    "600M":    dict(embed_dim=1280, depth=32, num_heads=16, patch_size=14),
    "600M-TL": dict(embed_dim=1280, depth=32, num_heads=16, patch_size=14),
}

def _select_indices(depth: int, n_scales: int = 4) -> list:
    """Evenly-spaced layer indices for multi-scale feature extraction.
    depth=24 -> [5, 11, 17, 23], depth=32 -> [7, 15, 23, 31]"""
    return [depth * (i + 1) // n_scales - 1 for i in range(n_scales)]

# Burn severity class names (aligned with ibm-nasa burn_intensity dataset)
SEVERITY_NAMES = {
    0: "no_burn",
    1: "very_low",
    2: "low",
    3: "moderate",
    4: "high",
}

_DECODER_CH     = [512, 256, 128, 64]
_SPATIAL_AUX_CH = 32
_WEATHER_DIM    = 128


# =============================================================================
# UNet Decoder Block with FiLM Conditioning
# =============================================================================

class FiLMDecoderBlock(nn.Module):
    """
    One scale of the UNet decoder with FiLM conditioning + spatial aux residual.

    Flow:
        1.  Upsample 2x (if not first block)
        2.  Concatenate skip connection from backbone
        3.  Add spatial aux features
        4.  Conv-BN-GELU  x2
        5.  FiLM modulate with weather conditioning
    """

    def __init__(
        self,
        in_ch:       int,
        skip_ch:     int,
        out_ch:      int,
        aux_ch:      int = _SPATIAL_AUX_CH,
        cond_dim:    int = _WEATHER_DIM,
        upsample:    bool = True,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                      align_corners=False) if upsample else nn.Identity()

        total_in = in_ch + skip_ch + aux_ch

        self.conv1 = nn.Sequential(
            nn.Conv2d(total_in, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        # Dropout2d after second conv -- regularises each decoder scale independently.
        # Applied before FiLM so weather modulation always sees full-channel features.
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.film = FiLMBlock(out_ch, cond_dim=cond_dim)

    def forward(
        self,
        x:    torch.Tensor,         # (B, in_ch, H, W) -- from previous scale
        skip: torch.Tensor,         # (B, skip_ch, H', W') -- from backbone
        aux:  torch.Tensor,         # (B, aux_ch, H', W') -- spatial auxiliary
        cond: torch.Tensor,         # (B, cond_dim) -- weather conditioning
    ) -> torch.Tensor:
        x = self.upsample(x)

        # Resize skip & aux to match x
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if aux.shape[-2:] != x.shape[-2:]:
            aux = F.interpolate(aux, size=x.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip, aux], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.film(x, cond)
        return x


# =============================================================================
# VanAgni Model
# =============================================================================

class VanAgni(nn.Module):
    """
    Multi-modal fire prediction model with burn severity output.

    Backbone:    Prithvi-EO-2.0-600M-TL  (ViT, embed_dim=1280, depth=32)
    Decoder:     4-scale UNet with FiLM weather conditioning
    Aux inputs:  terrain, land cover, burn-age rasters (via SpatialAuxEncoder)
    Output:      5-class burn severity segmentation at 30m resolution
                 (or 2-class binary fire/no-fire in legacy mode)
    """

    def __init__(
        self,
        # Backbone
        eo2_variant:         str  = "600M-TL",
        num_frames:          int  = 3,
        in_chans:            int  = 6,
        img_size:            int  = 224,
        backbone_init_from:  Optional[str] = None,
        freeze_backbone:     bool = False,
        # Decoder
        decoder_channels:    List[int] = None,
        # Weather conditioning
        weather_input_dim:   int = 10,
        weather_cond_dim:    int = _WEATHER_DIM,
        # Spatial aux
        terrain_ch:          int = 4,
        landcover_ch:        int = 11,
        burn_age_ch:         int = 3,
        spatial_aux_ch:      int = _SPATIAL_AUX_CH,
        # Classification
        num_classes:         int = 5,
        dropout:             float = 0.1,
    ):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = list(_DECODER_CH)

        self.img_size    = img_size
        self.n_scales    = len(decoder_channels)
        self.num_classes = num_classes
        assert self.n_scales == 4, "VanAgni requires exactly 4 decoder scales"

        # Look up architecture params for the requested variant
        if eo2_variant not in _EO2_ARCH:
            valid = ", ".join(sorted(_EO2_ARCH.keys()))
            raise ValueError(f"Unknown eo2_variant '{eo2_variant}'. Choose from: {valid}")

        arch = _EO2_ARCH[eo2_variant]
        embed_dim  = arch["embed_dim"]
        patch_size = arch["patch_size"]
        self.grid_size = img_size // patch_size   # 224/14 = 16

        # -- Backbone ------------------------------------------------------
        log.info(f"Building Prithvi-EO-2.0-{eo2_variant} backbone ...")
        self.backbone = build_prithvi_eo2(
            variant    = eo2_variant,
            num_frames = num_frames,
            in_chans   = in_chans,
            img_size   = img_size,
        )
        if self.backbone is None:
            raise RuntimeError(
                "Failed to build Prithvi-EO-2.0 backbone. "
                "Ensure TerraTorch or transformers is installed."
            )
        self.embed_dim = self.backbone.embed_dim

        # Optionally load fine-tuned burn-scars checkpoint
        if backbone_init_from:
            self._load_backbone_weights(backbone_init_from)

        if freeze_backbone:
            log.info("[FROZEN] Backbone parameters frozen")
            for p in self.backbone.parameters():
                p.requires_grad = False

        # -- Token -> 2D Projections (one per selected layer) ---------------
        # Each ViT feature: (B, 1+T*16*16, 1024) -> (B, 1024, 16, 16)
        # Then project to decoder channel width
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(embed_dim, ch, 1)
            for ch in decoder_channels
        ])

        # -- Weather Encoders ----------------------------------------------
        self.weather_enc    = WeatherEncoder(weather_input_dim, 64, weather_cond_dim)
        self.forecast_enc   = ForecastWeatherEncoder(weather_input_dim, 64, weather_cond_dim)
        # Fuse current + 7-day weather into single conditioning vector
        self.cond_fuse = nn.Sequential(
            nn.Linear(weather_cond_dim * 2, weather_cond_dim),
            nn.GELU(),
        )

        # -- Spatial Auxiliary Encoder -------------------------------------
        self.spatial_aux = SpatialAuxEncoder(
            terrain_ch   = terrain_ch,
            lc_ch        = landcover_ch,
            burn_age_ch  = burn_age_ch,
            hidden_ch    = 64,
            out_ch       = spatial_aux_ch,
            decoder_scales = self.n_scales,
        )

        # -- UNet Decoder with FiLM ---------------------------------------
        self.decoder_blocks = nn.ModuleList()
        for i, out_ch in enumerate(decoder_channels):
            in_ch = embed_dim if i == 0 else decoder_channels[i - 1]
            skip_ch = out_ch   # after skip_proj
            self.decoder_blocks.append(
                FiLMDecoderBlock(
                    in_ch    = in_ch,
                    skip_ch  = skip_ch,
                    out_ch   = out_ch,
                    aux_ch   = spatial_aux_ch,
                    cond_dim = weather_cond_dim,
                    upsample = (i > 0),   # first block: no upsample (16->16)
                    dropout  = dropout,   # per-block Dropout2d for regularisation
                )
            )

        # -- Temporal Attention for tokens_to_2d --------------------------
        self.temporal_attn = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # -- Final Upsample + Classifier ----------------------------------
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(decoder_channels[-1], 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(64, num_classes, 1)
        # Binary fire detection head (multi-task regression mode only)
        if num_classes == 1:
            self.fire_head = nn.Conv2d(64, 1, 1)

        # -- Log parameter counts -----------------------------------------
        self._log_params()

    # -- Weight loading ----------------------------------------------------

    def _load_backbone_weights(self, path: str) -> None:
        """Load fine-tuned backbone weights (e.g. from BurnScars checkpoint)."""
        import os
        if not os.path.isfile(path):
            log.warning(f"Backbone init file not found: {path}")
            return

        log.info(f"Loading backbone weights from: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)

        # Handle various checkpoint formats
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]

        # Filter for backbone keys only and strip prefix
        backbone_state = {}
        for k, v in state.items():
            # Common prefixes in TerraTorch/Lightning checkpoints
            for prefix in ["model.backbone.", "backbone.", "encoder.", ""]:
                if k.startswith(prefix) and len(prefix) > 0:
                    new_k = k[len(prefix):]
                    backbone_state[new_k] = v
                    break
            else:
                backbone_state[k] = v

        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        log.info(
            f"  Loaded backbone: {len(backbone_state)} keys  "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    # -- Forward -----------------------------------------------------------

    def _tokens_to_2d(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Reshape ViT tokens to 2D spatial map.

        Handles multiple input formats:
          4D (B, D, H, W) -- already spatial (timm forward_intermediates) -> pass-through
          3D (B, N+1, D)  -- ViT tokens with CLS -> temporal avg -> (B, D, G, G)
          2D (B, D)        -- global pooled -> expand to (B, D, G, G)
        """
        # 4D: already a spatial feature map -- pass through
        if tokens.ndim == 4:
            return tokens

        # 2D: global pooled, no spatial info; expand to (B, D, G, G)
        if tokens.ndim == 2:
            B, D = tokens.shape
            return tokens.unsqueeze(-1).unsqueeze(-1).expand(
                B, D, self.grid_size, self.grid_size
            ).contiguous()

        # 3D: (B, N_total, D) -- ViT token sequence
        B, N_total, D = tokens.shape
        G = self.grid_size

        # If N_total == G*G (no CLS token), skip CLS removal
        if N_total == G * G:
            spatial = tokens
            T = 1
        else:
            T = (N_total - 1) // (G * G)  # infer T from token count
            spatial = tokens[:, 1:, :]                        # (B, T*G*G, D)
            spatial = spatial[:, : T * G * G, :]              # safety clip

        if T > 1:
            spatial = spatial.reshape(B, T, G * G, D)
            # Temporal Cross-Attention to preserve chronology instead of mean
            attn_weights = torch.softmax(self.temporal_attn(spatial), dim=1) # (B, T, G*G, 1)
            spatial = (spatial * attn_weights).sum(dim=1)                     # (B, G*G, D)

        spatial = spatial.permute(0, 2, 1).contiguous().reshape(B, D, G, G)
        return spatial

    def forward(
        self,
        hls:        torch.Tensor,              # (B, 6, T, H, W)
        weather:    torch.Tensor,              # (B, 10)
        weather_7d: torch.Tensor,              # (B, 10, 7)
        terrain:    torch.Tensor,              # (B, 4, H, W)
        burn_age:   torch.Tensor,              # (B, 3, H, W)
        landcover:  torch.Tensor,              # (B, 11, H, W)
    ) -> torch.Tensor:
        """
        Returns: (B, num_classes, H, W) fire-prediction logits
        """
        B = hls.shape[0]

        # -- Backbone: extract 4 multi-scale features ---------------------
        # Prithvi expects (B, C, T, H, W) -- hls is already in that format
        multi_scale = self.backbone.get_intermediate_layers(hls, n_last=4)
        # multi_scale: [layer5 (shallowest), layer11, layer17, layer23 (deepest)]
        # Reverse so skip_maps[0]=deepest (16x16), skip_maps[3]=shallowest (128x128)
        # This is standard UNet: deepest features at coarsest resolution,
        # shallowest features provide high-res skip connections.
        reversed_feats = list(reversed(multi_scale))
        skip_maps = [
            proj(self._tokens_to_2d(feat))
            for proj, feat in zip(self.skip_projs, reversed_feats)
        ]
        # skip_maps[i]: (B, decoder_channels[i], G, G)

        # -- Weather conditioning ------------------------------------------
        w_curr = self.weather_enc(weather)             # (B, cond_dim)
        w_7d   = self.forecast_enc(weather_7d)         # (B, cond_dim)
        cond   = self.cond_fuse(torch.cat([w_curr, w_7d], dim=1))  # (B, cond_dim)

        # -- Spatial auxiliary features ------------------------------------
        # We need target sizes for each decoder scale
        # Scale sizes: 16, 32, 64, 128 (the decoder progressively upsamples)
        G = self.grid_size   # 16
        target_sizes = [(G, G)]
        for i in range(1, self.n_scales):
            target_sizes.append((G * (2 ** i), G * (2 ** i)))
        aux_maps = self.spatial_aux(terrain, landcover, burn_age, target_sizes)

        # -- Decoder: first block processes raw backbone output ------------
        # Initial feature: the deepest backbone feature (layer 23) -> 2D
        x = self._tokens_to_2d(multi_scale[-1])   # (B, embed_dim, G, G)

        for i, block in enumerate(self.decoder_blocks):
            x = block(x, skip_maps[i], aux_maps[i], cond)

        # -- Final upsample to full resolution ----------------------------
        x = self.final_up(x)   # (B, 64, 2*last_scale_H, 2*last_scale_W)
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)

        logits = self.classifier(x)   # (B, num_classes, H, W)

        # Multi-task regression mode: return dict with both heads
        if self.num_classes == 1:
            dnbr = F.relu(logits.squeeze(1))           # (B, H, W) non-negative dNBR
            fire_logit = self.fire_head(x).squeeze(1)  # (B, H, W) fire detection logit
            return {"dnbr": dnbr, "fire_logit": fire_logit}

        return logits

    # -- Cached Forward (precomputed frozen features) ----------------------

    def forward_cached(
        self,
        cached_feat: torch.Tensor,           # (B, N_tokens, embed_dim)
        weather:     torch.Tensor,            # (B, 10)
        weather_7d:  torch.Tensor,            # (B, 10, 7)
        terrain:     torch.Tensor,            # (B, 4, H, W)
        burn_age:    torch.Tensor,            # (B, 3, H, W)
        landcover:   torch.Tensor,            # (B, 11, H, W)
    ) -> torch.Tensor:
        """
        Forward pass using precomputed backbone features from blocks 0-27.
        Runs only blocks 28-31 + decoder. ~4.75x faster than full forward.

        cached_feat: output of block 27, shape (B, 197, 1280) for 600M
        """
        inner = self.backbone._backbone
        detach_at = getattr(self, "_detach_prefix_n", 28)
        depth = len(inner.blocks)

        # Run trainable blocks (28-31), collecting intermediate outputs
        x = cached_feat.detach().requires_grad_(True)
        multi_scale = []
        for i in range(detach_at, depth):
            x = inner.blocks[i](x)
            multi_scale.append(x)

        # Apply norm to the final block output (matches TerraTorch behavior)
        if hasattr(inner, "norm"):
            multi_scale[-1] = inner.norm(multi_scale[-1])

        # Build skip maps from trainable block outputs [28, 29, 30, 31]
        # Pad if fewer blocks than scales (shouldn't happen for depth=32, detach=28)
        while len(multi_scale) < self.n_scales:
            multi_scale.insert(0, multi_scale[0])

        reversed_feats = list(reversed(multi_scale[-self.n_scales:]))
        skip_maps = [
            proj(self._tokens_to_2d(feat))
            for proj, feat in zip(self.skip_projs, reversed_feats)
        ]

        # Weather conditioning (same as full forward)
        w_curr = self.weather_enc(weather)
        w_7d   = self.forecast_enc(weather_7d)
        cond   = self.cond_fuse(torch.cat([w_curr, w_7d], dim=1))

        # Spatial auxiliary features
        G = self.grid_size
        target_sizes = [(G, G)]
        for i in range(1, self.n_scales):
            target_sizes.append((G * (2 ** i), G * (2 ** i)))
        aux_maps = self.spatial_aux(terrain, landcover, burn_age, target_sizes)

        # Decoder
        x = self._tokens_to_2d(multi_scale[-1])
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, skip_maps[i], aux_maps[i], cond)

        # Final upsample
        x = self.final_up(x)
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)

        logits = self.classifier(x)

        if self.num_classes == 1:
            dnbr = F.relu(logits.squeeze(1))
            fire_logit = self.fire_head(x).squeeze(1)
            return {"dnbr": dnbr, "fire_logit": fire_logit}

        return logits

    # -- LoRA Injection ----------------------------------------------------

    _lora_active: bool = False

    def inject_lora(
        self,
        rank:           int = 16,
        alpha:          int = 32,
        dropout:        float = 0.05,
        target_modules: Optional[List[str]] = None,
    ) -> int:
        """Inject LoRA adapters into backbone and freeze base weights.

        Uses HuggingFace ``peft`` to add low-rank adapters to the ViT's
        attention projections.  The backbone base weights are frozen; only
        LoRA matrices A/B (plus decoder + FiLM) are trainable.

        Returns the number of trainable LoRA parameters added.
        """
        from peft import LoraConfig, get_peft_model

        if target_modules is None:
            target_modules = ["attn.qkv", "attn.proj"]

        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )

        # Wrap the inner backbone (the actual ViT, not our PrithviEO2Backbone)
        self.backbone._backbone = get_peft_model(
            self.backbone._backbone, lora_cfg
        )
        self._lora_active = True

        # Remove gradient checkpointing: peft's LoraLinear adds extra
        # matmul ops (dropout + A + B + scale) inside each attention block,
        # which pushes the per-block checkpoint recomputation over DirectML's
        # TDR (Timeout Detection and Recovery) limit (~2s), causing device
        # suspension.  Without grad ckpt, the backward uses stored activations
        # (more VRAM but no recomputation, avoiding the TDR crash).
        inner = self.backbone._backbone
        if hasattr(inner, "base_model"):
            inner = inner.base_model.model  # unwrap PeftModel
        if hasattr(inner, "blocks"):
            n_unpatched = 0
            for block in inner.blocks:
                if "forward" in block.__dict__:
                    del block.__dict__["forward"]
                    n_unpatched += 1
            if n_unpatched:
                log.info(f"[LoRA] Removed gradient checkpointing from "
                         f"{n_unpatched} blocks (avoids DML TDR crash)")

        # Count trainable LoRA params
        lora_trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
        lora_frozen = sum(
            p.numel() for p in self.backbone.parameters() if not p.requires_grad
        )
        log.info(
            f"[LoRA] Injected: rank={rank}, alpha={alpha}, "
            f"targets={target_modules}"
        )
        log.info(
            f"[LoRA] Backbone: {lora_trainable:,} trainable (LoRA) + "
            f"{lora_frozen:,} frozen (base)"
        )
        return lora_trainable

    def merge_lora(self) -> None:
        """Merge LoRA weights into backbone for zero-overhead inference."""
        if self._lora_active and hasattr(self.backbone._backbone, "merge_and_unload"):
            self.backbone._backbone = self.backbone._backbone.merge_and_unload()
            self._lora_active = False
            log.info("[LoRA] Merged adapters into backbone weights")

    # -- Detach Prefix (DML stability) -------------------------------------

    _detach_prefix_n: int = 0

    def set_detach_prefix(self, n: int) -> int:
        """Freeze backbone blocks 0..n-1 and add detach boundary at block n.

        Blocks 0..n-1 run in torch.no_grad() during forward (no backward).
        Block n receives a detached input, starting a fresh grad graph.
        This eliminates backward computation through the frozen prefix,
        reducing GPU load by ~75% for a 32-block ViT.

        Also freezes patch_embed / pos_embed / cls_token parameters so
        the entire prefix is gradient-free.

        Returns the number of blocks in the no_grad prefix.
        """
        bb = self.backbone._backbone
        if not hasattr(bb, "blocks"):
            log.warning("[DetachPrefix] Backbone has no 'blocks' -- skipped")
            return 0

        depth = len(bb.blocks)
        n = min(n, depth)
        if n <= 0:
            return 0

        # 1. Freeze params: patch_embed + positional embeds + blocks 0..n-1
        frozen_count = 0
        for name, p in bb.named_parameters():
            is_prefix_block = any(
                name.startswith(f"blocks.{i}.") for i in range(n)
            )
            is_embed = any(k in name for k in [
                "patch_embed", "pos_embed", "cls_token",
                "temporal_embed", "location_embed",
            ])
            if is_prefix_block or is_embed:
                p.requires_grad = False
                frozen_count += 1

        trainable_bb = sum(1 for p in bb.parameters() if p.requires_grad)
        log.info(f"[DetachPrefix] Froze {frozen_count} params "
                 f"(blocks 0-{n-1} + embeddings), "
                 f"{trainable_bb} backbone params remain trainable")

        # 2. Patch forward methods: no_grad for 0..n-1, detach at n
        _patch_detach_prefix(bb, n)

        self._detach_prefix_n = n
        return n


    # -- Utilities ---------------------------------------------------------

    def predict_with_uncertainty(self, *args, mc_samples: int = 10, **kwargs) -> tuple:
        """
        Monte Carlo Dropout inference for uncertainty mapping.
        Returns: (mean_prediction, uncertainty_std)
        """
        is_training = self.training
        self.train()  # Enable dropout layers

        preds = []
        with torch.no_grad():
            for _ in range(mc_samples):
                out = self.forward(*args, **kwargs)
                if isinstance(out, dict):
                    preds.append(out["dnbr"] if "dnbr" in out else list(out.values())[0])
                else:
                    preds.append(out)

        preds = torch.stack(preds)  # (mc_samples, B, ...)
        mean_pred = preds.mean(dim=0)
        uncertainty = preds.std(dim=0)

        if not is_training:
            self.eval()

        return mean_pred, uncertainty

    def _log_params(self) -> None:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone  = sum(p.numel() for p in self.backbone.parameters())
        decoder   = total - backbone
        log.info(
            f"VanAgni parameters: {total:,} total  "
            f"({trainable:,} trainable)  "
            f"backbone={backbone:,}  decoder={decoder:,}"
        )

    def get_param_groups(
        self,
        base_lr: float,
        llrd_decay: float = 0.0,
    ) -> list:
        """Return parameter groups with differential learning rates.

        Modes:
          - llrd_decay > 0:  Layer-Wise LR Decay.  Each backbone block gets
            ``base_lr * backbone_lr_scale * decay^(depth-1-i)`` where deeper
            layers get higher LR.  All params stay requires_grad=True (DML
            compatible -- avoids the partial-freeze bug in DirectML).
          - LoRA active:  3 groups (lora / film / decoder).
          - Otherwise:    legacy 4 groups (backbone_early/late / film / decoder).

        Args:
            base_lr:    Base learning rate for decoder.
            llrd_decay: Per-layer LR decay factor (0.0 = disabled).
                        Typical: 0.65-0.75 for ViT-L.
        """
        # -- Collect FiLM + weather params (same for all modes) ------------
        film_params = (
            list(self.weather_enc.parameters()) +
            list(self.forecast_enc.parameters()) +
            list(self.cond_fuse.parameters())
        )
        for block in self.decoder_blocks:
            film_params.extend(list(block.film.parameters()))
        film_ids = {id(p) for p in film_params}

        # -- Backbone params -----------------------------------------------
        backbone_params = list(self.backbone.parameters())
        bb_ids = {id(p) for p in backbone_params}

        # -- Decoder (everything not backbone or film) ---------------------
        decoder_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in film_ids and id(p) not in bb_ids
        ]

        # =================================================================
        # LLRD: Layer-Wise LR Decay
        # =================================================================
        if llrd_decay > 0:
            groups = self._build_llrd_groups(base_lr, llrd_decay)
            groups.append({"params": film_params,    "lr": base_lr * 2.0,
                           "name": "film_weather"})
            groups.append({"params": decoder_params, "lr": base_lr * 1.0,
                           "name": "decoder"})

        # =================================================================
        # LoRA mode
        # =================================================================
        elif self._lora_active:
            lora_params = [p for p in backbone_params if p.requires_grad]
            groups = [
                {"params": lora_params,    "lr": base_lr * 1.0,  "name": "lora"},
                {"params": film_params,    "lr": base_lr * 2.0,  "name": "film_weather"},
                {"params": decoder_params, "lr": base_lr * 1.0,  "name": "decoder"},
            ]

        # =================================================================
        # Legacy: coarse 2-group backbone split
        # =================================================================
        else:
            n_bb = len(backbone_params)
            mid  = n_bb // 2
            early_bb = [p for p in backbone_params[:mid] if p.requires_grad]
            late_bb  = [p for p in backbone_params[mid:] if p.requires_grad]
            groups = [
                {"params": early_bb,       "lr": base_lr * 0.01,
                 "name": "backbone_early"},
                {"params": late_bb,        "lr": base_lr * 0.05,
                 "name": "backbone_late"},
                {"params": film_params,    "lr": base_lr * 2.0,
                 "name": "film_weather"},
                {"params": decoder_params, "lr": base_lr * 1.0,
                 "name": "decoder"},
            ]

        for g in groups:
            n = sum(p.numel() for p in g["params"])
            log.info(f"  Param group '{g['name']}': {n:,} params, lr={g['lr']:.1e}")

        return groups

    def _build_llrd_groups(
        self,
        base_lr: float,
        decay: float,
    ) -> list:
        """Create per-layer backbone param groups with exponential LR decay.

        Layer 0 (shallowest) gets the smallest LR:
            lr_i = base_lr * backbone_scale * decay^(depth-1-i)

        The backbone_scale (0.1) ensures even the deepest layer trains
        slower than the decoder, preventing catastrophic forgetting.

        When detach_prefix is active, frozen blocks are skipped entirely
        (no optimizer entries = no wasted state memory for ~474M params).
        """
        bb = self.backbone._backbone
        depth = len(bb.blocks) if hasattr(bb, "blocks") else 32
        backbone_scale = 0.1  # backbone deepest layer = 10% of base_lr
        prefix_n = self._detach_prefix_n

        groups = []

        # Patch embed + positional embeddings (shallowest, lowest LR)
        embed_params = [
            p for name, p in bb.named_parameters()
            if p.requires_grad and any(k in name for k in [
                "patch_embed", "pos_embed", "cls_token",
                "temporal_embed", "location_embed",
            ])
        ]
        if embed_params:
            embed_lr = base_lr * backbone_scale * (decay ** depth)
            groups.append({"params": embed_params, "lr": embed_lr,
                           "name": "bb_embed"})

        # Per-block groups (skip frozen prefix blocks)
        for i, block in enumerate(bb.blocks):
            params = [p for p in block.parameters() if p.requires_grad]
            if not params:
                continue  # frozen block — skip entirely
            layer_lr = base_lr * backbone_scale * (decay ** (depth - 1 - i))
            groups.append({"params": params, "lr": layer_lr,
                           "name": f"bb_block_{i:02d}"})

        # Final norm (deepest, highest backbone LR)
        norm_params = [
            p for name, p in bb.named_parameters()
            if p.requires_grad and "norm" in name and "blocks" not in name
        ]
        if norm_params:
            norm_lr = base_lr * backbone_scale
            groups.append({"params": norm_params, "lr": norm_lr,
                           "name": "bb_norm"})

        # Summarize
        first_trainable = prefix_n if prefix_n > 0 else 0
        log.info(f"[LLRD] decay={decay}, depth={depth}, "
                 f"backbone_scale={backbone_scale}"
                 + (f", detach_prefix={prefix_n}" if prefix_n else ""))
        log.info(f"[LLRD] LR range: block_{first_trainable}="
                 f"{base_lr * backbone_scale * decay**(depth-1-first_trainable):.2e}"
                 f" -> block_{depth-1}={base_lr * backbone_scale:.2e}")

        return groups


# =============================================================================
# Factory
# =============================================================================

def build_vanaagni(cfg: dict) -> VanAgni:
    """
    Build VanAgni model from config dict.

    Expected cfg keys:
      model:
        eo2_variant:       "600M-TL"       # 300M | 300M-TL | 600M | 600M-TL
        backbone_init_from: null            # optional local .pt checkpoint
        freeze_backbone:    false
        decoder_channels:   [512, 256, 128, 64]
        weather_cond_dim:   128
        num_classes:        5               # 5=severity, 2=binary
        dropout:            0.1
    """
    m = cfg.get("model", cfg)

    model = VanAgni(
        eo2_variant       = m.get("eo2_variant", "600M-TL"),
        num_frames        = m.get("num_frames", 3),
        in_chans          = m.get("in_chans", 6),
        img_size          = m.get("img_size", 224),
        backbone_init_from= m.get("backbone_init_from", None),
        freeze_backbone   = m.get("freeze_backbone", False),
        decoder_channels  = m.get("decoder_channels", list(_DECODER_CH)),
        weather_input_dim = m.get("weather_input_dim", 10),
        weather_cond_dim  = m.get("weather_cond_dim", _WEATHER_DIM),
        num_classes       = m.get("num_classes", 5),
        dropout           = m.get("dropout", 0.1),
    )
    return model


# =============================================================================
# Quick Forward-Pass Test
# =============================================================================

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    logging.basicConfig(level=logging.INFO)

    NC = 5
    variant = "600M-TL"

    print("=" * 60)
    print(f"VanAgni -- Forward Pass Test ({variant}, {NC}-class severity)")
    print("=" * 60)

    cfg = {
        "model": {
            "eo2_variant":      variant,
            "num_frames":       3,
            "in_chans":         6,
            "img_size":         224,
            "freeze_backbone":  False,
            "decoder_channels": [512, 256, 128, 64],
            "weather_cond_dim": 128,
            "num_classes":      NC,
            "dropout":          0.1,
        }
    }

    model = build_vanaagni(cfg)

    B = 1
    hls        = torch.randn(B, 6, 3, 224, 224)
    weather    = torch.randn(B, 10)
    weather_7d = torch.randn(B, 10, 7)
    terrain    = torch.randn(B, 4, 224, 224)
    burn_age   = torch.randn(B, 3, 224, 224)
    landcover  = torch.randn(B, 11, 224, 224)

    with torch.no_grad():
        logits = model(hls, weather, weather_7d, terrain, burn_age, landcover)

    print(f"\nOutput shape: {logits.shape}  Expected: ({B}, {NC}, 224, 224)")
    print(f"Output range: [{logits.min():.3f}, {logits.max():.3f}]")
    assert logits.shape == (B, NC, 224, 224), f"Shape mismatch: {logits.shape}"

    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total:,}")
    print(f"Trainable parameters: {train:,}")
    sev_names = list(SEVERITY_NAMES.values())[:NC]
    print(f"Severity classes:     {NC} -> {sev_names}")

    print("\n[PASS] VanAgni forward-pass test passed")
    print("=" * 60)
