"""
src/model/film_layers.py
========================
Feature-wise Linear Modulation (FiLM) conditioning layers for VanAgni.

FiLM injects scalar weather / FWI context into every spatial decoder scale by
learning per-channel affine transforms:   x' = gamma(cond) * Norm(x) + beta(cond)

Initialisation follows adaLN-Zero (Peebles & Xie, DiT 2023): gamma->1, beta->0
so the conditioning is identity at the start of training and the decoder
behaves as an ordinary UNet until the FiLM branches warm up.

References:
  - FiLM:  Perez et al. (2018) arXiv:1709.07871
  - adaLN-Zero: Peebles & Xie (2023) arXiv:2212.09748
  - Flamingo cross-attention: Alayrac et al. (2022) arXiv:2204.14198
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# FiLM Block  (applied at every UNet decoder scale)
# =============================================================================

class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation with adaLN-Zero initialisation.

    Given a conditioning vector c (e.g. 128-dim weather embedding):
        gamma = Linear(c)       # (B, num_features)
        beta  = Linear(c)       # (B, num_features)
        x' = gamma[:,:,None,None] * GroupNorm(x) + beta[:,:,None,None]

    At init: gamma outputs constant 1 (weight=0, bias=1),
             beta  outputs constant 0 (weight=0, bias=0)
    -> identity transform -> decoder starts as a standard UNet.
    """

    def __init__(self, num_features: int, cond_dim: int = 128, num_groups: int = 8):
        super().__init__()
        self.norm  = nn.GroupNorm(min(num_groups, num_features), num_features)
        self.gamma = nn.Linear(cond_dim, num_features)
        self.beta  = nn.Linear(cond_dim, num_features)

        # adaLN-Zero init: identity at start
        # gamma(c) = W_g @ c + b_g  ->  W_g=0, b_g=1  ->  output = 1 for any c
        # beta(c)  = W_b @ c + b_b  ->  W_b=0, b_b=0  ->  output = 0 for any c
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, C, H, W)  decoder feature map
            cond: (B, cond_dim) conditioning vector
        Returns:
            (B, C, H, W) modulated feature map
        """
        x = self.norm(x)
        g = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        b = self.beta(cond).unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
        return g * x + b


# =============================================================================
# Weather Encoders
# =============================================================================

class WeatherEncoder(nn.Module):
    """
    MLP encoder for current-day FWI / meteorological scalars.

    Input:  (B, 10) -> [temp, rh, wind, precip, FFMC, DMC, DC, ISI, BUI, FWI]
    Output: (B, out_dim) conditioning vector
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """w: (B, 10) -> (B, out_dim)"""
        return self.mlp(w)


class ForecastWeatherEncoder(nn.Module):
    """
    Temporal encoder for weather look-back window.

    Input:  (B, 10, T) -> 10 weather features x T days
    Output: (B, out_dim)

    Architecture: Linear proj -> GRU -> final hidden state.
    This captures multi-day sequence dynamics and long-term memory.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        # Using GRU for better long-term drought memory instead of Conv1d
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, w7: torch.Tensor) -> torch.Tensor:
        """
        w7: (B, 10, T)
        Returns: (B, out_dim)
        """
        # (B, 10, T) -> transpose -> (B, T, 10)
        x = w7.permute(0, 2, 1)             # (B, T, 10)
        x = torch.nn.functional.gelu(self.proj(x))  # (B, T, hidden)
        _, (h_n, c_n) = self.gru(x) # Handle LSTM tuple return
        x = self.out_proj(h_n.squeeze(0))   # (B, out_dim)
        return x


# =============================================================================
# Spatial Auxiliary Encoder
# =============================================================================

class SpatialAuxEncoder(nn.Module):
    """
    Encodes terrain + land cover + burn-age rasters into multi-scale spatial
    features that are ADDED to the UNet decoder at each scale.

    Inputs (pre-computed, static per-patch):
      terrain   : (B, 4,  H, W)  -> elev_norm, slope_norm, sin_asp, cos_asp
      landcover : (B, 11, H, W)  -> ESA WorldCover one-hot (11 classes)
      burn_age  : (B, 3,  H, W)  -> recent / high-risk / mature fuel channels

    Total input: 4 + 11 + 3 = 18 channels

    Output: list of (B, out_ch, H_s, W_s) at each decoder scale s
    """

    def __init__(
        self,
        terrain_ch:    int = 4,
        lc_ch:         int = 11,
        burn_age_ch:   int = 3,
        hidden_ch:     int = 64,
        out_ch:        int = 32,
        decoder_scales: int = 4,
    ):
        super().__init__()
        in_ch = terrain_ch + lc_ch + burn_age_ch   # 18

        # Shared feature extraction at full resolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
        )

        # Per-scale projection (each decoder scale gets its own 1x1 conv)
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            for _ in range(decoder_scales)
        ])

    def forward(
        self,
        terrain:   torch.Tensor,
        landcover: torch.Tensor,
        burn_age:  torch.Tensor,
        target_sizes: list,            # [(H0, W0), (H1, W1), ...]
    ) -> list:
        """
        Returns list of (B, out_ch, H_s, W_s) feature maps, one per decoder scale.
        """
        x = torch.cat([terrain, landcover, burn_age], dim=1)   # (B, 18, H, W)
        x = self.stem(x)                                       # (B, hidden, H, W)

        out = []
        for proj, (h, w) in zip(self.scale_projs, target_sizes):
            xs = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
            out.append(proj(xs))   # (B, out_ch, h, w)

        return out
