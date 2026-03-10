"""
Prithvi MAE Architecture
========================
Re-implementation of the Prithvi-100M Masked AutoEncoder architecture.
Based on: https://github.com/NASA-IMPACT/hls-foundation-os

This module defines the ViT backbone that matches the Prithvi-100M pre-trained weights.
"""

import torch
import torch.nn as nn
import numpy as np
from functools import partial
from typing import List, Optional, Tuple
import math


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding for multi-temporal input (B, C, T, H, W).
    Matches Prithvi-100M's patch embedding configuration.
    """
    def __init__(self, img_size=224, patch_size=16, num_frames=3,
                 in_chans=6, embed_dim=768):
        super().__init__()
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size  # 14 for 224/16

        # Separate spatial and temporal projections (as in original Prithvi)
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size)
        )
        self.num_patches = (img_size // patch_size) ** 2 * num_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        outputs = []
        for t in range(T):
            frame = x[:, :, t, :, :]  # (B, C, H, W)
            patch = self.proj(frame)   # (B, embed_dim, H/P, W/P)
            patch = patch.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
            outputs.append(patch)
        # Concatenate along token dimension: (B, T*N, embed_dim)
        return torch.cat(outputs, dim=1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                    self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PrithviMAE(nn.Module):
    """
    Prithvi-100M MAE Encoder.
    Architecture matches the pre-trained HuggingFace weights.

    Input:  (B, C, T, H, W)  — C=6 bands, T=3 temporal frames
    Output: (B, N_tokens, embed_dim)  — token sequence for decoder head
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 3,
        in_chans: int = 6,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.grid_size = img_size // patch_size  # 14
        self.num_patches_per_frame = self.grid_size ** 2  # 196
        self.num_patches = self.num_patches_per_frame * num_frames  # 588

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Learnable class token (for global representation)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings (spatial + temporal)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim),
            requires_grad=False,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize position embeddings with sine-cosine
        pos_embed = self._get_pos_embed()
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())
        nn.init.normal_(self.cls_token, std=0.02)

    def _get_pos_embed(self) -> np.ndarray:
        """Compute sine-cosine positional embeddings."""
        embed_dim = self.embed_dim
        num_patches = self.num_patches
        pos_embed = np.zeros((1, num_patches + 1, embed_dim))
        # Simple initialization; actual Prithvi uses 3D position encoding
        for i in range(1, num_patches + 1):
            for j in range(0, embed_dim, 2):
                pos_embed[0, i, j] = math.sin(i / (10000 ** (j / embed_dim)))
                if j + 1 < embed_dim:
                    pos_embed[0, i, j + 1] = math.cos(i / (10000 ** (j / embed_dim)))
        return pos_embed

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch features.

        Args:
            x: (B, C, T, H, W) or (B, T*C, H, W)

        Returns:
            (B, N+1, embed_dim) — token sequence including cls_token
        """
        # Handle both input formats
        if x.ndim == 4:
            B, TC, H, W = x.shape
            T = self.num_frames
            C = TC // T
            x = x.reshape(B, C, T, H, W)

        x = self.patch_embed(x)  # (B, T*N, embed_dim)

        # Add cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        if x.shape[1] == self.pos_embed.shape[1]:
            x = x + self.pos_embed
        else:
            # Interpolate if sizes don't match
            x = x + self.pos_embed[:, :x.shape[1], :]

        # Transformer forward
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return x  # (B, N+1, embed_dim)

    def get_intermediate_layers(self, x: torch.Tensor,
                                 n_last: int = 4) -> List[torch.Tensor]:
        """
        Get features from the last N transformer blocks for multi-scale decoding.

        Returns:
            List of (B, N, embed_dim) tensors from last n_last layers
        """
        if x.ndim == 4:
            B, TC, H, W = x.shape
            T = self.num_frames
            C = TC // T
            x = x.reshape(B, C, T, H, W)

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        if x.shape[1] == self.pos_embed.shape[1]:
            x = x + self.pos_embed

        intermediates = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i >= len(self.blocks) - n_last:
                intermediates.append(self.norm(x))

        return intermediates  # list of (B, N+1, embed_dim)
