"""
Temporal Prediction Head
========================
ConvLSTM-based head that predicts future forest change probability
from a sequence of temporal Prithvi embeddings.

Architecture:
    Prithvi Encoder (per-frame) → ConvLSTM →
    Convolutional Decoder → Risk probability map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell for spatial-temporal modeling.
    Operates on 2D feature maps instead of 1D vectors.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        # Gates: input, forget, cell, output — all fused in one conv
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size=kernel_size, padding=padding, bias=True
        )

    def forward(self, x: torch.Tensor,
                h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim, H, W) current input
            h: (B, hidden_dim, H, W) previous hidden state
            c: (B, hidden_dim, H, W) previous cell state

        Returns:
            (h_next, c_next) tuples
        """
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)  # (B, 4*hidden_dim, H, W)
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)    # input gate
        f = torch.sigmoid(f)    # forget gate
        g = torch.tanh(g)       # cell gate
        o = torch.sigmoid(o)    # output gate

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size: int, h: int, w: int,
                     device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
        )


class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM for temporal sequence modeling."""

    def __init__(self, input_dim: int, hidden_dims: List[int], kernel_size: int = 3):
        super().__init__()
        self.num_layers = len(hidden_dims)
        self.cells = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims):
            in_dim = input_dim if i == 0 else hidden_dims[i - 1]
            self.cells.append(ConvLSTMCell(in_dim, h_dim, kernel_size))
        self.out_dim = hidden_dims[-1]

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: (B, T, C, H, W) sequence of spatial feature maps

        Returns:
            (B, hidden_dim, H, W) final hidden state representing temporal context
        """
        B, T, C, H, W = seq.shape
        device = seq.device

        # Initialize hidden states for each layer
        h_states = []
        c_states = []
        for cell in self.cells:
            h, c = cell.init_hidden(B, H, W, device)
            h_states.append(h)
            c_states.append(c)

        # Process sequence step by step
        for t in range(T):
            x = seq[:, t]  # (B, C, H, W)
            for i, cell in enumerate(self.cells):
                h, c = cell(x, h_states[i], c_states[i])
                h_states[i] = h
                c_states[i] = c
                x = h  # output of layer i is input to layer i+1

        return h_states[-1]  # Final hidden state from last layer


class TemporalPredictionHead(nn.Module):
    """
    Predicts future forest change probability from a time-series of
    Prithvi feature maps using ConvLSTM.

    Input flow:
        Per-frame features (B, T, embed_dim, H, W)
        → ConvLSTM temporal modeling
        → Convolutional decoder
        → (B, 1, H, W) change probability in [0, 1]
    """

    def __init__(
        self,
        embed_dim: int = 768,
        grid_size: int = 14,
        num_frames: int = 6,
        hidden_dims: Optional[List[int]] = None,
        img_size: int = 224,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.img_size = img_size

        # Project ViT tokens to 2D feature maps
        self.token_proj = nn.Linear(embed_dim, embed_dim)

        # ConvLSTM for temporal modeling
        self.convlstm = ConvLSTM(
            input_dim=embed_dim,
            hidden_dims=hidden_dims,
            kernel_size=3,
        )

        # Decoder to full resolution
        lstm_out_dim = hidden_dims[-1]
        self.decoder = nn.Sequential(
            nn.Conv2d(lstm_out_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        # Risk output: raw logits — do NOT apply sigmoid here.
        # BCEWithLogitsLoss in train.py uses the numerically stable log-sum-exp path,
        # which is critical under fp16 AMP to prevent log(0) → -inf gradients.
        # Apply torch.sigmoid() only at inference time (detect.py).
        self.risk_head = nn.Conv2d(32, 1, kernel_size=1)

    def tokens_to_2d(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert ViT token sequence to 2D spatial map.
        tokens: (B, 1+N, embed_dim) → (B, embed_dim, H, W)
        """
        spatial = tokens[:, 1:, :]   # remove cls token → (B, N, C)
        N = self.grid_size ** 2
        B, _, C = spatial.shape
        spatial = spatial[:, :N, :]  # take spatial tokens
        spatial = self.token_proj(spatial)
        spatial = spatial.permute(0, 2, 1).reshape(B, C, self.grid_size, self.grid_size)
        return spatial

    def forward(self, feature_sequence: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feature_sequence: List of T token tensors, each (B, N, embed_dim)
                              Output from PrithviMAE.forward_features() per timestep

        Returns:
            (B, 1, H, W) risk probability map in [0, 1]
        """
        # Convert each frame's tokens to 2D maps
        maps = [self.tokens_to_2d(t) for t in feature_sequence]  # T × (B, C, H, W)

        # Stack into sequence: (B, T, C, H, W)
        seq = torch.stack(maps, dim=1)

        # ConvLSTM temporal modeling → (B, hidden, H, W)
        temporal_ctx = self.convlstm(seq)

        # Decode to full resolution
        out = self.decoder(temporal_ctx)  # (B, 32, H', W')

        # Upsample to img_size if needed
        if out.shape[-1] != self.img_size:
            out = F.interpolate(out, size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False)

        risk_map = self.risk_head(out)  # (B, 1, H, W)
        return risk_map
