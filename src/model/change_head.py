"""
Change Detection Head
=====================
UperNet-style segmentation head for bi-temporal forest change detection.
Takes multi-scale features from Prithvi encoder -> outputs change mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ConvBNReLU(nn.Sequential):
    """Conv2d + BatchNorm + ReLU building block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class FPN(nn.Module):
    """
    Feature Pyramid Network - aggregates multi-scale features from ViT.
    Converts ViT token sequences to 2D feature maps.
    """

    def __init__(self, embed_dim: int = 768, fpn_out_dim: int = 256,
                 grid_size: int = 14, num_frames: int = 3, num_scales: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_frames = num_frames
        self.num_scales = num_scales

        # Lateral connections: project each scale's tokens to fpn_out_dim
        self.laterals = nn.ModuleList([
            nn.Conv2d(embed_dim, fpn_out_dim, kernel_size=1)
            for _ in range(num_scales)
        ])

        # Top-down connections
        self.fpn_convs = nn.ModuleList([
            ConvBNReLU(fpn_out_dim, fpn_out_dim)
            for _ in range(num_scales)
        ])

        # Final channel-wise aggregation
        self.merge = nn.Sequential(
            ConvBNReLU(fpn_out_dim * num_scales, fpn_out_dim * 2),
            nn.Dropout2d(0.1),
        )
        self.out_dim = fpn_out_dim * 2

    def tokens_to_2d(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Reshape ViT tokens back to 2D spatial feature maps.
        tokens: (B, 1+T*N, embed_dim) -> (B, embed_dim, H, W)
        """
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)

        B, N_total, C = tokens.shape
        H = W = self.grid_size
        
        # Remove cls token
        spatial_tokens = tokens[:, 1:, :]  # (B, N_spatial, C)
        N_spatial = spatial_tokens.shape[1]

        N = self.grid_size * self.grid_size  # spatial patches per frame
        T_actual = N_spatial // N

        if T_actual == 0:
             return spatial_tokens.mean(dim=1).view(B, C, 1, 1).expand(-1, -1, H, W)

        # Reshape: (B, T, N, C)
        spatial_tokens = spatial_tokens[:, :T_actual * N, :].reshape(B, T_actual, N, C)
        
        # Enhance temporal features: use mean AND diff between last and first frame
        if T_actual >= 2:
            diff = (spatial_tokens[:, -1] - spatial_tokens[:, 0]).abs()
            spatial_tokens = (spatial_tokens.mean(dim=1) + diff) / 2
        else:
            spatial_tokens = spatial_tokens.mean(dim=1)
            
        # Now spatial_tokens is (B, N, C)
        # Reshape to 2D: (B, C, H, W)
        spatial_2d = spatial_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return spatial_2d

    def forward(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        # Convert tokens to 2D
        maps = [self.tokens_to_2d(f) for f in feature_list]

        # Get target spatial size (from first map)
        target_h, target_w = maps[0].shape[-2:]

        # Apply lateral convolutions
        laterals = [lat(m) for lat, m in zip(self.laterals, maps)]

        # Upsample to same resolution and fuse
        fused = []
        for feat in laterals:
            if feat.shape[-2:] != (target_h, target_w):
                feat = F.interpolate(feat, size=(target_h, target_w),
                                     mode="bilinear", align_corners=False)
            fused.append(feat)

        # Apply FPN convolutions
        fused = [conv(f) for conv, f in zip(self.fpn_convs, fused)]

        # Concatenate and merge
        merged = self.merge(torch.cat(fused, dim=1))
        return merged


class ChangeDetectionHead(nn.Module):
    """
    Segmentation head for forest change detection.
    Takes Prithvi multi-scale features -> binary/multi-class change mask.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        fpn_out_dim: int = 256,
        num_classes: int = 2,
        img_size: int = 224,
        grid_size: int = 14,
        num_frames: int = 3,
        num_scales: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.grid_size = grid_size

        # Feature pyramid
        self.fpn = FPN(
            embed_dim=embed_dim,
            fpn_out_dim=fpn_out_dim,
            grid_size=grid_size,
            num_frames=num_frames,
            num_scales=num_scales,
        )

        # Decoder head: progressive upsampling
        in_ch = self.fpn.out_dim
        self.decoder = nn.Sequential(
            ConvBNReLU(in_ch, 256),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(256, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(128, 64),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(64, 64),
            nn.Dropout2d(dropout),
        )

        # Final classifier
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, feature_list: List[torch.Tensor],
                target_size: Optional[int] = None) -> torch.Tensor:
        target_size = target_size or self.img_size
        x = self.fpn(feature_list)
        x = self.decoder(x)

        if x.shape[-1] != target_size:
            x = F.interpolate(x, size=(target_size, target_size),
                              mode="bilinear", align_corners=False)

        logits = self.classifier(x)
        return logits


class DiceLoss(nn.Module):
    """Soft Dice Loss for segmentation."""
    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = predictions.shape[1]
        probs = F.softmax(predictions, dim=1)

        B, H, W = targets.shape
        targets_clamped = targets.clamp(0).long()
        targets_one_hot = torch.zeros(B, num_classes, H, W,
                                      dtype=torch.float32,
                                      device=predictions.device)
        targets_one_hot.scatter_(1, targets_clamped.unsqueeze(1), 1.0)

        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for severe pixel imbalance."""
    def __init__(self, gamma: float = 2.0, class_weights: Optional[torch.Tensor] = None,
                 ignore_index: int = -1):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            predictions, targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.exp(-ce)
        focal_loss = (1.0 - pt) ** self.gamma * ce
        return focal_loss.mean()


class FocalDiceLoss(nn.Module):
    """Focal Loss + Dice Loss."""
    def __init__(self, class_weights: Optional[torch.Tensor] = None,
                 gamma: float = 2.0, dice_weight: float = 0.5):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, class_weights=class_weights)
        self.dice  = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.dice_weight       * self.dice(logits, targets) +
                (1 - self.dice_weight) * self.focal(logits, targets))


class CombinedLoss(FocalDiceLoss):
    pass

