"""
Dataset Classes
===============
PyTorch Dataset wrappers for change detection and temporal prediction tasks.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple
import logging

log = logging.getLogger(__name__)


# ─── Augmentation pipeline ────────────────────────────────────────────────────

class SatelliteAugment:
    """
    Satellite-appropriate augmentations that preserve spatial–spectral
    consistency across all temporal frames AND the label mask.

    Augmentations applied:
      Spatial (applied identically to all frames + label):
        • Random H/V flip          (p=0.5 each)
        • Random 90° rotation      (p=0.5 — covers 0°/90°/180°/270°)
      Spectral (applied per-image, NOT to label):
        • Per-band additive Gaussian noise  σ ~ U(0, noise_std)
        • Per-band brightness shift         Δ ~ U(-bright_delta, +bright_delta)
        • Random band dropout               (zeroes one random band, p=band_drop_p)

    All operations work on raw numpy float32 arrays:
        image: (T, C, H, W)
        label: (H, W) int
    """

    def __init__(
        self,
        noise_std: float = 0.02,
        bright_delta: float = 0.05,
        band_drop_p: float = 0.15,
        flip_p: float = 0.5,
        rot_p: float = 0.5,
    ):
        self.noise_std = noise_std
        self.bright_delta = bright_delta
        self.band_drop_p = band_drop_p
        self.flip_p = flip_p
        self.rot_p = rot_p

    # ── Spatial ops (consistent across all T frames + label) ──────────────────

    def _hflip(self, img, lbl):
        return np.flip(img, axis=-1).copy(), np.flip(lbl, axis=-1).copy()

    def _vflip(self, img, lbl):
        return np.flip(img, axis=-2).copy(), np.flip(lbl, axis=-2).copy()

    def _rot90(self, img, lbl, k):
        # img: (T, C, H, W) — rotate in (H, W) plane
        img = np.rot90(img, k=k, axes=(-2, -1)).copy()
        lbl = np.rot90(lbl, k=k, axes=(-2, -1)).copy()
        return img, lbl

    # ── Spectral ops (image only) ──────────────────────────────────────────────

    def _add_noise(self, img):
        if self.noise_std > 0:
            sigma = np.random.uniform(0, self.noise_std)
            img = img + np.random.randn(*img.shape).astype(np.float32) * sigma
        return img

    def _brightness(self, img):
        # Independent shift per band (C dimension = axis 1)
        T, C, H, W = img.shape
        delta = np.random.uniform(-self.bright_delta, self.bright_delta, size=(1, C, 1, 1)).astype(np.float32)
        return img + delta

    def _band_dropout(self, img):
        if np.random.rand() < self.band_drop_p:
            c_idx = np.random.randint(0, img.shape[1])
            img = img.copy()
            img[:, c_idx, :, :] = 0.0
        return img

    # ── Main call ──────────────────────────────────────────────────────────────

    def __call__(self, image: np.ndarray, label: np.ndarray):
        """
        Args:
            image: (T, C, H, W) float32
            label: (H, W) int

        Returns:
            image, label  (same shapes, augmented)
        """
        # ── Spatial ──────────────────────────────────────────────────────────
        if np.random.rand() < self.flip_p:
            image, label = self._hflip(image, label)
        if np.random.rand() < self.flip_p:
            image, label = self._vflip(image, label)
        if np.random.rand() < self.rot_p:
            k = np.random.choice([1, 2, 3])
            image, label = self._rot90(image, label, k)

        # ── Spectral ─────────────────────────────────────────────────────────
        image = self._add_noise(image)
        image = self._brightness(image)
        image = self._band_dropout(image)

        return image, label


# ─── Datasets ─────────────────────────────────────────────────────────────────

class ForestChangeDataset(Dataset):
    """
    PyTorch Dataset for bi-temporal / multi-temporal forest change detection.

    Each sample:
        image: (T, C, H, W) float32 tensor  — T temporal frames, C=6 bands
        label: (H, W) long tensor           — 0=no change, 1=forest loss

    If oversample_positives=True, positive (change) patches are sampled
    at 2× the rate of negatives to help break class collapse.
    """

    def __init__(
        self,
        manifest_csv: str,
        transform: Optional[Callable] = None,
        augment: bool = False,
        use_last_two_frames_only: bool = False,
        oversample_positives: bool = False,
        aug_cfg: Optional[dict] = None,
    ):
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform
        self.use_last_two = use_last_two_frames_only

        # Build augmenter (from aug config if provided, else sensible defaults)
        if augment:
            cfg = aug_cfg or {}
            self.augmenter = SatelliteAugment(
                noise_std    = cfg.get("noise_std",    0.02),
                bright_delta = cfg.get("bright_delta", 0.05),
                band_drop_p  = cfg.get("band_drop_p",  0.15),
                flip_p       = cfg.get("flip_p",       0.5),
                rot_p        = cfg.get("rot_p",        0.5),
            )
        else:
            self.augmenter = None

        # Build index (with optional positive oversampling)
        pos_rows  = self.df[self.df["has_change"] == 1]
        neg_rows  = self.df[self.df["has_change"] == 0]

        if oversample_positives and len(pos_rows) > 0:
            # Duplicate positive patches 2× to break class collapse
            # (was incorrectly `* 1` which is a no-op)
            oversample_ratio = (aug_cfg or {}).get("oversample_ratio", 2)
            extra = pd.concat([pos_rows] * oversample_ratio, ignore_index=True)
            self.df = pd.concat([self.df, extra], ignore_index=True).sample(
                frac=1, random_state=42
            ).reset_index(drop=True)

        log.info(f"Dataset: {len(self.df)} samples from {manifest_csv}")
        log.info(f"  Positive (change) samples: {self.df['has_change'].sum()}")
        if oversample_positives:
            log.info("  Positive oversampling: ON")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image = np.load(row["img_path"])   # (T, C, H, W)
        label = np.load(row["lbl_path"])   # (H, W)

        # Optionally use only last 2 frames (simpler change detection)
        if self.use_last_two and image.shape[0] > 2:
            image = image[-2:]  # (2, C, H, W)

        if self.augmenter is not None:
            image, label = self.augmenter(image, label)

        image = torch.from_numpy(image.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights to handle class imbalance."""
        n_neg = (self.df["has_change"] == 0).sum()
        n_pos = (self.df["has_change"] == 1).sum()
        total = n_neg + n_pos
        w_neg = total / (2 * n_neg + 1e-6)
        w_pos = total / (2 * n_pos + 1e-6)
        return torch.tensor([w_neg, w_pos], dtype=torch.float32)


class TemporalPredictionDataset(Dataset):
    """
    Dataset for temporal prediction:
    Input: sequence of N frames, Output: change probability at T+1

    Builds sequences from the manifest by grouping consecutive year patches.
    """

    def __init__(
        self,
        manifest_csv: str,
        num_time_steps: int = 6,
        augment: bool = False,
        aug_cfg: Optional[dict] = None,
    ):
        self.df = pd.read_csv(manifest_csv)
        self.num_time_steps = num_time_steps

        if augment:
            cfg = aug_cfg or {}
            import inspect as _inspect
            _valid_aug_params = set(_inspect.signature(SatelliteAugment.__init__).parameters) - {"self"}
            self.augmenter = SatelliteAugment(**{k: cfg[k] for k in cfg if k in _valid_aug_params})
        else:
            self.augmenter = None

        # Group by spatial position
        self.df["pos"] = self.df["row"].astype(str) + "_" + self.df["col"].astype(str)
        self.sequences = self._build_sequences()
        log.info(f"TemporalPrediction: {len(self.sequences)} sequences from {manifest_csv}")

    def _build_sequences(self):
        """Build sequences of consecutive temporal patches at same spatial position."""
        sequences = []
        for pos, group in self.df.groupby("pos"):
            group = group.sort_values("label_year")
            if len(group) < self.num_time_steps + 1:
                continue
            for i in range(len(group) - self.num_time_steps):
                input_rows = group.iloc[i: i + self.num_time_steps]
                target_row = group.iloc[i + self.num_time_steps]
                sequences.append((input_rows, target_row))
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_rows, target_row = self.sequences[idx]

        # Load input frames: each is (T, C, H, W) — take last frame of each
        frames = []
        for _, row in input_rows.iterrows():
            img = np.load(row["img_path"])  # (T, C, H, W)
            frames.append(img[-1])  # last frame → (C, H, W)

        # Stack into (N, C, H, W)
        image_seq = np.stack(frames, axis=0)

        # Load target: future change label
        target_lbl = np.load(target_row["lbl_path"])  # (H, W)

        if self.augmenter is not None:
            image_seq, target_lbl = self.augmenter(image_seq, target_lbl)

        image_seq = torch.from_numpy(image_seq.copy()).float()
        target_lbl = torch.from_numpy(target_lbl.copy()).long()

        return image_seq, target_lbl
