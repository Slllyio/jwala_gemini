"""
Segmentation Metrics
====================
IoU, F1, Precision, Recall for binary/multi-class segmentation.

Supports three modes:
  - Binary (num_classes=2):  bg/fg naming, threshold-based prediction
  - Severity (num_classes=5): named classes, argmax prediction,
    plus aggregated "any fire" binary metrics for operational use
  - Regression (num_classes=1): MAE, RMSE, R-squared for continuous dNBR,
    plus threshold-based binary fire detection metrics
"""

import torch
import numpy as np
from typing import Dict, List, Optional


# Burn severity class names (aligned with ibm-nasa burn_intensity dataset)
SEVERITY_NAMES = ["no_burn", "very_low", "low", "moderate", "high"]


class SegmentationMetrics:
    """
    Tracks and computes segmentation metrics per epoch.
    Handles binary (2-class) and multi-class severity segmentation.
    """

    def __init__(self, num_classes: int = 5, ignore_index: int = -1,
                 class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        # Class names for logging
        if class_names is not None:
            self.class_names = class_names
        elif num_classes == 2:
            self.class_names = ["bg", "fg"]
        elif num_classes == 5:
            self.class_names = SEVERITY_NAMES[:5]
        else:
            self.class_names = [f"c{i}" for i in range(num_classes)]

    def reset(self):
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor,
                probs: Optional[torch.Tensor] = None, threshold: float = 0.5):
        """
        Update metrics from a batch.

        Args:
            preds:     (B, H, W) long tensor of class predictions (argmax)
            targets:   (B, H, W) long tensor of ground truth
            probs:     (B, C, H, W) float softmax probabilities (optional)
            threshold: if probs supplied AND binary mode (2 classes), use this
                       threshold for fg class instead of argmax
        """
        if probs is not None and self.num_classes == 2:
            # Binary mode: threshold on fg probability
            fg_prob = probs[:, 1, :, :]          # (B, H, W)
            preds = (fg_prob >= threshold).long() # (B, H, W)
        # Multi-class: always use argmax (threshold doesn't apply)

        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        # Remove ignored pixels (ignore_index can be -1 or 255)
        mask = (targets != self.ignore_index)
        # Also filter out-of-range values
        mask &= (targets >= 0) & (targets < self.num_classes)
        mask &= (preds >= 0) & (preds < self.num_classes)
        preds = preds[mask]
        targets = targets[mask]

        # Update confusion matrix -- vectorised (np.bincount >> Python for-loop)
        if len(targets) > 0:
            n = self.num_classes
            indices = targets * n + preds
            cm_flat = np.bincount(indices.astype(np.int64), minlength=n * n)
            self.confusion_matrix += cm_flat.reshape(n, n)

    def compute(self) -> Dict[str, float]:
        """Compute metrics from accumulated confusion matrix."""
        cm = self.confusion_matrix.astype(np.float64)
        scores: Dict[str, float] = {}

        # Per-class metrics with human-readable names
        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp  # predicted c but not c
            fn = cm[c, :].sum() - tp  # actually c but predicted other

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)

            name = self.class_names[c] if c < len(self.class_names) else f"c{c}"
            scores[f"precision_{name}"] = float(precision)
            scores[f"recall_{name}"]    = float(recall)
            scores[f"f1_{name}"]        = float(f1)
            scores[f"iou_{name}"]       = float(iou)

        # Mean IoU across foreground classes (class 0 = background excluded)
        fg_iou_keys = [f"iou_{self.class_names[c]}"
                       for c in range(1, self.num_classes)
                       if c < len(self.class_names)]
        fg_ious = [scores[k] for k in fg_iou_keys if k in scores]
        scores["mean_iou"] = float(np.mean(fg_ious)) if fg_ious else 0.0
        scores["pixel_accuracy"] = float(np.diag(cm).sum() / (cm.sum() + 1e-8))

        # -- Aggregated "any fire" binary metrics (for operational use) --------
        # Collapse severity classes 1-4 into "fire" vs class 0 "no_burn"
        if self.num_classes > 2:
            tp_fire = cm[1:, 1:].sum()           # any severity predicted as any severity
            fp_fire = cm[0, 1:].sum()             # no_burn predicted as severity
            fn_fire = cm[1:, 0].sum()             # severity predicted as no_burn
            tn_fire = cm[0, 0]                    # no_burn predicted as no_burn

            p = tp_fire / (tp_fire + fp_fire + 1e-8)
            r = tp_fire / (tp_fire + fn_fire + 1e-8)
            scores["fire_precision"] = float(p)
            scores["fire_recall"]    = float(r)
            scores["fire_f1"]        = float(2 * p * r / (p + r + 1e-8))
            scores["fire_iou"]       = float(tp_fire / (tp_fire + fp_fire + fn_fire + 1e-8))

        return scores


class RegressionMetrics:
    """
    Tracks regression metrics for continuous dNBR prediction.

    Accumulates predictions and targets over batches, then computes:
      - MAE (Mean Absolute Error)
      - RMSE (Root Mean Squared Error)
      - R-squared (coefficient of determination)
      - fire_precision, fire_recall, fire_f1 (threshold-based binary detection)

    Only valid (non-NaN) pixels contribute to metrics.
    """

    def __init__(self, fire_threshold: float = 0.05):
        """
        Parameters
        ----------
        fire_threshold : dNBR threshold for binary fire detection metrics.
                         Pixels with dNBR > threshold are considered "fire".
        """
        self.fire_threshold = fire_threshold
        self.reset()

    def reset(self):
        self._sum_ae  = 0.0   # sum of absolute errors
        self._sum_se  = 0.0   # sum of squared errors
        self._sum_y   = 0.0   # sum of targets
        self._sum_y2  = 0.0   # sum of targets squared
        self._n       = 0     # total valid pixels
        # Binary fire detection counts (threshold-based)
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update from a batch.

        Args:
            preds:   (B, H, W) float predicted dNBR
            targets: (B, H, W) float ground truth dNBR (NaN = ignore)
        """
        preds_np   = preds.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()

        # Mask out NaN pixels (nodata)
        valid = np.isfinite(targets_np) & np.isfinite(preds_np)
        if not valid.any():
            return

        p = preds_np[valid]
        t = targets_np[valid]
        n = len(t)

        # Regression stats
        ae = np.abs(p - t)
        se = (p - t) ** 2
        self._sum_ae += float(ae.sum())
        self._sum_se += float(se.sum())
        self._sum_y  += float(t.sum())
        self._sum_y2 += float((t ** 2).sum())
        self._n      += n

        # Binary fire detection (thresholded)
        pred_fire = p > self.fire_threshold
        true_fire = t > self.fire_threshold
        self._tp += int((pred_fire & true_fire).sum())
        self._fp += int((pred_fire & ~true_fire).sum())
        self._fn += int((~pred_fire & true_fire).sum())
        self._tn += int((~pred_fire & ~true_fire).sum())

    def compute(self) -> Dict[str, float]:
        """Compute regression and binary fire detection metrics."""
        scores: Dict[str, float] = {}
        n = max(self._n, 1)

        # Regression metrics
        mae  = self._sum_ae / n
        rmse = (self._sum_se / n) ** 0.5
        scores["mae"]  = float(mae)
        scores["rmse"] = float(rmse)

        # R-squared: 1 - SS_res / SS_tot
        ss_res = self._sum_se
        mean_y = self._sum_y / n
        ss_tot = self._sum_y2 - n * mean_y * mean_y
        if ss_tot > 1e-8:
            scores["r2"] = float(1.0 - ss_res / ss_tot)
        else:
            scores["r2"] = 0.0

        # Binary fire detection metrics
        tp, fp, fn = self._tp, self._fp, self._fn
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        fire_iou = tp / (tp + fp + fn + 1e-8)

        scores["fire_precision"] = float(p)
        scores["fire_recall"]    = float(r)
        scores["fire_f1"]        = float(f1)
        scores["fire_iou"]       = float(fire_iou)

        # Use fire_f1 as "mean_iou" proxy for checkpoint selection
        # (higher = better, consistent with classification mode)
        # Previously used -MAE which rewards predicting 0 everywhere
        # on background-dominated data. fire_f1 forces the model to
        # actually detect burns to get a good checkpoint score.
        scores["mean_iou"] = float(f1)

        # Pixel accuracy: fraction of pixels within 0.05 of target
        scores["pixel_accuracy"] = float(1.0 - mae)  # approximate

        return scores
