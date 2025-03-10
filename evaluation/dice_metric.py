"""
Dice coefficient and segmentation metrics for LUNA16 lung nodule evaluation.

All functions operate on PyTorch tensors (CPU or CUDA) with minimal overhead.
Functions accept batched inputs of shape (B, 1, H, W) or flat binary masks.

Run this file for a demo::

    python evaluation/dice_metric.py
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------


def _check_shapes(pred: Tensor, target: Tensor) -> None:
    """Raise ValueError if pred and target have mismatched shapes."""
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, "
            f"got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )


def compute_dice(
    pred: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    spatial_dims: Sequence[int] = (1, 2, 3),
) -> Tensor:
    """Compute per-sample Dice coefficient.

    Parameters
    ----------
    pred : Tensor
        Prediction logits or probabilities, shape (B, 1, H, W).
        If values are not in [0, 1], sigmoid is applied automatically.
    target : Tensor
        Binary ground-truth mask, shape (B, 1, H, W), values in {0, 1}.
    threshold : float
        Binarisation threshold for ``pred``. Default 0.5.
    smooth : float
        Laplace smoothing constant. Default 1e-6.
    spatial_dims : sequence of int
        Dimensions to reduce over. Default (1, 2, 3) = channel + spatial.

    Returns
    -------
    Tensor
        Per-sample Dice coefficient, shape (B,). Values ∈ [0, 1].
    """
    _check_shapes(pred, target)
    # Convert logits → probabilities if needed
    if pred.min() < 0.0 or pred.max() > 1.0:
        pred = torch.sigmoid(pred)
    binary = (pred >= threshold).float()
    target = target.float()

    dims = tuple(spatial_dims)
    inter = (binary * target).sum(dim=dims)
    union = binary.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + smooth) / (union + smooth)
    return dice


def compute_iou(
    pred: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    spatial_dims: Sequence[int] = (1, 2, 3),
) -> Tensor:
    """Compute per-sample Intersection-over-Union (Jaccard index).

    Parameters
    ----------
    pred : Tensor
        Prediction logits or probabilities, shape (B, 1, H, W).
    target : Tensor
        Binary ground-truth mask, shape (B, 1, H, W).
    threshold : float
        Binarisation threshold. Default 0.5.
    smooth : float
        Laplace smoothing. Default 1e-6.
    spatial_dims : sequence of int
        Dims to reduce over. Default (1, 2, 3).

    Returns
    -------
    Tensor
        Per-sample IoU, shape (B,). Values ∈ [0, 1].
    """
    _check_shapes(pred, target)
    if pred.min() < 0.0 or pred.max() > 1.0:
        pred = torch.sigmoid(pred)
    binary = (pred >= threshold).float()
    target = target.float()

    dims = tuple(spatial_dims)
    inter = (binary * target).sum(dim=dims)
    union = binary.sum(dim=dims) + target.sum(dim=dims) - inter
    iou = (inter + smooth) / (union + smooth)
    return iou


def compute_precision_recall(
    pred: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    spatial_dims: Sequence[int] = (1, 2, 3),
) -> Tuple[Tensor, Tensor]:
    """Compute per-sample precision and recall.

    Parameters
    ----------
    pred : Tensor
        Prediction logits or probabilities, shape (B, 1, H, W).
    target : Tensor
        Binary ground-truth mask, shape (B, 1, H, W).
    threshold : float
        Binarisation threshold. Default 0.5.
    smooth : float
        Laplace smoothing. Default 1e-6.
    spatial_dims : sequence of int
        Dims to reduce over. Default (1, 2, 3).

    Returns
    -------
    precision : Tensor
        Per-sample precision, shape (B,).
    recall : Tensor
        Per-sample recall (sensitivity), shape (B,).
    """
    _check_shapes(pred, target)
    if pred.min() < 0.0 or pred.max() > 1.0:
        pred = torch.sigmoid(pred)
    binary = (pred >= threshold).float()
    target = target.float()

    dims = tuple(spatial_dims)
    tp = (binary * target).sum(dim=dims)
    fp = (binary * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - binary) * target).sum(dim=dims)

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return precision, recall


def compute_hausdorff(
    pred: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    percentile: float = 95.0,
) -> List[float]:
    """Compute per-sample Hausdorff distance (95th percentile).

    Falls back to NumPy / SciPy for the distance transform computation.
    Works on CPU tensors only; CUDA tensors are moved to CPU automatically.

    Parameters
    ----------
    pred : Tensor
        Prediction, shape (B, 1, H, W). Logits or probabilities.
    target : Tensor
        Ground-truth binary mask, shape (B, 1, H, W).
    threshold : float
        Binarisation threshold for pred. Default 0.5.
    percentile : float
        Hausdorff percentile. 95.0 = HD95. Default 95.0.

    Returns
    -------
    list of float
        Per-sample HD95 (or np.inf if a mask is empty).
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        logger.warning("scipy not installed — Hausdorff skipped, returning inf")
        return [float("inf")] * pred.shape[0]

    if pred.min() < 0.0 or pred.max() > 1.0:
        pred = torch.sigmoid(pred)
    binary_np = (pred >= threshold).float().cpu().numpy()  # (B, 1, H, W)
    target_np = target.float().cpu().numpy()

    hd_list: List[float] = []
    for b in range(binary_np.shape[0]):
        p = binary_np[b, 0]  # (H, W)
        t = target_np[b, 0]  # (H, W)
        if p.sum() == 0 or t.sum() == 0:
            hd_list.append(float("inf"))
            continue
        dt_p = distance_transform_edt(1 - p)  # dist from every non-p pixel to p border
        dt_t = distance_transform_edt(1 - t)  # dist from every non-t pixel to t border
        # HD: symmetric surface distance
        dist_p_to_t = dt_t[p > 0.5]   # distances from p surface to t
        dist_t_to_p = dt_p[t > 0.5]   # distances from t surface to p
        all_dists = np.concatenate([dist_p_to_t, dist_t_to_p])
        hd = float(np.percentile(all_dists, percentile))
        hd_list.append(hd)

    return hd_list


# ---------------------------------------------------------------------------
# Stateful accumulator
# ---------------------------------------------------------------------------


class DiceMetric:
    """Streaming Dice coefficient accumulator over multiple batches.

    Maintains running per-sample sums so ``compute()`` returns the exact
    mean Dice over the entire dataset regardless of batch size.

    Parameters
    ----------
    threshold : float
        Binarisation threshold for predicted probabilities. Default 0.5.
    smooth : float
        Laplace smoothing constant. Default 1e-6.
    compute_hd95 : bool
        If True, also compute HD95 (requires scipy). Default False.

    Examples
    --------
    >>> metric = DiceMetric(threshold=0.5)
    >>> for batch in loader:
    ...     metric.update(batch["pred"], batch["mask"])
    >>> results = metric.compute()
    >>> print(results)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        smooth: float = 1e-6,
        compute_hd95: bool = False,
    ) -> None:
        self.threshold = threshold
        self.smooth = smooth
        self.compute_hd95 = compute_hd95
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated values."""
        self._dice_sum: float = 0.0
        self._iou_sum: float = 0.0
        self._prec_sum: float = 0.0
        self._rec_sum: float = 0.0
        self._hd95_sum: float = 0.0
        self._hd95_valid: int = 0
        self._n_samples: int = 0

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Accumulate metrics for a batch.

        Parameters
        ----------
        pred : Tensor
            Prediction logits or probabilities, shape (B, 1, H, W).
        target : Tensor
            Ground-truth binary mask, shape (B, 1, H, W).
        """
        dice = compute_dice(pred, target, self.threshold, self.smooth)
        iou = compute_iou(pred, target, self.threshold, self.smooth)
        prec, rec = compute_precision_recall(pred, target, self.threshold, self.smooth)

        self._dice_sum += dice.sum().item()
        self._iou_sum += iou.sum().item()
        self._prec_sum += prec.sum().item()
        self._rec_sum += rec.sum().item()
        self._n_samples += pred.shape[0]

        if self.compute_hd95:
            hd_vals = compute_hausdorff(pred, target)
            for hd in hd_vals:
                if hd != float("inf"):
                    self._hd95_sum += hd
                    self._hd95_valid += 1

    def compute(self) -> Dict[str, float]:
        """Return mean metrics over all accumulated samples.

        Returns
        -------
        dict with keys ``dice``, ``iou``, ``precision``, ``recall``,
        and optionally ``hd95``.
        """
        n = max(self._n_samples, 1)
        results: Dict[str, float] = {
            "dice": self._dice_sum / n,
            "iou": self._iou_sum / n,
            "precision": self._prec_sum / n,
            "recall": self._rec_sum / n,
            "n_samples": float(self._n_samples),
        }
        if self.compute_hd95:
            results["hd95"] = (
                self._hd95_sum / max(self._hd95_valid, 1)
                if self._hd95_valid > 0
                else float("inf")
            )
        return results


# ---------------------------------------------------------------------------
# Convenience all-in-one function
# ---------------------------------------------------------------------------


def compute_all_metrics(
    pred: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    include_hd95: bool = False,
) -> Dict[str, float]:
    """Compute Dice, IoU, precision, recall (and HD95) in one call.

    Parameters
    ----------
    pred : Tensor
        Predictions, shape (B, 1, H, W).
    target : Tensor
        Ground truth, shape (B, 1, H, W).
    threshold : float
        Binarisation threshold. Default 0.5.
    smooth : float
        Smoothing constant. Default 1e-6.
    include_hd95 : bool
        Also compute HD95 (slow, requires scipy). Default False.

    Returns
    -------
    dict
        ``dice``, ``iou``, ``precision``, ``recall``, optionally ``hd95``.
    """
    dice = compute_dice(pred, target, threshold, smooth).mean().item()
    iou = compute_iou(pred, target, threshold, smooth).mean().item()
    prec, rec = compute_precision_recall(pred, target, threshold, smooth)
    results = {
        "dice": dice,
        "iou": iou,
        "precision": prec.mean().item(),
        "recall": rec.mean().item(),
    }
    if include_hd95:
        hd_vals = compute_hausdorff(pred, target, threshold)
        finite_vals = [v for v in hd_vals if v != float("inf")]
        results["hd95"] = float(np.mean(finite_vals)) if finite_vals else float("inf")
    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("DiceMetric — sanity check")
    print("=" * 60)

    B, H, W = 4, 96, 96

    # Perfect prediction
    target = torch.zeros(B, 1, H, W)
    target[:, :, 30:60, 30:60] = 1.0  # 30×30 square nodule
    pred_perfect = target.clone() * 2.0 - 1.0  # convert to logits ≈ ±1

    metrics_perfect = compute_all_metrics(pred_perfect, target)
    print(f"\nPerfect prediction:")
    for k, v in metrics_perfect.items():
        print(f"  {k}: {v:.6f}")
    assert abs(metrics_perfect["dice"] - 1.0) < 1e-4, "Perfect dice should be ≈1"

    # Random prediction
    pred_random = torch.randn(B, 1, H, W)
    metrics_random = compute_all_metrics(pred_random, target)
    print(f"\nRandom prediction:")
    for k, v in metrics_random.items():
        print(f"  {k}: {v:.6f}")
    assert metrics_random["dice"] < 0.9, "Random dice should be < 0.9"

    # Streaming accumulator
    dm = DiceMetric(threshold=0.5)
    for _ in range(5):
        dm.update(pred_perfect, target)
    acc = dm.compute()
    print(f"\nDiceMetric accumulator ({dm._n_samples} samples):")
    for k, v in acc.items():
        print(f"  {k}: {v:.6f}")
    assert abs(acc["dice"] - 1.0) < 1e-4

    print("\nAll checks passed. ✓")
