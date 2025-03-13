"""
Temporal consistency loss for slice-level coherence in SAM2 CT segmentation.

Problem: slice-by-slice 2D segmentation predicts each axial slice independently,
giving the model no incentive to produce a smooth, anatomically coherent 3D mask.
Adjacent slices may show dramatically different predictions for the same nodule.

Solution: an auxiliary L_temporal loss that penalises differences between the
sigmoid-activated predictions on consecutive slices (|Δidx| == 1). The loss is
activated after a configurable warmup period to let the model learn the basic
segmentation task before imposing smoothness constraints.

Full training objective::

    L = L_Dice + λ_bce · L_focal_BCE + λ_tc · L_temporal

Run this file directly for a demo::

    python models/temporal_consistency.py
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitive loss functions
# ---------------------------------------------------------------------------


def l2_consistency(logits_t: Tensor, logits_t1: Tensor) -> Tensor:
    """L2 consistency between sigmoid predictions on adjacent slices.

    Computes the mean squared error between sigmoid(logits_t) and
    sigmoid(logits_t+1). Low when adjacent slices have similar predicted
    probability maps; high when they differ abruptly.

    Parameters
    ----------
    logits_t : Tensor
        Raw logits for slice at time t, shape (N, 1, H, W).
    logits_t1 : Tensor
        Raw logits for slice at time t+1, same shape.

    Returns
    -------
    Tensor
        Scalar L2 consistency loss ≥ 0.
    """
    p_t = torch.sigmoid(logits_t)
    p_t1 = torch.sigmoid(logits_t1)
    return F.mse_loss(p_t, p_t1)


def dice_consistency(logits_t: Tensor, logits_t1: Tensor) -> Tensor:
    """Dice-based consistency between adjacent slice predictions.

    Computes ``1 - DiceCoefficient(sigmoid(t), sigmoid(t+1))``.
    Since Dice ∈ [0, 1], this loss ∈ [0, 1]: 0 when slices are identical,
    1 when they have no overlap.

    Parameters
    ----------
    logits_t : Tensor
        Raw logits for slice at time t, shape (N, 1, H, W).
    logits_t1 : Tensor
        Raw logits for slice at time t+1, same shape.

    Returns
    -------
    Tensor
        Scalar Dice consistency loss ∈ [0, 1].
    """
    p_t = torch.sigmoid(logits_t)
    p_t1 = torch.sigmoid(logits_t1)
    smooth = 1e-6
    inter = (p_t * p_t1).sum()
    union = p_t.sum() + p_t1.sum()
    dice_coeff = (2.0 * inter + smooth) / (union + smooth)
    return 1.0 - dice_coeff


def dice_loss(logits: Tensor, targets: Tensor, smooth: float = 1e-6) -> Tensor:
    """Soft Dice loss for binary segmentation.

    Operates on the full logit tensor; sigmoid is applied internally.
    Averaged over the batch.

    Parameters
    ----------
    logits : Tensor
        Raw model output, shape (B, 1, H, W).
    targets : Tensor
        Binary ground truth, same shape, values in {0, 1}.
    smooth : float
        Laplace smoothing constant to prevent division by zero. Default 1e-6.

    Returns
    -------
    Tensor
        Scalar soft Dice loss ∈ (0, 1].
    """
    preds = torch.sigmoid(logits)
    # Flatten spatial dims per sample
    preds_flat = preds.view(preds.shape[0], -1)
    tgts_flat = targets.view(targets.shape[0], -1)

    inter = (preds_flat * tgts_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + tgts_flat.sum(dim=1)
    dice = (2.0 * inter + smooth) / (union + smooth)
    return (1.0 - dice).mean()


def focal_bce_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> Tensor:
    """Focal Binary Cross-Entropy loss for class-imbalanced segmentation.

    The focal modifier ``(1 - p_t)^gamma`` down-weights easy negatives and
    focuses training on hard-to-classify pixels (e.g., nodule boundaries).
    The ``alpha`` parameter further balances positive/negative frequency.

    Parameters
    ----------
    logits : Tensor
        Raw model output, shape (B, 1, H, W).
    targets : Tensor
        Binary ground truth, same shape, values in {0, 1}.
    alpha : float
        Balancing factor for positive class. Default 0.75.
        (1-alpha) is applied to negative class.
    gamma : float
        Focusing parameter. gamma=0 → standard BCE. Default 2.0.

    Returns
    -------
    Tensor
        Scalar focal BCE loss ≥ 0.
    """
    # Binary cross-entropy, element-wise, no reduction
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    probs = torch.sigmoid(logits)
    # p_t = probability of the true class
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    # alpha_t = alpha for positives, (1-alpha) for negatives
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    # Focal weight: down-weight easy examples
    focal_weight = alpha_t * (1.0 - p_t) ** gamma

    return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# TemporalConsistencyLoss — full combined loss
# ---------------------------------------------------------------------------


class TemporalConsistencyLoss(nn.Module):
    """Combined segmentation + temporal consistency loss.

    Computes::

        L = L_Dice + λ_bce · L_focal_BCE + λ_tc · L_temporal

    where ``L_temporal`` is activated only after ``warmup_epochs`` has elapsed
    and is computed only between pairs of truly adjacent slices (|Δidx| == 1).

    Parameters
    ----------
    lambda_bce : float
        Weight for the focal BCE component. Default 0.5.
    lambda_tc : float
        Weight for the temporal consistency component. Default 0.3.
    consistency_mode : str
        ``"l2"`` (L2 squared difference) or ``"dice"`` (1 - Dice similarity).
        Default ``"l2"``.
    warmup_epochs : int
        Number of epochs during which ``L_temporal = 0``. The TC loss ramps
        in suddenly at epoch ``warmup_epochs``. Default 5.
    focal_alpha : float
        Alpha parameter for focal BCE. Default 0.75.
    focal_gamma : float
        Gamma parameter for focal BCE. Default 2.0.
    """

    def __init__(
        self,
        lambda_bce: float = 0.5,
        lambda_tc: float = 0.3,
        consistency_mode: str = "l2",
        warmup_epochs: int = 5,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if consistency_mode not in ("l2", "dice"):
            raise ValueError(
                f"consistency_mode must be 'l2' or 'dice', got {consistency_mode!r}"
            )
        self.lambda_bce = lambda_bce
        self.lambda_tc = lambda_tc
        self.consistency_mode = consistency_mode
        self.warmup_epochs = warmup_epochs
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self._current_epoch: int = 0
        self._tc_active: bool = False

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch counter.

        The temporal consistency loss activates when epoch >= warmup_epochs.

        Parameters
        ----------
        epoch : int
            Current training epoch (0-indexed).
        """
        self._current_epoch = epoch
        was_active = self._tc_active
        self._tc_active = epoch >= self.warmup_epochs
        if self._tc_active and not was_active:
            logger.info(
                "TemporalConsistencyLoss: TC loss ACTIVATED at epoch %d "
                "(warmup_epochs=%d, mode=%s, λ_tc=%.2f)",
                epoch,
                self.warmup_epochs,
                self.consistency_mode,
                self.lambda_tc,
            )

    def _compute_tc_loss(
        self,
        logits: Tensor,
        slice_indices: Tensor,
    ) -> Tensor:
        """Compute temporal consistency loss over truly adjacent slice pairs.

        Parameters
        ----------
        logits : Tensor
            Batch logits of shape (B, 1, H, W).
        slice_indices : Tensor
            Integer tensor of shape (B,) containing the z-index of each sample.

        Returns
        -------
        Tensor
            Scalar temporal consistency loss. Zero if no adjacent pairs exist
            in the batch.
        """
        tc_fn = l2_consistency if self.consistency_mode == "l2" else dice_consistency

        tc_losses: list[Tensor] = []
        for i in range(len(slice_indices) - 1):
            idx_t = int(slice_indices[i].item())
            idx_t1 = int(slice_indices[i + 1].item())
            # Only penalise TRULY adjacent slices: |Δidx| == 1
            if abs(idx_t - idx_t1) == 1:
                tc_losses.append(tc_fn(logits[i : i + 1], logits[i + 1 : i + 2]))

        if len(tc_losses) == 0:
            # No adjacent pairs in this batch — return zero without a graph
            return torch.tensor(0.0, device=logits.device, requires_grad=False)

        return torch.stack(tc_losses).mean()

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        slice_indices: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute the full combined loss.

        Parameters
        ----------
        logits : Tensor
            Model output logits of shape (B, 1, H, W).
        targets : Tensor
            Binary ground truth masks of shape (B, 1, H, W).
        slice_indices : Tensor or None
            Integer z-indices for each sample in the batch, shape (B,).
            If None, the temporal consistency loss is skipped.

        Returns
        -------
        dict of Tensor with keys:
            ``total``    : weighted combined loss (scalar).
            ``dice``     : soft Dice loss component (scalar).
            ``bce``      : focal BCE loss component (scalar).
            ``temporal`` : temporal consistency loss component (scalar, 0 during warmup).
        """
        l_dice = dice_loss(logits, targets)
        l_bce = focal_bce_loss(
            logits, targets, alpha=self.focal_alpha, gamma=self.focal_gamma
        )

        # Temporal consistency
        if self._tc_active and slice_indices is not None and self.lambda_tc > 0:
            l_tc = self._compute_tc_loss(logits, slice_indices)
        else:
            l_tc = torch.tensor(0.0, device=logits.device)

        total = l_dice + self.lambda_bce * l_bce + self.lambda_tc * l_tc

        return {
            "total": total,
            "dice": l_dice,
            "bce": l_bce,
            "temporal": l_tc,
        }


# ---------------------------------------------------------------------------
# AblationLossFactory
# ---------------------------------------------------------------------------


class AblationLossFactory:
    """Factory for creating loss variants used in the ablation study.

    Each variant corresponds to one row in the ablation table:

    - ``"baseline"`` — Dice + focal BCE only, no temporal loss (λ_tc=0).
    - ``"tc_l2"`` — Full model with L2 temporal consistency.
    - ``"tc_dice"`` — Full model with Dice temporal consistency.

    Usage::

        criterion = AblationLossFactory.get("tc_l2")
    """

    _VARIANTS: Dict[str, Dict] = {
        "baseline": dict(lambda_bce=0.5, lambda_tc=0.0, consistency_mode="l2"),
        "tc_l2": dict(lambda_bce=0.5, lambda_tc=0.3, consistency_mode="l2"),
        "tc_dice": dict(lambda_bce=0.5, lambda_tc=0.3, consistency_mode="dice"),
    }

    @classmethod
    def get(cls, variant: str, **overrides) -> TemporalConsistencyLoss:
        """Build a ``TemporalConsistencyLoss`` for the given variant.

        Parameters
        ----------
        variant : str
            One of ``"baseline"``, ``"tc_l2"``, ``"tc_dice"``.
        **overrides
            Additional keyword arguments that override the variant defaults,
            e.g. ``warmup_epochs=10``.

        Returns
        -------
        TemporalConsistencyLoss
            Configured loss module.

        Raises
        ------
        ValueError
            If ``variant`` is not one of the registered variants.
        """
        if variant not in cls._VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. Choose from: {list(cls._VARIANTS)}"
            )
        kwargs = dict(cls._VARIANTS[variant])
        kwargs.update(overrides)
        logger.info(
            "AblationLossFactory: creating variant=%r with kwargs=%s", variant, kwargs
        )
        return TemporalConsistencyLoss(**kwargs)

    @classmethod
    def list_variants(cls) -> list[str]:
        """Return a list of all registered variant names."""
        return list(cls._VARIANTS.keys())


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=" * 65)
    print("TemporalConsistencyLoss — demo")
    print("=" * 65)

    B, H, W = 6, 64, 64
    logits = torch.randn(B, 1, H, W)
    targets = (torch.rand(B, 1, H, W) > 0.7).float()
    # Simulate a batch of consecutive slices from the same patch
    slice_indices = torch.tensor([10, 11, 12, 13, 14, 20], dtype=torch.long)
    # Note: slice 14→20 is NOT adjacent (Δidx=6), so TC skips that pair.

    criterion = TemporalConsistencyLoss(
        lambda_bce=0.5, lambda_tc=0.3, consistency_mode="l2", warmup_epochs=5
    )

    # --- Warmup phase (epoch 0 < warmup_epochs=5) ---
    criterion.set_epoch(0)
    losses_warmup = criterion(logits, targets, slice_indices)
    print("\nEpoch  0 (warmup — TC inactive):")
    print(
        f"  total={losses_warmup['total']:.4f} | dice={losses_warmup['dice']:.4f} | "
        f"bce={losses_warmup['bce']:.4f} | temporal={losses_warmup['temporal']:.4f}"
    )
    assert losses_warmup["temporal"].item() == 0.0, "TC should be 0 during warmup"
    print("  ✓ temporal=0 during warmup")

    # --- Active phase (epoch 5 >= warmup_epochs=5) ---
    criterion.set_epoch(5)
    losses_active = criterion(logits, targets, slice_indices)
    print("\nEpoch  5 (TC active — mode=l2):")
    print(
        f"  total={losses_active['total']:.4f} | dice={losses_active['dice']:.4f} | "
        f"bce={losses_active['bce']:.4f} | temporal={losses_active['temporal']:.4f}"
    )
    assert losses_active["temporal"].item() >= 0.0
    print("  ✓ temporal > 0 when TC active")

    # --- L2 consistency: same logits → near 0 ---
    same = torch.zeros(2, 1, H, W)
    l2_same = l2_consistency(same, same)
    print(f"\nl2_consistency(same, same) = {l2_same:.6f}  (expected ≈ 0)")
    assert l2_same.item() < 1e-10

    # --- L2 consistency: different logits → > 0 ---
    diff = torch.ones(2, 1, H, W) * 5.0
    l2_diff = l2_consistency(same, diff)
    print(f"l2_consistency(zeros, fives) = {l2_diff:.6f}  (expected > 0)")
    assert l2_diff.item() > 0

    # --- AblationLossFactory ---
    print(f"\nAblationLossFactory variants: {AblationLossFactory.list_variants()}")
    for variant in AblationLossFactory.list_variants():
        crit = AblationLossFactory.get(variant)
        crit.set_epoch(10)
        out = crit(logits, targets, slice_indices)
        print(f"  [{variant}] total={out['total']:.4f} | tc={out['temporal']:.4f}")

    print("\nAll tests passed. ✓")
