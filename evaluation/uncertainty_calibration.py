"""
Uncertainty calibration analysis for MC Dropout predictions.

A well-calibrated model outputs high uncertainty where it is actually wrong and
low uncertainty where it is correct. This module provides:

- ``expected_calibration_error`` (ECE) — reliability of confidence scores.
- ``brier_score``                  — proper scoring rule for probabilistic predictions.
- ``reliability_diagram``          — calibration curve plotting utility.
- ``entropy_auc``                  — AUROC of uncertainty as an error detector.
- ``CalibrationAnalyzer``          — stateful accumulator for the full test set.

Reference: Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017).

Run this file for a demo::

    python evaluation/uncertainty_calibration.py
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Optional matplotlib for plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.warning("matplotlib not installed — plotting functions disabled")


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error (ECE) over binned confidence.

    Divides predictions into ``n_bins`` equal-width confidence bins and
    measures the weighted mean absolute difference between mean confidence
    and mean accuracy within each bin.

    ECE = Σ_b  |B_b| / N  *  |acc(B_b) − conf(B_b)|

    Parameters
    ----------
    confidences : np.ndarray
        Predicted confidence (probability estimate), shape (N,). Values ∈ [0, 1].
    correctness : np.ndarray
        Binary array: 1 if the prediction was correct, 0 otherwise. Shape (N,).
    n_bins : int
        Number of calibration bins. Default 15.

    Returns
    -------
    ece : float
        Expected calibration error ∈ [0, 1]. Lower is better.
    bin_accs : np.ndarray
        Mean accuracy per bin, shape (n_bins,).
    bin_confs : np.ndarray
        Mean confidence per bin, shape (n_bins,).
    bin_freqs : np.ndarray
        Fraction of samples per bin, shape (n_bins,).
    """
    N = len(confidences)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            bin_accs[b] = correctness[mask].mean()
            bin_confs[b] = confidences[mask].mean()
            bin_counts[b] = mask.sum()

    bin_freqs = bin_counts / max(N, 1)
    ece = float((bin_freqs * np.abs(bin_accs - bin_confs)).sum())
    return ece, bin_accs, bin_confs, bin_freqs


def brier_score(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute pixel-level Brier score.

    Brier score = (1/N) Σ (p_i − y_i)²

    A proper scoring rule: 0.0 is perfect, 1.0 is worst.

    Parameters
    ----------
    probabilities : np.ndarray
        Predicted probability for class 1, shape (N,). Values ∈ [0, 1].
    targets : np.ndarray
        Binary ground truth, shape (N,). Values in {0, 1}.

    Returns
    -------
    float
        Brier score ∈ [0, 1].
    """
    return float(np.mean((probabilities - targets) ** 2))


def entropy_auc(
    uncertainty: np.ndarray,
    errors: np.ndarray,
) -> float:
    """Compute AUROC of uncertainty as a binary error detector.

    Measures how well the uncertainty map identifies wrongly-predicted pixels.
    High AUROC means high uncertainty correlates with prediction errors — a
    well-calibrated model.

    Parameters
    ----------
    uncertainty : np.ndarray
        Per-pixel uncertainty (MC variance or entropy), shape (N,). Values ≥ 0.
    errors : np.ndarray
        Binary array: 1 if pixel was mis-classified, 0 otherwise. Shape (N,).

    Returns
    -------
    float
        AUROC ∈ [0, 1]. 0.5 = random; 1.0 = perfect uncertainty detector.
    """
    try:
        from sklearn.metrics import roc_auc_score
        if errors.sum() == 0 or errors.sum() == len(errors):
            return float("nan")
        return float(roc_auc_score(errors, uncertainty))
    except ImportError:
        # Manual AUROC via trapezoidal rule
        thresholds = np.linspace(0, uncertainty.max(), 200)
        tpr_list, fpr_list = [0.0], [0.0]
        pos = errors.sum()
        neg = len(errors) - pos
        if pos == 0 or neg == 0:
            return float("nan")
        for t in sorted(thresholds, reverse=True):
            pred_pos = uncertainty >= t
            tpr = (pred_pos & errors.astype(bool)).sum() / pos
            fpr = (pred_pos & ~errors.astype(bool)).sum() / neg
            tpr_list.append(float(tpr))
            fpr_list.append(float(fpr))
        tpr_list.append(1.0)
        fpr_list.append(1.0)
        return float(np.trapz(tpr_list, fpr_list))


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------


def reliability_diagram(
    bin_accs: np.ndarray,
    bin_confs: np.ndarray,
    bin_freqs: np.ndarray,
    ece: float,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
) -> Optional[object]:
    """Plot a calibration reliability diagram.

    Shows the gap between ideal diagonal (perfect calibration) and the
    model's actual accuracy-confidence relationship per bin.

    Parameters
    ----------
    bin_accs : np.ndarray
        Mean accuracy per bin, shape (n_bins,).
    bin_confs : np.ndarray
        Mean confidence per bin, shape (n_bins,).
    bin_freqs : np.ndarray
        Fraction of samples per bin (for bar heights), shape (n_bins,).
    ece : float
        ECE value shown in the legend.
    title : str
        Plot title. Default "Reliability Diagram".
    save_path : str or None
        If provided, save the figure to this path. Default None.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object, or None if matplotlib is not available.
    """
    if not _MPL_AVAILABLE:
        logger.warning("reliability_diagram: matplotlib not available")
        return None

    n_bins = len(bin_accs)
    bin_centers = np.linspace(1 / (2 * n_bins), 1 - 1 / (2 * n_bins), n_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Reliability diagram
    ax1.bar(bin_centers, bin_accs, width=1.0 / n_bins, alpha=0.7, color="#4C72B0", label="Model accuracy")
    ax1.bar(bin_centers, np.abs(bin_accs - bin_confs), width=1.0 / n_bins,
            bottom=np.minimum(bin_accs, bin_confs), alpha=0.6, color="#DD8452", label="Gap")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax1.set_xlabel("Mean confidence")
    ax1.set_ylabel("Mean accuracy")
    ax1.set_title(f"Reliability Diagram  (ECE = {ece:.4f})")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Frequency histogram
    ax2.bar(bin_centers, bin_freqs, width=1.0 / n_bins, alpha=0.7, color="#55A868")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Fraction of samples")
    ax2.set_title("Confidence Distribution")
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Reliability diagram saved → %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Stateful accumulator
# ---------------------------------------------------------------------------


class CalibrationAnalyzer:
    """Stateful calibration analyzer over multiple MC Dropout batches.

    Accumulates pixel-level (probability, target, uncertainty, error) tuples
    across the entire validation / test set, then computes calibration metrics
    and optionally saves a reliability diagram.

    Parameters
    ----------
    n_bins : int
        Number of calibration bins for ECE. Default 15.
    subsample_rate : float
        Fraction of pixels to keep per batch (to bound memory). Default 0.05.
        At 0.05, a 512×512 image contributes ~1300 pixel samples.
    """

    def __init__(self, n_bins: int = 15, subsample_rate: float = 0.05) -> None:
        self.n_bins = n_bins
        self.subsample_rate = subsample_rate
        self._probs: List[np.ndarray] = []
        self._targets: List[np.ndarray] = []
        self._uncertainties: List[np.ndarray] = []
        self._errors: List[np.ndarray] = []

    def update(
        self,
        mean_pred: Tensor,
        uncertainty: Tensor,
        target: Tensor,
        threshold: float = 0.5,
    ) -> None:
        """Accumulate one batch of MC Dropout predictions.

        Parameters
        ----------
        mean_pred : Tensor
            Mean MC prediction, shape (B, 1, H, W). Values ∈ [0, 1].
        uncertainty : Tensor
            MC variance map, shape (B, 1, H, W). Values ≥ 0.
        target : Tensor
            Binary ground truth, shape (B, 1, H, W). Values in {0, 1}.
        threshold : float
            Binarisation threshold. Default 0.5.
        """
        num_pixels = mean_pred.numel()
        n_keep = max(1, int(num_pixels * self.subsample_rate))
        idx = torch.randperm(num_pixels)[:n_keep]

        p_flat = mean_pred.flatten()[idx].cpu().float().numpy()
        t_flat = target.float().flatten()[idx].cpu().numpy()
        u_flat = uncertainty.flatten()[idx].cpu().float().numpy()
        binary = (mean_pred >= threshold).float()
        err_flat = (binary.flatten()[idx] != target.float().flatten()[idx]).cpu().numpy().astype(float)

        self._probs.append(p_flat)
        self._targets.append(t_flat)
        self._uncertainties.append(u_flat)
        self._errors.append(err_flat)

    def compute(
        self,
        save_diagram: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute all calibration metrics over accumulated data.

        Parameters
        ----------
        save_diagram : str or None
            If provided, save reliability diagram PNG to this path.

        Returns
        -------
        dict with keys:
            ``ece``          : Expected Calibration Error.
            ``brier``        : Brier score.
            ``uncertainty_auc`` : AUROC of uncertainty as error detector.
            ``n_pixels``     : Total accumulated pixel count.
        """
        if not self._probs:
            return {"ece": float("nan"), "brier": float("nan"), "uncertainty_auc": float("nan"), "n_pixels": 0.0}

        probs = np.concatenate(self._probs)
        targets = np.concatenate(self._targets)
        uncertainties = np.concatenate(self._uncertainties)
        errors = np.concatenate(self._errors)

        ece, bin_accs, bin_confs, bin_freqs = expected_calibration_error(probs, 1 - errors, self.n_bins)
        bs = brier_score(probs, targets)
        unc_auc = entropy_auc(uncertainties, errors)

        if save_diagram:
            reliability_diagram(
                bin_accs, bin_confs, bin_freqs, ece,
                title="SAM2 Lung Nodule Seg — Calibration",
                save_path=save_diagram,
            )

        results = {
            "ece": ece,
            "brier": bs,
            "uncertainty_auc": unc_auc,
            "n_pixels": float(len(probs)),
        }
        logger.info(
            "Calibration: ECE=%.4f | Brier=%.4f | UncAUC=%.4f | pixels=%d",
            ece, bs, unc_auc, len(probs),
        )
        return results

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._probs.clear()
        self._targets.clear()
        self._uncertainties.clear()
        self._errors.clear()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("CalibrationAnalyzer — demo with synthetic predictions")
    print("=" * 60)

    rng = np.random.default_rng(42)
    N = 10_000

    # Simulate confident correct predictions
    probs_correct = rng.beta(8, 2, N // 2)   # high confidence
    targets_correct = np.ones(N // 2)
    # Simulate uncertain incorrect predictions
    probs_wrong = rng.beta(2, 8, N // 2)     # low confidence
    targets_wrong = np.ones(N // 2)          # actual label = 1, but model predicts ~0

    probs_all = np.concatenate([probs_correct, probs_wrong])
    targets_all = np.concatenate([targets_correct, targets_wrong])

    ece, bin_accs, bin_confs, bin_freqs = expected_calibration_error(probs_all, (probs_all >= 0.5).astype(float))
    print(f"\nECE: {ece:.4f}")

    bs = brier_score(probs_all, targets_all)
    print(f"Brier: {bs:.4f}")

    # Uncertainty AUC
    uncertainty_sim = rng.exponential(0.02, N)
    errors_sim = rng.binomial(1, 0.15, N).astype(float)  # 15% error rate
    # Make high-error pixels have higher uncertainty on average
    uncertainty_sim[errors_sim == 1] += rng.exponential(0.04, int(errors_sim.sum()))
    auc = entropy_auc(uncertainty_sim, errors_sim)
    print(f"UncertaintyAUC: {auc:.4f}")

    reliability_diagram(
        bin_accs, bin_confs, bin_freqs, ece,
        title="Demo Reliability Diagram",
        save_path="calibration_demo.png",
    )

    # CalibrationAnalyzer
    analyzer = CalibrationAnalyzer(n_bins=10, subsample_rate=0.5)
    for _ in range(5):
        B, H, W = 4, 64, 64
        mean_pred = torch.rand(B, 1, H, W)
        uncertainty = torch.rand(B, 1, H, W) * 0.05
        target = (torch.rand(B, 1, H, W) > 0.5).float()
        analyzer.update(mean_pred, uncertainty, target)

    results = analyzer.compute(save_diagram="calibration_test.png")
    print(f"\nCalibrationAnalyzer results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    print("\nAll checks passed. ✓")
