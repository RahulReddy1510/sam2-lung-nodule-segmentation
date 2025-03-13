"""
SAM2 Lung Nodule Segmentation — evaluation package.

Quick start::

    from evaluation import run_evaluation
    from evaluation import DiceMetric, compute_dice
    from evaluation import CalibrationAnalyzer, reliability_diagram
    from evaluation import RadiologistAgreement, cohens_kappa
"""

from evaluation.dice_metric import (
    DiceMetric,
    compute_all_metrics,
    compute_dice,
    compute_iou,
    compute_precision_recall,
)
from evaluation.evaluate import run_evaluation
from evaluation.radiologist_agreement import (
    RadiologistAgreement,
    bland_altman,
    cohens_kappa,
    fleiss_kappa,
    percent_agreement,
)
from evaluation.uncertainty_calibration import (
    CalibrationAnalyzer,
    brier_score,
    entropy_auc,
    expected_calibration_error,
    reliability_diagram,
)

__all__ = [
    # Dice / segmentation metrics
    "DiceMetric",
    "compute_dice",
    "compute_iou",
    "compute_precision_recall",
    "compute_all_metrics",
    # Uncertainty calibration
    "CalibrationAnalyzer",
    "reliability_diagram",
    "expected_calibration_error",
    "brier_score",
    "entropy_auc",
    # Radiologist agreement
    "RadiologistAgreement",
    "cohens_kappa",
    "fleiss_kappa",
    "percent_agreement",
    "bland_altman",
    # Top-level evaluation runner
    "run_evaluation",
]
