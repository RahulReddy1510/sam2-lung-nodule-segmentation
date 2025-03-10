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
    compute_dice,
    compute_iou,
    compute_precision_recall,
    compute_all_metrics,
)
from evaluation.uncertainty_calibration import (
    CalibrationAnalyzer,
    reliability_diagram,
    expected_calibration_error,
    brier_score,
    entropy_auc,
)
from evaluation.radiologist_agreement import (
    RadiologistAgreement,
    cohens_kappa,
    fleiss_kappa,
    percent_agreement,
    bland_altman,
)
from evaluation.evaluate import run_evaluation

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
