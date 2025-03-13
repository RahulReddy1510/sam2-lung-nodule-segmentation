"""
SAM2 Lung Nodule Segmentation — models package.

Exposes the primary model classes and factory functions.

Quick start::

    from models import build_model, SAM2LungSegmentor
    from models import mc_predict, TemporalConsistencyLoss

    model = build_model()
    from models.mc_dropout import mc_predict
    from models.temporal_consistency import TemporalConsistencyLoss
"""

from models.mc_dropout import (
    compute_uncertainty_stats,
    enable_dropout_modules,
    entropy_from_samples,
    mc_dropout_mode,
    mc_predict,
    mc_predict_volume,
)
from models.registry import ModelRegistry, get_model
from models.sam2_finetune import (
    DropoutMultiheadAttention,
    FallbackEncoder,
    LightweightMaskDecoder,
    SAM2LungSegmentor,
    SinusoidalPosEmbed,
    build_model,
)
from models.temporal_consistency import (
    AblationLossFactory,
    TemporalConsistencyLoss,
    dice_consistency,
    dice_loss,
    focal_bce_loss,
    l2_consistency,
)

__all__ = [
    # Core model
    "SAM2LungSegmentor",
    "FallbackEncoder",
    "LightweightMaskDecoder",
    "SinusoidalPosEmbed",
    "DropoutMultiheadAttention",
    "build_model",
    # MC Dropout
    "enable_dropout_modules",
    "mc_dropout_mode",
    "mc_predict",
    "mc_predict_volume",
    "entropy_from_samples",
    "compute_uncertainty_stats",
    # Loss functions
    "TemporalConsistencyLoss",
    "AblationLossFactory",
    "dice_loss",
    "focal_bce_loss",
    "l2_consistency",
    "dice_consistency",
    # Registry
    "ModelRegistry",
    "get_model",
]
