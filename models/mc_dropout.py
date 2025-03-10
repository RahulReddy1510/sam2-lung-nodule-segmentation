"""
Monte Carlo Dropout inference utilities.

MC Dropout (Gal & Ghahramani, 2016) keeps dropout layers stochastic at
inference time by calling ``enable_dropout_modules()`` after ``model.eval()``.
Running T stochastic forward passes produces a distribution of predictions
whose variance is a spatially-resolved uncertainty map.

Key functions::

    with mc_dropout_mode(model):
        mean_pred, uncertainty = mc_predict(model, x, n_samples=25)

Run this file directly for a demo::

    python models/mc_dropout.py
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, Generator, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode-switching utilities
# ---------------------------------------------------------------------------


def enable_dropout_modules(model: nn.Module) -> None:
    """Set all Dropout layers to training mode while leaving BatchNorm in eval.

    This achieves the Monte Carlo Dropout inference regime:
    - Dropout layers: stochastic (training mode) → samples differ each pass
    - BatchNorm layers: deterministic (eval mode) → use running statistics

    Calling this after ``model.eval()`` is the standard MC Dropout setup.

    Parameters
    ----------
    model : nn.Module
        The model to modify in-place.

    Notes
    -----
    The function recurses through the entire module tree. It targets any
    module whose class name contains "Dropout" (``nn.Dropout``,
    ``nn.Dropout2d``, ``nn.Dropout3d``, ``nn.AlphaDropout``, and the custom
    ``DropoutMultiheadAttention.attn_dropout``). This broad pattern ensures
    that custom dropout wrappers like those in ``LightweightMaskDecoder`` are
    also activated.
    """
    n_dropout = 0
    n_bn = 0
    for module in model.modules():
        class_name = module.__class__.__name__
        if "Dropout" in class_name:
            module.train()
            n_dropout += 1
        elif "BatchNorm" in class_name:
            module.eval()
            n_bn += 1
    logger.debug(
        "enable_dropout_modules: %d Dropout layers → train, %d BN layers → eval",
        n_dropout,
        n_bn,
    )


@contextlib.contextmanager
def mc_dropout_mode(model: nn.Module) -> Generator[nn.Module, None, None]:
    """Context manager that enables MC Dropout inference.

    Sets the model to eval mode, then activates all Dropout layers,
    and restores full eval mode on exit.

    Parameters
    ----------
    model : nn.Module
        The model to run in MC Dropout mode.

    Yields
    ------
    model : nn.Module
        The same model, now in MC Dropout mode.

    Examples
    --------
    >>> with mc_dropout_mode(model):
    ...     mean_pred, uncertainty = mc_predict(model, x, n_samples=25)
    """
    was_training = model.training
    try:
        model.eval()
        enable_dropout_modules(model)
        yield model
    finally:
        # Restore original training state
        model.train(was_training)
        # Ensure all BN layers are back in eval if model was in eval
        if not was_training:
            model.eval()


# ---------------------------------------------------------------------------
# MC Dropout prediction — 2D
# ---------------------------------------------------------------------------


@torch.no_grad()
def mc_predict(
    model: nn.Module,
    x: Tensor,
    n_samples: int = 25,
    mc_batch_size: int = 4,
    sigmoid: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Run Monte Carlo Dropout inference on a batch of 2D slices.

    Performs ``n_samples`` stochastic forward passes through the model
    (with dropout active) and returns the pixel-wise mean and variance
    of the predictions.

    Parameters
    ----------
    model : nn.Module
        The segmentation model, configured with MC-compatible dropout.
    x : Tensor
        Input batch of shape (B, C, H, W).
    n_samples : int
        Number of stochastic forward passes. Default 25.
    mc_batch_size : int
        Number of MC samples processed per forward pass. Reduces peak VRAM
        usage at the cost of more passes. Default 4.
    sigmoid : bool
        If True, apply sigmoid to logits before averaging. Mean of sigmoid
        predictions is the Bayesian predictive mean; variance is the epistemic
        uncertainty. Default True.

    Returns
    -------
    mean_pred : Tensor
        Pixel-wise mean prediction of shape (B, 1, H, W), in [0, 1] if
        ``sigmoid=True``, otherwise raw logit scale.
    uncertainty : Tensor
        Pixel-wise variance across MC passes, shape (B, 1, H, W), ≥ 0.

    Notes
    -----
    To save GPU memory, samples are processed in chunks of ``mc_batch_size``.
    The function is decorated with ``@torch.no_grad()`` — no gradients are
    computed regardless of the model's grad state.
    """
    device = next(model.parameters()).device
    x = x.to(device)
    B = x.shape[0]

    all_preds: list[Tensor] = []

    with mc_dropout_mode(model):
        remaining = n_samples
        while remaining > 0:
            chunk = min(mc_batch_size, remaining)
            # Repeat x chunk times along a new axis
            x_chunk = x.unsqueeze(0).expand(chunk, -1, -1, -1, -1)
            # (chunk, B, C, H, W) → (chunk*B, C, H, W)
            x_flat = x_chunk.reshape(chunk * B, *x.shape[1:])

            logits = model(x_flat)  # (chunk*B, 1, H, W)
            preds = torch.sigmoid(logits) if sigmoid else logits
            # (chunk*B, 1, H, W) → (chunk, B, 1, H, W)
            preds = preds.reshape(chunk, B, *preds.shape[1:])
            all_preds.append(preds)
            remaining -= chunk

    # Stack: (n_samples, B, 1, H, W)
    stacked = torch.cat(all_preds, dim=0)

    mean_pred = stacked.mean(dim=0)          # (B, 1, H, W)
    uncertainty = stacked.var(dim=0)         # (B, 1, H, W), unbiased=True (default)

    logger.debug(
        "mc_predict: n_samples=%d | mean range [%.4f, %.4f] | unc max=%.4f",
        n_samples,
        mean_pred.min().item(),
        mean_pred.max().item(),
        uncertainty.max().item(),
    )
    return mean_pred, uncertainty


# ---------------------------------------------------------------------------
# MC Dropout prediction — 3D volume (slice-by-slice)
# ---------------------------------------------------------------------------


@torch.no_grad()
def mc_predict_volume(
    model: nn.Module,
    volume: Tensor,
    n_samples: int = 25,
    mc_batch_size: int = 4,
    threshold: float = 0.5,
    slice_batch_size: int = 8,
) -> Dict[str, Tensor]:
    """Run MC Dropout inference on a full 3D CT volume slice-by-slice.

    The model processes each 2D axial slice independently. Results are
    assembled back into 3D volumes.

    Parameters
    ----------
    model : nn.Module
        The segmentation model.
    volume : Tensor
        Full CT volume of shape (1, 1, Z, H, W) or (1, Z, H, W).
        Values should be in [0, 1] (HU-windowed and normalized).
    n_samples : int
        MC Dropout passes per slice. Default 25.
    mc_batch_size : int
        MC samples per forward pass. Default 4.
    threshold : float
        Binary threshold applied to the mean prediction. Default 0.5.
    slice_batch_size : int
        Number of slices processed simultaneously (batched inference). Default 8.

    Returns
    -------
    dict with keys:
        ``mean_seg`` : Tensor (1, 1, Z, H, W) — float mean predictions ∈ [0, 1].
        ``uncertainty`` : Tensor (1, 1, Z, H, W) — pixel-wise variance ∈ [0, ∞).
        ``binary_mask`` : Tensor (1, 1, Z, H, W) — thresholded binary mask (uint8).

    Notes
    -----
    Slices are processed in batches of ``slice_batch_size`` to avoid OOM on
    long CT volumes. GPU cache is cleared between slice batches to prevent
    memory accumulation.
    """
    # Normalize input shape to (1, Z, H, W)
    if volume.ndim == 5 and volume.shape[0] == 1:
        volume = volume.squeeze(0)  # (1, Z, H, W)
    assert volume.ndim == 4, f"Expected (1, Z, H, W) or (1, 1, Z, H, W), got {volume.shape}"
    _, Z, H, W = volume.shape

    mean_slices: list[Tensor] = []
    unc_slices: list[Tensor] = []

    # Process slices in batches to bound peak memory
    for z_start in range(0, Z, slice_batch_size):
        z_end = min(z_start + slice_batch_size, Z)
        slice_batch = volume[:, z_start:z_end, :, :]  # (1, batch_z, H, W)
        # → (batch_z, 1, H, W) — each slice as a separate sample
        slices_2d = slice_batch.permute(1, 0, 2, 3)  # (batch_z, 1, H, W)

        mean_batch, unc_batch = mc_predict(
            model, slices_2d, n_samples=n_samples, mc_batch_size=mc_batch_size
        )
        mean_slices.append(mean_batch)  # (batch_z, 1, H, W)
        unc_slices.append(unc_batch)

        # Free GPU cache between slice batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stack along Z: (Z, 1, H, W) → (1, 1, Z, H, W)
    mean_vol = torch.cat(mean_slices, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)
    unc_vol = torch.cat(unc_slices, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)
    binary_vol = (mean_vol >= threshold).to(torch.uint8)

    logger.info(
        "mc_predict_volume: Z=%d | mean∈[%.3f, %.3f] | unc max=%.4f | "
        "mask voxels=%d",
        Z,
        mean_vol.min().item(),
        mean_vol.max().item(),
        unc_vol.max().item(),
        int(binary_vol.sum().item()),
    )

    return {
        "mean_seg": mean_vol,
        "uncertainty": unc_vol,
        "binary_mask": binary_vol,
    }


# ---------------------------------------------------------------------------
# Entropy utility
# ---------------------------------------------------------------------------


def entropy_from_samples(samples: Tensor) -> Tensor:
    """Compute predictive entropy from MC Dropout sample predictions.

    Predictive entropy: H[p̄] = -p̄ log p̄ - (1 - p̄) log(1 - p̄)

    where p̄ = E_T[p(y=1|x, w_t)] is the mean prediction across T samples.

    This is a global uncertainty measure combining both epistemic and aleatoric
    uncertainty, unlike variance which is purely epistemic.

    Parameters
    ----------
    samples : Tensor
        Stacked MC predictions of shape (T, B, 1, H, W) or (T, 1, H, W),
        with values in [0, 1].

    Returns
    -------
    Tensor
        Predictive entropy map of the same shape as the mean (B, 1, H, W).
    """
    eps = 1e-7
    p_bar = samples.mean(dim=0)  # Mean across T samples
    p_bar = p_bar.clamp(eps, 1.0 - eps)
    entropy = -(p_bar * p_bar.log() + (1.0 - p_bar) * (1.0 - p_bar).log())
    return entropy


# ---------------------------------------------------------------------------
# Uncertainty statistics
# ---------------------------------------------------------------------------


def compute_uncertainty_stats(
    uncertainty_map: Tensor,
    binary_pred: Tensor,
    binary_gt: Tensor | None = None,
) -> Dict[str, float]:
    """Compute summary statistics for an uncertainty map.

    Parameters
    ----------
    uncertainty_map : Tensor
        Pixel-wise variance map of shape (B, 1, H, W) or (1, H, W) or (H, W).
    binary_pred : Tensor
        Binary segmentation prediction, same shape as uncertainty_map.
        Values in {0, 1}.
    binary_gt : Tensor or None
        Ground truth binary mask, same shape. If provided, boundary uncertainty
        is also computed (uncertainty in the XOR region). Default None.

    Returns
    -------
    dict with keys:
        ``unc_mean``           : float — mean uncertainty across all pixels.
        ``unc_max``            : float — maximum uncertainty.
        ``unc_std``            : float — std of uncertainty.
        ``unc_in_pred_mean``   : float — mean uncertainty inside predicted mask.
        ``unc_out_pred_mean``  : float — mean uncertainty outside predicted mask.
        ``unc_at_boundary``    : float — mean uncertainty in GT boundary region
                                         (only if binary_gt is provided).
    """
    u = uncertainty_map.float().flatten()
    p = binary_pred.float().flatten()

    stats: Dict[str, float] = {
        "unc_mean": float(u.mean().item()),
        "unc_max": float(u.max().item()),
        "unc_std": float(u.std().item()),
    }

    pred_mask = p > 0.5
    if pred_mask.any():
        stats["unc_in_pred_mean"] = float(u[pred_mask].mean().item())
    else:
        stats["unc_in_pred_mean"] = 0.0

    not_pred = ~pred_mask
    if not_pred.any():
        stats["unc_out_pred_mean"] = float(u[not_pred].mean().item())
    else:
        stats["unc_out_pred_mean"] = 0.0

    if binary_gt is not None:
        gt = binary_gt.float().flatten()
        # Boundary = XOR between prediction and GT
        boundary = (p.round() != gt.round())
        if boundary.any():
            stats["unc_at_boundary"] = float(u[boundary].mean().item())
        else:
            stats["unc_at_boundary"] = 0.0

    return stats


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch
    from models.sam2_finetune import build_model

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=" * 65)
    print("MC Dropout — demo with synthetic data")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(embed_dim=256, num_heads=8, encoder_frozen=False).to(device)

    # Test 1: mc_predict on a 2D batch
    B, H, W = 2, 96, 96
    x = torch.randn(B, 1, H, W, device=device)

    print(f"\n[Test 1] mc_predict | input: {tuple(x.shape)} | n_samples=25")
    mean_pred, uncertainty = mc_predict(model, x, n_samples=25, mc_batch_size=5)
    print(f"  mean_pred shape:  {tuple(mean_pred.shape)}")
    print(f"  uncertainty shape:{tuple(uncertainty.shape)}")
    print(f"  mean_pred range:  [{mean_pred.min():.4f}, {mean_pred.max():.4f}]")
    print(f"  uncertainty max:  {uncertainty.max():.6f}")
    assert mean_pred.shape == (B, 1, H, W)
    assert (uncertainty >= 0).all(), "Variance must be non-negative"
    print("  ✓ shapes and value ranges correct")

    # Test 2: entropy_from_samples
    T = 10
    samples = torch.rand(T, B, 1, H, W)
    entropy = entropy_from_samples(samples)
    print(f"\n[Test 2] entropy_from_samples | samples: {tuple(samples.shape)}")
    print(f"  entropy shape: {tuple(entropy.shape)}")
    print(f"  entropy range: [{entropy.min():.4f}, {entropy.max():.4f}]")
    assert (entropy >= 0).all()
    print("  ✓ entropy non-negative")

    # Test 3: compute_uncertainty_stats
    binary_pred = (mean_pred > 0.5).float()
    binary_gt = torch.zeros_like(binary_pred)
    binary_gt[:, :, H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1.0

    stats = compute_uncertainty_stats(uncertainty, binary_pred, binary_gt)
    print(f"\n[Test 3] compute_uncertainty_stats")
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}")
    assert "unc_in_pred_mean" in stats and "unc_at_boundary" in stats
    print("  ✓ all expected keys present")

    # Test 4: mc_dropout_mode context manager
    print(f"\n[Test 4] mc_dropout_mode context manager")
    model.eval()
    # Check that dropout layers are in eval before
    n_train_before = sum(1 for m in model.modules() if "Dropout" in m.__class__.__name__ and m.training)
    with mc_dropout_mode(model):
        n_train_during = sum(1 for m in model.modules() if "Dropout" in m.__class__.__name__ and m.training)
    n_train_after = sum(1 for m in model.modules() if "Dropout" in m.__class__.__name__ and m.training)
    print(f"  Dropout modules in train mode: before={n_train_before}, during={n_train_during}, after={n_train_after}")
    assert n_train_during > 0, "Should have dropout active during mc_dropout_mode"
    print("  ✓ context manager correctly activates/restores dropout")

    print("\nAll tests passed. ✓")
