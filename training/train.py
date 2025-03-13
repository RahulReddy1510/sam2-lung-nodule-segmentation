"""
End-to-end training script for SAM2 Lung Nodule Segmentation.

Implements the full 50-epoch fine-tuning pipeline::

    Epochs  1-5  : encoder frozen, decoder + prompt token only          [warmup + TC warmup]
    Epochs  6-50 : encoder unfrozen, full end-to-end fine-tuning         [encoder_lr × 0.1]

Training objective::

    L = L_Dice + 0.5 · L_focal_BCE + 0.3 · L_temporal

Usage::

    # Synthetic data (no LUNA16 download needed)
    python training/train.py data.data_dir=SYNTHETIC

    # Real data
    python training/train.py data.data_dir=data/processed/

    # Resume from checkpoint
    python training/train.py training.resume_from=runs/my_run/checkpoint_epoch_10.pt

    # Ablation profile
    python training/train.py --profile tc_disabled

    # Override any key
    python training/train.py optimizer.lr=5e-5 data.batch_size=32

Requires: PyTorch ≥ 2.1, PyYAML, tqdm, (optional) wandb
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parents[1]))
from data.augmentation import get_augmentation_pipeline
from data.dataset import build_dataset
from models.mc_dropout import mc_predict
from models.registry import get_model
from models.temporal_consistency import TemporalConsistencyLoss
from training.lr_scheduler import build_scheduler, get_lr

logger = logging.getLogger(__name__)

# Optional W&B
try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set all RNG seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value applied to Python random, NumPy, and PyTorch (CPU + CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Seed set to %d", seed)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file.

    Parameters
    ----------
    config_path : str
        Path to ``training/config.yaml``.

    Returns
    -------
    dict
        Parsed config dictionary.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_profile(cfg: Dict, profile_name: str) -> Dict:
    """Merge an ablation profile into the base config (deep update).

    Parameters
    ----------
    cfg : dict
        Base config dictionary.
    profile_name : str
        Key of the profile inside ``cfg["ablation_profiles"]``.

    Returns
    -------
    dict
        Updated config dict.

    Raises
    ------
    KeyError
        If the profile is not found.
    """
    profiles = cfg.get("ablation_profiles", {})
    if profile_name not in profiles:
        raise KeyError(
            f"Profile {profile_name!r} not found. " f"Available: {list(profiles)}"
        )
    profile = profiles[profile_name]
    cfg = copy.deepcopy(cfg)
    _deep_update(cfg, profile)
    logger.info("Applied ablation profile: %s", profile_name)
    return cfg


def _deep_update(base: Dict, override: Dict) -> None:
    """Recursively merge override into base in-place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def apply_cli_overrides(cfg: Dict, overrides: list[str]) -> Dict:
    """Apply dot-notation CLI overrides to the config dict.

    Example: ``["optimizer.lr=5e-5", "data.batch_size=32"]``

    Parameters
    ----------
    cfg : dict
        Config dict to update in-place (a deepcopy is made internally).
    overrides : list of str
        Each element of the form ``"section.key=value"`` or
        ``"section.subsection.key=value"``.

    Returns
    -------
    dict
        Updated config dict.
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            logger.warning("Skipping malformed override (no '='): %s", item)
            continue
        key_path, raw_value = item.split("=", 1)
        keys = key_path.strip().split(".")
        # Parse value: try int, float, bool, then str
        value: Any
        try:
            value = int(raw_value)
        except ValueError:
            try:
                value = float(raw_value)
            except ValueError:
                if raw_value.lower() in ("true", "yes"):
                    value = True
                elif raw_value.lower() in ("false", "no"):
                    value = False
                elif raw_value.lower() in ("null", "none"):
                    value = None
                else:
                    value = raw_value

        # Traverse and set
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
        logger.info("CLI override: %s = %r", key_path, value)

    return cfg


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: TemporalConsistencyLoss,
    device: torch.device,
    mc_samples: int = 10,
    mc_batch_size: int = 5,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate the model on a validation DataLoader.

    Uses MC Dropout for prediction. Computes Dice, IoU, precision, recall,
    and the combined loss.

    Parameters
    ----------
    model : nn.Module
        The segmentation model.
    loader : DataLoader
        Validation DataLoader.
    criterion : TemporalConsistencyLoss
        Loss function (used for val_loss computation; TC is always 0 at eval).
    device : torch.device
        Computation device.
    mc_samples : int
        MC Dropout forward passes. Default 10.
    mc_batch_size : int
        Samples per GPU launch. Default 5.
    threshold : float
        Binary segmentation threshold. Default 0.5.

    Returns
    -------
    dict
        Validation metrics: ``val_loss``, ``val_dice``, ``val_iou``,
        ``val_precision``, ``val_recall``, ``val_uncertainty_mean``.
    """
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    total_prec = 0.0
    total_rec = 0.0
    total_unc = 0.0
    total_loss = 0.0
    n_batches = 0
    eps = 1e-6

    for batch in loader:
        images: Tensor = batch["image"].to(device)
        masks: Tensor = batch["mask"].to(device)

        # MC Dropout prediction
        mean_pred, uncertainty = mc_predict(
            model, images, n_samples=mc_samples, mc_batch_size=mc_batch_size
        )

        # Val loss (uses mean logit; no TC at eval)
        logits = model(images)
        loss_dict = criterion(logits, masks, slice_indices=None)
        total_loss += loss_dict["total"].item()

        # Binary metrics
        binary = (mean_pred >= threshold).float()
        tp = (binary * masks).sum(dim=(1, 2, 3))
        fp = (binary * (1.0 - masks)).sum(dim=(1, 2, 3))
        fn = ((1.0 - binary) * masks).sum(dim=(1, 2, 3))

        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        prec = (tp + eps) / (tp + fp + eps)
        rec = (tp + eps) / (tp + fn + eps)

        total_dice += dice.mean().item()
        total_iou += iou.mean().item()
        total_prec += prec.mean().item()
        total_rec += rec.mean().item()
        total_unc += uncertainty.mean().item()
        n_batches += 1

    n = max(n_batches, 1)
    metrics = {
        "val_loss": total_loss / n,
        "val_dice": total_dice / n,
        "val_iou": total_iou / n,
        "val_precision": total_prec / n,
        "val_recall": total_rec / n,
        "val_uncertainty_mean": total_unc / n,
    }
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    cfg: Dict,
) -> None:
    """Save a full training checkpoint.

    Parameters
    ----------
    path : Path
        Output file path (``*.pt``).
    model : nn.Module
        Model whose ``state_dict`` is saved.
    optimizer : Optimizer
        Optimizer state.
    scheduler : LRScheduler
        Scheduler state.
    epoch : int
        Current epoch.
    metrics : dict
        Validation metrics at this checkpoint.
    cfg : dict
        Training config (serialised for reproducibility).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if hasattr(scheduler, "state_dict") else {}
        ),
        "metrics": metrics,
        "config": cfg,
    }
    torch.save(state, path)
    logger.info("Checkpoint saved → %s", path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, Dict[str, float]]:
    """Load a checkpoint and restore model (+ optionally optimizer) state.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.
    model : nn.Module
        Model to restore.
    optimizer : Optimizer, optional
        If provided, restore optimizer state.
    scheduler : LRScheduler, optional
        If provided, restore scheduler state.
    device : torch.device, optional
        Map location for tensors.

    Returns
    -------
    epoch : int
        Epoch at which the checkpoint was saved.
    metrics : dict
        Validation metrics at the checkpoint.
    """
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as exc:
            logger.warning("Could not restore scheduler state: %s", exc)
    epoch = ckpt.get("epoch", 0)
    metrics = ckpt.get("metrics", {})
    logger.info(
        "Checkpoint loaded: epoch=%d  val_dice=%.4f", epoch, metrics.get("val_dice", 0)
    )
    return epoch, metrics


# ---------------------------------------------------------------------------
# W&B initialisation
# ---------------------------------------------------------------------------


def init_wandb(cfg: Dict, run_dir: Path) -> bool:
    """Initialise Weights & Biases if available and configured.

    Parameters
    ----------
    cfg : dict
        Full training config.
    run_dir : Path
        Run output directory (used as W&B dir).

    Returns
    -------
    bool
        True if W&B was successfully initialised.
    """
    log_cfg = cfg.get("logging", {})
    if not (log_cfg.get("use_wandb", False) and _WANDB_AVAILABLE):
        return False
    try:
        wandb.init(
            project=log_cfg.get("wandb_project", "sam2-lung-nodule-seg"),
            entity=log_cfg.get("wandb_entity", None) or None,
            name=cfg["experiment"]["name"],
            id=cfg["experiment"].get("run_id") or None,
            config=cfg,
            dir=str(run_dir),
            tags=log_cfg.get("wandb_tags", []),
            resume="allow",
        )
        logger.info("W&B run: %s", wandb.run.url)
        return True
    except Exception as exc:
        logger.warning("W&B init failed: %s  (continuing without W&B)", exc)
        return False


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(cfg: Dict) -> None:
    """Full training loop.

    Parameters
    ----------
    cfg : dict
        Merged training config dict (loaded from YAML + CLI overrides + profile).
    """
    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    opt_cfg = cfg["optimizer"]
    loss_cfg = cfg["loss"]
    train_cfg = cfg["training"]
    mc_cfg = cfg["mc_dropout"]

    # ── Reproducibility ────────────────────────────────────────────────────
    set_seed(int(exp_cfg.get("seed", 42)))

    # ── Device and mixed precision ─────────────────────────────────────────
    device_str: str = exp_cfg.get("device", "cuda")
    device = torch.device(
        device_str if torch.cuda.is_available() or device_str != "cuda" else "cpu"
    )
    use_amp: bool = bool(exp_cfg.get("mixed_precision", True)) and device.type == "cuda"
    logger.info("Device: %s | AMP: %s", device, use_amp)

    # ── Output directory ───────────────────────────────────────────────────
    run_id = exp_cfg.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(exp_cfg.get("output_dir", "runs")) / exp_cfg["name"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Save resolved config next to checkpoints for reproducibility
    with open(run_dir / "config_resolved.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    logger.info("Run directory: %s", run_dir)

    # ── W&B ───────────────────────────────────────────────────────────────
    cfg["experiment"]["run_id"] = run_id
    wb_active = init_wandb(cfg, run_dir)

    # ── TensorBoard ───────────────────────────────────────────────────────
    tb_writer = None
    if cfg.get("logging", {}).get("tensorboard", True):
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
            logger.info("TensorBoard logging → %s", run_dir / "tensorboard")
        except ImportError:
            logger.warning("torch.utils.tensorboard not available — TB disabled")

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir: str = data_cfg.get("data_dir", "SYNTHETIC")
    aug_cfg: Dict = data_cfg.get("augmentation", {})
    aug_transform = get_augmentation_pipeline(aug_cfg, augment=True)
    no_aug_transform = get_augmentation_pipeline({}, augment=False)

    train_ds = build_dataset(
        data_dir, split="train", mode="slice", transform=aug_transform
    )
    val_ds = build_dataset(
        data_dir, split="val", mode="slice", transform=no_aug_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(data_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        prefetch_factor=(
            int(data_cfg.get("prefetch_factor", 2))
            if data_cfg.get("num_workers", 4) > 0
            else None
        ),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(data_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=False,
    )
    logger.info(
        "Datasets: train=%d slices | val=%d slices | batch=%d",
        len(train_ds),
        len(val_ds),
        data_cfg.get("batch_size", 16),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = get_model(
        model_cfg.get("name", "sam2_lung_seg"),
        embed_dim=int(model_cfg.get("embed_dim", 256)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        attn_dropout=float(model_cfg.get("attn_dropout", 0.1)),
        proj_dropout=float(model_cfg.get("proj_dropout", 0.1)),
        encoder_frozen=bool(model_cfg.get("encoder_frozen", True)),
        sam2_checkpoint=model_cfg.get("sam2_checkpoint"),
        sam2_config=model_cfg.get("sam2_config"),
    ).to(device)

    if bool(exp_cfg.get("compile_model", False)):
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            logger.info("torch.compile enabled")
        except Exception as exc:
            logger.warning("torch.compile failed: %s", exc)

    # ── Optimizer ─────────────────────────────────────────────────────────
    enc_lr_mult = float(opt_cfg.get("encoder_lr_multiplier", 0.1))
    base_lr = float(opt_cfg.get("lr", 1e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))

    # Separate LR groups: encoder vs. decoder + adapter
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    other_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("encoder")
    ]
    param_groups = [
        {"params": encoder_params, "lr": base_lr * enc_lr_mult, "name": "encoder"},
        {"params": other_params, "lr": base_lr, "name": "decoder+adapter"},
    ]

    opt_name = opt_cfg.get("name", "AdamW").lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=weight_decay,
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=float(opt_cfg.get("eps", 1e-8)),
        )
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            param_groups, lr=base_lr, weight_decay=weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # ── Scheduler ─────────────────────────────────────────────────────────
    scheduler = build_scheduler(optimizer, cfg)

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = TemporalConsistencyLoss(
        lambda_bce=float(loss_cfg.get("lambda_bce", 0.5)),
        lambda_tc=float(loss_cfg.get("lambda_tc", 0.3)),
        consistency_mode=loss_cfg.get("consistency_mode", "l2"),
        warmup_epochs=int(loss_cfg.get("warmup_epochs", 5)),
        focal_alpha=float(loss_cfg.get("focal_alpha", 0.75)),
        focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
    )

    # ── AMP scaler ────────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 0
    best_val_dice = 0.0
    epochs_no_improve = 0
    resume_path = train_cfg.get("resume_from")
    if resume_path and Path(resume_path).exists():
        start_epoch, resume_metrics = load_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
        best_val_dice = resume_metrics.get("val_dice", 0.0)
        start_epoch += 1
        logger.info("Resuming from epoch %d", start_epoch)

    # ── Training loop ─────────────────────────────────────────────────────
    total_epochs = int(train_cfg.get("epochs", 50))
    val_interval = int(train_cfg.get("val_interval", 1))
    log_interval = int(train_cfg.get("log_interval", 10))
    save_interval = int(train_cfg.get("save_interval", 5))
    early_pat = int(train_cfg.get("early_stopping_patience", 10))
    clip_norm = opt_cfg.get("clip_grad_norm")
    enc_frozen_epochs = int(model_cfg.get("encoder_frozen_epochs", 5))
    log_images_every = cfg.get("logging", {}).get("log_images_every_n_epochs", 5)
    mc_val = int(cfg.get("evaluation", {}).get("mc_samples_val", 10))

    logger.info("Starting training: %d → %d epochs", start_epoch, total_epochs)
    global_step = 0
    history: list[Dict] = []

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()

        # ── Encoder freeze / unfreeze schedule ────────────────────────────
        if epoch == enc_frozen_epochs and bool(model_cfg.get("encoder_frozen", True)):
            logger.info("Epoch %d: unfreezing encoder", epoch)
            model.unfreeze_encoder()

        # ── Update TC loss epoch ──────────────────────────────────────────
        criterion.set_epoch(epoch)

        # ── Train epoch ───────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_tc = 0.0
        n_batches_train = 0

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [train]", leave=False
        )
        for batch_idx, batch in enumerate(pbar):
            images: Tensor = batch["image"].to(device, non_blocking=True)
            masks: Tensor = batch["mask"].to(device, non_blocking=True)
            slice_indices: Tensor = batch.get(
                "slice_idx", torch.zeros(images.shape[0], dtype=torch.long)
            ).to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss_dict = criterion(logits, masks, slice_indices=slice_indices)
                loss = loss_dict["total"]

            scaler.scale(loss).backward()

            if clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))

            scaler.step(optimizer)
            scaler.update()

            # Batch metrics
            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).float()
                eps = 1e-6
                tp = (preds * masks).sum(dim=(1, 2, 3))
                fp = (preds * (1 - masks)).sum(dim=(1, 2, 3))
                fn = ((1 - preds) * masks).sum(dim=(1, 2, 3))
                batch_dice = ((2 * tp + eps) / (2 * tp + fp + fn + eps)).mean().item()

            running_loss += loss.item()
            running_dice += batch_dice
            running_tc += loss_dict["temporal"].item()
            n_batches_train += 1
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{batch_dice:.4f}",
                lr=f"{get_lr(optimizer):.2e}",
            )

            if batch_idx % log_interval == 0:
                lr_now = get_lr(optimizer)
                step_metrics = {
                    "train/loss": loss.item(),
                    "train/dice": batch_dice,
                    "train/loss_dice": loss_dict["dice"].item(),
                    "train/loss_bce": loss_dict["bce"].item(),
                    "train/loss_tc": loss_dict["temporal"].item(),
                    "train/lr": lr_now,
                    "epoch": epoch,
                }
                if wb_active:
                    wandb.log(step_metrics, step=global_step)
                if tb_writer is not None:
                    for k, v in step_metrics.items():
                        tb_writer.add_scalar(k, v, global_step)

        epoch_lr = get_lr(optimizer)
        avg_train_loss = running_loss / max(n_batches_train, 1)
        avg_train_dice = running_dice / max(n_batches_train, 1)
        avg_tc = running_tc / max(n_batches_train, 1)

        # ── LR step ───────────────────────────────────────────────────────
        # ReduceLROnPlateau needs a metric; step after eval below
        is_plateau = hasattr(scheduler, "patience")

        # ── Validation ────────────────────────────────────────────────────
        val_metrics: Dict[str, float] = {}
        if (epoch + 1) % val_interval == 0:
            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                mc_samples=mc_val,
                mc_batch_size=int(mc_cfg.get("mc_batch_size", 5)),
                threshold=float(mc_cfg.get("threshold", 0.5)),
            )
            val_dice = val_metrics.get("val_dice", 0.0)
            epoch_time = time.time() - epoch_start

            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_dice": avg_train_dice,
                "train_tc_loss": avg_tc,
                "lr": epoch_lr,
                **val_metrics,
                "epoch_time_s": epoch_time,
            }
            history.append(epoch_log)

            # Log to console
            logger.info(
                "Epoch %3d/%d | tr_loss=%.4f tr_dice=%.4f | "
                "val_dice=%.4f val_iou=%.4f | lr=%.2e | tc_loss=%.4f | %.1fs",
                epoch + 1,
                total_epochs,
                avg_train_loss,
                avg_train_dice,
                val_dice,
                val_metrics.get("val_iou", 0.0),
                epoch_lr,
                avg_tc,
                epoch_time,
            )

            # Log to W&B / TB
            if wb_active:
                wandb.log(
                    {
                        "train/avg_loss": avg_train_loss,
                        "train/avg_dice": avg_train_dice,
                        **{f"eval/{k}": v for k, v in val_metrics.items()},
                        "lr": epoch_lr,
                    },
                    step=global_step,
                )
            if tb_writer is not None:
                for k, v in val_metrics.items():
                    tb_writer.add_scalar(f"eval/{k}", v, epoch)

            # Log image grid every N epochs
            if tb_writer is not None and (epoch + 1) % log_images_every == 0:
                try:
                    sample_batch = next(iter(val_loader))
                    with torch.no_grad():
                        s_img = sample_batch["image"][:4].to(device)
                        s_msk = sample_batch["mask"][:4]
                        s_pred = torch.sigmoid(model(s_img)).cpu()
                    import torchvision

                    grid = torchvision.utils.make_grid(
                        torch.cat([s_img.cpu(), s_msk, s_pred], dim=0),
                        nrow=4,
                        normalize=True,
                    )
                    tb_writer.add_image("val/predictions", grid, epoch)
                except Exception as exc:
                    logger.debug("Image logging failed: %s", exc)

            # ── ReduceLROnPlateau step ────────────────────────────────────
            if is_plateau:
                monitor_metric = val_metrics.get("val_loss", avg_train_loss)
                scheduler.step(monitor_metric)

            # ── Early stopping ────────────────────────────────────────────
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                epochs_no_improve = 0
                # Save best checkpoint
                save_checkpoint(
                    ckpt_dir / "best_model.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    val_metrics,
                    cfg,
                )
                logger.info(
                    "★ New best val_dice=%.4f — checkpoint saved", best_val_dice
                )
                if wb_active:
                    wandb.run.summary["best_val_dice"] = best_val_dice
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_pat:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch + 1,
                        early_pat,
                    )
                    break

        # ── Regular checkpoint ────────────────────────────────────────────
        if not is_plateau:
            scheduler.step()

        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                ckpt_dir / f"checkpoint_epoch_{epoch+1:03d}.pt",
                model,
                optimizer,
                scheduler,
                epoch + 1,
                val_metrics or {"val_dice": avg_train_dice},
                cfg,
            )

    # ── Post-training ──────────────────────────────────────────────────────
    logger.info("Training complete. Best val_dice=%.4f", best_val_dice)

    # Save training history as JSON
    hist_path = run_dir / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved → %s", hist_path)

    if wb_active:
        wandb.finish()
    if tb_writer is not None:
        tb_writer.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train SAM2 Lung Nodule Segmentation model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training/train.py                              # synthetic data, default config
  python training/train.py data.data_dir=SYNTHETIC     # explicit synthetic
  python training/train.py data.data_dir=data/processed/ optimizer.lr=5e-5
  python training/train.py --profile tc_disabled
  python training/train.py --config training/my_config.yaml
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config YAML. Default: training/config.yaml",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Ablation profile name (from ablation_profiles in config.yaml).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dot-notation key=value overrides, e.g. optimizer.lr=5e-5",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # Load config
    cfg = load_config(args.config)

    # Apply ablation profile
    if args.profile:
        cfg = apply_profile(cfg, args.profile)

    # Apply CLI overrides
    if args.overrides:
        cfg = apply_cli_overrides(cfg, args.overrides)

    # Run training
    train(cfg)


if __name__ == "__main__":
    main()
