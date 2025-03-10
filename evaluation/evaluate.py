"""
Full evaluation pipeline for SAM2 Lung Nodule Segmentation.

Orchestrates models/mc_dropout, evaluation/dice_metric,
evaluation/uncertainty_calibration, and evaluation/radiologist_agreement
over the test split or any DataLoader.

Usage::

    # Synthetic data (no LUNA16 required)
    python evaluation/evaluate.py --data_dir SYNTHETIC

    # Real preprocessed data
    python evaluation/evaluate.py --data_dir data/processed/ \\
        --checkpoint runs/sam2_lung_seg_v1/best/best_model.pt \\
        --output_dir results/ --n_mc 25

The script produces:
    results/metrics.json          — all numerical metrics
    results/calibration.png       — reliability diagram
    results/bland_altman.png      — Bland-Altman plot (if radiologist CSV provided)
    results/per_case_metrics.csv  — per-patch Dice / IoU / uncertainty
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))

from data.dataset import build_dataset
from models.registry import get_model
from models.mc_dropout import mc_predict, compute_uncertainty_stats
from evaluation.dice_metric import DiceMetric, compute_all_metrics
from evaluation.uncertainty_calibration import CalibrationAnalyzer
from evaluation.radiologist_agreement import RadiologistAgreement

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation runner
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_mc_samples: int = 25,
    mc_batch_size: int = 5,
    threshold: float = 0.5,
    output_dir: Optional[str] = None,
    radiologist_csv: Optional[str] = None,
) -> Dict[str, object]:
    """Run the complete evaluation pipeline on a DataLoader.

    Processes every batch through MC Dropout inference and accumulates:
    - Dice / IoU / precision / recall via ``DiceMetric``
    - Calibration (ECE, Brier, UncAUC) via ``CalibrationAnalyzer``
    - Per-case records for the CSV report

    Parameters
    ----------
    model : nn.Module
        The trained SAM2LungSegmentor.
    loader : DataLoader
        Test / validation DataLoader (returns dicts with ``image``, ``mask``,
        ``patch_id``, optionally ``slice_idx``).
    device : torch.device
        Compute device.
    n_mc_samples : int
        MC Dropout forward passes. Default 25.
    mc_batch_size : int
        Samples per GPU launch. Default 5.
    threshold : float
        Binary segmentation threshold. Default 0.5.
    output_dir : str or None
        If provided, write all output artefacts here.
    radiologist_csv : str or None
        Path to a CSV with columns ``study_id, radiologist_label, rad_volume_mm3``
        for each radiologist. Enables Bland-Altman + κ analysis.

    Returns
    -------
    dict
        All evaluation metrics (segmentation + calibration + agreement).
    """
    model.eval()

    dice_metric = DiceMetric(threshold=threshold, compute_hd95=False)
    calib_analyzer = CalibrationAnalyzer(n_bins=15, subsample_rate=0.05)

    per_case_rows: List[Dict] = []

    logger.info(
        "Evaluating: %d batches | MC samples=%d | threshold=%.2f",
        len(loader), n_mc_samples, threshold,
    )

    from tqdm import tqdm
    pbar = tqdm(loader, desc="Evaluating", unit="batch")

    for batch in pbar:
        images: Tensor = batch["image"].to(device, non_blocking=True)
        masks: Tensor = batch["mask"].to(device, non_blocking=True)
        patch_ids: List[str] = batch["patch_id"] if isinstance(batch["patch_id"], list) else [batch["patch_id"]]

        # MC Dropout inference
        mean_pred, uncertainty = mc_predict(
            model, images,
            n_samples=n_mc_samples,
            mc_batch_size=mc_batch_size,
            sigmoid=True,
        )

        # Accumulate segmentation metrics
        dice_metric.update(mean_pred, masks)

        # Accumulate calibration metrics
        calib_analyzer.update(mean_pred, uncertainty, masks, threshold=threshold)

        # Per-case metrics
        B = images.shape[0]
        for b in range(B):
            m = compute_all_metrics(
                mean_pred[b : b + 1],
                masks[b : b + 1],
                threshold=threshold,
                include_hd95=False,
            )
            unc_stats = compute_uncertainty_stats(
                uncertainty[b : b + 1],
                (mean_pred[b : b + 1] >= threshold).float(),
            )
            row = {
                "patch_id": patch_ids[b] if b < len(patch_ids) else f"sample_{b}",
                "dice": round(m["dice"], 6),
                "iou": round(m["iou"], 6),
                "precision": round(m["precision"], 6),
                "recall": round(m["recall"], 6),
                "unc_mean": round(unc_stats["unc_mean"], 8),
                "unc_max": round(unc_stats["unc_max"], 8),
                "unc_in_pred": round(unc_stats["unc_in_pred_mean"], 8),
            }
            per_case_rows.append(row)

        pbar.set_postfix(dice=f"{m['dice']:.4f}")

    # Compute aggregate metrics
    seg_metrics = dice_metric.compute()
    calib_results = calib_analyzer.compute()

    # Radiologist agreement (if CSV provided)
    ra_results: Dict = {}
    if radiologist_csv and Path(radiologist_csv).exists():
        ra_results = _run_radiologist_agreement(
            radiologist_csv, per_case_rows,
            save_ba_plot=str(Path(output_dir) / "bland_altman.png") if output_dir else None,
        )

    # Assemble full results
    results: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "n_mc_samples": n_mc_samples,
        "threshold": threshold,
        "n_cases": len(per_case_rows),
        # Segmentation
        "dice": seg_metrics["dice"],
        "iou": seg_metrics["iou"],
        "precision": seg_metrics["precision"],
        "recall": seg_metrics["recall"],
        # Calibration
        "ece": calib_results.get("ece"),
        "brier": calib_results.get("brier"),
        "uncertainty_auc": calib_results.get("uncertainty_auc"),
        # Radiologist agreement
        **{f"agreement_{k}": v for k, v in ra_results.items() if isinstance(v, (int, float, str))},
    }

    # Log summary
    logger.info(
        "Results: Dice=%.4f | IoU=%.4f | Prec=%.4f | Rec=%.4f | "
        "ECE=%.4f | Brier=%.4f | UncAUC=%.4f",
        results["dice"], results["iou"], results["precision"], results["recall"],
        results.get("ece", float("nan")),
        results.get("brier", float("nan")),
        results.get("uncertainty_auc", float("nan")),
    )

    # Write outputs
    if output_dir:
        _write_outputs(
            output_dir=output_dir,
            results=results,
            per_case_rows=per_case_rows,
            calib_analyzer=calib_analyzer,
        )

    return results


# ---------------------------------------------------------------------------
# Radiologist agreement helper
# ---------------------------------------------------------------------------


def _run_radiologist_agreement(
    csv_path: str,
    per_case_rows: List[Dict],
    save_ba_plot: Optional[str] = None,
) -> Dict:
    """Load radiologist CSV and compute agreement vs. model.

    CSV format (tab or comma separated)::

        study_id, radiologist_id, label, volume_mm3

    Parameters
    ----------
    csv_path : str
        Path to radiologist ratings CSV.
    per_case_rows : list of dict
        Per-case model results (from ``run_evaluation``).
    save_ba_plot : str or None
        Path to save Bland-Altman plot.

    Returns
    -------
    dict
        Agreement statistics from ``RadiologistAgreement.compute()``.
    """
    import csv as csv_mod

    # Build lookup: study_id → {rad_id: (label, volume)}
    study_data: Dict[str, Dict[str, Tuple[int, float]]] = {}
    with open(csv_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            sid = row["study_id"].strip()
            rid = row["radiologist_id"].strip()
            lbl = int(row["label"])
            vol = float(row.get("volume_mm3", 0) or 0)
            study_data.setdefault(sid, {})[rid] = (lbl, vol)

    # Determine unique radiologists
    all_rads = sorted({rid for d in study_data.values() for rid in d})
    n_rads = len(all_rads)
    if n_rads == 0:
        return {}

    logger.info("RadiologistAgreement from CSV: %d studies, %d raters", len(study_data), n_rads)

    # Build model label lookup from per_case_rows
    model_lookup: Dict[str, int] = {}
    model_vol_lookup: Dict[str, float] = {}
    for row in per_case_rows:
        pid = row["patch_id"]
        # Derive study_id: strip '_slice_*' suffix if present
        sid = pid.split("_slice_")[0] if "_slice_" in pid else pid
        model_lookup[sid] = 1 if row["dice"] > 0.3 else 0
        model_vol_lookup[sid] = 0.0  # Placeholder (real vol from 3D inference)

    ra = RadiologistAgreement(n_radiologists=n_rads)
    for sid, rad_dict in study_data.items():
        rad_labels = [rad_dict.get(r, (0, 0.0))[0] for r in all_rads]
        rad_vols = [rad_dict.get(r, (0, 0.0))[1] for r in all_rads]
        model_lbl = model_lookup.get(sid, 0)
        ra.add_study(
            study_id=sid,
            model_label=model_lbl,
            radiologist_labels=rad_labels,
            model_volume_mm3=model_vol_lookup.get(sid),
            radiologist_volumes_mm3=rad_vols if any(v > 0 for v in rad_vols) else None,
        )

    return ra.compute(save_ba_plot=save_ba_plot)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_outputs(
    output_dir: str,
    results: Dict,
    per_case_rows: List[Dict],
    calib_analyzer: CalibrationAnalyzer,
) -> None:
    """Write JSON + CSV + calibration plot to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # metrics.json
    metrics_path = out / "metrics.json"
    # Convert non-serialisable values
    serialisable = {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in results.items()
    }
    with open(metrics_path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    logger.info("Metrics saved → %s", metrics_path)

    # per_case_metrics.csv
    csv_path = out / "per_case_metrics.csv"
    if per_case_rows:
        fieldnames = list(per_case_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_case_rows)
        logger.info("Per-case CSV saved → %s", csv_path)

    # Calibration plot
    calib_path = str(out / "calibration.png")
    calib_analyzer.compute(save_diagram=calib_path)


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    model_name: str = "sam2_lung_seg",
    **model_kwargs,
) -> nn.Module:
    """Load a model from a training checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to ``*.pt`` checkpoint saved by ``train.py``.
    device : torch.device
        Computation device.
    model_name : str
        Model variant name from the registry. Default ``"sam2_lung_seg"``.
    **model_kwargs
        Additional kwargs forwarded to ``get_model()``.

    Returns
    -------
    nn.Module
        Model with weights restored, set to eval mode.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    model = get_model(
        model_cfg.get("name", model_name),
        embed_dim=int(model_cfg.get("embed_dim", 256)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        attn_dropout=float(model_cfg.get("attn_dropout", 0.1)),
        proj_dropout=float(model_cfg.get("proj_dropout", 0.1)),
        encoder_frozen=False,  # doesn't matter at eval
        **model_kwargs,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    val_dice = ckpt.get("metrics", {}).get("val_dice", "?")
    logger.info(
        "Loaded checkpoint: epoch=%s | val_dice=%s | %s",
        epoch, val_dice, checkpoint_path,
    )
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate SAM2 Lung Nodule Segmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/evaluate.py --data_dir SYNTHETIC
  python evaluation/evaluate.py --data_dir data/processed/ \\
      --checkpoint runs/best_model.pt --output_dir results/ --n_mc 25
        """,
    )
    p.add_argument("--data_dir", default="SYNTHETIC", help="Data directory or 'SYNTHETIC'")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    p.add_argument("--output_dir", default="results/eval", help="Output directory for reports")
    p.add_argument("--n_mc", type=int, default=25, help="MC Dropout samples")
    p.add_argument("--mc_batch_size", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--radiologist_csv", default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load dataset
    ds = build_dataset(args.data_dir, split=args.split, mode="slice", augment=False)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    logger.info("Dataset: %d samples | split=%s", len(ds), args.split)

    # Load model
    if args.checkpoint:
        model = load_model_from_checkpoint(args.checkpoint, device)
    else:
        logger.warning("No checkpoint provided — using randomly initialised model (for smoke test)")
        model = get_model("sam2_lung_seg", encoder_frozen=False).to(device)
        model.eval()

    # Run evaluation
    results = run_evaluation(
        model=model,
        loader=loader,
        device=device,
        n_mc_samples=args.n_mc,
        mc_batch_size=args.mc_batch_size,
        threshold=args.threshold,
        output_dir=args.output_dir,
        radiologist_csv=args.radiologist_csv,
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<35s} {v:.6f}")
        elif isinstance(v, (int, str)):
            print(f"  {k:<35s} {v}")
    print(f"\nOutputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
