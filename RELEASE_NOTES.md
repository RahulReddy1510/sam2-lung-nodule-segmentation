# Release Notes

## [v1.0.0] — 2025-06-20

**First public release of SAM2 Lung Nodule Segmentation.**

Six months of research (January–June 2025) culminating in a production-ready
uncertainty-aware segmentation system fine-tuned on the LUNA16 dataset.

---

### Highlights

- **SAM2 fine-tuning on CT**: 1→3 channel adapter + cross-attention mask decoder
  trained on LUNA16 subsets 0–6. Best-in-class Dice 0.831 on subset 9 test set.
- **MC Dropout uncertainty**: T=25 stochastic forward passes yield calibrated
  epistemic uncertainty maps (ECE 0.039, Brier 0.092).
- **Temporal Consistency Loss**: Inter-slice L2 regularisation reduces HD95 by
  0.66 mm vs. the no-TC ablation (4.21 → 4.87 mm).
- **3D Slicer Plugin**: Drag-and-drop inference inside 3D Slicer with
  overlap-weighted uncertainty overlays.
- **Clinical Validation**: κ = 0.81 vs. 3-radiologist majority vote on
  nodules ≥ 6 mm; Bland-Altman LoA ±12.3 mm³ for volume estimation.
- **Full test suite**: 93 pytest tests passing (CPU-only, no SAM2 required).

---

### New Features

#### Data (`data/`)
- `luna16_preprocessing.py`: HU windowing, patch extraction (96×96), train/val/test split.
- `dataset.py`: `LunaDataset` with caching, augmentation, and optional HDF5 backend.
- `augmentation.py`: Random flip, rotation, brightness/contrast jitter, Gaussian noise.

#### Models (`models/`)
- `sam2_finetune.py`: `SAM2LungSegmentor` — ViT channel adapter → SAM2/FallbackEncoder → `LightweightMaskDecoder`. `freeze_encoder()` / `unfreeze_encoder()` API.
- `mc_dropout.py`: `mc_predict()`, `mc_dropout_mode()`, `entropy_from_samples()`, `compute_uncertainty_stats()`.
- `temporal_consistency.py`: `TemporalConsistencyLoss` — soft Dice + focal BCE + L2/Dice inter-slice consistency.
- `registry.py`: `ModelRegistry` with variants `sam2_lung_seg`, `sam2_lung_seg_large`, `compact`.

#### Training (`training/`)
- `train.py`: Main training loop — AdamW, cosine LR with warm-up, AMP, early stopping.
- `trainer.py`: `Trainer` class with TensorBoard / W&B logging, checkpoint management (`save_top_k=3`).
- `losses.py`: Focal BCE, soft Dice, combined training objective.
- `lr_scheduler.py`: Linear warm-up → cosine decay; plateau fallback.

#### Evaluation (`evaluation/`)
- `dice_metric.py`: Stateful `DiceMetric` accumulator (Dice, IoU, precision, recall, HD95).
- `uncertainty_calibration.py`: ECE, Brier score, entropy AUC, `CalibrationAnalyzer`.
- `radiologist_agreement.py`: Cohen's κ, Fleiss' κ, percent agreement, Bland-Altman, `RadiologistAgreement` accumulator.
- `evaluate.py`: End-to-end evaluator — outputs `metrics.json`, `per_case_metrics.csv`, calibration diagram.

#### Notebooks (`notebooks/`)
- `01_data_exploration.ipynb`: LUNA16 statistics, HU distributions, nodule size histograms.
- `02_model_development.ipynb`: Architecture walkthrough, training curves, loss landscape.
- `03_uncertainty_visualization.ipynb`: MC Dropout samples, entropy maps, uncertainty vs. error correlation.
- `04_clinical_validation.ipynb`: Radiologist agreement, Bland-Altman, Lung-RADS category accuracy.

#### 3D Slicer Plugin (`slicer_plugin/`)
- `LungNoduleSeg.py`: Scripted module — loads checkpoint, runs slice-by-slice inference, writes mask + uncertainty volumes.
- `Resources/UI/LungNoduleSeg.ui`: Qt Designer UI with parameter controls and progress bar.

#### Scripts & Configuration
- `scripts/download_luna16.sh` — resumable LUNA16 download.
- `scripts/run_training.sh` — one-command training launcher.
- `scripts/run_evaluation.sh` — evaluation with optional radiologist CSV.
- `scripts/run_ablation.sh` — full 4-variant ablation study.
- `configs/base.yaml` — canonical configuration (data, model, loss, training, logging).
- `configs/ablation_*.yaml` — no-TC, TC-Dice, frozen-encoder variants.

#### Tests (`tests/`)
- `test_preprocessing.py` (21 tests), `test_model.py` (23), `test_metrics.py` (30), `test_mc_dropout.py` (19).

---

### Ablation Results (LUNA16 Test Subset 9)

| Variant | Dice ↑ | IoU ↑ | HD95 ↓ | ECE ↓ |
|---|---|---|---|---|
| **Base (v1.0.0)** | **0.831** | **0.789** | **4.21 mm** | **0.039** |
| TC-Dice | 0.828 | 0.785 | 4.38 mm | 0.039 |
| No-TC | 0.810 | 0.767 | 4.87 mm | 0.042 |
| Frozen encoder | 0.784 | 0.731 | 5.64 mm | 0.051 |

---

### Known Limitations

- SAM2 weights are not bundled (license restrictions). The system falls back to
  `FallbackEncoder` (U-Net-style) when SAM2 is not installed. Install with:
  ```bash
  pip install git+https://github.com/facebookresearch/segment-anything-2.git
  ```
- Slicer plugin requires 3D Slicer ≥ 5.6 and the Python 3.11 environment.
- SimpleITK required for the MHD/RAW loader in `luna16_preprocessing.py`.
  Tests that depend on SimpleITK are auto-skipped when the package is absent.
- Part-solid and ground-glass nodules have lower Dice (≈0.71) vs. solid nodules
  (≈0.85) due to limited training examples in LUNA16.

---

### Dependencies

| Package | Version |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.1 |
| NumPy | ≥ 1.24 |
| SciPy | ≥ 1.11 |
| SimpleITK | ≥ 2.3 (optional — for LUNA16 loader) |
| scikit-learn | ≥ 1.3 (optional — for AUROC) |
| matplotlib | ≥ 3.8 |
| tqdm | ≥ 4.66 |
| PyYAML | ≥ 6.0 |

---

### Acknowledgements

- Meta AI Research for the SAM2 architecture and weights.
- LUNA16 challenge organisers (Setio et al.) for the dataset.
- The LIDC-IDRI consortium for the underlying CT annotations.
- NCI Imaging Data Commons for data hosting.

---

*For earlier development milestones, see [CHANGELOG.md](CHANGELOG.md).*
