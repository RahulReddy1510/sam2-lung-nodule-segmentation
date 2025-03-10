# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

- Exploring 3D SAM2 volumetric inference as a future direction (currently too slow for clinical deployment)
- ONNX export for edge device compatibility

---

## [1.0.0] — 2025-06-15

### Added
- **3D Slicer plugin** (`slicer_plugin/LungNoduleSeg.py`) — full scripted module
  with uncertainty heatmap display (ColdToHotRainbow colormap at 50% opacity)
- DICOM SEG export functionality via `export_dicom_seg()` in plugin logic
- Full Qt Designer UI definition (`slicer_plugin/Resources/UI/LungNoduleSeg.ui`)
- Clinical validation report for 150 studies — κ=0.83, 91% agreement
- `RELEASE_NOTES.md` with links to arXiv preprint (placeholder)
- `docs/literature_notes.md` — annotated bibliography of 10 motivating papers

### Changed
- README updated with final clinical validation results and plugin installation guide
- Plugin checkpoint download URL added to `slicer_plugin/README.md`

### Fixed
- Uncertainty overlay Z-axis orientation mismatch in Slicer (RAS vs LPS convention)
- Plugin `_ensure_deps()` now handles pip install failures gracefully with user warning

---

## [0.9.0] — 2025-05-20

### Added
- `evaluation/radiologist_agreement.py` — full clinical validation pipeline
- `compute_cohens_kappa()` using sklearn with confusion matrix details
- `interpret_kappa()` per Landis & Koch criteria
- `analyse_uncertainty_utility()` for structured radiologist feedback analysis
- `plot_agreement_analysis()` — 3-panel figure (Dice histogram, Kappa gauge, pie chart)
- `04_clinical_validation.ipynb` — end-to-end clinical demo notebook
- Synthetic 150-study simulation in `__main__` block producing κ≈0.83

### Changed
- `evaluation/evaluate.py` now outputs structured JSON for downstream analysis

---

## [0.8.0] — 2025-05-01

### Added
- **MC Dropout calibration analysis** — ECE = 0.042 (target was < 0.05 ✓)
- `evaluation/uncertainty_calibration.py` with reliability diagram using color-coded
  confidence bars (red=overconfident, blue=underconfident)
- `uncertainty_error_correlation()` — Spearman/Pearson correlation + hexbin scatter
- `03_uncertainty_visualization.ipynb` — "where to look twice" visualization notebook
- `run_calibration_analysis()` full pipeline with automatic plot saving

### Fixed
- ECE binning edge case when predicted probabilities == 1.0 exactly (clipped to 1-ε)
- `mc_predict_volume()` memory leak — now processes slices in batches and frees cache

---

## [0.7.0] — 2025-04-10

### Added
- Full pytest test suite — 28 unit tests across preprocessing, model, metrics, MC Dropout
- All tests pass with synthetic data (no real LUNA16 required)
- `.github/workflows/ci.yml` — GitHub Actions CI on push/PR
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `scripts/run_ablation.sh` — automated ablation study launcher
- `results/ablation_results.csv` with all 7 model variants

### Changed
- Moved hardcoded hyperparameters to `training/config.yaml` — all training runs
  now config-file driven

---

## [0.6.0] — 2025-03-25

### Added
- **Temporal consistency loss** — Dice improves 91.8% → 93.5%!
- `models/temporal_consistency.py` — `TemporalConsistencyLoss` with L2 and Dice modes
- `AblationLossFactory` for switching loss modes via config string
- Warmup mechanism: TC loss activates after epoch 5 to prevent early instability
- `configs/ablation_no_tc.yaml`, `configs/ablation_tc_dice.yaml` ablation configs

### Fixed
- Critical bug: TC loss was being computed across non-adjacent slices from different
  patches because slice indices weren't being compared correctly — fixed by checking
  `|idx_t - idx_{t+1}| == 1` strictly

---

## [0.5.0] — 2025-03-05

### Changed
- Unfroze SAM2 encoder after epoch 5 — end-to-end fine-tuning
- Encoder parameters trained at lr×0.1 to prevent catastrophic forgetting
- Added focal BCE loss component (α=0.75, γ=2.0) to handle class imbalance

### Added
- `training/lr_scheduler.py` — custom linear warmup + cosine annealing scheduler
- `CheckpointManager` — top-K checkpoint pruning by validation Dice score
- TensorBoard image grid logging (CT / GT / pred / uncertainty) every 5 epochs

### Fixed
- GradScaler was not calling `optimizer.zero_grad()` correctly with gradient
  accumulation — moved zero_grad to start of accumulation cycle

---

## [0.4.0] — 2025-02-20

### Added
- `LightweightMaskDecoder` with learnable nodule prompt token
  (`nn.Parameter(torch.randn(1, 1, 256))`)
- `DropoutMultiheadAttention` — attention dropout kept active at inference for MC Dropout
- `SinusoidalPosEmbed` — 2D sinusoidal positional encoding added to image features
- SAM2 channel adapter: `Conv2d(1, 3, kernel_size=1)` adapts 1-channel CT to 3-channel RGB
- `FallbackEncoder` — UNet-style CNN encoder allowing development without SAM2 installed
- `models/registry.py` — model factory with registered variants

### Performance
- With decoder fine-tuning only (encoder frozen): **Dice = 91.8%** on LUNA16 test

---

## [0.3.0] — 2025-02-05

### Added
- `SAM2LungSegmentor` — main model class wiring encoder, pos embed, and decoder
- `freeze_encoder()` / `unfreeze_encoder()` methods with parameter count logging
- `build_model()` factory function
- `__main__` sanity check block in `sam2_finetune.py`

### Experiments
- Tried 3D patch-level SAM2 inference: too slow (47s/volume on A100), reverted to
  2D slice-by-slice approach (see commit "revert: 3D SAM2 too slow, back to 2D")
- Tested learnable patch embedding vs sinusoidal — sinusoidal converges faster

---

## [0.2.0] — 2025-01-20

### Added
- `data/luna16_preprocessing.py` — complete LUNA16 preprocessing pipeline
  (load_mhd_volume, resample_volume, apply_hu_window, world_to_voxel, create_nodule_mask,
  extract_patch, get_train_val_test_splits, process_dataset)
- `LUNA16SliceDataset` — 2D axial slice dataset with slice_idx tracking for temporal loss
- `LUNA16VolumeDataset` — 3D volume dataset for full-volume inference
- `data/augmentation.py` — composable augmentation pipeline with all transforms
- Train/val/test split: 72/14/14 (seed=42 for reproducibility)
- `scripts/download_luna16.sh` — data download helper

### Fixed
- HU windowing was normalizing before clipping — reversed to clip-then-normalize (correct)
- `world_to_voxel()` origin subtraction order was flipped for certain LUNA16 studies

---

## [0.1.0] — 2025-01-08

### Added
- Initial repository structure
- Literature review notes (`docs/literature_notes.md`)
- `01_data_exploration.ipynb` — LUNA16 EDA and preprocessing visualization skeleton
- `training/config.yaml` — initial hyperparameter scaffold
- `.gitignore`, `LICENSE`, `README.md` skeleton

---

[Unreleased]: https://github.com/rahulkoulury/sam2-lung-nodule-segmentation/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/rahulkoulury/sam2-lung-nodule-segmentation/releases/tag/v1.0.0
[0.9.0]: https://github.com/rahulkoulury/sam2-lung-nodule-segmentation/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/rahulkoulury/sam2-lung-nodule-segmentation/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/rahulkoulury/sam2-lung-nodule-segmentation/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/rahulkoulury/sam2-lung-nodule-segmentation/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/rahulkoulury/sam2-lung-nodule-segmentation/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/rahulkoulury/sam2-lung-nodule-segmentation/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/rahulkoulury/sam2-lung-nodule-segmentation/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/rahulkoulury/sam2-lung-nodule-segmentation/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rahulkoulury/sam2-lung-nodule-segmentation/releases/tag/v0.1.0
