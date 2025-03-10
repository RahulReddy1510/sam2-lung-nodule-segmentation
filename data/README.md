# LUNA16 Dataset — Download & Setup Guide

## Overview

This directory contains the data preprocessing pipeline for the
[LUNA16 (LUng Nodule Analysis 2016)](https://luna16.grand-challenge.org/) challenge dataset.
LUNA16 provides 888 CT scans with annotated pulmonary nodules ≥ 3mm, derived from the
LIDC-IDRI collection. It is the standard benchmark for lung nodule detection and segmentation.

---

## Data Access

LUNA16 data **requires registration** at the grand-challenge website. The annotation CSV is
publicly available via Zenodo (no registration required).

### Step 1 — Register and Download CT Volumes

1. Create an account at: https://luna16.grand-challenge.org/
2. Accept the data use agreement
3. Download the 10 subsets (`subset0.zip` through `subset9.zip`, ~120 GB total)
4. Also download `seg-lungs-LUNA16.zip` (lung segmentation masks, optional but useful)

**Automated download helper** (after registration, saves cookies):

```bash
bash scripts/download_luna16.sh
```

### Step 2 — Download Annotations CSV (No Registration Needed)

The nodule annotations CSV is available from Zenodo:

```bash
# Download annotations.csv (~500 KB)
wget https://zenodo.org/record/3723295/files/annotations.csv -O data/annotations.csv

# Download candidates.csv (for false positive reduction, optional)
wget https://zenodo.org/record/3723295/files/candidates.csv -O data/candidates.csv
```

---

## Expected Raw Directory Structure

After downloading and extracting, your raw data directory should look like:

```
/data/LUNA16/raw/
├── subset0/
│   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.mhd
│   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.zraw
│   └── ...  (88 scans per subset)
├── subset1/ ... subset9/   (10 subsets total, 888 scans)
└── seg-lungs-LUNA16/       (optional lung masks)

/data/LUNA16/
└── annotations.csv         (nodule center + radius annotations)
```

### annotations.csv Format

```
seriesuid,coordX,coordY,coordZ,diameter_mm
1.3.6.1.4...860,  68.42, -74.48, -288.7,  5.65
```

---

## Preprocessing Pipeline

Run the full preprocessing pipeline with:

```bash
python data/luna16_preprocessing.py \
    --input_dir  /data/LUNA16/raw \
    --output_dir /data/LUNA16/preprocessed \
    --annotations_csv /data/LUNA16/annotations.csv \
    --patch_size 96 \
    --num_workers 8
```

### What the Pipeline Does

1. **Load** `.mhd`/`.zraw` volumes using SimpleITK
2. **Resample** to isotropic 1×1×1 mm spacing using linear interpolation
3. **Window** HU values to `[-1000, 400]` and normalize to `[0, 1]`
4. **Convert** world-coordinate nodule centers to voxel indices
5. **Create** spherical binary masks (radius from annotations)
6. **Extract** 96×96×96 patches centered on each nodule (zero-padded at boundaries)
7. **Save** `{uid}_image.npy` and `{uid}_mask.npy` pairs

### Output Structure

```
/data/LUNA16/preprocessed/
├── train/
│   ├── 1.3.6...860_image.npy    # shape: (96, 96, 96), float32
│   ├── 1.3.6...860_mask.npy     # shape: (96, 96, 96), uint8
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### Split Ratios

| Split | Fraction | Approximate Count |
|-------|----------|-------------------|
| Train | 72%      | ~640 CT studies   |
| Val   | 14%      | ~124 CT studies   |
| Test  | 14%      | ~124 CT studies   |

Split is deterministic with `seed=42`. All nodules from a single CT study are kept
in the same split to prevent data leakage.

---

## Data Statistics (LUNA16)

| Property                    | Value                    |
|-----------------------------|--------------------------|
| Total CT studies            | 888                      |
| Total annotated nodules     | 1,186                    |
| Nodule diameter range       | 3.0 – 32.8 mm            |
| Nodule diameter (mean ± σ)  | 8.5 ± 5.2 mm             |
| CT slice thickness range    | 0.6 – 2.5 mm             |
| In-plane resolution range   | 0.46 – 0.98 mm/pixel     |
| Min nodule filtered         | < 3.0 mm (excluded)      |

---

## Minimum Disk Space Requirements

| Stage           | Size     |
|-----------------|----------|
| Raw CT (10 zips)| ~120 GB  |
| Extracted .mhd  | ~240 GB  |
| Preprocessed    | ~18 GB   |
| Peak (both)     | ~360 GB  |

---

## Synthetic Data for Development

All training and evaluation scripts work **without real LUNA16 data** via a synthetic
data fallback. To run a smoke test without any downloads:

```bash
# Runs the training loop for 2 epochs on synthetic Gaussian nodule patches
python training/train.py --config training/config.yaml \
    --data_dir SYNTHETIC --debug

# Runs pytest unit tests (all pass on synthetic data)
pytest tests/ -v
```

---

## References

- Setio, A.A.A., et al. (2017). *Validation, comparison, and combination of algorithms
  for automatic detection of pulmonary nodules in computed tomography images.*
  Medical Image Analysis, 42, 1–13. https://doi.org/10.1016/j.media.2017.06.015
- LUNA16 Grand Challenge: https://luna16.grand-challenge.org/
- LIDC-IDRI Collection (source): https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
