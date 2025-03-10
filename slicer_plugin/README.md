# 3D Slicer Plugin — SAM2 Lung Nodule Segmentation

A 3D Slicer extension that brings the SAM2-based lung nodule segmenter
directly into the radiology workstation. Radiologists load a CT volume,
click **Run Segmentation**, and get back:

- A segmentation labelmap with the detected nodule
- A per-voxel **uncertainty heatmap** (MC Dropout variance) displayed as a scalar volume
- Volume and confidence statistics in the Results panel

---

## Requirements

| Requirement | Version |
|---|---|
| 3D Slicer | ≥ 5.4.0 |
| Python (bundled with Slicer) | 3.9+ |
| PyTorch | ≥ 2.1 (see install below) |
| SAM2 Lung Seg package | this repository |

> **Note:** Slicer ships its own Python interpreter. You must install
> PyTorch into *Slicer's* Python, not your system Python.

---

## Installation

### 1. Install PyTorch into Slicer's Python

Open **Slicer → Python Interactor** and run:

```python
import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "torch==2.1.2", "torchvision==0.16.2",
    "--index-url", "https://download.pytorch.org/whl/cu118"
])
```

### 2. Install the sam2-lung-nodule-segmentation package

```python
import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-e",
    "/path/to/sam2-lung-nodule-segmentation"
])
```

### 3. Load the Extension into Slicer

1. Go to **Edit → Application Settings → Modules → Additional module paths**
2. Add the `slicer_plugin/` directory of this repository
3. Restart Slicer
4. The module appears under **Segmentation → Lung Nodule Seg (SAM2)**

---

## Usage

### Loading a CT Volume

1. Load your CT DICOM via **File → Add DICOM Data** (or drag-and-drop)
2. In the **Lung Nodule Seg (SAM2)** panel:
   - Select the CT volume from the **Input Volume** dropdown
   - Set the **Model Checkpoint** path (`.pt` file from `training/`)
   - Adjust **MC Dropout Samples** (default 25; higher = more accurate uncertainty)
   - Click **Run Segmentation**

### Interpreting Results

| Output | Description |
|---|---|
| **Segmentation Labelmap** | Binary nodule mask (label=1) overlaid on CT slices |
| **Uncertainty Volume** | Per-voxel MC variance — warmer colour = higher uncertainty |
| **Volume (mm³)** | Estimated nodule volume from the binary mask |
| **Mean Confidence** | 1 − mean uncertainty, as a % |

### Uncertainty Heatmap

The uncertainty volume is automatically assigned a heat colormap in 3D Slicer:

- **Blue/cool** — model is confident the region is background or nodule  
- **Red/warm** — model is uncertain; consider manual review

Uncertainty > 0.05 at the nodule boundary is highlighted automatically via
the threshold set in the **Display** panel.

---

## Module Architecture

```
LungNoduleSeg.py
├── LungNoduleSegWidget          # Qt GUI / Slicer panel widget
│   ├── setup()                  # load .ui, connect signals
│   ├── onRunSegmentation()      # main callback: MRML → model → MRML
│   └── _show_results()          # write stats to results text box
├── LungNoduleSegLogic           # headless inference engine
│   ├── load_model()             # lazy-load SAM2LungSegmentor from checkpoint
│   ├── run()                    # volume tensor → mean_seg + uncertainty
│   └── _volume_to_tensor()      # MRML vtkMRMLScalarVolumeNode → Tensor
└── LungNoduleSegTest            # self-test with synthetic data
    └── runTest()
```

---

## Checkpoint File

The plugin expects a `.pt` checkpoint created by `training/train.py`.
After training, your best checkpoint is at:

```
runs/<experiment_name>/<run_id>/checkpoints/best_model.pt
```

Set this path in the **Model Checkpoint** field of the Slicer panel.

---

## Known Limitations

- Currently supports **axial slice-by-slice** inference only.
  3D SAM2 support is planned for v1.1.0.
- Very large CT volumes (> 512 slices) may be slow; use GPU for <30 s/volume.
- Input CT must be **already HU-windowed** or the plugin will apply
  the default window `[-1000, 400]` automatically.

---

## Citation

If you use this Slicer plugin in clinical research, please cite:

```bibtex
@misc{sam2_lung_nodule_2025,
  title     = {SAM2 Lung Nodule Segmentation: Uncertainty-Aware CT Analysis},
  author    = {Rahul Reddy},
  year      = {2025},
  url       = {https://github.com/rahul-reddy/sam2-lung-nodule-segmentation}
}
```
