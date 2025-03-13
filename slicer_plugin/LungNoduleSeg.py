"""
LungNoduleSeg — 3D Slicer Extension
=====================================
SAM2-based lung nodule segmentation with Monte Carlo Dropout uncertainty
estimation, packaged as a standard Slicer scripted module.

Module hierarchy (Slicer convention)
--------------------------------------
  LungNoduleSeg          — module metadata singleton
  LungNoduleSegWidget    — Qt/CTK UI, wires signals → logic
  LungNoduleSegLogic     — headless inference; no Slicer GUI dependencies
  LungNoduleSegTest      — slicer.util.VTKObservationMixin self-test

Supported Slicer version: ≥ 5.4.0
Python environment: Slicer's bundled Python 3.9+

Usage (from Python Interactor)::

    logic = LungNoduleSegLogic()
    logic.load_model("/path/to/best_model.pt", device="cuda")
    seg, unc = logic.run(input_volume_node, n_mc=25)
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ── Slicer imports ────────────────────────────────────────────────────────
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------


class LungNoduleSeg(ScriptedLoadableModule):
    """Slicer module descriptor.

    Sets the module's display name, category, description, and contributors
    shown in the Slicer Module panel.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Lung Nodule Seg (SAM2)"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Rahul Reddy (SAM2 Lung Nodule Team)"]
        self.parent.helpText = (
            "Automated lung nodule segmentation using a fine-tuned SAM2 model "
            "with Monte Carlo Dropout uncertainty estimation on CT volumes. "
            "Load a CT, select a trained checkpoint, and click Run Segmentation."
        )
        self.parent.acknowledgementText = (
            "Based on Meta AI's Segment Anything Model 2 (SAM2). "
            "Trained on the LUNA16 dataset. "
            "Developed as part of a 6-month research project (Jan–Jun 2025)."
        )


# ---------------------------------------------------------------------------
# Widget (GUI)
# ---------------------------------------------------------------------------


class LungNoduleSegWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Qt-based panel widget for the Lung Nodule Seg module.

    Loads the designer .ui file, connects signals from UI widgets to logic
    methods, and updates the display when inference completes.
    """

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic: Optional[LungNoduleSegLogic] = None
        self._param_node = None
        self._updating_gui = False

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup(self):
        """Initialise the module widget: load .ui, create logic, connect signals."""
        super().setup()

        # Load .ui file from Resources/UI/
        ui_path = Path(__file__).parent / "Resources" / "UI" / "LungNoduleSeg.ui"
        if not ui_path.exists():
            slicer.util.errorDisplay(
                f"UI file not found:\n{ui_path}\n\nPlugin may be incomplete.",
                windowTitle="LungNoduleSeg Error",
            )
            return

        self.ui = slicer.util.loadUI(str(ui_path))
        self.layout.addWidget(self.ui)

        # Grab widget references (must match objectName in .ui)
        self.inputVolumeSelector = self.ui.findChild(object, "inputVolumeSelector")
        self.outputSegSelector = self.ui.findChild(object, "outputSegSelector")
        self.uncertaintyVolSelector = self.ui.findChild(
            object, "uncertaintyVolSelector"
        )
        self.checkpointPathEdit = self.ui.findChild(object, "checkpointPathEdit")
        self.mcSamplesSpinBox = self.ui.findChild(object, "mcSamplesSpinBox")
        self.thresholdSpinBox = self.ui.findChild(object, "thresholdSpinBox")
        self.useGpuCheckBox = self.ui.findChild(object, "useGpuCheckBox")
        self.runSegmentationButton = self.ui.findChild(object, "runSegmentationButton")
        self.progressBar = self.ui.findChild(object, "progressBar")
        self.resultsTextEdit = self.ui.findChild(object, "resultsTextEdit")

        # Create logic
        self.logic = LungNoduleSegLogic()

        # Connect signals
        self.runSegmentationButton.clicked.connect(self.onRunSegmentation)
        self.inputVolumeSelector.currentNodeChanged.connect(self._on_input_changed)

        # Initialise state
        self._on_input_changed()
        self.progressBar.setVisible(False)

        logger.info("LungNoduleSegWidget setup complete")

    def cleanup(self):
        """Called when the module widget is destroyed."""
        self.removeObservers()

    # ── Signal handlers ───────────────────────────────────────────────────

    def _on_input_changed(self):
        """Enable Run button only when an input volume is selected."""
        has_input = self.inputVolumeSelector.currentNode() is not None
        self.runSegmentationButton.setEnabled(has_input)

    def onRunSegmentation(self):
        """Main callback: validate inputs, run inference, update MRML scene."""
        input_node = self.inputVolumeSelector.currentNode()
        if input_node is None:
            slicer.util.errorDisplay(
                "Please select an input CT volume.",
                windowTitle="No Input Selected",
            )
            return

        checkpoint_path = (
            self.checkpointPathEdit.currentPath
            if hasattr(self.checkpointPathEdit, "currentPath")
            else ""
        )
        n_mc = self.mcSamplesSpinBox.value
        threshold = self.thresholdSpinBox.value
        use_gpu = self.useGpuCheckBox.isChecked()
        device = "cuda" if (use_gpu and self._cuda_available()) else "cpu"

        # Prepare output nodes
        out_seg_node = self.outputSegSelector.currentNode()
        if out_seg_node is None:
            out_seg_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode",
                f"{input_node.GetName()}_LungNoduleSeg",
            )
            self.outputSegSelector.setCurrentNode(out_seg_node)

        unc_vol_node = self.uncertaintyVolSelector.currentNode()
        if unc_vol_node is None:
            unc_vol_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode",
                f"{input_node.GetName()}_Uncertainty",
            )
            self.uncertaintyVolSelector.setCurrentNode(unc_vol_node)

        # Start inference
        self.runSegmentationButton.setEnabled(False)
        self.progressBar.setVisible(True)
        self.progressBar.setValue(0)
        self.resultsTextEdit.setPlainText("Running segmentation…")
        slicer.app.processEvents()

        try:
            # Load model (lazy — cached after first call)
            self.logic.load_model(
                checkpoint_path=checkpoint_path or None,
                device=device,
            )

            # Run inference with progress reporting
            def progress_cb(pct: float):
                self.progressBar.setValue(int(pct * 100))
                slicer.app.processEvents()

            t0 = time.time()
            seg_array, unc_array = self.logic.run(
                input_volume_node=input_node,
                n_mc=n_mc,
                threshold=threshold,
                progress_callback=progress_cb,
            )
            elapsed = time.time() - t0

            # Write segmentation → output labelmap
            self._array_to_labelmap(seg_array, input_node, out_seg_node)

            # Write uncertainty → scalar volume
            self._array_to_scalar_volume(unc_array, input_node, unc_vol_node)

            # Apply heat colormap to uncertainty
            self._apply_uncertainty_display(unc_vol_node)

            # Display results
            self._show_results(seg_array, unc_array, input_node, elapsed)

            logger.info("Segmentation complete in %.1f s", elapsed)

        except Exception as exc:
            msg = f"Segmentation failed:\n{exc}\n\n{traceback.format_exc()}"
            logger.error(msg)
            slicer.util.errorDisplay(msg, windowTitle="Segmentation Error")
            self.resultsTextEdit.setPlainText(f"ERROR:\n{exc}")

        finally:
            self.runSegmentationButton.setEnabled(True)
            self.progressBar.setVisible(False)
            self.progressBar.setValue(0)

    # ── MRML helpers ──────────────────────────────────────────────────────

    def _array_to_labelmap(
        self,
        seg_array: np.ndarray,
        reference_node,
        out_node,
    ) -> None:
        """Write numpy binary array into a vtkMRMLLabelMapVolumeNode.

        Parameters
        ----------
        seg_array : np.ndarray
            Binary segmentation, shape (D, H, W), dtype uint8.
        reference_node : vtkMRMLScalarVolumeNode
            Source CT node (provides geometry / IJK-to-RAS matrix).
        out_node : vtkMRMLLabelMapVolumeNode
            Target MRML node.
        """
        import vtk.util.numpy_support as nps

        seg_array = seg_array.astype(np.uint8)
        flat = seg_array.flatten(order="C")
        vtk_array = nps.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_array.SetName("LungNoduleSeg")

        image_data = vtk.vtkImageData()
        image_data.SetDimensions(
            seg_array.shape[2], seg_array.shape[1], seg_array.shape[0]
        )
        image_data.GetPointData().SetScalars(vtk_array)

        out_node.SetAndObserveImageData(image_data)
        # Copy geometry from reference
        ref_to_world = vtk.vtkMatrix4x4()
        reference_node.GetIJKToRASMatrix(ref_to_world)
        out_node.SetIJKToRASMatrix(ref_to_world)
        out_node.Modified()

    def _array_to_scalar_volume(
        self,
        array: np.ndarray,
        reference_node,
        out_node,
    ) -> None:
        """Write numpy float array into a vtkMRMLScalarVolumeNode.

        Parameters
        ----------
        array : np.ndarray
            Float array (e.g., uncertainty variance), shape (D, H, W).
        reference_node : vtkMRMLScalarVolumeNode
            Reference geometry node.
        out_node : vtkMRMLScalarVolumeNode
            Target MRML node.
        """
        import vtk.util.numpy_support as nps

        arr = array.astype(np.float32)
        flat = arr.flatten(order="C")
        vtk_array = nps.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_array.SetName("Uncertainty")

        image_data = vtk.vtkImageData()
        image_data.SetDimensions(arr.shape[2], arr.shape[1], arr.shape[0])
        image_data.GetPointData().SetScalars(vtk_array)

        out_node.SetAndObserveImageData(image_data)
        ref_to_world = vtk.vtkMatrix4x4()
        reference_node.GetIJKToRASMatrix(ref_to_world)
        out_node.SetIJKToRASMatrix(ref_to_world)
        out_node.Modified()

    def _apply_uncertainty_display(self, unc_node) -> None:
        """Apply a Heat1 colormap to the uncertainty volume for display."""
        try:
            display_node = unc_node.GetDisplayNode()
            if display_node is None:
                unc_node.CreateDefaultDisplayNodes()
                display_node = unc_node.GetDisplayNode()
            display_node.SetAndObserveColorNodeID(slicer.util.getNode("Heat1").GetID())
            display_node.AutoWindowLevelOn()
        except Exception as exc:
            logger.debug("Could not apply uncertainty colormap: %s", exc)

    def _show_results(
        self,
        seg_array: np.ndarray,
        unc_array: np.ndarray,
        input_node,
        elapsed: float,
    ) -> None:
        """Populate the results text box with summary statistics."""
        # Voxel spacing for volume estimation
        spacing = [1.0, 1.0, 1.0]
        try:
            spacing = list(input_node.GetSpacing())  # (sx, sy, sz) in mm
        except Exception:
            pass

        voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
        n_voxels = int(seg_array.sum())
        volume_mm3 = n_voxels * voxel_vol_mm3
        volume_cm3 = volume_mm3 / 1000.0

        mean_unc = float(unc_array.mean())
        max_unc = float(unc_array.max())
        confidence_pct = max(0.0, (1.0 - mean_unc)) * 100.0

        text = (
            "══════════════════════════════════\n"
            "  SAM2 Lung Nodule Segmentation\n"
            "══════════════════════════════════\n"
            f"  Status         : ✓ Complete\n"
            f"  Time           : {elapsed:.1f} s\n"
            f"──────────────────────────────────\n"
            f"  Nodule voxels  : {n_voxels:,}\n"
            f"  Volume         : {volume_mm3:.1f} mm³  ({volume_cm3:.3f} cm³)\n"
            f"──────────────────────────────────\n"
            f"  Mean uncertainty: {mean_unc:.5f}\n"
            f"  Max uncertainty : {max_unc:.5f}\n"
            f"  Confidence      : {confidence_pct:.1f}%\n"
            "══════════════════════════════════\n"
        )
        self.resultsTextEdit.setPlainText(text)

    @staticmethod
    def _cuda_available() -> bool:
        """Return True if PyTorch finds a CUDA device."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# Logic (headless inference engine)
# ---------------------------------------------------------------------------


class LungNoduleSegLogic(ScriptedLoadableModuleLogic):
    """Headless inference engine — no GUI dependencies.

    Can be used from the Python Interactor or other modules::

        logic = LungNoduleSegLogic()
        logic.load_model("best_model.pt", device="cuda")
        seg, unc = logic.run(volume_node, n_mc=25)

    The model is lazily loaded and cached; repeated calls to ``run()``
    with the same ``load_model()`` call reuse the cached weights.

    Parameters
    ----------
    parent : optional
        Slicer parent object (passed through to ScriptedLoadableModuleLogic).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None
        self._device = "cpu"
        self._checkpoint_path: Optional[str] = None

    # ── Model management ──────────────────────────────────────────────────

    def load_model(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        """Load (or reload) the SAM2LungSegmentor from a checkpoint.

        If called again with the same path and device, this is a no-op.

        Parameters
        ----------
        checkpoint_path : str or None
            Path to the ``best_model.pt`` checkpoint. If None, a randomly
            initialised model is used (for smoke-testing only).
        device : str
            ``"cuda"`` or ``"cpu"``. Default ``"cpu"``.
        """
        import torch

        # Skip reload if nothing changed
        if (
            self._model is not None
            and self._checkpoint_path == checkpoint_path
            and self._device == device
        ):
            return

        self._device = device
        self._checkpoint_path = checkpoint_path

        # Add project root to path so we can import models/
        project_root = str(Path(__file__).parents[1])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from models.registry import get_model

        dev = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=dev)
            cfg = ckpt.get("config", {})
            mcfg = cfg.get("model", {})
            model = get_model(
                mcfg.get("name", "sam2_lung_seg"),
                embed_dim=int(mcfg.get("embed_dim", 256)),
                num_heads=int(mcfg.get("num_heads", 8)),
                attn_dropout=float(mcfg.get("attn_dropout", 0.1)),
                proj_dropout=float(mcfg.get("proj_dropout", 0.1)),
                encoder_frozen=False,
            ).to(dev)
            model.load_state_dict(ckpt["model_state_dict"])
            epoch = ckpt.get("epoch", "?")
            val_dice = ckpt.get("metrics", {}).get("val_dice", "?")
            logger.info("Loaded checkpoint: epoch=%s val_dice=%s", epoch, val_dice)
        else:
            if checkpoint_path:
                logger.warning(
                    "Checkpoint not found at %s — using random weights", checkpoint_path
                )
            model = get_model(
                "sam2_lung_seg",
                encoder_frozen=False,
            ).to(dev)
            logger.info("Initialised model with random weights (no checkpoint)")

        model.eval()
        self._model = model
        self._device = str(dev)
        logger.info("Model ready on device=%s", self._device)

    # ── Inference ─────────────────────────────────────────────────────────

    def run(
        self,
        input_volume_node,
        n_mc: int = 25,
        threshold: float = 0.5,
        progress_callback=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run slice-by-slice MC Dropout segmentation on a CT volume.

        Parameters
        ----------
        input_volume_node : vtkMRMLScalarVolumeNode
            Slicer MRML node containing the CT data.
        n_mc : int
            Number of MC Dropout forward passes per slice. Default 25.
        threshold : float
            Binary segmentation threshold. Default 0.5.
        progress_callback : callable or None
            Called with a float ∈ [0, 1] indicating progress.

        Returns
        -------
        seg_array : np.ndarray
            Binary segmentation volume, shape (D, H, W), dtype uint8.
        unc_array : np.ndarray
            MC variance uncertainty map, shape (D, H, W), dtype float32.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() before run().")

        import torch

        from models.mc_dropout import mc_predict

        # Convert MRML volume → numpy array → torch tensor (D, H, W)
        volume_np = self._volume_to_numpy(input_volume_node)  # (D, H, W) float32
        volume_np = self._apply_hu_window(volume_np)  # normalise → [0, 1]

        D, H, W = volume_np.shape
        dev = torch.device(self._device)

        seg_volume = np.zeros((D, H, W), dtype=np.float32)
        unc_volume = np.zeros((D, H, W), dtype=np.float32)

        for z in range(D):
            # Shape: (1, 1, H, W)
            slice_np = volume_np[z : z + 1, :, :]
            slice_t = torch.from_numpy(slice_np).unsqueeze(0).unsqueeze(0).to(dev)

            with torch.no_grad():
                mean_pred, variance = mc_predict(
                    self._model,
                    slice_t,
                    n_samples=n_mc,
                    mc_batch_size=min(n_mc, 5),
                    sigmoid=True,
                )

            seg_volume[z] = (mean_pred.squeeze().cpu().numpy() >= threshold).astype(
                np.float32
            )
            unc_volume[z] = variance.squeeze().cpu().numpy()

            if progress_callback is not None:
                progress_callback((z + 1) / D)

        return seg_volume.astype(np.uint8), unc_volume.astype(np.float32)

    # ── MRML / numpy helpers ──────────────────────────────────────────────

    @staticmethod
    def _volume_to_numpy(volume_node) -> np.ndarray:
        """Extract voxel data from a vtkMRMLScalarVolumeNode as float32 numpy array.

        Returns array shaped (D, H, W) in patient RAS order
        (matching the axial-first convention used in training).

        Parameters
        ----------
        volume_node : vtkMRMLScalarVolumeNode
            Input MRML node.

        Returns
        -------
        np.ndarray
            Shape (D, H, W), dtype float32.
        """
        import vtk.util.numpy_support as nps

        image_data = volume_node.GetImageData()
        dims = image_data.GetDimensions()  # (W, H, D) in VTK order
        scalars = image_data.GetPointData().GetScalars()
        np_array = nps.vtk_to_numpy(scalars).astype(np.float32)
        # VTK stores in Fortran-order (X=cols, Y=rows, Z=slices)
        np_array = np_array.reshape(dims[2], dims[1], dims[0])  # → (D, H, W)
        return np_array

    @staticmethod
    def _apply_hu_window(
        volume: np.ndarray,
        hu_min: float = -1000.0,
        hu_max: float = 400.0,
    ) -> np.ndarray:
        """Apply HU windowing ([-1000, 400]) and normalise to [0, 1].

        Parameters
        ----------
        volume : np.ndarray
            Raw CT Hounsfield unit array.
        hu_min : float
            Lower window bound. Default -1000.
        hu_max : float
            Upper window bound. Default 400.

        Returns
        -------
        np.ndarray
            Normalised float32 array with values in [0, 1].
        """
        volume = np.clip(volume, hu_min, hu_max)
        volume = (volume - hu_min) / (hu_max - hu_min)
        return volume.astype(np.float32)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


class LungNoduleSegTest(ScriptedLoadableModuleTest):
    """Self-test class for 3D Slicer's built-in test framework.

    Run via: Edit → Application Settings → Testing → Run All Tests
    or via slicer.util.mainWindow()…

    The test uses a synthetic random CT volume so no real LUNA16 data
    is required.
    """

    def runTest(self):
        """Execute all test cases."""
        self.delayDisplay("Starting LungNoduleSeg self-test…")
        slicer.mrmlScene.Clear(0)

        self.test_logic_synthetic()
        self.test_hu_window()

        self.delayDisplay("LungNoduleSeg self-test PASSED ✓")

    def test_logic_synthetic(self):
        """Test the inference logic with a synthetic 64×64×32 volume."""
        import numpy as np
        import vtk
        import vtk.util.numpy_support as nps

        self.delayDisplay("Creating synthetic CT volume…")

        # Build a synthetic 64×64×32 random CT
        W, H, D = 64, 64, 32
        synthetic = np.random.randn(D, H, W).astype(np.float32) * 200 - 500

        vtk_array = nps.numpy_to_vtk(
            synthetic.flatten(), deep=True, array_type=vtk.VTK_FLOAT
        )
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(W, H, D)
        image_data.SetSpacing(1.0, 1.0, 1.0)
        image_data.GetPointData().SetScalars(vtk_array)

        volume_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", "SyntheticCT"
        )
        volume_node.SetAndObserveImageData(image_data)
        volume_node.SetSpacing(1.0, 1.0, 1.0)

        # Instantiate logic (no checkpoint → random weights)
        logic = LungNoduleSegLogic()
        logic.load_model(checkpoint_path=None, device="cpu")

        # Run inference with n_mc=2 for speed
        seg, unc = logic.run(volume_node, n_mc=2, threshold=0.5)

        assert seg.shape == (D, H, W), f"Expected ({D},{H},{W}), got {seg.shape}"
        assert unc.shape == (D, H, W), "Uncertainty shape mismatch"
        assert seg.dtype == np.uint8, "Segmentation should be uint8"
        assert unc.dtype == np.float32, "Uncertainty should be float32"
        assert 0 <= seg.max() <= 1, "Segmentation values should be binary"
        assert unc.min() >= 0.0, "Uncertainty should be non-negative"

        self.delayDisplay("test_logic_synthetic PASSED ✓")

    def test_hu_window(self):
        """Unit test HU windowing logic."""
        import numpy as np

        self.delayDisplay("Testing HU windowing…")

        logic = LungNoduleSegLogic()
        vol = np.array([-2000.0, -1000.0, 0.0, 400.0, 1000.0], dtype=np.float32)
        windowed = logic._apply_hu_window(vol, hu_min=-1000.0, hu_max=400.0)

        assert windowed[0] == 0.0, "Below-min should clip to 0"
        assert windowed[1] == 0.0, "hu_min should map to 0"
        assert abs(windowed[3] - 1.0) < 1e-5, "hu_max should map to 1"
        assert windowed[4] == 1.0, "Above-max should clip to 1"
        assert 0.0 < windowed[2] < 1.0, "HU=0 should be between 0 and 1"

        self.delayDisplay("test_hu_window PASSED ✓")
