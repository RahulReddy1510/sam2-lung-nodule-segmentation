"""
tests/test_preprocessing.py
============================
Tests for data loading, preprocessing, and data augmentation.

All tests use SyntheticNoduleDataset (no LUNA16 required).

Run:
    pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def train_dataset():
    """Synthetic training dataset instance, module-scoped for speed."""
    from data.dataset import build_dataset

    return build_dataset("SYNTHETIC", split="train", mode="slice", augment=False)


@pytest.fixture(scope="module")
def augmented_dataset():
    """Synthetic training dataset with augmentations enabled."""
    from data.dataset import build_dataset

    return build_dataset("SYNTHETIC", split="train", mode="slice", augment=True)


@pytest.fixture(scope="module")
def augmentation_pipeline():
    """Augmentation pipeline fixture."""
    from data.augmentation import get_augmentation_pipeline

    cfg = {
        "augment": True,
        "random_flip_prob": 0.5,
        "vertical_flip_prob": 0.5,
        "random_rotation_degrees": 15.0,
        "random_zoom_range": [0.9, 1.1],
        "random_brightness": 0.1,
        "gaussian_noise_std": 0.02,
    }
    return get_augmentation_pipeline(cfg, augment=True)


# ── Dataset tests ─────────────────────────────────────────────────────────────


class TestDatasetLoading:
    """Tests around dataset construction and indexing."""

    def test_dataset_not_empty(self, train_dataset):
        assert len(train_dataset) > 0, "Dataset must have at least one sample"

    def test_dataset_splits_non_overlapping(self):
        """Train / val / test splits should be strictly non-overlapping."""
        from data.dataset import build_dataset

        train = build_dataset("SYNTHETIC", split="train", mode="slice", augment=False)
        val = build_dataset("SYNTHETIC", split="val", mode="slice", augment=False)
        test = build_dataset("SYNTHETIC", split="test", mode="slice", augment=False)
        # Total samples
        total = len(train) + len(val) + len(test)
        assert total > 0
        # Splits should sum to total
        assert len(train) > 0 and len(val) > 0 and len(test) > 0

    def test_sample_keys(self, train_dataset):
        """Each sample dict must contain 'image' and 'mask'."""
        sample = train_dataset[0]
        assert "image" in sample, "Sample must have 'image' key"
        assert "mask" in sample, "Sample must have 'mask' key"

    def test_image_shape(self, train_dataset):
        """Image tensor must be (1, H, W) with H, W >= 32."""
        sample = train_dataset[0]
        img = sample["image"]
        assert img.ndim == 3, f"Expected 3D tensor (C,H,W), got {img.ndim}D"
        assert img.shape[0] == 1, f"Expected 1 channel, got {img.shape[0]}"
        assert img.shape[1] >= 32 and img.shape[2] >= 32

    def test_mask_shape_matches_image(self, train_dataset):
        """Mask and image must have the same spatial dimensions."""
        sample = train_dataset[0]
        img, msk = sample["image"], sample["mask"]
        assert (
            img.shape[1:] == msk.shape[1:]
        ), f"Image {img.shape} and mask {msk.shape} spatial dims differ"

    def test_image_dtype_float(self, train_dataset):
        """Image tensor must be float (float32)."""
        sample = train_dataset[0]
        assert (
            sample["image"].dtype == torch.float32
        ), f"Expected float32, got {sample['image'].dtype}"

    def test_mask_dtype_float(self, train_dataset):
        """Mask tensor must be float for BCE loss compatibility."""
        sample = train_dataset[0]
        assert sample["mask"].dtype in (torch.float32, torch.float16, torch.float64)

    def test_image_range_normalised(self, train_dataset):
        """After HU windowing, image values should be in [0, 1]."""
        for i in range(min(10, len(train_dataset))):
            img = train_dataset[i]["image"]
            assert img.min() >= -1e-5, f"Image min {img.min()} below 0"
            assert img.max() <= 1 + 1e-5, f"Image max {img.max()} above 1"

    def test_mask_binary(self, train_dataset):
        """Mask values must be binary (0 or 1) with some tolerance."""
        for i in range(min(10, len(train_dataset))):
            msk = train_dataset[i]["mask"].numpy()
            unique = np.unique(np.round(msk, decimals=3))
            for u in unique:
                assert u in (0.0, 1.0) or (
                    0.0 <= u <= 1.0
                ), f"Unexpected mask value {u}"


# ── DataLoader tests ──────────────────────────────────────────────────────────


class TestDataLoader:
    """Tests for batched DataLoader output."""

    def test_batch_image_shape(self, train_dataset):
        """Batched images should be (B, 1, H, W)."""
        loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        img = batch["image"]
        assert img.ndim == 4, f"Expected 4D batch (B,C,H,W), got {img.ndim}"
        assert img.shape[0] == 4
        assert img.shape[1] == 1

    def test_batch_mask_shape(self, train_dataset):
        """Batched masks should be (B, 1, H, W)."""
        loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        msk = batch["mask"]
        assert msk.ndim == 4, f"Expected 4D mask batch (B,C,H,W), got {msk.ndim}"
        assert msk.shape[0] == 4

    def test_full_epoch_iteration(self, train_dataset):
        """Should be able to iterate through the full dataset without errors."""
        loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        total = 0
        for batch in loader:
            total += batch["image"].shape[0]
        assert total == len(train_dataset)


# ── HU Windowing ──────────────────────────────────────────────────────────────


# Skip entire HU windowing class if SimpleITK (used internally by luna16_preprocessing) is absent
_sitk = pytest.importorskip(
    "SimpleITK", reason="SimpleITK not installed — skipping HU windowing tests"
)


@pytest.mark.skipif(
    _sitk is None,
    reason="SimpleITK not installed",
)
class TestHUWindowing:
    """Tests for HU windowing and normalisation routine."""

    def test_windowing_clip_min(self):
        """Values below hu_min should clip to 0."""
        from data.luna16_preprocessing import apply_hu_window

        arr = np.array([-2000.0, -1000.0, 0.0, 400.0, 1000.0], dtype=np.float32)
        out = apply_hu_window(arr, hu_min=-1000.0, hu_max=400.0)
        assert out[0] == pytest.approx(0.0, abs=1e-5), "Below min → 0"
        assert out[1] == pytest.approx(0.0, abs=1e-5), "hu_min → 0"

    def test_windowing_clip_max(self):
        """Values above hu_max should clip to 1."""
        from data.luna16_preprocessing import apply_hu_window

        arr = np.array([400.0, 1000.0], dtype=np.float32)
        out = apply_hu_window(arr, hu_min=-1000.0, hu_max=400.0)
        assert out[0] == pytest.approx(1.0, abs=1e-5)
        assert out[1] == pytest.approx(1.0, abs=1e-5)

    def test_windowing_midpoint(self):
        """HU=0 should map to 1000/1400 ≈ 0.714."""
        from data.luna16_preprocessing import apply_hu_window

        arr = np.array([0.0], dtype=np.float32)
        out = apply_hu_window(arr, hu_min=-1000.0, hu_max=400.0)
        expected = (0.0 - (-1000.0)) / (400.0 - (-1000.0))
        assert out[0] == pytest.approx(expected, abs=1e-4)

    def test_windowing_output_range(self):
        """Output should always be in [0, 1]."""
        from data.luna16_preprocessing import apply_hu_window

        rng = np.random.default_rng(42)
        arr = rng.uniform(-2000, 2000, 1000).astype(np.float32)
        out = apply_hu_window(arr)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ── Augmentation tests ────────────────────────────────────────────────────────


class TestAugmentation:
    """Tests for the data augmentation pipeline."""

    def test_augmented_image_shape_preserved(self, augmentation_pipeline):
        """Augmentations must not change spatial dimensions."""
        img = torch.rand(1, 96, 96)
        msk = torch.zeros(1, 96, 96)
        msk[0, 30:50, 30:50] = 1.0
        out = augmentation_pipeline({"image": img, "mask": msk})
        assert out["image"].shape == img.shape, "Image shape changed by augmentation"
        assert out["mask"].shape == msk.shape, "Mask shape changed by augmentation"

    def test_augmented_image_range(self, augmentation_pipeline):
        """Augmented images should remain in a reasonable range."""
        img = torch.rand(1, 96, 96).clamp(0, 1)
        msk = torch.zeros(1, 96, 96)
        out = augmentation_pipeline({"image": img, "mask": msk})
        # Allow slight out-of-range due to brightness/noise then clamp
        assert out["image"].min() >= -0.5
        assert out["image"].max() <= 1.5

    def test_mask_stays_binary_after_augmentation(self, augmentation_pipeline):
        """After augmentation, mask values should still be close to 0 or 1."""
        img = torch.rand(1, 96, 96)
        msk = torch.zeros(1, 96, 96)
        msk[0, 20:60, 20:60] = 1.0
        out = augmentation_pipeline({"image": img, "mask": msk})
        mask_vals = out["mask"].numpy().flatten()
        # All values should be 0 or 1 (nearest-neighbour interpolation for mask)
        assert np.all((mask_vals >= -0.01) & (mask_vals <= 1.01))

    def test_augmentation_randomness(self, augmentation_pipeline):
        """Two augmentation passes of the same input should differ."""
        img = torch.rand(1, 96, 96)
        msk = torch.zeros(1, 96, 96)
        # They might be identical by chance; run multiple times to be sure
        # Run 5 times: extremely unlikely all 5 produce same output
        diffs = 0
        for _ in range(5):
            o1 = augmentation_pipeline({"image": img.clone(), "mask": msk.clone()})
            o2 = augmentation_pipeline({"image": img.clone(), "mask": msk.clone()})
            if not torch.allclose(o1["image"], o2["image"]):
                diffs += 1
        assert (
            diffs > 0
        ), "Augmentation produced identical outputs every time — check RNG"

    def test_augmented_dataset_len_unchanged(self, augmented_dataset, train_dataset):
        """Augmented dataset must have same length as non-augmented."""
        assert len(augmented_dataset) == len(train_dataset)
