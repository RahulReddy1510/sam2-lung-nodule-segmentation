"""
PyTorch Dataset classes for LUNA16 preprocessed patches.

Two dataset classes are provided:

- ``LUNA16SliceDataset`` — serves individual 2D axial slices from 3D .npy patches,
  with ``slice_idx`` tracking for the temporal consistency loss.
- ``LUNA16VolumeDataset`` — serves full 3D volumes for validation and Slicer inference.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LUNA16SliceDataset — 2D slices with temporal index tracking
# ---------------------------------------------------------------------------


class LUNA16SliceDataset(Dataset):
    """2D axial slice dataset for LUNA16 patch pairs.

    Each item is a single 2D axial (z-indexed) slice extracted from a 96³
    patch. The dataset exposes ``slice_idx`` so that the temporal consistency
    loss can identify truly adjacent slices (|Δidx| == 1) within a batch.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``*_image.npy`` and ``*_mask.npy`` files.
    augment : bool
        If True, apply the data augmentation pipeline. Default False.
    transform : callable, optional
        A callable that accepts a dict ``{"image": Tensor, "mask": Tensor}``
        and returns the same dict with transforms applied. Overrides ``augment``
        if provided.
    patch_size : tuple of int
        Expected spatial dimensions of each patch (Z, Y, X). Default (96, 96, 96).
    min_mask_voxels : int
        Patches where the mask has fewer than this many positive voxels are
        skipped at dataset construction. Default 10.

    Notes
    -----
    The dataset is built by scanning ``data_dir`` for ``*_image.npy`` files
    at construction time. For each patch of shape (Z, Y, X), all Z slices are
    registered as separate items. The ``slice_idx`` field tracks the z-position
    within the original patch (0 … Z-1), enabling the training loop to compute
    temporal consistency loss only between consecutive slices from the same patch.
    """

    def __init__(
        self,
        data_dir: str | Path,
        augment: bool = False,
        transform: Optional[Callable] = None,
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        min_mask_voxels: int = 10,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.transform = transform
        self.patch_size = patch_size
        self.min_mask_voxels = min_mask_voxels

        self._items: List[Dict] = []  # {patch_id, slice_idx, image_path, mask_path}
        self._build_index()

        if len(self._items) == 0:
            logger.warning(
                "LUNA16SliceDataset: no valid slices found in %s", self.data_dir
            )
        else:
            logger.info(
                "LUNA16SliceDataset: %d slices from %d patches in %s",
                len(self._items),
                len(self._patch_ids),
                self.data_dir,
            )

    def _build_index(self) -> None:
        """Scan data_dir and build the flat slice index."""
        image_paths = sorted(self.data_dir.glob("*_image.npy"))
        self._patch_ids: List[str] = []

        for img_path in image_paths:
            mask_path = img_path.parent / img_path.name.replace("_image.npy", "_mask.npy")
            if not mask_path.exists():
                logger.debug("Mask missing for %s — skipping", img_path.name)
                continue

            # Quick mask validation: load mask, check positive voxel count
            try:
                mask = np.load(mask_path, mmap_mode="r")
            except Exception as exc:
                logger.warning("Could not load %s: %s", mask_path, exc)
                continue

            if int(mask.sum()) < self.min_mask_voxels:
                logger.debug(
                    "Skipping %s — mask has only %d positive voxels",
                    img_path.name,
                    int(mask.sum()),
                )
                continue

            # Derive patch_id from filename
            patch_id = img_path.name.replace("_image.npy", "")
            self._patch_ids.append(patch_id)
            n_slices = mask.shape[0]  # Z dimension

            for z in range(n_slices):
                self._items.append(
                    {
                        "patch_id": patch_id,
                        "slice_idx": z,
                        "image_path": img_path,
                        "mask_path": mask_path,
                    }
                )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        """Return a single 2D slice dict.

        Returns
        -------
        dict with keys:
            ``image``     : Tensor of shape (1, H, W), float32, values in [0, 1].
            ``mask``      : Tensor of shape (1, H, W), float32, values in {0, 1}.
            ``slice_idx`` : Tensor (scalar int64), z-position within the patch.
            ``patch_id``  : str, unique patch identifier for traceability.
        """
        item = self._items[idx]
        patch_id: str = item["patch_id"]
        z: int = item["slice_idx"]

        # Memory-map the full patch and extract single slice
        try:
            image_vol: np.ndarray = np.load(item["image_path"], mmap_mode="r")
            mask_vol: np.ndarray = np.load(item["mask_path"], mmap_mode="r")
        except Exception as exc:
            # Return a zero slice on IO failure rather than crashing the dataloader
            logger.error("IO error loading %s slice %d: %s", patch_id, z, exc)
            h, w = self.patch_size[1], self.patch_size[2]
            return {
                "image": torch.zeros(1, h, w, dtype=torch.float32),
                "mask": torch.zeros(1, h, w, dtype=torch.float32),
                "slice_idx": torch.tensor(z, dtype=torch.int64),
                "patch_id": patch_id,
            }

        image_slice: np.ndarray = image_vol[z].astype(np.float32)  # (H, W)
        mask_slice: np.ndarray = mask_vol[z].astype(np.float32)  # (H, W)

        # Add channel dimension → (1, H, W)
        image_tensor = torch.from_numpy(image_slice).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_slice).unsqueeze(0)

        sample: Dict = {
            "image": image_tensor,
            "mask": mask_tensor,
            "slice_idx": torch.tensor(z, dtype=torch.int64),
            "patch_id": patch_id,
        }

        # Apply transforms
        if self.transform is not None:
            sample = self.transform(sample)
        elif self.augment:
            from data.augmentation import get_augmentation_pipeline

            aug = get_augmentation_pipeline({"augment": True})
            sample = aug(sample)

        return sample


# ---------------------------------------------------------------------------
# LUNA16VolumeDataset — full 3D volumes for inference
# ---------------------------------------------------------------------------


class LUNA16VolumeDataset(Dataset):
    """Full 3D volume dataset for inference and the 3D Slicer plugin.

    Returns complete 96³ volume patches (not individual slices). Used during
    evaluation and in the Slicer plugin's ``run()`` method for slice-by-slice
    MC Dropout inference.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``*_image.npy`` and ``*_mask.npy`` files.
    return_metadata : bool
        If True, includes ``spacing`` and ``origin`` keys (loaded from
        sidecar ``*_meta.npz`` if present). Default False.

    Notes
    -----
    The dataset does **not** apply augmentation — volumes are served as-is
    for inference. HU windowing and resampling are applied during preprocessing
    (see ``luna16_preprocessing.py``), so values already lie in [0, 1].
    """

    def __init__(
        self,
        data_dir: str | Path,
        return_metadata: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.return_metadata = return_metadata
        self._patch_ids: List[str] = []
        self._build_index()

        logger.info(
            "LUNA16VolumeDataset: %d volumes in %s",
            len(self._patch_ids),
            self.data_dir,
        )

    def _build_index(self) -> None:
        image_paths = sorted(self.data_dir.glob("*_image.npy"))
        for img_path in image_paths:
            mask_path = img_path.parent / img_path.name.replace("_image.npy", "_mask.npy")
            if mask_path.exists():
                patch_id = img_path.name.replace("_image.npy", "")
                self._patch_ids.append(patch_id)

    def __len__(self) -> int:
        return len(self._patch_ids)

    def __getitem__(self, idx: int) -> Dict:
        """Return a full 3D volume dict.

        Returns
        -------
        dict with keys:
            ``image``    : Tensor of shape (1, Z, H, W), float32.
            ``mask``     : Tensor of shape (1, Z, H, W), float32.
            ``patch_id`` : str, unique patch identifier.
        """
        patch_id = self._patch_ids[idx]
        img_path = self.data_dir / f"{patch_id}_image.npy"
        msk_path = self.data_dir / f"{patch_id}_mask.npy"

        image_vol = np.load(img_path).astype(np.float32)  # (Z, H, W)
        mask_vol = np.load(msk_path).astype(np.float32)  # (Z, H, W)

        # Add channel dimension: (1, Z, H, W)
        image_tensor = torch.from_numpy(image_vol).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_vol).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "patch_id": patch_id,
        }


# ---------------------------------------------------------------------------
# Synthetic dataset for smoke-testing without real LUNA16 data
# ---------------------------------------------------------------------------


class SyntheticNoduleDataset(Dataset):
    """Synthetic Gaussian nodule dataset for CI/CD and development.

    Generates random 3D patches with synthetic spherical nodule masks at
    runtime. No real CT data needed. Used by pytest and the --data_dir SYNTHETIC
    flag in train.py.

    Parameters
    ----------
    n_patches : int
        Number of synthetic patches to generate. Default 64.
    patch_size : tuple of int
        Patch dimensions (Z, Y, X). Default (96, 96, 96).
    mode : str
        ``"slice"`` returns individual 2D slices (like LUNA16SliceDataset).
        ``"volume"`` returns full 3D volumes (like LUNA16VolumeDataset).
    seed : int
        RNG seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        n_patches: int = 64,
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        mode: str = "slice",
        seed: int = 42,
    ) -> None:
        assert mode in ("slice", "volume"), f"mode must be 'slice' or 'volume', got {mode}"
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.mode = mode
        self._rng = np.random.RandomState(seed)

        # Pre-generate patches for consistency across epochs
        self._patches = [self._make_patch(i) for i in range(n_patches)]

        if mode == "slice":
            # Flatten into individual slices
            self._items: List[Tuple[np.ndarray, np.ndarray, str, int]] = []
            for pid, (img, msk) in enumerate(self._patches):
                for z in range(patch_size[0]):
                    self._items.append((img[z], msk[z], f"synthetic_{pid:04d}", z))
        else:
            self._items = self._patches  # type: ignore[assignment]

        logger.info(
            "SyntheticNoduleDataset: mode=%s | %d items", mode, len(self._items)
        )

    def _make_patch(
        self, seed_offset: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single synthetic CT patch with a Gaussian-blurred nodule."""
        rng = np.random.RandomState(self._rng.randint(0, 2**31) + seed_offset)
        Z, Y, X = self.patch_size
        # Background: lung-like texture (near 0, Gaussian noise)
        image = rng.normal(loc=0.1, scale=0.05, size=(Z, Y, X)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        # Nodule: bright sphere in center with Gaussian falloff
        radius_vox = rng.randint(6, 18)
        cz, cy, cx = Z // 2, Y // 2, X // 2
        zz, yy, xx = np.ogrid[:Z, :Y, :X]
        dist_sq = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        nodule = np.exp(-dist_sq / (2 * (radius_vox / 2) ** 2)).astype(np.float32)
        image = np.clip(image + 0.6 * nodule, 0.0, 1.0)

        mask = (dist_sq <= radius_vox**2).astype(np.uint8)
        return image, mask

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        if self.mode == "slice":
            img_slice, msk_slice, patch_id, z = self._items[idx]
            return {
                "image": torch.from_numpy(img_slice).unsqueeze(0),
                "mask": torch.from_numpy(msk_slice.astype(np.float32)).unsqueeze(0),
                "slice_idx": torch.tensor(z, dtype=torch.int64),
                "patch_id": patch_id,
            }
        else:
            img_vol, msk_vol = self._items[idx]
            return {
                "image": torch.from_numpy(img_vol).unsqueeze(0),
                "mask": torch.from_numpy(msk_vol.astype(np.float32)).unsqueeze(0),
                "patch_id": f"synthetic_{idx:04d}",
            }


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def build_dataset(
    data_dir: str,
    split: str = "train",
    mode: str = "slice",
    augment: bool = False,
    **kwargs,
) -> Dataset:
    """Build the appropriate dataset given the data_dir and split.

    Parameters
    ----------
    data_dir : str
        Path to preprocessed data root, OR ``"SYNTHETIC"`` for a synthetic dataset.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    mode : str
        ``"slice"`` for LUNA16SliceDataset, ``"volume"`` for LUNA16VolumeDataset.
    augment : bool
        Passed to LUNA16SliceDataset. Ignored for volumes.
    **kwargs
        Additional kwargs forwarded to the dataset constructor.

    Returns
    -------
    Dataset
        The constructed dataset.
    """
    if data_dir == "SYNTHETIC":
        n_patches = kwargs.pop("n_patches", 64)
        return SyntheticNoduleDataset(n_patches=n_patches, mode=mode, **kwargs)

    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}. "
            "Run luna16_preprocessing.py first, or use data_dir='SYNTHETIC'."
        )

    if mode == "slice":
        return LUNA16SliceDataset(split_dir, augment=augment, **kwargs)
    elif mode == "volume":
        return LUNA16VolumeDataset(split_dir, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'slice' or 'volume'.")


if __name__ == "__main__":
    # Smoke test — runs without any real LUNA16 data
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("LUNA16 Dataset — smoke test with synthetic data")
    print("=" * 60)

    # Test slice dataset
    ds_slice = SyntheticNoduleDataset(n_patches=4, mode="slice", seed=42)
    sample = ds_slice[0]
    print(f"\n[SliceDataset] len={len(ds_slice)}")
    print(f"  image:     {sample['image'].shape}  dtype={sample['image'].dtype}")
    print(f"  mask:      {sample['mask'].shape}   dtype={sample['mask'].dtype}")
    print(f"  slice_idx: {sample['slice_idx'].item()}")
    print(f"  patch_id:  {sample['patch_id']}")

    # Test volume dataset
    ds_vol = SyntheticNoduleDataset(n_patches=4, mode="volume", seed=42)
    vol_sample = ds_vol[0]
    print(f"\n[VolumeDataset] len={len(ds_vol)}")
    print(f"  image:    {vol_sample['image'].shape}")
    print(f"  mask:     {vol_sample['mask'].shape}")
    print(f"  patch_id: {vol_sample['patch_id']}")

    print("\nAll checks passed.")
