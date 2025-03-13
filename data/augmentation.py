"""
Data augmentation pipeline for LUNA16 CT slice patches.

All augmentations operate on a dict of ``{"image": Tensor, "mask": Tensor}``
where both tensors have shape (1, H, W). The same spatial transformation is
applied to both image and mask to preserve annotation alignment.

Factory function ``get_augmentation_pipeline(config)`` is the primary API.

Example::

    from data.augmentation import get_augmentation_pipeline

    aug = get_augmentation_pipeline({
        "random_flip_prob": 0.5,
        "random_rotation_degrees": 15,
        "random_zoom_range": [0.85, 1.15],
        "gaussian_noise_std": 0.02,
        "random_brightness": 0.1,
    })

    sample = {"image": torch.randn(1, 96, 96), "mask": torch.zeros(1, 96, 96)}
    augmented = aug(sample)
"""

from __future__ import annotations

import logging
import math
import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# Type alias for a sample dict
Sample = Dict[str, Union[Tensor, str, int]]


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------


class RandomHorizontalFlip:
    """Randomly flip image and mask horizontally (along the W axis).

    Parameters
    ----------
    p : float
        Probability of applying the flip. Default 0.5.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            sample = dict(sample)
            sample["image"] = torch.flip(sample["image"], dims=[-1])
            sample["mask"] = torch.flip(sample["mask"], dims=[-1])
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomVerticalFlip:
    """Randomly flip image and mask vertically (along the H axis).

    Parameters
    ----------
    p : float
        Probability of applying the flip. Default 0.3.
    """

    def __init__(self, p: float = 0.3) -> None:
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            sample = dict(sample)
            sample["image"] = torch.flip(sample["image"], dims=[-2])
            sample["mask"] = torch.flip(sample["mask"], dims=[-2])
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class GaussianNoise:
    """Add zero-mean Gaussian noise to the image only (not the mask).

    The noise standard deviation is sampled uniformly from ``std_range``
    each time the transform is called, providing variability in noise level.

    Parameters
    ----------
    std_range : tuple of float
        Range ``(min_std, max_std)`` for the noise standard deviation.
        Default (0.0, 0.02).
    """

    def __init__(self, std_range: Tuple[float, float] = (0.0, 0.02)) -> None:
        self.std_range = std_range

    def __call__(self, sample: Sample) -> Sample:
        std = random.uniform(*self.std_range)
        if std > 0:
            sample = dict(sample)
            noise = torch.randn_like(sample["image"]) * std
            sample["image"] = torch.clamp(sample["image"] + noise, 0.0, 1.0)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std_range={self.std_range})"


class RandomBrightness:
    """Add a random brightness offset to the image only.

    A scalar offset is sampled uniformly from ``[-brightness, +brightness]``
    and added to all pixels. The result is clamped to [0, 1].

    Parameters
    ----------
    brightness : float
        Maximum absolute brightness shift. Default 0.1.
    """

    def __init__(self, brightness: float = 0.1) -> None:
        self.brightness = brightness

    def __call__(self, sample: Sample) -> Sample:
        if self.brightness > 0:
            sample = dict(sample)
            delta = random.uniform(-self.brightness, self.brightness)
            sample["image"] = torch.clamp(sample["image"] + delta, 0.0, 1.0)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(brightness={self.brightness})"


class RandomZoom:
    """Apply random isotropic zoom (scale) to both image and mask.

    If scale > 1, the image is zoomed in (cropped); if scale < 1, it is
    zoomed out (padded with zeros). In both cases the output is resized
    back to the original spatial dimensions.

    Parameters
    ----------
    zoom_range : tuple of float
        Range ``(min_scale, max_scale)`` for the zoom factor.
        Default (0.85, 1.15).
    """

    def __init__(self, zoom_range: Tuple[float, float] = (0.85, 1.15)) -> None:
        self.zoom_range = zoom_range

    def __call__(self, sample: Sample) -> Sample:
        scale = random.uniform(*self.zoom_range)
        if abs(scale - 1.0) < 1e-4:
            return sample  # effectively identity

        sample = dict(sample)
        image = sample["image"]  # (1, H, W)
        mask = sample["mask"]  # (1, H, W)

        # Build affine matrix for scaling (identity + scale)
        theta = torch.tensor(
            [[scale, 0.0, 0.0], [0.0, scale, 0.0]],
            dtype=torch.float32,
        ).unsqueeze(
            0
        )  # (1, 2, 3)

        # Upsample to 4D for grid_sample
        img_4d = image.unsqueeze(0)  # (1, 1, H, W)
        msk_4d = mask.unsqueeze(0)  # (1, 1, H, W)

        grid = F.affine_grid(theta, img_4d.size(), align_corners=False)
        img_zoomed = F.grid_sample(img_4d, grid, mode="bilinear", align_corners=False)
        msk_zoomed = F.grid_sample(msk_4d, grid, mode="nearest", align_corners=False)

        sample["image"] = img_zoomed.squeeze(0)
        sample["mask"] = msk_zoomed.squeeze(0)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(zoom_range={self.zoom_range})"


class RandomRotation:
    """Rotate both image and mask by a random angle.

    The rotation angle is sampled uniformly from ``[-degrees, +degrees]``.
    Rotation is performed via bilinear interpolation for the image and
    nearest-neighbour for the mask (to preserve binary values).

    Parameters
    ----------
    degrees : float
        Maximum rotation angle in degrees (both CW and CCW). Default 15.
    """

    def __init__(self, degrees: float = 15.0) -> None:
        self.degrees = degrees

    def __call__(self, sample: Sample) -> Sample:
        angle_deg = random.uniform(-self.degrees, self.degrees)
        if abs(angle_deg) < 0.1:
            return sample  # skip near-zero rotation

        sample = dict(sample)
        image = sample["image"]  # (1, H, W)
        mask = sample["mask"]  # (1, H, W)

        angle_rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        # 2D rotation affine matrix (no translation)
        theta = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
            dtype=torch.float32,
        ).unsqueeze(0)

        img_4d = image.unsqueeze(0)
        msk_4d = mask.unsqueeze(0)

        grid = F.affine_grid(theta, img_4d.size(), align_corners=False)
        img_rot = F.grid_sample(img_4d, grid, mode="bilinear", align_corners=False)
        msk_rot = F.grid_sample(msk_4d, grid, mode="nearest", align_corners=False)

        sample["image"] = img_rot.squeeze(0)
        sample["mask"] = msk_rot.squeeze(0)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(degrees={self.degrees})"


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class Compose:
    """Compose a sequence of transforms, calling each in order.

    Parameters
    ----------
    transforms : list of callable
        Sequence of transforms to apply. Each must accept and return a Sample dict.
    """

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        lines = [f"  {t}" for t in self.transforms]
        return "Compose([\n" + ",\n".join(lines) + "\n])"


class IdentityTransform:
    """No-op transform — returns the sample unchanged."""

    def __call__(self, sample: Sample) -> Sample:
        return sample

    def __repr__(self) -> str:
        return "IdentityTransform()"


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def get_augmentation_pipeline(
    config: Dict,
    augment: Optional[bool] = None,
) -> Callable[[Sample], Sample]:
    """Build a composable augmentation pipeline from a config dict.

    Parameters
    ----------
    config : dict
        May contain the following optional keys (with their defaults):

        - ``augment`` (bool): Master switch. If False, returns IdentityTransform.
          Overridable by the ``augment`` argument.
        - ``random_flip_prob`` (float): Horizontal flip probability. Default 0.5.
        - ``vertical_flip_prob`` (float): Vertical flip probability. Default 0.3.
        - ``gaussian_noise_std`` (float): Max noise std. Default 0.02.
        - ``random_brightness`` (float): Max brightness shift. Default 0.1.
        - ``random_zoom_range`` (list/tuple): Zoom range. Default [0.85, 1.15].
        - ``random_rotation_degrees`` (float): Max rotation. Default 15.0.

    augment : bool, optional
        If provided, overrides ``config.get("augment", True)``.

    Returns
    -------
    Compose or IdentityTransform
        A callable that transforms a sample dict.

    Examples
    --------
    >>> aug = get_augmentation_pipeline({"random_flip_prob": 0.5})
    >>> sample = {"image": torch.zeros(1, 96, 96), "mask": torch.zeros(1, 96, 96)}
    >>> out = aug(sample)
    >>> out["image"].shape
    torch.Size([1, 96, 96])
    """
    # Master switch
    should_augment = augment if (augment is not None) else config.get("augment", True)
    if not should_augment:
        return IdentityTransform()

    transforms: List[Callable] = []

    flip_p = float(config.get("random_flip_prob", 0.5))
    if flip_p > 0:
        transforms.append(RandomHorizontalFlip(p=flip_p))

    vflip_p = float(config.get("vertical_flip_prob", 0.3))
    if vflip_p > 0:
        transforms.append(RandomVerticalFlip(p=vflip_p))

    rotate_deg = float(config.get("random_rotation_degrees", 15.0))
    if rotate_deg > 0:
        transforms.append(RandomRotation(degrees=rotate_deg))

    zoom_range = config.get("random_zoom_range", [0.85, 1.15])
    zoom_range = tuple(zoom_range)
    if zoom_range[0] != 1.0 or zoom_range[1] != 1.0:
        transforms.append(RandomZoom(zoom_range=zoom_range))

    brightness = float(config.get("random_brightness", 0.1))
    if brightness > 0:
        transforms.append(RandomBrightness(brightness=brightness))

    noise_std = float(config.get("gaussian_noise_std", 0.02))
    if noise_std > 0:
        transforms.append(GaussianNoise(std_range=(0.0, noise_std)))

    pipeline = Compose(transforms)
    logger.debug("Built augmentation pipeline: %s", pipeline)
    return pipeline


# ---------------------------------------------------------------------------
# Smoke test / demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Augmentation Pipeline — demo")
    print("=" * 60)

    config = {
        "random_flip_prob": 0.5,
        "vertical_flip_prob": 0.3,
        "random_rotation_degrees": 15.0,
        "random_zoom_range": [0.85, 1.15],
        "random_brightness": 0.1,
        "gaussian_noise_std": 0.02,
    }

    aug = get_augmentation_pipeline(config, augment=True)
    print(f"\nPipeline:\n{aug}\n")

    # Create a synthetic sample with a disc mask
    H, W = 96, 96
    image = torch.rand(1, H, W)
    mask = torch.zeros(1, H, W)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    mask[0] = ((yy - H // 2) ** 2 + (xx - W // 2) ** 2 <= 20**2).float()

    sample: Sample = {
        "image": image,
        "mask": mask,
        "slice_idx": torch.tensor(10, dtype=torch.int64),
        "patch_id": "demo_patch",
    }

    print(
        f"Input  — image: {sample['image'].shape}, mask sum: {sample['mask'].sum():.0f}"
    )

    for i in range(3):
        out = aug(sample)
        print(
            f"Aug {i+1}  — image: {out['image'].shape}, "
            f"mask sum: {out['mask'].sum():.1f}, "
            f"image range: [{out['image'].min():.3f}, {out['image'].max():.3f}]"
        )

    # Test identity (no augmentation)
    no_aug = get_augmentation_pipeline(config, augment=False)
    out_id = no_aug(sample)
    assert torch.allclose(out_id["image"], sample["image"]), "Identity should be no-op"
    print("\nIdentity transform: ✓")
    print("No stochastic transforms applied.")
