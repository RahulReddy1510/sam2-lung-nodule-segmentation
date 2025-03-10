"""
LUNA16 CT Preprocessing Pipeline.

Transforms raw LUNA16 .mhd/.zraw volumes into normalized, resampled,
patch-extracted NumPy arrays ready for PyTorch training.

Pipeline:
    .mhd volume → resample to 1mm isotropic → HU window → nodule mask →
    96³ patch extraction → train/val/test split → save as .npy pairs

Usage:
    python data/luna16_preprocessing.py \\
        --input_dir  /data/LUNA16/raw \\
        --output_dir /data/LUNA16/preprocessed \\
        --annotations_csv /data/LUNA16/annotations.csv \\
        --patch_size 96 \\
        --num_workers 8
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_mhd_volume(
    path: str | Path,
) -> Tuple[np.ndarray, Tuple[float, ...], Tuple[float, ...]]:
    """Load a LUNA16 .mhd volume using SimpleITK.

    Parameters
    ----------
    path : str or Path
        Path to the .mhd file (companion .zraw or .raw must be in the same dir).

    Returns
    -------
    volume : np.ndarray
        Float32 array of shape (Z, Y, X) in Hounsfield Units.
    spacing : tuple of float
        Voxel spacing in mm as (sz, sy, sx) — note SimpleITK returns (x,y,z)
        and we reverse to match (Z, Y, X) array convention.
    origin : tuple of float
        World-coordinate origin as (oz, oy, ox).

    Notes
    -----
    SimpleITK reads images in (x, y, z) order. We convert the array to
    (z, y, x) order (depth-first) to match the standard radiological convention
    used throughout this codebase.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MHD file not found: {path}")

    sitk_img: sitk.Image = sitk.ReadImage(str(path))
    # SimpleITK spacing/origin are in (x, y, z) order
    spacing_xyz: Tuple[float, ...] = sitk_img.GetSpacing()
    origin_xyz: Tuple[float, ...] = sitk_img.GetOrigin()

    # Convert to numpy (z, y, x)
    volume: np.ndarray = sitk.GetArrayFromImage(sitk_img).astype(np.float32)

    # Reverse ordering to match (z, y, x)
    spacing: Tuple[float, ...] = tuple(reversed(spacing_xyz))  # (sz, sy, sx)
    origin: Tuple[float, ...] = tuple(reversed(origin_xyz))  # (oz, oy, ox)

    logger.debug(
        "Loaded %s | shape=%s | spacing=%s mm | origin=%s",
        path.name,
        volume.shape,
        spacing,
        origin,
    )
    return volume, spacing, origin


def resample_volume(
    sitk_img: sitk.Image,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> sitk.Image:
    """Resample a SimpleITK image to the given isotropic voxel spacing.

    Parameters
    ----------
    sitk_img : sitk.Image
        Input image (any spacing).
    target_spacing : tuple of float
        Desired output spacing in mm, given as (x, y, z) to match SimpleITK
        convention. Default is (1.0, 1.0, 1.0).

    Returns
    -------
    resampled : sitk.Image
        Resampled image with target_spacing voxels.

    Notes
    -----
    New size is computed as::

        new_size[i] = round(orig_size[i] * orig_spacing[i] / target_spacing[i])

    Linear (sitkLinear) interpolation is used for the image values.
    """
    orig_spacing: Tuple[float, ...] = sitk_img.GetSpacing()
    orig_size: Tuple[int, ...] = sitk_img.GetSize()

    new_size: List[int] = [
        int(round(orig_size[i] * orig_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)  # air HU outside FOV

    resampled: sitk.Image = resampler.Execute(sitk_img)
    logger.debug(
        "Resampled: %s → %s | spacing: %s → %s mm",
        orig_size,
        new_size,
        orig_spacing,
        target_spacing,
    )
    return resampled


def apply_hu_window(
    volume: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
) -> np.ndarray:
    """Clip Hounsfield Units to [hu_min, hu_max] and normalize to [0, 1].

    Parameters
    ----------
    volume : np.ndarray
        Raw HU array of any shape and dtype.
    hu_min : float
        Lower bound of the HU window (default -1000 = air).
    hu_max : float
        Upper bound of the HU window (default 400 = soft tissue/bone boundary).

    Returns
    -------
    windowed : np.ndarray
        Float32 array in [0, 1] with the same shape as input.

    Notes
    -----
    We clip BEFORE normalizing (not after). Normalizing before clipping would
    produce incorrect scale — a bug present in many open LUNA16 baselines.
    """
    # Order matters: clip first, then normalize
    clipped = np.clip(volume, hu_min, hu_max)
    windowed = (clipped - hu_min) / (hu_max - hu_min)
    return windowed.astype(np.float32)


def world_to_voxel(
    world_coord: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    """Convert a world-coordinate point to voxel indices.

    Parameters
    ----------
    world_coord : tuple of float
        World coordinates as (wz, wy, wx) in mm.
    origin : tuple of float
        Volume origin as (oz, oy, ox) in mm.
    spacing : tuple of float
        Voxel spacing as (sz, sy, sx) in mm/voxel.

    Returns
    -------
    voxel_idx : tuple of int
        Integer voxel indices as (iz, iy, ix).

    Notes
    -----
    The formula is: voxel[i] = round((world[i] - origin[i]) / spacing[i]).
    We round (not truncate) because nodule centers may fall at sub-voxel
    positions after world-to-voxel conversion.
    """
    voxel_idx = tuple(
        int(round((world_coord[i] - origin[i]) / spacing[i])) for i in range(3)
    )
    return voxel_idx  # type: ignore[return-value]


def create_nodule_mask(
    volume_shape: Tuple[int, int, int],
    center_voxel: Tuple[int, int, int],
    radius_mm: float,
    spacing: Tuple[float, float, float],
) -> np.ndarray:
    """Create a spherical binary mask for a nodule.

    Parameters
    ----------
    volume_shape : tuple of int
        Full volume shape (Z, Y, X).
    center_voxel : tuple of int
        Nodule center in voxel coordinates (iz, iy, ix).
    radius_mm : float
        Nodule radius in millimetres.
    spacing : tuple of float
        Voxel spacing (sz, sy, sx) in mm, used to convert radius to voxels
        per axis.

    Returns
    -------
    mask : np.ndarray
        Binary uint8 array of shape volume_shape. Voxels inside the sphere = 1.

    Notes
    -----
    We use a bounding-box optimization: instead of checking all voxels in the
    full volume (which can be 512³ = 134M iterations), we only iterate over
    a tight bounding box around the nodule center. For a 30mm diameter nodule
    at 1mm spacing, this reduces the search space from ~134M to ~27K voxels —
    roughly a 5000× speedup.
    """
    mask = np.zeros(volume_shape, dtype=np.uint8)
    cz, cy, cx = center_voxel
    sz, sy, sx = spacing

    # Convert radius from mm to voxels along each axis
    rz = int(np.ceil(radius_mm / sz)) + 1
    ry = int(np.ceil(radius_mm / sy)) + 1
    rx = int(np.ceil(radius_mm / sx)) + 1

    # Clamp bounding box to valid volume bounds
    z0, z1 = max(0, cz - rz), min(volume_shape[0], cz + rz + 1)
    y0, y1 = max(0, cy - ry), min(volume_shape[1], cy + ry + 1)
    x0, x1 = max(0, cx - rx), min(volume_shape[2], cx + rx + 1)

    # Build coordinate grids for the bounding box
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]

    # Ellipsoidal distance test (accounts for anisotropic spacing)
    dist_sq = (
        ((zz - cz) * sz) ** 2 + ((yy - cy) * sy) ** 2 + ((xx - cx) * sx) ** 2
    )
    inside = dist_sq <= radius_mm**2
    mask[z0:z1, y0:y1, x0:x1][inside] = 1

    logger.debug(
        "Created nodule mask | center=%s | radius=%.1f mm | n_voxels=%d",
        center_voxel,
        radius_mm,
        int(mask.sum()),
    )
    return mask


def extract_patch(
    volume: np.ndarray,
    mask: np.ndarray,
    center: Tuple[int, int, int],
    patch_size: Tuple[int, int, int] = (96, 96, 96),
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a fixed-size 3D patch centered on a nodule.

    Parameters
    ----------
    volume : np.ndarray
        Full CT volume of shape (Z, Y, X), float32.
    mask : np.ndarray
        Full binary mask of shape (Z, Y, X), uint8.
    center : tuple of int
        Patch center in voxel coordinates (iz, iy, ix).
    patch_size : tuple of int
        Output patch dimensions (pz, py, px). Default (96, 96, 96).

    Returns
    -------
    vol_patch : np.ndarray
        Image patch of shape patch_size, float32.
    mask_patch : np.ndarray
        Mask patch of shape patch_size, uint8.

    Notes
    -----
    When the center is close to the volume boundary, the requested patch
    may extend outside the volume. In this case we extract only the valid
    interior region and embed it in a zero-padded patch of patch_size.
    This avoids discarding boundary nodules.
    """
    pz, py, px = patch_size
    cz, cy, cx = center
    Z, Y, X = volume.shape

    # Patch half-extents
    hz, hy, hx = pz // 2, py // 2, px // 2

    # Desired slice coordinates (may be out of bounds)
    z_start, z_end = cz - hz, cz - hz + pz
    y_start, y_end = cy - hy, cy - hy + py
    x_start, x_end = cx - hx, cx - hx + px

    # Clamped (valid) source region
    src_z0, src_z1 = max(0, z_start), min(Z, z_end)
    src_y0, src_y1 = max(0, y_start), min(Y, y_end)
    src_x0, src_x1 = max(0, x_start), min(X, x_end)

    # Destination region in the output patch
    dst_z0 = src_z0 - z_start
    dst_y0 = src_y0 - y_start
    dst_x0 = src_x0 - x_start
    dst_z1 = dst_z0 + (src_z1 - src_z0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    vol_patch = np.zeros(patch_size, dtype=np.float32)
    mask_patch = np.zeros(patch_size, dtype=np.uint8)

    vol_patch[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = volume[
        src_z0:src_z1, src_y0:src_y1, src_x0:src_x1
    ]
    mask_patch[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = mask[
        src_z0:src_z1, src_y0:src_y1, src_x0:src_x1
    ]

    return vol_patch, mask_patch


def get_train_val_test_splits(
    uids: List[str],
    train: float = 0.72,
    val: float = 0.14,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Randomly split series UIDs into train / val / test sets.

    Parameters
    ----------
    uids : list of str
        All unique series UIDs to split.
    train : float
        Fraction for training. Default 0.72.
    val : float
        Fraction for validation. Default 0.14.
        Test fraction is implicitly ``1 - train - val``.
    seed : int
        Random seed for reproducibility. Default 42.

    Returns
    -------
    splits : dict
        Dictionary with keys ``"train"``, ``"val"``, ``"test"``, each
        mapping to a list of series UIDs.

    Notes
    -----
    All nodules from a single CT study (same seriesuid) are placed in the
    same split, preventing any data leakage across splits. The split is
    performed at the study level, not the nodule level.
    """
    rng = random.Random(seed)
    shuffled = list(uids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train))
    n_val = int(round(n * val))

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


# ---------------------------------------------------------------------------
# Single-study processing worker
# ---------------------------------------------------------------------------


def _process_one_study(
    seriesuid: str,
    nodule_rows: pd.DataFrame,
    raw_dir: Path,
    output_dir: Path,
    patch_size: Tuple[int, int, int],
    hu_min: float,
    hu_max: float,
    min_diameter_mm: float,
) -> List[str]:
    """Process a single LUNA16 CT study and save patch pairs.

    Parameters
    ----------
    seriesuid : str
        Study identifier.
    nodule_rows : pd.DataFrame
        Rows from annotations.csv for this study.
    raw_dir : Path
        Root directory containing the subset folders.
    output_dir : Path
        Directory where .npy patches will be saved.
    patch_size : tuple of int
        Patch dimensions (Z, Y, X).
    hu_min, hu_max : float
        HU window bounds.
    min_diameter_mm : float
        Nodules smaller than this are skipped.

    Returns
    -------
    saved_ids : list of str
        List of patch_id strings that were successfully saved.
    """
    # Locate the .mhd file in any of the 10 subsets
    mhd_path: Optional[Path] = None
    for subset_dir in sorted(raw_dir.glob("subset*")):
        candidate = subset_dir / f"{seriesuid}.mhd"
        if candidate.exists():
            mhd_path = candidate
            break

    if mhd_path is None:
        logger.warning("MHD not found for seriesuid=%s — skipping", seriesuid)
        return []

    try:
        volume_raw, _, _ = load_mhd_volume(mhd_path)
        sitk_img = sitk.ReadImage(str(mhd_path))
        sitk_img = resample_volume(sitk_img, target_spacing=(1.0, 1.0, 1.0))

        volume = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        spacing_resampled = (1.0, 1.0, 1.0)
        origin_resampled = tuple(reversed(sitk_img.GetOrigin()))

        # Apply HU windowing
        volume = apply_hu_window(volume, hu_min=hu_min, hu_max=hu_max)

        saved_ids: List[str] = []

        for _, row in nodule_rows.iterrows():
            diameter_mm: float = float(row["diameter_mm"])
            if diameter_mm < min_diameter_mm:
                logger.debug(
                    "Skipping small nodule %.1f mm < %.1f mm threshold",
                    diameter_mm,
                    min_diameter_mm,
                )
                continue

            radius_mm = diameter_mm / 2.0
            # Annotations are in (x, y, z) world coordinates
            world_xyz = (float(row["coordZ"]), float(row["coordY"]), float(row["coordX"]))
            center_voxel = world_to_voxel(world_xyz, origin_resampled, spacing_resampled)

            # Safety: clamp center to valid range
            center_voxel = tuple(
                max(0, min(center_voxel[i], volume.shape[i] - 1)) for i in range(3)
            )

            # Create full-volume mask
            full_mask = create_nodule_mask(
                volume.shape, center_voxel, radius_mm, spacing_resampled
            )

            vol_patch, mask_patch = extract_patch(
                volume, full_mask, center_voxel, patch_size=patch_size
            )

            # Unique ID per nodule: seriesuid + rounded coordinates
            cx, cy_, cz = row["coordX"], row["coordY"], row["coordZ"]
            patch_id = f"{seriesuid}_{cx:.1f}_{cy_:.1f}_{cz:.1f}"

            np.save(output_dir / f"{patch_id}_image.npy", vol_patch)
            np.save(output_dir / f"{patch_id}_mask.npy", mask_patch)
            saved_ids.append(patch_id)

        return saved_ids

    except Exception as exc:
        logger.error("Failed to process %s: %s", seriesuid, exc)
        return []


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def process_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    annotations_csv: str | Path,
    patch_size: Tuple[int, int, int] = (96, 96, 96),
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
    min_nodule_diameter_mm: float = 3.0,
    train_frac: float = 0.72,
    val_frac: float = 0.14,
    seed: int = 42,
    num_workers: int = 4,
) -> None:
    """Orchestrate the full LUNA16 preprocessing pipeline.

    Parameters
    ----------
    input_dir : str or Path
        Root directory containing subset0/ … subset9/ folders.
    output_dir : str or Path
        Root directory for preprocessed .npy patches.
        Subdirectories ``train/``, ``val/``, ``test/`` will be created.
    annotations_csv : str or Path
        Path to LUNA16 annotations.csv.
    patch_size : tuple of int
        Patch dimensions (pz, py, px). Default (96, 96, 96).
    hu_min : float
        Lower HU window bound. Default -1000.
    hu_max : float
        Upper HU window bound. Default 400.
    min_nodule_diameter_mm : float
        Minimum nodule size to include. Smaller nodules are skipped.
    train_frac : float
        Train split fraction. Default 0.72.
    val_frac : float
        Validation split fraction. Default 0.14.
    seed : int
        Random seed. Default 42.
    num_workers : int
        Number of parallel worker processes. Default 4.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    annotations_csv = Path(annotations_csv)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not annotations_csv.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {annotations_csv}")

    # Load annotations
    annotations = pd.read_csv(annotations_csv)
    logger.info(
        "Loaded %d nodule annotations for %d unique studies",
        len(annotations),
        annotations["seriesuid"].nunique(),
    )

    # Split at study level
    all_uids: List[str] = annotations["seriesuid"].unique().tolist()
    splits = get_train_val_test_splits(all_uids, train=train_frac, val=val_frac, seed=seed)
    logger.info(
        "Split: %d train | %d val | %d test studies",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    # Create output directories
    for split_name in ("train", "val", "test"):
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)

    # Process each split
    total_saved = 0
    for split_name, uid_list in splits.items():
        split_out = output_dir / split_name
        logger.info("Processing %s split (%d studies)...", split_name, len(uid_list))

        uid_to_rows = {
            uid: annotations[annotations["seriesuid"] == uid]
            for uid in uid_list
        }

        if num_workers <= 1:
            results = []
            for uid in tqdm(uid_list, desc=split_name, unit="study"):
                saved = _process_one_study(
                    uid,
                    uid_to_rows[uid],
                    input_dir,
                    split_out,
                    patch_size,
                    hu_min,
                    hu_max,
                    min_nodule_diameter_mm,
                )
                results.extend(saved)
        else:
            results = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        _process_one_study,
                        uid,
                        uid_to_rows[uid],
                        input_dir,
                        split_out,
                        patch_size,
                        hu_min,
                        hu_max,
                        min_nodule_diameter_mm,
                    ): uid
                    for uid in uid_list
                }
                with tqdm(total=len(uid_list), desc=split_name, unit="study") as pbar:
                    for future in as_completed(futures):
                        saved = future.result()
                        results.extend(saved)
                        pbar.update(1)

        logger.info("  Saved %d patches to %s", len(results), split_out)
        total_saved += len(results)

    logger.info("Preprocessing complete. Total patches saved: %d", total_saved)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess LUNA16 CT volumes into normalized patch pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing LUNA16 subset0/ … subset9/ folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write preprocessed train/val/test .npy patches.",
    )
    parser.add_argument(
        "--annotations_csv",
        type=str,
        required=True,
        help="Path to LUNA16 annotations.csv.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=96,
        help="Cubic patch size in voxels.",
    )
    parser.add_argument(
        "--hu_min",
        type=float,
        default=-1000.0,
        help="Lower HU window bound.",
    )
    parser.add_argument(
        "--hu_max",
        type=float,
        default=400.0,
        help="Upper HU window bound.",
    )
    parser.add_argument(
        "--min_nodule_diameter_mm",
        type=float,
        default=3.0,
        help="Minimum nodule diameter to include.",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.72,
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.14,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel preprocessing workers.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        annotations_csv=args.annotations_csv,
        patch_size=(args.patch_size, args.patch_size, args.patch_size),
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        min_nodule_diameter_mm=args.min_nodule_diameter_mm,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
