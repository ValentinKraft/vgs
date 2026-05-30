# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.

"""
Volume data loading and preprocessing for 3D Gaussian Splatting.
"""

import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Optional, Union
from pathlib import Path
try:
    import nibabel as nib
except ModuleNotFoundError:  # pragma: no cover
    nib = None
# import SimpleITK as sitk
import torch.nn.functional as F

class VolumeLoader:
    """
    Loader for volumetric data supporting various medical imaging formats.
    Handles loading, optional resampling, and coordinate system alignment.
    """

    def __init__(
                 self,
                 target_shape: Optional[Tuple[int, int, int]] = None,
                 device: torch.device = torch.device('cuda'),
                 downscale_factor: Optional[int] = None,
                 storage_dtype: str = "fp32",
                 enable_overflow_guard: bool = True):
        """
        Args:
            target_shape: Optional target shape for resampling. If None, keeps original dimensions
            device: Device to load tensors to
            downscale_factor: Optional integer downscale factor applied to the input volume shape.
                When provided and > 1, volumes are resampled to (D//factor, H//factor, W//factor).
                When provided and <= 1, resampling is disabled (native resolution), unless the
                automatic overflow guard triggers.
            storage_dtype: Storage dtype used for the returned tensor on device.
                One of {'fp32', 'fp16', 'bf16'}.
            enable_overflow_guard: When True (default), automatically resample very large volumes
                to avoid downstream operations that cannot handle >~16M voxels. Set to False to
                force native resolution loading (may require more memory).
        """
        self.target_shape = target_shape
        self.device = device
        self.downscale_factor = downscale_factor
        self.storage_dtype = str(storage_dtype).lower()
        if self.storage_dtype not in {"fp32", "fp16", "bf16"}:
            raise ValueError(
                "storage_dtype must be one of {'fp32','fp16','bf16'}, "
                f"got {storage_dtype!r}."
            )
        self.enable_overflow_guard = bool(enable_overflow_guard)
        self.last_loaded_raw_min: Optional[float] = None
        self.last_loaded_raw_max: Optional[float] = None
        self.last_loaded_spacing_xyz: Optional[Tuple[float, float, float]] = None

    @staticmethod
    def peek_raw_range(path: Union[str, Path]) -> Tuple[float, float]:
        """Return raw (pre-normalization) min/max values for the input volume file."""
        path = Path(path)
        if path.suffix in ['.nii', '.gz']:
            if nib is None:
                raise ModuleNotFoundError(
                    "nibabel is required to load NIfTI volumes. Install it or use a .npy input."
                )
            volume_np = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
        elif path.suffix == '.npy':
            volume_np = np.asarray(np.load(str(path)), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return float(volume_np.min()), float(volume_np.max())

    def _target_dtype(self) -> torch.dtype:
        """Return torch dtype configured for stored supervision volumes."""
        if self.storage_dtype == "fp16":
            return torch.float16
        if self.storage_dtype == "bf16":
            return torch.bfloat16
        return torch.float32

    @staticmethod
    def _normalize_volume(volume: Tensor) -> Tensor:
        """Normalize to [0, 1] while preserving constant positive masks."""
        vol_min = volume.min()
        vol_max = volume.max()
        if float((vol_max - vol_min).item()) <= 1e-8:
            fill_value = 1.0 if float(vol_max.item()) > 0.0 else 0.0
            return torch.full_like(volume, fill_value)
        return (volume - vol_min) / (vol_max - vol_min)

    def load_nifti(self, path: Union[str, Path]) -> Tensor:
        """Load a NIfTI volume file."""
        if nib is None:
            raise ModuleNotFoundError(
                "nibabel is required to load NIfTI volumes. Install it or use a .npy input."
            )
        nii = nib.load(str(path))
        zooms = nii.header.get_zooms()
        if len(zooms) >= 3:
            # nibabel reports spacing in (X, Y, Z) voxel index order.
            self.last_loaded_spacing_xyz = (
                float(zooms[0]),
                float(zooms[1]),
                float(zooms[2]),
            )
        else:
            self.last_loaded_spacing_xyz = None
        volume = torch.from_numpy(nii.get_fdata()).float()
        # nibabel returns arrays in voxel index order (i, j, k) which typically
        # corresponds to (X, Y, Z). This project consistently represents volumes
        # as torch tensors in (D, H, W) = (Z, Y, X) order.
        if volume.dim() == 3:
            volume = volume.permute(2, 1, 0).contiguous()
        return self._process_volume(volume)

    def load_npy(self, path: Union[str, Path]) -> Tensor:
        """Load a NumPy volume file."""
        self.last_loaded_spacing_xyz = None
        volume = torch.from_numpy(np.load(str(path))).float()
        return self._process_volume(volume)

    # def load_mhd(self, path: Union[str, Path]) -> Tensor:
    #     """Load a MetaImage (MHD/Raw) volume file."""
    #     img = sitk.ReadImage(str(path))
    #     volume = torch.from_numpy(sitk.GetArrayFromImage(img)).float()
    #     return self._process_volume(volume)

    def load_volume(self, path: Union[str, Path]) -> Tensor:
        """
        Load a volume file based on its extension.
        
        Args:
            path: Path to volume file (.nii, .nii.gz, .npy, .mhd)
            
        Returns:
            Normalized and resampled volume tensor
        """
        path = Path(path)

        if path.suffix in ['.nii', '.gz']:
            volume = self.load_nifti(path)
        elif path.suffix == '.npy':
            volume = self.load_npy(path)
        # elif path.suffix == '.mhd':
        #     volume = self.load_mhd(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return volume

    def _process_volume(self, volume: Tensor) -> Tensor:
        """
        Process loaded volume:
        1. Normalize to [0, 1]
        2. Automatically resample if volume is too large
        3. Optionally resample to target shape
        4. Move to device
        """
        # Capture raw range before normalization.
        self.last_loaded_raw_min = float(volume.min().item())
        self.last_loaded_raw_max = float(volume.max().item())

        # Normalize while keeping constant positive masks as full foreground.
        volume = self._normalize_volume(volume)

        requested_target_shape = self.target_shape
        original_shape_dhw = tuple(int(v) for v in volume.shape)

        # Optional downscale relative to the input volume shape.
        if requested_target_shape is None and self.downscale_factor is not None:
            factor = int(self.downscale_factor)
            if factor > 1:
                D, H, W = volume.shape
                requested_target_shape = (
                    max(1, D // factor),
                    max(1, H // factor),
                    max(1, W // factor),
                )

        effective_target_shape = requested_target_shape

        if self.enable_overflow_guard:
            # Automatically determine (or adjust) target shape to prevent downstream operations
            # from failing on extremely large volumes.
            # Important: consider the voxel count *after* requested downscaling/target_shape.
            max_voxels = 2**24 - 1  # Conservative safety cap (~16M voxels)
            if effective_target_shape is None:
                candidate_shape = tuple(int(v) for v in volume.shape)
            else:
                candidate_shape = tuple(int(v) for v in effective_target_shape)

            candidate_voxels = (
                int(candidate_shape[0])
                * int(candidate_shape[1])
                * int(candidate_shape[2])
            )
            if candidate_voxels > max_voxels:
                # Keep aspect ratio while ensuring total voxels < max_voxels.
                scale = (max_voxels / float(candidate_voxels)) ** (1.0 / 3.0)
                D0, H0, W0 = candidate_shape
                adjusted = (
                    max(32, int(D0 * scale)),
                    max(32, int(H0 * scale)),
                    max(32, int(W0 * scale)),
                )
                # Ensure we actually reduce the voxel count.
                adjusted_voxels = (
                    int(adjusted[0]) * int(adjusted[1]) * int(adjusted[2])
                )
                if adjusted_voxels >= candidate_voxels:
                    adjusted = (
                        max(32, D0 - 1),
                        max(32, H0 - 1),
                        max(32, W0 - 1),
                    )

                if effective_target_shape is None:
                    print(
                        "Auto-resizing volume from "
                        f"{tuple(int(v) for v in volume.shape)} to {adjusted} "
                        "to prevent overflow"
                    )
                else:
                    print(
                        "Auto-resizing requested volume shape "
                        f"{candidate_shape} to {adjusted} to prevent overflow"
                    )
                effective_target_shape = adjusted

        # Resample if a target shape is specified
        if effective_target_shape is not None:
            # Add batch and channel dimensions for resampling
            volume = volume.unsqueeze(0).unsqueeze(0)

            # Resample to target shape
            volume = F.interpolate(
                volume,
                size=effective_target_shape,
                mode='trilinear',
                align_corners=True
            )

            # Remove batch and channel dimensions
            volume = volume.squeeze(0).squeeze(0)

        # Adjust physical spacing if known and resampling changed the grid shape.
        if self.last_loaded_spacing_xyz is not None:
            old_dhw = original_shape_dhw
            new_dhw = tuple(int(v) for v in volume.shape)
            old_xyz = np.array([old_dhw[2], old_dhw[1], old_dhw[0]], dtype=np.float32)
            new_xyz = np.array([new_dhw[2], new_dhw[1], new_dhw[0]], dtype=np.float32)
            old_spacing = np.array(self.last_loaded_spacing_xyz, dtype=np.float32)
            scale = np.maximum(old_xyz - 1.0, 1.0) / np.maximum(new_xyz - 1.0, 1.0)
            new_spacing = old_spacing * scale
            self.last_loaded_spacing_xyz = (
                float(new_spacing[0]),
                float(new_spacing[1]),
                float(new_spacing[2]),
            )

        return volume.to(device=self.device, dtype=self._target_dtype())

    def align_to_space(self, 
                      volume: Tensor,
                      bbox_min: Tensor,
                      bbox_max: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Align volume coordinates to match the Gaussian splat coordinate system.
        
        Args:
            volume: Input volume tensor
            bbox_min: Minimum point of bounding box in world space
            bbox_max: Maximum point of bounding box in world space
            
        Returns:
            Tuple of (aligned volume, coordinate grid)
        """
        # Create normalized coordinate grid
        coords = create_grid_points(volume.shape, volume.device)

        # Scale coordinates to bounding box
        scale = bbox_max - bbox_min
        coords = coords * scale.view(1, 1, 1, 3) + bbox_min.view(1, 1, 1, 3)

        return volume, coords
