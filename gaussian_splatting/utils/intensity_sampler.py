# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.

"""
Utility functions for sampling and computing intensity and opacity values for Gaussian splats.
"""

import os

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Tuple

from gaussian_splatting.utils.orientation_field import (
    default_origin_and_spacing,
    world_to_grid,
    world_to_voxel,
)

_SAMPLING_VALIDATED = False


def _ensure_n3(points: Tensor) -> Tensor:
    """Return point tensor shaped [N, 3], cloning only if needed."""
    if points.dim() != 2:
        raise ValueError("Points tensor must have shape [..., 3].")
    if points.shape[1] == 3:
        return points
    if points.shape[0] == 3:
        return points.permute(1, 0)
    raise ValueError("Unexpected point shape; expected [N,3] or [3,N].")


def _normalize_points(points: Tensor, volume_shape: Tuple[int, int, int]) -> Tensor:
    """Normalize points from voxel index space to world [0,1]^3 if needed."""
    pts = points
    if pts.max() > 1.0 + 1e-6 or pts.min() < -1e-6:
        D, H, W = volume_shape
        pts = pts.clone()
        pts[:, 0] /= max(W - 1, 1)
        pts[:, 1] /= max(H - 1, 1)
        pts[:, 2] /= max(D - 1, 1)
    return pts


def sample_intensities_from_volume(
    points: Tensor,
    volume: Tensor,
    scale: Optional[Tensor] = None,
    radius_scale: float = 2.0,
    enable_footprint_pooling: bool = False,
    normalize: bool = False,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    padding_mode: str = "zeros",
) -> Tuple[Tensor, float, float]:
    """
    Sample intensity values from a volume for each point position.
    Computes a mean intensity within the Gaussian's influence region.

    Args:
        points: Point coordinates in normalized [0,1] space, shape [N, 3]
    volume: Input volume tensor with intensity values, shape [D, H, W]
    scale: Optional scale parameters for each point, shape [N, 3] or [N]
    radius_scale: How many standard deviations to consider for intensity computation
    normalize: Whether to normalize intensity values to [0,1] range
    min_val: Optional precomputed global minimum used for normalization
    max_val: Optional precomputed global maximum used for normalization

    Returns:
        Tuple of:
            - Intensity values for each point, shape [N, 1]
            - Global minimum intensity value in volume
            - Global maximum intensity value in volume
    """
    device = points.device
    volume = volume.to(device=device, dtype=torch.float32)
    points_n3 = _ensure_n3(points).to(device=device, dtype=torch.float32)
    points_n3 = _normalize_points(points_n3, volume.shape)

    if min_val is not None and max_val is not None:
        volume_min = float(min_val)
        volume_max = float(max_val)
    else:
        volume_min = float(volume.min().item())
        volume_max = float(volume.max().item())

    if (volume_max - volume_min) <= 1e-8:
        print("Warning: Volume has near-constant intensity; returning mid-gray.")
        base_value = 0.5 if normalize else volume_min
        fallback = torch.full((points_n3.shape[0], 1), base_value, device=device)
        return fallback, volume_min, volume_max

    origin, voxel = default_origin_and_spacing(volume.shape, device)
    grid = world_to_grid(points_n3, origin, voxel, volume.shape)
    grid = grid.view(1, -1, 1, 1, 3)

    global _SAMPLING_VALIDATED
    if not _SAMPLING_VALIDATED and os.environ.get("GS_VALIDATE_SAMPLING") == "1":
        _SAMPLING_VALIDATED = True
        idx = torch.stack(
            torch.meshgrid(
                torch.arange(volume.shape[0], device=device),
                torch.arange(volume.shape[1], device=device),
                torch.arange(volume.shape[2], device=device),
                indexing="ij",
            ),
            dim=-1,
        ).view(-1, 3)
        idx_xyz = torch.stack([idx[:, 2], idx[:, 1], idx[:, 0]], dim=-1).float()
        pts_full = idx_xyz / torch.tensor(
            [volume.shape[2] - 1, volume.shape[1] - 1, volume.shape[0] - 1],
            device=device,
        ).clamp_min(1)
        full_grid = world_to_grid(pts_full, origin, voxel, volume.shape)
        full_grid = full_grid.view(1, -1, 1, 1, 3)
        gt_samples = volume.unsqueeze(0).unsqueeze(0)
        round_trip = F.grid_sample(
            gt_samples,
            full_grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=True,
        ).view(-1)
        diff = (round_trip - volume.view(-1)).abs().max().item()
        print(f"Trilinear sampling check max |diff| = {diff:.4e}")

    volume_5d = volume.unsqueeze(0).unsqueeze(0)

    # Base (center) sample.
    samples = F.grid_sample(
        volume_5d,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
    intensities = samples.view(-1, 1)

    # By default, intensity sampling is defined as the trilinearly-sampled center voxel.
    # Footprint pooling is an explicit opt-in used by mean-covered modes.
    if (
        enable_footprint_pooling
        and scale is not None
        and scale.numel() > 0
        and radius_scale > 0.0
    ):
        scales_n3 = _ensure_n3(scale).to(device=device, dtype=torch.float32)
        # Voxel size in normalized [0,1]^3 coordinates.
        D, H, W = volume.shape
        voxel_norm = torch.tensor(
            [1.0 / max(W - 1, 1), 1.0 / max(H - 1, 1), 1.0 / max(D - 1, 1)],
            device=device,
            dtype=torch.float32,
        )
        large_mask = scales_n3.max(dim=1).values >= voxel_norm.max()

        if large_mask.any():
            # Offsets in "sigma units"; weights are independent of the actual scale.
            # Using [-1,0,1]^3 with a 0.5 factor keeps samples within ~1.5*sigma.
            offset_vals = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=torch.float32)
            offsets = torch.stack(
                torch.meshgrid(offset_vals, offset_vals, offset_vals, indexing="ij"),
                dim=-1,
            ).view(-1, 3)
            offsets = offsets * (0.5 * float(radius_scale))
            weights = torch.exp(-0.5 * (offsets ** 2).sum(dim=1))
            weights = weights / weights.sum().clamp_min(1e-8)

            idx = torch.nonzero(large_mask, as_tuple=False).view(-1)
            pts_large = points_n3[idx]
            scale_large = scales_n3[idx]

            pts_samples = pts_large[:, None, :] + scale_large[:, None, :] * offsets[None, :, :]
            pts_samples = pts_samples.clamp(0.0, 1.0)

            grid_large = world_to_grid(
                pts_samples.reshape(-1, 3),
                origin,
                voxel,
                volume.shape,
            ).view(1, -1, 1, 1, 3)
            samples_large = F.grid_sample(
                volume_5d,
                grid_large,
                mode="bilinear",
                padding_mode=padding_mode,
                align_corners=True,
            ).view(idx.shape[0], -1)

            pooled = (samples_large * weights[None, :]).sum(dim=1, keepdim=True)
            intensities = intensities.clone()
            intensities[idx] = pooled

    if normalize and volume_max > volume_min:
        denominator = max(volume_max - volume_min, 1e-8)
        intensities = (intensities - volume_min) / denominator
        intensities = intensities.clamp_(0.0, 1.0)

    return intensities, volume_min, volume_max


def sample_opacities_from_mask(
    points: Tensor,
    mask: Tensor,
    scale: Optional[Tensor] = None,
    radius_scale: float = 2.0,
    min_opacity: float = 0.0,
    max_opacity: float = 1.0,
) -> Tuple[Tensor, float, float]:
    """
    Sample opacity values from a mask volume for each point position.
    Computes mean opacity within the Gaussian's influence region.

    Args:
        points: Point coordinates in normalized [0,1] space, shape [N, 3]
        mask: Input mask tensor with values in [0,1], shape [D, H, W]
        scale: Optional scale parameters for each point, shape [N, 3] or [N]
        radius_scale: How many standard deviations to consider for opacity computation
        min_opacity: Minimum opacity value to ensure visibility
        max_opacity: Maximum opacity value to prevent complete occlusion

    Returns:
        Tuple of:
            - Opacity values for each point, shape [N, 1]
            - Global minimum mask value
            - Global maximum mask value
    """
    # Use the same sampling approach as intensity but with opacity range constraints
    raw_opacities, mask_min, mask_max = sample_intensities_from_volume(
        points,
        mask,
        scale,
        radius_scale,
        enable_footprint_pooling=True,
    )

    # Map mask values to opacity. Defaults preserve the raw [0,1] mask range.
    opacities = min_opacity + raw_opacities * (max_opacity - min_opacity)

    return opacities, mask_min, mask_max


def sample_mean_covered_voxel_intensities(
    points: Tensor,
    volume: Tensor,
    scales: Optional[Tensor],
    origin: Tensor,
    voxel_size: Tensor,
    *,
    radius_scale: float = 2.5,
    coverage_mask: Optional[Tensor] = None,
    normalize: bool = False,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    padding_mode: str = "zeros",
) -> Tuple[Tensor, float, float]:
    """Average voxel intensities across mean-covered regions for each splat."""
    if points.numel() == 0:
        device = volume.device
        empty = torch.empty(0, 1, device=device, dtype=volume.dtype)
        default_min = (
            float(min_val) if min_val is not None else float(volume.min().item())
        )
        default_max = (
            float(max_val) if max_val is not None else float(volume.max().item())
        )
        return empty, default_min, default_max

    device = volume.device
    work_dtype = torch.float32

    volume_f = volume.to(device=device, dtype=work_dtype)
    points_n3 = _ensure_n3(points).to(device=device, dtype=work_dtype)

    if min_val is not None and max_val is not None:
        volume_min = float(min_val)
        volume_max = float(max_val)
    else:
        volume_min = float(volume_f.min().item())
        volume_max = float(volume_f.max().item())

    base_samples, _, _ = sample_intensities_from_volume(
        points_n3,
        volume_f,
        normalize=False,
        min_val=volume_min,
        max_val=volume_max,
        padding_mode=padding_mode,
    )
    raw_values = base_samples.view(-1)

    if coverage_mask is None:
        coverage_mask = torch.ones_like(raw_values, dtype=torch.bool, device=device)
    else:
        coverage_mask = coverage_mask.to(device=device, dtype=torch.bool)

    if (
        scales is None
        or scales.numel() == 0
        or radius_scale <= 0.0
        or not coverage_mask.any()
    ):
        processed = raw_values
    else:
        scales_n3 = _ensure_n3(scales).to(device=device, dtype=work_dtype)
        scale_indices = coverage_mask.nonzero(as_tuple=False).view(-1)

        if scale_indices.numel() == 0:
            processed = raw_values
        else:
            origin_xyz = origin.to(device=device, dtype=work_dtype)
            voxel_size_xyz = voxel_size.to(device=device, dtype=work_dtype).clamp_min(
                1e-8
            )

            pts_cov = points_n3[scale_indices]
            scale_cov = scales_n3[scale_indices] * float(radius_scale)

            centers = world_to_voxel(pts_cov, origin_xyz, voxel_size_xyz)
            extent_xyz = scale_cov / voxel_size_xyz.unsqueeze(0)
            extent_zyx = extent_xyz[:, [2, 1, 0]]

            min_idx = torch.floor(centers - extent_zyx)
            max_idx = torch.ceil(centers + extent_zyx)

            zeros_vec = torch.zeros(3, device=device, dtype=work_dtype)
            dims = torch.tensor(
                [volume.shape[0] - 1, volume.shape[1] - 1, volume.shape[2] - 1],
                device=device,
                dtype=work_dtype,
            )
            min_idx = torch.maximum(min_idx, zeros_vec)
            max_idx = torch.minimum(max_idx, dims)

            min_idx_long = min_idx.to(torch.long).cpu()
            max_idx_long = max_idx.to(torch.long).cpu()

            updated_samples = raw_values.clone()
            subset = updated_samples[scale_indices].clone()

            for i in range(scale_indices.shape[0]):
                z0 = int(min_idx_long[i, 0])
                y0 = int(min_idx_long[i, 1])
                x0 = int(min_idx_long[i, 2])
                z1 = int(max_idx_long[i, 0])
                y1 = int(max_idx_long[i, 1])
                x1 = int(max_idx_long[i, 2])

                if z1 < z0 or y1 < y0 or x1 < x0:
                    continue

                region = volume_f[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
                if region.numel() == 0:
                    continue
                subset[i] = region.mean()

            updated_samples[scale_indices] = subset
            processed = updated_samples

    if normalize:
        if volume_max > volume_min + 1e-8:
            denom = max(volume_max - volume_min, 1e-8)
            processed = (processed - volume_min) / denom
            processed = processed.clamp_(0.0, 1.0)
        else:
            processed = torch.full_like(processed, 0.5)

    return processed.view(-1, 1).to(volume.dtype), volume_min, volume_max


def update_intensities(
    points: Tensor,
    volume: Tensor,
    scale: Optional[Tensor] = None,
    normalize: bool = False,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    padding_mode: str = "zeros",
) -> Tuple[Tensor, float, float]:
    """
    Update intensity values for points based on their current positions.
    Should be called whenever point positions or scales change significantly.

    Args:
        points: Point coordinates, shape [3, N] or [N, 3]
    volume: Reference volume with intensity values
    scale: Scale parameters for points
    normalize: Whether to normalize intensities to [0,1] range
    min_val: Optional precomputed global minimum used for normalization
    max_val: Optional precomputed global maximum used for normalization

    Returns:
        Tuple of:
            - Updated intensity values, shape [N, 1]
            - Global minimum intensity value in volume
            - Global maximum intensity value in volume
    """
    points_n3 = _ensure_n3(points)
    return sample_intensities_from_volume(
        points_n3,
        volume,
        scale,
        enable_footprint_pooling=False,
        normalize=normalize,
        min_val=min_val,
        max_val=max_val,
        padding_mode=padding_mode,
    )


def update_opacities(
    points: Tensor, mask: Tensor, scale: Optional[Tensor] = None
) -> Tuple[Tensor, float, float]:
    """
    Update opacity values for points based on their current positions.
    Should be called whenever point positions or scales change significantly.

    Args:
        points: Point coordinates, shape [3, N] or [N, 3]
        mask: Reference mask volume with values in [0,1]
        scale: Scale parameters for points

    Returns:
        Tuple of:
            - Updated opacity values, shape [N, 1]
            - Minimum mask value (usually 0)
            - Maximum mask value (usually 1)
    """
    points_n3 = _ensure_n3(points)

    # Get global min/max of the mask
    mask_min = float(mask.min().item())
    mask_max = float(mask.max().item())

    # Get opacity values
    opacities, _, _ = sample_opacities_from_mask(points_n3, mask, scale)

    return opacities, mask_min, mask_max


def update_intensities_and_opacities(
    points: Tensor,
    volume: Tensor,
    mask: Optional[Tensor] = None,
    scale: Optional[Tensor] = None,
    normalize: bool = False,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    padding_mode: str = "zeros",
) -> Tuple[Tensor, Optional[Tensor], float, float]:
    """
    Update both intensity and opacity values for points based on their current positions.
    Should be called whenever point positions or scales change significantly.

    Args:
        points: Point coordinates, shape [3, N] or [N, 3]
    volume: Reference volume with intensity values
    mask: Optional reference mask with opacity values
    scale: Scale parameters for points
    normalize: Whether to normalize intensities to [0,1] range
    min_val: Optional precomputed global minimum used for normalization
    max_val: Optional precomputed global maximum used for normalization

    Returns:
        Tuple of:
            - Intensity values, shape [N, 1]
            - Opacity values, shape [N, 1] or None
            - Global minimum intensity value in volume
            - Global maximum intensity value in volume
    """
    # Update intensities and get global min/max
    intensities, volume_min, volume_max = update_intensities(
        points,
        volume,
        scale,
        normalize,
        min_val=min_val,
        max_val=max_val,
        padding_mode=padding_mode,
    )

    # Update opacities if mask is provided
    opacities = None
    if mask is not None:
        opacities, _, _ = update_opacities(points, mask, scale)

    return intensities, opacities, volume_min, volume_max
