# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.

"""
Initialize Gaussian points from volume data for 3D Gaussian Splatting.
"""

import math
import heapq
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.general_utils import inverse_sigmoid

from scene.gaussian_model import GaussianModel
from gaussian_splatting.data.volume_loader import VolumeLoader
from gaussian_splatting.utils.intensity_sampler import (
    sample_intensities_from_volume,
    update_intensities,
)
from gaussian_splatting.utils.orientation_field import (
    compute_gradient_field,
    default_origin_and_spacing,
    gather_rotation_from_gradient,
    quat_from_directions,
    random_quat_perturb,
    structure_from_mask_at_ijk,
    world_to_voxel,
)

if TYPE_CHECKING:
    from gaussian_splatting.utils.volume_supervisor import VolumeSupervisor


def _blend_quaternions(
    source_quat: Tensor,
    target_quat: Tensor,
    blend: Tensor,
) -> Tensor:
    """Blend quaternion pairs with hemisphere alignment and renormalization."""
    if source_quat.numel() == 0:
        return source_quat

    blend = blend.to(device=source_quat.device, dtype=source_quat.dtype).view(-1, 1)
    blend = blend.clamp(0.0, 1.0)
    target_quat = target_quat.to(device=source_quat.device, dtype=source_quat.dtype)

    dot = (source_quat * target_quat).sum(dim=1, keepdim=True)
    aligned_target = torch.where(dot < 0.0, -target_quat, target_quat)
    mixed = (1.0 - blend) * source_quat + blend * aligned_target
    return F.normalize(mixed, dim=1, eps=1e-6)

def _compute_distance_field(mask: Tensor, threshold: float = 0.1) -> Tensor:
    """Approximate Euclidean distance transform using a weighted grid Dijkstra."""
    mask_cpu = mask.detach().float().cpu()
    D, H, W = mask_cpu.shape
    outside = mask_cpu <= threshold
    outside_np = outside.numpy()

    if torch.all(~outside):
        # Entire volume is foreground; return zeros to avoid NaNs
        return torch.zeros_like(mask, dtype=torch.float32)

    dist = np.full((D, H, W), np.inf, dtype=np.float32)
    visited = np.zeros((D, H, W), dtype=bool)

    heap: List[Tuple[float, int, int, int]] = []
    outside_idx = np.argwhere(outside_np)
    for z, y, x in outside_idx:
        dist[z, y, x] = 0.0
        heapq.heappush(heap, (0.0, int(z), int(y), int(x)))

    offsets: List[Tuple[float, int, int, int]] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                cost = math.sqrt(dx * dx + dy * dy + dz * dz)
                offsets.append((cost, dz, dy, dx))

    while heap:
        current_dist, z, y, x = heapq.heappop(heap)
        if visited[z, y, x]:
            continue
        visited[z, y, x] = True

        for cost, dz, dy, dx in offsets:
            nz, ny, nx = z + dz, y + dy, x + dx
            if nz < 0 or nz >= D or ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            new_dist = current_dist + cost
            if new_dist < dist[nz, ny, nx]:
                dist[nz, ny, nx] = new_dist
                heapq.heappush(heap, (new_dist, nz, ny, nx))

    dist[outside_np] = 0.0
    return torch.from_numpy(dist).to(mask.device)


def _hash_indices(coords: Tensor, grid_size: Tuple[int, int, int]) -> Tensor:
    """Hash integer voxel coordinates for uniqueness filtering."""
    W = grid_size[2]
    H = grid_size[1]
    stride_y = W + 1
    stride_z = (H + 1) * stride_y
    return coords[:, 2] * stride_z + coords[:, 1] * stride_y + coords[:, 0]


def _enforce_cell_quota(
    coords: Tensor,
    grid_shape: Tuple[int, int, int],
    cell_size: int,
    max_per_cell: int,
    cell_counts: Dict[int, int],
) -> Tensor:
    """Return boolean mask keeping at most `max_per_cell` samples per coarse cell."""
    if coords.numel() == 0:
        return torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)

    cell_coords = torch.div(coords, cell_size, rounding_mode="floor").long()
    cell_coords[:, 0].clamp_(0, grid_shape[2] - 1)
    cell_coords[:, 1].clamp_(0, grid_shape[1] - 1)
    cell_coords[:, 2].clamp_(0, grid_shape[0] - 1)

    cell_keys = _hash_indices(cell_coords, grid_shape)
    keep = torch.zeros(cell_keys.shape[0], dtype=torch.bool)
    for idx, key in enumerate(cell_keys.detach().cpu().tolist()):
        count = cell_counts.get(key, 0)
        if count < max_per_cell:
            keep[idx] = True
            cell_counts[key] = count + 1
    return keep.to(coords.device)


def _sample_uniform_voxels(
    coords: Tensor,
    values: Tensor,
    distances: Tensor,
    n_points: int,
    grid_shape: Tuple[int, int, int],
    *,
    cell_size: int,
    max_per_cell: int,
    oversample_factor: float,
    max_attempts: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Sample voxels uniformly with a mild per-cell quota to limit duplicates."""
    if coords.shape[0] == 0:
        raise ValueError("No candidate voxels available for initialization.")

    device = coords.device
    selected_indices: List[Tensor] = []
    cell_counts: Dict[int, int] = {}
    total_kept = 0
    attempts = 0

    while total_kept < n_points and attempts < max_attempts:
        need = n_points - total_kept
        draw = max(int(math.ceil(need * oversample_factor)), need)
        draw = min(draw, coords.shape[0])
        rand_idx = torch.randint(0, coords.shape[0], (draw,), device=device)
        batch_coords = coords[rand_idx]
        keep_mask = _enforce_cell_quota(
            batch_coords, grid_shape, cell_size, max_per_cell, cell_counts
        )
        if keep_mask.any():
            kept_idx = rand_idx[keep_mask]
            selected_indices.append(kept_idx)
            total_kept += kept_idx.shape[0]
        attempts += 1

    if total_kept == 0:
        raise RuntimeError(
            "Failed to sample any voxels with the current mask threshold and dedup quota."
        )

    sampled_idx = torch.cat(selected_indices, dim=0)

    if sampled_idx.shape[0] < n_points:
        shortfall = n_points - sampled_idx.shape[0]
        extra_idx = torch.randint(0, coords.shape[0], (shortfall,), device=device)
        sampled_idx = torch.cat([sampled_idx, extra_idx], dim=0)

    if sampled_idx.shape[0] > n_points:
        shuffle = torch.randperm(sampled_idx.shape[0], device=device)[:n_points]
        sampled_idx = sampled_idx[shuffle]

    return coords[sampled_idx], values[sampled_idx], distances[sampled_idx]


def initialize_from_volume(
    mask_path: str,
    n_points: int = 5000,
    noise_std: float = 0.01,
    device: torch.device = torch.device("cuda"),
    mask_threshold: float = 0.05,
    volume_downscale_factor: Optional[int] = None,
    volume_storage_dtype: str = "fp32",
    disable_volume_overflow_guard: bool = False,
    voxel_size_override: Optional[Tensor] = None,
    init_scale_min_vox: float = 1.0,
    init_scale_max_vox: float = 3.0,
    dedup_cell_size: int = 2,
    dedup_max_per_cell: int = 4,
    oversample_factor: float = 2.5,
    max_sampling_attempts: int = 6,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Sample Gaussian seeds uniformly from mask voxels above a threshold."""

    downscale = int(volume_downscale_factor) if volume_downscale_factor is not None else 1
    loader = VolumeLoader(
        target_shape=None,
        device=device,
        downscale_factor=downscale,
        storage_dtype=str(volume_storage_dtype),
        enable_overflow_guard=not bool(disable_volume_overflow_guard),
    )
    sampling_volume = loader.load_volume(mask_path)
    sampling_volume = sampling_volume.to(device=device, dtype=torch.float32)

    D, H, W = sampling_volume.shape
    z, y, x = torch.meshgrid(
        torch.arange(D, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    coords = torch.stack([x, y, z], dim=-1).float()
    coords_flat = coords.reshape(-1, 3)
    volume_flat = sampling_volume.reshape(-1)

    if volume_flat.max() <= 0:
        raise ValueError("Sampling volume contains no positive entries.")

    distance_field = _compute_distance_field(
        sampling_volume, threshold=max(mask_threshold, 1e-4)
    )
    distance_flat = distance_field.reshape(-1)

    foreground_mask = volume_flat >= mask_threshold
    if not foreground_mask.any():
        foreground_mask = volume_flat > 0
    if not foreground_mask.any():
        fallback_k = min(volume_flat.numel(), max(n_points * 4, 2048))
        top_vals, top_idx = torch.topk(volume_flat, k=fallback_k)
        candidate_coords = coords_flat[top_idx]
        candidate_vals = top_vals
        candidate_dist = distance_flat[top_idx]
    else:
        candidate_coords = coords_flat[foreground_mask]
        candidate_vals = volume_flat[foreground_mask]
        candidate_dist = distance_flat[foreground_mask]

    grid_shape = (
        max(1, math.ceil(D / dedup_cell_size)),
        max(1, math.ceil(H / dedup_cell_size)),
        max(1, math.ceil(W / dedup_cell_size)),
    )

    sampled_coords, sampled_vals, sampled_dist = _sample_uniform_voxels(
        candidate_coords,
        candidate_vals,
        candidate_dist,
        n_points,
        grid_shape,
        cell_size=max(1, dedup_cell_size),
        max_per_cell=max(1, dedup_max_per_cell),
        oversample_factor=max(1.0, oversample_factor),
        max_attempts=max(1, max_sampling_attempts),
    )

    # Always place initial points continuously inside voxel cells so seeds are
    # not restricted to integer lattice centers.
    subvoxel = torch.rand_like(sampled_coords) - 0.5
    jittered = sampled_coords + subvoxel

    jitter_scale = max(noise_std, 0.0)
    if jitter_scale > 0:
        jitter = (torch.rand_like(sampled_coords) - 0.5) * (jitter_scale * 2.0)
        jittered = jittered + jitter
    jittered[:, 0].clamp_(0, W - 1)
    jittered[:, 1].clamp_(0, H - 1)
    jittered[:, 2].clamp_(0, D - 1)

    # Ensure jitter does not push samples outside the mask threshold.
    # This is important for float masks with soft boundaries.
    if mask_threshold is not None and candidate_coords.numel() > 0:
        threshold = float(mask_threshold)
        if threshold > 0.0:
            nearest = jittered.round().long()
            x_idx = nearest[:, 0].clamp(0, W - 1)
            y_idx = nearest[:, 1].clamp(0, H - 1)
            z_idx = nearest[:, 2].clamp(0, D - 1)
            valid = sampling_volume[z_idx, y_idx, x_idx] >= threshold

            if not bool(valid.all().item()):
                invalid = torch.nonzero(~valid, as_tuple=False).view(-1)
                attempts = 0
                while invalid.numel() > 0 and attempts < max(1, max_sampling_attempts):
                    need = int(invalid.numel())
                    draw = max(int(math.ceil(need * max(1.0, oversample_factor))), need)
                    draw = min(draw, candidate_coords.shape[0])
                    rand_idx = torch.randint(
                        0, candidate_coords.shape[0], (draw,), device=device
                    )
                    res_coords = candidate_coords[rand_idx]
                    res_vals = candidate_vals[rand_idx]
                    res_dist = candidate_dist[rand_idx]

                    # Keep replacement samples continuous within voxel cells.
                    res_coords = res_coords + (torch.rand_like(res_coords) - 0.5)

                    if jitter_scale > 0:
                        res_jitter = (torch.rand_like(res_coords) - 0.5) * (
                            jitter_scale * 2.0
                        )
                        res_coords = res_coords + res_jitter

                    res_coords[:, 0].clamp_(0, W - 1)
                    res_coords[:, 1].clamp_(0, H - 1)
                    res_coords[:, 2].clamp_(0, D - 1)

                    nearest_res = res_coords.round().long()
                    rx = nearest_res[:, 0].clamp(0, W - 1)
                    ry = nearest_res[:, 1].clamp(0, H - 1)
                    rz = nearest_res[:, 2].clamp(0, D - 1)
                    valid_res = sampling_volume[rz, ry, rx] >= threshold

                    if bool(valid_res.any().item()):
                        chosen = torch.nonzero(valid_res, as_tuple=False).view(-1)
                        take = min(need, int(chosen.numel()))
                        chosen = chosen[:take]
                        tgt = invalid[:take]
                        jittered[tgt] = res_coords[chosen]
                        sampled_vals[tgt] = res_vals[chosen]
                        sampled_dist[tgt] = res_dist[chosen]
                        invalid = invalid[take:]

                    attempts += 1

                # Final fallback: keep replacement points continuous inside in-mask voxels.
                if invalid.numel() > 0:
                    need = int(invalid.numel())
                    rand_idx = torch.randint(
                        0, candidate_coords.shape[0], (need,), device=device
                    )
                    fallback_coords = candidate_coords[rand_idx]
                    fallback_coords = fallback_coords + (
                        torch.rand_like(fallback_coords) - 0.5
                    )
                    if jitter_scale > 0:
                        fallback_jitter = (
                            torch.rand_like(fallback_coords) - 0.5
                        ) * (jitter_scale * 2.0)
                        fallback_coords = fallback_coords + fallback_jitter
                    fallback_coords[:, 0].clamp_(0, W - 1)
                    fallback_coords[:, 1].clamp_(0, H - 1)
                    fallback_coords[:, 2].clamp_(0, D - 1)
                    jittered[invalid] = fallback_coords
                    sampled_vals[invalid] = candidate_vals[rand_idx]
                    sampled_dist[invalid] = candidate_dist[rand_idx]

    scale_den = torch.tensor([W - 1, H - 1, D - 1], device=device).clamp_min(1)
    points = jittered / scale_den

    _, voxel_size = default_origin_and_spacing((D, H, W), device)
    voxel_sizes_xyz = voxel_size
    if voxel_size_override is not None:
        override = torch.as_tensor(voxel_size_override, device=device, dtype=torch.float32)
        if override.numel() == 1:
            override = override.view(1).repeat(3)
        if override.numel() == 3:
            voxel_sizes_xyz = override

    # Global initialization scale band (in voxel units).
    # Use an isotropic voxel reference (mean axis spacing) to avoid introducing
    # default axis-dependent elongation when volume dimensions differ by axis.
    min_vox = max(float(init_scale_min_vox), 1e-3)
    max_vox = max(float(init_scale_max_vox), min_vox)
    voxel_iso = voxel_sizes_xyz.mean().clamp_min(1e-12)
    scale_min = torch.full((1, 3), voxel_iso * min_vox, device=device, dtype=torch.float32)
    scale_max = torch.full((1, 3), voxel_iso * max_vox, device=device, dtype=torch.float32)
    u = torch.rand(points.shape[0], 1, device=device, dtype=torch.float32)
    scales = scale_min + u * (scale_max - scale_min)

    val_min = float(candidate_vals.min().item())
    val_max = float(candidate_vals.max().item())
    if val_max > val_min:
        norm_vals = (sampled_vals - val_min) / (val_max - val_min)
    else:
        norm_vals = torch.ones_like(sampled_vals)
    opacities = norm_vals.clamp(0.1, 1.0).unsqueeze(1)

    return points, scales, opacities

    # Initialize scales and opacities
    scales = torch.ones(len(points), 3, device=device) * 0.01
    opacities = torch.ones(len(points), 1, device=device)

    return points, scales, opacities


def _sample_structure_from_mask(
    points_normalized: Tensor,
    mask_volume: Tensor,
    mask_threshold: float,
    sigma_pre: float,
) -> Tuple[Tensor, Tensor]:
    """Sample Hessian-based quaternions/vesselness directly from a mask volume."""
    if mask_volume is None or mask_volume.numel() == 0:
        empty = torch.zeros(
            points_normalized.shape[0], 1, device=points_normalized.device
        )
        identity = torch.zeros(
            points_normalized.shape[0], 4, device=points_normalized.device
        )
        identity[:, 0] = 1.0
        return identity, empty

    origin, spacing = default_origin_and_spacing(
        mask_volume.shape, points_normalized.device
    )
    ijk = world_to_voxel(points_normalized, origin, spacing)
    return structure_from_mask_at_ijk(
        mask_volume,
        ijk,
        mask_threshold=float(mask_threshold),
        sigma_pre=float(sigma_pre),
    )


def transform_points_to_world(
    points: Tensor,
    volume_transform: Optional[Tensor] = None,
    scene_bounds: Optional[Tuple[Tensor, Tensor]] = None
) -> Tensor:
    """
    Transform points from volume space to world space.

    Args:
        points: Points in normalized volume space [0,1]^3 (shape [N, 3])
        volume_transform: Optional 4x4 transform matrix
        scene_bounds: Optional (min, max) scene bounds to scale into

    Returns:
        Points in world space (shape [N, 3])
    """
    device = points.device

    if scene_bounds is not None:
        min_bound, max_bound = scene_bounds
        # Ensure bounds are on same device as points
        min_bound = min_bound.to(device)
        max_bound = max_bound.to(device)
        scale = max_bound - min_bound
        points = points * scale + min_bound

    if volume_transform is not None:
        # Ensure transform is on same device as points
        volume_transform = volume_transform.to(device)
        # Add homogeneous coordinate
        points_h = torch.cat([points, torch.ones(len(points), 1, device=device)], dim=1)

        # Transform
        points = (volume_transform @ points_h.T).T[:, :3]

    return points


def _setup_model_parameters(
    model: GaussianModel,
    points: Tensor,
    scales: Tensor,
    opacities: Tensor,
    opacity_values: Optional[Tensor] = None,
    initial_rotations: Optional[Tensor] = None,
) -> None:
    """
    Set up core model parameters (positions, scales, rotations, opacities).

    Args:
        model: The Gaussian model to initialize
        points: Point positions [N, 3]
        scales: Scale values [N, 3]
        opacities: Default opacity values [N, 1]
        opacity_values: Optional volume-derived opacity values [N, 1]
    """
    # Keep trainable parameters in FP32 for stable AMP+GradScaler behavior.
    points = points.to(dtype=torch.float32)
    scales = scales.to(device=points.device, dtype=torch.float32)
    opacities = opacities.to(device=points.device, dtype=torch.float32)
    if opacity_values is not None:
        opacity_values = opacity_values.to(device=points.device, dtype=torch.float32)
    if initial_rotations is not None and initial_rotations.numel() != 0:
        initial_rotations = initial_rotations.to(
            device=points.device,
            dtype=torch.float32,
        )

    # Get shapes and device
    num_points = points.shape[0]
    device = points.device

    # Initialize all model tensors with proper nn.Parameters
    model._xyz = nn.Parameter(
        points.T.contiguous().requires_grad_(True)
    )  # Convert [N, 3] -> [3, N]
    model._initial_xyz = model._xyz.detach().clone()
    model._scaling = nn.Parameter(
        torch.log(scales).contiguous().requires_grad_(True)
    )  # [N, 3], model expects log-scales
    model._initial_scaling = (
        torch.log(scales).clone().detach()
    )  # Store initial scales for max size constraint

    # Initialize opacity based on whether we're using mask-buffered opacity or not.
    # Note: model stores opacity logits; convert probabilities -> logits via inverse_sigmoid.
    safe_prob = opacities.clamp(1e-6, 1.0 - 1e-6)
    opacity_logits = inverse_sigmoid(safe_prob)

    if opacity_values is not None:
        # Store non-learnable opacity values from the mask.
        model.opacities = opacity_values.detach().contiguous()
        model.opacities.requires_grad = False
        # Keep a non-trainable opacity parameter for compatibility with legacy code.
        model._opacity = nn.Parameter(opacity_logits.detach().contiguous().requires_grad_(False))
        print("Using non-learnable opacities from mask")
    else:
        # Use learnable opacity parameters.
        model._opacity = nn.Parameter(opacity_logits.contiguous().requires_grad_(True))

    # Initialize rotation quaternions
    if initial_rotations is not None and initial_rotations.numel() != 0:
        rotations = initial_rotations.to(device)
        rotations = rotations / (rotations.norm(dim=1, keepdim=True) + 1e-8)
    else:
        rotations = torch.zeros((num_points, 4), device=device)
        rotations[..., 0] = 1  # Identity quaternion
    model._rotation = nn.Parameter(rotations.contiguous().requires_grad_(True))

    # Initialize max 2D radii
    model.max_radii2D = torch.zeros(num_points, device=device)


def _setup_feature_tensors(
    model: GaussianModel,
    intensities: Tensor,
    volume_min: float,
    volume_max: float,
) -> None:
    """
    Set up feature tensors based on intensity values.

    Args:
        model: The Gaussian model to initialize
        intensities: Intensity values [N, 1]
        volume_min: Global minimum intensity value
        volume_max: Global maximum intensity value
    """
    intensities = intensities.to(dtype=torch.float32)
    num_points = intensities.shape[0]
    device = intensities.device

    # Store intensity values and volume range (not learnable parameters)
    model.intensities = intensities.detach().contiguous()
    model.intensities.requires_grad = False
    model.volume_min = volume_min
    model.volume_max = volume_max

    print(f"Initialized {num_points} Gaussians with intensity values")
    print(f"Stored volume min/max values: [{volume_min:.4f}, {volume_max:.4f}]")

    if getattr(model, "intensity_mode", "learned") in {
        "sampled",
        "sampled_mean_covered",
    }:
        # Disable SH features; rely entirely on sampled intensities
        model._features_dc = torch.zeros((num_points, 0, 3), device=device)
        model._features_rest = torch.zeros((num_points, 0, 3), device=device)
        return

    # Map intensity values to spherical harmonic coefficients for learnable color modes
    normalized_intensities = model._map_intensities_to_sh_coefficients(
        intensities, volume_min, volume_max
    )

    if torch.allclose(normalized_intensities, torch.zeros_like(normalized_intensities)):
        print(
            "Warning: Using default mid-gray intensities. Check if volume sampling worked correctly."
        )

    if model.max_sh_degree > 0:
        # Renderer expects SH features in [N, K, 3] split into dc/rest:
        # - _features_dc:   [N, 1, 3]
        # - _features_rest: [N, K-1, 3]
        sh_coeff_count = (model.max_sh_degree + 1) ** 2
        features_dc = normalized_intensities.expand(-1, 3).unsqueeze(1)
        features_rest = torch.zeros(
            (num_points, sh_coeff_count - 1, 3),
            device=device,
            dtype=features_dc.dtype,
        )
        model._features_dc = nn.Parameter(features_dc.contiguous().requires_grad_(True))
        model._features_rest = nn.Parameter(
            features_rest.contiguous().requires_grad_(True)
        )
    else:
        intensity_tensor = normalized_intensities.expand(-1, 3).unsqueeze(1)
        print(
            f"Creating feature_dc from normalized intensities: shape {intensity_tensor.shape}, "
            f"range [{intensity_tensor.min().item():.4f}, {intensity_tensor.max().item():.4f}]"
        )
        if num_points > 0:
            print(
                f"First 5 RGB values: {intensity_tensor[:min(5, num_points), 0, :].cpu().numpy()}"
            )
        model._features_dc = nn.Parameter(
            intensity_tensor.contiguous().requires_grad_(True)
        )
        model._features_rest = nn.Parameter(
            torch.zeros((num_points, 0, 3), device=device)
            .contiguous()
            .requires_grad_(True)
        )


def _is_valid_sampling(intensities: Tensor) -> bool:
    """
    Check if sampled intensities have valid range.

    Args:
        intensities: Sampled intensity values

    Returns:
        bool: True if intensities are valid, False otherwise
    """
    # Check common failure cases
    if (
        intensities.max() <= intensities.min()
        or torch.allclose(intensities, torch.full_like(intensities, 0.5))
        or (intensities.max() - intensities.min()) < 1e-4
    ):
        return False
    return True


def _sample_fallback_intensities(
    points: Tensor, volume: Tensor, device: torch.device
) -> Tuple[Tensor, float, float]:
    """
    Fallback method for sampling intensities directly from volume.

    Args:
        points: Point positions in normalized [0,1] coordinates
        volume: Volume tensor
        device: Torch device

    Returns:
        Tuple of:
            - Sampled intensities
            - Volume min value
            - Volume max value
    """
    D, H, W = volume.shape
    # Convert normalized points to indices
    point_indices = (points * torch.tensor([W - 1, H - 1, D - 1], device=device)).long()
    point_indices = torch.clamp(
        point_indices,
        min=torch.tensor([0, 0, 0], device=device),
        max=torch.tensor([W - 1, H - 1, D - 1], device=device),
    )

    # Get intensity values at nearest voxels
    x, y, z = point_indices[:, 0], point_indices[:, 1], point_indices[:, 2]
    direct_intensities = volume[z, y, x].unsqueeze(1)

    print(
        f"Raw intensity range: [{direct_intensities.min().item():.4f}, {direct_intensities.max().item():.4f}]"
    )

    # If direct sampling didn't work, try nonzero sampling
    if direct_intensities.max() <= 1e-4:
        print("Direct sampling failed, sampling from nonzero regions...")
        nonzero = torch.nonzero(volume > 1e-4, as_tuple=False)
        if len(nonzero) > 0:
            # Sample random points from nonzero regions
            indices = torch.randint(0, len(nonzero), (len(points),), device=device)
            sampled_points = nonzero[indices]
            # Get intensity values
            sampled_intensities = volume[
                sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2]
            ]
            direct_intensities = sampled_intensities.unsqueeze(1)

    # Get global min/max
    volume_min = float(volume.min().item())
    volume_max = float(volume.max().item())

    print(
        f"Updated intensity range: [{direct_intensities.min().item():.4f}, {direct_intensities.max().item():.4f}]"
    )
    print(f"Updated volume range: [{volume_min:.4f}, {volume_max:.4f}]")

    return direct_intensities, volume_min, volume_max


def initialize_gaussians(
    model: GaussianModel,
    n_points: int = 5000,
    volume_transform: Optional[Tensor] = None,
    scene_bounds: Optional[Tuple[Tensor, Tensor]] = None,
    volume_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    orientation_helper: Optional["VolumeSupervisor"] = None,
    **kwargs,
):
    """
    Initialize a Gaussian model from a volume or mask.

    Args:
        model: Gaussian model to initialize
        n_points: Number of points to sample
        volume_transform: Optional 4x4 transform matrix
        scene_bounds: Optional (min, max) scene bounds
        volume_path: Optional path to volume file
        mask_path: Optional path to mask file
        **kwargs: Additional args for initialize_from_volume
    """
    structure_mask_threshold = kwargs.pop("structure_mask_threshold", 0.1)
    structure_sigma = kwargs.pop("structure_sigma", 1.0)
    structure_min_vesselness = kwargs.pop("structure_min_vesselness", 0.2)
    anisotropy_strength = kwargs.pop("anisotropy_strength", 0.0)
    structure_orientation_strength = float(
        kwargs.pop("structure_orientation_strength", 0.0)
    )
    init_anisotropy_ratio = float(kwargs.pop("init_anisotropy_ratio", 1.0))
    border_distance_vox = float(kwargs.pop("border_distance_vox", 0.0))
    border_flatten_ratio = float(kwargs.pop("border_flatten_ratio", 1.0))
    border_grad_sigma = float(kwargs.pop("border_grad_sigma", 1.5))
    volume_downscale_factor = kwargs.pop("volume_downscale_factor", None)
    disable_volume_overflow_guard = bool(
        kwargs.pop("disable_volume_overflow_guard", False)
    )
    volume_storage_dtype = str(kwargs.pop("volume_storage_dtype", "fp32"))
    opacity_gamma = float(kwargs.pop("opacity_gamma", 1.0))
    opacity_mode = str(
        kwargs.pop("opacity_mode", getattr(model, "opacity_mode", "sampled"))
    )
    model.set_opacity_mode(opacity_mode)

    # Keep initialization sampling space consistent with the configured loader downscale.
    # This ensures the mask volume is downscaled the same way as the main input volume.
    init_sampling_downscale = volume_downscale_factor

    # Get points in volume space
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points, scales, opacities = initialize_from_volume(
        mask_path if mask_path else volume_path,
        n_points,
        device=device,
        volume_downscale_factor=init_sampling_downscale,
        volume_storage_dtype=volume_storage_dtype,
        disable_volume_overflow_guard=disable_volume_overflow_guard,
        voxel_size_override=(
            getattr(orientation_helper, "voxel_size", None)
            if orientation_helper is not None
            else None
        ),
        **kwargs,
    )
    volume_points = points.clone()

    if init_anisotropy_ratio > 1.0:
        # Make all Gaussians anisotropic at init time in their local frame.
        # This uses the axis-2 scale component as the "long" axis by convention.
        ratio = float(init_anisotropy_ratio)
        ratio = max(ratio, 1.0 + 1e-6)
        shrink = 1.0 / math.sqrt(ratio)
        scales[:, 2] = scales[:, 2] * ratio
        scales[:, 0] = scales[:, 0] * shrink
        scales[:, 1] = scales[:, 1] * shrink

    # Sample intensity values from the volume if available
    intensities = None
    mask_volume = None
    volume_min = 0.0
    volume_max = 1.0
    downscale = int(volume_downscale_factor) if volume_downscale_factor is not None else 1
    # Keep mask/opacity sampling aligned with the init sampling space.
    # Also load the intensity volume with the same downscale factor so voxel-count
    # safeguards are evaluated against the requested (downscaled) resolution.
    loader_mask = VolumeLoader(
        target_shape=None,
        device=device,
        downscale_factor=downscale,
        storage_dtype=volume_storage_dtype,
        enable_overflow_guard=not disable_volume_overflow_guard,
    )
    loader_intensity = VolumeLoader(
        target_shape=None,
        device=device,
        downscale_factor=downscale,
        storage_dtype=volume_storage_dtype,
        enable_overflow_guard=not disable_volume_overflow_guard,
    )

    # Load mask early for opacity sampling / structural initialization.
    if mask_path:
        mask_volume = loader_mask.load_volume(mask_path)

    # Load and sample intensities from volume if provided
    if volume_path:
        # Load volume for intensity sampling
        print(f"Loading intensity volume from: {volume_path}")
        volume = loader_intensity.load_volume(volume_path)
        global_min = float(volume.min().item())
        global_max = float(volume.max().item())

        mask_intensity_min = None
        mask_intensity_max = None
        if mask_volume is not None and mask_volume.numel() > 0:
            mask_threshold = float(kwargs.get("mask_threshold", 0.05))
            mask_bool = mask_volume >= max(mask_threshold, 1e-4)
            if not bool(mask_bool.any().item()):
                mask_bool = mask_volume > 0
            if bool(mask_bool.any().item()):
                masked_vals = volume[mask_bool]
                if masked_vals.numel() > 0:
                    mask_intensity_min = float(masked_vals.min().item())
                    mask_intensity_max = float(masked_vals.max().item())
                    print(
                        "Mask-bounded intensity range: "
                        f"[{mask_intensity_min:.4f}, {mask_intensity_max:.4f}]"
                    )

        # Only sampled intensity modes should store normalized [0,1] values.
        # In learned mode, keep raw intensities here and let SH conversion apply
        # the single intended normalization step via (min_ref, max_ref).
        normalize_samples = getattr(model, "intensity_mode", "learned") in {
            "sampled",
            "sampled_mean_covered",
        }
        min_ref = mask_intensity_min if mask_intensity_min is not None else global_min
        max_ref = mask_intensity_max if mask_intensity_max is not None else global_max

        # Sample intensities using the utility function
        print("Sampling intensity values from volume...")
        intensities, volume_min, volume_max = update_intensities(
            points,
            volume,
            scales,
            normalize=normalize_samples,
            min_val=min_ref,
            max_val=max_ref,
            padding_mode="border",
        )

        # Mask-aware correction: trilinear interpolation near boundaries can blend
        # outside-mask voxels into otherwise valid seeds and create very dark SH DC
        # values at iteration 1. Detect those cases and replace with nearest-voxel
        # samples from the intensity volume.
        if mask_volume is not None and mask_volume.numel() > 0 and intensities.numel() > 0:
            mask_threshold = float(kwargs.get("mask_threshold", 0.05))
            mask_samples, _, _ = sample_intensities_from_volume(
                points,
                mask_volume,
                scale=None,
                normalize=False,
                min_val=0.0,
                max_val=1.0,
                padding_mode="border",
            )
            outside_soft = mask_samples.view(-1) < max(mask_threshold, 1e-4)
            if bool(outside_soft.any().item()):
                Dv, Hv, Wv = volume.shape
                point_indices = (
                    points
                    * torch.tensor(
                        [Wv - 1, Hv - 1, Dv - 1],
                        device=device,
                        dtype=points.dtype,
                    )
                ).round().long()
                point_indices = torch.clamp(
                    point_indices,
                    min=torch.tensor([0, 0, 0], device=device),
                    max=torch.tensor([Wv - 1, Hv - 1, Dv - 1], device=device),
                )
                x_idx, y_idx, z_idx = (
                    point_indices[:, 0],
                    point_indices[:, 1],
                    point_indices[:, 2],
                )
                nearest_vals = volume[z_idx, y_idx, x_idx].unsqueeze(1)
                nearest_vals = nearest_vals.to(
                    device=intensities.device,
                    dtype=intensities.dtype,
                )

                if normalize_samples:
                    denom = max(max_ref - min_ref, 1e-8)
                    if denom <= 1e-8:
                        nearest_vals = torch.full_like(nearest_vals, 0.5)
                    else:
                        nearest_vals = (nearest_vals - min_ref) / denom
                        nearest_vals = nearest_vals.clamp_(0.0, 1.0)

                intensities = intensities.clone()
                intensities[outside_soft] = nearest_vals[outside_soft]
                print(
                    "Applied mask-boundary intensity correction to "
                    f"{int(outside_soft.sum().item())} initialized seeds."
                )

        # Check if sampling was successful
        if not _is_valid_sampling(intensities):
            print(
                "Warning: Invalid intensity range detected. Trying alternative sampling..."
            )

            # Try direct sampling at nearest voxels
            intensities, volume_min, volume_max = _sample_fallback_intensities(
                points, volume, device
            )

            if normalize_samples:
                denom = max(max_ref - min_ref, 1e-8)
                if denom <= 1e-8:
                    intensities = torch.full_like(intensities, 0.5)
                else:
                    intensities = (intensities - min_ref) / denom
                    intensities = intensities.clamp_(0.0, 1.0)
                volume_min = min_ref
                volume_max = max_ref
        elif normalize_samples:
            volume_min = min_ref
            volume_max = max_ref
        print(f"Final volume global range: [{volume_min:.4f}, {volume_max:.4f}]")
    else:
        # Default mid-gray if no volume is provided
        intensities = torch.full((points.shape[0], 1), 0.5, device=device)

    opacity_values = None
    if mask_volume is not None:
        # Sample opacity values from the mask
        print("Sampling opacity values from mask...")
        opacity_values, _, _ = sample_intensities_from_volume(
            points,
            mask_volume,
            scale=scales,
            normalize=False,
            min_val=0.0,
            max_val=1.0,
            padding_mode="border",
        )

        mask_min = float(mask_volume.min().item())
        mask_max = float(mask_volume.max().item())

        if opacity_gamma != 1.0 and opacity_values is not None:
            opacity_values = opacity_values.clamp(0.0, 1.0).pow(opacity_gamma)

        print(
            f"Opacity range: [{opacity_values.min().item():.4f}, {opacity_values.max().item():.4f}]"
        )
        print(f"Mask global range: [{mask_min:.4f}, {mask_max:.4f}]")

    # Decide how opacities are represented on the model.
    opacity_values_for_model = (
        opacity_values
        if opacity_mode in {"sampled", "sampled_mean_covered"}
        else None
    )

    # Transform to world space
    points = transform_points_to_world(points, volume_transform, scene_bounds)

    initial_rotations = None
    orientation_field = None
    fallback_count = 0
    structure_quats: Optional[Tensor] = None
    structure_vesselness: Optional[Tensor] = None
    if orientation_helper is not None:
        quats, fallback_count = orientation_helper.get_quat_for_points(points)
        initial_rotations = quats.detach()
        orientation_field = orientation_helper.export_orientation_field()
        print(
            f"Orientation initialized for {quats.shape[0]} points "
            f"(fallback {fallback_count})."
        )
        structure_quats, structure_vesselness = (
            orientation_helper.get_structure_for_points(points)
        )
    else:
        identity = torch.zeros(points.shape[0], 4, device=points.device)
        identity[:, 0] = 1.0
        initial_rotations = random_quat_perturb(identity, deg=2.0)
        fallback_count = points.shape[0]
        print(f"Orientation initialized without field (fallback {fallback_count}).")
        if mask_volume is not None:
            structure_quats, structure_vesselness = _sample_structure_from_mask(
                volume_points, mask_volume, structure_mask_threshold, structure_sigma
            )

    if structure_quats is not None and structure_vesselness is not None:
        vessel_vals = structure_vesselness.squeeze(1)
        active = vessel_vals >= structure_min_vesselness
        if active.any():
            vessel_strength = vessel_vals[active].clamp(0.0, 1.0).sqrt()
            stretch = 1.0 + anisotropy_strength * vessel_strength
            shrink = torch.clamp(1.0 / stretch, min=0.15)
            scales_active = scales[active]
            scales_active[:, 2] = scales_active[:, 2] * stretch
            scales_active[:, 0] = scales_active[:, 0] * shrink
            scales_active[:, 1] = scales_active[:, 1] * shrink
            scales[active] = scales_active
            # Blend in Hessian orientations when requested so strong vessel cues
            # can sharpen the principal axis instead of only changing scales.
            if orientation_helper is None:
                initial_rotations[active] = structure_quats[active]
            elif structure_orientation_strength > 0.0:
                blend_weight = (
                    float(structure_orientation_strength) * vessel_strength
                ).clamp(0.0, 1.0)
                initial_rotations[active] = _blend_quaternions(
                    initial_rotations[active],
                    structure_quats[active],
                    blend_weight,
                )

            orientation_note = ""
            if orientation_helper is None:
                orientation_note = "; used Hessian orientations"
            elif structure_orientation_strength > 0.0:
                orientation_note = (
                    "; blended Hessian orientations "
                    f"(strength={structure_orientation_strength:.2f})"
                )
            print(
                f"Applied Hessian anisotropy to {active.sum().item()} seeds "
                f"(threshold={structure_min_vesselness:.2f})"
                f"{orientation_note}."
            )

    # Border splats: align directly to the continuously sampled mask-gradient
    # normal, then flatten along that normal (init-only). Using rounded-voxel
    # Hessian directions here made the voxel lattice visible on smooth surfaces.
    enable_border = (
        mask_volume is not None
        and border_distance_vox > 0.0
        and border_flatten_ratio > 1.0
        and volume_points.numel() != 0
    )
    if enable_border:
        with torch.no_grad():
            init_mask_threshold = float(kwargs.get("mask_threshold", structure_mask_threshold))
            origin, spacing = default_origin_and_spacing(
                mask_volume.shape, volume_points.device
            )
            ijk = world_to_voxel(volume_points, origin, spacing)
            ijk_round = ijk.round().long()
            D, H, W = mask_volume.shape
            ijk_round[:, 0].clamp_(0, D - 1)
            ijk_round[:, 1].clamp_(0, H - 1)
            ijk_round[:, 2].clamp_(0, W - 1)

            dist_field = _compute_distance_field(
                mask_volume, threshold=max(float(init_mask_threshold), 1e-4)
            )
            dist_at = dist_field[
                ijk_round[:, 0], ijk_round[:, 1], ijk_round[:, 2]
            ]
            border_mask = dist_at <= float(border_distance_vox)

            if border_mask.any():
                grad_field, mag_field = compute_gradient_field(
                    mask_volume, sigma_pre=float(border_grad_sigma)
                )
                rot_g, fallback_g = gather_rotation_from_gradient(
                    grad_field, mag_field, ijk
                )
                normal_dir = rot_g[:, :, 2]

                # Skip overrides where the gradient field fell back to identity.
                border_idx = border_mask.nonzero(as_tuple=False).squeeze(1)
                good = ~fallback_g[border_idx]
                if good.any():
                    border_normals = normal_dir[border_idx[good]]
                    quats_border, _ = quat_from_directions(border_normals)
                    quats_border = quats_border.to(
                        device=initial_rotations.device,
                        dtype=initial_rotations.dtype,
                    )
                    initial_rotations[border_idx[good]] = quats_border

                    # Flatten along local axis-2 (normal) and expand tangential axes.
                    ratio = float(border_flatten_ratio)
                    ratio = max(ratio, 1.0 + 1e-6)
                    tangential = math.sqrt(ratio)
                    idx_good = border_idx[good]
                    scales[idx_good, 2] = scales[idx_good, 2] / ratio
                    scales[idx_good, 0] = scales[idx_good, 0] * tangential
                    scales[idx_good, 1] = scales[idx_good, 1] * tangential

                    print(
                        f"Applied border normal alignment + flattening to {idx_good.numel()} seeds "
                        f"(dist<= {border_distance_vox:.2f} vox, ratio={border_flatten_ratio:.2f})."
                    )

    # Set up model parameters and feature tensors
    opacity_param_init = opacity_values if opacity_values is not None else opacities
    _setup_model_parameters(
        model,
        points,
        scales,
        opacity_param_init,
        opacity_values_for_model,
        initial_rotations,
    )
    _setup_feature_tensors(model, intensities, volume_min, volume_max)

    # Cache orientation data for densification if available
    model.orientation_field = orientation_field

    return model
