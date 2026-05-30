import os
import math
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from gaussian_splatting.utils.orientation_field import default_origin_and_spacing

# NOTE: Refactored to (1) minimize Python loops, (2) actually use rotation by
# constructing covariances, and (3) avoid reallocation / unnecessary empty_cache calls.


_GRID_CACHE: dict[tuple, Tensor] = {}


def _sparse_support_sigma_from_cutoff(support_cutoff: float) -> float:
    """Convert a kernel cutoff into an equivalent sigma-space radius."""
    cutoff = min(max(float(support_cutoff), 1e-8), 0.999999)
    return math.sqrt(-2.0 * math.log(cutoff))


def _sparse_support_gate(
    sq_mahalanobis: Tensor,
    support_sigma: float,
    support_softness: float,
) -> Tensor:
    """Return a smooth support gate for sparse splat truncation."""
    if support_softness <= 0.0:
        return (sq_mahalanobis <= (support_sigma * support_sigma)).to(
            dtype=sq_mahalanobis.dtype
        )

    sigma_distance = torch.sqrt(sq_mahalanobis.clamp_min(0.0) + 1e-6)
    softness = max(float(support_softness), 1e-6)
    return torch.sigmoid((support_sigma - sigma_distance) / softness)


def _grid_bounds_cache_key(
    grid_bounds: Optional[Tuple[Tensor, Tensor]],
    *,
    precision: float = 1e-6,
) -> Optional[tuple[int, int, int, int, int, int]]:
    """Build a stable, hashable key for grid bounds.

    Rounds to a fixed precision to avoid cache misses from tiny float noise.
    """
    if grid_bounds is None:
        return None
    bounds_min, bounds_max = grid_bounds
    scale = 1.0 / max(float(precision), 1e-12)
    bmin = bounds_min.detach().to(dtype=torch.float32)
    bmax = bounds_max.detach().to(dtype=torch.float32)
    vals = torch.cat([bmin, bmax], dim=0)
    # Convert to python ints.
    return tuple(int(round(float(v.item()) * scale)) for v in vals)


def create_grid_points(
    volume_shape: Tuple[int, int, int],
    device: torch.device,
    grid_bounds: Optional[Tuple[Tensor, Tensor]] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Create a grid of 3D points for volume rendering.
    
    Args:
        volume_shape: (depth, height, width) of output volume
        device: Device to create tensors on
    
    Returns:
        Grid points tensor (D, H, W, 3)
    """
    D, H, W = volume_shape

    cache_key = (
        volume_shape,
        _grid_bounds_cache_key(grid_bounds),
        str(device),
        str(dtype),
    )
    cached = _GRID_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if grid_bounds is None:
        bounds_min = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
        bounds_max = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
    else:
        bounds_min, bounds_max = grid_bounds
        bounds_min = bounds_min.to(device=device, dtype=dtype)
        bounds_max = bounds_max.to(device=device, dtype=dtype)

    # Create normalized coordinate grid for the requested bounds.
    z = torch.linspace(
        float(bounds_min[2]), float(bounds_max[2]), D, device=device, dtype=dtype
    )
    y = torch.linspace(
        float(bounds_min[1]), float(bounds_max[1]), H, device=device, dtype=dtype
    )
    x = torch.linspace(
        float(bounds_min[0]), float(bounds_max[0]), W, device=device, dtype=dtype
    )

    # Create meshgrid
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')

    # Stack coordinates
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    _GRID_CACHE[cache_key] = grid
    return grid


def gaussian_kernel_3d(
    points: Tensor,
    means: Tensor,
    covs: Optional[Tensor] = None,
    scale: float = 1.0,
    scaling: Optional[Tensor] = None  # Add scaling parameter
) -> Tensor:
    """
    Compute batched 3D Gaussian kernel values for multiple centers.
    
    Args:
        points: Grid points (D, H, W, 3)
        means: Gaussian centers (N, 3)
        covs: Optional covariance matrices (N, 3, 3)  
        scale: Scale factor for isotropic Gaussian
        scaling: Optional per-point scaling factors (N, 3)
    
    Returns:
        Combined kernel values at each point (D, H, W)
    """
    D, H, W = points.shape[:3]
    N = means.shape[0]

    # Reshape points for broadcasting against means
    # points: (D, H, W, 3) -> (D, H, W, 1, 3)
    points_exp = points.unsqueeze(3)

    # Reshape means for broadcasting against points
    # means: (N, 3) -> (1, 1, 1, N, 3)
    means_exp = means.reshape(1, 1, 1, N, 3)

    # Compute differences for all points and means at once
    # diff: (D, H, W, N, 3)
    diff = points_exp - means_exp

    if covs is None and scaling is None:
        # Use isotropic Gaussian for all centers
        sq_dist = torch.sum(diff * diff, dim=-1)  # (D, H, W, N)
        kernels = torch.exp(-0.5 * sq_dist / (scale ** 2))  # (D, H, W, N)
    elif scaling is not None:
        # Use anisotropic Gaussian with per-axis scaling
        # Reshape scaling for broadcasting
        scaling_exp = scaling.reshape(1, 1, 1, N, 3)
        # Apply per-axis scaling
        scaled_diff = diff / (scaling_exp + 1e-6)
        sq_dist = torch.sum(scaled_diff * scaled_diff, dim=-1)  # (D, H, W, N)
        kernels = torch.exp(-0.5 * sq_dist)
    else:
        # Use full covariance matrices
        # covs: (N, 3, 3)
        # Compute inverse of each covariance matrix
        cov_invs = torch.inverse(covs)  # (N, 3, 3)

        # Expand covs for broadcasting
        # (N, 3, 3) -> (1, 1, 1, N, 3, 3)
        cov_invs = cov_invs.reshape(1, 1, 1, N, 3, 3)

        # Reshape diff for matrix multiplication
        # (D, H, W, N, 3) -> (D, H, W, N, 1, 3)
        diff_exp = diff.unsqueeze(-2)

        # Compute mahalanobis distance
        # (D, H, W, N)
        mahalanobis = torch.sum(
            (diff_exp @ cov_invs) * diff_exp.transpose(-2, -1),
            dim=(-2, -1)
        )
        kernels = torch.exp(-0.5 * mahalanobis)

    # Sum contributions from all Gaussians
    # (D, H, W)
    return kernels.sum(dim=-1)


def splat_to_volume(
    points: Tensor,
    point_scales: Optional[Tensor] = None,
    point_rotations: Optional[Tensor] = None,
    point_opacities: Optional[Tensor] = None,
    point_intensities: Optional[Tensor] = None,
    volume_shape: Tuple[int, int, int] = (64, 64, 64),
    covariances: Optional[Tensor] = None,
    scale: float = 0.1,
    batch_size: int = 50,  # Process points in batches to save memory
    device: Optional[torch.device] = None,
    active_idx: Optional[Tensor] = None,
    grid_bounds: Optional[Tuple[Tensor, Tensor]] = None,
    render_mode: str = "intensity",
    density_scale: float = 1.0,
    working_grid_downscale_factor: int = 2,
    sparse_support_cutoff: float = 0.2,
    sparse_max_radius_vox: int = 10,
    sparse_support_softness: float = 0.75,
    render_min_sigma_vox: float = 0.35,
) -> Tensor:
    """
    Convert 3D Gaussian splats to a volumetric representation.

    Args:
        points: Tensor of point centers (3, N) or (N, 3)
        point_scales: Optional per-point scaling factors (N, 3) or (N,)
        point_rotations: Optional per-point rotation quaternions (N, 4)
        point_opacities: Optional per-point opacity values (N, 1) or (N,)
        point_intensities: Optional per-point intensity values (N, 1) or (N,)
        volume_shape: Output volume shape (depth, height, width)
        covariances: Optional covariance matrices (N, 3, 3)
        scale: Default scale factor for isotropic Gaussians when point_scales not provided
        batch_size: Number of points to process at once to manage memory
        device: Device to use for computation

    Returns:
        Volume tensor (D, H, W)
    """
    device = points.device if device is None else device

    def _index_param(t: Optional[Tensor]) -> Optional[Tensor]:
        if t is None or active_idx is None:
            return t
        if t.dim() == 2 and t.shape[0] == 3 and t.shape[1] != 3:
            return torch.index_select(t, 1, active_idx)
        return torch.index_select(t, 0, active_idx)

    points = _index_param(points)
    point_scales = _index_param(point_scales)
    point_rotations = _index_param(point_rotations)
    point_opacities = _index_param(point_opacities)
    point_intensities = _index_param(point_intensities)
    if covariances is not None and active_idx is not None:
        covariances = torch.index_select(covariances, 0, active_idx)

    # Lightweight debug (can be silenced by setting ENV var later)
    if torch.is_grad_enabled() and points.grad_fn is None:
        pass  # avoid noisy prints in tight loops

    # Check if input requires gradients - if not, force it to require gradients
    if not points.requires_grad:
        print(
            "WARNING: Input tensor does not require gradients - forcing requires_grad=True"
        )
        # Create a differentiable copy
        points = points.clone().detach().requires_grad_(True)

    # Handle different input formats WITHOUT detaching - we need to keep the computation graph
    if points.shape[0] == 3 and points.shape[1] != 3:
        # Convert from (3, N) to (N, 3) without breaking gradient chain
        points_n3 = points.permute(
            1, 0
        )  # Use permute instead of T to maintain gradient connections
    else:
        # Keep the tensor as is
        points_n3 = points

    accum_dtype = torch.float32
    compute_dtype = accum_dtype
    if device.type == "cuda" and torch.is_autocast_enabled():
        compute_dtype = torch.get_autocast_gpu_dtype()

    points_n3 = points_n3.to(compute_dtype)
    if point_scales is not None:
        point_scales = point_scales.to(compute_dtype)
    if point_opacities is not None:
        point_opacities = point_opacities.to(compute_dtype)
    if point_intensities is not None:
        point_intensities = point_intensities.to(compute_dtype)
    if point_rotations is not None:
        point_rotations = point_rotations.to(compute_dtype)
    if covariances is not None:
        covariances = covariances.to(compute_dtype)

    total_points = points_n3.shape[0]
    # Avoid verbose printing here for performance

    # Create volume grid - these don't need gradients
    grid_points = create_grid_points(
        volume_shape,
        device,
        grid_bounds=grid_bounds,
        dtype=compute_dtype,
    )

    # Allocate final volume (grad will flow through ops populating it)
    volume = torch.zeros(volume_shape, device=device, dtype=accum_dtype)

    # Process splats in batches to save memory
    batch_size = min(batch_size, 100)  # Increased batch size for better performance
    num_batches = (total_points + batch_size - 1) // batch_size

    working_factor = max(1, int(working_grid_downscale_factor))

    # Optionally use a smaller working grid for memory efficiency when we have many points.
    # Set working_factor=1 to force full-resolution rasterization.
    if total_points > 1000 and working_factor > 1:
        small_shape = tuple(max(16, d // working_factor) for d in volume_shape)
        small_grid_points = create_grid_points(
            small_shape,
            device,
            grid_bounds=grid_bounds,
            dtype=compute_dtype,
        )
    else:
        small_shape = volume_shape
        small_grid_points = grid_points

    # Create a small working volume for accumulating results
    small_volume = torch.zeros(small_shape, device=device, dtype=accum_dtype)
    weight_volume = torch.zeros_like(small_volume)

    # Compute native and working-grid voxel spacing in normalized coordinates.
    if grid_bounds is None:
        native_voxel_spacing = default_origin_and_spacing(volume_shape, device)[1].to(
            accum_dtype
        )
    else:
        bounds_min, bounds_max = grid_bounds
        bounds_min = bounds_min.to(device=device, dtype=accum_dtype)
        bounds_max = bounds_max.to(device=device, dtype=accum_dtype)
        dims_xyz = torch.tensor(
            [volume_shape[2], volume_shape[1], volume_shape[0]],
            device=device,
            dtype=accum_dtype,
        ).clamp_min(1)
        denom = (dims_xyz - 1.0).clamp_min(1.0)
        native_voxel_spacing = (bounds_max - bounds_min) / denom
    if small_shape == volume_shape:
        scale_ratio = torch.ones(3, device=device, dtype=points_n3.dtype)
    else:
        scale_ratio = torch.tensor(
            [
                volume_shape[2] / small_shape[2],
                volume_shape[1] / small_shape[1],
                volume_shape[0] / small_shape[0],
            ],
            device=device,
            dtype=accum_dtype,
        )
    working_voxel_spacing = native_voxel_spacing * scale_ratio
    min_sigma = working_voxel_spacing * max(float(render_min_sigma_vox), 0.0)
    min_sigma = min_sigma.to(accum_dtype)
    min_sigma_broadcast = min_sigma.unsqueeze(0)

    def _splat_sparse_batch(
        *,
        bp: Tensor,
        scales_batch: Tensor,
        rb: Optional[Tensor],
        alpha: Tensor,
        value_scale: Tensor,
        out_shape: Tuple[int, int, int],
        bounds_min: Tensor,
        bounds_max: Tensor,
        voxel_spacing_local: Tensor,
        render_mode_local: str,
        density_scale_local: float,
        support_cutoff: float = 0.2,
        max_radius_vox: int = 10,
        support_softness: float = 0.75,
    ) -> Tuple[Optional[Tensor], Tensor]:
        """Sparse splat into flat buffers for one batch.

        Returns (contrib_flat or None, weight_flat), both shaped [G].
        """
        D, H, W = out_shape
        G = D * H * W

        # Convert batch point positions to voxel-space coordinates in this grid.
        denom = torch.tensor(
            [max(W - 1, 1), max(H - 1, 1), max(D - 1, 1)],
            device=device,
            dtype=bp.dtype,
        )
        extent = (bounds_max - bounds_min).clamp_min(1e-8)
        centers_vox = (bp - bounds_min.unsqueeze(0)) / extent.unsqueeze(0) * denom

        alpha = alpha.view(-1)
        value_scale = value_scale.view(-1)

        # Adaptive support: define a finite neighborhood from the requested
        # kernel cutoff, then soften the support boundary to reduce lattice-like
        # clipping artifacts.
        support_sigma = _sparse_support_sigma_from_cutoff(support_cutoff)
        support_margin = max(float(support_softness), 0.0) * 4.0
        candidate_sigma = support_sigma + support_margin

        sigma_vox_axes = scales_batch / voxel_spacing_local.unsqueeze(0).clamp_min(1e-8)
        sigma_vox_max = sigma_vox_axes.max(dim=1).values
        radii = torch.ceil(candidate_sigma * sigma_vox_max).to(torch.long).clamp_min(1)
        R = int(radii.max().item()) if radii.numel() > 0 else 0
        if R <= 0:
            if render_mode_local == "intensity":
                return torch.zeros(G, device=device, dtype=accum_dtype), torch.zeros(
                    G, device=device, dtype=accum_dtype
                )
            return None, torch.zeros(G, device=device, dtype=accum_dtype)

        # Keep sparse mode bounded.
        # IMPORTANT: Falling back to the dense path can build a massive autograd graph
        # (grid_chunk loops over the entire ROI grid), which can OOM during backward/
        # checkpoint recompute even for small volumes.
        # Instead, cap the neighborhood radius and effectively truncate large splats.
        if R > max_radius_vox:
            radii = radii.clamp_max(max_radius_vox)
            R = max_radius_vox

        offset_vals = torch.arange(-R, R + 1, device=device, dtype=torch.long)
        offsets = torch.stack(
            torch.meshgrid(offset_vals, offset_vals, offset_vals, indexing="ij"),
            dim=-1,
        ).view(-1, 3)
        K = offsets.shape[0]

        # Build voxel coordinates (B,K,3) around each center.
        diff_vox = offsets.to(device=device, dtype=bp.dtype).unsqueeze(0)
        diff_vox = diff_vox.expand(centers_vox.shape[0], -1, -1)  # (B,K,3)

        # Per-point radius mask in voxel space (conservative prefilter).
        within = diff_vox.abs().max(dim=-1).values <= radii.to(device=device).unsqueeze(1).to(diff_vox.dtype)

        # Absolute voxel coords in (x,y,z) indexing.
        base = torch.floor(centers_vox).to(torch.long)
        vox = base.unsqueeze(1) + offsets.unsqueeze(0)
        x = vox[..., 0]
        y = vox[..., 1]
        z = vox[..., 2]
        valid = (
            (x >= 0)
            & (x < W)
            & (y >= 0)
            & (y < H)
            & (z >= 0)
            & (z < D)
            & within
        )

        if not valid.any():
            if render_mode_local == "intensity":
                return torch.zeros(G, device=device, dtype=accum_dtype), torch.zeros(
                    G, device=device, dtype=accum_dtype
                )
            return None, torch.zeros(G, device=device, dtype=accum_dtype)

        # Compute continuous diffs for kernel evaluation.
        # IMPORTANT: using only integer voxel offsets would make the kernel
        # independent of bp (center positions), yielding near-zero xyz gradients.
        denom_f = denom.to(device=device, dtype=bp.dtype).clamp_min(1.0)
        extent_f = extent.to(device=device, dtype=bp.dtype)
        bmin_f = bounds_min.to(device=device, dtype=bp.dtype).unsqueeze(0).unsqueeze(0)
        vox_f = vox.to(device=device, dtype=bp.dtype)
        grid_pos = bmin_f + (vox_f / denom_f.view(1, 1, 3)) * extent_f.view(1, 1, 3)
        diff_norm = grid_pos - bp.unsqueeze(1)

        # Anisotropic kernel in local Gaussian frame.
        if rb is not None:
            diff_local = torch.einsum("bkj,bji->bki", diff_norm, rb)
        else:
            diff_local = diff_norm
        inv_scales = 1.0 / scales_batch.clamp_min(1e-6)
        diff_scaled = diff_local * inv_scales.unsqueeze(1)
        sq = (diff_scaled * diff_scaled).sum(dim=-1)
        kern = torch.exp(-0.5 * sq)
        support_gate = _sparse_support_gate(
            sq,
            support_sigma=support_sigma,
            support_softness=support_softness,
        )
        kern = kern * support_gate
        valid = valid & (support_gate > 1e-4)

        if not valid.any():
            if render_mode_local == "intensity":
                return torch.zeros(G, device=device, dtype=accum_dtype), torch.zeros(
                    G, device=device, dtype=accum_dtype
                )
            return None, torch.zeros(G, device=device, dtype=accum_dtype)

        # Flatten valid contributions.
        idx_lin = (z * (H * W) + y * W + x)
        idx_lin = idx_lin[valid].view(-1)
        w_vals = (kern * alpha.view(-1, 1))[valid].to(accum_dtype).view(-1)

        weight_flat = torch.zeros(G, device=device, dtype=accum_dtype)
        weight_flat.index_add_(0, idx_lin, w_vals)

        if render_mode_local == "intensity":
            c_vals = (kern * (alpha * value_scale).view(-1, 1))[valid].to(
                accum_dtype
            ).view(-1)
            contrib_flat = torch.zeros(G, device=device, dtype=accum_dtype)
            contrib_flat.index_add_(0, idx_lin, c_vals)
            return contrib_flat, weight_flat

        return None, weight_flat

    # Handle scaling parameters
    # point_scales, point_opacities, point_intensities already passed as parameters

    if render_mode not in {"intensity", "density"}:
        raise ValueError(
            "render_mode must be one of {'intensity','density'}, got "
            f"{render_mode!r}."
        )

    # Use default intensity values if not provided.
    # (Not needed in density mode.)
    if render_mode == "intensity" and point_intensities is None:
        point_intensities = torch.ones(total_points, device=device, dtype=accum_dtype)

    # Use torch.cuda.empty_cache() to clear memory periodically
    # if device.type == 'cuda':
    #     torch.cuda.empty_cache()

    # Vectorized accumulation over batches.
    # Prepare rotation -> covariance if provided (quaternions expected normalized).
    def quat_to_rotmat(q: Tensor) -> Tensor:
        # q: (B,4) (w,x,y,z) or (x,y,z,w); assume either – normalize then compute matrix
        if q.shape[-1] != 4:
            raise ValueError("Quaternion tensor must have shape (N,4)")
        # Heuristic: if mean(abs(q[...,0])) < mean(abs(q[..., -1])) swap ordering; keep simple
        # We won't modify ordering aggressively; assume (N,4) already in proper order matching training code
        q = F.normalize(q, dim=-1)
        w, x, y, z = q.unbind(-1)
        B = q.shape[0]
        R = torch.empty(B, 3, 3, device=q.device, dtype=q.dtype)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)
        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - x * w)
        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R

    if point_rotations is not None and point_rotations.numel() > 0:
        rot_mats = quat_to_rotmat(point_rotations)
    else:
        rot_mats = None

    # Pre-normalize optional per-point vectors to 1D for consistent slicing
    if point_opacities is not None:
        if point_opacities.ndim == 2:
            point_opacities = point_opacities.view(-1)
        elif point_opacities.ndim > 2:
            point_opacities = point_opacities.view(point_opacities.shape[0], -1)[:, 0]

    if point_intensities is not None and render_mode == "intensity":
        if point_intensities.ndim == 2:
            point_intensities = point_intensities.view(-1)
        elif point_intensities.ndim > 2:
            point_intensities = point_intensities.view(point_intensities.shape[0], -1)[
                :, 0
            ]

    # Flatten grid and chunk to limit memory.
    work_grid = small_grid_points.view(-1, 3)
    G = work_grid.shape[0]
    # Lower chunk size to reduce peak memory (important under checkpoint recompute).
    grid_chunk = 8192

    # Precompute sparse-mode ROI bounds and voxel spacing for the working grid.
    if grid_bounds is None:
        sparse_bounds_min = torch.zeros(3, device=device, dtype=accum_dtype)
        sparse_bounds_max = torch.ones(3, device=device, dtype=accum_dtype)
    else:
        bmin, bmax = grid_bounds
        sparse_bounds_min = bmin.to(device=device, dtype=accum_dtype)
        sparse_bounds_max = bmax.to(device=device, dtype=accum_dtype)

    dims_xyz = torch.tensor(
        [small_shape[2], small_shape[1], small_shape[0]],
        device=device,
        dtype=accum_dtype,
    ).clamp_min(1)
    denom = (dims_xyz - 1.0).clamp_min(1.0)
    sparse_voxel_spacing = (sparse_bounds_max - sparse_bounds_min) / denom

    for i in range(num_batches):
        s = i * batch_size
        e = min((i + 1) * batch_size, total_points)
        bp = points_n3[s:e]
        Bcur = bp.shape[0]
        if Bcur == 0:
            continue

        # Scales to (B,3)
        if point_scales is None:
            scales_batch = torch.full((Bcur, 3), scale, device=device, dtype=bp.dtype)
        else:
            sb = point_scales[s:e]
            if sb.ndim == 2 and sb.shape[1] == 3:
                scales_batch = sb
            else:
                scales_batch = sb.view(-1, 1).repeat(1, 3)
        # Keep splats from collapsing below voxel resolution while preserving gradients
        below_min = scales_batch < min_sigma_broadcast
        if below_min.any():
            scales_batch = torch.where(
                below_min,
                min_sigma_broadcast
                + (scales_batch - scales_batch.detach()),
                scales_batch,
            )

        if rot_mats is not None:
            rb = rot_mats[s:e]  # (B,3,3)
        else:
            rb = None

        if point_opacities is not None:
            alpha = point_opacities[s:e]
        else:
            alpha = torch.ones(Bcur, device=device, dtype=bp.dtype)

        if point_intensities is not None and render_mode == "intensity":
            value_scale = point_intensities[s:e]
        else:
            value_scale = torch.ones(Bcur, device=device, dtype=bp.dtype)

        # Prefer sparse splatting.
        # The dense fallback can be prohibitively expensive in memory during backward.
        # Env overrides are only intended for tests/debug.
        force_sparse = os.environ.get("GS_FORCE_SPARSE", "0") == "1"
        disable_sparse = os.environ.get("GS_DISABLE_SPARSE", "0") == "1"
        use_sparse = True
        if disable_sparse:
            use_sparse = False
        elif force_sparse:
            use_sparse = True
        # Note: Large splats are handled by radius capping inside _splat_sparse_batch.

        if use_sparse:
            try:
                contrib_flat, weight_flat = _splat_sparse_batch(
                    bp=bp,
                    scales_batch=scales_batch,
                    rb=rb,
                    alpha=alpha,
                    value_scale=value_scale,
                    out_shape=small_shape,
                    bounds_min=sparse_bounds_min,
                    bounds_max=sparse_bounds_max,
                    voxel_spacing_local=sparse_voxel_spacing,
                    render_mode_local=render_mode,
                    density_scale_local=density_scale,
                    support_cutoff=float(sparse_support_cutoff),
                    max_radius_vox=max(1, int(sparse_max_radius_vox)),
                    support_softness=float(sparse_support_softness),
                )
            except RuntimeError as exc:
                # Unexpected sparse failure; fall back to dense.
                if str(exc) != "sparse_radius_too_large":
                    raise
                use_sparse = False

        if not use_sparse:
            inv_scales = 1.0 / (scales_batch + 1e-6)  # (B,3)
            contrib_flat = None
            if render_mode == "intensity":
                contrib_flat = torch.zeros(G, device=device, dtype=accum_dtype)
            weight_flat = torch.zeros(G, device=device, dtype=accum_dtype)

            for g0 in range(0, G, grid_chunk):
                g1 = min(g0 + grid_chunk, G)
                grid_chunk_pts = work_grid[g0:g1]  # (Cg,3)
                diff = grid_chunk_pts.unsqueeze(1) - bp.unsqueeze(0)  # (Cg,B,3)
                if rb is not None:
                    # Vectorized rotation: for each b, apply diff[:, b, :] @ rb[b].T.
                    # einsum uses rb indices (b, j, i) to represent transpose.
                    diff_local = torch.einsum("gbi,bji->gbj", diff, rb)
                else:
                    diff_local = diff
                diff_scaled = diff_local * inv_scales.unsqueeze(0)  # (Cg,B,3)
                support_mask = (diff_scaled.abs() <= 3.0).all(dim=-1)
                sq = (diff_scaled * diff_scaled).sum(-1)  # (Cg,B)
                sq = sq.masked_fill(~support_mask, 36.0)
                kern = torch.exp(-0.5 * sq)
                weight_contrib = kern * alpha.unsqueeze(0)
                weight_flat[g0:g1] += weight_contrib.to(accum_dtype).sum(dim=1)

                if render_mode == "intensity":
                    value_contrib = kern * (alpha * value_scale).unsqueeze(0)
                    contrib_flat[g0:g1] += value_contrib.to(accum_dtype).sum(dim=1)
                    del value_contrib

                del diff, diff_local, diff_scaled, sq, kern, weight_contrib

        if render_mode == "intensity":
            small_volume = small_volume + contrib_flat.view(small_shape)
        weight_volume = weight_volume + weight_flat.view(small_shape)
        if contrib_flat is not None:
            del contrib_flat
        del weight_flat

    if render_mode == "intensity":
        # Normalize the working volume to [0, 1] - ensure we preserve gradient flow
        small_volume = torch.where(
            weight_volume > 1e-6,
            small_volume / (weight_volume + 1e-6),
            torch.zeros_like(small_volume),
        )
    else:
        # Density rendering: accumulate opacity mass and squash to [0,1].
        # The squash keeps the output bounded while preserving monotonicity.
        density = weight_volume * float(density_scale)
        small_volume = 1.0 - torch.exp(-density)

    # If we used a smaller working grid, upsample back to full resolution
    if small_shape != volume_shape:
        # Convert to 5D tensor for F.interpolate (batch, channels, D, H, W)
        volume_5d = small_volume.unsqueeze(0).unsqueeze(0)
        # Upsample to original size with trilinear interpolation
        volume_5d = F.interpolate(volume_5d, size=volume_shape, mode='trilinear', align_corners=False)
        # Convert back to 3D tensor
        volume = volume_5d.squeeze(0).squeeze(0)
    else:
        volume = small_volume

    # Free memory (do not call torch.cuda.empty_cache() here; it typically hurts performance).
    del small_volume

    # Optional debug prints removed for performance; caller can inspect externally.

    # Ensure we have gradients flowing
    if not volume.requires_grad:
        print("WARNING: Volume doesn't require gradients after computation")
        # Create a proper connection to the input
        volume = volume + (points[0, 0] * 0)

    # No threshold - let the gradients flow naturally
    # volume = torch.sigmoid((volume - 0.1) * 10)

    return volume


def differentiable_max_pooling(volume: Tensor, kernel_size: int = 3) -> Tensor:
    """
    Differentiable approximate maximum pooling using softmax.
    Useful for reducing noise in the volume.
    
    Args:
        volume: Input volume (D, H, W)
        kernel_size: Size of pooling kernel
        
    Returns:
        Pooled volume (D, H, W)
    """
    volume_shape = volume.shape
    padding = kernel_size // 2
    
    # Add batch and channel dimensions
    x = volume.unsqueeze(0).unsqueeze(0)
    
    # Extract patches
    patches = F.unfold(
        F.pad(x, (padding, padding, padding, padding, padding, padding)),
        kernel_size=kernel_size
    )
    
    # Soft maximum using softmax
    softmax = F.softmax(patches * 10.0, dim=1)  # Scale factor for sharper maximum
    pooled = (patches * softmax).sum(dim=1)
    
    # Reshape back to volume
    return pooled.view(volume_shape)
