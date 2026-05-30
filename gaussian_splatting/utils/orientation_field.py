"""Utility functions for orientation initialization from volumetric structure."""

from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

_DEFAULT_SIGMA_EPS = 1e-6
_GRAD_EPS = 1e-6
_FALLBACK_EPS = (
    1e-8  # Very small threshold - only fallback when gradient is essentially zero
)

_DEBUG_ORIENTATION = os.environ.get("GS_ORIENTATION_DEBUG", "0") == "1"
_ORIENTATION_DEBUG_STATE = {"points": 0, "fallbacks": 0, "calls": 0, "reported": False}


def _axis_derivative_3d(
    data_5d: Tensor,
    axis: str,
    order: int = 1,
) -> Tensor:
    """Apply centered finite differences along a chosen axis on a [1,1,D,H,W] tensor."""
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be one of {'x','y','z'}")
    if order not in {1, 2}:
        raise ValueError("Only first or second derivatives are supported.")

    if order == 1:
        kernel_vals = torch.tensor(
            [-0.5, 0.0, 0.5], device=data_5d.device, dtype=data_5d.dtype
        )
    else:
        kernel_vals = torch.tensor(
            [1.0, -2.0, 1.0], device=data_5d.device, dtype=data_5d.dtype
        )

    if axis == "x":
        kernel = kernel_vals.view(1, 1, 1, 1, -1)
        padding = (0, 0, kernel_vals.shape[0] // 2)
    elif axis == "y":
        kernel = kernel_vals.view(1, 1, 1, -1, 1)
        padding = (0, kernel_vals.shape[0] // 2, 0)
    else:  # z
        kernel = kernel_vals.view(1, 1, -1, 1, 1)
        padding = (kernel_vals.shape[0] // 2, 0, 0)

    return F.conv3d(data_5d, kernel, padding=padding)


def _normalize_index(idx: Tensor, size: int) -> Tensor:
    """Normalize voxel indices to [-1, 1] for grid_sample with align_corners=True."""
    if size <= 1:
        return torch.zeros_like(idx)
    denom = float(size - 1)
    return (idx / denom) * 2.0 - 1.0


def default_origin_and_spacing(
    volume_shape: Tuple[int, int, int],
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Return origin and voxel spacing vectors for a normalized [0, 1]^3 volume."""
    dims_dhw = torch.tensor(volume_shape, device=device, dtype=torch.float32)
    dims_xyz = dims_dhw[[2, 1, 0]].clamp_min(1.0)
    origin = torch.zeros(3, device=device, dtype=torch.float32)
    voxel = 1.0 / (dims_xyz - 1.0).clamp_min(1.0)
    return origin, voxel


def _gauss1d_kernel(sigma: float, device: torch.device) -> Tensor:
    """Return a 1D Gaussian kernel normalised to sum 1."""
    if sigma <= _DEFAULT_SIGMA_EPS:
        return torch.tensor([1.0], device=device, dtype=torch.float32)

    radius = max(1, int(3.0 * sigma))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def _separable_gaussian_blur3d(volume: Tensor, sigma: float) -> Tensor:
    """Apply separable 3D Gaussian smoothing to a [1, 1, D, H, W] tensor."""
    if sigma <= _DEFAULT_SIGMA_EPS:
        return volume

    kernel = _gauss1d_kernel(sigma, volume.device).to(volume.dtype)
    kx = kernel.view(1, 1, -1, 1, 1)
    ky = kernel.view(1, 1, 1, -1, 1)
    kz = kernel.view(1, 1, 1, 1, -1)

    pad = kernel.numel() // 2
    volume = F.conv3d(volume, kx, padding=(pad, 0, 0))
    volume = F.conv3d(volume, ky, padding=(0, pad, 0))
    volume = F.conv3d(volume, kz, padding=(0, 0, pad))
    return volume


def _frames_from_directions(direction: Tensor, fallback: Tensor) -> Tensor:
    """Build orthonormal frames whose third axis follows the supplied directions."""
    if direction.numel() == 0:
        return torch.empty(0, 3, 3, device=direction.device, dtype=direction.dtype)

    ref_axis = torch.zeros_like(direction)
    ref_axis[:, 2] = 1.0
    close_to_z = direction[:, 2].abs() > 0.9
    if close_to_z.any():
        ref_axis[close_to_z] = torch.tensor(
            [0.0, 1.0, 0.0], device=direction.device, dtype=direction.dtype
        )

    tangent = torch.cross(ref_axis, direction, dim=1)
    tangent_norm = tangent.norm(dim=1, keepdim=True).clamp_min(_GRAD_EPS)
    tangent = tangent / tangent_norm
    bitangent = torch.cross(direction, tangent, dim=1)

    rot = torch.stack([tangent, bitangent, direction], dim=2)
    rot = torch.nan_to_num(rot, nan=0.0, posinf=0.0, neginf=0.0)

    qr_dtype = torch.float32 if rot.dtype in {torch.float16, torch.bfloat16} else rot.dtype
    q, _ = torch.linalg.qr(rot.to(dtype=qr_dtype))
    det = torch.det(q)
    neg = det < 0
    if neg.any():
        q[neg, :, 0] = -q[neg, :, 0]

    if q.dtype != direction.dtype:
        q = q.to(dtype=direction.dtype)

    if fallback.any():
        q[fallback] = torch.eye(3, device=direction.device, dtype=direction.dtype)

    return q


def structure_from_mask_at_ijk(
    mask_volume: Tensor,
    ijk: Tensor,
    *,
    mask_threshold: float = 0.1,
    sigma_pre: float = 0.0,
    vesselness_eps: float = 1e-4,
) -> Tuple[Tensor, Tensor]:
    """Compute Hessian-based quaternions/vesselness for a set of voxel indices.

    This is a lightweight alternative to building a dense [D,H,W] structure field.
    It estimates the Hessian at each query location using local finite differences
    (optionally on a globally smoothed copy of the mask).

    Args:
        mask_volume: Mask tensor with shape [D, H, W].
        ijk: Query voxel coordinates (z, y, x) with shape [N, 3]. Values may be
            fractional; sampling uses nearest-voxel indices.

    Returns:
        (quats, vesselness) where:
          - quats has shape [N, 4] (w, x, y, z)
          - vesselness has shape [N, 1]
    """
    if mask_volume is None or mask_volume.numel() == 0:
        empty_v = torch.zeros(0, 1, device=ijk.device)
        empty_q = torch.zeros(0, 4, device=ijk.device)
        return empty_q, empty_v

    if mask_volume.dim() != 3:
        raise ValueError("mask_volume must be [D, H, W].")
    if ijk.dim() != 2 or ijk.shape[1] != 3:
        raise ValueError("ijk must be shaped [N, 3].")

    device = mask_volume.device
    dtype = mask_volume.dtype

    n = int(ijk.shape[0])
    if n == 0:
        return (
            torch.empty(0, 4, device=device, dtype=dtype),
            torch.empty(0, 1, device=device, dtype=dtype),
        )

    vol = mask_volume
    if sigma_pre > _DEFAULT_SIGMA_EPS:
        vol5 = vol.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        vol5 = _separable_gaussian_blur3d(vol5, float(sigma_pre))
        vol = vol5.squeeze(0).squeeze(0).to(dtype=dtype)

    D, H, W = vol.shape
    centers = ijk.round().to(device=device, dtype=torch.long)
    z = centers[:, 0].clamp(1, max(D - 2, 1))
    y = centers[:, 1].clamp(1, max(H - 2, 1))
    x = centers[:, 2].clamp(1, max(W - 2, 1))

    f = vol[z, y, x]
    in_mask = f >= float(mask_threshold)

    # Default outputs.
    vesselness = torch.zeros(n, device=device, dtype=torch.float32)
    direction = torch.zeros(n, 3, device=device, dtype=torch.float32)
    direction[:, 2] = 1.0
    fallback = ~in_mask

    if in_mask.any():
        zi = z[in_mask]
        yi = y[in_mask]
        xi = x[in_mask]

        f0 = vol[zi, yi, xi].to(torch.float32)

        fx1 = vol[zi, yi, xi + 1].to(torch.float32)
        fxm1 = vol[zi, yi, xi - 1].to(torch.float32)
        fy1 = vol[zi, yi + 1, xi].to(torch.float32)
        fym1 = vol[zi, yi - 1, xi].to(torch.float32)
        fz1 = vol[zi + 1, yi, xi].to(torch.float32)
        fzm1 = vol[zi - 1, yi, xi].to(torch.float32)

        fxx = fx1 - 2.0 * f0 + fxm1
        fyy = fy1 - 2.0 * f0 + fym1
        fzz = fz1 - 2.0 * f0 + fzm1

        fxy = (
            vol[zi, yi + 1, xi + 1].to(torch.float32)
            - vol[zi, yi + 1, xi - 1].to(torch.float32)
            - vol[zi, yi - 1, xi + 1].to(torch.float32)
            + vol[zi, yi - 1, xi - 1].to(torch.float32)
        ) * 0.25
        fxz = (
            vol[zi + 1, yi, xi + 1].to(torch.float32)
            - vol[zi + 1, yi, xi - 1].to(torch.float32)
            - vol[zi - 1, yi, xi + 1].to(torch.float32)
            + vol[zi - 1, yi, xi - 1].to(torch.float32)
        ) * 0.25
        fyz = (
            vol[zi + 1, yi + 1, xi].to(torch.float32)
            - vol[zi + 1, yi - 1, xi].to(torch.float32)
            - vol[zi - 1, yi + 1, xi].to(torch.float32)
            + vol[zi - 1, yi - 1, xi].to(torch.float32)
        ) * 0.25

        hess = torch.stack(
            [
                torch.stack([fxx, fxy, fxz], dim=-1),
                torch.stack([fxy, fyy, fyz], dim=-1),
                torch.stack([fxz, fyz, fzz], dim=-1),
            ],
            dim=-2,
        )
        hess = torch.nan_to_num(hess, nan=0.0, posinf=0.0, neginf=0.0)

        evals, evecs = torch.linalg.eigh(hess)
        abs_vals = evals.abs()
        order = torch.argsort(abs_vals, dim=-1)
        order_expand = order.unsqueeze(1).expand(-1, 3, -1)
        sorted_evals = torch.gather(evals, 1, order)
        sorted_evecs = torch.gather(evecs, 2, order_expand)

        dir_i = sorted_evecs[:, :, 0]
        dir_i = torch.nan_to_num(dir_i, nan=0.0, posinf=0.0, neginf=0.0)
        dir_i = dir_i / dir_i.norm(dim=1, keepdim=True).clamp_min(_GRAD_EPS)

        lambda1 = sorted_evals[:, 0]
        lambda2 = sorted_evals[:, 1]
        lambda3 = sorted_evals[:, 2]
        denom = lambda2.abs() + lambda3.abs() + 1e-6
        vessel_i = (1.0 - (lambda1.abs() / denom)).clamp(0.0, 1.0)

        masked_fallback = vessel_i < float(vesselness_eps)
        if masked_fallback.any():
            dir_i = dir_i.clone()
            dir_i[masked_fallback] = torch.tensor(
                [0.0, 0.0, 1.0], device=dir_i.device, dtype=dir_i.dtype
            )

        direction[in_mask] = dir_i
        vesselness[in_mask] = vessel_i
        fallback[in_mask] = masked_fallback

    rot = _frames_from_directions(direction.to(dtype=dtype), fallback)
    quats = rotmat_to_quat(rot)
    vesselness_out = vesselness.to(device=device, dtype=dtype).view(-1, 1)
    return quats.contiguous(), vesselness_out.contiguous()


def compute_gradient_field(
    volume: Tensor,
    sigma_pre: float = 1.5,
    sigma_post: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """Return gradient vectors and magnitudes for each voxel in a scalar field."""
    if volume.dim() != 3:
        raise ValueError("Volume tensor must have shape [D, H, W].")

    device = volume.device
    dtype = volume.dtype

    with torch.no_grad():
        data = volume.unsqueeze(0).unsqueeze(0)
        data = _separable_gaussian_blur3d(data, sigma_pre)

        kernel_dx = torch.tensor([[[[[-1.0, 0.0, 1.0]]]]], device=device, dtype=dtype)
        kernel_dy = torch.tensor([[[[[-1.0], [0.0], [1.0]]]]], device=device, dtype=dtype)
        kernel_dz = torch.tensor([[[[[-1.0]], [[0.0]], [[1.0]]]]], device=device, dtype=dtype)
        kernel_dx *= 0.5
        kernel_dy *= 0.5
        kernel_dz *= 0.5

        gx = F.conv3d(data, kernel_dx, padding=(0, 0, 1))
        gy = F.conv3d(data, kernel_dy, padding=(0, 1, 0))
        gz = F.conv3d(data, kernel_dz, padding=(1, 0, 0))

        gx = gx.squeeze(0).squeeze(0)
        gy = gy.squeeze(0).squeeze(0)
        gz = gz.squeeze(0).squeeze(0)

    if sigma_post > _DEFAULT_SIGMA_EPS:
        gx = _separable_gaussian_blur3d(gx.unsqueeze(0).unsqueeze(0), sigma_post)
        gy = _separable_gaussian_blur3d(gy.unsqueeze(0).unsqueeze(0), sigma_post)
        gz = _separable_gaussian_blur3d(gz.unsqueeze(0).unsqueeze(0), sigma_post)
        gx = gx.squeeze(0).squeeze(0)
        gy = gy.squeeze(0).squeeze(0)
        gz = gz.squeeze(0).squeeze(0)

    grad = torch.stack([gx, gy, gz], dim=-1)
    magnitude = torch.linalg.norm(grad, dim=-1)
    return grad.contiguous(), magnitude.contiguous()


def compute_hessian_field(
    volume: Tensor,
    sigma_pre: float = 1.0,
    sigma_post: float = 0.0,
) -> Tensor:
    """Return the Hessian matrix H(x) for every voxel in a scalar field."""
    if volume.dim() != 3:
        raise ValueError("Volume tensor must have shape [D, H, W].")

    with torch.no_grad():
        data = volume.unsqueeze(0).unsqueeze(0)
        data = _separable_gaussian_blur3d(data, sigma_pre)

        h_xx = _axis_derivative_3d(data, "x", order=2)
        h_yy = _axis_derivative_3d(data, "y", order=2)
        h_zz = _axis_derivative_3d(data, "z", order=2)
        h_xy = _axis_derivative_3d(_axis_derivative_3d(data, "y"), "x")
        h_xz = _axis_derivative_3d(_axis_derivative_3d(data, "z"), "x")
        h_yz = _axis_derivative_3d(_axis_derivative_3d(data, "z"), "y")

    if sigma_post > _DEFAULT_SIGMA_EPS:

        def _smooth(comp: Tensor) -> Tensor:
            return _separable_gaussian_blur3d(comp, sigma_post)

        h_xx, h_yy, h_zz = _smooth(h_xx), _smooth(h_yy), _smooth(h_zz)
        h_xy, h_xz, h_yz = _smooth(h_xy), _smooth(h_xz), _smooth(h_yz)

    h_xx = h_xx.squeeze(0).squeeze(0)
    h_yy = h_yy.squeeze(0).squeeze(0)
    h_zz = h_zz.squeeze(0).squeeze(0)
    h_xy = h_xy.squeeze(0).squeeze(0)
    h_xz = h_xz.squeeze(0).squeeze(0)
    h_yz = h_yz.squeeze(0).squeeze(0)

    hessian = torch.stack(
        [
            torch.stack([h_xx, h_xy, h_xz], dim=-1),
            torch.stack([h_xy, h_yy, h_yz], dim=-1),
            torch.stack([h_xz, h_yz, h_zz], dim=-1),
        ],
        dim=-2,
    )
    return hessian.contiguous()


def build_structure_field(
    mask_volume: Tensor,
    mask_threshold: float = 0.1,
    sigma_pre: float = 1.0,
    sigma_post: float = 0.0,
    vesselness_eps: float = 1e-4,
) -> Tuple[Tensor, Tensor]:
    """Compute quaternion and vesselness fields from a mask using Hessian eigendecomposition."""
    if mask_volume.dim() != 3:
        raise ValueError("mask_volume must be [D, H, W].")

    hessian = compute_hessian_field(
        mask_volume, sigma_pre=sigma_pre, sigma_post=sigma_post
    )
    D, H, W = mask_volume.shape
    hessian_flat = hessian.view(-1, 3, 3)

    hessian_flat = torch.nan_to_num(hessian_flat, nan=0.0, posinf=0.0, neginf=0.0)

    mask_bool = mask_volume.reshape(-1) >= mask_threshold

    # Defaults for voxels outside the mask (or below threshold).
    direction = torch.zeros((hessian_flat.shape[0], 3), device=hessian_flat.device, dtype=hessian_flat.dtype)
    direction[:, 2] = 1.0
    vesselness = torch.zeros((hessian_flat.shape[0],), device=hessian_flat.device, dtype=hessian_flat.dtype)
    fallback = ~mask_bool

    if mask_bool.any():
        masked_hessian = hessian_flat[mask_bool]
        n = masked_hessian.shape[0]

        # Chunked batched-eigh to avoid cuSOLVER limitations on very large batches.
        chunk = 500_000
        evals_chunks = []
        evecs_chunks = []
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            evals_c, evecs_c = torch.linalg.eigh(masked_hessian[start:end])
            evals_chunks.append(evals_c)
            evecs_chunks.append(evecs_c)

        evals = torch.cat(evals_chunks, dim=0)
        evecs = torch.cat(evecs_chunks, dim=0)

        abs_vals = evals.abs()
        order = torch.argsort(abs_vals, dim=-1)
        order_expand = order.unsqueeze(1).expand(-1, 3, -1)
        sorted_evals = torch.gather(evals, 1, order)
        sorted_evecs = torch.gather(evecs, 2, order_expand)

        direction_masked = sorted_evecs[:, :, 0]
        direction_masked = torch.nan_to_num(
            direction_masked, nan=0.0, posinf=0.0, neginf=0.0
        )
        direction_norm = direction_masked.norm(dim=1, keepdim=True).clamp_min(_GRAD_EPS)
        direction_masked = direction_masked / direction_norm

        lambda1 = sorted_evals[:, 0]
        lambda2 = sorted_evals[:, 1]
        lambda3 = sorted_evals[:, 2]
        denom = lambda2.abs() + lambda3.abs() + 1e-6
        vesselness_masked = (1.0 - (lambda1.abs() / denom)).clamp(0.0, 1.0)

        masked_fallback = vesselness_masked < vesselness_eps
        if masked_fallback.any():
            direction_masked[masked_fallback] = torch.tensor(
                [0.0, 0.0, 1.0], device=direction_masked.device, dtype=direction_masked.dtype
            )

        # Scatter results back into full fields.
        direction[mask_bool] = direction_masked
        vesselness[mask_bool] = vesselness_masked
        fallback[mask_bool] = masked_fallback

    rot = _frames_from_directions(direction, fallback)
    quats = rotmat_to_quat(rot).view(D, H, W, 4)
    vessel_field = vesselness.view(D, H, W)
    return quats.contiguous(), vessel_field.contiguous()


def sample_structure_field(
    quat_field: Tensor,
    vesselness_field: Tensor,
    ijk: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Sample quaternion/vesselness fields at fractional voxel indices."""
    if ijk.numel() == 0:
        empty = torch.empty(0, 1, device=quat_field.device)
        return torch.empty(0, 4, device=quat_field.device), empty

    if quat_field.shape[:3] != vesselness_field.shape:
        raise ValueError("Quaternion and vesselness grids must share spatial dims.")

    D, H, W = quat_field.shape[:3]
    grid = torch.stack(
        [
            _normalize_index(ijk[:, 2], W),
            _normalize_index(ijk[:, 1], H),
            _normalize_index(ijk[:, 0], D),
        ],
        dim=-1,
    ).to(device=quat_field.device, dtype=quat_field.dtype)
    grid = grid.view(1, -1, 1, 1, 3)

    quat_field_5d = quat_field.permute(3, 0, 1, 2).unsqueeze(0)
    vessel_field_5d = vesselness_field.unsqueeze(0).unsqueeze(0)

    sampled_quat = (
        F.grid_sample(
            quat_field_5d,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        .view(4, -1)
        .t()
    )
    sampled_quat = sampled_quat / (
        sampled_quat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    )

    sampled_vessel = F.grid_sample(
        vessel_field_5d,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).view(-1, 1)

    return sampled_quat.contiguous(), sampled_vessel.contiguous()


def world_to_voxel(
    xyz_world: Tensor,
    origin_xyz: Tensor,
    voxel_size_xyz: Tensor,
) -> Tensor:
    """Convert world coordinates (x, y, z) to voxel indices (z, y, x)."""
    if xyz_world.dim() != 2 or xyz_world.shape[1] != 3:
        raise ValueError("Expected xyz_world with shape [N, 3].")

    rel = xyz_world - origin_xyz.unsqueeze(0)
    rel = rel / voxel_size_xyz.unsqueeze(0)
    iz = rel[:, 2]
    iy = rel[:, 1]
    ix = rel[:, 0]

    ijk = torch.stack([iz, iy, ix], dim=-1)

    # Debug: check coordinate transformation (show for first call only)
    if not hasattr(world_to_voxel, "_printed"):
        unique_ijk = torch.unique(ijk.round(), dim=0)
        print(
            f"[world_to_voxel] {xyz_world.shape[0]} points -> {unique_ijk.shape[0]} unique voxels"
        )
        print(
            f"[world_to_voxel] Voxel range: z=[{ijk[:, 0].min():.2f}, {ijk[:, 0].max():.2f}], "
            f"y=[{ijk[:, 1].min():.2f}, {ijk[:, 1].max():.2f}], x=[{ijk[:, 2].min():.2f}, {ijk[:, 2].max():.2f}]"
        )
        world_to_voxel._printed = True

    return ijk


def world_to_grid(
    xyz_world: Tensor,
    origin_xyz: Tensor,
    voxel_size_xyz: Tensor,
    volume_shape: Tuple[int, int, int],
) -> Tensor:
    """Convert world coordinates to grid_sample coordinates in [-1, 1].

    Note: For 5D `grid_sample` with input shaped [N, C, D, H, W], the grid's last
    dimension must be ordered (x, y, z), corresponding to (W, H, D).
    """
    ijk = world_to_voxel(xyz_world, origin_xyz, voxel_size_xyz)
    D, H, W = volume_shape
    norm_z = _normalize_index(ijk[:, 0], D)
    norm_y = _normalize_index(ijk[:, 1], H)
    norm_x = _normalize_index(ijk[:, 2], W)
    return torch.stack([norm_x, norm_y, norm_z], dim=-1)


def gather_rotation_from_gradient(
    grad_field: Tensor,
    mag_field: Tensor,
    ijk: Tensor,
    eps: float = _FALLBACK_EPS,
) -> Tuple[Tensor, Tensor]:
    """
    Sample rotation matrices from gradient field.

    The gradient direction becomes the main axis (principal eigenvector),
    and gradient magnitude represents the eigenvalue (structural strength).

    Args:
        grad_field: [D, H, W, 3] gradient vectors at each voxel
        mag_field: [D, H, W] gradient magnitudes at each voxel
        ijk: [N, 3] voxel indices (z, y, x) to sample
        eps: Magnitude threshold below which to use identity rotation

    Returns:
        rot: [N, 3, 3] orthonormal rotation matrices
        fallback: [N] bool mask indicating identity fallback was used
    """
    if ijk.numel() == 0:
        return (
            torch.empty(0, 3, 3, device=grad_field.device),
            torch.empty(0, dtype=torch.bool, device=grad_field.device),
        )

    if grad_field.shape[:3] != mag_field.shape:
        raise ValueError("Gradient and magnitude fields must share spatial dimensions.")

    device = grad_field.device
    dtype = grad_field.dtype
    D, H, W = grad_field.shape[:3]

    # Debug: Check volume and index ranges (first call only)
    if not hasattr(gather_rotation_from_gradient, "_printed"):
        ijk_min = ijk.min(dim=0).values
        ijk_max = ijk.max(dim=0).values
        print(f"[gather_rotation_from_gradient] Gradient field shape: [{D}, {H}, {W}]")
        print(
            f"[gather_rotation_from_gradient] Index range: z=[{ijk_min[0]:.2f}, {ijk_max[0]:.2f}], "
            f"y=[{ijk_min[1]:.2f}, {ijk_max[1]:.2f}], x=[{ijk_min[2]:.2f}, {ijk_max[2]:.2f}]"
        )
        gather_rotation_from_gradient._printed = True

    # Convert voxel indices (z, y, x) to normalized grid coordinates (x, y, z)
    # for grid_sample on [N, C, D, H, W] tensors.
    grid = torch.stack(
        [
            _normalize_index(ijk[:, 2], W),
            _normalize_index(ijk[:, 1], H),
            _normalize_index(ijk[:, 0], D),
        ],
        dim=-1,
    ).to(device=device, dtype=dtype)
    grid = grid.view(1, -1, 1, 1, 3)

    # Prepare fields for grid_sample: [1, C, D, H, W]
    vec_field = grad_field.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, D, H, W]
    mag_field_5d = mag_field.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

    # Sample gradient vectors and magnitudes at point locations
    sampled_vecs = F.grid_sample(
        vec_field,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    sampled_mag = F.grid_sample(
        mag_field_5d,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    # Reshape to [N, 3] and [N]
    grad = sampled_vecs.view(3, -1).t().contiguous()
    mag = sampled_mag.view(-1)

    # Clean up NaN/Inf values
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    mag = torch.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)

    # Identify low-magnitude points that need identity fallback
    fallback = mag < eps

    # Normalize gradient to get main direction (principal axis)
    safe_mag = mag.clamp_min(_GRAD_EPS)
    direction = grad / safe_mag.unsqueeze(1)

    # Build orthonormal frames from gradient direction
    q = _frames_from_directions(direction, fallback)

    # Debug logging - show actual magnitude statistics
    fallback_count = fallback.sum().item()
    total_count = len(fallback)
    if _DEBUG_ORIENTATION:
        mag_min = mag.min().item()
        mag_max = mag.max().item()
        mag_mean = mag.mean().item()
        print(
            f"[Orientation] {fallback_count}/{total_count} fallbacks ({fallback_count/total_count*100:.1f}%)"
        )
        print(
            f"[Orientation] Magnitude: min={mag_min:.6f}, max={mag_max:.6f}, mean={mag_mean:.6f}, threshold={eps:.6f}"
        )

    _ORIENTATION_DEBUG_STATE["calls"] += 1

    return q.contiguous(), fallback


def rotmat_to_quat(rot_mats: Tensor) -> Tensor:
    """Convert rotation matrices [N, 3, 3] to unit quaternions [N, 4]."""
    if rot_mats.numel() == 0:
        return torch.empty(0, 4, device=rot_mats.device, dtype=rot_mats.dtype)

    trace = rot_mats[:, 0, 0] + rot_mats[:, 1, 1] + rot_mats[:, 2, 2]
    quats = torch.empty(
        rot_mats.shape[0],
        4,
        device=rot_mats.device,
        dtype=rot_mats.dtype,
    )

    positive = trace > 0.0
    if positive.any():
        t = torch.sqrt(trace[positive] + 1.0) * 2.0
        quats[positive, 0] = 0.25 * t
        quats[positive, 1] = (rot_mats[positive, 2, 1] - rot_mats[positive, 1, 2]) / t
        quats[positive, 2] = (rot_mats[positive, 0, 2] - rot_mats[positive, 2, 0]) / t
        quats[positive, 3] = (rot_mats[positive, 1, 0] - rot_mats[positive, 0, 1]) / t

    remaining = ~positive
    if remaining.any():
        r = rot_mats[remaining]
        diag = torch.stack([r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]], dim=1)
        max_idx = diag.argmax(dim=1)
        q = torch.empty_like(quats[remaining])
        for axis in range(3):
            mask = max_idx == axis
            if not mask.any():
                continue
            i = axis
            j = (axis + 1) % 3
            k = (axis + 2) % 3
            diag_term = r[mask, i, i] - r[mask, j, j] - r[mask, k, k] + 1.0
            s = torch.sqrt(diag_term.clamp_min(1e-8)) * 2.0
            q[mask, i + 1] = 0.25 * s
            q[mask, 0] = (r[mask, k, j] - r[mask, j, k]) / s
            q[mask, j + 1] = (r[mask, j, i] + r[mask, i, j]) / s
            q[mask, k + 1] = (r[mask, k, i] + r[mask, i, k]) / s
        quats[remaining] = q

    quats = quats / (quats.norm(dim=1, keepdim=True) + 1e-8)
    return quats


def quat_from_directions(direction: Tensor, eps: float = 1e-6) -> Tuple[Tensor, Tensor]:
    """Return quaternions whose local z-axis aligns to the provided direction.

    Args:
        direction: Direction vectors in (x, y, z) order, shape [N, 3].
        eps: Threshold below which identity quaternions are returned.

    Returns:
        Tuple of:
            - quats: Unit quaternions [N, 4]
            - fallback: Bool mask [N] indicating identity fallback.
    """
    if direction.numel() == 0:
        return (
            torch.empty(0, 4, device=direction.device, dtype=direction.dtype),
            torch.empty(0, dtype=torch.bool, device=direction.device),
        )

    if direction.dim() != 2 or direction.shape[1] != 3:
        raise ValueError("direction must have shape [N, 3].")

    norm = direction.norm(dim=1)
    fallback = norm < eps
    safe = direction / norm.clamp_min(eps).unsqueeze(1)
    rot = _frames_from_directions(safe, fallback)
    quats = rotmat_to_quat(rot)
    return quats.contiguous(), fallback.contiguous()


def random_quat_perturb(quats: Tensor, deg: float = 2.0) -> Tensor:
    """Apply a small random axis-angle perturbation to quaternions."""
    if quats.numel() == 0 or deg <= 0.0:
        return quats

    device = quats.device
    dtype = quats.dtype
    angle = (torch.rand(quats.shape[0], 1, device=device, dtype=dtype) - 0.5) * 2.0
    angle *= torch.deg2rad(torch.tensor(deg, device=device, dtype=dtype))

    axis = torch.randn(quats.shape[0], 3, device=device, dtype=dtype)
    axis /= axis.norm(dim=1, keepdim=True).clamp_min(1e-8)

    half = angle * 0.5
    sin_half = torch.sin(half)
    dq = torch.cat([torch.cos(half), axis * sin_half], dim=1)

    w1, x1, y1, z1 = dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3]
    w2, x2, y2, z2 = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    out = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=1,
    )
    out = out / (out.norm(dim=1, keepdim=True) + 1e-8)
    return out
