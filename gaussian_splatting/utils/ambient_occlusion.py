"""Ambient occlusion utilities for mask/volume supervision.

The AO computation here is intentionally lightweight: it precomputes a per-voxel
occlusion factor from the input mask volume once at startup and can then be
sampled at Gaussian positions during PLY export.

All volumes use the project convention: tensors are shaped [D, H, W] = [Z, Y, X]
while Gaussian positions are in normalized [0, 1]^3 with coordinates ordered as
[x, y, z].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


AOMethod = Literal["isotropic", "normal"]


@dataclass(frozen=True)
class AOResult:
    """Container returned by AO precompute."""

    ao_volume: Tensor  # [D, H, W] float32 in [0,1]
    surface_mask: Optional[Tensor] = None  # [D,H,W] bool


def _make_spherical_kernel(radius_vox: int, device: torch.device) -> Tensor:
    if radius_vox <= 0:
        raise ValueError("radius_vox must be >= 1")

    k = 2 * radius_vox + 1
    zz, yy, xx = torch.meshgrid(
        torch.arange(k, device=device),
        torch.arange(k, device=device),
        torch.arange(k, device=device),
        indexing="ij",
    )
    zz = zz - radius_vox
    yy = yy - radius_vox
    xx = xx - radius_vox

    dist2 = (xx * xx + yy * yy + zz * zz).float()
    kernel = (dist2 <= float(radius_vox * radius_vox)).float()
    kernel[radius_vox, radius_vox, radius_vox] = 0.0
    kernel = kernel.view(1, 1, k, k, k)
    return kernel


def _compute_surface_mask(mask_bool: Tensor) -> Tensor:
    """Return voxels inside the mask that touch outside (6-neighborhood)."""
    if mask_bool.dtype != torch.bool:
        raise ValueError("mask_bool must be a bool tensor")

    z, y, x = mask_bool.shape
    padded = F.pad(mask_bool, (1, 1, 1, 1, 1, 1), mode="constant", value=False)

    center = padded[1 : 1 + z, 1 : 1 + y, 1 : 1 + x]
    n0 = padded[0:z, 1 : 1 + y, 1 : 1 + x]
    n1 = padded[2 : 2 + z, 1 : 1 + y, 1 : 1 + x]
    n2 = padded[1 : 1 + z, 0:y, 1 : 1 + x]
    n3 = padded[1 : 1 + z, 2 : 2 + y, 1 : 1 + x]
    n4 = padded[1 : 1 + z, 1 : 1 + y, 0:x]
    n5 = padded[1 : 1 + z, 1 : 1 + y, 2 : 2 + x]

    fully_inside = n0 & n1 & n2 & n3 & n4 & n5
    surface = center & (~fully_inside)
    return surface


def _compute_normals_from_mask(mask_float: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute outward normals from a (soft) mask volume via central differences.

    Returns normals shaped [D,H,W,3] in xyz order.
    """
    if mask_float.dim() != 3:
        raise ValueError("mask_float must have shape [D,H,W]")

    mask_float = mask_float.to(dtype=torch.float32)
    # Replication padding is implemented for 5D tensors (N,C,D,H,W).
    mask_5d = mask_float.view(1, 1, *mask_float.shape)
    padded = F.pad(mask_5d, (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]

    # Note: tensor axes are [z,y,x] == [D,H,W].
    dz = padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1]
    dy = padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]
    dx = padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]

    # Mask increases from outside(0) -> inside(1), so gradient points inward.
    # Use -grad for outward-facing hemisphere.
    nx = -dx
    ny = -dy
    nz = -dz

    n = torch.stack([nx, ny, nz], dim=-1)
    norm = torch.linalg.vector_norm(n, dim=-1, keepdim=True).clamp_min(eps)
    return n / norm


@torch.no_grad()
def compute_isotropic_ao_from_mask(mask_bool: Tensor, radius_vox: int) -> Tensor:
    """Fast local-occupancy AO: ao = 1 - fraction of occupied voxels in a sphere."""
    if mask_bool.dtype != torch.bool:
        raise ValueError("mask_bool must be a bool tensor")

    device = mask_bool.device
    kernel = _make_spherical_kernel(radius_vox, device)
    denom = float(kernel.sum().item())
    denom = max(denom, 1.0)

    x = mask_bool.to(dtype=torch.float32).view(1, 1, *mask_bool.shape)
    counts = F.conv3d(x, kernel, padding=radius_vox)
    occ = (counts / denom).view_as(mask_bool).clamp_(0.0, 1.0)

    ao = (1.0 - occ).clamp_(0.0, 1.0)
    ao = torch.where(mask_bool, ao, torch.ones_like(ao))
    return ao


@torch.no_grad()
def compute_normal_hemisphere_ao_from_mask(
    mask_volume: Tensor,
    mask_bool: Tensor,
    radius_vox: int,
    *,
    chunk_size: int = 65536,
) -> AOResult:
    """Compute a hemisphere-oriented AO on the surface, isotropic inside.

    - Interior voxels use isotropic AO.
    - Surface voxels use a hemisphere neighborhood test using normals derived
      from the mask gradient.

    This is an approximation (no ray marching) designed to be computed once.
    """
    if mask_bool.dtype != torch.bool:
        raise ValueError("mask_bool must be a bool tensor")
    if mask_volume.shape != mask_bool.shape:
        raise ValueError("mask_volume and mask_bool must have the same shape")

    ao = compute_isotropic_ao_from_mask(mask_bool, radius_vox)
    surface = _compute_surface_mask(mask_bool)

    if surface.sum().item() == 0:
        return AOResult(ao_volume=ao, surface_mask=surface)

    device = mask_bool.device
    offsets_xyz: list[list[int]] = []
    offsets_zyx: list[list[int]] = []
    for oz in range(-radius_vox, radius_vox + 1):
        for oy in range(-radius_vox, radius_vox + 1):
            for ox in range(-radius_vox, radius_vox + 1):
                if ox == 0 and oy == 0 and oz == 0:
                    continue
                if (ox * ox + oy * oy + oz * oz) > radius_vox * radius_vox:
                    continue
                offsets_xyz.append([ox, oy, oz])
                offsets_zyx.append([oz, oy, ox])

    offsets_xyz_t = torch.tensor(offsets_xyz, device=device, dtype=torch.float32)  # [M,3]
    offsets_zyx_t = torch.tensor(offsets_zyx, device=device, dtype=torch.int64)  # [M,3]

    normals = _compute_normals_from_mask(mask_volume)

    idx_zyx = surface.nonzero(as_tuple=False)  # [N,3] in zyx
    normals_surface = normals[idx_zyx[:, 0], idx_zyx[:, 1], idx_zyx[:, 2]]  # [N,3]

    n_total = idx_zyx.shape[0]
    d, h, w = mask_bool.shape

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        idx_chunk = idx_zyx[start:end]
        n_chunk = normals_surface[start:end]

        # Hemisphere selection for each voxel.
        hemi = (offsets_xyz_t[None, :, :] * n_chunk[:, None, :]).sum(dim=-1) > 0.0  # [B,M]
        denom = hemi.float().sum(dim=1).clamp_min(1.0)  # [B]

        coords = idx_chunk[:, None, :] + offsets_zyx_t[None, :, :]  # [B,M,3]
        zc = coords[..., 0].clamp(0, d - 1)
        yc = coords[..., 1].clamp(0, h - 1)
        xc = coords[..., 2].clamp(0, w - 1)

        neigh_inside = mask_bool[zc, yc, xc]  # [B,M]
        occ = (neigh_inside & hemi).float().sum(dim=1) / denom
        ao_chunk = (1.0 - occ).clamp_(0.0, 1.0)

        ao[idx_chunk[:, 0], idx_chunk[:, 1], idx_chunk[:, 2]] = ao_chunk

    ao = torch.where(mask_bool, ao, torch.ones_like(ao))
    return AOResult(ao_volume=ao, surface_mask=surface)


@torch.no_grad()
def compute_ao_volume_from_mask(
    mask_volume: Tensor,
    mask_bool: Tensor,
    *,
    radius_vox: int,
    method: AOMethod = "isotropic",
) -> AOResult:
    """Compute AO volume using the selected method."""
    if method == "isotropic":
        ao = compute_isotropic_ao_from_mask(mask_bool, radius_vox)
        return AOResult(ao_volume=ao, surface_mask=None)

    if method == "normal":
        return compute_normal_hemisphere_ao_from_mask(mask_volume, mask_bool, radius_vox)

    raise ValueError(f"Unknown AO method: {method}")
