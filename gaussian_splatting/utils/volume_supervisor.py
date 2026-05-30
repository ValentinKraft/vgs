#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.

"""
Volume supervision manager for 3D Gaussian Splatting.
Handles volume loading, loss computation, and optimization tracking.
"""

import torch
from torch import Tensor
from typing import Optional, Dict, Tuple

from gaussian_splatting.losses.volume_loss import VolumeLoss
from torch.utils.checkpoint import checkpoint
from gaussian_splatting.utils.splat_to_volume import splat_to_volume
from gaussian_splatting.data.volume_loader import VolumeLoader
from gaussian_splatting.utils.intensity_sampler import (
    sample_mean_covered_voxel_intensities,
    update_intensities,
)
from gaussian_splatting.utils.orientation_field import (
    build_structure_field,
    compute_gradient_field,
    default_origin_and_spacing,
    gather_rotation_from_gradient,
    random_quat_perturb,
    rotmat_to_quat,
    structure_from_mask_at_ijk,
    world_to_voxel,
)

class VolumeSupervisor:
    """
    Manages volume supervision during training:
    - Loads and preprocesses ground truth volumes
    - Computes volume supervision loss
    - Tracks metrics and optimization progress
    """

    def __init__(
        self,
        volume_path: str,
        volume_shape: Tuple[int, int, int] = (64, 64, 64),
        volume_downscale_factor: Optional[int] = None,
        volume_render_downscale_factor: int = 2,
        volume_storage_dtype: str = "fp32",
        disable_volume_overflow_guard: bool = False,
        mask_path: Optional[str] = None,
        loss_type: str = "dice",
        ct_loss_type: str = "mse",
        loss_weight: float = 1.0,
        supervision_target: str = "mask",
        mask_loss_weight: float = 1.0,
        ct_loss_weight: float = 1.0,
        mask_loss_threshold_rel: float = 0.01,
        opacity_gamma: float = 1.0,
        density_scale: float = 1.0,
        outside_mask_weight: float = 0.1,
        device: torch.device = torch.device("cuda"),
        intensity_update_interval: int = 10,
        opacity_update_interval: Optional[int] = None,
        dirty_threshold_xyz: float = 1e-3,
        dirty_threshold_scale: float = 5e-3,
        dirty_threshold_rot: float = 8.726646e-3,
        verbose: bool = False,
        sampling_padding_mode: str = "border",
        sparse_support_cutoff: float = 0.2,
        sparse_max_radius_vox: int = 10,
        sparse_support_softness: float = 0.75,
        render_min_sigma_vox: float = 0.35,
        sparse_support_cutoff_final: Optional[float] = None,
        sparse_support_softness_final: Optional[float] = None,
        render_min_sigma_vox_final: Optional[float] = None,
        raster_schedule_start_iter: int = -1,
        raster_schedule_end_iter: int = -1,
    ):
        """
        Args:
            volume_path: Path to ground truth volume
            volume_shape: Target shape for volume optimization
            mask_path: Optional path to mask volume for opacity values
            loss_type: Type of volume loss ('mse', 'dice', 'tversky', 'kl')
            loss_weight: Weight for volume loss term
            device: Device to use for computations
            verbose: When True, print detailed diagnostics during sampling
        """
        self.device = device
        self.volume_shape = volume_shape
        self.loss_weight = loss_weight
        self.verbose = bool(verbose)

        if supervision_target not in {"mask", "ct", "joint"}:
            raise ValueError(
                "supervision_target must be one of {'mask','ct','joint'}, got "
                f"{supervision_target!r}."
            )
        self.supervision_target = str(supervision_target)
        self.mask_loss_weight = float(mask_loss_weight)
        self.ct_loss_weight = float(ct_loss_weight)
        self.opacity_gamma = float(opacity_gamma)
        self.density_scale = float(density_scale)
        self.mask_loss_threshold_rel = float(mask_loss_threshold_rel)
        self.outside_mask_weight = float(outside_mask_weight)
        self.volume_render_downscale_factor = max(
            1, int(volume_render_downscale_factor)
        )
        self.volume_storage_dtype = str(volume_storage_dtype).lower()
        if self.volume_storage_dtype not in {"fp32", "fp16", "bf16"}:
            raise ValueError(
                "volume_storage_dtype must be one of {'fp32','fp16','bf16'}, "
                f"got {volume_storage_dtype!r}."
            )
        self.disable_volume_overflow_guard = bool(disable_volume_overflow_guard)

        # Training is defined to be mask-driven; without a mask the objective is
        # ill-posed for the intended medical workflows.
        if not mask_path:
            raise ValueError(
                "mask_path is required for volume supervision (training without a mask is not supported)."
            )

        # Loss masking: only voxels above a fraction of mask max contribute.
        # Default matches the medical workflow requirement: 1% of mask max.

        # Initialize volume loader and loss
        # Default behavior (omitted flag) matches downscale_factor=1: keep native resolution.
        downscale_factor = (
            int(volume_downscale_factor)
            if volume_downscale_factor is not None
            else 1
        )
        self.loader = VolumeLoader(
            target_shape=None,
            device=device,
            downscale_factor=downscale_factor,
            storage_dtype=self.volume_storage_dtype,
            enable_overflow_guard=not self.disable_volume_overflow_guard,
        )
        # Apply loss_weight once in compute_loss for clarity.
        self.mask_criterion = VolumeLoss(loss_type, 1.0)
        self.ct_criterion = VolumeLoss(ct_loss_type, 1.0)
        # Back-compat: keep a "primary" criterion.
        self.criterion = self.ct_criterion if self.supervision_target == "ct" else self.mask_criterion

        # Load ground truth volume used for supervision/rendering.
        # The same downscale factor is also used for sampling per-splat intensities/colors so
        # the overflow guard is evaluated against the requested downscaled resolution.
        self.volume_gt = self.loader.load_volume(volume_path)
        self.volume_color = self.volume_gt
        self.raw_intensity_min = self.loader.last_loaded_raw_min
        self.raw_intensity_max = self.loader.last_loaded_raw_max

        # Always trust the loaded tensor shape for supervision/rendering.
        self.volume_shape = tuple(int(v) for v in self.volume_gt.shape)
        self.global_intensity_min = float(self.volume_color.min().item())
        self.global_intensity_max = float(self.volume_color.max().item())
        if self.verbose:
            print(
                "Loaded color volume intensity range: "
                f"[{self.global_intensity_min:.4f}, {self.global_intensity_max:.4f}]"
            )
        if abs(self.global_intensity_max - self.global_intensity_min) <= 1e-8:
            print(
                "Warning: Volume intensity range is nearly zero; outputs will default to mid-gray."
            )

        # Orientation defaults (no CLI controls for now)
        self.orientation_sigma_grad = 0.8  # Reduced for sharper gradients
        self.orientation_sigma_tensor = 0.0  # No post-smoothing needed
        self.orientation_perturb_deg = 2.0
        self._orientation_grad: Optional[Tensor] = None
        self._orientation_mag: Optional[Tensor] = None
        self.structure_sigma = 1.0
        self.structure_mask_threshold = 0.1
        self._structure_quat: Optional[Tensor] = None
        self._structure_vesselness: Optional[Tensor] = None

        # Coordinate mapping assumes volume space normalised to [0, 1]
        origin, spacing = default_origin_and_spacing(
            self.volume_gt.shape, self.device
        )
        self.volume_origin = origin
        self.voxel_size = spacing
        raw_spacing = getattr(self.loader, "last_loaded_spacing_xyz", None)
        self.voxel_spacing_xyz = (
            None
            if raw_spacing is None
            else torch.tensor(raw_spacing, device=self.device, dtype=torch.float32)
        )

        # Load mask volume (required)
        self.mask_volume = self.loader.load_volume(mask_path)
        if self.mask_volume.shape != self.volume_gt.shape:
            raise ValueError(
                "Mask and volume shapes must match after loading. "
                f"volume_shape={tuple(self.volume_gt.shape)}, mask_shape={tuple(self.mask_volume.shape)}"
            )
        if self.verbose:
            print(
                "Loaded mask volume with range "
                f"[{self.mask_volume.min().item():.4f}, {self.mask_volume.max().item():.4f}]"
            )

        # Cache mask-derived data for reuse during loss computation.
        self.mask_max = float(self.mask_volume.max().item())
        self.mask_threshold = float(self.mask_loss_threshold_rel) * self.mask_max
        self.mask_bool = (self.mask_volume > self.mask_threshold).to(device=self.device)
        if not self.mask_bool.any():
            raise RuntimeError(
                "Mask thresholding produced an empty region. "
                f"mask_max={self.mask_max:.6f}, rel_thr={self.mask_loss_threshold_rel:.6f}"
            )

        # Derive intensity normalization range from the intensity volume, but only
        # at voxels where the mask is non-zero.
        masked_vals = self.volume_color[self.mask_bool]
        if masked_vals.numel() > 0:
            self.global_intensity_min = float(masked_vals.min().item())
            self.global_intensity_max = float(masked_vals.max().item())
            if self.verbose:
                print(
                    "Using mask-bounded intensity range: "
                    f"[{self.global_intensity_min:.4f}, {self.global_intensity_max:.4f}]"
                )
            if abs(self.global_intensity_max - self.global_intensity_min) <= 1e-8:
                print(
                    "Warning: Mask-bounded intensity range is nearly zero; outputs will default to mid-gray."
                )

        nz = torch.nonzero(self.mask_bool, as_tuple=False)
        z0 = int(nz[:, 0].min().item())
        z1 = int(nz[:, 0].max().item())
        y0 = int(nz[:, 1].min().item())
        y1 = int(nz[:, 1].max().item())
        x0 = int(nz[:, 2].min().item())
        x1 = int(nz[:, 2].max().item())
        self.mask_bounds = (z0, z1, y0, y1, x0, x1)

        D, H, W = self.volume_shape
        # Disable ROI cropping: supervise/render on the full loaded volume.
        self.roi_shape = (D, H, W)
        self.bounds_min = torch.tensor(
            [0.0, 0.0, 0.0], device=self.device, dtype=torch.float32
        )
        self.bounds_max = torch.tensor(
            [1.0, 1.0, 1.0], device=self.device, dtype=torch.float32
        )
        self.bounds_min_padded = self.bounds_min.clone()
        self.bounds_max_padded = self.bounds_max.clone()
        self.roi_pad_vox = 0.0

        # Cache full-volume tensors for per-iteration reuse.
        self._roi_slices = (
            slice(0, D),
            slice(0, H),
            slice(0, W),
        )
        zsl, ysl, xsl = self._roi_slices
        self.volume_gt_roi = self.volume_gt[zsl, ysl, xsl]
        self.mask_volume_roi = self.mask_volume[zsl, ysl, xsl]
        self.mask_bool_roi = self.mask_bool[zsl, ysl, xsl]

        # Initialize metrics tracking
        self.metrics = {
            "volume_loss": 0.0,
            "volume_loss_unweighted": 0.0,
            "mask_loss": 0.0,
            "ct_loss": 0.0,
            "dice_score": 0.0,
            "outside_mask_loss": 0.0,
            "active_points": 0.0,
            "intensity_update_count": 0.0,
            "opacity_update_count": 0.0,
            "mean_covered_intensity_count": 0.0,
            "mean_covered_opacity_count": 0.0,
        }
        self._step = 0
        self.enable_diagnostics = False
        self.last_intensity_update_count = 0
        self.last_opacity_update_count = 0
        self.last_mean_covered_intensity_count = 0
        self.last_mean_covered_opacity_count = 0
        self.intensity_update_interval = max(1, int(intensity_update_interval))
        if opacity_update_interval is None:
            opacity_update_interval = self.intensity_update_interval
        self.opacity_update_interval = max(1, int(opacity_update_interval))
        self.dirty_threshold_xyz = float(dirty_threshold_xyz)
        self.dirty_threshold_scale = float(dirty_threshold_scale)
        self.dirty_threshold_rot = float(dirty_threshold_rot)
        self.sampling_padding_mode = str(sampling_padding_mode)
        self.sparse_support_cutoff = float(
            min(max(float(sparse_support_cutoff), 1e-5), 0.9999)
        )
        self.sparse_max_radius_vox = max(1, int(sparse_max_radius_vox))
        self.sparse_support_softness = max(float(sparse_support_softness), 0.0)
        self.render_min_sigma_vox = max(float(render_min_sigma_vox), 0.0)
        self.sparse_support_cutoff_final = (
            None
            if sparse_support_cutoff_final is None
            else float(min(max(float(sparse_support_cutoff_final), 1e-5), 0.9999))
        )
        self.sparse_support_softness_final = (
            None
            if sparse_support_softness_final is None
            else max(float(sparse_support_softness_final), 0.0)
        )
        self.render_min_sigma_vox_final = (
            None
            if render_min_sigma_vox_final is None
            else max(float(render_min_sigma_vox_final), 0.0)
        )
        self.raster_schedule_start_iter = int(raster_schedule_start_iter)
        self.raster_schedule_end_iter = int(raster_schedule_end_iter)
        self.enable_render_checkpoint = True

    def _scheduled_raster_value(
        self,
        start_value: float,
        final_value: Optional[float],
    ) -> float:
        """Return the scheduled raster value for the current iteration."""
        if final_value is None:
            return float(start_value)

        start_iter = int(getattr(self, "raster_schedule_start_iter", -1))
        end_iter = int(getattr(self, "raster_schedule_end_iter", -1))
        iteration = int(getattr(self, "iteration", 0))

        if start_iter < 0 or end_iter <= start_iter:
            return float(start_value)
        if iteration <= start_iter:
            return float(start_value)
        if iteration >= end_iter:
            return float(final_value)

        alpha = float(iteration - start_iter) / float(end_iter - start_iter)
        return float(start_value) + alpha * (float(final_value) - float(start_value))

    def _effective_raster_params(self) -> Dict[str, float]:
        """Return sparse raster parameters after applying any active schedule."""
        return {
            "sparse_support_cutoff": self._scheduled_raster_value(
                self.sparse_support_cutoff,
                self.sparse_support_cutoff_final,
            ),
            "sparse_support_softness": self._scheduled_raster_value(
                self.sparse_support_softness,
                self.sparse_support_softness_final,
            ),
            "render_min_sigma_vox": self._scheduled_raster_value(
                self.render_min_sigma_vox,
                self.render_min_sigma_vox_final,
            ),
        }

    def _orientation_source(self) -> Tensor:
        """Return the tensor used to derive orientations."""
        # Use the intensity volume for orientation - it has richer gradients
        # than binary/float masks which are mostly uniform
        return self.volume_color

    def _intensity_sampler(self, gaussians, indices: Optional[Tensor]) -> Tensor:
        """Sample mean intensities for selected indices."""
        xyz = gaussians.get_xyz
        scaling_full = gaussians.get_scaling
        if indices is None:
            pts = xyz
            scales = scaling_full if scaling_full.numel() > 0 else None
            idx_tensor = None
        else:
            idx = indices.long()
            pts = xyz[:, idx] if xyz.shape[0] == 3 else xyz[idx]
            if scaling_full.numel() > 0:
                if (
                    scaling_full.dim() == 2
                    and scaling_full.shape[0] == 3
                    and scaling_full.shape[1] != 3
                ):
                    scales = scaling_full[:, idx]
                else:
                    scales = scaling_full[idx]
            else:
                scales = None
            idx_tensor = idx

        intensities, v_min, v_max = update_intensities(
            pts,
            self.volume_color,
            scale=scales,
            normalize=True,
            min_val=self.global_intensity_min,
            max_val=self.global_intensity_max,
            padding_mode=self.sampling_padding_mode,
        )

        intensity_mode = getattr(gaussians, "intensity_mode", "learned")
        if (
            intensity_mode == "sampled_mean_covered"
            and scales is not None
            and scales.numel() > 0
        ):
            large_mask_global = gaussians.large_splat_mask(
                getattr(gaussians, "intensity_large_splat_threshold", 0.0)
            ).to(device=intensities.device)
            coverage_mask = (
                large_mask_global
                if idx_tensor is None
                else large_mask_global[idx_tensor]
            )
            if coverage_mask is not None and coverage_mask.any():
                self.last_mean_covered_intensity_count = int(
                    coverage_mask.sum().item()
                )
                refined, _, _ = sample_mean_covered_voxel_intensities(
                    pts,
                    self.volume_color,
                    scales,
                    self.volume_origin,
                    self.voxel_size,
                    radius_scale=getattr(gaussians, "mean_covered_radius", 2.5),
                    coverage_mask=coverage_mask,
                    normalize=True,
                    min_val=self.global_intensity_min,
                    max_val=self.global_intensity_max,
                    padding_mode=self.sampling_padding_mode,
                )
                refined = refined.to(
                    device=intensities.device,
                    dtype=intensities.dtype,
                )
                intensities = intensities.clone()
                intensities[coverage_mask] = refined[coverage_mask]

        if indices is None:
            gaussians.volume_min = v_min
            gaussians.volume_max = v_max
        return intensities

    def _opacity_sampler(self, gaussians, indices: Optional[Tensor]) -> Tensor:
        """Sample mask-derived opacities for selected indices."""
        if self.mask_volume is None:
            return torch.empty(0, 1, device=self.device)

        xyz = gaussians.get_xyz
        scaling_full = gaussians.get_scaling
        if indices is None:
            pts = xyz
            scales = scaling_full if scaling_full.numel() > 0 else None
            idx_tensor = None
        else:
            idx = indices.long()
            pts = xyz[:, idx] if xyz.shape[0] == 3 else xyz[idx]
            if scaling_full.numel() > 0:
                if (
                    scaling_full.dim() == 2
                    and scaling_full.shape[0] == 3
                    and scaling_full.shape[1] != 3
                ):
                    scales = scaling_full[:, idx]
                else:
                    scales = scaling_full[idx]
            else:
                scales = None
            idx_tensor = idx

        from gaussian_splatting.utils.intensity_sampler import sample_intensities_from_volume

        opacities, _, _ = sample_intensities_from_volume(
            pts,
            self.mask_volume,
            scale=scales,
            enable_footprint_pooling=True,
            normalize=False,
            min_val=0.0,
            max_val=1.0,
            padding_mode=self.sampling_padding_mode,
        )

        opacity_mode = getattr(gaussians, "opacity_mode", "sampled")
        if (
            opacity_mode == "sampled_mean_covered"
            and scales is not None
            and scales.numel() > 0
        ):
            large_mask_global = gaussians.large_splat_mask(
                getattr(gaussians, "intensity_large_splat_threshold", 0.0)
            ).to(device=opacities.device)
            coverage_mask = large_mask_global if idx_tensor is None else large_mask_global[idx_tensor]
            if coverage_mask is not None and coverage_mask.any():
                self.last_mean_covered_opacity_count = int(
                    coverage_mask.sum().item()
                )
                refined, _, _ = sample_mean_covered_voxel_intensities(
                    pts,
                    self.mask_volume,
                    scales,
                    self.volume_origin,
                    self.voxel_size,
                    radius_scale=getattr(gaussians, "mean_covered_radius", 2.5),
                    coverage_mask=coverage_mask,
                    normalize=False,
                    min_val=0.0,
                    max_val=1.0,
                    padding_mode=self.sampling_padding_mode,
                )
                refined = refined.to(
                    device=opacities.device,
                    dtype=opacities.dtype,
                )
                opacities = opacities.clone()
                opacities[coverage_mask] = refined[coverage_mask]

        if self.opacity_gamma != 1.0:
            opacities = opacities.clamp(0.0, 1.0).pow(self.opacity_gamma)

        return opacities

    def _merge_index_sets(
        self,
        device: torch.device,
        *index_sets: Optional[Tensor],
    ) -> Tensor:
        """Merge optional index tensors into one unique long tensor."""
        merged = []
        for idx in index_sets:
            if isinstance(idx, torch.Tensor) and idx.numel() > 0:
                merged.append(idx.long().to(device=device).view(-1))

        if not merged:
            return torch.empty(0, dtype=torch.long, device=device)
        if len(merged) == 1:
            return merged[0].unique()
        return torch.unique(torch.cat(merged, dim=0))

    def refresh_cached_appearance(
        self,
        gaussians,
        *,
        intensity_indices: Optional[Tensor] = None,
        opacity_indices: Optional[Tensor] = None,
        force_all: bool = False,
    ) -> Dict[str, int]:
        """Refresh sampled appearance buffers outside the main loss path."""
        counts = {"intensity": 0, "opacity": 0}

        intensity_mode = getattr(gaussians, "intensity_mode", "learned")
        if intensity_mode in {"sampled", "sampled_mean_covered"}:
            idx = None if force_all else intensity_indices
            counts["intensity"] = int(
                gaussians.update_sampled_intensities(
                    sampler=self._intensity_sampler,
                    indices=idx,
                )
            )
            self.last_intensity_update_count = counts["intensity"]

        opacity_mode = getattr(gaussians, "opacity_mode", "sampled")
        if (
            opacity_mode in {"sampled", "sampled_mean_covered"}
            and self.mask_volume is not None
        ):
            idx = None if force_all else opacity_indices
            counts["opacity"] = int(
                gaussians.update_sampled_opacities(
                    sampler=self._opacity_sampler,
                    indices=idx,
                )
            )
            self.last_opacity_update_count = counts["opacity"]

        return counts

    def _resolve_eval_target(self, target: str) -> str:
        """Resolve the requested evaluation target to either mask or ct."""
        target_norm = str(target).lower()
        if target_norm == "auto":
            return "ct" if self.supervision_target in {"ct", "joint"} else "mask"
        if target_norm not in {"mask", "ct"}:
            raise ValueError(
                "eval target must be one of {'auto','mask','ct'}, got "
                f"{target!r}."
            )
        return target_norm

    def _resolve_eval_intensities(self, gaussians, n_points: int, device: torch.device) -> Tensor:
        """Return a scalar intensity buffer suitable for full-model evaluation."""
        intensity_mode = getattr(gaussians, "intensity_mode", "learned")

        if intensity_mode in {"sampled", "sampled_mean_covered"}:
            has_buffer = (
                hasattr(gaussians, "intensities")
                and isinstance(gaussians.intensities, torch.Tensor)
                and gaussians.intensities.numel() > 0
                and gaussians.intensities.shape[0] == n_points
            )
            if not has_buffer:
                self.refresh_cached_appearance(gaussians, force_all=True)

            has_buffer = (
                hasattr(gaussians, "intensities")
                and isinstance(gaussians.intensities, torch.Tensor)
                and gaussians.intensities.numel() > 0
                and gaussians.intensities.shape[0] == n_points
            )
            if has_buffer:
                intensities = gaussians.intensities.detach()
            else:
                intensities = gaussians.ensure_intensity_buffer(
                    n_points,
                    1,
                    device=device,
                    dtype=torch.float32,
                    fill_value=0.5,
                )
                gaussians.intensities = intensities.detach()

            return self._ensure_scalar_point_attribute(
                "intensity",
                intensities,
                n_points,
            )

        if (
            hasattr(gaussians, "_features_dc")
            and gaussians._features_dc is not None
            and gaussians._features_dc.numel() > 0
        ):
            learned = gaussians.learned_intensity_from_features()
            if learned is not None and learned.numel() > 0:
                return self._ensure_scalar_point_attribute(
                    "intensity",
                    learned,
                    n_points,
                )

        has_buffer = (
            hasattr(gaussians, "intensities")
            and isinstance(gaussians.intensities, torch.Tensor)
            and gaussians.intensities.numel() > 0
            and gaussians.intensities.shape[0] == n_points
        )
        if has_buffer:
            return self._ensure_scalar_point_attribute(
                "intensity",
                gaussians.intensities.detach(),
                n_points,
            )

        fallback = gaussians.ensure_intensity_buffer(
            n_points,
            1,
            device=device,
            dtype=torch.float32,
            fill_value=0.5,
        )
        gaussians.intensities = fallback.detach()
        return self._ensure_scalar_point_attribute("intensity", fallback, n_points)

    @torch.no_grad()
    def compute_full_roi_masked_mse(
        self,
        gaussians,
        *,
        target: str = "auto",
        working_grid_downscale_factor: int = 1,
        refresh_appearance: bool = True,
    ) -> Tuple[float, str]:
        """Evaluate full-model masked MSE inside the ROI on the current volume grid."""
        resolved_target = self._resolve_eval_target(target)

        if refresh_appearance:
            self.refresh_cached_appearance(gaussians, force_all=True)

        xyz = gaussians.get_xyz
        scaling = gaussians.get_scaling
        rotation = gaussians.get_rotation
        n_points = xyz.shape[1] if xyz.shape[0] == 3 else xyz.shape[0]

        use_opacity = self._ensure_scalar_point_attribute(
            "opacity",
            gaussians.get_opacity,
            n_points,
        )

        bounds_min = self.bounds_min.to(xyz.device)
        bounds_max = self.bounds_max.to(xyz.device)
        roi_shape = self.roi_shape
        mask_roi = self.mask_bool_roi.to(device=xyz.device)
        eval_downscale = max(1, int(working_grid_downscale_factor))

        if resolved_target == "mask":
            pred = splat_to_volume(
                points=xyz,
                point_scales=scaling,
                point_rotations=rotation,
                point_opacities=use_opacity,
                point_intensities=None,
                volume_shape=roi_shape,
                device=xyz.device,
                active_idx=None,
                grid_bounds=(bounds_min, bounds_max),
                render_mode="density",
                density_scale=float(getattr(self, "density_scale", 1.0)),
                working_grid_downscale_factor=eval_downscale,
                sparse_support_cutoff=self.sparse_support_cutoff,
                sparse_max_radius_vox=self.sparse_max_radius_vox,
                sparse_support_softness=self.sparse_support_softness,
                render_min_sigma_vox=self.render_min_sigma_vox,
            )
            target_roi = self.mask_volume_roi.to(device=pred.device, dtype=pred.dtype)
        else:
            use_intensities = self._resolve_eval_intensities(gaussians, n_points, xyz.device)
            pred = splat_to_volume(
                points=xyz,
                point_scales=scaling,
                point_rotations=rotation,
                point_opacities=use_opacity,
                point_intensities=use_intensities,
                volume_shape=roi_shape,
                device=xyz.device,
                active_idx=None,
                grid_bounds=(bounds_min, bounds_max),
                render_mode="intensity",
                density_scale=float(getattr(self, "density_scale", 1.0)),
                working_grid_downscale_factor=eval_downscale,
                sparse_support_cutoff=self.sparse_support_cutoff,
                sparse_max_radius_vox=self.sparse_max_radius_vox,
                sparse_support_softness=self.sparse_support_softness,
                render_min_sigma_vox=self.render_min_sigma_vox,
            )
            target_ct_roi = self.volume_gt_roi.to(device=pred.device, dtype=pred.dtype)
            denom = max(self.global_intensity_max - self.global_intensity_min, 1e-8)
            if denom <= 1e-8:
                target_roi = torch.full_like(target_ct_roi, 0.5)
            else:
                target_roi = (
                    (target_ct_roi - float(self.global_intensity_min)) / float(denom)
                ).clamp_(0.0, 1.0)

        pred_vals = pred[mask_roi]
        tgt_vals = target_roi[mask_roi]
        mse = (pred_vals - tgt_vals).square().mean()
        return float(mse.item()), resolved_target

    def _ensure_scalar_point_attribute(
        self,
        name: str,
        tensor: Tensor,
        n_points: int,
    ) -> Tensor:
        """Validate and reshape a per-point scalar attribute to [N, 1]."""
        if tensor is None:
            raise ValueError(f"{name} tensor is missing")

        if tensor.dim() == 1 and tensor.shape[0] == n_points:
            return tensor.view(n_points, 1)

        if tensor.dim() == 2 and tensor.shape[0] == n_points and tensor.shape[1] == 1:
            return tensor

        if tensor.numel() == n_points:
            return tensor.reshape(n_points, 1)

        raise RuntimeError(
            f"{name} shape mismatch: expected {n_points} scalar values, "
            f"got shape {tuple(tensor.shape)} with {tensor.numel()} values"
        )

    def _ensure_orientation_field(self) -> None:
        """Compute and cache gradient field if needed."""
        if self._orientation_grad is not None:
            return
        source = self._orientation_source().to(self.device)

        # Print source statistics for debugging
        if self.verbose:
            print(
                "Computing orientation field from intensity volume "
                f"(range: [{source.min().item():.4f}, {source.max().item():.4f}])"
            )

        grad, mag = compute_gradient_field(
            source,
            sigma_pre=self.orientation_sigma_grad,
            sigma_post=self.orientation_sigma_tensor,
        )
        self._orientation_grad = grad
        self._orientation_mag = mag

        # Print gradient field statistics
        if self.verbose:
            print(
                "Gradient magnitude range: "
                f"[{mag.min().item():.6f}, {mag.max().item():.6f}], mean: {mag.mean().item():.6f}"
            )
            print(
                "Orientation field computed "
                f"(sigma_pre={self.orientation_sigma_grad}, sigma_post={self.orientation_sigma_tensor})"
            )

    def _ensure_structure_field(self) -> None:
        """Compute quaternion/vesselness fields when a mask is available."""
        if self._structure_quat is not None or self.mask_volume is None:
            return

        quat_field, vessel_field = build_structure_field(
            self.mask_volume,
            mask_threshold=self.structure_mask_threshold,
            sigma_pre=self.structure_sigma,
        )
        self._structure_quat = quat_field
        self._structure_vesselness = vessel_field

    def get_quat_for_points(self, xyz_world: Tensor) -> Tuple[Tensor, int]:
        """Return orientation quaternions and fallback count for points."""
        if xyz_world.numel() == 0:
            print("Warning: Empty point set provided for orientation query.")
            return torch.empty(0, 4, device=self.device), 0

        self._ensure_orientation_field()

        # Debug: Print world coordinate statistics
        if self.verbose:
            print(f"[get_quat_for_points] Processing {xyz_world.shape[0]} points")
            print(
                f"[get_quat_for_points] World coords: x=[{xyz_world[:, 0].min():.4f}, {xyz_world[:, 0].max():.4f}], "
                f"y=[{xyz_world[:, 1].min():.4f}, {xyz_world[:, 1].max():.4f}], "
                f"z=[{xyz_world[:, 2].min():.4f}, {xyz_world[:, 2].max():.4f}]"
            )
            print(
                f"[get_quat_for_points] Origin: {self.volume_origin.tolist()}, Voxel size: {self.voxel_size.tolist()}"
            )

        ijk = world_to_voxel(xyz_world, self.volume_origin, self.voxel_size)
        rotmats, fallback = gather_rotation_from_gradient(
            self._orientation_grad, self._orientation_mag, ijk, eps=1e-6
        )
        quats = rotmat_to_quat(rotmats)
        quats = random_quat_perturb(quats, self.orientation_perturb_deg)
        return quats, int(fallback.sum().item())

    def export_orientation_field(self) -> Dict[str, Tensor]:
        """Expose cached orientation data for reuse by the Gaussian model."""
        self._ensure_orientation_field()
        payload = {
            "gradient": self._orientation_grad,
            "magnitude": self._orientation_mag,
            "origin": self.volume_origin,
            "voxel_size": self.voxel_size,
            "perturb_deg": torch.tensor(
                self.orientation_perturb_deg, device=self.device
            ),
        }

        # Structure fields are intentionally NOT forced here: dense [D,H,W] structure
        # grids can be extremely memory-heavy at native CT resolution. If structure
        # was computed elsewhere, include it; otherwise keep export lightweight.
        if self._structure_quat is not None and self._structure_vesselness is not None:
            payload["structure_quat"] = self._structure_quat
            payload["structure_vesselness"] = self._structure_vesselness
        return payload

    def get_structure_for_points(self, xyz_world: Tensor) -> Tuple[Tensor, Tensor]:
        """Sample Hessian-based orientation quaternions and vesselness values at points.

        Uses a lightweight pointwise Hessian estimate to avoid building a dense
        structure field for the entire volume.
        """
        if xyz_world.numel() == 0:
            empty = torch.empty(0, 1, device=self.device)
            return torch.empty(0, 4, device=self.device), empty

        if self.mask_volume is None or self.mask_volume.numel() == 0:
            empty = torch.zeros(xyz_world.shape[0], 1, device=self.device)
            identity = torch.zeros(xyz_world.shape[0], 4, device=self.device)
            identity[:, 0] = 1.0
            return identity, empty

        ijk = world_to_voxel(xyz_world, self.volume_origin, self.voxel_size)
        quats, vessel = structure_from_mask_at_ijk(
            self.mask_volume,
            ijk,
            mask_threshold=float(self.structure_mask_threshold),
            sigma_pre=float(self.structure_sigma),
        )
        return quats, vessel

    def compute_loss(
        self,
        gaussians,
        active_idx: Optional[Tensor] = None,
        total_points: Optional[int] = None,
        *,
        compute_volume_gradients: bool = False,
        volume_gradient_interval: int = 10,
    ) -> Tuple[Tensor, Dict[str, float], Tensor]:
        """
        Compute volume supervision loss for current gaussians.
        
        Args:
            gaussians: Current gaussian model
            
        Returns:
            Tuple of (loss tensor, metrics dict, volume_gradients)
        """
        # Check if xyz requires gradients
        xyz = gaussians.get_xyz

        # Ensure parameters require gradients
        if not xyz.requires_grad:
            # Enable requires_grad without breaking optimizer reference
            gaussians._xyz.requires_grad_(True)
            xyz = gaussians._xyz

        # Get scaling and rotation values
        scaling = gaussians.get_scaling
        rotation = gaussians.get_rotation

        self._step += 1
        self.iteration = getattr(self, "iteration", 0) + 1
        self.last_mean_covered_intensity_count = 0
        self.last_mean_covered_opacity_count = 0

        n_points = xyz.shape[1] if xyz.shape[0] == 3 else xyz.shape[0]
        intensity_mode = getattr(gaussians, "intensity_mode", "learned")
        opacity_mode = getattr(gaussians, "opacity_mode", "sampled")
        pending_refresh_idx = torch.empty(0, dtype=torch.long, device=xyz.device)
        if (
            intensity_mode in {"sampled", "sampled_mean_covered"}
            or opacity_mode in {"sampled", "sampled_mean_covered"}
        ) and hasattr(gaussians, "consume_pending_appearance_indices"):
            pending_refresh_idx = gaussians.consume_pending_appearance_indices()

        gaussians.reference_volume = self.volume_color
        if self.mask_volume is not None:
            gaussians.reference_mask = self.mask_volume

        use_intensities: Tensor
        if intensity_mode in {"sampled", "sampled_mean_covered"}:
            if self.mask_volume is not None:
                gaussians.reference_mask = self.mask_volume

            needs_resize = (
                not hasattr(gaussians, "intensities")
                or gaussians.intensities.numel() == 0
                or gaussians.intensities.shape[0] != n_points
            )

            is_mean_mode = intensity_mode == "sampled_mean_covered"
            interval = (
                getattr(gaussians, "mean_covered_interval", 1)
                if is_mean_mode
                else self.intensity_update_interval
            )
            interval = max(int(interval), 1)
            update_due = ((self._step - 1) % interval) == 0

            dirty_subset = torch.empty(0, dtype=torch.long, device=xyz.device)
            if active_idx is not None and active_idx.numel() > 0:
                dirty_subset = gaussians.dirty_indices(
                    active_idx,
                    self.dirty_threshold_xyz,
                    self.dirty_threshold_scale,
                    self.dirty_threshold_rot,
                )
            global_dirty_subset = torch.empty(0, dtype=torch.long, device=xyz.device)
            if update_due:
                global_dirty_subset = gaussians.dirty_indices(
                    None,
                    self.dirty_threshold_xyz,
                    self.dirty_threshold_scale,
                    self.dirty_threshold_rot,
                )
            dirty_subset = self._merge_index_sets(
                xyz.device,
                dirty_subset,
                pending_refresh_idx,
                global_dirty_subset,
            )

            indices_for_update: Optional[Tensor]
            if needs_resize:
                indices_for_update = None
            else:
                if is_mean_mode:
                    large_mask = gaussians.large_splat_mask(
                        getattr(gaussians, "intensity_large_splat_threshold", 0.0)
                    )
                    large_mask = large_mask.to(device=xyz.device)
                    if active_idx is not None and active_idx.numel() > 0:
                        active_idx_long = active_idx.long().to(device=xyz.device)
                        subset_mask = large_mask[active_idx_long]
                        candidate = active_idx_long[subset_mask]
                        if candidate.numel() == 0:
                            candidate = torch.nonzero(large_mask, as_tuple=False).view(
                                -1
                            )
                    else:
                        candidate = torch.nonzero(large_mask, as_tuple=False).view(-1)

                    if dirty_subset.numel() > 0:
                        indices_for_update = dirty_subset
                    elif update_due and candidate.numel() > 0:
                        if getattr(self, "debug_intensity", False):
                            total_large = int(large_mask.sum().item())
                            print(
                                f"[Intensity] mean-covered refresh updating {int(candidate.numel())}"
                                f" large splats (total tracked: {total_large})."
                            )
                        indices_for_update = candidate
                    elif update_due:
                        indices_for_update = active_idx
                    else:
                        indices_for_update = None
                else:
                    if dirty_subset.numel() > 0:
                        indices_for_update = dirty_subset
                    elif update_due:
                        indices_for_update = active_idx
                    else:
                        indices_for_update = None

            if indices_for_update is not None or needs_resize:
                updated = gaussians.update_sampled_intensities(
                    sampler=self._intensity_sampler,
                    indices=indices_for_update,
                )
                self.last_intensity_update_count = updated
            else:
                self.last_intensity_update_count = 0

            has_prev = (
                hasattr(gaussians, "intensities")
                and isinstance(gaussians.intensities, torch.Tensor)
                and gaussians.intensities.numel() > 0
            )
            channels = gaussians.intensities.shape[1] if has_prev else 1
            dtype = gaussians.intensities.dtype if has_prev else xyz.dtype
            use_intensities = gaussians.ensure_intensity_buffer(
                n_points,
                channels,
                device=xyz.device,
                dtype=dtype,
                fill_value=0.5,
            )
            gaussians.intensities = use_intensities.detach()
            gaussians.intensities.requires_grad_(False)
            gaussians.volume_min = self.global_intensity_min
            gaussians.volume_max = self.global_intensity_max
            use_intensities = gaussians.intensities
            if (
                self.verbose
                and self._step % 200 == 0
                and use_intensities.numel() > 0
            ):
                batch_min = float(use_intensities.min().item())
                batch_max = float(use_intensities.max().item())
                print(
                    (
                        f"[Intensity] global_min={self.global_intensity_min:.4f}, "
                        f"global_max={self.global_intensity_max:.4f}, "
                        f"sampled_batch=[{batch_min:.4f},{batch_max:.4f}]"
                    )
                )
        else:
            needs_resize = (
                not hasattr(gaussians, "intensities")
                or gaussians.intensities is None
                or gaussians.intensities.numel() == 0
                or gaussians.intensities.shape[0] != n_points
            )

            interval = max(int(self.intensity_update_interval), 1)
            update_due = ((self._step - 1) % interval) == 0

            dirty_subset = torch.empty(0, dtype=torch.long, device=xyz.device)
            if active_idx is not None and active_idx.numel() > 0:
                dirty_subset = gaussians.dirty_indices(
                    active_idx,
                    self.dirty_threshold_xyz,
                    self.dirty_threshold_scale,
                    self.dirty_threshold_rot,
                )
            global_dirty_subset = torch.empty(0, dtype=torch.long, device=xyz.device)
            if update_due:
                global_dirty_subset = gaussians.dirty_indices(
                    None,
                    self.dirty_threshold_xyz,
                    self.dirty_threshold_scale,
                    self.dirty_threshold_rot,
                )
            dirty_subset = self._merge_index_sets(
                xyz.device,
                dirty_subset,
                global_dirty_subset,
            )

            indices_for_update: Optional[Tensor]
            if needs_resize:
                indices_for_update = None
            elif dirty_subset.numel() > 0:
                indices_for_update = dirty_subset
            elif update_due:
                indices_for_update = active_idx
            else:
                indices_for_update = None

            if indices_for_update is not None or needs_resize:
                sampled_buffer = self._intensity_sampler(gaussians, indices_for_update)
                if sampled_buffer is not None and sampled_buffer.numel() > 0:
                    sampled_buffer = sampled_buffer.detach()
                    channels = sampled_buffer.shape[1]
                    intensity_buffer = gaussians.ensure_intensity_buffer(
                        n_points,
                        channels,
                        device=xyz.device,
                        dtype=sampled_buffer.dtype,
                        fill_value=0.5,
                    )
                    with torch.no_grad():
                        if indices_for_update is None:
                            intensity_buffer.copy_(sampled_buffer)
                            gaussians.snapshot_params_for_dirty_check(None)
                            self.last_intensity_update_count = int(n_points)
                        else:
                            idx_update = indices_for_update.long()
                            intensity_buffer[idx_update] = sampled_buffer
                            gaussians.snapshot_params_for_dirty_check(idx_update)
                            self.last_intensity_update_count = int(idx_update.numel())
                    gaussians.intensities = intensity_buffer.detach()
                else:
                    self.last_intensity_update_count = 0
            else:
                self.last_intensity_update_count = 0

            if (
                hasattr(gaussians, "intensities")
                and gaussians.intensities is not None
                and gaussians.intensities.numel() > 0
            ):
                gaussians.intensities = gaussians.intensities.detach()
                gaussians.intensities.requires_grad_(False)
                gaussians.volume_min = self.global_intensity_min
                gaussians.volume_max = self.global_intensity_max

            if (
                hasattr(gaussians, "_features_dc")
                and gaussians._features_dc is not None
                and gaussians._features_dc.numel() > 0
            ):
                learned_intensity = gaussians.learned_intensity_from_features()
                if learned_intensity is not None and learned_intensity.numel() > 0:
                    use_intensities = learned_intensity
                else:
                    use_intensities = torch.full(
                        (n_points, 1),
                        0.5,
                        device=xyz.device,
                        dtype=xyz.dtype,
                    )
            else:
                if (
                    not hasattr(gaussians, "intensities")
                    or gaussians.intensities.numel() == 0
                    or gaussians.intensities.shape[0] != n_points
                ):
                    gaussians.ensure_intensity_buffer(
                        n_points,
                        1,
                        device=xyz.device,
                        dtype=xyz.dtype,
                        fill_value=0.5,
                    )
                gaussians.intensities = gaussians.intensities.detach()
                gaussians.intensities.requires_grad_(False)
                use_intensities = gaussians.intensities

        # --- Opacity refresh (independent of intensity mode) ---
        if (
            opacity_mode in {"sampled", "sampled_mean_covered"}
            and self.mask_volume is not None
        ):
            gaussians.opacity_gamma = float(getattr(self, "opacity_gamma", 1.0))

            needs_resize = (
                not hasattr(gaussians, "opacities")
                or gaussians.opacities is None
                or gaussians.opacities.numel() == 0
                or gaussians.opacities.shape[0] != n_points
            )

            is_mean_mode = opacity_mode == "sampled_mean_covered"
            interval = (
                getattr(gaussians, "mean_covered_interval", 1)
                if is_mean_mode
                else self.opacity_update_interval
            )
            interval = max(int(interval), 1)
            update_due = ((self._step - 1) % interval) == 0

            dirty_subset = torch.empty(0, dtype=torch.long, device=xyz.device)
            if active_idx is not None and active_idx.numel() > 0:
                dirty_subset = gaussians.dirty_indices(
                    active_idx,
                    self.dirty_threshold_xyz,
                    self.dirty_threshold_scale,
                    self.dirty_threshold_rot,
                )
            global_dirty_subset = torch.empty(0, dtype=torch.long, device=xyz.device)
            if update_due:
                global_dirty_subset = gaussians.dirty_indices(
                    None,
                    self.dirty_threshold_xyz,
                    self.dirty_threshold_scale,
                    self.dirty_threshold_rot,
                )
            dirty_subset = self._merge_index_sets(
                xyz.device,
                dirty_subset,
                pending_refresh_idx,
                global_dirty_subset,
            )

            indices_for_update: Optional[Tensor]
            if needs_resize:
                indices_for_update = None
            else:
                if is_mean_mode:
                    large_mask = gaussians.large_splat_mask(
                        getattr(gaussians, "intensity_large_splat_threshold", 0.0)
                    ).to(device=xyz.device)
                    if active_idx is not None and active_idx.numel() > 0:
                        active_idx_long = active_idx.long().to(device=xyz.device)
                        subset_mask = large_mask[active_idx_long]
                        candidate = active_idx_long[subset_mask]
                        if candidate.numel() == 0:
                            candidate = torch.nonzero(large_mask, as_tuple=False).view(-1)
                    else:
                        candidate = torch.nonzero(large_mask, as_tuple=False).view(-1)

                    if dirty_subset.numel() > 0:
                        indices_for_update = dirty_subset
                    elif update_due and candidate.numel() > 0:
                        indices_for_update = candidate
                    elif update_due:
                        indices_for_update = active_idx
                    else:
                        indices_for_update = None
                else:
                    if dirty_subset.numel() > 0:
                        indices_for_update = dirty_subset
                    elif update_due:
                        indices_for_update = active_idx
                    else:
                        indices_for_update = None

            if indices_for_update is not None or needs_resize:
                self.last_opacity_update_count = int(
                    gaussians.update_sampled_opacities(
                        sampler=self._opacity_sampler,
                        indices=indices_for_update,
                    )
                )
            else:
                self.last_opacity_update_count = 0
        else:
            self.last_opacity_update_count = 0

        # Convert gaussians to volume using intensity values (or density for mask supervision)
        # Opacity is provided via gaussians.get_opacity, which is mode-aware.
        use_opacity = gaussians.get_opacity

        # FIX: Ensure opacity and intensity tensors have correct shape to match number of points
        # Get the number of points from xyz (whether [3, N] or [N, 3])
        n_points = xyz.shape[1] if xyz.shape[0] == 3 else xyz.shape[0]

        use_opacity = self._ensure_scalar_point_attribute(
            "opacity",
            use_opacity,
            n_points,
        )

        if total_points is None:
            total_points = n_points

        if (
            getattr(self, "debug_intensity", False)
            and intensity_mode in {"sampled", "sampled_mean_covered"}
            and use_intensities.numel() > 0
        ):
            if not torch.isfinite(use_intensities).all():
                raise AssertionError("Sampled intensities contain non-finite values")
            min_val = float(use_intensities.min().item())
            max_val = float(use_intensities.max().item())
            if min_val < -1e-4 or max_val > 1.0 + 1e-4:
                raise AssertionError(
                    f"Sampled intensities out of [0,1] range: [{min_val:.4f}, {max_val:.4f}]"
                )

        use_intensities = self._ensure_scalar_point_attribute(
            "intensity",
            use_intensities,
            n_points,
        )

        # Debug tensor shapes is no longer needed

        # Compute predictions on the full supervision volume (no ROI cropping).
        roi_shape = self.roi_shape
        bounds_min = self.bounds_min.to(xyz.device)
        bounds_max = self.bounds_max.to(xyz.device)

        checkpoint_ok = bool(getattr(self, "enable_render_checkpoint", True))

        render_use_amp = bool(getattr(self, "render_use_amp", False))
        render_amp_dtype = getattr(self, "render_amp_dtype", torch.float16)
        raster_params = self._effective_raster_params()

        def _render_density(points, scales, rotations, opacities):
            # IMPORTANT: checkpoint recompute happens during backward, outside the
            # training-loop autocast context. Re-enable autocast here so recompute
            # uses the same dtype and avoids unnecessary peak memory.
            with torch.cuda.amp.autocast(
                enabled=render_use_amp,
                dtype=render_amp_dtype if render_use_amp else None,
            ):
                return splat_to_volume(
                    points=points,
                    point_scales=scales,
                    point_rotations=rotations,
                    point_opacities=opacities,
                    point_intensities=None,
                    volume_shape=roi_shape,
                    device=xyz.device,
                    active_idx=active_idx,
                    grid_bounds=(bounds_min, bounds_max),
                    render_mode="density",
                    density_scale=float(getattr(self, "density_scale", 1.0)),
                    working_grid_downscale_factor=self.volume_render_downscale_factor,
                    sparse_support_cutoff=raster_params["sparse_support_cutoff"],
                    sparse_max_radius_vox=self.sparse_max_radius_vox,
                    sparse_support_softness=raster_params["sparse_support_softness"],
                    render_min_sigma_vox=raster_params["render_min_sigma_vox"],
                )

        def _render_intensity(points, scales, rotations, opacities, intensities):
            with torch.cuda.amp.autocast(
                enabled=render_use_amp,
                dtype=render_amp_dtype if render_use_amp else None,
            ):
                return splat_to_volume(
                    points=points,
                    point_scales=scales,
                    point_rotations=rotations,
                    point_opacities=opacities,
                    point_intensities=intensities,
                    volume_shape=roi_shape,
                    device=xyz.device,
                    active_idx=active_idx,
                    grid_bounds=(bounds_min, bounds_max),
                    render_mode="intensity",
                    density_scale=float(getattr(self, "density_scale", 1.0)),
                    working_grid_downscale_factor=self.volume_render_downscale_factor,
                    sparse_support_cutoff=raster_params["sparse_support_cutoff"],
                    sparse_max_radius_vox=self.sparse_max_radius_vox,
                    sparse_support_softness=raster_params["sparse_support_softness"],
                    render_min_sigma_vox=raster_params["render_min_sigma_vox"],
                )

        volume_pred_mask_roi: Optional[Tensor] = None
        volume_pred_ct_roi: Optional[Tensor] = None

        if self.supervision_target in {"mask", "joint"}:
            density_inputs = (xyz, scaling, rotation, use_opacity)
            if checkpoint_ok and any(t.requires_grad for t in density_inputs):
                volume_pred_mask_roi = checkpoint(
                    _render_density, *density_inputs, use_reentrant=False
                )
            else:
                volume_pred_mask_roi = _render_density(*density_inputs)

        if self.supervision_target in {"ct", "joint"}:
            intensity_inputs = (xyz, scaling, rotation, use_opacity, use_intensities)
            if checkpoint_ok and any(t.requires_grad for t in intensity_inputs):
                volume_pred_ct_roi = checkpoint(
                    _render_intensity, *intensity_inputs, use_reentrant=False
                )
            else:
                volume_pred_ct_roi = _render_intensity(*intensity_inputs)

        # Optionally retain grad for debugging
        if getattr(self, "debug", False):
            xyz.retain_grad()

        # Debug if needed
        # Slice targets/mask to ROI for loss.
        # Use whichever prediction exists for device/dtype alignment.
        ref_pred = volume_pred_mask_roi if volume_pred_mask_roi is not None else volume_pred_ct_roi
        assert ref_pred is not None
        mask_roi = self.mask_bool_roi.to(device=ref_pred.device)

        # Store predicted volume for visualization only when needed.
        # For joint mode, we store the mask/density branch for compatibility.
        if getattr(self, "iteration", 0) % 1000 == 0:
            full_pred = torch.zeros(
                self.volume_shape,
                device=ref_pred.device,
                dtype=ref_pred.dtype,
            )
            insert = (
                volume_pred_mask_roi
                if volume_pred_mask_roi is not None
                else volume_pred_ct_roi
            )
            assert insert is not None
            zsl, ysl, xsl = self._roi_slices
            full_pred[zsl, ysl, xsl] = insert
            self.volume_pred = full_pred.detach().clone()

        mask_loss = None
        ct_loss = None

        if self.supervision_target in {"mask", "joint"}:
            assert volume_pred_mask_roi is not None
            target_mask_roi = self.mask_volume_roi.to(device=volume_pred_mask_roi.device)
            if self.mask_criterion.loss_type == "mse":
                pred_vals = volume_pred_mask_roi[mask_roi]
                tgt_vals = target_mask_roi[mask_roi]
                diff = pred_vals - tgt_vals
                mask_loss = (diff * diff).mean()
            else:
                masked_pred = volume_pred_mask_roi * mask_roi.to(dtype=volume_pred_mask_roi.dtype)
                masked_tgt = target_mask_roi * mask_roi.to(dtype=target_mask_roi.dtype)
                mask_loss = self.mask_criterion(masked_pred, masked_tgt)

        if self.supervision_target in {"ct", "joint"}:
            assert volume_pred_ct_roi is not None
            target_ct_roi = self.volume_gt_roi.to(device=volume_pred_ct_roi.device)
            denom = max(self.global_intensity_max - self.global_intensity_min, 1e-8)
            if denom <= 1e-8:
                target_ct_norm = torch.full_like(target_ct_roi, 0.5)
            else:
                target_ct_norm = (target_ct_roi - float(self.global_intensity_min)) / float(denom)
                target_ct_norm = target_ct_norm.clamp_(0.0, 1.0)

            if self.ct_criterion.loss_type == "mse":
                pred_vals = volume_pred_ct_roi[mask_roi]
                tgt_vals = target_ct_norm[mask_roi]
                diff = pred_vals - tgt_vals
                ct_loss = (diff * diff).mean()
            else:
                masked_pred = volume_pred_ct_roi * mask_roi.to(dtype=volume_pred_ct_roi.dtype)
                masked_tgt = target_ct_norm * mask_roi.to(dtype=target_ct_norm.dtype)
                ct_loss = self.ct_criterion(masked_pred, masked_tgt)

        loss = torch.zeros((), device=ref_pred.device, dtype=ref_pred.dtype)
        if mask_loss is not None:
            loss = loss + float(self.mask_loss_weight) * mask_loss
        if ct_loss is not None:
            loss = loss + float(self.ct_loss_weight) * ct_loss

        outside_loss = None
        outside_weight = float(getattr(self, "outside_mask_weight", 0.0))
        if outside_weight > 0.0 and self.supervision_target in {"mask", "joint"}:
            outside_roi = ~mask_roi
            if outside_roi.any() and volume_pred_mask_roi is not None:
                outside_vals = volume_pred_mask_roi[outside_roi]
                outside_loss = (outside_vals * outside_vals).mean()
                loss = loss + outside_weight * outside_loss

        unweighted_loss = loss

        # Scale loss by weight
        if self.loss_weight != 1.0:
            loss = loss * self.loss_weight

        # Optionally compute gradients of loss w.r.t. xyz periodically (for analysis/alignment)
        volume_grads = None
        if compute_volume_gradients:
            interval = max(int(volume_gradient_interval), 1)
            if hasattr(self, "iteration") and (self.iteration % interval) == 0:
                grad_list = torch.autograd.grad(
                    loss, xyz, retain_graph=True, allow_unused=True
                )
                volume_grads = grad_list[0]
                self.volume_gradients = volume_grads
            else:
                volume_grads = getattr(self, "volume_gradients", None)

        # Update metrics
        with torch.no_grad():
            self.metrics["volume_loss"] = float(loss.item())
            self.metrics["volume_loss_unweighted"] = float(unweighted_loss.item())
            self.metrics["mask_loss"] = float(mask_loss.item()) if mask_loss is not None else 0.0
            self.metrics["ct_loss"] = float(ct_loss.item()) if ct_loss is not None else 0.0
            self.metrics["active_points"] = float(
                active_idx.numel() if active_idx is not None else total_points
            )
            self.metrics["intensity_update_count"] = float(
                self.last_intensity_update_count
            )
            self.metrics["opacity_update_count"] = float(
                self.last_opacity_update_count
            )
            self.metrics["mean_covered_intensity_count"] = float(
                self.last_mean_covered_intensity_count
            )
            self.metrics["mean_covered_opacity_count"] = float(
                self.last_mean_covered_opacity_count
            )
            if outside_loss is not None:
                self.metrics["outside_mask_loss"] = float(outside_loss.item())
            else:
                self.metrics["outside_mask_loss"] = 0.0
            if mask_loss is not None and self.mask_criterion.loss_type == "dice":
                self.metrics["dice_score"] = 1.0 - float(mask_loss.item())
            else:
                self.metrics["dice_score"] = 0.0

        # Return both loss and volume gradients for parameter diversity losses
        return loss, self.metrics.copy(), volume_grads

    def log_metrics(self, writer, iteration: int):
        """Log current metrics to tensorboard."""
        if writer is not None:
            for name, value in self.metrics.items():
                writer.add_scalar(f'volume/{name}', value, iteration)

            # Log volume visualizations periodically
            if iteration % 1000 == 0:
                writer.add_image('volume/ground_truth',
                               self.volume_gt[None, None],
                               iteration)
                writer.add_image('volume/prediction',
                               self.volume_pred[None, None],
                               iteration)
