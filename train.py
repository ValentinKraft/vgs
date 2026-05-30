#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import sys
import numpy as np
from dataclasses import dataclass
from scene.gaussian_model import GaussianModel
from scene.volume_scene import VolumeScene
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from typing import Optional
from arguments import (
    ExportParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    TrainingScriptParams,
)
from gaussian_splatting.utils.parameter_monitor import (
    ParameterMonitor,
    add_parameter_regularization_loss,
)
from gaussian_splatting.utils.volume_supervisor import VolumeSupervisor
from gaussian_splatting.utils.ambient_occlusion import compute_ao_volume_from_mask
from gaussian_splatting.utils.intensity_sampler import sample_intensities_from_volume
from torch.cuda.amp import autocast, GradScaler


DEFAULT_MAX_POINTS_PER_ITER = 10000  # Upper bound of splats per forward pass to limit memory


@dataclass
class MedicalPresetState:
    """Container describing how medical presets modify the training loop."""

    mode: str
    active: bool
    diversity_enabled: bool
    diagnostics_enabled: bool
    densification_enabled: bool
    scale_constraints_enabled: bool
    init_points: int


@dataclass
class ActiveSubsetState:
    """Track fair coverage when active-point sampling is memory capped."""

    order: Optional[torch.Tensor] = None
    cursor: int = 0
    total_points: int = 0


def _configure_medical_presets(args: Namespace, opt) -> MedicalPresetState:
    """Apply organ/vessel presets and return the resulting state."""

    mode = getattr(args, "medical_mode", "none")
    active = mode in ("organ", "vessel", "vessel_anisotropy")
    diversity_enabled = bool(getattr(args, "enable_diversity", False))
    diagnostics_enabled = bool(getattr(args, "enable_diagnostics", False))
    enable_densification = bool(getattr(args, "enable_densification", False))
    disable_densification = bool(getattr(args, "disable_densification", False))
    densification_enabled = bool(enable_densification and not disable_densification)
    init_points = getattr(args, "init_n_points", 0)

    state = MedicalPresetState(
        mode=mode,
        active=active,
        diversity_enabled=diversity_enabled,
        diagnostics_enabled=diagnostics_enabled,
        densification_enabled=densification_enabled,
        scale_constraints_enabled=True,
        init_points=init_points,
    )

    if not state.densification_enabled:
        opt.densify_from_iter = max(opt.iterations + 1, opt.densify_from_iter)
        opt.densify_until_iter = opt.iterations

    if not active:
        return state

    target_points = 8000 if mode == "organ" else 6000
    if hasattr(args, "init_n_points") and args.init_n_points < target_points:
        args.init_n_points = target_points
    state.init_points = getattr(args, "init_n_points", target_points)

    if not diversity_enabled:
        opt.diversity_warmup_iterations = 0
        opt.diversity_scale_weight = 0.0
        opt.diversity_rotation_weight = 0.0
        opt.diversity_scale_range_weight = 0.0
        opt.diversity_target_range_weight = 0.0
        opt.diversity_rotation_entropy_weight = 0.0
        opt.diversity_dispersion_weight = 0.0
        opt.diversity_alignment_weight = 0.0
        state.scale_constraints_enabled = False
        if hasattr(args, "scale_l2_weight"):
            args.scale_l2_weight = 0.0
        if hasattr(opt, "scale_l2_weight"):
            opt.scale_l2_weight = 0.0
    else:
        state.scale_constraints_enabled = True

    # Allow densification for both organ and vessel presets when enabled.
    state.densification_enabled = densification_enabled
    # if state.densification_enabled:
    # opt.densification_interval = max(opt.densification_interval, 200)
    # opt.densify_grad_threshold = max(opt.densify_grad_threshold, 5e-4)
    # opt.densify_from_iter = max(opt.densify_from_iter, 400)
    # opt.densify_until_iter = min(
    #     opt.densify_until_iter, opt.iterations, opt.densify_from_iter + 2000
    # )

    if disable_densification and state.densification_enabled:
        state.densification_enabled = False
        opt.densify_from_iter = max(opt.iterations + 1, opt.densify_from_iter)
        opt.densify_until_iter = opt.iterations

    if mode == "vessel_anisotropy":
        if hasattr(args, "anisotropy_strength"):
            args.anisotropy_strength = 2.0
        if hasattr(args, "init_anisotropy_ratio"):
            args.init_anisotropy_ratio = 3.5
        if hasattr(args, "anisotropy_reg_weight"):
            args.anisotropy_reg_weight = 0.02
        if hasattr(args, "anisotropy_target_ratio"):
            args.anisotropy_target_ratio = 3.0
        if hasattr(args, "anisotropy_reg_warmup_iters"):
            args.anisotropy_reg_warmup_iters = 200

        if hasattr(opt, "anisotropy_reg_weight"):
            opt.anisotropy_reg_weight = 0.02
        if hasattr(opt, "anisotropy_target_ratio"):
            opt.anisotropy_target_ratio = 3.0
        if hasattr(opt, "anisotropy_reg_warmup_iters"):
            opt.anisotropy_reg_warmup_iters = 200

    return state

def _select_active_indices(
    xyz: torch.Tensor,
    max_points_per_iter: int,
    state: ActiveSubsetState,
) -> tuple[Optional[torch.Tensor], int]:
    """Return a capped active subset while cycling through all points over time."""
    if xyz.dim() != 2:
        total = xyz.shape[0]
        state.order = None
        state.cursor = 0
        state.total_points = total
        return None, total

    if xyz.shape[0] == 3 and xyz.shape[1] != 3:
        total = xyz.shape[1]
    else:
        total = xyz.shape[0]

    if total <= max_points_per_iter:
        state.order = None
        state.cursor = 0
        state.total_points = total
        return None, total

    device = xyz.device
    if (
        state.order is None
        or state.total_points != total
        or state.order.device != device
        or state.order.numel() != total
    ):
        state.order = torch.randperm(total, device=device)
        state.cursor = 0
        state.total_points = total

    remaining = total - state.cursor
    if remaining >= max_points_per_iter:
        idx = state.order[state.cursor : state.cursor + max_points_per_iter]
        state.cursor += max_points_per_iter
        return idx, total

    tail = state.order[state.cursor:]
    state.order = torch.randperm(total, device=device)
    state.total_points = total
    head_count = max_points_per_iter - tail.numel()
    head = state.order[:head_count]
    state.cursor = head_count
    idx = torch.cat((tail, head), dim=0)
    return idx, total


def _densify_due(
    iteration: int,
    densify_from_iter: int,
    densification_interval: int,
) -> bool:
    """Return whether densification should run on this iteration."""
    interval = max(int(densification_interval), 1)
    if int(iteration) < int(densify_from_iter):
        return False
    return ((int(iteration) - int(densify_from_iter)) % interval) == 0


def _log_gpu_memory(
    tag: str, iteration: int, total_points: int, active_points: int
) -> None:
    """Print lightweight CUDA memory diagnostics for early iterations."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9
    print(
        (
            f"[MEM][iter={iteration}] {tag}: alloc={alloc:.2f} GB, "
            f"reserved={reserved:.2f} GB | points {active_points}/{total_points}"
        )
    )


def _log_used_vram(iteration: int) -> None:
    """Print current CUDA VRAM usage for periodic console monitoring."""
    if not torch.cuda.is_available():
        return

    used = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(
        f"[VRAM][iter={iteration}] used={used:.2f} GB, reserved={reserved:.2f} GB"
    )


def _maybe_reset_opacity(
    gaussians: GaussianModel, iteration: int, interval: int
) -> bool:
    """Reset learnable opacities when enabled and no mask buffer is active."""
    if interval <= 0 or iteration % interval != 0:
        return False

    if getattr(gaussians, "opacity_mode", "learned") != "learned":
        return False

    mask_check = getattr(gaussians, "_mask_opacity_active", None)
    if callable(mask_check) and mask_check():
        return False

    has_mask_buffer = (
        hasattr(gaussians, "opacities")
        and isinstance(gaussians.opacities, torch.Tensor)
        and gaussians.opacities.numel() > 0
    )
    if has_mask_buffer:
        return False

    gaussians.reset_opacity()
    return True


def _ensure_core_params_require_grad(gaussians: GaussianModel) -> None:
    """Make sure the core tensors stay connected to the optimizer graph."""
    for name in ("_xyz", "_scaling", "_rotation"):
        tensor = getattr(gaussians, name, None)
        if tensor is None or not isinstance(tensor, torch.nn.Parameter):
            continue
        if not tensor.requires_grad:
            print(f"WARNING: {name} had requires_grad=False – enabling in-place.")
            tensor.requires_grad_(True)


def _collect_grad_norms(gaussians: GaussianModel) -> dict[str, float]:
    """Return gradient norms for the main parameter tensors."""
    norms: dict[str, float] = {}
    if gaussians._xyz.grad is not None:
        norms["xyz"] = gaussians._xyz.grad.norm().item()
    if gaussians._scaling.grad is not None:
        norms["scaling"] = gaussians._scaling.grad.norm().item()
    if gaussians._rotation.grad is not None:
        norms["rotation"] = gaussians._rotation.grad.norm().item()
    return norms


def _clip_gradients(gaussians: GaussianModel, max_norm: float) -> None:
    """Apply gradient clipping to the core tensors if gradients exist."""
    clip_candidates: list[torch.Tensor] = []

    if gaussians._xyz.grad is not None:
        clip_candidates.append(gaussians._xyz)

    if gaussians._scaling.grad is not None:
        clip_candidates.append(gaussians._scaling)

    if gaussians._rotation.grad is not None:
        clip_candidates.append(gaussians._rotation)

    if (
        hasattr(gaussians, "_features_dc")
        and gaussians._features_dc is not None
        and gaussians._features_dc.requires_grad
        and gaussians._features_dc.grad is not None
    ):
        clip_candidates.append(gaussians._features_dc)

    if clip_candidates:
        torch.nn.utils.clip_grad_norm_(clip_candidates, max_norm=max_norm)


def _log_gradient_snapshot(
    gaussians: GaussianModel, iteration: int, interval: int
) -> None:
    """Log gradient norms to stdout at a fixed interval."""
    if iteration % interval != 0:
        return
    with torch.no_grad():
        if gaussians._xyz.grad is not None:
            print(f"XYZ grad norm: {gaussians._xyz.grad.norm().item():.6f}")
        if gaussians._scaling.grad is not None:
            print(f"Scaling grad norm: {gaussians._scaling.grad.norm().item():.6f}")
        if gaussians._rotation.grad is not None:
            print(f"Rotation grad norm: {gaussians._rotation.grad.norm().item():.6f}")


def _log_learning_rates(gaussians: GaussianModel, iteration: int) -> None:
    """Dump learning rate information for the first iterations."""
    if iteration > 3:
        return
    print(f"\n[ITER {iteration}] Learning Rates:")
    for group in gaussians.optimizer.param_groups:
        name = group.get("name", "?")
        lr = group.get("lr", 0.0)
        print(f"  {name:15s}: {lr:.8f}")
    print(f"  spatial_lr_scale: {gaussians.spatial_lr_scale:.8f}")


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    args,
):

    max_points_per_iter = int(
        max(1, getattr(args, "max_points_per_iter", DEFAULT_MAX_POINTS_PER_ITER))
    )

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":

        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(args)

    # SH degree for volume-only training.
    # Historically this code used degree 0; allow overriding via --sh_degree.
    sh_degree = int(getattr(args, "sh_degree", 0))

    gaussians = GaussianModel(
        sh_degree, opt.optimizer_type
    )
    gaussians.set_intensity_mode(getattr(opt, "intensity_mode", "sampled"))
    gaussians.set_opacity_mode(getattr(opt, "opacity_mode", "sampled"))
    gaussians.configure_mean_covered_sampling(
        large_splat_threshold=getattr(opt, "intensity_large_splat_threshold", 0.03),
        radius_scale=getattr(opt, "intensity_mean_cover_radius", 2.5),
        update_interval=getattr(
            opt,
            "intensity_mean_cover_interval",
            getattr(opt, "intensity_update_interval", 10),
        ),
    )
    gaussians.set_intensity_color_divisor(getattr(opt, "intensity_color_divisor", 1.0))

    # Absolute voxel-unit scale clamps (applied per-axis).
    gaussians.min_scale_vox = float(getattr(opt, "min_scale_vox", 1.0))
    gaussians.max_scale_vox = float(getattr(opt, "max_scale_vox", 10.0))

    # Propagate constraint-related knobs from CLI into the Gaussian model.
    gaussians.max_scale_factor = getattr(opt, "max_scale_factor", gaussians.max_scale_factor)
    gaussians._max_scale_factor_base = gaussians.max_scale_factor
    gaussians.scaling_constraint_warmup_iters = getattr(
        opt, "scaling_constraint_warmup_iters", gaussians.scaling_constraint_warmup_iters
    )
    gaussians.scaling_constraint_relaxation = getattr(
        opt, "scaling_constraint_relaxation", gaussians.scaling_constraint_relaxation
    )
    gaussians.early_stats_window = getattr(
        opt, "early_stats_window", gaussians.early_stats_window
    )
    gaussians.max_position_displacement_scale = getattr(
        opt,
        "position_displacement_scale",
        gaussians.max_position_displacement_scale,
    )
    if gaussians.max_position_displacement_scale > 0.0:
        print(
            "WARNING: position displacement constraint is active "
            f"(position_displacement_scale={gaussians.max_position_displacement_scale:.4f}). "
            "Set <= 0 for fully free motion within volume bounds."
        )

    preset_state = _configure_medical_presets(args, opt)
    diagnostics_enabled = preset_state.diagnostics_enabled
    scale_constraints_enabled = (
        preset_state.scale_constraints_enabled or not preset_state.active
    )

    # Initialize parameter monitoring with increased log interval for better performance
    parameter_monitor = (
        ParameterMonitor(args.model_path, log_interval=50)
        if diagnostics_enabled
        else None
    )

    # Initialize parameter update tracker
    from utils.parameter_update_tracking import ParameterUpdateTracker

    update_tracker = ParameterUpdateTracker() if diagnostics_enabled else None

    # Create scene for volume-based training
    scene = VolumeScene(args, gaussians)

    volume_shape = tuple(
        args.volume_shape if hasattr(args, "volume_shape") else opt.volume_shape
    )
    volume_downscale_factor = getattr(args, "volume_downscale_factor", None)
    loss_type = (
        args.volume_loss_type
        if hasattr(args, "volume_loss_type")
        else opt.volume_loss_type
    )
    loss_weight = (
        args.volume_loss_weight
        if hasattr(args, "volume_loss_weight")
        else opt.volume_loss_weight
    )
    ct_loss_type = getattr(args, "ct_loss_type", "mse")
    mask_loss_weight = float(getattr(args, "mask_loss_weight", 1.0))
    ct_loss_weight = float(getattr(args, "ct_loss_weight", 1.0))
    volume_supervisor = VolumeSupervisor(
        volume_path=args.volume_path,
        volume_shape=volume_shape,
        volume_downscale_factor=volume_downscale_factor,
        volume_render_downscale_factor=getattr(args, "volume_render_downscale_factor", 1),
        disable_volume_overflow_guard=bool(
            getattr(args, "disable_volume_overflow_guard", True)
        ),
        mask_path=args.mask_path if hasattr(args, "mask_path") else None,
        loss_type=loss_type,
        ct_loss_type=ct_loss_type,
        loss_weight=loss_weight,
        supervision_target=getattr(args, "supervision_target", "joint"),
        mask_loss_weight=mask_loss_weight,
        ct_loss_weight=ct_loss_weight,
        density_scale=getattr(args, "density_scale", 1.0),
        mask_loss_threshold_rel=getattr(args, "mask_loss_threshold_rel", 0.01),
        opacity_gamma=getattr(args, "opacity_gamma", 1.0),
        outside_mask_weight=getattr(args, "outside_mask_weight", 0.1),
        intensity_update_interval=getattr(opt, "intensity_update_interval", 10),
        opacity_update_interval=getattr(
            opt,
            "opacity_update_interval",
            getattr(opt, "intensity_update_interval", 10),
        ),
        sampling_padding_mode=getattr(opt, "sampling_padding_mode", "border"),
        sparse_support_cutoff=getattr(opt, "sparse_support_cutoff", 0.2),
        sparse_max_radius_vox=getattr(opt, "sparse_max_radius_vox", 10),
        sparse_support_softness=getattr(opt, "sparse_support_softness", 0.75),
        render_min_sigma_vox=getattr(opt, "render_min_sigma_vox", 0.35),
        sparse_support_cutoff_final=getattr(
            opt, "sparse_support_cutoff_final", None
        ),
        sparse_support_softness_final=getattr(
            opt, "sparse_support_softness_final", None
        ),
        render_min_sigma_vox_final=getattr(
            opt, "render_min_sigma_vox_final", None
        ),
        raster_schedule_start_iter=getattr(opt, "raster_schedule_start_iter", -1),
        raster_schedule_end_iter=getattr(opt, "raster_schedule_end_iter", -1),
        volume_storage_dtype=getattr(args, "volume_storage_dtype", "fp32"),
    )
    volume_supervisor.enable_render_checkpoint = not bool(
        getattr(args, "disable_render_checkpoint", False)
    )
    volume_supervisor.enable_diagnostics = diagnostics_enabled

    ao_volume = None
    ao_strength = float(getattr(args, "export_ao_strength", 1.0))
    if bool(getattr(args, "export_ao", False)):
        ao_radius = int(getattr(args, "export_ao_radius_vox", 2))
        ao_method = str(getattr(args, "export_ao_method", "isotropic"))
        ao_result = compute_ao_volume_from_mask(
            volume_supervisor.mask_volume,
            volume_supervisor.mask_bool,
            radius_vox=ao_radius,
            method=ao_method,  # type: ignore[arg-type]
        )
        ao_volume = ao_result.ao_volume
        volume_supervisor.ao_volume = ao_volume

    tb_log_interval = max(1, int(getattr(args, "tb_log_interval", 10)))
    postfix_interval = max(1, int(getattr(args, "progress_postfix_interval", 10)))
    eval_masked_mse_full_roi_interval = max(
        0, int(getattr(args, "eval_masked_mse_full_roi_interval", 0))
    )
    eval_masked_mse_full_roi_target = str(
        getattr(args, "eval_masked_mse_full_roi_target", "auto")
    )
    eval_masked_mse_full_roi_downscale_factor = max(
        1, int(getattr(args, "eval_masked_mse_full_roi_downscale_factor", 1))
    )

    # Provide voxel spacing to the Gaussian model so voxel-unit clamps work.
    gaussians.voxel_size = volume_supervisor.voxel_size
    gaussians.voxel_spacing_xyz = getattr(volume_supervisor, "voxel_spacing_xyz", None)
    gaussians.raw_volume_min = getattr(volume_supervisor, "raw_intensity_min", None)
    gaussians.raw_volume_max = getattr(volume_supervisor, "raw_intensity_max", None)
    gaussians.sampling_padding_mode = getattr(
        opt, "sampling_padding_mode", volume_supervisor.sampling_padding_mode
    )
    gaussians.position_bounds = (
        volume_supervisor.bounds_min,
        volume_supervisor.bounds_max,
    )
    gaussians.structure_guidance_helper = volume_supervisor
    gaussians.reference_mask_threshold = float(
        getattr(args, "init_mask_threshold", 0.05)
    )
    if hasattr(args, "orientation_sigma_grad"):
        volume_supervisor.orientation_sigma_grad = float(
            args.orientation_sigma_grad
        )
    if hasattr(args, "orientation_sigma_tensor"):
        volume_supervisor.orientation_sigma_tensor = float(
            args.orientation_sigma_tensor
        )
    if hasattr(args, "orientation_perturb_deg"):
        volume_supervisor.orientation_perturb_deg = float(
            args.orientation_perturb_deg
        )
    if hasattr(args, "structure_sigma"):
        volume_supervisor.structure_sigma = float(args.structure_sigma)
    if hasattr(args, "structure_mask_threshold"):
        volume_supervisor.structure_mask_threshold = float(
            args.structure_mask_threshold
        )

    from gaussian_splatting.utils.volume_initializer import initialize_gaussians

    # Load volume transform if provided
    volume_transform = None
    if args.volume_transform:
        # Volume supervision (splat_to_volume) operates in normalized [0,1]^3.
        # Applying an arbitrary transform here can move points out of that domain.
        # If you need a transform, the supervision mapping must be updated consistently.
        print(
            "WARNING: --volume_transform provided, but normalized volume-space training "
            "keeps Gaussians in [0,1]^3 and will ignore volume_transform."
        )
        volume_transform = None

    # Get scene bounds for scaling
    # Volume supervision assumes normalized volume coordinates; do not remap
    # points into camera/world bounds.
    scene_bounds = None

    # Initialize gaussians by sampling from mask/volume inputs
    initialize_gaussians(
        model=gaussians,
        mask_path=args.mask_path,
        volume_path=args.volume_path,
        n_points=args.init_n_points,
        volume_transform=volume_transform,
        scene_bounds=scene_bounds,
        volume_downscale_factor=volume_downscale_factor,
        disable_volume_overflow_guard=bool(
            getattr(args, "disable_volume_overflow_guard", True)
        ),
        volume_storage_dtype=getattr(args, "volume_storage_dtype", "fp32"),
        init_scale_min_vox=getattr(args, "init_scale_min_vox", 1.0),
        init_scale_max_vox=getattr(args, "init_scale_max_vox", 3.0),
        opacity_gamma=getattr(args, "opacity_gamma", 1.0),
        opacity_mode=getattr(opt, "opacity_mode", "sampled"),
        noise_std=(
            args.position_noise
            if hasattr(args, "position_noise")
            else opt.position_noise
        ),
        orientation_helper=volume_supervisor,
        mask_threshold=float(getattr(args, "init_mask_threshold", 0.05)),
        structure_mask_threshold=getattr(args, "structure_mask_threshold", 0.1),
        structure_sigma=getattr(args, "structure_sigma", 1.0),
        structure_min_vesselness=getattr(args, "structure_min_vesselness", 0.1),
        anisotropy_strength=getattr(args, "anisotropy_strength", 0.0),
        structure_orientation_strength=getattr(
            args, "structure_orientation_strength", 0.0
        ),
        init_anisotropy_ratio=getattr(args, "init_anisotropy_ratio", 1.0),
        border_distance_vox=getattr(args, "border_distance_vox", 0.0),
        border_flatten_ratio=getattr(args, "border_flatten_ratio", 1.0),
        border_grad_sigma=getattr(args, "border_grad_sigma", 1.5),
    )

    # Set spatial_lr_scale after volume initialization
    # For normalized [0,1]^3 coordinates, the natural spatial scale is 1.0.
    gaussians.spatial_lr_scale = 1.0

    print(
        f"Initialized {gaussians._xyz.shape[1]} Gaussians; spatial_lr_scale={gaussians.spatial_lr_scale:.3f}"
    )

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

        # Initialize intensity values if they don't exist

    # Initialize mixed precision training
    scaler = GradScaler()
    use_amp = (
        not args.disable_mixed_precision
    )  # Use mixed precision unless explicitly disabled
    if use_amp:
        print("Using mixed precision training for better performance")
        bf16_supported = False
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_available():
            bf16_supported = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if bf16_supported else torch.float16
    else:
        print("Mixed precision training disabled")
        amp_dtype = torch.float32

    # Ensure checkpoint recompute in VolumeSupervisor uses the same autocast settings.
    volume_supervisor.render_use_amp = bool(use_amp)
    volume_supervisor.render_amp_dtype = amp_dtype

    # Initialize tracking variables
    ema_loss_for_log = 0.0
    ema_vol_loss_for_log = 0.0
    param_stats = {"scale_change_rate": 0.0, "rot_change_rate": 0.0}

    diversity_warmup_iters = getattr(opt, "diversity_warmup_iterations", 0)
    diversity_log_interval = max(1, getattr(opt, "diversity_log_interval", 25))
    diversity_scale_weight = getattr(opt, "diversity_scale_weight", 0.0)
    diversity_rotation_weight = getattr(opt, "diversity_rotation_weight", 0.0)
    diversity_scale_range_weight = getattr(
        opt, "diversity_scale_range_weight", 0.2
    )
    diversity_target_range_weight = getattr(
        opt, "diversity_target_range_weight", 0.2
    )
    diversity_rotation_entropy_weight = getattr(
        opt, "diversity_rotation_entropy_weight", 0.2
    )
    diversity_dispersion_weight = getattr(opt, "diversity_dispersion_weight", 0.2)
    diversity_alignment_weight = getattr(opt, "diversity_alignment_weight", 0.1)
    diversity_enabled = (
        preset_state.diversity_enabled
        and diversity_warmup_iters > 0
        and (diversity_scale_weight > 0 or diversity_rotation_weight > 0)
    )

    class _NoopCudaEvent:
        """Minimal stand-in when CUDA timing events are unavailable."""

        def record(self) -> None:
            return None

        def elapsed_time(self, other) -> float:
            return 0.0

    if torch.cuda.is_available():
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
    else:
        iter_start = _NoopCudaEvent()
        iter_end = _NoopCudaEvent()

    progress_bar = tqdm(
        range(first_iter, opt.iterations),
        desc="#### Training progress ####",
        dynamic_ncols=True,
        leave=True,
    )
    first_iter += 1
    active_subset_state = ActiveSubsetState()
    for iteration in range(first_iter, opt.iterations + 1):
        # Skip GUI network in volume-only mode

        iter_start.record()

        log_mem = iteration <= 3
        if log_mem and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        xyz_grad_norm = None
        scaling_grad_norm = None
        rotation_grad_norm = None
        scaling_lr = next(
            (
                group["lr"]
                for group in gaussians.optimizer.param_groups
                if group["name"] == "scaling"
            ),
            None,
        )

        gaussians.update_learning_rate(iteration)
        gaussians.optimizer.zero_grad(set_to_none=True)

        xyz_for_sampling = gaussians.get_xyz
        active_idx, total_points = _select_active_indices(
            xyz_for_sampling,
            max_points_per_iter=max_points_per_iter,
            state=active_subset_state,
        )
        active_points = active_idx.numel() if active_idx is not None else total_points
        if log_mem and total_points > 0:
            _log_gpu_memory("before_forward", iteration, total_points, active_points)
            if active_idx is not None:
                print(
                    f"[Points][iter={iteration}] using {active_points} / {total_points} splats"
                )

        # No SH updates needed for volume-only training

        # Initialize total loss
        loss = 0.0
        # Volume supervision loss
        volume_loss = 0.0
        reg_loss_value = None
        reg_metrics = None
        structure_guidance_metrics = None
        warmup_active = diversity_enabled and iteration <= diversity_warmup_iters

        # Compute the volume loss and get volume gradients for parameter diversity loss
        with autocast(enabled=use_amp, dtype=amp_dtype if use_amp else None):
            vol_loss, vol_metrics, vol_gradients = volume_supervisor.compute_loss(
                gaussians,
                active_idx=active_idx,
                total_points=total_points,
                compute_volume_gradients=bool(warmup_active),
            )

            # CRITICAL: Don't call item() on the loss until after backward() is called!
            loss = vol_loss

            # Optional global scale L2 regularization to discourage oversized splats
            scale_l2_weight = getattr(args, "scale_l2_weight", 0.0)
            if scale_constraints_enabled and scale_l2_weight > 0.0:
                scales = gaussians.get_scaling
                if scales.numel() > 0:
                    scale_norm = scales.norm(dim=1)
                    scale_reg = scale_norm.mean() * float(scale_l2_weight)
                    loss = loss + scale_reg
                    vol_metrics["scale_l2_reg"] = float(scale_reg.detach().item())

            # Optional log-scale spread penalty (global). Encourages more uniform splat sizes
            # without forcing them identical.
            scale_logvar_weight = float(getattr(args, "scale_logvar_weight", 0.0))
            scale_logvar_warmup = int(getattr(args, "scale_logvar_warmup_iters", 0))
            if scale_logvar_weight > 0.0 and iteration >= scale_logvar_warmup:
                scales = gaussians.get_scaling
                voxel_size = getattr(gaussians, "voxel_size", None)
                if (
                    scales.numel() > 0
                    and isinstance(voxel_size, torch.Tensor)
                    and voxel_size.numel() == 3
                ):
                    voxel_size_xyz = voxel_size.to(
                        device=scales.device, dtype=scales.dtype
                    ).clamp_min(1e-8)
                    scales_vox = scales / voxel_size_xyz.unsqueeze(0)
                    log_scales_vox = torch.log(scales_vox.clamp_min(1e-8))
                    centered = log_scales_vox - log_scales_vox.mean(
                        dim=0, keepdim=True
                    )
                    spread = (centered * centered).mean()
                    scale_spread_reg = spread * scale_logvar_weight
                    loss = loss + scale_spread_reg
                    vol_metrics["scale_logvar_reg"] = float(
                        scale_spread_reg.detach().item()
                    )

            anisotropy_reg_weight = float(
                getattr(args, "anisotropy_reg_weight", 0.0)
            )
            anisotropy_warmup = int(
                getattr(args, "anisotropy_reg_warmup_iters", 0)
            )
            if anisotropy_reg_weight > 0.0 and iteration >= anisotropy_warmup:
                scales = gaussians.get_scaling
                if scales.numel() > 0:
                    voxel_size = getattr(gaussians, "voxel_size", None)
                    if isinstance(voxel_size, torch.Tensor) and voxel_size.numel() == 3:
                        voxel_size_xyz = voxel_size.to(
                            device=scales.device, dtype=scales.dtype
                        ).clamp_min(1e-8)
                        scales_axes = scales / voxel_size_xyz.unsqueeze(0)
                    else:
                        scales_axes = scales

                    max_axis = scales_axes.max(dim=1).values.clamp_min(1e-8)
                    min_axis = scales_axes.min(dim=1).values.clamp_min(1e-8)
                    axis_ratio = max_axis / min_axis
                    target_ratio = max(
                        1.0, float(getattr(args, "anisotropy_target_ratio", 2.0))
                    )
                    ratio_excess = torch.relu(axis_ratio - target_ratio) / target_ratio
                    anisotropy_reg = (
                        (ratio_excess * ratio_excess).mean() * anisotropy_reg_weight
                    )
                    loss = loss + anisotropy_reg
                    vol_metrics["anisotropy_reg"] = float(
                        anisotropy_reg.detach().item()
                    )
                    vol_metrics["anisotropy_ratio_mean"] = float(
                        axis_ratio.mean().detach().item()
                    )

        if log_mem and total_points > 0:
            _log_gpu_memory("after_forward", iteration, total_points, active_points)

        # Store value for logging only (avoid per-iteration GPU->CPU sync; updated later).
        volume_loss_tensor = vol_loss.detach()

        # Apply diversity warmup regularization when requested
        if warmup_active:
            base_loss = loss
            reg_scale_weight = diversity_scale_weight * 0.5
            reg_rotation_weight = diversity_rotation_weight * 0.5
            reg_scale_range_weight = diversity_scale_range_weight * 0.25
            reg_target_range_weight = diversity_target_range_weight * 0.25
            reg_entropy_weight = diversity_rotation_entropy_weight * 0.25
            reg_dispersion_weight = diversity_dispersion_weight * 0.25
            reg_alignment_weight = diversity_alignment_weight * 0.5
            loss, reg_metrics = add_parameter_regularization_loss(
                model=gaussians,
                loss=loss,
                scale_diversity_weight=reg_scale_weight,
                rotation_diversity_weight=reg_rotation_weight,
                scale_range_weight=reg_scale_range_weight,
                rotation_entropy_weight=reg_entropy_weight,
                target_range_weight=reg_target_range_weight,
                dispersion_weight=reg_dispersion_weight,
                alignment_weight=reg_alignment_weight,
                volume_gradients=vol_gradients,
            )
            reg_loss_value = reg_metrics.get("total") if reg_metrics else None
            if reg_loss_value is None:
                reg_loss_value = (loss - base_loss).detach().item()

            if iteration <= 3 or iteration % diversity_log_interval == 0:
                remaining = diversity_warmup_iters - iteration
                scale_total = (
                    reg_metrics.get("scale_total", 0.0) if reg_metrics else 0.0
                )
                rotation_total = (
                    reg_metrics.get("rotation_total", 0.0) if reg_metrics else 0.0
                )
                print(
                    (
                        f"[REG][iter={iteration}] total={reg_loss_value:.6f} "
                        f"scale={scale_total:.6f} rotation={rotation_total:.6f} "
                        f"remaining={max(0, remaining)}"
                    )
                )

        # Track parameter statistics for monitoring (only on every 50th iteration)
        if diagnostics_enabled and parameter_monitor is not None:
            if iteration % 50 == 0:
                new_stats = parameter_monitor.update(
                    iteration,
                    gaussians._xyz,
                    gaussians.get_scaling,
                    gaussians.get_rotation,
                    loss=loss.item(),
                    volume_loss=vol_loss.item() if vol_loss is not None else None,
                    reg_loss=reg_loss_value,
                )
                if new_stats:
                    param_stats.update(new_stats)

        # Log volume metrics
        if tb_writer and iteration % 10 == 0:
            for name, value in vol_metrics.items():
                tb_writer.add_scalar(f"volume/{name}", value, iteration)

            if diagnostics_enabled:
                tb_writer.add_scalar(
                    "diversity/scale_weight", diversity_scale_weight, iteration
                )
                tb_writer.add_scalar(
                    "diversity/rotation_weight", diversity_rotation_weight, iteration
                )
                if reg_loss_value is not None:
                    tb_writer.add_scalar(
                        "loss/regularization", reg_loss_value, iteration
                    )
                    if reg_metrics:
                        for metric_name, metric_value in reg_metrics.items():
                            tb_writer.add_scalar(
                                f"diversity/{metric_name}", metric_value, iteration
                            )

        # Make sure the loss requires gradients before calling backward
        if loss.requires_grad:
            # Make sure parameters require gradients BEFORE calling backward
            _ensure_core_params_require_grad(gaussians)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(gaussians.optimizer)
            else:
                loss.backward()

            grad_norms = _collect_grad_norms(gaussians)
            xyz_grad_norm = grad_norms.get("xyz")
            scaling_grad_norm = grad_norms.get("scaling")
            rotation_grad_norm = grad_norms.get("rotation")

            if log_mem and total_points > 0:
                _log_gpu_memory(
                    "after_backward", iteration, total_points, active_points
                )

            # Debug: Save pre-step values to verify updates happen
            if iteration <= 5 or iteration % 500 == 0:
                pre_xyz_mean = gaussians._xyz.mean().item()
                pre_scaling_mean = gaussians._scaling.mean().item()

            if diagnostics_enabled:
                _log_gradient_snapshot(gaussians, iteration, interval=50)

            # Clip gradients to prevent numerical instability
            if iteration > 1:
                _clip_gradients(gaussians, max_norm=10.0)

            if use_amp:
                prev_scale = scaler.get_scale()
                scaler.step(gaussians.optimizer)
                scaler.update()
            else:
                gaussians.optimizer.step()

            if log_mem and total_points > 0:
                _log_gpu_memory("after_step", iteration, total_points, active_points)

            if iteration % 100 == 0:
                _log_used_vram(iteration)

            # Debug: Check if scaler skipped the step (happens when gradients are inf/nan)
            if diagnostics_enabled and (iteration <= 10 or iteration % 500 == 0):
                if use_amp:
                    scale = scaler.get_scale()
                    print(
                        f"[ITER {iteration}] GradScaler scale: {scale:.2f} (prev {prev_scale:.2f})"
                    )

                if iteration <= 5 or iteration % 500 == 0:
                    post_xyz_mean = gaussians._xyz.mean().item()
                    post_scaling_mean = gaussians._scaling.mean().item()
                    xyz_change = abs(post_xyz_mean - pre_xyz_mean)
                    scaling_change = abs(post_scaling_mean - pre_scaling_mean)
                    print(f"  XYZ mean change: {xyz_change:.10f}")
                    print(f"  Scaling mean change: {scaling_change:.10f}")
                    if scaling_lr is not None and scaling_grad_norm is not None:
                        print(
                            "  Scaling lr: {:.6e} | grad norm: {:.6e}".format(
                                scaling_lr,
                                scaling_grad_norm,
                            )
                        )
                    if xyz_grad_norm is not None:
                        print(f"  XYZ grad norm: {xyz_grad_norm:.6e}")
                    if rotation_grad_norm is not None:
                        print(f"  Rotation grad norm: {rotation_grad_norm:.6e}")

            # Verify learning rates on first few iterations
            _log_learning_rates(gaussians, iteration)

            # Enforce scale clamps (absolute clamps always; relative clamp can be disabled by presets).
            gaussians.enforce_scaling_constraint(
                iteration=iteration,
                apply_relative=bool(scale_constraints_enabled),
            )
            gaussians.enforce_position_displacement_constraint()
            gaussians.enforce_position_bounds()
            structure_guidance_metrics = gaussians.apply_structure_guidance(
                iteration,
                indices=active_idx,
            )
            if structure_guidance_metrics:
                gaussians.enforce_scaling_constraint(
                    iteration=iteration,
                    apply_relative=bool(scale_constraints_enabled),
                )

            # Adaptive density control for volume-based training
            with torch.no_grad():
                # For volume-based training, use position gradients instead of viewspace gradients
                if (
                    preset_state.densification_enabled
                    and iteration >= opt.densify_from_iter
                    and iteration <= opt.densify_until_iter
                ):
                    # Accumulate gradients for densification (every iteration during densification period)
                    if gaussians._xyz.grad is not None:
                        # _xyz has shape [3, N], so grad also has shape [3, N]
                        # Compute norm across the 3D dimension (dim=0) to get magnitude per point
                        # Result shape: [N]
                        xyz_grad_per_point = torch.norm(
                            gaussians._xyz.grad, dim=0, keepdim=False
                        )
                        # Reshape to [N, 1] to match xyz_gradient_accum shape
                        xyz_grad_per_point = xyz_grad_per_point.unsqueeze(1)
                        if active_idx is None:
                            gaussians.xyz_gradient_accum += xyz_grad_per_point
                            gaussians.denom += 1
                        else:
                            gaussians.xyz_gradient_accum[active_idx] += (
                                xyz_grad_per_point[active_idx]
                            )
                            gaussians.denom[active_idx] += 1

                    # Perform densification and pruning at intervals
                    if (
                        iteration < opt.iterations
                        and _densify_due(
                            iteration,
                            opt.densify_from_iter,
                            opt.densification_interval,
                        )
                    ):
                        # Volume-only training uses normalized [0,1]^3 coordinates.
                        # Keep densification heuristics in the same normalized scale.
                        extent = 1.0

                        points_before = gaussians._xyz.shape[1]

                        # Perform densification and pruning
                        gaussians.densify_and_prune(
                            max_grad=opt.densify_grad_threshold,
                            min_opacity=float(getattr(opt, "prune_min_opacity", 1e-4)),
                            extent=extent,
                            max_screen_size=None,  # No screen size limit for volume training
                            radii=None,  # No radii for volume training
                        )

                        points_after = gaussians._xyz.shape[1]
                        last_counts = getattr(
                            gaussians,
                            "last_densify_counts",
                            {"split": 0, "clone": 0, "hole_fill": 0},
                        )
                        added = int(sum(int(v) for v in last_counts.values()))
                        pruned_est = max(0, points_before + added - points_after)

                        # Log densification. With small intervals the previous
                        # cadence could hide most densification passes.
                        should_log_densify = (
                            bool(getattr(args, "enable_diagnostics", False))
                            or iteration % 100 == 0
                            or iteration == opt.densify_from_iter
                        )
                        if should_log_densify:
                            print(
                                (
                                    f"\n[ITER {iteration}] Densify: +{added} pruned~{pruned_est} "
                                    f"(split={int(last_counts.get('split', 0))}, "
                                    f"clone={int(last_counts.get('clone', 0))}, "
                                    f"hole_fill={int(last_counts.get('hole_fill', 0))}) "
                                    f"-> {points_after} points"
                                )
                            )


            gaussians.optimizer.zero_grad(set_to_none=True)

            # Add a manual gradient perturbation if gradients are zero
            # This is a drastic measure to force parameter updates
            # REMOVE direct random parameter perturbations (they break true gradient-based optimization)
        else:
            # This should no longer happen with the fixed gradient chain
            print("WARNING: Loss does not require gradients! Check gradient chain.")
            dummy_loss = (gaussians._xyz.sum() * 0) + loss
            dummy_loss.backward()

        iter_end.record()

        with torch.no_grad():
            progress_bar.update(1)  # Update by 1 each iteration

            eval_masked_mse_full_roi = None
            eval_masked_mse_full_roi_target_used = None
            should_eval_masked_mse_full_roi = (
                eval_masked_mse_full_roi_interval > 0
                and (
                    iteration == opt.iterations
                    or (iteration % eval_masked_mse_full_roi_interval) == 0
                )
            )
            if should_eval_masked_mse_full_roi:
                eval_masked_mse_full_roi, eval_masked_mse_full_roi_target_used = (
                    volume_supervisor.compute_full_roi_masked_mse(
                        gaussians,
                        target=eval_masked_mse_full_roi_target,
                        working_grid_downscale_factor=(
                            eval_masked_mse_full_roi_downscale_factor
                        ),
                        refresh_appearance=True,
                    )
                )
                print(
                    (
                        f"\n[ITER {iteration}] eval/masked_mse_full_roi="
                        f"{eval_masked_mse_full_roi:.6f} "
                        f"(target={eval_masked_mse_full_roi_target_used}, "
                        f"downscale={eval_masked_mse_full_roi_downscale_factor})"
                    )
                )

            should_postfix = (
                iteration == 1
                or iteration == opt.iterations
                or (iteration % postfix_interval) == 0
            )
            if should_postfix:
                loss_scalar = float(loss.detach().item())
                volume_loss = float(volume_loss_tensor.item())

                ema_loss_for_log = 0.4 * loss_scalar + 0.6 * ema_loss_for_log
                ema_vol_loss_for_log = 0.4 * volume_loss + 0.6 * ema_vol_loss_for_log

                scaling = gaussians.get_scaling
                scaling_mean = float(scaling.mean().item())
                scaling_std = float(scaling.std().item())

                rotation = gaussians.get_rotation
                rotation_magnitude = float(
                    torch.norm(rotation[:, 1:], dim=1).mean().item()
                )

                scale_change = param_stats.get("scale_change_rate", 0.0)
                rot_change = param_stats.get("rot_change_rate", 0.0)

                postfix = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Vol": f"{ema_vol_loss_for_log:.{5}f}",
                    "Scale": f"{scaling_mean:.{3}f}±{scaling_std:.{3}f}",
                    "Rot": f"{rotation_magnitude:.{3}f}",
                    "Δs": f"{scale_change:.{3}f}",
                    "Δr": f"{rot_change:.{3}f}",
                }
                progress_bar.set_postfix(postfix)

            if iteration == opt.iterations:
                progress_bar.close()

            should_tb = tb_writer is not None and (
                iteration == 1
                or iteration == opt.iterations
                or (iteration % tb_log_interval) == 0
            )
            if should_tb and tb_writer is not None:
                tb_writer.add_scalar("loss/total", float(loss.detach().item()), iteration)
                tb_writer.add_scalar("loss/volume", float(volume_loss_tensor.item()), iteration)
                tb_writer.add_scalar(
                    "timing/iter_ms", iter_start.elapsed_time(iter_end), iteration
                )
                tb_writer.add_scalar("model/points", gaussians._xyz.shape[1], iteration)
                if eval_masked_mse_full_roi is not None:
                    tb_writer.add_scalar(
                        "eval/masked_mse_full_roi",
                        eval_masked_mse_full_roi,
                        iteration,
                    )

                if diagnostics_enabled and should_postfix:
                    tb_writer.add_scalar(
                        "parameters/scaling_mean", scaling_mean, iteration
                    )
                    tb_writer.add_scalar(
                        "parameters/scaling_std", scaling_std, iteration
                    )
                    tb_writer.add_scalar(
                        "parameters/rotation_magnitude", rotation_magnitude, iteration
                    )

                    if xyz_grad_norm is not None:
                        tb_writer.add_scalar("grads/xyz_norm", xyz_grad_norm, iteration)
                    if scaling_grad_norm is not None:
                        tb_writer.add_scalar(
                            "grads/scaling_norm", scaling_grad_norm, iteration
                        )
                    if rotation_grad_norm is not None:
                        tb_writer.add_scalar(
                            "grads/rotation_norm", rotation_grad_norm, iteration
                        )
                    if scaling_lr is not None:
                        tb_writer.add_scalar("lr/scaling", scaling_lr, iteration)

                    if structure_guidance_metrics:
                        for metric_name, metric_value in (
                            structure_guidance_metrics.items()
                        ):
                            tb_writer.add_scalar(
                                f"structure_guidance/{metric_name}",
                                metric_value,
                                iteration,
                            )

                    if update_tracker is not None:
                        update_metrics = update_tracker.update(gaussians)
                        for name, value in update_metrics.items():
                            tb_writer.add_scalar(f"updates/{name}", value, iteration)

                        if iteration % 100 == 0:
                            xyz_delta = update_metrics.get("xyz_delta_avg", 0)
                            scale_delta = update_metrics.get("scaling_delta_avg", 0)
                            rot_delta = update_metrics.get("rotation_delta_avg", 0)
                            print(
                                f"\n[ITER {iteration}] Parameter updates - XYZ: {xyz_delta:.5f}, Scale: {scale_delta:.5f}, Rot: {rot_delta:.5f}"
                            )

            # Save PLY file at specified iterations
            save_ply_every = (
                args.save_ply_every if hasattr(args, "save_ply_every") else 1
            )
            needs_serialization_refresh = (
                iteration % save_ply_every == 0
                or iteration == 1
                or iteration == opt.iterations
                or (iteration in saving_iterations)
                or (iteration in checkpoint_iterations)
            )
            if needs_serialization_refresh:
                volume_supervisor.refresh_cached_appearance(
                    gaussians,
                    force_all=True,
                )

            if (
                iteration % save_ply_every == 0
                or iteration == 1
                or iteration == opt.iterations
            ):
                ply_output_dir = os.path.join(args.model_path, "ply_sequence")
                prefix = (
                    args.ply_output_prefix
                    if hasattr(args, "ply_output_prefix")
                    else "gaussians"
                )
                ao_values = None
                if ao_volume is not None:
                    ao_values, _, _ = sample_intensities_from_volume(
                        gaussians.get_xyz,
                        ao_volume,
                        normalize=False,
                    )
                    ao_values = ao_values.clamp(0.0, 1.0)

                ply_output_path = gaussians.save_ply_sequence(
                    ply_output_dir,
                    iteration,
                    prefix,
                    ao=ao_values,
                    ao_strength=ao_strength,
                )

                # Log PLY saving every 100 iterations to avoid console spam
                if (
                    iteration % 100 == 0
                    or iteration == 1
                    or iteration == opt.iterations
                ):
                    print(f"\n[ITER {iteration}] Saved model as PLY: {ply_output_path}")

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Reset opacity only when learnable opacities are active
            reset_interval = getattr(opt, "opacity_reset_interval", 0)
            _maybe_reset_opacity(gaussians, iteration, reset_interval)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Generate parameter report on last iteration
            if (
                diagnostics_enabled
                and parameter_monitor is not None
                and iteration == opt.iterations
            ):
                parameter_monitor.update(
                    iteration,
                    gaussians._xyz,
                    gaussians.get_scaling,
                    gaussians.get_rotation,
                    force=True,
                    loss=loss.item(),
                    volume_loss=vol_loss.item() if vol_loss is not None else None,
                    reg_loss=None,
                )
                parameter_monitor.final_report()
                print(
                    "\nParameter monitoring report saved to:",
                    os.path.join(args.model_path, "parameter_stats"),
                )


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, loss, elapsed):
    """Simple training report for volume-based training"""
    if tb_writer:
        tb_writer.add_scalar("train/loss", loss.item(), iteration)
        tb_writer.add_scalar("train/iter_time_ms", elapsed, iteration)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    # Core parameter groups
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    xp = ExportParams(parser)
    tsp = TrainingScriptParams(parser)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    class VolumeDataset:
        def __init__(self, model_path: str):
            self.cameras_extent = 1.0
            self.white_background = False
            self.model_path = model_path
            self.source_path = ""
            self.sh_degree = int(getattr(args, "sh_degree", 0))

    dataset = VolumeDataset(args.model_path)

    # Start training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        dataset,
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args,  # Pass the full arguments
    )

    # All done
    print("\nTraining complete.")
