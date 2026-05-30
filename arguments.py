"""Unified, well-documented CLI surface for the volume-only trainer."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from types import SimpleNamespace
import os
import sys


class GroupParams(SimpleNamespace):
    """Simple namespace used for legacy compatibility in training/render scripts."""


class ParamGroup:
    """Base helper that wires attributes into an ``ArgumentParser`` group.

    This mirrors the behavior of the original ``arguments/__init__.py`` so older
    code that expects shorthand flags (e.g. ``-m`` for ``--model_path``) keeps
    working, while newer callsites can opt into the explicit registration-based
    API via ``_register``.
    """

    def __init__(self, parser: ArgumentParser | None = None, name: str | None = None, fill_none: bool = False) -> None:  # type: ignore[override]
        self._fields: list[str] = []

        if parser is not None and name is not None:
            group = parser.add_argument_group(name)
            for key, value in vars(self).items():
                shorthand = False
                if key.startswith("_"):
                    shorthand = True
                    key = key[1:]
                arg_type = type(value)
                default_val = value if not fill_none else None
                if shorthand:
                    if arg_type is bool:
                        group.add_argument(
                            "--" + key,
                            "-" + key[0:1],
                            default=default_val,
                            action="store_true",
                        )
                    else:
                        group.add_argument(
                            "--" + key,
                            "-" + key[0:1],
                            default=default_val,
                            type=arg_type,
                        )
                else:
                    if arg_type is bool:
                        group.add_argument(
                            "--" + key, default=default_val, action="store_true"
                        )
                    else:
                        group.add_argument(
                            "--" + key, default=default_val, type=arg_type
                        )

    def _register(self, name: str) -> None:
        """Track a field for the newer explicit registration-style API."""
        if name not in self._fields:
            self._fields.append(name)

    def extract(self, args: Namespace) -> GroupParams:
        """Populate a ``GroupParams`` view from an ``argparse.Namespace``."""
        params = GroupParams()

        # Legacy behavior: copy anything that matches our attributes (with or
        # without a leading underscore) so older training/render scripts that
        # rely on ``arguments.GroupParams`` keep working.
        for key, value in vars(args).items():
            if key in vars(self) or ("_" + key) in vars(self):
                setattr(params, key, value)

        # New behavior: also ensure any explicitly registered fields are copied
        # even if they do not correspond to pre-populated attributes.
        for field in self._fields:
            if hasattr(args, field):
                setattr(params, field, getattr(args, field))

        return params


class ModelParams(ParamGroup):
    """Filesystem inputs and data-shaping configuration exposed to end users."""

    def __init__(self, parser: ArgumentParser, sentinel: bool = False) -> None:
        super().__init__()

        core = parser.add_argument_group("Core Volume Inputs")

        # Where checkpoints and exported PLYs live.
        core.add_argument(
            "--model_path",
            type=str,
            required=not sentinel,
            help="Output directory that stores checkpoints, logs, and PLY exports.",
        )
        self._register("model_path")

        # Supervision volume (nii/npy/mhd) resampled during training.
        core.add_argument(
            "--volume_path",
            type=str,
            required=True,
            help="Path to the CT/MR volume that acts as the optimization target.",
        )
        self._register("volume_path")

        # Segmentation mask for initializing Gaussian seeds.
        core.add_argument(
            "--mask_path",
            type=str,
            required=True,
            help="Binary/probability mask used to sample initial Gaussians.",
        )
        self._register("mask_path")

        # Output resolution for voxelized supervision.
        core.add_argument(
            "--volume_shape",
            type=int,
            nargs=3,
            default=[64, 64, 64],
            metavar=("D", "H", "W"),
            help="Voxel grid resolution used when loading and supervising the volume.",
        )
        self._register("volume_shape")

        core.add_argument(
            "--volume_downscale_factor",
            type=int,
            default=1,
            help=(
                "Optional integer downscale factor applied when loading volume/mask. "
                "Defaults to 1 (native resolution). Values <= 1 keep the native resolution; "
                "values 2/4/... downsample each axis by that factor."
            ),
        )
        self._register("volume_downscale_factor")

        core.add_argument(
            "--volume_render_downscale_factor",
            type=int,
            default=1,
            help=(
                "Internal working-grid downscale used when rasterizing many Gaussians into the supervision volume. "
                "Defaults to 1 (full-resolution supervision). Values greater than 1 compute on a lower-resolution "
                "working grid and upsample. Note: full-resolution rasterization can be significantly slower and use more VRAM."
            ),
        )
        self._register("volume_render_downscale_factor")

        core.add_argument(
            "--volume_storage_dtype",
            type=str,
            default="fp32",
            choices=["fp32", "fp16", "bf16"],
            help=(
                "Storage dtype for loaded supervision volumes on GPU. "
                "Use fp16/bf16 to reduce persistent VRAM usage; fp32 keeps maximum precision."
            ),
        )
        self._register("volume_storage_dtype")

        core.add_argument(
            "--disable_volume_overflow_guard",
            default=True,
            action="store_true",
            help=(
                "Disable the loader's automatic safety resampling for very large volumes. "
                "This is the default behavior; use --enable_volume_overflow_guard to restore the safety resampling."
            ),
        )
        core.add_argument(
            "--enable_volume_overflow_guard",
            dest="disable_volume_overflow_guard",
            action="store_false",
            help="Re-enable the loader's automatic safety resampling for very large volumes.",
        )
        self._register("disable_volume_overflow_guard")

        # Optional 4x4 transform to align volume/mask with world coordinates.
        core.add_argument(
            "--volume_transform",
            type=str,
            default="",
            help="Numpy 4x4 transform bringing volume voxels into the training frame.",
        )
        self._register("volume_transform")

        # Loss family used when comparing rendered splats to the target volume.
        core.add_argument(
            "--volume_loss_type",
            type=str,
            default="dice",
            choices=["mse", "dice", "tversky", "kl"],
            help="Volume-domain loss to optimize (dice/mse/tversky/kl).",
        )
        self._register("volume_loss_type")

        # Scalar weight applied to the chosen volume loss.
        core.add_argument(
            "--volume_loss_weight",
            type=float,
            default=1.0,
            help="Weight applied to the chosen volume supervision loss.",
        )
        self._register("volume_loss_weight")

        core.add_argument(
            "--supervision_target",
            type=str,
            default="joint",
            choices=["mask", "ct", "joint"],
            help=(
                "Supervision target volume: 'mask' optimizes a probability/density field; "
                "'ct' optimizes the CT/MR intensity volume; "
                "'joint' optimizes both (dual render) with configurable weights."
            ),
        )
        self._register("supervision_target")

        core.add_argument(
            "--mask_loss_weight",
            type=float,
            default=1.0,
            help=(
                "Weight applied to the mask/density loss when supervision_target='joint'. "
                "Ignored for supervision_target='ct'."
            ),
        )
        self._register("mask_loss_weight")

        core.add_argument(
            "--ct_loss_type",
            type=str,
            default="mse",
            choices=["mse", "dice", "tversky", "kl"],
            help=(
                "Loss family for CT/intensity supervision when supervision_target is 'ct' or 'joint'. "
                "In joint mode, this is the CT branch; the mask branch uses --volume_loss_type."
            ),
        )
        self._register("ct_loss_type")

        core.add_argument(
            "--ct_loss_weight",
            type=float,
            default=1.0,
            help=(
                "Weight applied to the CT/intensity loss when supervision_target='joint'. "
                "Ignored for supervision_target='mask'."
            ),
        )
        self._register("ct_loss_weight")

        core.add_argument(
            "--density_scale",
            type=float,
            default=1.0,
            help=(
                "Multiplier applied to accumulated density before squashing when render_mode='density'. "
                "Lower values reduce saturation (fewer hard-white blobs)."
            ),
        )
        self._register("density_scale")

        core.add_argument(
            "--mask_loss_threshold_rel",
            type=float,
            default=0.01,
            help=(
                "Relative mask threshold used to define the loss support. "
                "Loss is computed only where mask > (mask_loss_threshold_rel * mask.max())."
            ),
        )
        self._register("mask_loss_threshold_rel")

        core.add_argument(
            "--opacity_gamma",
            type=float,
            default=1.0,
            help=(
                "Gamma applied to sampled mask probabilities when converting to per-Gaussian opacity: "
                "opacity = clamp(p,0,1) ** opacity_gamma. Use 1.0 for identity."
            ),
        )
        self._register("opacity_gamma")

        core.add_argument(
            "--outside_mask_weight",
            type=float,
            default=0.1,
            help=(
                "Soft penalty applied to predictions outside the mask within the ROI. "
                "Adds outside_mask_weight * mean(pred[outside]^2) to the loss when supervision_target is 'mask' or 'joint'."
            ),
        )
        self._register("outside_mask_weight")

        # Number of Gaussians sampled from the mask.
        core.add_argument(
            "--init_n_points",
            type=int,
            default=5000,
            help="How many Gaussians to sample from the mask during initialization.",
        )
        self._register("init_n_points")

        # Random jitter applied to initial Gaussian positions.
        core.add_argument(
            "--position_noise",
            type=float,
            default=0.01,
            help="Std-dev of positional noise added to mask samples at init time.",
        )
        self._register("position_noise")

        core.add_argument(
            "--init_mask_threshold",
            type=float,
            default=0.05,
            help="Minimum mask intensity (0-1) required for a voxel to spawn an initial Gaussian.",
        )
        self._register("init_mask_threshold")

        core.add_argument(
            "--structure_mask_threshold",
            type=float,
            default=0.1,
            help="Mask cutoff applied when building the Hessian field used for vessel alignment.",
        )
        self._register("structure_mask_threshold")

        core.add_argument(
            "--structure_sigma",
            type=float,
            default=1.5,
            help="Gaussian blur (voxels) applied before computing Hessians for vessel cues.",
        )
        self._register("structure_sigma")

        core.add_argument(
            "--structure_min_vesselness",
            type=float,
            default=0.35,
            help="Minimum vesselness required before anisotropic stretching is applied.",
        )
        self._register("structure_min_vesselness")

        core.add_argument(
            "--anisotropy_strength",
            type=float,
            default=0.0,
            help="Amount of stretch applied along vessel axes when Hessian cues are reliable.",
        )
        self._register("anisotropy_strength")

        core.add_argument(
            "--structure_orientation_strength",
            type=float,
            default=0.0,
            help=(
                "Blend Hessian-derived vessel orientations into the initial "
                "Gaussian rotations when structure cues are reliable. "
                "0.0 keeps the current gradient-driven orientation only; "
                "1.0 fully adopts the Hessian orientation for eligible seeds."
            ),
        )
        self._register("structure_orientation_strength")

        core.add_argument(
            "--init_anisotropy_ratio",
            type=float,
            default=1.0,
            help=(
                "Global anisotropy ratio applied to all Gaussians at initialization. "
                "Values > 1.0 make splats anisotropic (stretch axis-2 in the local frame and shrink the other axes to roughly preserve volume). "
                "Set to 1.0 to keep isotropic initialization."
            ),
        )
        self._register("init_anisotropy_ratio")

        core.add_argument(
            "--border_distance_vox",
            type=float,
            default=0.0,
            help=(
                "If > 0, classifies Gaussians within this many voxels of the mask boundary as 'border' and aligns them to the surface normal (mask gradient) at init time."
            ),
        )
        self._register("border_distance_vox")

        core.add_argument(
            "--border_flatten_ratio",
            type=float,
            default=1.0,
            help=(
                "Border-only flattening ratio applied at init time (requires --border_distance_vox > 0). "
                "Values > 1.0 make border splats flatter by shrinking the local axis-2 scale and expanding the tangential axes to roughly preserve volume."
            ),
        )
        self._register("border_flatten_ratio")

        core.add_argument(
            "--border_grad_sigma",
            type=float,
            default=1.5,
            help=(
                "Gaussian blur (voxels) applied before computing mask gradients for border surface normals. "
                "Used only when --border_distance_vox > 0."
            ),
        )
        self._register("border_grad_sigma")

        core.add_argument(
            "--structure_guidance_start_iter",
            type=int,
            default=-1,
            help=(
                "Iteration at which runtime structure guidance begins. "
                "Set < 0 to disable late structure-aligned geometry updates."
            ),
        )
        self._register("structure_guidance_start_iter")

        core.add_argument(
            "--structure_guidance_end_iter",
            type=int,
            default=-1,
            help=(
                "Iteration at which runtime structure guidance reaches full "
                "strength. Values after this keep using the final guidance."
            ),
        )
        self._register("structure_guidance_end_iter")

        core.add_argument(
            "--structure_guidance_interval",
            type=int,
            default=0,
            help=(
                "Iteration cadence for runtime structure guidance updates. "
                "Use 0 to disable."
            ),
        )
        self._register("structure_guidance_interval")

        core.add_argument(
            "--structure_guidance_rotation_strength",
            type=float,
            default=0.0,
            help=(
                "Blend strength for nudging runtime Gaussian rotations toward "
                "local structure directions."
            ),
        )
        self._register("structure_guidance_rotation_strength")

        core.add_argument(
            "--structure_guidance_anisotropy_strength",
            type=float,
            default=0.0,
            help=(
                "Blend strength for nudging runtime Gaussian scales toward a "
                "moderate structure-aligned anisotropy target."
            ),
        )
        self._register("structure_guidance_anisotropy_strength")

        core.add_argument(
            "--structure_guidance_target_ratio",
            type=float,
            default=1.6,
            help=(
                "Target axial-to-radial scale ratio used by runtime structure "
                "guidance. Values near 1.4-1.8 are usually stable."
            ),
        )
        self._register("structure_guidance_target_ratio")

        core.add_argument(
            "--structure_guidance_threshold",
            type=float,
            default=0.12,
            help=(
                "Minimum structure confidence before runtime guidance starts "
                "blending rotations and anisotropy."
            ),
        )
        self._register("structure_guidance_threshold")

        core.add_argument(
            "--orientation_sigma_grad",
            type=float,
            default=0.8,
            help=(
                "Gaussian blur (voxels) applied before computing the gradient "
                "orientation field used for initial rotations."
            ),
        )
        self._register("orientation_sigma_grad")

        core.add_argument(
            "--orientation_sigma_tensor",
            type=float,
            default=0.0,
            help=(
                "Additional post-smoothing (voxels) applied to the gradient "
                "orientation field before quaternion extraction."
            ),
        )
        self._register("orientation_sigma_tensor")

        core.add_argument(
            "--orientation_perturb_deg",
            type=float,
            default=2.0,
            help=(
                "Small random perturbation in degrees applied to initial "
                "orientation quaternions to avoid perfectly identical axes."
            ),
        )
        self._register("orientation_perturb_deg")

        legacy = parser.add_argument_group("Legacy RGB Inputs (kept for compatibility)")

        # COLMAP/SfM dataset directory.
        legacy.add_argument(
            "--source_path",
            type=str,
            default="",
            help="Directory with RGB assets (ignored for pure volume training).",
        )
        self._register("source_path")

        # Relative image folder.
        legacy.add_argument(
            "--images",
            type=str,
            default="images",
            help="Image folder name inside --source_path (RGB workflows only).",
        )
        self._register("images")

        # Relative depth folder.
        legacy.add_argument(
            "--depths",
            type=str,
            default="",
            help="Depth folder name inside --source_path (RGB workflows only).",
        )
        self._register("depths")

        # Target render resolution.
        legacy.add_argument(
            "--resolution",
            type=int,
            default=-1,
            help="Image resolution override for RGB renders (ignored for volumes).",
        )
        self._register("resolution")

        # Spherical-harmonics degree for RGB appearance.
        legacy.add_argument(
            "--sh_degree",
            type=int,
            default=3,
            help="Maximum SH degree used when RGB rendering is enabled.",
        )
        self._register("sh_degree")

        # Background color toggle for legacy renders.
        legacy.add_argument(
            "--white_background",
            action="store_true",
            help="Render RGB frames on white instead of black background.",
        )
        self._register("white_background")

        # Device used to stage datasets before training.
        legacy.add_argument(
            "--data_device",
            type=str,
            default="cuda",
            help="Device used to host dataset tensors (cuda or cpu).",
        )
        self._register("data_device")

        # Toggles train/test exposure split (RGB feature, retained for parity).
        legacy.add_argument(
            "--train_test_exp",
            action="store_true",
            help="Enable separate train/test exposure parameters (RGB only).",
        )
        self._register("train_test_exp")

        # Evaluation-only flag.
        legacy.add_argument(
            "--eval",
            action="store_true",
            help="Skip training and run evaluation only (legacy behavior).",
        )
        self._register("eval")

    def extract(self, args: Namespace) -> GroupParams:  # type: ignore[override]
        params = super().extract(args)
        params.source_path = os.path.abspath(params.source_path)
        return params


class ExportParams(ParamGroup):
    """Controls related to periodic PLY exports during training."""

    def __init__(self, parser: ArgumentParser) -> None:
        super().__init__()
        group = parser.add_argument_group("PLY Export Options")

        group.add_argument(
            "--save_ply_every",
            type=int,
            default=1,
            help="Save a PLY snapshot every N iterations (1 = every iteration).",
        )
        self._register("save_ply_every")

        group.add_argument(
            "--ply_output_prefix",
            type=str,
            default="gaussians",
            help="Filename prefix used for exported PLY files.",
        )
        self._register("ply_output_prefix")

        group.add_argument(
            "--export_ao",
            action="store_true",
            help="Bake fast ambient occlusion from the mask volume into exported PLY colors and add an 'ao' attribute.",
        )
        self._register("export_ao")

        group.add_argument(
            "--export_ao_radius_vox",
            type=int,
            default=2,
            help="AO neighborhood radius in voxels (used when --export_ao is set).",
        )
        self._register("export_ao_radius_vox")

        group.add_argument(
            "--export_ao_n_samples",
            type=int,
            default=12,
            help="Reserved for future ray-sampled AO (currently unused by the fast precompute).",
        )
        self._register("export_ao_n_samples")

        group.add_argument(
            "--export_ao_method",
            type=str,
            default="isotropic",
            choices=["isotropic", "normal"],
            help="AO precompute method: 'isotropic' uses local occupancy; 'normal' refines surface voxels using hemisphere occupancy oriented by mask-gradient normals.",
        )
        self._register("export_ao_method")

        group.add_argument(
            "--export_ao_strength",
            type=float,
            default=1.0,
            help="Strength of AO applied to exported PLY colors (0 = off, 1 = full).",
        )
        self._register("export_ao_strength")


class TrainingScriptParams(ParamGroup):
    """General-purpose knobs for debug/IO behavior of train.py."""

    def __init__(self, parser: ArgumentParser) -> None:
        super().__init__()
        group = parser.add_argument_group("Training Script Controls")

        group.add_argument(
            "--debug_from",
            type=int,
            default=-1,
            help="Iteration at which to enable verbose debugging output.",
        )
        self._register("debug_from")

        group.add_argument(
            "--detect_anomaly",
            action="store_true",
            help="Enable torch.autograd anomaly detection for debugging.",
        )
        self._register("detect_anomaly")

        group.add_argument(
            "--test_iterations",
            nargs="+",
            type=int,
            default=[7_000, 30_000],
            help="Iteration indices used for intermediate test renders.",
        )
        self._register("test_iterations")

        group.add_argument(
            "--save_iterations",
            nargs="+",
            type=int,
            default=[7_000, 30_000],
            help="Iteration indices at which checkpoints/PLYs are saved.",
        )
        self._register("save_iterations")

        group.add_argument(
            "--quiet",
            action="store_true",
            help="Silence non-critical logging (except progress bars).",
        )
        self._register("quiet")

        group.add_argument(
            "--checkpoint_iterations",
            nargs="+",
            type=int,
            default=[],
            help="Extra iteration ids where checkpoints are forced.",
        )
        self._register("checkpoint_iterations")

        group.add_argument(
            "--start_checkpoint",
            type=str,
            default=None,
            help="Path to an existing checkpoint to warm-start training.",
        )
        self._register("start_checkpoint")

        group.add_argument(
            "--disable_mixed_precision",
            action="store_true",
            help="Run training entirely in FP32 instead of mixed precision.",
        )
        self._register("disable_mixed_precision")

        group.add_argument(
            "--medical_mode",
            type=str,
            choices=["none", "organ", "vessel", "vessel_anisotropy"],
            default="none",
            help=(
                "Optional medical-training presets. Use 'organ', 'vessel', or 'vessel_anisotropy' to enable preset overrides; "
                "use 'none' (default) to keep all CLI knobs (including --init_n_points) fully user-controlled."
            ),
        )
        self._register("medical_mode")

        group.add_argument(
            "--enable_diversity",
            action="store_true",
            help="Opt back into diversity regularizers and scale constraints when using medical presets.",
        )
        self._register("enable_diversity")

        group.add_argument(
            "--enable_diagnostics",
            action="store_true",
            help="Enable verbose monitoring (parameter stats, gradient norms, detailed TensorBoard logs).",
        )
        self._register("enable_diagnostics")

        group.add_argument(
            "--tb_log_interval",
            type=int,
            default=10,
            help=(
                "Write TensorBoard scalars every N iterations (reduces CPU/disk overhead). "
                "Set to 1 for per-iteration logging."
            ),
        )
        self._register("tb_log_interval")

        group.add_argument(
            "--progress_postfix_interval",
            type=int,
            default=10,
            help=(
                "Update the tqdm progress postfix every N iterations (reduces GPU->CPU sync). "
                "Set to 1 for per-iteration postfix updates."
            ),
        )
        self._register("progress_postfix_interval")

        group.add_argument(
            "--disable_render_checkpoint",
            action="store_true",
            help=(
                "Disable torch.utils.checkpoint for volume rendering. Can improve throughput "
                "but increases VRAM usage."
            ),
        )
        self._register("disable_render_checkpoint")

        group.add_argument(
            "--max_points_per_iter",
            type=int,
            default=10000,
            help=(
                "Maximum number of Gaussians rendered per optimization step. "
                "Lower values reduce VRAM peak at the cost of slower convergence per iteration."
            ),
        )
        self._register("max_points_per_iter")

        group.add_argument(
            "--eval_masked_mse_full_roi_interval",
            type=int,
            default=0,
            help=(
                "Run a full-model masked-MSE evaluation every N iterations and at the final iteration. "
                "Set to 0 to disable this evaluation pass."
            ),
        )
        self._register("eval_masked_mse_full_roi_interval")

        group.add_argument(
            "--eval_masked_mse_full_roi_target",
            type=str,
            default="auto",
            choices=["auto", "mask", "ct"],
            help=(
                "Target used for full-model masked-MSE evaluation. "
                "'auto' uses CT for ct/joint supervision and mask for mask-only supervision."
            ),
        )
        self._register("eval_masked_mse_full_roi_target")

        group.add_argument(
            "--eval_masked_mse_full_roi_downscale_factor",
            type=int,
            default=1,
            help=(
                "Working-grid downscale factor used by the periodic full-model masked-MSE evaluation. "
                "Use 1 for native-grid evaluation; larger values trade accuracy for speed."
            ),
        )
        self._register("eval_masked_mse_full_roi_downscale_factor")


class OptimizationParams(ParamGroup):
    """Learning schedules, densification knobs, and regularization weights."""

    def __init__(self, parser: ArgumentParser) -> None:
        super().__init__()

        schedule = parser.add_argument_group("Training Schedule & Optimizer")

        # Total number of optimization iterations.
        schedule.add_argument(
            "--iterations",
            type=int,
            default=30_000,
            help="Total number of gradient steps to run before finishing training.",
        )
        self._register("iterations")

        # Choice of optimizer backend.
        schedule.add_argument(
            "--optimizer_type",
            type=str,
            default="default",
            choices=["default", "sparse_adam"],
            help="Optimizer backend (use 'sparse_adam' only when supported).",
        )
        self._register("optimizer_type")

        # Random background blending for RGB rendering.
        schedule.add_argument(
            "--random_background",
            action="store_true",
            help="Enable random background colors for RGB supervision.",
        )
        self._register("random_background")

        lr = parser.add_argument_group("Learning-Rate Schedule")

        # Initial center learning rate.
        lr.add_argument(
            "--position_lr_init",
            type=float,
            default=0.00010,
            help="Initial learning rate for Gaussian centers (slightly reduced for smoother early motion).",
        )
        self._register("position_lr_init")

        # Final center learning rate.
        lr.add_argument(
            "--position_lr_final",
            type=float,
            default=0.0000016,
            help="Final learning rate for Gaussian centers.",
        )
        self._register("position_lr_final")

        # Delay multiplier for the LR ramp.
        lr.add_argument(
            "--position_lr_delay_mult",
            type=float,
            default=1.0,
            help="Multiplier that delays how fast the position LR ramps up. Use 1.0 to start moving immediately.",
        )
        self._register("position_lr_delay_mult")

        # Steps used to transition between LR endpoints.
        lr.add_argument(
            "--position_lr_max_steps",
            type=int,
            default=20_000,
            help="Iterations over which the position LR decays from init to final (shorter decay for longer refinement plateau).",
        )
        self._register("position_lr_max_steps")

        # Appearance-feature LR.
        lr.add_argument(
            "--feature_lr",
            type=float,
            default=0.0025,
            help="Learning rate for SH feature coefficients.",
        )
        self._register("feature_lr")

        # Opacity LR.
        lr.add_argument(
            "--opacity_lr",
            type=float,
            default=0.025,
            help="Learning rate applied to Gaussian opacity logits.",
        )
        self._register("opacity_lr")

        # Scale LR.
        lr.add_argument(
            "--scaling_lr",
            type=float,
            default=0.005,
            help="Learning rate driving Gaussian scale updates.",
        )
        self._register("scaling_lr")

        # Rotation LR.
        lr.add_argument(
            "--rotation_lr",
            type=float,
            default=0.001,
            help="Learning rate for quaternion rotations.",
        )
        self._register("rotation_lr")

        # Exposure LR init (legacy RGB).
        lr.add_argument(
            "--exposure_lr_init",
            type=float,
            default=0.01,
            help="Initial exposure LR (legacy RGB feature).",
        )
        self._register("exposure_lr_init")

        # Exposure LR final (legacy RGB).
        lr.add_argument(
            "--exposure_lr_final",
            type=float,
            default=0.001,
            help="Final exposure LR after scheduling (legacy RGB feature).",
        )
        self._register("exposure_lr_final")

        # Delay before exposure LR decay.
        lr.add_argument(
            "--exposure_lr_delay_steps",
            type=int,
            default=0,
            help="Iterations before exposure LR decay starts (legacy RGB feature).",
        )
        self._register("exposure_lr_delay_steps")

        # Exposure LR decay multiplier.
        lr.add_argument(
            "--exposure_lr_delay_mult",
            type=float,
            default=0.0,
            help="Multiplier applied to exposure LR after the delay window.",
        )
        self._register("exposure_lr_delay_mult")

        densify = parser.add_argument_group("Densification & Regularization")

        densify.add_argument(
            "--enable_densification",
            default=True,
            action="store_true",
            help=(
                "Enable densification + pruning passes during training. "
                "Enabled by default; use --disable_densification to opt out."
            ),
        )
        self._register("enable_densification")

        densify.add_argument(
            "--disable_densification",
            default=False,
            action="store_true",
            help=(
                "Disable densification + pruning passes during training (override). "
                "Overrides --enable_densification."
            ),
        )
        self._register("disable_densification")

        # Dense fraction (legacy compatibility).
        densify.add_argument(
            "--percent_dense",
            type=float,
            default=0.01,
            help="Fraction of Gaussians kept dense (legacy RGB compatibility).",
        )
        self._register("percent_dense")

        # DSSIM weight when RGB supervision is active.
        densify.add_argument(
            "--lambda_dssim",
            type=float,
            default=0.2,
            help="DSSIM loss weight (RGB workflows).",
        )
        self._register("lambda_dssim")

        # Interval between densification passes.
        densify.add_argument(
            "--densification_interval",
            type=int,
            default=100,
            help="Iterations between densification/splitting passes (volume training: keep this relatively large).",
        )
        self._register("densification_interval")

        # Opacity reset cadence.
        densify.add_argument(
            "--opacity_reset_interval",
            type=int,
            default=0,
            help=(
                "Iterations between global opacity resets. "
                "Default 0 disables resets (recommended for volume-only supervision)."
            ),
        )
        self._register("opacity_reset_interval")

        # Iteration at which densification starts.
        densify.add_argument(
            "--densify_from_iter",
            type=int,
            default=100,
            help="Iteration to begin spawning/splitting Gaussians.",
        )
        self._register("densify_from_iter")

        # Iteration after which densification stops.
        densify.add_argument(
            "--densify_until_iter",
            type=int,
            default=2000,
            help=(
                "Iteration after which densification/pruning stops. "
                "Default uses a short early window to avoid late noisy topology changes."
            ),
        )
        self._register("densify_until_iter")

        # Gradient magnitude threshold for densification.
        densify.add_argument(
            "--densify_grad_threshold",
            type=float,
            default=0.0,
            help=(
                "Minimum gradient energy required for densification. "
                "Default 0 relies on adaptive percentile thresholding."
            ),
        )
        self._register("densify_grad_threshold")

        densify.add_argument(
            "--densify_grad_percentile",
            type=float,
            default=0.60,
            help=(
                "Percentile used for adaptive densification thresholding. "
                "Lower values densify more aggressively."
            ),
        )
        self._register("densify_grad_percentile")

        densify.add_argument(
            "--densify_max_new_points",
            type=int,
            default=10000,
            help=(
                "Maximum net new Gaussians added per densification pass "
                "(clone + split + hole-fill). Set <= 0 to disable this cap."
            ),
        )
        self._register("densify_max_new_points")

        densify.add_argument(
            "--low_density_threshold",
            type=float,
            default=4.0,
            help=(
                "Local-density cutoff below which splats are treated as "
                "under-covered for adaptive densification."
            ),
        )
        self._register("low_density_threshold")

        densify.add_argument(
            "--target_coverage",
            type=float,
            default=0.78,
            help=(
                "Desired occupancy coverage before sparse-region hole-fill "
                "is skipped."
            ),
        )
        self._register("target_coverage")

        densify.add_argument(
            "--density_radius_factor",
            type=float,
            default=3.0,
            help=(
                "Multiplier applied to the mean splat scale when estimating "
                "local density."
            ),
        )
        self._register("density_radius_factor")

        densify.add_argument(
            "--density_update_interval",
            type=int,
            default=10,
            help="Iterations between density-cache refreshes.",
        )
        self._register("density_update_interval")

        densify.add_argument(
            "--hole_fill_fraction",
            type=float,
            default=0.05,
            help=(
                "Fraction of the current point count proposed for sparse-region "
                "hole-fill when coverage is below target."
            ),
        )
        self._register("hole_fill_fraction")

        densify.add_argument(
            "--dynamics_log_interval",
            type=int,
            default=50,
            help="Iterations between adaptive-dynamics diagnostic snapshots.",
        )
        self._register("dynamics_log_interval")

        densify.add_argument(
            "--scale_boost_window",
            type=int,
            default=16,
            help="History window used to detect stalled scale updates.",
        )
        self._register("scale_boost_window")

        densify.add_argument(
            "--scale_stall_epsilon",
            type=float,
            default=7e-5,
            help="Average scale-change threshold below which scale motion counts as stalled.",
        )
        self._register("scale_stall_epsilon")

        densify.add_argument(
            "--scale_boost_factor",
            type=float,
            default=1.20,
            help="Temporary scaling-LR multiplier applied when low-density regions stall.",
        )
        self._register("scale_boost_factor")

        densify.add_argument(
            "--scale_boost_duration",
            type=int,
            default=6,
            help="Iterations for which the temporary scaling-LR boost stays active.",
        )
        self._register("scale_boost_duration")

        densify.add_argument(
            "--scale_cooldown",
            type=int,
            default=40,
            help="Iterations to wait before another adaptive scaling boost can trigger.",
        )
        self._register("scale_cooldown")

        densify.add_argument(
            "--structure_gradient_boost",
            type=float,
            default=0.0,
            help=(
                "Boost factor applied to densification gradients in regions with "
                "strong structure cues."
            ),
        )
        self._register("structure_gradient_boost")

        densify.add_argument(
            "--structure_gradient_exponent",
            type=float,
            default=1.0,
            help="Exponent applied to normalized structure strength before reuse.",
        )
        self._register("structure_gradient_exponent")

        densify.add_argument(
            "--structure_gradient_threshold",
            type=float,
            default=0.1,
            help="Minimum normalized structure strength before strong vessel behavior activates.",
        )
        self._register("structure_gradient_threshold")

        densify.add_argument(
            "--vessel_axial_scale",
            type=float,
            default=1.0,
            help="Scale multiplier applied to runtime offspring along the local vessel axis.",
        )
        self._register("vessel_axial_scale")

        densify.add_argument(
            "--vessel_radial_scale",
            type=float,
            default=1.0,
            help="Scale multiplier applied to runtime offspring on the local radial axes.",
        )
        self._register("vessel_radial_scale")

        densify.add_argument(
            "--densify_spawn_jitter_vox",
            type=float,
            default=0.0,
            help="Uniform spawn jitter in voxel units applied to cloned or hole-fill splats.",
        )
        self._register("densify_spawn_jitter_vox")

        densify.add_argument(
            "--densify_vessel_spawn_bias",
            type=float,
            default=0.0,
            help=(
                "Additional weighting applied when hole-fill sampling prefers "
                "strong-structure candidates."
            ),
        )
        self._register("densify_vessel_spawn_bias")

        densify.add_argument(
            "--densify_vessel_spawn_power",
            type=float,
            default=1.0,
            help="Exponent controlling how strongly structure-biased hole-fill sampling concentrates.",
        )
        self._register("densify_vessel_spawn_power")

        densify.add_argument(
            "--prune_min_opacity",
            type=float,
            default=1e-4,
            help="Prune Gaussians whose opacity falls below this threshold.",
        )
        self._register("prune_min_opacity")

        # Depth L1 initial weight (RGB compatibility).
        densify.add_argument(
            "--depth_l1_weight_init",
            type=float,
            default=1.0,
            help="Initial depth-L1 weight (legacy RGB feature).",
        )
        self._register("depth_l1_weight_init")

        # Depth L1 final weight (RGB compatibility).
        densify.add_argument(
            "--depth_l1_weight_final",
            type=float,
            default=0.01,
            help="Final depth-L1 weight after scheduling (legacy RGB feature).",
        )
        self._register("depth_l1_weight_final")

        intensity = parser.add_argument_group("Intensity & Appearance Controls")

        # How grayscale intensities are produced.
        intensity.add_argument(
            "--intensity_mode",
            type=str,
            default="sampled",
            choices=["sampled", "learned", "sampled_mean_covered"],
            help="Strategy for assigning per-Gaussian intensity values.",
        )
        self._register("intensity_mode")

        intensity.add_argument(
            "--opacity_mode",
            type=str,
            default="sampled",
            choices=["sampled", "learned", "sampled_mean_covered"],
            help=(
                "Strategy for assigning per-Gaussian opacity values. "
                "'sampled' samples the input mask and stores a non-learnable buffer; "
                "'learned' optimizes opacity logits; "
                "'sampled_mean_covered' applies mean-covered sampling for large splats."
            ),
        )
        self._register("opacity_mode")

        # Interval between intensity statistic updates.
        intensity.add_argument(
            "--intensity_update_interval",
            type=int,
            default=20,
            help="Iterations between intensity statistic updates (less frequent to reduce jitter).",
        )
        self._register("intensity_update_interval")

        intensity.add_argument(
            "--opacity_update_interval",
            type=int,
            default=20,
            help=(
                "Iterations between sampled-opacity refreshes when opacity_mode is 'sampled'. "
                "Ignored for 'learned'."
            ),
        )
        self._register("opacity_update_interval")

        # Brightness divisor for intensity-to-color mapping.
        intensity.add_argument(
            "--intensity_color_divisor",
            type=float,
            default=1.0,
            help="Divisor applied when mapping intensities to pseudo-RGB colors.",
        )
        self._register("intensity_color_divisor")

        # Threshold for classifying large splats.
        intensity.add_argument(
            "--intensity_large_splat_threshold",
            type=float,
            default=0.03,
            help='Radius threshold used to treat splats as "large" during sampling.',
        )
        self._register("intensity_large_splat_threshold")

        # Radius multiplier for mean-covered sampling.
        intensity.add_argument(
            "--intensity_mean_cover_radius",
            type=float,
            default=2.5,
            help="Neighborhood radius multiplier for mean-covered sampling.",
        )
        self._register("intensity_mean_cover_radius")

        # Interval for recomputing mean-covered intensities.
        intensity.add_argument(
            "--intensity_mean_cover_interval",
            type=int,
            default=30,
            help="Iterations between mean-covered intensity updates (smoother large-splat behavior).",
        )
        self._register("intensity_mean_cover_interval")

        intensity.add_argument(
            "--sparse_support_cutoff",
            type=float,
            default=0.2,
            help=(
                "Minimum Gaussian kernel weight kept in sparse volume splatting. "
                "Lower values produce smoother, less point-like accumulation "
                "at the cost of more compute."
            ),
        )
        self._register("sparse_support_cutoff")

        intensity.add_argument(
            "--sparse_max_radius_vox",
            type=int,
            default=10,
            help=(
                "Maximum sparse splat neighborhood radius (voxels) used during "
                "volume rasterization. Larger values reduce truncation artifacts "
                "but increase memory/compute."
            ),
        )
        self._register("sparse_max_radius_vox")

        intensity.add_argument(
            "--sparse_support_softness",
            type=float,
            default=0.75,
            help=(
                "Soft transition width in sigma units at the sparse-support "
                "boundary. Larger values reduce lattice-like hard cutoffs at "
                "the cost of more overlap and compute."
            ),
        )
        self._register("sparse_support_softness")

        intensity.add_argument(
            "--render_min_sigma_vox",
            type=float,
            default=0.35,
            help=(
                "Minimum rendered Gaussian sigma in working-grid voxel units. "
                "Lower values preserve detail and reduce blur, but values that "
                "are too small can make optimization noisier."
            ),
        )
        self._register("render_min_sigma_vox")

        intensity.add_argument(
            "--sparse_support_cutoff_final",
            type=float,
            default=None,
            help=(
                "Optional final sparse-support cutoff used by the raster schedule. "
                "When omitted, --sparse_support_cutoff stays fixed throughout training."
            ),
        )
        self._register("sparse_support_cutoff_final")

        intensity.add_argument(
            "--sparse_support_softness_final",
            type=float,
            default=None,
            help=(
                "Optional final sparse-support softness used by the raster schedule. "
                "Lower values sharpen detail late in training."
            ),
        )
        self._register("sparse_support_softness_final")

        intensity.add_argument(
            "--render_min_sigma_vox_final",
            type=float,
            default=None,
            help=(
                "Optional final rendered sigma floor used by the raster schedule. "
                "Lower values sharpen the model after topology stabilizes."
            ),
        )
        self._register("render_min_sigma_vox_final")

        intensity.add_argument(
            "--raster_schedule_start_iter",
            type=int,
            default=-1,
            help=(
                "Iteration at which raster parameters begin interpolating toward "
                "their *_final values. Set < 0 to disable scheduling."
            ),
        )
        self._register("raster_schedule_start_iter")

        intensity.add_argument(
            "--raster_schedule_end_iter",
            type=int,
            default=-1,
            help=(
                "Iteration at which raster interpolation reaches the *_final "
                "values. Must be greater than --raster_schedule_start_iter."
            ),
        )
        self._register("raster_schedule_end_iter")

        constraint = parser.add_argument_group("Scale Constraints & Diagnostics")

        # Optional L2 penalty on absolute scales.
        constraint.add_argument(
            "--scale_l2_weight",
            type=float,
            default=0.01,
            help="Weight for the L2 penalty applied to physical Gaussian scales.",
        )
        self._register("scale_l2_weight")

        # Max growth relative to the initial scale.
        constraint.add_argument(
            "--max_scale_factor",
            type=float,
            default=1.5,
            help="Cap on how much a Gaussian scale may grow vs. initialization (slightly tighter to avoid oversized splats).",
        )
        self._register("max_scale_factor")

        constraint.add_argument(
            "--min_scale_vox",
            type=float,
            default=1.5,
            help="Absolute minimum Gaussian scale in voxel units (applied per-axis).",
        )
        self._register("min_scale_vox")

        constraint.add_argument(
            "--max_scale_vox",
            type=float,
            default=10.0,
            help="Absolute maximum Gaussian scale in voxel units (applied per-axis).",
        )
        self._register("max_scale_vox")

        # Initialization scale band (in voxel units).
        constraint.add_argument(
            "--init_scale_min_vox",
            type=float,
            default=1.5,
            help="Minimum initial Gaussian scale in voxel units (used during volume initialization).",
        )
        self._register("init_scale_min_vox")

        constraint.add_argument(
            "--init_scale_max_vox",
            type=float,
            default=4.0,
            help="Maximum initial Gaussian scale in voxel units (used during volume initialization).",
        )
        self._register("init_scale_max_vox")

        # Global spread penalty on log-scales (encourages more uniform splat sizes).
        constraint.add_argument(
            "--scale_logvar_weight",
            type=float,
            default=0.0,
            help="Weight for a global log-scale spread penalty (0 disables).",
        )
        self._register("scale_logvar_weight")

        constraint.add_argument(
            "--scale_logvar_warmup_iters",
            type=int,
            default=0,
            help="Iterations to wait before enabling the log-scale spread penalty.",
        )
        self._register("scale_logvar_warmup_iters")

        constraint.add_argument(
            "--anisotropy_reg_weight",
            type=float,
            default=0.0,
            help=(
                "Weight for the soft anisotropy ratio penalty "
                "(0 disables)."
            ),
        )
        self._register("anisotropy_reg_weight")

        constraint.add_argument(
            "--anisotropy_target_ratio",
            type=float,
            default=2.0,
            help=(
                "Target max/min axis ratio used by anisotropy regularization. "
                "Penalty applies only above this value."
            ),
        )
        self._register("anisotropy_target_ratio")

        constraint.add_argument(
            "--anisotropy_reg_warmup_iters",
            type=int,
            default=0,
            help="Iterations to wait before enabling anisotropy regularization.",
        )
        self._register("anisotropy_reg_warmup_iters")

        # Warmup iterations for the scaling constraint.
        constraint.add_argument(
            "--scaling_constraint_warmup_iters",
            type=int,
            default=1000,
            help="Iterations over which the scale constraint tightens to its final value (earlier stabilization).",
        )
        self._register("scaling_constraint_warmup_iters")

        # Initial relaxation multiplier for the constraint.
        constraint.add_argument(
            "--scaling_constraint_relaxation",
            type=float,
            default=3.0,
            help="Initial relaxation multiplier applied to the scale constraint.",
        )
        self._register("scaling_constraint_relaxation")

        # Cap for how far points can drift relative to their size.
        constraint.add_argument(
            "--position_displacement_scale",
            type=float,
            default=0.0,
            help="Multiplier on the max-axis scale limiting splat displacement from its spawn point. Set <= 0 to disable and allow fully free motion.",
        )
        self._register("position_displacement_scale")

        # Window for logging early Gaussian statistics.
        constraint.add_argument(
            "--early_stats_window",
            type=int,
            default=256,
            help="Number of early iterations that log detailed Gaussian statistics.",
        )
        self._register("early_stats_window")

        diversity = parser.add_argument_group("Parameter Diversity Warmup")

        # Duration of the diversity warmup phase.
        diversity.add_argument(
            "--diversity_warmup_iterations",
            type=int,
            default=1500,
            help="Iterations to keep diversity losses enabled (shorter warmup for smoother shapes).",
        )
        self._register("diversity_warmup_iterations")

        # Logging cadence for diversity diagnostics.
        diversity.add_argument(
            "--diversity_log_interval",
            type=int,
            default=25,
            help="Iterations between diversity diagnostic prints.",
        )
        self._register("diversity_log_interval")

        # Base weight for the scale diversity loss.
        diversity.add_argument(
            "--diversity_scale_weight",
            type=float,
            default=0.02,
            help="Overall strength of the scale diversity loss (kept low to avoid overpowering the data term).",
        )
        self._register("diversity_scale_weight")

        # Base weight for the rotation diversity loss.
        diversity.add_argument(
            "--diversity_rotation_weight",
            type=float,
            default=0.02,
            help="Overall strength of the rotation diversity loss (kept low for stability).",
        )
        self._register("diversity_rotation_weight")

        # Range penalty weight for scale diversity.
        diversity.add_argument(
            "--diversity_scale_range_weight",
            type=float,
            default=0.05,
            help="Penalty pushing scales toward a desired range without dominating the loss.",
        )
        self._register("diversity_scale_range_weight")

        # Target clamp weight for scale diversity.
        diversity.add_argument(
            "--diversity_target_range_weight",
            type=float,
            default=0.05,
            help="Clamp weight reinforcing the preferred scale interval with softer influence.",
        )
        self._register("diversity_target_range_weight")

        # Rotation entropy weight.
        diversity.add_argument(
            "--diversity_rotation_entropy_weight",
            type=float,
            default=0.1,
            help="Encourages diverse quaternion orientations without overwhelming gradients.",
        )
        self._register("diversity_rotation_entropy_weight")

        # Quaternion dispersion weight.
        diversity.add_argument(
            "--diversity_dispersion_weight",
            type=float,
            default=0.1,
            help="Penalizes quaternions collapsing toward identity while staying gentle.",
        )
        self._register("diversity_dispersion_weight")

        # Alignment weight with gradient-derived directions.
        diversity.add_argument(
            "--diversity_alignment_weight",
            type=float,
            default=0.1,
            help="Aligns Gaussians with local volume gradients when set > 0.",
        )
        self._register("diversity_alignment_weight")


class PipelineParams(ParamGroup):
    """Low-level pipeline toggles retained for debugging/compatibility."""

    def __init__(self, parser: ArgumentParser) -> None:
        super().__init__()
        group = parser.add_argument_group("Pipeline Toggles")

        # Forces SH conversion to happen on the Python side.
        group.add_argument(
            "--convert_SHs_python",
            action="store_true",
            help="Convert spherical harmonics coefficients on CPU/Python for debugging.",
        )
        self._register("convert_SHs_python")

        # Forces covariance computation to happen on CPU/Python.
        group.add_argument(
            "--compute_cov3D_python",
            action="store_true",
            help="Compute 3D covariance matrices on CPU/Python for debugging.",
        )
        self._register("compute_cov3D_python")

        # Enables verbose diagnostics in the rasterization pipeline.
        group.add_argument(
            "--debug",
            action="store_true",
            help="Enable verbose debugging prints inside the rasterizer.",
        )
        self._register("debug")

        # Enables antialiasing when using the legacy RGB rasterizer.
        group.add_argument(
            "--antialiasing",
            action="store_true",
            help="Turn on rasterizer anti-aliasing (legacy RGB workflows).",
        )
        self._register("antialiasing")


def get_combined_args(parser: ArgumentParser) -> Namespace:
    """Merge CLI arguments with a saved cfg_args file when present."""

    cmdline_args = parser.parse_args(sys.argv[1:])
    cfg_contents = "Namespace()"

    try:
        cfg_path = os.path.join(cmdline_args.model_path, "cfg_args")
        print("Looking for config file in", cfg_path)
        with open(cfg_path) as cfg_file:
            print(f"Config file found: {cfg_path}")
            cfg_contents = cfg_file.read()
    except (TypeError, FileNotFoundError):
        print("Config file not found; using command-line arguments only.")

    cfg_args = eval(cfg_contents)
    merged = vars(cfg_args).copy()
    for key, value in vars(cmdline_args).items():
        if value is not None:
            merged[key] = value
    return Namespace(**merged)
