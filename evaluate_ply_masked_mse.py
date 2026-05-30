"""Evaluate a Gaussian PLY with the full-ROI masked-MSE metric."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from gaussian_splatting.utils.ply_masked_mse_eval import (
    configure_gaussian_for_ply_evaluation,
    parse_cfg_args_namespace,
    resolve_cli_or_cfg,
    resolve_eval_target,
)
from gaussian_splatting.utils.volume_supervisor import VolumeSupervisor
from scene.gaussian_model import GaussianModel


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for external PLY masked-MSE evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Load an external Gaussian PLY, rasterize it with the same "
            "splat-to-volume path used during training, and compute masked MSE."
        )
    )
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument(
        "--cfg_args_path",
        type=str,
        default=None,
        help="Path to a saved cfg_args file written by train.py.",
    )
    parser.add_argument(
        "--training_model_path",
        type=str,
        default=None,
        help=(
            "Path to a training output directory containing cfg_args. "
            "This is the run folder, not the input volume path."
        ),
    )
    parser.add_argument("--volume_path", type=str, default=None)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        choices=["auto", "mask", "ct"],
        help=(
            "Evaluation target. If omitted, inherit "
            "eval_masked_mse_full_roi_target from cfg_args when available."
        ),
    )
    parser.add_argument(
        "--intensity_source",
        type=str,
        default="auto",
        choices=["auto", "intensity_01", "features_dc"],
        help=(
            "PLY appearance source for CT evaluation. 'auto' prefers the "
            "exported intensity_01 buffer when present, otherwise falls back "
            "to SH/DC features."
        ),
    )
    parser.add_argument("--working_grid_downscale_factor", type=int, default=None)
    parser.add_argument("--volume_downscale_factor", type=int, default=None)
    parser.add_argument("--volume_render_downscale_factor", type=int, default=None)
    parser.add_argument(
        "--volume_storage_dtype",
        type=str,
        default=None,
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument(
        "--disable_volume_overflow_guard",
        action="store_true",
        help="Disable the volume loader overflow guard for this evaluation.",
    )
    parser.add_argument(
        "--supervision_target",
        type=str,
        default=None,
        choices=["mask", "ct", "joint"],
    )
    parser.add_argument("--density_scale", type=float, default=None)
    parser.add_argument("--opacity_gamma", type=float, default=None)
    parser.add_argument("--sparse_support_cutoff", type=float, default=None)
    parser.add_argument("--sparse_max_radius_vox", type=int, default=None)
    parser.add_argument("--sparse_support_softness", type=float, default=None)
    parser.add_argument("--render_min_sigma_vox", type=float, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    return parser


def _resolve_cfg_args_path(args: argparse.Namespace) -> str | None:
    """Return the cfg_args path requested by the user, if any."""
    if args.cfg_args_path:
        cfg_path = Path(args.cfg_args_path)
        if cfg_path.is_dir():
            cfg_path = cfg_path / "cfg_args"
        return str(cfg_path)

    if args.training_model_path:
        model_path = Path(args.training_model_path)
        if model_path.is_file():
            if model_path.name == "cfg_args":
                return str(model_path)
            raise ValueError(
                "--training_model_path must point to a training output directory "
                "that contains cfg_args, not a volume file. Use --volume_path and "
                "--mask_path for direct evaluation inputs, or pass --cfg_args_path "
                "to a saved cfg_args file."
            )
        return str(model_path / "cfg_args")

    return None


def _require_path(value: Any, name: str) -> str:
    """Require that a path-like argument has been resolved."""
    if not value:
        raise ValueError(
            f"Missing required setting {name!r}. Provide it explicitly or via cfg_args."
        )
    return str(value)


def main() -> int:
    """Run the external PLY masked-MSE evaluation."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        cfg_args_path = _resolve_cfg_args_path(args)
        cfg_args = parse_cfg_args_namespace(cfg_args_path) if cfg_args_path else {}
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        parser.error(str(exc))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    supervision_target = str(
        resolve_cli_or_cfg(
            args.supervision_target,
            cfg_args,
            "supervision_target",
            "mask",
        )
    )
    target = str(
        resolve_cli_or_cfg(
            args.target,
            cfg_args,
            "eval_masked_mse_full_roi_target",
            "auto",
        )
    )
    effective_target = resolve_eval_target(target, supervision_target)

    volume_path = _require_path(
        resolve_cli_or_cfg(args.volume_path, cfg_args, "volume_path", None),
        "volume_path",
    )
    mask_path = _require_path(
        resolve_cli_or_cfg(args.mask_path, cfg_args, "mask_path", None),
        "mask_path",
    )
    volume_shape = tuple(
        resolve_cli_or_cfg(None, cfg_args, "volume_shape", [64, 64, 64])
    )
    volume_downscale_factor = int(
        resolve_cli_or_cfg(
            args.volume_downscale_factor,
            cfg_args,
            "volume_downscale_factor",
            1,
        )
    )
    volume_render_downscale_factor = int(
        resolve_cli_or_cfg(
            args.volume_render_downscale_factor,
            cfg_args,
            "volume_render_downscale_factor",
            2,
        )
    )
    volume_storage_dtype = str(
        resolve_cli_or_cfg(
            args.volume_storage_dtype,
            cfg_args,
            "volume_storage_dtype",
            "fp32",
        )
    )
    density_scale = float(
        resolve_cli_or_cfg(args.density_scale, cfg_args, "density_scale", 1.0)
    )
    opacity_gamma = float(
        resolve_cli_or_cfg(args.opacity_gamma, cfg_args, "opacity_gamma", 1.0)
    )
    sparse_support_cutoff = float(
        resolve_cli_or_cfg(
            args.sparse_support_cutoff,
            cfg_args,
            "sparse_support_cutoff",
            0.2,
        )
    )
    sparse_max_radius_vox = int(
        resolve_cli_or_cfg(
            args.sparse_max_radius_vox,
            cfg_args,
            "sparse_max_radius_vox",
            10,
        )
    )
    sparse_support_softness = float(
        resolve_cli_or_cfg(
            args.sparse_support_softness,
            cfg_args,
            "sparse_support_softness",
            0.75,
        )
    )
    render_min_sigma_vox = float(
        resolve_cli_or_cfg(
            args.render_min_sigma_vox,
            cfg_args,
            "render_min_sigma_vox",
            0.35,
        )
    )
    working_grid_downscale_factor = int(
        resolve_cli_or_cfg(
            args.working_grid_downscale_factor,
            cfg_args,
            "eval_masked_mse_full_roi_downscale_factor",
            1,
        )
    )

    volume_supervisor = VolumeSupervisor(
        volume_path=volume_path,
        mask_path=mask_path,
        volume_shape=volume_shape,
        volume_downscale_factor=volume_downscale_factor,
        volume_render_downscale_factor=volume_render_downscale_factor,
        volume_storage_dtype=volume_storage_dtype,
        disable_volume_overflow_guard=bool(
            args.disable_volume_overflow_guard
            or bool(cfg_args.get("disable_volume_overflow_guard", False))
        ),
        supervision_target=supervision_target,
        density_scale=density_scale,
        opacity_gamma=opacity_gamma,
        sparse_support_cutoff=sparse_support_cutoff,
        sparse_max_radius_vox=sparse_max_radius_vox,
        sparse_support_softness=sparse_support_softness,
        render_min_sigma_vox=render_min_sigma_vox,
        device=device,
    )

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(args.ply_path, device=device)
    gaussians.set_opacity_mode("learned")

    resolved_intensity_source = "unused"
    if effective_target == "ct":
        resolved_intensity_source = configure_gaussian_for_ply_evaluation(
            gaussians,
            intensity_source=args.intensity_source,
        )

    masked_mse, resolved_target = volume_supervisor.compute_full_roi_masked_mse(
        gaussians,
        target=target,
        working_grid_downscale_factor=working_grid_downscale_factor,
        refresh_appearance=False,
    )

    result = {
        "ply_path": os.path.abspath(args.ply_path),
        "cfg_args_path": None if cfg_args_path is None else os.path.abspath(cfg_args_path),
        "volume_path": os.path.abspath(volume_path),
        "mask_path": os.path.abspath(mask_path),
        "target": resolved_target,
        "intensity_source": resolved_intensity_source,
        "masked_mse": float(masked_mse),
        "working_grid_downscale_factor": working_grid_downscale_factor,
        "device": str(device),
    }

    print(f"PLY: {result['ply_path']}")
    print(f"Target: {result['target']}")
    print(f"Intensity source: {result['intensity_source']}")
    print(f"Working-grid downscale: {result['working_grid_downscale_factor']}")
    print(f"Masked MSE: {result['masked_mse']:.6f}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"Saved JSON summary to {os.path.abspath(args.output_json)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())