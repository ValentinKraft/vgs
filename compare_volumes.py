"""Compare two volumes inside a binary mask using masked MSE and PSNR."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from gaussian_splatting.data.volume_loader import VolumeLoader


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for volume-to-volume masked metrics."""
    parser = argparse.ArgumentParser(
        description=(
            "Load a reference volume, a comparative volume, and a binary mask, "
            "then report masked MSE and PSNR over voxels inside the mask."
        )
    )
    parser.add_argument("--volume_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--comparative_volume_path", type=str, required=True)
    parser.add_argument(
        "--mask_threshold_rel",
        type=float,
        default=0.01,
        help="Relative threshold applied to the normalized mask volume.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for loading and metric computation.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional JSON file path for saving the metric report.",
    )
    return parser


def _resolve_device(requested_device: str) -> torch.device:
    """Resolve the runtime device from the CLI selection."""
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_volume(path: str, device: torch.device) -> torch.Tensor:
    """Load a volume with the same normalization path used by training."""
    loader = VolumeLoader(
        target_shape=None,
        device=device,
        downscale_factor=1,
        storage_dtype="fp32",
        enable_overflow_guard=False,
    )
    return loader.load_volume(path).to(dtype=torch.float32)


def compute_masked_volume_metrics(
    reference_volume: torch.Tensor,
    comparative_volume: torch.Tensor,
    mask_volume: torch.Tensor,
    mask_threshold_rel: float = 0.01,
) -> dict[str, float | int | list[int]]:
    """Compute masked MSE and PSNR for two already loaded volumes."""
    if reference_volume.shape != comparative_volume.shape:
        raise ValueError(
            "Volume shapes must match for comparison. "
            f"reference_shape={tuple(reference_volume.shape)}, "
            f"comparative_shape={tuple(comparative_volume.shape)}"
        )
    if mask_volume.shape != reference_volume.shape:
        raise ValueError(
            "Mask and volume shapes must match for comparison. "
            f"volume_shape={tuple(reference_volume.shape)}, "
            f"mask_shape={tuple(mask_volume.shape)}"
        )

    if mask_threshold_rel < 0.0:
        raise ValueError("mask_threshold_rel must be non-negative.")

    mask_max = float(mask_volume.max().item())
    mask_threshold = float(mask_threshold_rel) * mask_max
    mask_bool = mask_volume > mask_threshold
    if not bool(mask_bool.any()):
        raise RuntimeError(
            "Mask thresholding produced an empty region. "
            f"mask_max={mask_max:.6f}, rel_thr={mask_threshold_rel:.6f}"
        )

    masked_sq_error = (reference_volume[mask_bool] - comparative_volume[mask_bool]) ** 2
    masked_mse = float(masked_sq_error.mean().item())
    masked_psnr = math.inf
    if masked_mse > 0.0:
        masked_psnr = float(20.0 * math.log10(1.0 / math.sqrt(masked_mse)))

    return {
        "masked_mse": masked_mse,
        "masked_psnr": masked_psnr,
        "masked_voxel_count": int(mask_bool.sum().item()),
        "volume_shape": [int(v) for v in reference_volume.shape],
        "mask_threshold_rel": float(mask_threshold_rel),
        "mask_threshold": mask_threshold,
    }


def compare_volumes(
    volume_path: str,
    comparative_volume_path: str,
    mask_path: str,
    device: torch.device,
    mask_threshold_rel: float = 0.01,
) -> dict[str, float | int | list[int]]:
    """Load three volumes and compute masked comparison metrics."""
    reference_volume = _load_volume(volume_path, device)
    comparative_volume = _load_volume(comparative_volume_path, device)
    mask_volume = _load_volume(mask_path, device)
    return compute_masked_volume_metrics(
        reference_volume=reference_volume,
        comparative_volume=comparative_volume,
        mask_volume=mask_volume,
        mask_threshold_rel=mask_threshold_rel,
    )


def main() -> int:
    """Run the volume-to-volume masked metric CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        device = _resolve_device(args.device)
        result = compare_volumes(
            volume_path=args.volume_path,
            comparative_volume_path=args.comparative_volume_path,
            mask_path=args.mask_path,
            device=device,
            mask_threshold_rel=args.mask_threshold_rel,
        )
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    result["device"] = str(device)
    result["volume_path"] = str(Path(args.volume_path))
    result["comparative_volume_path"] = str(Path(args.comparative_volume_path))
    result["mask_path"] = str(Path(args.mask_path))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Masked MSE: {result['masked_mse']:.6f}")
    if math.isinf(float(result["masked_psnr"])):
        print("Masked PSNR: inf")
    else:
        print(f"Masked PSNR: {result['masked_psnr']:.6f} dB")
    print(f"Masked Voxels: {result['masked_voxel_count']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())