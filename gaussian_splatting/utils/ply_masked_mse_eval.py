"""Helpers for evaluating external Gaussian PLY files against volumes."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import torch


def parse_cfg_args_namespace(path: str) -> dict[str, Any]:
    """Parse a ``train.py`` ``cfg_args`` file written as ``Namespace(...)``."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"cfg_args file not found: {cfg_path}. Provide --cfg_args_path or "
            "a valid --training_model_path that contains cfg_args."
        )
    if cfg_path.is_dir():
        raise IsADirectoryError(
            f"Expected cfg_args file path but got directory: {cfg_path}"
        )

    text = cfg_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"cfg_args file is empty: {cfg_path}")

    try:
        expression = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid cfg_args syntax in {path!r}.") from exc

    if not isinstance(expression, ast.Call):
        raise ValueError("cfg_args must contain a Namespace(...) expression.")
    if not isinstance(expression.func, ast.Name) or expression.func.id != "Namespace":
        raise ValueError("cfg_args must start with Namespace(...).")
    if expression.args:
        raise ValueError("cfg_args Namespace(...) must use keyword arguments only.")

    parsed: dict[str, Any] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError("cfg_args may not contain **kwargs expansion.")
        parsed[keyword.arg] = ast.literal_eval(keyword.value)
    return parsed


def resolve_cli_or_cfg(
    cli_value: Any,
    cfg_args: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    """Resolve a setting from CLI override, then cfg_args, then fallback."""
    if cli_value is not None:
        return cli_value
    if key in cfg_args:
        return cfg_args[key]
    return default


def resolve_eval_target(target: str, supervision_target: str) -> str:
    """Resolve the effective full-ROI evaluation target."""
    target_norm = str(target).lower()
    if target_norm == "auto":
        return "ct" if str(supervision_target).lower() in {"ct", "joint"} else "mask"
    if target_norm not in {"mask", "ct"}:
        raise ValueError(
            "target must be one of {'auto', 'mask', 'ct'}, "
            f"got {target!r}."
        )
    return target_norm


def _point_count(gaussians: Any) -> int:
    """Return the number of Gaussians represented by the model."""
    xyz = gaussians.get_xyz
    if xyz.dim() == 2 and xyz.shape[0] == 3 and xyz.shape[1] != 3:
        return int(xyz.shape[1])
    return int(xyz.shape[0])


def _loaded_ply_attributes(gaussians: Any) -> set[str]:
    """Return the attribute names that were present in the last loaded PLY."""
    names = getattr(gaussians, "_loaded_ply_attribute_names", set())
    return set(names) if isinstance(names, (set, frozenset, list, tuple)) else set()


def _has_loaded_intensity_buffer(gaussians: Any) -> bool:
    """Return whether the loaded PLY provided ``intensity_01`` values."""
    n_points = _point_count(gaussians)
    intensities = getattr(gaussians, "intensities", None)
    if not isinstance(intensities, torch.Tensor):
        return False
    if intensities.numel() == 0 or intensities.shape[0] != n_points:
        return False
    return "intensity_01" in _loaded_ply_attributes(gaussians)


def _has_loaded_feature_dc(gaussians: Any) -> bool:
    """Return whether the loaded PLY provided SH/DC appearance values."""
    names = _loaded_ply_attributes(gaussians)
    if not any(name.startswith("f_dc_") for name in names):
        return False

    features_dc = getattr(gaussians, "_features_dc", None)
    if not isinstance(features_dc, torch.Tensor) or features_dc.numel() == 0:
        return False

    return features_dc.shape[0] == _point_count(gaussians)


def configure_gaussian_for_ply_evaluation(
    gaussians: Any,
    *,
    intensity_source: str = "auto",
) -> str:
    """Select which PLY appearance source should drive CT masked-MSE evaluation."""
    source = str(intensity_source).lower()
    if source not in {"auto", "intensity_01", "features_dc"}:
        raise ValueError(
            "intensity_source must be one of "
            "{'auto', 'intensity_01', 'features_dc'}."
        )

    if hasattr(gaussians, "set_opacity_mode"):
        gaussians.set_opacity_mode("learned")

    if source == "auto":
        if _has_loaded_intensity_buffer(gaussians):
            source = "intensity_01"
        elif _has_loaded_feature_dc(gaussians):
            source = "features_dc"
        else:
            raise ValueError(
                "Loaded PLY does not contain a usable CT appearance source. "
                "Expected either 'intensity_01' or 'f_dc_*' attributes."
            )

    if source == "intensity_01":
        if not _has_loaded_intensity_buffer(gaussians):
            raise ValueError(
                "Requested intensity_source='intensity_01', but the loaded "
                "PLY does not provide an 'intensity_01' attribute."
            )
        if hasattr(gaussians, "set_intensity_mode"):
            gaussians.set_intensity_mode("sampled")
        return source

    if not _has_loaded_feature_dc(gaussians):
        raise ValueError(
            "Requested intensity_source='features_dc', but the loaded PLY "
            "does not provide 'f_dc_*' attributes."
        )
    if hasattr(gaussians, "set_intensity_mode"):
        gaussians.set_intensity_mode("learned")
    return source