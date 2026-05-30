"""PLY import/export helpers for :mod:`scene.gaussian_model`."""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from utils.system_utils import mkdir_p


SH_C0 = 0.28209479177387814
SH_SCALE = 1.77
TRAIN_TEST_EXPECTED_ROWS = 29060
DEBUG_PLY_EXPORT = os.environ.get("GS_PLY_DEBUG", "0") == "1"


def extract_ply_attributes(
    plydata: PlyData,
    prefix: str,
    use_train_test_exp: bool,
) -> np.ndarray:
    """Extract sequential scalar PLY attributes with a common prefix."""
    attributes = []
    i = 0
    while True:
        key = f"{prefix}{i}"
        if key in plydata.elements[0].data.dtype.names:
            attributes.append(np.asarray(plydata.elements[0][key]))
            i += 1
        else:
            break

    if len(attributes) > 0:
        stacked = np.stack(attributes, axis=1)
        if use_train_test_exp:
            stacked = stacked[:TRAIN_TEST_EXPECTED_ROWS, :]
        return stacked

    return np.array([])


def extract_optional_ply_scalar_attribute(
    plydata: PlyData,
    name: str,
    use_train_test_exp: bool,
) -> np.ndarray:
    """Extract one optional scalar PLY attribute as a ``[N, 1]`` array."""
    names = plydata.elements[0].data.dtype.names or ()
    if name not in names:
        return np.array([], dtype=np.float32)

    values = np.asarray(plydata.elements[0][name], dtype=np.float32).reshape(-1, 1)
    if use_train_test_exp:
        values = values[:TRAIN_TEST_EXPECTED_ROWS, :]
    return values


def map_intensities_to_sh_coefficients(
    intensity_values: torch.Tensor,
    volume_min: Optional[float] = None,
    volume_max: Optional[float] = None,
) -> torch.Tensor:
    """Map intensity values to the SH DC coefficient range used for grayscale."""
    intensity_tensor = intensity_values.clone()

    if volume_min is None:
        volume_min = intensity_tensor.min()
    if volume_max is None:
        volume_max = intensity_tensor.max()

    if volume_max > volume_min:
        intensity_tensor = (intensity_tensor - volume_min) / (
            volume_max - volume_min
        )
        intensity_tensor = intensity_tensor.clamp_(0.0, 1.0)
        intensity_tensor = (intensity_tensor * 2.0 - 1.0) * SH_SCALE

    return intensity_tensor


def learned_intensity_from_features(model) -> Optional[torch.Tensor]:
    """Decode scalar [0,1] intensity from SH DC features."""
    if model._features_dc is None or model._features_dc.numel() == 0:
        return None
    if model._features_dc.dim() != 3 or model._features_dc.shape[1] < 1:
        return None

    dc_rgb = model._features_dc[:, 0, :]
    if dc_rgb.numel() == 0:
        return None

    rgb = dc_rgb * float(SH_C0) + 0.5
    return rgb.mean(dim=1, keepdim=True).clamp(0.0, 1.0)


def prepare_colors_for_ply(model, num_points: int) -> np.ndarray:
    """Prepare SH DC color values for PLY export."""
    has_intensity_buffer = (
        hasattr(model, "intensities")
        and model.intensities is not None
        and model.intensities.numel() > 0
        and model.intensities.view(-1).shape[0] == num_points
    )
    has_feature_colors = (
        model._features_dc is not None
        and model._features_dc.numel() > 0
        and model._features_dc.shape[0] == num_points
        and torch.sum(torch.abs(model._features_dc)) > 0
    )
    intensity_mode = getattr(model, "intensity_mode", "learned")

    if not has_intensity_buffer and intensity_mode in {
        "sampled",
        "learned",
        "sampled_mean_covered",
    }:
        print(
            "Warning: export intensity buffer missing or size-mismatched; "
            "falling back to feature-based color source."
        )

    if intensity_mode == "learned" and has_feature_colors:
        if DEBUG_PLY_EXPORT:
            print("Using learned feature DC values for PLY export.")
        return _flatten_feature_dc(model)

    if has_intensity_buffer:
        return create_colors_from_intensities(model, num_points)

    if intensity_mode in {"sampled", "sampled_mean_covered"}:
        return create_colors_from_intensities(model, num_points)

    if has_feature_colors:
        if DEBUG_PLY_EXPORT:
            print("Using provided features for volume rendering.")
        f_dc = _flatten_feature_dc(model)

        if np.allclose(f_dc, 0.0):
            if DEBUG_PLY_EXPORT:
                print("Warning: f_dc values are all zeros; using intensity values.")
            f_dc = create_colors_from_intensities(model, num_points)
    else:
        f_dc = create_colors_from_intensities(model, num_points)

    if DEBUG_PLY_EXPORT:
        print(f"RGB value examples (from features): {f_dc[:5]}")

    return f_dc


def create_colors_from_intensities(model, num_points: int) -> np.ndarray:
    """Create SH DC color values from cached intensity values."""
    if hasattr(model, "intensities") and model.intensities.numel() > 0:
        if DEBUG_PLY_EXPORT:
            print("Creating colors from intensities.")
        raw_tensor = model.intensities.detach().cpu()
        intensity_values = raw_tensor.view(-1).numpy()
        if DEBUG_PLY_EXPORT:
            print(
                f"Raw intensity shape: {tuple(raw_tensor.shape)}, "
                f"range: [{intensity_values.min():.4f}, "
                f"{intensity_values.max():.4f}]"
            )

        normalized = _normalize_intensity_values(model, intensity_values)
        divisor = max(
            abs(float(getattr(model, "intensity_color_divisor", 1.0))),
            1e-8,
        )
        gray01 = np.clip(normalized / divisor, 0.0, 1.0)
        gray_dc = ((gray01 - 0.5) / float(SH_C0)).astype(np.float32)
        return np.repeat(gray_dc[:, None], 3, axis=1)

    if DEBUG_PLY_EXPORT:
        print("Could not find intensity values, using default mid-gray.")
    return np.zeros((num_points, 3), dtype=np.float32)


def prepare_export_intensity01(
    model,
    num_points: int,
    f_dc: np.ndarray,
) -> np.ndarray:
    """Prepare normalized [0,1] scalar intensity values for PLY export."""
    if getattr(model, "intensity_mode", "learned") == "learned":
        normalized = _learned_export_intensity01(model, num_points, f_dc)
    elif (
        hasattr(model, "intensities")
        and model.intensities is not None
        and model.intensities.numel() > 0
    ):
        intensity_values = model.intensities.detach().float().view(-1).cpu().numpy()
        normalized = _normalize_intensity_values(model, intensity_values)
    elif f_dc.shape[0] == num_points:
        normalized = _intensity01_from_f_dc(f_dc)
    else:
        normalized = np.full((num_points,), 0.5, dtype=np.float32)

    normalized = _fit_scalar_array(normalized, num_points)
    return normalized.reshape(-1, 1)


def construct_list_of_attributes(
    model,
    *,
    include_ao: bool = False,
    include_hu: bool = False,
) -> list[str]:
    """Construct attribute names for GaussianModel PLY export."""
    attributes = ["x", "y", "z", "nx", "ny", "nz"]

    if model._features_dc is not None and model._features_dc.numel() > 0:
        for i in range(model._features_dc.shape[1] * model._features_dc.shape[2]):
            attributes.append(f"f_dc_{i}")
    else:
        for i in range(3):
            attributes.append(f"f_dc_{i}")

    if model._features_rest is not None and model._features_rest.numel() > 0:
        for i in range(model._features_rest.shape[1] * model._features_rest.shape[2]):
            attributes.append(f"f_rest_{i}")

    attributes.append("intensity_01")
    if include_hu:
        attributes.append("hu")
    if include_ao:
        attributes.append("ao")

    attributes.append("opacity")

    if model._scaling.numel() > 0:
        for i in range(model._scaling.shape[1]):
            attributes.append(f"scale_{i}")
    else:
        for i in range(3):
            attributes.append(f"scale_{i}")

    if model._rotation.numel() > 0:
        for i in range(model._rotation.shape[1]):
            attributes.append(f"rot_{i}")
    else:
        for i in range(4):
            attributes.append(f"rot_{i}")

    return attributes


def save_ply(
    model,
    path: str,
    *,
    ao: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ao_strength: float = 1.0,
) -> None:
    """Save a GaussianModel-compatible PLY file."""
    mkdir_p(os.path.dirname(path))

    num_points, xyz = _export_xyz(model)
    normals = np.zeros_like(xyz)
    f_dc = prepare_colors_for_ply(model, num_points)
    intensity_01 = prepare_export_intensity01(model, num_points, f_dc)
    hu_values = _prepare_hu_values(model, intensity_01)

    ao_np = _coerce_ao_values(ao, num_points)
    if ao_np is not None:
        f_dc = _apply_ambient_occlusion(f_dc, ao_np, ao_strength)

    f_rest = _export_rest_features(model, num_points)
    opacities = _export_opacities(model, num_points)
    scale = _export_scales(model, num_points)
    xyz, scale = _apply_voxel_size_export(model, xyz, scale)
    xyz, scale = _apply_voxel_spacing_export(model, xyz, scale)
    rotation = _export_rotations(model, num_points)

    create_ply_file(
        model,
        path,
        xyz,
        normals,
        f_dc,
        f_rest,
        intensity_01,
        hu_values,
        opacities,
        scale,
        rotation,
        ao=ao_np,
    )


def create_ply_file(
    model,
    path: str,
    xyz: np.ndarray,
    normals: np.ndarray,
    f_dc: np.ndarray,
    f_rest: np.ndarray,
    intensity_01: np.ndarray,
    hu: Optional[np.ndarray],
    opacities: np.ndarray,
    scale: np.ndarray,
    rotation: np.ndarray,
    ao: Optional[np.ndarray] = None,
) -> None:
    """Create a PLY file from prepared GaussianModel export arrays."""
    num_points = xyz.shape[0]
    attributes_list = construct_list_of_attributes(
        model,
        include_ao=ao is not None,
        include_hu=hu is not None,
    )
    dtype_full = [(attribute, "f4") for attribute in attributes_list]
    elements = np.empty(num_points, dtype=dtype_full)

    all_attributes = [
        xyz,
        normals,
        f_dc,
    ]
    if f_rest.shape[1] > 0:
        all_attributes.append(f_rest)
    all_attributes.append(intensity_01)
    if hu is not None:
        all_attributes.append(hu)
    if ao is not None:
        all_attributes.append(ao)
    all_attributes.extend([opacities, scale, rotation])

    attributes = np.concatenate(all_attributes, axis=1)
    elements[:] = list(map(tuple, attributes))

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)


def save_ply_sequence(
    model,
    output_dir: str,
    iteration: int,
    prefix: str = "gaussians",
    *,
    ao: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ao_strength: float = 1.0,
) -> str:
    """Write the current Gaussian set to a numbered PLY inside ``ply_sequence``."""
    ply_dir = os.path.join(output_dir, "ply_sequence")
    mkdir_p(ply_dir)

    path = os.path.join(ply_dir, f"{prefix}_{iteration:06d}.ply")
    save_ply(model, path, ao=ao, ao_strength=ao_strength)

    if DEBUG_PLY_EXPORT:
        print(f"[ITER {iteration}] Saved model as PLY: {path}")
    return path


def _flatten_feature_dc(model) -> np.ndarray:
    features_tensor = model._features_dc.detach()
    features_tensor = features_tensor.transpose(1, 2)
    features_tensor = features_tensor.flatten(start_dim=1)
    return features_tensor.contiguous().cpu().numpy()


def _normalize_intensity_values(model, intensity_values: np.ndarray) -> np.ndarray:
    already_normalized = (
        intensity_values.size > 0
        and float(intensity_values.min()) >= -0.05
        and float(intensity_values.max()) <= 1.05
    )

    if already_normalized:
        normalized = intensity_values.astype(np.float32)
    elif (
        hasattr(model, "volume_min")
        and hasattr(model, "volume_max")
        and model.volume_max > model.volume_min
    ):
        denom = max(float(model.volume_max) - float(model.volume_min), 1e-8)
        normalized = ((intensity_values - float(model.volume_min)) / denom).astype(
            np.float32
        )
        if DEBUG_PLY_EXPORT:
            print(
                f"Applying cached global min/max "
                f"[{float(model.volume_min):.4f}, {float(model.volume_max):.4f}]"
            )
    else:
        local_min = float(intensity_values.min()) if intensity_values.size > 0 else 0.0
        local_max = float(intensity_values.max()) if intensity_values.size > 0 else 1.0
        if local_max > local_min:
            normalized = (
                (intensity_values - local_min) / (local_max - local_min)
            ).astype(np.float32)
        else:
            normalized = np.full_like(intensity_values, 0.5, dtype=np.float32)
        if DEBUG_PLY_EXPORT:
            print(f"Fallback normalization range [{local_min:.4f}, {local_max:.4f}]")

    return np.clip(normalized, 0.0, 1.0)


def _learned_export_intensity01(
    model,
    num_points: int,
    f_dc: np.ndarray,
) -> np.ndarray:
    learned = learned_intensity_from_features(model)
    if learned is not None and learned.numel() > 0 and learned.shape[0] == num_points:
        return learned.detach().view(-1).cpu().numpy().astype(np.float32)
    if f_dc.shape[0] == num_points:
        return _intensity01_from_f_dc(f_dc)
    return np.full((num_points,), 0.5, dtype=np.float32)


def _intensity01_from_f_dc(f_dc: np.ndarray) -> np.ndarray:
    return np.clip(
        (f_dc * float(SH_C0) + 0.5).mean(axis=1),
        0.0,
        1.0,
    ).astype(np.float32)


def _fit_scalar_array(values: np.ndarray, count: int) -> np.ndarray:
    if values.shape[0] == count:
        return values.astype(np.float32)
    if values.shape[0] > count:
        return values[:count].astype(np.float32)

    pad = np.full((count - values.shape[0],), 0.5, dtype=np.float32)
    return np.concatenate([values.astype(np.float32), pad], axis=0)


def _export_xyz(model) -> tuple[int, np.ndarray]:
    if model._xyz.shape[0] == 3:
        return model._xyz.shape[1], model._xyz.detach().cpu().numpy().T
    return model._xyz.shape[0], model._xyz.detach().cpu().numpy()


def _prepare_hu_values(model, intensity_01: np.ndarray) -> Optional[np.ndarray]:
    raw_min = getattr(model, "raw_volume_min", None)
    raw_max = getattr(model, "raw_volume_max", None)
    if raw_min is None or raw_max is None or float(raw_max) <= float(raw_min):
        return None
    return (
        float(raw_min) + intensity_01 * (float(raw_max) - float(raw_min))
    ).astype(np.float32)


def _coerce_ao_values(
    ao: Optional[Union[torch.Tensor, np.ndarray]],
    num_points: int,
) -> Optional[np.ndarray]:
    if ao is None:
        return None
    if isinstance(ao, torch.Tensor):
        ao_np = ao.detach().float().view(-1, 1).cpu().numpy()
    else:
        ao_np = np.asarray(ao, dtype=np.float32).reshape(-1, 1)

    if ao_np.shape[0] != num_points:
        raise ValueError(
            f"AO length mismatch: expected {num_points}, got {ao_np.shape[0]}"
        )
    return ao_np


def _apply_ambient_occlusion(
    f_dc: np.ndarray,
    ao: np.ndarray,
    ao_strength: float,
) -> np.ndarray:
    strength = max(0.0, min(1.0, float(ao_strength)))
    ao_applied = (1.0 - strength) + strength * np.clip(ao, 0.0, 1.0)
    rgb = f_dc * float(SH_C0) + 0.5
    rgb = np.clip(rgb * ao_applied, 0.0, 1.0)
    return (rgb - 0.5) / float(SH_C0)


def _export_rest_features(model, num_points: int) -> np.ndarray:
    if model._features_rest is not None and model._features_rest.numel() > 0:
        return (
            model._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
    return np.zeros((num_points, 0))


def _export_opacities(model, num_points: int) -> np.ndarray:
    opacity_tensor = model.get_opacity
    if opacity_tensor.numel() == 0:
        opacity_tensor = torch.ones((num_points, 1), device=model._xyz.device)
    opacities = opacity_tensor.detach().cpu().numpy()
    if opacities.shape[0] != num_points:
        return np.ones((num_points, 1))
    return opacities


def _export_scales(model, num_points: int) -> np.ndarray:
    scale = model._scaling.detach().cpu().numpy()
    if scale.shape[0] != num_points:
        return np.ones((num_points, 3)) * 0.01
    return scale


def _apply_voxel_size_export(
    model,
    xyz: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    voxel_size = getattr(model, "voxel_size", None)
    if voxel_size is None:
        return xyz, scale

    voxel_np = np.asarray(
        torch.as_tensor(voxel_size, dtype=torch.float32).view(-1).cpu().numpy(),
        dtype=np.float32,
    )
    if voxel_np.size == 1:
        voxel_np = np.repeat(voxel_np, 3)
    if voxel_np.size < 3:
        return xyz, scale

    voxel_xyz = np.clip(voxel_np[:3], 1e-8, None)
    xyz = (xyz / voxel_xyz.reshape(1, 3)).astype(np.float32)
    scale = (scale - np.log(voxel_xyz.reshape(1, 3))).astype(np.float32)
    return xyz, scale


def _apply_voxel_spacing_export(
    model,
    xyz: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    voxel_spacing = getattr(model, "voxel_spacing_xyz", None)
    if voxel_spacing is None:
        return xyz, scale

    spacing_np = np.asarray(
        torch.as_tensor(voxel_spacing, dtype=torch.float32).view(-1).cpu().numpy(),
        dtype=np.float32,
    )
    if spacing_np.size == 1:
        spacing_np = np.repeat(spacing_np, 3)
    if spacing_np.size < 3:
        return xyz, scale

    spacing_xyz = np.clip(spacing_np[:3], 1e-8, None)
    xyz = (xyz * spacing_xyz.reshape(1, 3)).astype(np.float32)
    scale = (scale + np.log(spacing_xyz.reshape(1, 3))).astype(np.float32)
    return xyz, scale


def _export_rotations(model, num_points: int) -> np.ndarray:
    rotation = model._rotation.detach().cpu().numpy()
    if rotation.shape[0] == num_points:
        return rotation

    rotation = np.zeros((num_points, 4))
    rotation[:, 0] = 1
    return rotation
