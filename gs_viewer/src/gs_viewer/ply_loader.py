"""PLY loader for GaussianModel-exported splats.

This loader supports the PLY schema produced by `scene/gaussian_model.py`:
- x,y,z
- f_dc_0..2 (SH DC coefficients)
- intensity_01 (optional explicit normalized per-splat intensity)
- hu (optional per-splat HU value)
- opacity
- scale_0..2 (log-scale)
- rot_0..3 (quaternion)
- optional ao

The viewer uses DC-only shading; any `f_rest_*` SH coefficients are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import numpy as np
from plyfile import PlyData

SH_C0: float = 0.28209479177387814


@dataclass(frozen=True)
class GaussianModelPly:
    """Gaussian splat model loaded from PLY."""

    positions: np.ndarray  # (N, 3) float32
    f_dc: np.ndarray  # (N, 3) float32
    opacity: np.ndarray  # (N,) float32 in [0,1]
    log_scale: np.ndarray  # (N, 3) float32
    quat: np.ndarray  # (N, 4) float32
    ao: np.ndarray | None  # (N,) float32 in [0,1]
    hu: np.ndarray | None  # (N,) float32 HU values

    base_rgb01: np.ndarray  # (N, 3) float32 in [0,1]
    intensity01: np.ndarray  # (N,) float32 in [0,1]

    bounds_center: np.ndarray  # (3,) float32
    bounds_radius: float

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    def normalized_for_view(self) -> "GaussianModelPly":
        """Center + scale model to a unit-ish space for stable navigation."""

        pos = self.positions
        pmin = pos.min(axis=0)
        pmax = pos.max(axis=0)
        center = (pmin + pmax) * 0.5
        extent = (pmax - pmin)
        radius = float(np.linalg.norm(extent) * 0.5)
        scale = max(radius, 1e-6)

        positions = ((pos - center) / scale).astype(np.float32)
        log_scale = (self.log_scale - np.log(scale)).astype(np.float32)

        return GaussianModelPly(
            positions=positions,
            f_dc=self.f_dc,
            opacity=self.opacity,
            log_scale=log_scale,
            quat=self.quat,
            ao=self.ao,
            hu=self.hu,
            base_rgb01=self.base_rgb01,
            intensity01=self.intensity01,
            bounds_center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            bounds_radius=1.0,
        )


def load_gaussian_model_ply(path: Path) -> GaussianModelPly:
    """Load a GaussianModel-exported PLY.

    Raises:
        ValueError: If required properties are missing or the schema is not supported.
    """

    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError("PLY has no 'vertex' element")

    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or [])

    if {"red", "green", "blue"}.issubset(names) and "f_dc_0" not in names:
        raise ValueError("Unsupported PLY schema (appears to be a point cloud with red/green/blue)")

    required = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"Missing required PLY properties: {missing}")

    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
    opacity = np.asarray(vertex["opacity"], dtype=np.float32)
    log_scale = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1).astype(np.float32)
    quat = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1).astype(np.float32)

    ao: np.ndarray | None = None
    if "ao" in names:
        ao = np.asarray(vertex["ao"], dtype=np.float32)

    hu: np.ndarray | None = None
    if "hu" in names:
        hu = np.asarray(vertex["hu"], dtype=np.float32)

    base_rgb = np.clip(f_dc * float(SH_C0) + 0.5, 0.0, 1.0).astype(np.float32)
    if "intensity_01" in names:
        intensity = np.clip(np.asarray(vertex["intensity_01"], dtype=np.float32), 0.0, 1.0)
    elif hu is not None:
        hu_min = float(hu.min())
        hu_max = float(hu.max())
        if hu_max > hu_min:
            intensity = np.clip((hu - hu_min) / (hu_max - hu_min), 0.0, 1.0).astype(np.float32)
        else:
            intensity = np.full((positions.shape[0],), 0.5, dtype=np.float32)
    else:
        intensity = np.clip(base_rgb.mean(axis=1), 0.0, 1.0).astype(np.float32)

    pmin = positions.min(axis=0)
    pmax = positions.max(axis=0)
    center = ((pmin + pmax) * 0.5).astype(np.float32)
    radius = float(np.linalg.norm((pmax - pmin) * 0.5))

    quat = _normalize_quat(quat)

    return GaussianModelPly(
        positions=positions,
        f_dc=f_dc,
        opacity=np.clip(opacity, 0.0, 1.0),
        log_scale=log_scale,
        quat=quat,
        ao=None if ao is None else np.clip(ao, 0.0, 1.0),
        hu=hu,
        base_rgb01=base_rgb,
        intensity01=intensity,
        bounds_center=center,
        bounds_radius=max(radius, 1e-6),
    )


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternions along the last axis."""

    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return (q / n).astype(np.float32)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GaussianModel PLY loader self-check")
    parser.add_argument("--ply", type=str, required=True, help="Path to GaussianModel PLY")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load a PLY and print basic stats."""

    args = _parse_args(argv)
    model = load_gaussian_model_ply(Path(args.ply))
    print(f"Loaded: {args.ply}")
    print(f"Splats: {model.count}")
    print(f"Bounds center: {model.bounds_center.tolist()}")
    print(f"Bounds radius: {model.bounds_radius:.6f}")
    print(f"Has AO: {model.ao is not None}")


if __name__ == "__main__":
    main()
