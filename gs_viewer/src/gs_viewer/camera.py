"""Orbit/pan/zoom camera (Unity-like)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v
    return v / n


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Create a right-handed look-at matrix."""

    f = _normalize(target - eye)
    s = _normalize(np.cross(f, up))
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -float(np.dot(s, eye))
    m[1, 3] = -float(np.dot(u, eye))
    m[2, 3] = float(np.dot(f, eye))
    return m


def _perspective(fovy_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    """Create a right-handed perspective projection matrix."""

    fovy_rad = np.deg2rad(fovy_deg)
    f = 1.0 / float(np.tan(fovy_rad / 2.0))

    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / float(aspect)
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


@dataclass
class OrbitCamera:
    """Simple orbit camera.

    Mouse mapping:
    - Left drag: orbit
    - Right drag: pan
    - Middle drag (vertical): zoom
    - Wheel: zoom
    """

    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    yaw: float = 0.0
    pitch: float = 0.0
    distance: float = 2.5
    fovy_deg: float = 45.0

    def frame_bounds(self, center: np.ndarray, radius: float) -> None:
        self.target = center.astype(np.float32).copy()
        self.yaw = 0.0
        self.pitch = 0.0
        self.distance = max(0.25, float(radius) * 3.0)

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw += dx * 0.005
        self.pitch += dy * 0.005
        self.pitch = float(np.clip(self.pitch, -1.55, 1.55))

    def pan(self, dx: float, dy: float) -> None:
        right, up = self._basis_vectors()
        scale = self.distance * 0.001
        self.target = (self.target - right * (dx * scale) + up * (dy * scale)).astype(np.float32)

    def zoom(self, scroll_y: float) -> None:
        factor = 1.0 - scroll_y * 0.1
        factor = float(np.clip(factor, 0.2, 5.0))
        self.distance = float(np.clip(self.distance * factor, 0.05, 1e6))

    def eye_position(self) -> np.ndarray:
        cy = float(np.cos(self.yaw))
        sy = float(np.sin(self.yaw))
        cp = float(np.cos(self.pitch))
        sp = float(np.sin(self.pitch))

        direction = np.array([cp * sy, sp, cp * cy], dtype=np.float32)
        return self.target + direction * float(self.distance)

    def view_matrix(self) -> np.ndarray:
        eye = self.eye_position()
        return _look_at(eye, self.target, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    def proj_matrix(self, aspect: float) -> np.ndarray:
        return _perspective(self.fovy_deg, aspect, 0.01, 1000.0)

    def _basis_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        eye = self.eye_position()
        forward = _normalize(self.target - eye)
        right = _normalize(np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32)))
        up = _normalize(np.cross(right, forward))
        return right, up
