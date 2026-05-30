from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def main() -> None:
    out = Path(__file__).resolve().parent / "minimal_gaussian.ply"

    n = 3
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)

    # Encode grayscale into SH DC the same way this repo does.
    # rgb = f_dc * SH_C0 + 0.5
    sh_c0 = 0.28209479177387814
    gray = np.array([0.2, 0.6, 0.9], np.float32)
    f_dc = ((gray - 0.5) / sh_c0)[:, None].repeat(3, axis=1)

    opacity = np.array([0.2, 0.6, 0.9], np.float32)
    log_scale = np.log(np.array([[0.03, 0.03, 0.03], [0.05, 0.05, 0.05], [0.08, 0.08, 0.08]], np.float32))
    quat = np.array([[1, 0, 0, 0]] * n, np.float32)
    ao = np.array([1.0, 0.8, 0.6], np.float32)

    vertex = np.empty(
        n,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("f_dc_0", "f4"),
            ("f_dc_1", "f4"),
            ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
            ("ao", "f4"),
        ],
    )

    vertex["x"] = pos[:, 0]
    vertex["y"] = pos[:, 1]
    vertex["z"] = pos[:, 2]
    vertex["nx"] = 0
    vertex["ny"] = 0
    vertex["nz"] = 0
    vertex["f_dc_0"] = f_dc[:, 0]
    vertex["f_dc_1"] = f_dc[:, 1]
    vertex["f_dc_2"] = f_dc[:, 2]
    vertex["opacity"] = opacity
    vertex["scale_0"] = log_scale[:, 0]
    vertex["scale_1"] = log_scale[:, 1]
    vertex["scale_2"] = log_scale[:, 2]
    vertex["rot_0"] = quat[:, 0]
    vertex["rot_1"] = quat[:, 1]
    vertex["rot_2"] = quat[:, 2]
    vertex["rot_3"] = quat[:, 3]
    vertex["ao"] = ao

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=False).write(str(out))
    print(f"Wrote sample PLY: {out}")


if __name__ == "__main__":
    main()
