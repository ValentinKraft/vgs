"""Standalone Gaussian PLY viewer entrypoint.

This viewer is intentionally isolated from the training code in this repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Gaussian PLY viewer")
    parser.add_argument("--ply", type=str, default="", help="Path to a GaussianModel PLY file")
    return parser.parse_args()


def main() -> None:
    # Ensure `gs_viewer/src` is on sys.path when running this file directly.
    this_file = Path(__file__).resolve()
    src_dir = this_file.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from gs_viewer.app import run_viewer

    args = _parse_args()
    run_viewer(initial_ply_path=args.ply)


if __name__ == "__main__":
    main()
