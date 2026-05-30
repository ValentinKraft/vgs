"""Standalone Gaussian PLY viewer application.

This application is intentionally self-contained and does not import the training
or rendering code from the repository.
"""

from __future__ import annotations

import argparse

from gs_viewer.viewer import Viewer


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Gaussian PLY viewer")
    parser.add_argument("--ply", type=str, default="", help="Path to a GaussianModel PLY file")
    return parser.parse_args(argv)


def run_viewer(initial_ply_path: str = "") -> None:
    """Run the interactive viewer."""

    viewer = Viewer(initial_ply_path=initial_ply_path)
    viewer.run()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_viewer(initial_ply_path=args.ply)
