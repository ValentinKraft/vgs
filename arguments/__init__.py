# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

"""Compatibility shim that forwards legacy imports to ``arguments.py``."""

from __future__ import annotations

from importlib import util
from pathlib import Path
from types import ModuleType
import sys

_UNIFIED_MODULE_NAME = "_arguments_unified_impl"
_UNIFIED_SOURCE = Path(__file__).resolve().parent.parent / "arguments.py"


def _load_unified_module() -> ModuleType:
    """Load the root ``arguments.py`` once and cache it in ``sys.modules``."""

    if _UNIFIED_MODULE_NAME in sys.modules:
        return sys.modules[_UNIFIED_MODULE_NAME]

    if not _UNIFIED_SOURCE.exists():
        raise ImportError(f"Unified arguments module missing at {_UNIFIED_SOURCE}")

    spec = util.spec_from_file_location(_UNIFIED_MODULE_NAME, _UNIFIED_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to create a module spec for unified arguments")

    module = util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)  # type: ignore[assignment]
    sys.modules[_UNIFIED_MODULE_NAME] = module
    return module


_unified = _load_unified_module()

GroupParams = _unified.GroupParams
ParamGroup = _unified.ParamGroup
ModelParams = _unified.ModelParams
PipelineParams = _unified.PipelineParams
OptimizationParams = _unified.OptimizationParams
ExportParams = _unified.ExportParams
TrainingScriptParams = _unified.TrainingScriptParams
get_combined_args = _unified.get_combined_args

__all__ = [
    "GroupParams",
    "ParamGroup",
    "ModelParams",
    "PipelineParams",
    "OptimizationParams",
    "ExportParams",
    "TrainingScriptParams",
    "get_combined_args",
]
