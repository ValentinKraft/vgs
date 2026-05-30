"""
Parameter update monitoring for 3D Gaussian Splatting.
Tracks the magnitude of parameter changes during optimization.
"""

import torch
from typing import Dict, Optional

class ParameterUpdateTracker:
    """
    Tracks parameter updates during training to detect stagnation.
    """

    def __init__(self):
        """Initialize the parameter update tracker."""
        # Previous parameter values
        self.prev_xyz = None
        self.prev_scaling = None 
        self.prev_rotation = None
        self.prev_opacity = None

        # Running averages of update magnitudes
        self.xyz_update_avg = 0.0
        self.scaling_update_avg = 0.0
        self.rotation_update_avg = 0.0
        self.opacity_update_avg = 0.0

        # EMA decay factor
        self.ema_decay = 0.9

    def update(self, model) -> Dict[str, float]:
        """
        Calculate parameter update magnitudes.
        
        Args:
            model: Gaussian model with current parameters
            
        Returns:
            Dictionary of update magnitudes
        """
        with torch.no_grad():
            result = {}

            # Check if topology changed (densification/pruning occurred)
            topology_changed = getattr(model, "_param_topology_changed", False)
            if topology_changed:
                # Reset tracking when point count changes
                self.prev_xyz = None
                self.prev_scaling = None
                self.prev_rotation = None
                self.prev_opacity = None
                # Reset the flag
                model._param_topology_changed = False

            # Position updates
            if self.prev_xyz is not None and self.prev_xyz.shape == model._xyz.shape:
                xyz_delta = torch.norm(model._xyz - self.prev_xyz) / model._xyz.shape[1]
                result["xyz_delta"] = xyz_delta.item()
                self.xyz_update_avg = self.ema_decay * self.xyz_update_avg + (1 - self.ema_decay) * xyz_delta.item()
                result["xyz_delta_avg"] = self.xyz_update_avg

            # Scale updates
            if (
                self.prev_scaling is not None
                and model._scaling.numel() > 0
                and self.prev_scaling.shape == model._scaling.shape
            ):
                scale_delta = torch.norm(model._scaling - self.prev_scaling) / max(1, model._scaling.shape[0])
                result["scaling_delta"] = scale_delta.item()
                self.scaling_update_avg = self.ema_decay * self.scaling_update_avg + (1 - self.ema_decay) * scale_delta.item()
                result["scaling_delta_avg"] = self.scaling_update_avg

            # Rotation updates
            if (
                self.prev_rotation is not None
                and model._rotation.numel() > 0
                and self.prev_rotation.shape == model._rotation.shape
            ):
                rot_delta = torch.norm(model._rotation - self.prev_rotation) / max(1, model._rotation.shape[0])
                result["rotation_delta"] = rot_delta.item()
                self.rotation_update_avg = self.ema_decay * self.rotation_update_avg + (1 - self.ema_decay) * rot_delta.item()
                result["rotation_delta_avg"] = self.rotation_update_avg

            # Opacity updates
            if (
                self.prev_opacity is not None
                and model._opacity.numel() > 0
                and self.prev_opacity.shape == model._opacity.shape
            ):
                opacity_delta = torch.norm(model._opacity - self.prev_opacity) / max(1, model._opacity.shape[0])
                result["opacity_delta"] = opacity_delta.item()
                self.opacity_update_avg = self.ema_decay * self.opacity_update_avg + (1 - self.ema_decay) * opacity_delta.item()
                result["opacity_delta_avg"] = self.opacity_update_avg

            # Store current parameters for next iteration
            self.prev_xyz = model._xyz.clone()
            if model._scaling.numel() > 0:
                self.prev_scaling = model._scaling.clone()
            if model._rotation.numel() > 0:
                self.prev_rotation = model._rotation.clone()
            if model._opacity.numel() > 0:
                self.prev_opacity = model._opacity.clone()

            return result
