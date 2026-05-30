"""
Parameter diversity losses for 3D Gaussian Splatting.
These losses encourage diversity in scale and rotation parameters.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple, Union, List

class ScaleDiversityLoss(nn.Module):
    """
    Loss to encourage anisotropy and reasonable magnitudes for scales.

    Args:
        orthogonality_weight: Weight for orthogonality loss term
        target_range_weight: Weight for target range loss term
        target_min_scale: Minimum target scale value
        target_max_scale: Maximum target scale value
    """

    def __init__(
        self,
        orthogonality_weight: float = 0.002,
        target_range_weight: float = 0.001,
        target_min_scale: float = 0.0025,
        target_max_scale: float = 0.06,
    ):
        super().__init__()
        self.orthogonality_weight = orthogonality_weight
        self.target_range_weight = target_range_weight
        self.target_min_scale = target_min_scale
        self.target_max_scale = target_max_scale

    def forward(self, scale_params: Tensor) -> Dict[str, Tensor]:
        """
        Compute scale diversity loss components.
        
        Args:
            scale_params: Scale parameters (N, 3)
            
        Returns:
            Dictionary of named loss components and values
        """
        losses = {}

        # Ensure the scale_params are properly shaped
        if len(scale_params.shape) == 1:
            scale_params = scale_params.unsqueeze(-1)

        if scale_params.shape[1] != 3:
            if len(scale_params.shape) == 1:
                # Handle scalar scale parameters
                # Create isotropic scaling by repeating values
                scale_params = scale_params.unsqueeze(-1).repeat(1, 3)
            elif scale_params.shape[1] == 1:
                # Handle (N, 1) format
                scale_params = scale_params.repeat(1, 3)

        # 1. Orthogonality Loss: Penalize if all three dimensions are similar
        # Compute similarity between dimensions
        dim_sim_01 = torch.abs(scale_params[:, 0] - scale_params[:, 1])
        dim_sim_12 = torch.abs(scale_params[:, 1] - scale_params[:, 2])
        dim_sim_02 = torch.abs(scale_params[:, 0] - scale_params[:, 2])

        # We want at least one dimension to be significantly different
        min_diff = torch.minimum(dim_sim_01, torch.minimum(dim_sim_12, dim_sim_02))
        orthogonality_loss = -torch.mean(min_diff)
        losses["orthogonality"] = orthogonality_loss * self.orthogonality_weight

        # 2. Target Range Loss: Push scales toward a desired range
        # Compute distance from target range for each value
        below_min = torch.relu(self.target_min_scale - scale_params)
        above_max = torch.relu(scale_params - self.target_max_scale)
        range_loss = torch.mean(below_min + above_max)
        losses["target_range"] = range_loss * self.target_range_weight

        # Total loss
        losses["total"] = losses["orthogonality"] + losses["target_range"]

        return losses

class RotationDiversityLoss(nn.Module):
    """
    Loss to encourage diversity in rotation parameters of 3D Gaussians.
    Combines quaternion dispersion, entropy, and alignment losses.
    
    Args:
        dispersion_weight: Weight for dispersion loss term
        entropy_weight: Weight for entropy loss term
        alignment_weight: Weight for principal direction alignment loss term
    """
    def __init__(
        self,
        dispersion_weight: float = 0.002,
        entropy_weight: float = 0.0015,
        alignment_weight: float = 0.005
    ):
        super().__init__()
        self.dispersion_weight = dispersion_weight
        self.entropy_weight = entropy_weight
        self.alignment_weight = alignment_weight
        
    def forward(
        self, 
        rotation_params: Tensor,
        volume_gradients: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Compute rotation diversity loss components.
        
        Args:
            rotation_params: Rotation quaternions (N, 4)
            volume_gradients: Optional volume gradients for alignment loss (N, 3)
            
        Returns:
            Dictionary of named loss components and values
        """
        losses = {}
        
        # Ensure quaternions are normalized
        rotation_params = torch.nn.functional.normalize(rotation_params, dim=1)
        
        # Identity quaternion represents no rotation
        identity_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], 
            device=rotation_params.device
        ).view(1, 4)
        
        # 1. Quaternion Dispersion Loss: Encourage rotations to deviate from identity
        # Compute angular distance from identity quaternion
        # For quaternions, this is 2 * arccos(|dot product|)
        dot_products = torch.abs(torch.sum(rotation_params * identity_quat, dim=1))
        # Clip to avoid numerical issues
        dot_products = torch.clamp(dot_products, -1.0 + 1e-7, 1.0 - 1e-7)
        # Convert to angles (radians)
        angles = 2.0 * torch.acos(dot_products)
        # We want to maximize angles (i.e., minimize -angles)
        dispersion_loss = -torch.mean(angles)
        losses["dispersion"] = dispersion_loss * self.dispersion_weight
        
        # 2. Rotation Entropy Loss: Maximize entropy of rotation distribution
        # For quaternions, we use the variance across the batch
        # Higher variance = higher entropy = more diversity
        rot_variance = torch.var(rotation_params, dim=0).sum()
        entropy_loss = -rot_variance
        losses["entropy"] = entropy_loss * self.entropy_weight
        
        # 3. Principal Direction Loss: Align with volume gradients if provided
        if volume_gradients is not None and volume_gradients.shape[0] == rotation_params.shape[0]:
            # Convert quaternions to rotation matrices (N, 3, 3)
            r, i, j, k = rotation_params[:, 0], rotation_params[:, 1], rotation_params[:, 2], rotation_params[:, 3]
            
            # Quaternion to rotation matrix conversion
            rot_matrices = torch.stack([
                1 - 2 * (j**2 + k**2), 2 * (i*j - k*r), 2 * (i*k + j*r),
                2 * (i*j + k*r), 1 - 2 * (i**2 + k**2), 2 * (j*k - i*r),
                2 * (i*k - j*r), 2 * (j*k + i*r), 1 - 2 * (i**2 + j**2)
            ], dim=1).view(-1, 3, 3)
            
            # Get primary rotation axis (z-axis of rotation matrix)
            primary_axes = rot_matrices[:, :, 2]  # Extract z-axis from each rotation
            
            # Normalize volume gradients
            grad_norms = torch.norm(volume_gradients, dim=1, keepdim=True)
            normalized_grads = volume_gradients / (grad_norms + 1e-8)
            
            # Compute alignment (dot product)
            alignment = torch.sum(primary_axes * normalized_grads, dim=1)
            # We want to maximize alignment (minimize negative alignment)
            alignment_loss = -torch.mean(torch.abs(alignment))
            losses["alignment"] = alignment_loss * self.alignment_weight
        else:
            losses["alignment"] = torch.tensor(0.0, device=rotation_params.device)
            
        # 4. Total loss
        losses["total"] = losses["dispersion"] + losses["entropy"] + losses["alignment"]
        
        return losses


def compute_parameter_diversity_losses(
    model,
    volume_gradients: Optional[Tensor] = None,
    scale_diversity_weight: float = 0.01,
    rotation_diversity_weight: float = 0.01,
    target_range_weight: float = 0.005,
    dispersion_weight: float = 0.01,
    alignment_weight: float = 0.01,
) -> Dict[str, Tensor]:
    """
    Compute parameter diversity losses for a Gaussian model.

    Args:
        model: GaussianModel instance
        volume_gradients: Optional volume gradients for rotation alignment
        scale_diversity_weight: Overall weight for scale diversity loss
        rotation_diversity_weight: Overall weight for rotation diversity loss
        target_range_weight: Weight for target scale range
        dispersion_weight: Weight for quaternion dispersion
        alignment_weight: Weight for volume gradient alignment

    Returns:
        Dictionary of loss components and values
    """
    losses = {}

    # Get scaling parameters
    if model._scaling is not None and model._scaling.numel() > 0:
        scaling = model.get_scaling
        if not scaling.requires_grad:
            print(
                "WARNING: Scaling parameters lack gradients; diversity loss skipped to preserve optimizer state."
            )
            losses["scale_total"] = torch.tensor(0.0, device=model.get_xyz.device)
        else:
            scale_loss = ScaleDiversityLoss(
                orthogonality_weight=scale_diversity_weight / 2,
                target_range_weight=target_range_weight,
            )(scaling)
            scale_loss["total"] = scale_loss["total"] * scale_diversity_weight
            for name, value in scale_loss.items():
                losses[f"scale_{name}"] = value
    else:
        # No scale parameters
        losses["scale_total"] = torch.tensor(0.0, device=model.get_xyz.device)

    # Get rotation parameters
    if model._rotation is not None and model._rotation.numel() > 0:
        rotation = model.get_rotation
        if not rotation.requires_grad:
            print(
                "WARNING: Rotation parameters lack gradients; diversity loss skipped to preserve optimizer state."
            )
            losses["rotation_total"] = torch.tensor(0.0, device=model.get_xyz.device)
        else:
            rot_loss = RotationDiversityLoss(
                dispersion_weight=dispersion_weight,
                entropy_weight=rotation_diversity_weight / 2,
                alignment_weight=alignment_weight,
            )(rotation, volume_gradients)
            rot_loss["total"] = rot_loss["total"] * rotation_diversity_weight
            for name, value in rot_loss.items():
                losses[f"rotation_{name}"] = value
    else:
        # No rotation parameters
        losses["rotation_total"] = torch.tensor(0.0, device=model.get_xyz.device)

    # Compute combined loss
    losses["total"] = losses.get("scale_total", 0.0) + losses.get("rotation_total", 0.0)

    return losses
