# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.

"""
Volume supervision losses for 3D Gaussian Splatting.
Supports MSE, Dice, Tversky, and KL-Divergence losses for volumetric optimization.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Literal
import logging


logger = logging.getLogger(__name__)

class DiceLoss(nn.Module):
    """
    Dice coefficient loss for volumetric segmentation.
    Optimized for binary segmentation but works with soft predictions.
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: Predicted volume (B, D, H, W) or (D, H, W)
            target: Target volume (B, D, H, W) or (D, H, W)
            
        Returns:
            Dice loss (1 - Dice coefficient)
        """
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class TverskyLoss(nn.Module):
    """
    Tversky loss for handling class imbalance in volumetric segmentation.
    Particularly useful for vessel segmentation with imbalanced foreground/background.
    
    Args:
        alpha (float): Weight of false positives
        beta (float): Weight of false negatives
        smooth (float): Smoothing factor
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: Predicted volume (B, D, H, W) or (D, H, W)
            target: Target volume (B, D, H, W) or (D, H, W)
            
        Returns:
            Tversky loss
        """
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        
        tp = (pred * target).sum(dim=1)
        fp = ((1 - target) * pred).sum(dim=1)
        fn = (target * (1 - pred)).sum(dim=1)
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()

class VolumeLoss(nn.Module):
    """
    Combined volume supervision loss supporting multiple loss types.
    
    Args:
        loss_type: Type of loss function to use
        weight: Weight for this loss term
        **kwargs: Additional arguments for specific loss types
    """
    def __init__(self, 
                 loss_type: Literal['mse', 'dice', 'tversky', 'kl'] = 'dice',
                 weight: float = 1.0,
                 **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.weight = weight
        
        # Initialize loss functions
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'dice':
            self.criterion = DiceLoss(smooth=kwargs.get('smooth', 1.0))
        elif loss_type == 'tversky':
            self.criterion = TverskyLoss(
                alpha=kwargs.get('tversky_alpha', 0.5),
                beta=kwargs.get('tversky_beta', 0.5),
                smooth=kwargs.get('smooth', 1.0)
            )
        elif loss_type == 'kl':
            self.criterion = nn.KLDivLoss(reduction='batchmean')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute the volume supervision loss.
        
        Args:
            pred: Predicted volume (B, D, H, W) or (D, H, W)
            target: Target volume (B, D, H, W) or (D, H, W)
            
        Returns:
            Weighted loss value
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "VolumeLoss input - pred requires_grad=%s, target requires_grad=%s",
                pred.requires_grad,
                target.requires_grad,
            )
        
        if self.loss_type == 'kl':
            # KL divergence expects log probabilities for pred - preserve gradients
            pred = torch.clamp(pred, 1e-7, 1.0)
            target = torch.clamp(target, 1e-7, 1.0)
            pred = torch.log(pred)
            loss = self.criterion(pred, target)
        else:
            loss = self.criterion(pred, target)
        
        weighted_loss = self.weight * loss
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "VolumeLoss output - loss=%s, requires_grad=%s",
                float(loss.detach().cpu().item()),
                weighted_loss.requires_grad,
            )
            
        return weighted_loss
