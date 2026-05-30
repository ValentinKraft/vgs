#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from arguments import ModelParams

class VolumeScene:
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None):
        """
        Simplified Scene class for volume-based training without RGB supervision
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.extent = 1.0  # Default extent for volume space

        # Store reference to volume for intensity updates
        self.reference_volume = None
        if hasattr(args, "volume_path") and args.volume_path:
            from gaussian_splatting.data.volume_loader import VolumeLoader
            import torch

            # Load reference volume for intensity sampling
            volume_shape = tuple(
                args.volume_shape if hasattr(args, "volume_shape") else [64, 64, 64]
            )
            downscale_factor = getattr(args, "volume_downscale_factor", None)
            disable_overflow_guard = bool(
                getattr(args, "disable_volume_overflow_guard", False)
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if downscale_factor is None:
                loader = VolumeLoader(
                    volume_shape,
                    device,
                    enable_overflow_guard=not disable_overflow_guard,
                )
            else:
                loader = VolumeLoader(
                    target_shape=None,
                    device=device,
                    downscale_factor=downscale_factor,
                    enable_overflow_guard=not disable_overflow_guard,
                )
            self.reference_volume = loader.load_volume(args.volume_path)

            # Store in gaussians for intensity updates
            self.gaussians.reference_volume = self.reference_volume

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                               "point_cloud",
                                               "iteration_" + str(self.loaded_iter),
                                               "point_cloud.ply"))

    def save(self, iteration):
        """Save the Gaussian model to disk"""
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        os.makedirs(point_cloud_path, exist_ok=True)
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
