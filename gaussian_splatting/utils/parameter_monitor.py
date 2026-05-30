"""
Parameter monitoring utility for Gaussian Splatting training.
Tracks and visualizes parameter changes during optimization.
"""

import torch
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")  # Headless backend to avoid Tkinter dependency during training
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import time

class ParameterMonitor:
    """
    Monitors and tracks parameter changes during training.
    Provides visualization and statistics for scaling, rotation, and position changes.
    """

    def __init__(self, output_path: str, log_interval: int = 10):
        """
        Initialize parameter monitor.
        
        Args:
            output_path: Directory to save visualizations and logs
            log_interval: How often to log parameter statistics (iterations)
        """
        self.output_path = output_path
        self.log_interval = log_interval
        os.makedirs(os.path.join(output_path, "parameter_stats"), exist_ok=True)

        # History tracking for parameters
        self.iterations = []
        self.scaling_history = {
            "mean": [],
            "std": [],
            "min": [],
            "max": [],
            "x": [],
            "y": [],
            "z": []
        }
        self.rotation_history = {
            "mean": [],
            "std": [],
            "w": [],
            "x": [],
            "y": [],
            "z": []
        }
        self.position_history = {
            "delta_mean": [],
            "delta_max": []
        }

        # Track loss values
        self.loss_history = {
            "total": [],
            "volume": [],
            "regularization": []
        }

        # Track previous positions for measuring change
        self.prev_positions = None

        # For timing
        self.start_time = time.time()

    def update(
        self,
        iteration: int,
        model_xyz: torch.Tensor,
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        force: bool = False,
        loss: float = None,
        volume_loss: float = None,
        reg_loss: float = None,
    ) -> Dict[str, float]:
        """
        Update parameter statistics and track changes.

        Args:
            iteration: Current training iteration
            model_xyz: Current point positions [3, N] or [N, 3]
            scaling: Current scaling parameters [N, 3]
            rotation: Current rotation quaternions [N, 4]
            force: Force update even if not at log interval
            loss: Total loss value (optional)
            volume_loss: Volume supervision loss component (optional)
            reg_loss: Regularization loss component (optional)

        Returns:
            Dictionary of current statistics
        """
        # Skip if not a logging iteration (unless forced)
        if not force and iteration % self.log_interval != 0:
            return {}

        # Track iteration
        self.iterations.append(iteration)

        # Make sure tensors are detached and on CPU
        model_xyz = model_xyz.detach().cpu()
        scaling = scaling.detach().cpu()
        rotation = rotation.detach().cpu()

        # Handle position change tracking
        current_xyz = model_xyz.clone()
        if model_xyz.shape[0] == 3:  # [3, N] format
            current_xyz = current_xyz.permute(1, 0)  # Convert to [N, 3]

        if self.prev_positions is not None:
            # Only compare points that exist in both tensors
            min_points = min(self.prev_positions.shape[0], current_xyz.shape[0])
            position_delta = torch.norm(current_xyz[:min_points] - self.prev_positions[:min_points], dim=1)
            self.position_history["delta_mean"].append(position_delta.mean().item())
            self.position_history["delta_max"].append(position_delta.max().item())

        self.prev_positions = current_xyz.clone()

        # Track scaling statistics
        self.scaling_history["mean"].append(scaling.mean().item())
        self.scaling_history["std"].append(scaling.std().item())
        self.scaling_history["min"].append(scaling.min().item())
        self.scaling_history["max"].append(scaling.max().item())

        # Track per-axis scaling
        if scaling.shape[1] == 3:
            self.scaling_history["x"].append(scaling[:, 0].mean().item())
            self.scaling_history["y"].append(scaling[:, 1].mean().item())
            self.scaling_history["z"].append(scaling[:, 2].mean().item())

        # Track rotation statistics
        self.rotation_history["mean"].append(rotation.mean().item())
        self.rotation_history["std"].append(rotation.std().item())

        # Track quaternion components
        if rotation.shape[1] == 4:
            self.rotation_history["w"].append(rotation[:, 0].mean().item())
            self.rotation_history["x"].append(rotation[:, 1].mean().item())
            self.rotation_history["y"].append(rotation[:, 2].mean().item())
            self.rotation_history["z"].append(rotation[:, 3].mean().item())

        # Create visualization whenever we have logged data
        if self.iterations:
            self._create_visualizations(iteration)

        # Return current statistics
        current_stats = {
            "scaling_mean": self.scaling_history["mean"][-1],
            "scaling_std": self.scaling_history["std"][-1],
            "rotation_mean": self.rotation_history["mean"][-1],
            "rotation_std": self.rotation_history["std"][-1],
        }

        # Add position change stats if available
        if len(self.position_history["delta_mean"]) > 0:
            current_stats["xyz_delta"] = self.position_history["delta_mean"][-1]

        # Calculate rate of change over last few iterations
        if len(self.scaling_history["mean"]) >= 3:
            scale_change = (self.scaling_history["mean"][-1] - self.scaling_history["mean"][-3])
            current_stats["scale_change_rate"] = scale_change

            rot_change = (self.rotation_history["std"][-1] - self.rotation_history["std"][-3])
            current_stats["rot_change_rate"] = rot_change

        # Track loss values if provided
        if loss is not None:
            self.loss_history["total"].append(loss)
            current_stats["loss"] = loss

        if volume_loss is not None:
            self.loss_history["volume"].append(volume_loss)
            current_stats["volume_loss"] = volume_loss

        if reg_loss is not None:
            self.loss_history["regularization"].append(reg_loss)
            current_stats["reg_loss"] = reg_loss

        return current_stats

    def _create_visualizations(self, iteration: int):
        """
        Create visualizations of parameter statistics.
        
        Args:
            iteration: Current iteration number
        """
        # Path for saving the combined visualization
        viz_dir = os.path.join(self.output_path, "parameter_stats")
        os.makedirs(viz_dir, exist_ok=True)
        viz_path = os.path.join(viz_dir, "params_combined.png")

        # Create a 3x2 subplot for all parameters and loss
        fig, axs = plt.subplots(3, 2, figsize=(18, 14))

        # Plot scaling statistics
        axs[0, 0].plot(self.iterations, self.scaling_history["mean"], label="Mean", color='blue', linewidth=2)
        axs[0, 0].fill_between(
            self.iterations,
            np.array(self.scaling_history["mean"]) - np.array(self.scaling_history["std"]),
            np.array(self.scaling_history["mean"]) + np.array(self.scaling_history["std"]),
            alpha=0.3, color='blue'
        )
        axs[0, 0].set_title("Scaling Parameters", fontsize=14)
        axs[0, 0].set_xlabel("Iteration", fontsize=12)
        axs[0, 0].set_ylabel("Scale Value", fontsize=12)
        axs[0, 0].legend(fontsize=10)
        axs[0, 0].grid(True, linestyle='--', alpha=0.7)

        # Plot per-axis scaling
        if len(self.scaling_history["x"]) > 0:
            axs[0, 1].plot(self.iterations, self.scaling_history["x"], label="X Scale", color='red', linewidth=2)
            axs[0, 1].plot(self.iterations, self.scaling_history["y"], label="Y Scale", color='green', linewidth=2)
            axs[0, 1].plot(self.iterations, self.scaling_history["z"], label="Z Scale", color='blue', linewidth=2)
            axs[0, 1].set_title("Per-Axis Scaling", fontsize=14)
            axs[0, 1].set_xlabel("Iteration", fontsize=12)
            axs[0, 1].set_ylabel("Scale Value", fontsize=12)
            axs[0, 1].legend(fontsize=10)
            axs[0, 1].grid(True, linestyle='--', alpha=0.7)

        # Plot rotation statistics
        axs[1, 0].plot(self.iterations, self.rotation_history["mean"], label="Mean", color='purple', linewidth=2)
        axs[1, 0].plot(self.iterations, self.rotation_history["std"], label="Std Dev", color='magenta', linewidth=2)
        axs[1, 0].set_title("Rotation Parameters", fontsize=14)
        axs[1, 0].set_xlabel("Iteration", fontsize=12)
        axs[1, 0].set_ylabel("Rotation Value", fontsize=12)
        axs[1, 0].legend(fontsize=10)
        axs[1, 0].grid(True, linestyle='--', alpha=0.7)

        # Plot quaternion components
        if len(self.rotation_history["w"]) > 0:
            axs[1, 1].plot(self.iterations, self.rotation_history["w"], label="W", color='red', linewidth=2)
            axs[1, 1].plot(self.iterations, self.rotation_history["x"], label="X", color='green', linewidth=2)
            axs[1, 1].plot(self.iterations, self.rotation_history["y"], label="Y", color='blue', linewidth=2)
            axs[1, 1].plot(self.iterations, self.rotation_history["z"], label="Z", color='purple', linewidth=2)
            axs[1, 1].set_title("Quaternion Components", fontsize=14)
            axs[1, 1].set_xlabel("Iteration", fontsize=12)
            axs[1, 1].set_ylabel("Component Value", fontsize=12)
            axs[1, 1].legend(fontsize=10)
            axs[1, 1].grid(True, linestyle='--', alpha=0.7)

        # Plot position changes if available
        if len(self.position_history["delta_mean"]) >= 2:
            plot_iterations = self.iterations[1:] if len(self.iterations) > len(self.position_history["delta_mean"]) else self.iterations
            axs[2, 0].plot(
                plot_iterations, 
                self.position_history["delta_mean"], 
                label="Mean Position Change", 
                color='orange',
                linewidth=2
            )
            axs[2, 0].plot(
                plot_iterations, 
                self.position_history["delta_max"], 
                label="Max Position Change",
                color='red',
                linewidth=2
            )
            axs[2, 0].set_title("Position Changes Between Iterations", fontsize=14)
            axs[2, 0].set_xlabel("Iteration", fontsize=12)
            axs[2, 0].set_ylabel("Change Magnitude", fontsize=12)
            axs[2, 0].legend(fontsize=10)
            axs[2, 0].grid(True, linestyle='--', alpha=0.7)

        # Plot loss values if available
        if len(self.loss_history["total"]) > 0:
            # Make sure we use the same number of points for x and y axes
            iterations_for_loss = self.iterations[:len(self.loss_history["total"])]
            axs[2, 1].plot(iterations_for_loss, self.loss_history["total"], label="Total Loss", color='red', linewidth=2)

            # Plot volume loss component if available
            if len(self.loss_history["volume"]) > 0:
                iterations_for_volume = self.iterations[:len(self.loss_history["volume"])]
                axs[2, 1].plot(iterations_for_volume, self.loss_history["volume"], label="Volume Loss", color='blue', linewidth=2)

            # Plot regularization loss component if available
            if len(self.loss_history["regularization"]) > 0:
                iterations_for_reg = self.iterations[:len(self.loss_history["regularization"])]
                axs[2, 1].plot(iterations_for_reg, self.loss_history["regularization"], label="Regularization", color='green', linewidth=2)

            axs[2, 1].set_title("Loss Evolution", fontsize=14)
            axs[2, 1].set_xlabel("Iteration", fontsize=12)
            axs[2, 1].set_ylabel("Loss Value", fontsize=12)
            axs[2, 1].legend(fontsize=10)
            axs[2, 1].grid(True, linestyle='--', alpha=0.7)

            # Use log scale for loss if values vary significantly
            if max(self.loss_history["total"]) / (min(self.loss_history["total"]) + 1e-10) > 100:
                axs[2, 1].set_yscale('log')
        else:
            axs[2, 1].text(0.5, 0.5, 'No loss data available', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=axs[2, 1].transAxes, fontsize=12)

        # Add a title with timestamp
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        plt.suptitle(f"Parameter Evolution - Last Updated: {current_time} (Iteration {iteration})", fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        tmp_path = viz_path + ".png"
        try:
            # Write to a temporary file first to avoid partial writes on failure
            fig.savefig(tmp_path, dpi=100, bbox_inches="tight")
            os.replace(tmp_path, viz_path)
        except Exception as exc:
            print(f"[ParameterMonitor] Failed to save params_combined.png: {exc}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        finally:
            plt.close(fig)

    def final_report(self):
        """Generate a final report of parameter changes over training."""
        # Ensure output directory exists
        stats_dir = os.path.join(self.output_path, "parameter_stats")
        os.makedirs(stats_dir, exist_ok=True)

        report_path = os.path.join(stats_dir, "final_report.txt")

        with open(report_path, "w") as f:
            runtime = time.time() - self.start_time
            f.write(f"Training completed in {runtime:.2f} seconds\n")
            f.write(f"Total iterations logged: {len(self.iterations)}\n")
            f.write(f"Log interval: {self.log_interval}\n\n")

            if len(self.scaling_history["mean"]) > 0:
                initial_scale = self.scaling_history["mean"][0]
                final_scale = self.scaling_history["mean"][-1]
                f.write(f"Scaling parameters:\n")
                f.write(f"  Initial mean: {initial_scale:.6f}\n")
                f.write(f"  Final mean: {final_scale:.6f}\n")
                f.write(f"  Change: {final_scale - initial_scale:.6f}\n")
                f.write(f"  Relative change: {(final_scale - initial_scale)/initial_scale:.2%}\n\n")

            if len(self.rotation_history["std"]) > 0:
                initial_rot_std = self.rotation_history["std"][0]
                final_rot_std = self.rotation_history["std"][-1]
                f.write(f"Rotation parameters:\n")
                f.write(f"  Initial std dev: {initial_rot_std:.6f}\n")
                f.write(f"  Final std dev: {final_rot_std:.6f}\n")
                f.write(f"  Change: {final_rot_std - initial_rot_std:.6f}\n\n")

            if len(self.position_history["delta_mean"]) > 0:
                total_pos_change = sum(self.position_history["delta_mean"])
                f.write(f"Position changes:\n")
                f.write(f"  Total accumulated change: {total_pos_change:.6f}\n")
                f.write(f"  Average change per step: {total_pos_change/len(self.position_history['delta_mean']):.6f}\n")


def add_parameter_regularization_loss(
    model,
    loss: torch.Tensor,
    scale_diversity_weight: float = 0.01,
    rotation_diversity_weight: float = 0.01,
    scale_range_weight: float = 0.002,
    rotation_entropy_weight: float = 0.005,
    volume_gt: Optional[torch.Tensor] = None,
    principal_dir_weight: float = 0.0,
    target_range_weight: float = 0.005,
    dispersion_weight: float = 0.01,
    alignment_weight: float = 0.01,
    volume_gradients: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Add bounded regularization losses to discourage scale/rotation collapse.

    Args:
        model: GaussianModel instance
        loss: Current loss value
        scale_diversity_weight: Weight for a mild per-point anisotropy floor
        rotation_diversity_weight: Weight for avoiding quaternion collapse to identity
        scale_range_weight: Weight for pushing scales toward target range
        rotation_entropy_weight: Weight for encouraging diverse rotation distribution
        volume_gt: Optional ground truth volume for gradient-based alignment
        principal_dir_weight: Weight for principal direction alignment loss

    Returns:
        Tuple of (modified_loss, loss_metrics_dict)
    """
    loss_metrics = {}
    modified_loss = loss.clone()

    # ====================== SCALE DIVERSITY LOSSES ======================
    if hasattr(model, "_scaling") and model._scaling is not None and model._scaling.numel() > 0:
        scaling = model.get_scaling  # Get actual (non-log) scaling
        scale_total_contrib = torch.tensor(0.0, device=scaling.device)

        if scaling.shape[1] == 3:  # If we have per-axis scaling
            # Encourage only a mild minimum axis separation instead of rewarding
            # arbitrarily large per-point anisotropy. This keeps the term bounded
            # and avoids pushing splats into needle-like or axis-collapsed shapes.
            axis_gap = (
                torch.abs(scaling[:, 0] - scaling[:, 1])
                + torch.abs(scaling[:, 1] - scaling[:, 2])
                + torch.abs(scaling[:, 0] - scaling[:, 2])
            ) / 3.0
            mean_scale = scaling.mean(dim=1).detach()
            target_axis_gap = torch.clamp(mean_scale * 0.12, min=1e-4)
            orthogonality_loss = torch.relu(target_axis_gap - axis_gap).mean()
            orthogonality_loss = orthogonality_loss * scale_diversity_weight
            modified_loss = modified_loss + orthogonality_loss
            loss_metrics["scale_orthogonality_loss"] = orthogonality_loss.item()

            # 2. Target Scale Range Loss: Keep scales in reasonable range
            # Prefer scales in range [0.01, 0.2] - values outside get penalized.
            # Use mean penalties so the magnitude is independent of point count.
            min_scale = 0.01
            max_scale = 0.2
            too_small = torch.relu(min_scale - scaling)
            too_large = torch.relu(scaling - max_scale)
            range_loss = (too_small.mean() + too_large.mean()) * scale_range_weight
            modified_loss = modified_loss + range_loss
            loss_metrics["scale_range_loss"] = range_loss.item()

            scale_total_contrib = orthogonality_loss + range_loss

        loss_metrics["scale_total"] = scale_total_contrib.item()

    # ====================== ROTATION DIVERSITY LOSSES ======================
    if (
        hasattr(model, "_rotation")
        and model._rotation is not None
        and model._rotation.numel() > 0
    ):
        rot = model.get_rotation
        rotation_total_contrib = torch.tensor(0.0, device=rot.device)
        if rot.shape[1] == 4:  # If we have quaternion rotations
            # 1. Penalize only when rotations collapse too close to identity.
            identity_distance = torch.abs(rot[:, 0] - 1.0) + torch.norm(
                rot[:, 1:], dim=1
            )
            target_identity_distance = torch.full_like(identity_distance, 0.10)
            quaternion_loss = torch.relu(
                target_identity_distance - identity_distance
            ).mean() * rotation_diversity_weight
            modified_loss = modified_loss + quaternion_loss
            loss_metrics["quaternion_dispersion_loss"] = quaternion_loss.item()

            # 2. Penalize low quaternion variance rather than rewarding variance
            # unboundedly. This keeps the loss non-negative and bounded.
            quat_var = torch.var(rot, dim=0, unbiased=False).sum()
            target_quat_var = torch.tensor(0.02, device=rot.device, dtype=rot.dtype)
            entropy_loss = torch.relu(target_quat_var - quat_var)
            entropy_loss = entropy_loss * rotation_entropy_weight
            modified_loss = modified_loss + entropy_loss
            loss_metrics["rotation_entropy_loss"] = entropy_loss.item()

            # 3. Penalize near-zero spread in vector quaternion components.
            vector_std = rot[:, 1:].std(dim=0, unbiased=False).mean()
            target_vector_std = torch.tensor(0.03, device=rot.device, dtype=rot.dtype)
            dispersion_loss = torch.relu(target_vector_std - vector_std)
            dispersion_loss = dispersion_loss * dispersion_weight
            modified_loss = modified_loss + dispersion_loss
            loss_metrics["rotation_dispersion_loss"] = dispersion_loss.item()

            rotation_total_contrib = quaternion_loss + entropy_loss + dispersion_loss

            # 4. Principal Direction Loss: Align with volume gradients if available
            if volume_gt is not None and principal_dir_weight > 0:
                # Compute volume gradients (simplified - in real implementation compute proper gradients)
                if hasattr(model, "_xyz") and model._xyz is not None:
                    # This is just a placeholder - actual implementation would need volume gradients
                    # and proper conversion between quaternions and rotation matrices
                    principal_dir_loss = torch.tensor(0.0, device=rot.device)
                    modified_loss = modified_loss + principal_dir_loss
                    loss_metrics["principal_dir_loss"] = principal_dir_loss.item()
                    rotation_total_contrib = rotation_total_contrib + principal_dir_loss

        loss_metrics["rotation_total"] = rotation_total_contrib.item()

    if loss is not None:
        reg_delta = (modified_loss - loss).detach()
        loss_metrics["total"] = reg_delta.item()
    else:
        loss_metrics["total"] = modified_loss.detach().item()

    return modified_loss, loss_metrics
