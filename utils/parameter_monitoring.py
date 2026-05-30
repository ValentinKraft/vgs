import torch
import matplotlib.pyplot as plt
import numpy as np
import os

class ParameterMonitor:
    """
    Class for monitoring Gaussian parameter statistics during training.
    Tracks changes in scaling, rotation, position, opacity and other parameters.
    """
    def __init__(self, output_dir):
        """
        Initialize the parameter monitor.
        
        Args:
            output_dir: Directory to save statistics and plots
        """
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "parameter_stats"), exist_ok=True)

        self.stats = {
            "iteration": [],
            "scaling_mean": [],
            "scaling_std": [],
            "scaling_min": [],
            "scaling_max": [],
            "rotation_mean_angle": [],
            "rotation_std_angle": [],
            "opacity_mean": [],
            "opacity_std": [],
            "position_delta_mean": [],
            "position_delta_std": []
        }

        # Store initial positions for tracking movement
        self.initial_positions = None
        self.last_positions = None

    def update(self, iteration, gaussian_model):
        """
        Update statistics with current Gaussian parameters.

        Args:
            iteration: Current training iteration
            gaussian_model: The Gaussian model with parameters to monitor

        Returns:
            Dictionary of parameter statistics
        """
        with torch.no_grad():
            # Store iteration
            self.stats["iteration"].append(iteration)

            # Get scaling statistics
            scaling = gaussian_model.get_scaling
            scaling_np = scaling.detach().cpu().numpy()
            self.stats["scaling_mean"].append(np.mean(scaling_np))
            self.stats["scaling_std"].append(np.std(scaling_np))
            self.stats["scaling_min"].append(np.min(scaling_np))
            self.stats["scaling_max"].append(np.max(scaling_np))

            # Get rotation statistics (convert to angles)
            rot = gaussian_model.get_rotation
            # Calculate rotation angle from quaternions
            # For simplicity, we calculate approximate "angle" from quaternion magnitudes
            rot_np = rot.detach().cpu().numpy()
            angles = 2 * np.arccos(np.abs(rot_np[:, 0]))  # Simple approximation from w component
            self.stats["rotation_mean_angle"].append(np.mean(angles))
            self.stats["rotation_std_angle"].append(np.std(angles))

            # Get opacity statistics
            opacity = gaussian_model.get_opacity
            opacity_np = opacity.detach().cpu().numpy()
            self.stats["opacity_mean"].append(np.mean(opacity_np))
            self.stats["opacity_std"].append(np.std(opacity_np))

            # Track position changes
            positions = gaussian_model.get_xyz
            if self.initial_positions is None:
                self.initial_positions = positions.detach().clone()
                self.last_positions = positions.detach().clone()
                position_delta = torch.zeros_like(positions)
            else:
                position_delta = positions - self.last_positions
                self.last_positions = positions.detach().clone()

            position_delta_np = position_delta.detach().cpu().numpy()
            self.stats["position_delta_mean"].append(np.mean(np.abs(position_delta_np)))
            self.stats["position_delta_std"].append(np.std(np.abs(position_delta_np)))

            # Save stats periodically
            if iteration % 1000 == 0 or iteration == 1:
                self.save_stats()

            # Return current stats as a dictionary for logging
            current_stats = {
                "scaling_mean": np.mean(scaling_np),
                "scaling_std": np.std(scaling_np),
                "rotation_mean_angle": np.mean(angles),
                "position_delta_mean": np.mean(np.abs(position_delta_np)),
            }

            return current_stats

    def save_stats(self):
        """
        Save current statistics to disk and generate plots.
        """
        # Save raw statistics as numpy arrays
        stats_path = os.path.join(self.output_dir, "parameter_stats", "stats.npz")
        np.savez(stats_path, **{k: np.array(v) for k, v in self.stats.items()})

        # Generate plots
        self._plot_parameter_evolution("scaling", 
                                      ["scaling_mean", "scaling_std", "scaling_min", "scaling_max"],
                                      "Scaling Parameter Evolution")

        self._plot_parameter_evolution("rotation", 
                                      ["rotation_mean_angle", "rotation_std_angle"],
                                      "Rotation Parameter Evolution")

        self._plot_parameter_evolution("opacity", 
                                      ["opacity_mean", "opacity_std"],
                                      "Opacity Parameter Evolution")

        self._plot_parameter_evolution("position_delta", 
                                      ["position_delta_mean", "position_delta_std"],
                                      "Position Change Magnitude")

    def _plot_parameter_evolution(self, name, stat_keys, title):
        """
        Generate and save a plot for parameter evolution.
        
        Args:
            name: Name of parameter (used for filename)
            stat_keys: List of keys from self.stats to plot
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        iterations = self.stats["iteration"]

        for key in stat_keys:
            if key in self.stats and len(self.stats[key]) == len(iterations):
                plt.plot(iterations, self.stats[key], label=key.replace("_", " "))

        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save figure
        plt.savefig(os.path.join(self.output_dir, "parameter_stats", f"{name}_evolution.png"), dpi=150)
        plt.close()


def add_parameter_regularization_loss(
    model,
    loss: torch.Tensor = None,
    scale_diversity_weight: float = 0.01,
    rotation_diversity_weight: float = 0.01,
    target_range_weight: float = 0.005,
    dispersion_weight: float = 0.01,
    alignment_weight: float = 0.01,
    volume_gradients: torch.Tensor = None,
):
    """
    Add regularization loss terms to encourage diversity in scaling and rotation.

    Args:
        model: GaussianModel instance
        loss: Current loss value (if None, returns only regularization loss)
        scale_diversity_weight: Weight for scale diversity term
        rotation_diversity_weight: Weight for rotation diversity term
        target_range_weight: Weight for target scale range
        dispersion_weight: Weight for quaternion dispersion
        alignment_weight: Weight for volume gradient alignment
        volume_gradients: Optional tensor of volume gradients at point positions

    Returns:
        Modified loss with regularization terms
    """
    from gaussian_splatting.losses.parameter_diversity_loss import (
        compute_parameter_diversity_losses,
    )

    # Start with either the provided loss or zero
    if loss is None:
        device = model.get_xyz.device
        modified_loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        modified_loss = loss.clone()

    # Compute diversity losses
    diversity_losses = compute_parameter_diversity_losses(
        model=model,
        volume_gradients=volume_gradients,
        scale_diversity_weight=scale_diversity_weight,
        rotation_diversity_weight=rotation_diversity_weight,
        target_range_weight=target_range_weight,
        dispersion_weight=dispersion_weight,
        alignment_weight=alignment_weight,
    )

    # Add to total loss
    regularization_loss = diversity_losses["total"]
    modified_loss = modified_loss + regularization_loss

    # Verbose logging for debugging
    print(
        f"Parameter regularization: scale={diversity_losses.get('scale_total', 0.0):.6f}, "
        f"rotation={diversity_losses.get('rotation_total', 0.0):.6f}, "
        f"total={regularization_loss.item():.6f}"
    )

    return modified_loss
