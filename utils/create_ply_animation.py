#!/usr/bin/env python3
"""
Create an animated visualization from a sequence of PLY files.
This script helps visualize the training progression of 3D Gaussian Splatting.
"""

import os
import argparse
import glob
from pathlib import Path
import numpy as np
import pyvista as pl
from tqdm import tqdm


def load_ply_sequence(directory, pattern="gaussians_*.ply"):
    """Load all PLY files matching the pattern in the directory."""
    ply_files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not ply_files:
        raise ValueError(f"No PLY files found in {directory} matching pattern {pattern}")
    return ply_files


def create_animation(ply_files, output_path, fps=30, resolution=(1920, 1080),
                     background_color='white', point_size=5.0, camera_zoom=1.5,
                     rotation_speed=0.5):
    """
    Create an animation from a sequence of PLY files.
    
    Args:
        ply_files: List of PLY file paths
        output_path: Output video file path
        fps: Frames per second
        resolution: Video resolution (width, height)
        background_color: Background color
        point_size: Size of points
        camera_zoom: Camera zoom level
        rotation_speed: Rotation speed factor
    """
    pl.global_theme.background = background_color
    pl.global_theme.window_size = resolution
    pl.global_theme.antialiasing = True
    
    # Create plotter
    plotter = pl.Plotter(off_screen=True)
    plotter.open_gif(output_path, fps=fps)
    
    # Process each PLY file
    for ply_file in tqdm(ply_files, desc="Creating animation"):
        # Clear previous frame
        plotter.clear()
        
        # Load PLY data
        mesh = pl.read(ply_file)
        
        # Get point data
        xyz = np.array([mesh.points[:, 0], mesh.points[:, 1], mesh.points[:, 2]]).T
        
        # Extract opacity if available (varies by dataset format)
        if 'opacity' in mesh.point_data:
            opacity = mesh.point_data['opacity']
        else:
            opacity = np.ones(xyz.shape[0])
            
        # Extract scaling if available
        if all(f'scale_{i}' in mesh.point_data for i in range(3)):
            scales = np.array([
                mesh.point_data['scale_0'],
                mesh.point_data['scale_1'],
                mesh.point_data['scale_2']
            ]).T
        else:
            scales = np.ones((xyz.shape[0], 3)) * 0.01
            
        # Create point cloud
        cloud = pl.PolyData(xyz)
        
        # Apply point size based on scaling
        size = point_size * np.mean(scales, axis=1)
        
        # Add to plotter
        plotter.add_points(cloud, point_size=size, opacity=opacity)
        
        # Set camera position
        plotter.camera_position = 'xy'
        plotter.camera.zoom(camera_zoom)
        
        # Rotate for each frame
        frame_num = int(Path(ply_file).stem.split('_')[-1])
        angle = frame_num * rotation_speed % 360
        plotter.camera.azimuth = angle
        
        # Write frame
        plotter.write_frame()
    
    # Close plotter
    plotter.close()


def main():
    parser = argparse.ArgumentParser(description="Create animation from PLY sequence")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing PLY files")
    parser.add_argument("--pattern", type=str, default="gaussians_*.ply",
                        help="File pattern for PLY files")
    parser.add_argument("--output", type=str, default="animation.gif",
                        help="Output animation path (.gif or .mp4)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second")
    parser.add_argument("--point_size", type=float, default=5.0,
                        help="Base point size multiplier")
    parser.add_argument("--camera_zoom", type=float, default=1.5,
                        help="Camera zoom factor")
    parser.add_argument("--rotation_speed", type=float, default=0.5,
                        help="Camera rotation speed factor")
    parser.add_argument("--background", type=str, default="white",
                        help="Background color (white, black, etc.)")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1920, 1080],
                        help="Output resolution (width height)")
    
    args = parser.parse_args()
    
    # Load PLY files
    ply_files = load_ply_sequence(args.input_dir, args.pattern)
    print(f"Found {len(ply_files)} PLY files")
    
    # Create animation
    create_animation(
        ply_files,
        args.output,
        fps=args.fps,
        resolution=tuple(args.resolution),
        background_color=args.background,
        point_size=args.point_size,
        camera_zoom=args.camera_zoom,
        rotation_speed=args.rotation_speed
    )
    
    print(f"Animation saved to {args.output}")


if __name__ == "__main__":
    main()
