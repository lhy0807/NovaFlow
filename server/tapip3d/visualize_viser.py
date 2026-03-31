import os
import numpy as np
import cv2
import json
import struct
import zlib
import argparse
from einops import rearrange
from pathlib import Path
import viser
import viser.transforms as tf
from typing import Optional
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def process_point_cloud_data(npz_file, width=256, height=192):
    fixed_size = (width, height)
    
    data = np.load(npz_file)
    extrinsics = data["extrinsics"]
    intrinsics = data["intrinsics"]
    trajs = data["coords"]
    T, C, H, W = data["video"].shape
    
    fx = intrinsics[0, 0, 0]
    fy = intrinsics[0, 1, 1]
    fov_y = 2 * np.arctan(H / (2 * fy)) * (180 / np.pi)
    fov_x = 2 * np.arctan(W / (2 * fx)) * (180 / np.pi)
    
    rgb_video = (rearrange(data["video"], "T C H W -> T H W C") * 255).astype(np.uint8)
    rgb_video = np.stack([cv2.resize(frame, fixed_size, interpolation=cv2.INTER_AREA)
                          for frame in rgb_video])
    
    depth_video = data["depths"].astype(np.float32)
    depth_video = np.stack([cv2.resize(frame, fixed_size, interpolation=cv2.INTER_NEAREST)
                            for frame in depth_video])
    
    scale_x = fixed_size[0] / W
    scale_y = fixed_size[1] / H
    intrinsics = intrinsics.copy()
    intrinsics[:, 0, :] *= scale_x
    intrinsics[:, 1, :] *= scale_y
    
    min_depth = float(depth_video.min()) * 0.8
    max_depth = float(depth_video.max()) * 1.5
    
    first_frame_inv = np.linalg.inv(extrinsics[0])
    normalized_extrinsics = np.array([first_frame_inv @ ext for ext in extrinsics])
    
    normalized_trajs = np.zeros_like(trajs)
    for t in range(T):
        homogeneous_trajs = np.concatenate([trajs[t], np.ones((trajs.shape[1], 1))], axis=1)
        transformed_trajs = (first_frame_inv @ homogeneous_trajs.T).T
        normalized_trajs[t] = transformed_trajs[:, :3]
    
    return {
        "rgb_video": rgb_video,
        "depth_video": depth_video,
        "intrinsics": intrinsics,
        "extrinsics": normalized_extrinsics,
        "trajectories": normalized_trajs,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "fov_y": fov_y,
        "fov_x": fov_x,
        "num_frames": T
    }

def create_rgb_point_cloud(depth_frame, rgb_frame, intrinsics, extrinsics):
    h, w = depth_frame.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Convert to camera coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    X = (x - cx) * depth_frame / fx
    Y = (y - cy) * depth_frame / fy
    Z = depth_frame
    
    # Stack coordinates
    points = np.stack([X, Y, Z], axis=-1)
    
    # Reshape to Nx3
    points = points.reshape(-1, 3)
    
    # Get RGB colors
    colors = rgb_frame.reshape(-1, 3) / 255.0  # Normalize to [0, 1]
    
    # Transform points to world coordinates
    points_homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points_world = (extrinsics @ points_homogeneous.T).T[:, :3]
    
    # Filter out invalid points (where depth is 0 or very large)
    valid_mask = (depth_frame.reshape(-1) > 0.1) & (depth_frame.reshape(-1) < 10.0)
    points_world = points_world[valid_mask]
    colors = colors[valid_mask]
    
    return points_world, colors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Path to the input .result.npz file')
    parser.add_argument('--width', '-W', type=int, default=256, help='Target width')
    parser.add_argument('--height', '-H', type=int, default=192, help='Target height')
    parser.add_argument('--port', '-p', type=int, default=8080, help='Port to serve the visualization')
    
    args = parser.parse_args()
    
    # Process the data
    data = process_point_cloud_data(args.input_file, args.width, args.height)
    
    # Initialize viser server
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    
    # Add RGB point clouds
    rgb_point_clouds = {}
    for t in range(data["num_frames"]):
        points, colors = create_rgb_point_cloud(
            data["depth_video"][t],
            data["rgb_video"][t],
            data["intrinsics"][t],
            data["extrinsics"][t]
        )
        rgb_point_clouds[t] = server.scene.add_point_cloud(
            f"rgb_cloud_{t}",
            points,
            colors=colors,
            point_size=0.002
        )
        rgb_point_clouds[t].visible = (t == 0)  # Only show first frame initially
    
    # Add GUI controls
    with server.gui.add_folder("Visualization Controls"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=data["num_frames"] - 1,
            step=1,
            initial_value=0
        )
        rgb_point_size_slider = server.gui.add_slider(
            "RGB Point Size",
            min=0.001,
            max=0.1,
            step=0.001,
            initial_value=0.002
        )
        trajectory_point_size_slider = server.gui.add_slider(
            "Trajectory Point Size",
            min=0.001,
            max=0.1,
            step=0.001,
            initial_value=0.005
        )
        point_opacity_slider = server.gui.add_slider(
            "Point Opacity",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=1.0
        )
        show_trajectory_toggle = server.gui.add_checkbox(
            "Show Trajectory",
            initial_value=True
        )
        show_trajectory_trails_toggle = server.gui.add_checkbox(
            "Show Trajectory Trails",
            initial_value=True
        )
        use_motion_threshold_toggle = server.gui.add_checkbox(
            "Filter Trails by Motion",
            initial_value=False
        )
        motion_threshold_slider = server.gui.add_slider(
            "Motion Threshold",
            min=0.01,
            max=1.0,
            step=0.01,
            initial_value=0.01
        )

    with server.gui.add_folder("Playback Controls"):
        play_button = server.gui.add_button("Play")
        pause_button = server.gui.add_button("Pause")
        reset_button = server.gui.add_button("Reset")
        framerate_dropdown = server.gui.add_dropdown(
            "Playback Speed",
            options=["0.5x", "1x", "2x", "4x", "10x"],
            initial_value="1x"
        )

    # Create trajectory points
    trajectory_points = {}
    for t in range(data["num_frames"]):
        # Get current points
        current_points = data["trajectories"][t]
        
        # Calculate motion magnitude for each point
        if t > 0:
            prev_points = data["trajectories"][t-1]
            motion_magnitude = np.linalg.norm(current_points - prev_points, axis=1)
            
            # Create mask for points with significant motion if enabled
            if use_motion_threshold_toggle.value:
                motion_mask = motion_magnitude > motion_threshold_slider.value
                if not np.any(motion_mask):
                    continue
                current_points = current_points[motion_mask]
        
        # Create rainbow colors based on point indices
        num_points = len(current_points)
        indices = np.arange(num_points)
        normalized_indices = indices / (num_points - 1)  # Normalize to [0, 1]
        
        # Create rainbow colors using matplotlib's colormap
        colors = plt.cm.rainbow(normalized_indices)[:, :3]  # Get RGB values, ignore alpha
        
        # Add points to scene
        trajectory_points[t] = server.scene.add_point_cloud(
            f"/trajectory_points_{t}",
            points=current_points,
            colors=colors,
            point_size=trajectory_point_size_slider.value
        )
        trajectory_points[t].visible = False

    # Create trajectory trails
    trajectory_trails = {}
    for t in range(data["num_frames"]):
        if t == 0:
            continue
            
        # Get current and previous points
        current_points = data["trajectories"][t]
        prev_points = data["trajectories"][t-1]
        
        # Calculate motion magnitude for each point
        motion_magnitude = np.linalg.norm(current_points - prev_points, axis=1)
        
        # Create mask for points with significant motion if enabled
        if use_motion_threshold_toggle.value:
            motion_mask = motion_magnitude > motion_threshold_slider.value
            if not np.any(motion_mask):
                continue
            current_points = current_points[motion_mask]
            prev_points = prev_points[motion_mask]
            motion_magnitude = motion_magnitude[motion_mask]
        
        # Create points array with shape (N, 2, 3) for line segments
        points = np.stack([prev_points, current_points], axis=1)
        
        # Create rainbow colors based on point indices
        num_points = len(points)
        indices = np.arange(num_points)
        normalized_indices = indices / (num_points - 1)  # Normalize to [0, 1]
        
        # Create rainbow colors using matplotlib's colormap
        colors = plt.cm.rainbow(normalized_indices)[:, :3]  # Get RGB values, ignore alpha
        
        # Create colors array with shape (N, 2, 3) for line segments
        colors_array = np.stack([colors, colors], axis=1)
        
        # Add line segments to scene
        trajectory_trails[t] = server.scene.add_line_segments(
            f"/trajectory_trails_{t}",
            points=points,
            colors=colors_array,
            line_width=10,
            visible=False
        )

    # Playback state
    is_playing = False
    current_frame = 0

    def get_framerate():
        return float(framerate_dropdown.value.replace("x", ""))

    def on_play(event):
        nonlocal is_playing
        is_playing = True

    def on_pause(event):
        nonlocal is_playing
        is_playing = False

    def on_reset(event):
        nonlocal current_frame
        current_frame = 0
        frame_slider.value = 0

    def on_frame_change(event):
        frame = frame_slider.value
        for t in range(data["num_frames"]):
            rgb_point_clouds[t].visible = (t == frame)
            if t in trajectory_points:
                trajectory_points[t].visible = (t == frame) and show_trajectory_toggle.value
            if t in trajectory_trails:
                trajectory_trails[t].visible = (t == frame) and show_trajectory_trails_toggle.value

    def on_point_size_change(event):
        for t in range(data["num_frames"]):
            if t == frame_slider.value:
                rgb_point_clouds[t].point_size = rgb_point_size_slider.value
                if t in trajectory_points:
                    trajectory_points[t].point_size = trajectory_point_size_slider.value

    def on_point_opacity_change(event):
        for t in range(data["num_frames"]):
            if t == frame_slider.value:
                rgb_point_clouds[t].opacity = point_opacity_slider.value

    def on_show_trajectory_change(event):
        for t in range(data["num_frames"]):
            if t in trajectory_points:
                trajectory_points[t].visible = (t == frame_slider.value) and show_trajectory_toggle.value

    def on_show_trajectory_trails_change(event):
        for t in range(data["num_frames"]):
            if t in trajectory_trails:
                trajectory_trails[t].visible = (t == frame_slider.value) and show_trajectory_trails_toggle.value

    def on_motion_threshold_change(event):
        # Recreate trajectory trails and points with new threshold
        for t in trajectory_trails:
            trajectory_trails[t].remove()
        trajectory_trails.clear()
        
        for t in trajectory_points:
            trajectory_points[t].remove()
        trajectory_points.clear()
        
        for t in range(data["num_frames"]):
            if t == 0:
                continue
                
            current_points = data["trajectories"][t]
            prev_points = data["trajectories"][t-1]
            
            motion_magnitude = np.linalg.norm(current_points - prev_points, axis=1)
            
            if use_motion_threshold_toggle.value:
                motion_mask = motion_magnitude > motion_threshold_slider.value
                if not np.any(motion_mask):
                    continue
                current_points = current_points[motion_mask]
                prev_points = prev_points[motion_mask]
                motion_magnitude = motion_magnitude[motion_mask]
            
            # Create rainbow colors based on point indices
            num_points = len(current_points)
            indices = np.arange(num_points)
            normalized_indices = indices / (num_points - 1)  # Normalize to [0, 1]
            
            # Create rainbow colors using matplotlib's colormap
            colors = plt.cm.rainbow(normalized_indices)[:, :3]  # Get RGB values, ignore alpha
            
            # Create trajectory points
            trajectory_points[t] = server.scene.add_point_cloud(
                f"/trajectory_points_{t}",
                points=current_points,
                colors=colors,
                point_size=trajectory_point_size_slider.value
            )
            trajectory_points[t].visible = (t == frame_slider.value) and show_trajectory_toggle.value
            
            # Create trajectory trails
            points = np.stack([prev_points, current_points], axis=1)
            
            # Create colors array with shape (N, 2, 3) for line segments
            colors_array = np.stack([colors, colors], axis=1)
            
            trajectory_trails[t] = server.scene.add_line_segments(
                f"/trajectory_trails_{t}",
                points=points,
                colors=colors_array,
                line_width=10,
                visible=(t == frame_slider.value) and show_trajectory_trails_toggle.value
            )

    # Register event handlers
    frame_slider.on_update(on_frame_change)
    play_button.on_click(on_play)
    pause_button.on_click(on_pause)
    reset_button.on_click(on_reset)
    rgb_point_size_slider.on_update(on_point_size_change)
    trajectory_point_size_slider.on_update(on_point_size_change)
    point_opacity_slider.on_update(on_point_opacity_change)
    show_trajectory_toggle.on_update(on_show_trajectory_change)
    show_trajectory_trails_toggle.on_update(on_show_trajectory_trails_change)
    use_motion_threshold_toggle.on_update(on_motion_threshold_change)
    motion_threshold_slider.on_update(on_motion_threshold_change)

    # Main visualization loop
    while True:
        if is_playing:
            current_frame = (current_frame + 1) % data["num_frames"]
            frame_slider.value = current_frame
            time.sleep(1.0 / (10.0 * get_framerate()))
        else:
            time.sleep(0.1)

if __name__ == "__main__":
    main() 