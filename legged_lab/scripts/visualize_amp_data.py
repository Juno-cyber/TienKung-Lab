#!/usr/bin/env python3
"""
Visualize AMP motion data from txt files.

This script plots specified dimensions of AMP motion data with legends.
Supports both TienKung (52-dim) and Kuavo5 (66-dim) formats.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional


# Joint names for different robots
TIENKUNG_JOINT_NAMES = [
    # Left leg (6)
    "hip_roll_l",
    "hip_pitch_l", 
    "hip_yaw_l",
    "knee_pitch_l",
    "ankle_pitch_l",
    "ankle_roll_l",
    # Right leg (6)
    "hip_roll_r",
    "hip_pitch_r",
    "hip_yaw_r",
    "knee_pitch_r",
    "ankle_pitch_r",
    "ankle_roll_r",
    # Left arm (4)
    "shoulder_pitch_l",
    "shoulder_roll_l",
    "shoulder_yaw_l",
    "elbow_pitch_l",
    # Right arm (4)
    "shoulder_pitch_r",
    "shoulder_roll_r",
    "shoulder_yaw_r",
    "elbow_pitch_r",
]

KUAVO5_JOINT_NAMES = [
    # Left leg (6)
    "leg_l1",
    "leg_l2",
    "leg_l3",
    "leg_l4",
    "leg_l5",
    "leg_l6",
    # Right leg (6)
    "leg_r1",
    "leg_r2",
    "leg_r3",
    "leg_r4",
    "leg_r5",
    "leg_r6",
    # Waist (1)
    "waist_yaw",
    # Left arm (7)
    "zarm_l1",
    "zarm_l2",
    "zarm_l3",
    "zarm_l4",
    "zarm_l5",
    "zarm_l6",
    "zarm_l7",
    # Right arm (7)
    "zarm_r1",
    "zarm_r2",
    "zarm_r3",
    "zarm_r4",
    "zarm_r5",
    "zarm_r6",
    "zarm_r7",
]

# Data dimension labels (AMP format)
DIMENSION_LABELS = {
    0: "Root Pos X",
    1: "Root Pos Y",
    2: "Root Pos Z",
    3: "Root Rot X",
    4: "Root Rot Y",
    5: "Root Rot Z",
    # 6-32: DOF positions (depends on robot)
    # 33-35: Root linear velocity
    33: "Root Lin Vel X",
    34: "Root Lin Vel Y",
    35: "Root Lin Vel Z",
    # 36-38: Root angular velocity
    36: "Root Ang Vel X",
    37: "Root Ang Vel Y",
    38: "Root Ang Vel Z",
    # 39+: DOF velocities (depends on robot)
}


def get_dimension_name(dim_idx: int, total_dims: int, robot_type: str) -> str:
    """
    Get human-readable name for a dimension index.
    
    Args:
        dim_idx: Dimension index (0-based)
        total_dims: Total number of dimensions (52 for TienKung, 66 for Kuavo5)
        robot_type: Robot type ("tienkung" or "kuavo5")
    
    Returns:
        Human-readable dimension name
    """
    # Root state
    if dim_idx < 6:
        return DIMENSION_LABELS.get(dim_idx, f"Dim {dim_idx}")
    
    # DOF positions
    if robot_type == "tienkung":
        dof_start, dof_end = 6, 26
        vel_start = 32
    else:  # kuavo5
        dof_start, dof_end = 6, 33
        vel_start = 39
    
    if dof_start <= dim_idx < dof_end:
        joint_idx = dim_idx - dof_start
        if robot_type == "tienkung" and joint_idx < len(TIENKUNG_JOINT_NAMES):
            return f"{TIENKUNG_JOINT_NAMES[joint_idx]}_pos"
        elif robot_type == "kuavo5" and joint_idx < len(KUAVO5_JOINT_NAMES):
            return f"{KUAVO5_JOINT_NAMES[joint_idx]}_pos"
        else:
            return f"DOF_{joint_idx}_pos"
    
    # Root velocities
    if dim_idx < vel_start:
        return DIMENSION_LABELS.get(dim_idx, f"Dim {dim_idx}")
    
    # DOF velocities
    dof_vel_idx = dim_idx - vel_start
    if robot_type == "tienkung" and dof_vel_idx < len(TIENKUNG_JOINT_NAMES):
        return f"{TIENKUNG_JOINT_NAMES[dof_vel_idx]}_vel"
    elif robot_type == "kuavo5" and dof_vel_idx < len(KUAVO5_JOINT_NAMES):
        return f"{KUAVO5_JOINT_NAMES[dof_vel_idx]}_vel"
    else:
        return f"DOF_{dof_vel_idx}_vel"


def load_motion_data(txt_file: str) -> tuple:
    """
    Load AMP motion data from txt file.
    
    Args:
        txt_file: Path to AMP txt file
    
    Returns:
        Tuple of (frames_array, frame_duration, robot_type)
    """
    print(f"📥 Loading motion data: {txt_file}")
    
    with open(txt_file, 'r') as f:
        motion_data = json.load(f)
    
    frames = np.array(motion_data['Frames'])
    frame_duration = motion_data['FrameDuration']
    num_frames = frames.shape[0]
    num_dims = frames.shape[1]
    
    # Detect robot type
    if num_dims == 52:
        robot_type = "tienkung"
    elif num_dims == 66:
        robot_type = "kuavo5"
    else:
        robot_type = "unknown"
    
    print(f"  Total frames: {num_frames}")
    print(f"  Frame duration: {frame_duration:.4f}s")
    print(f"  Total duration: {num_frames * frame_duration:.2f}s")
    print(f"  Data dimensions: {num_dims}")
    print(f"  Robot type: {robot_type}")
    
    return frames, frame_duration, robot_type


def plot_dimensions(
    frames: np.ndarray,
    frame_duration: float,
    dimensions: List[int],
    robot_type: str,
    output_file: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 8),
    show_grid: bool = True,
    separate_plots: bool = False
):
    """
    Plot specified dimensions of motion data.
    
    Args:
        frames: Motion data frames (num_frames, num_dims)
        frame_duration: Duration of each frame in seconds
        dimensions: List of dimension indices to plot
        robot_type: Robot type ("tienkung" or "kuavo5")
        output_file: Optional path to save the plot
        title: Optional plot title
        figsize: Figure size (width, height)
        show_grid: Whether to show grid
        separate_plots: If True, create separate subplots for each dimension
    """
    num_frames = frames.shape[0]
    time = np.arange(num_frames) * frame_duration
    
    if separate_plots:
        # Create separate subplots for each dimension
        fig, axes = plt.subplots(len(dimensions), 1, figsize=(figsize[0], figsize[1] * len(dimensions)))
        if len(dimensions) == 1:
            axes = [axes]
        
        for i, dim_idx in enumerate(dimensions):
            ax = axes[i]
            dim_name = get_dimension_name(dim_idx, frames.shape[1], robot_type)
            
            ax.plot(time, frames[:, dim_idx], linewidth=1.5, label=dim_name)
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(f'Dimension {dim_idx}: {dim_name}', fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(show_grid, alpha=0.3)
            ax.set_xlim([time[0], time[-1]])
        
        plt.tight_layout()
    else:
        # Plot all dimensions in one figure
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(dimensions)))
        
        for i, dim_idx in enumerate(dimensions):
            dim_name = get_dimension_name(dim_idx, frames.shape[1], robot_type)
            ax.plot(time, frames[:, dim_idx], linewidth=1.5, color=colors[i], label=f'{dim_idx}: {dim_name}')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title or 'AMP Motion Data Visualization', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(show_grid, alpha=0.3)
        ax.set_xlim([time[0], time[-1]])
        
        plt.tight_layout()
    
    # Save if output path provided
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Plot saved to: {output_file}")
    
    plt.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize AMP motion data")
    parser.add_argument("--input_txt", type=str, required=True, help="Input AMP txt file")
    parser.add_argument("--dims", type=int, nargs='+', required=True, 
                       help="Dimension indices to plot (e.g., --dims 6 7 8)")
    parser.add_argument("--output", type=str, default=None, 
                       help="Output image file path (optional)")
    parser.add_argument("--title", type=str, default=None, 
                       help="Plot title")
    parser.add_argument("--separate", action="store_true", 
                       help="Create separate subplots for each dimension")
    parser.add_argument("--no-grid", action="store_true", 
                       help="Disable grid")
    parser.add_argument("--figsize", type=float, nargs=2, default=[12, 8], 
                       help="Figure size [width, height]")
    
    args = parser.parse_args()
    
    # Load data
    frames, frame_duration, robot_type = load_motion_data(args.input_txt)
    
    # Validate dimensions
    max_dim = frames.shape[1] - 1
    for dim in args.dims:
        if dim < 0 or dim > max_dim:
            print(f"❌ Error: Dimension {dim} out of range (0-{max_dim})")
            return
    
    # Print dimension info
    print(f"\n📊 Plotting dimensions:")
    for dim in args.dims:
        dim_name = get_dimension_name(dim, frames.shape[1], robot_type)
        print(f"  - Dim {dim}: {dim_name}")
    
    # Plot
    plot_dimensions(
        frames=frames,
        frame_duration=frame_duration,
        dimensions=args.dims,
        robot_type=robot_type,
        output_file=args.output,
        title=args.title,
        figsize=tuple(args.figsize),
        show_grid=not args.no_grid,
        separate_plots=args.separate
    )


if __name__ == "__main__":
    main()
