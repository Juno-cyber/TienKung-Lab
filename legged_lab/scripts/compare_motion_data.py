#!/usr/bin/env python3
"""
Compare TienKung walk.txt and converted Kuavo5 data to verify conversion correctness.
"""

import json
import numpy as np
from pathlib import Path


def load_motion_data(txt_file: str):
    """Load AMP motion data from txt file."""
    with open(txt_file, 'r') as f:
        motion_data = json.load(f)
    
    frames = np.array(motion_data['Frames'])
    frame_duration = motion_data['FrameDuration']
    
    return frames, frame_duration, motion_data


def analyze_data_structure(frames: np.ndarray, name: str):
    """Analyze the structure of motion data."""
    print(f"\n{'='*60}")
    print(f"📊 {name} Analysis")
    print(f"{'='*60}")
    
    num_frames, num_dims = frames.shape
    
    print(f"Total frames: {num_frames}")
    print(f"Dimensions per frame: {num_dims}")
    print(f"\nData shape: {frames.shape}")
    print(f"Data range: [{frames.min():.4f}, {frames.max():.4f}]")
    
    # Check if data is "vertical" (all values in similar ranges) or "horizontal"
    print(f"\n📈 Dimension statistics:")
    for i in range(min(num_dims, 10)):  # Show first 10 dims
        dim_data = frames[:, i]
        print(f"  Dim {i:2d}: mean={dim_data.mean():7.4f}, "
              f"std={dim_data.std():7.4f}, "
              f"range=[{dim_data.min():7.4f}, {dim_data.max():7.4f}]")
    
    if num_dims > 10:
        print(f"  ... ({num_dims - 10} more dimensions)")
    
    # Check root position (first 3 dims)
    print(f"\n📍 Root Position Analysis (dims 0-2):")
    print(f"  X: mean={frames[:, 0].mean():.4f}, range=[{frames[:, 0].min():.4f}, {frames[:, 0].max():.4f}]")
    print(f"  Y: mean={frames[:, 1].mean():.4f}, range=[{frames[:, 1].min():.4f}, {frames[:, 1].max():.4f}]")
    print(f"  Z: mean={frames[:, 2].mean():.4f}, range=[{frames[:, 2].min():.4f}, {frames[:, 2].max():.4f}]")
    
    return num_frames, num_dims


def compare_datasets(tienkung_file: str, kuavo5_file: str):
    """Compare TienKung and Kuavo5 datasets."""
    
    print("\n" + "="*60)
    print("🔍 COMPARISON ANALYSIS")
    print("="*60)
    
    # Load data
    tk_frames, tk_duration, tk_data = load_motion_data(tienkung_file)
    kv_frames, kv_duration, kv_data = load_motion_data(kuavo5_file)
    
    # Analyze structure
    analyze_data_structure(tk_frames, "TienKung walk.txt")
    analyze_data_structure(kv_frames, "Kuavo5 walk.txt")
    
    # Compare key metrics
    print(f"\n{'='*60}")
    print("📋 KEY COMPARISON METRICS")
    print(f"{'='*60}")
    
    print(f"\n⏱️  Temporal Properties:")
    print(f"  TienKung: {tk_frames.shape[0]} frames × {tk_duration:.4f}s = "
          f"{tk_frames.shape[0] * tk_duration:.2f}s")
    print(f"  Kuavo5:   {kv_frames.shape[0]} frames × {kv_duration:.4f}s = "
          f"{kv_frames.shape[0] * kv_duration:.2f}s")
    
    print(f"\n📐 Dimensional Properties:")
    print(f"  TienKung: {tk_frames.shape[1]} dimensions (52 expected)")
    print(f"  Kuavo5:   {kv_frames.shape[1]} dimensions (66 expected)")
    
    print(f"\n🎯 Root Position Comparison:")
    print(f"  TienKung X: [{tk_frames[:, 0].min():.3f}, {tk_frames[:, 0].max():.3f}]")
    print(f"  Kuavo5 X:   [{kv_frames[:, 0].min():.3f}, {kv_frames[:, 0].max():.3f}]")
    print(f"  ✓ Should be similar if conversion is correct")
    
    print(f"\n  TienKung Y: [{tk_frames[:, 1].min():.3f}, {tk_frames[:, 1].max():.3f}]")
    print(f"  Kuavo5 Y:   [{kv_frames[:, 1].min():.3f}, {kv_frames[:, 1].max():.3f}]")
    print(f"  ✓ Should be similar if conversion is correct")
    
    print(f"\n  TienKung Z: [{tk_frames[:, 2].min():.3f}, {tk_frames[:, 2].max():.3f}]")
    print(f"  Kuavo5 Z:   [{kv_frames[:, 2].min():.3f}, {kv_frames[:, 2].max():.3f}]")
    print(f"  ✓ Should be similar if conversion is correct")
    
    # Check joint positions
    print(f"\n🦾 Joint Position Analysis:")
    
    # TienKung: DOF positions at indices 6-25 (20 joints)
    # Kuavo5: DOF positions at indices 6-32 (27 joints)
    
    print(f"\n  Left Leg Joints (first 6 DOF positions):")
    min_frames = min(tk_frames.shape[0], kv_frames.shape[0])
    for i in range(6):
        tk_joint = tk_frames[:min_frames, 6 + i]
        kv_joint = kv_frames[:min_frames, 6 + i]
        print(f"    Joint {i}: TK [{tk_joint.mean():.3f}], KV [{kv_joint.mean():.3f}]")
        if np.allclose(tk_joint, kv_joint, atol=1e-3):
            print(f"      ✓ MATCH")
        else:
            print(f"      ⚠ DIFFERENT")
    
    print(f"\n  Right Leg Joints (next 6 DOF positions):")
    for i in range(6):
        tk_joint = tk_frames[:min_frames, 12 + i]
        kv_joint = kv_frames[:min_frames, 12 + i]
        print(f"    Joint {i+6}: TK [{tk_joint.mean():.3f}], KV [{kv_joint.mean():.3f}]")
        if np.allclose(tk_joint, kv_joint, atol=1e-3):
            print(f"      ✓ MATCH")
        else:
            print(f"      ⚠ DIFFERENT")
    
    # Check unmapped joints (should be zero in Kuavo5)
    print(f"\n🔒 Unmapped Kuavo5 Joints (should be zero):")
    unmapped_indices = [18, 24, 25, 26, 31, 32]  # waist, zarm_l5-7, zarm_r5-7
    for idx in unmapped_indices:
        joint_data = kv_frames[:, idx]
        print(f"    Index {idx}: mean={joint_data.mean():.6f}, max={joint_data.max():.6f}")
        if np.allclose(joint_data, 0, atol=1e-5):
            print(f"      ✓ CORRECT (zero)")
        else:
            print(f"      ⚠ NOT ZERO!")
    
    print(f"\n{'='*60}")
    print("✅ VERIFICATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare TienKung and Kuavo5 motion data")
    parser.add_argument("--tienkung_file", type=str, 
                       default="legged_lab/envs/tienkung/datasets/motion_visualization/walk.txt",
                       help="Path to TienKung walk.txt")
    parser.add_argument("--kuavo5_file", type=str,
                       default="legged_lab/envs/kuavo5/datasets/motion_amp_expert/kuavo5_walk.txt",
                       help="Path to Kuavo5 walk.txt")
    
    args = parser.parse_args()
    
    compare_datasets(args.tienkung_file, args.kuavo5_file)
