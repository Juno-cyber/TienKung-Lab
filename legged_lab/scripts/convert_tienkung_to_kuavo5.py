#!/usr/bin/env python3
"""
Convert TienKung AMP motion data to Kuavo5 format.

TienKung has 20 joints:
- Left leg (6): hip_roll, hip_pitch, hip_yaw, knee_pitch, ankle_pitch, ankle_roll
- Right leg (6): hip_roll, hip_pitch, hip_yaw, knee_pitch, ankle_pitch, ankle_roll  
- Left arm (4): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch
- Right arm (4): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch

Kuavo5 has 27 joints:
- Left leg (6): leg_l1-6_joint
- Right leg (6): leg_r1-6_joint
- Waist (1): waist_yaw_joint
- Left arm (7): zarm_l1-7_joint
- Right arm (7): zarm_r1-7_joint

This script maps TienKung joints to Kuavo5 joints and adds zero-padding for missing joints.
"""

import json
import numpy as np
from pathlib import Path


# Joint mapping from TienKung (20 joints) to Kuavo5 (27 joints)
# Order in AMP file: [left_leg(6), right_leg(6), waist(1), left_arm(7), right_arm(7)]
TIENKUNG_JOINT_NAMES = [
    # Left leg (6)
    "hip_roll_l_joint",
    "hip_pitch_l_joint", 
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    # Right leg (6)
    "hip_roll_r_joint",
    "hip_pitch_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    # Left arm (4)
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    # Right arm (4)
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
]

# Kuavo5 joint order in AMP format
KUAVO5_JOINT_NAMES = [
    # Left leg (6)
    "leg_l1_joint",
    "leg_l2_joint",
    "leg_l3_joint",
    "leg_l4_joint",
    "leg_l5_joint",
    "leg_l6_joint",
    # Right leg (6)
    "leg_r1_joint",
    "leg_r2_joint",
    "leg_r3_joint",
    "leg_r4_joint",
    "leg_r5_joint",
    "leg_r6_joint",
    # Waist (1)
    "waist_yaw_joint",
    # Left arm (7)
    "zarm_l1_joint",
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_l5_joint",
    "zarm_l6_joint",
    "zarm_l7_joint",
    # Right arm (7)
    "zarm_r1_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
    "zarm_r5_joint",
    "zarm_r6_joint",
    "zarm_r7_joint",
]

# Mapping indices from TienKung to Kuavo5
# This defines which TienKung joint corresponds to which Kuavo5 joint
JOINT_MAPPING = {
    # Left leg mapping (TienKung -> Kuavo5)
    0: 0,   # hip_roll_l -> leg_l1
    1: 2,   # hip_pitch_l -> leg_l2
    2: 1,   # hip_yaw_l -> leg_l3
    3: 3,   # knee_pitch_l -> leg_l4
    4: 4,   # ankle_pitch_l -> leg_l5
    5: 5,   # ankle_roll_l -> leg_l6
    
    # Right leg mapping (TienKung -> Kuavo5)
    6: 6,   # hip_roll_r -> leg_r1
    7: 8,   # hip_pitch_r -> leg_r2
    8: 7,   # hip_yaw_r -> leg_r3
    9: 9,   # knee_pitch_r -> leg_r4
    10: 10, # ankle_pitch_r -> leg_r5
    11: 11, # ankle_roll_r -> leg_r6
    
    # Waist (no mapping in TienKung, set to 0)
    # Index 12 (waist_yaw) has no source
    
    # Left arm mapping (TienKung -> Kuavo5)
    12: 13, # shoulder_pitch_l -> zarm_l1
    13: 14, # shoulder_roll_l -> zarm_l2
    14: 15, # shoulder_yaw_l -> zarm_l3
    15: 16, # elbow_pitch_l -> zarm_l4
    # zarm_l5, l6, l7 have no source (set to 0)
    
    # Right arm mapping (TienKung -> Kuavo5)
    16: 20, # shoulder_pitch_r -> zarm_r1
    17: 21, # shoulder_roll_r -> zarm_r2
    18: 22, # shoulder_yaw_r -> zarm_r3
    19: 23, # elbow_pitch_r -> zarm_r4
    # zarm_r5, r6, r7 have no source (set to 0)
}


def convert_tienkung_to_kuavo5(input_txt: str, output_txt: str):
    """
    Convert TienKung AMP motion file to Kuavo5 format.
    
    Args:
        input_txt: Path to TienKung AMP txt file
        output_txt: Path to output Kuavo5 AMP txt file
    """
    print(f"📥 Loading TienKung motion: {input_txt}")
    
    # Load TienKung motion data
    with open(input_txt, 'r') as f:
        motion_data = json.load(f)
    
    frame_duration = motion_data['FrameDuration']
    frames = motion_data['Frames']
    num_frames = len(frames)
    
    print(f"  Total frames: {num_frames}")
    print(f"  Frame duration: {frame_duration:.6f}s")
    print(f"  Input frame dimension: {len(frames[0])}")
    
    # AMP format: [root_pos(3), root_rot(3), dof_pos(N), root_lin_vel(3), root_ang_vel(3), dof_vel(N)]
    # TienKung: N=20, Total = 3+3+20+3+3+20 = 52
    # Kuavo5: N=27, Total = 3+3+27+3+3+27 = 66
    
    tienkung_dim = 52
    kuavo5_dim = 66
    
    if len(frames[0]) != tienkung_dim:
        print(f"  ⚠️  Warning: Expected TienKung dimension {tienkung_dim}, got {len(frames[0])}")
    
    # Convert each frame
    converted_frames = []
    for i, frame in enumerate(frames):
        # Extract components
        root_pos = frame[0:3]
        root_rot = frame[3:6]
        tienkung_dof_pos = frame[6:26]      # 20 joints
        root_lin_vel = frame[26:29]
        root_ang_vel = frame[29:32]
        tienkung_dof_vel = frame[32:52]     # 20 joints
        
        # Map to Kuavo5 DOF positions (27 joints)
        kuavo5_dof_pos = np.zeros(27)
        kuavo5_dof_vel = np.zeros(27)
        
        for tk_idx, kv_idx in JOINT_MAPPING.items():
            kuavo5_dof_pos[kv_idx] = tienkung_dof_pos[tk_idx]
            kuavo5_dof_vel[kv_idx] = tienkung_dof_vel[tk_idx]
        
        # Construct Kuavo5 frame
        kuavo5_frame = np.concatenate([
            root_pos,        # 3
            root_rot,        # 3
            kuavo5_dof_pos,  # 27
            root_lin_vel,    # 3
            root_ang_vel,    # 3
            kuavo5_dof_vel,  # 27
        ])
        
        converted_frames.append(kuavo5_frame.tolist())
    
    print(f"  Output frame dimension: {len(converted_frames[0])}")
    
    # Create output dictionary
    output_data = {
        "LoopMode": motion_data.get("LoopMode", "Wrap"),
        "FrameDuration": frame_duration,
        "EnableCycleOffsetPosition": motion_data.get("EnableCycleOffsetPosition", True),
        "EnableCycleOffsetRotation": motion_data.get("EnableCycleOffsetRotation", True),
        "MotionWeight": motion_data.get("MotionWeight", 0.5),
        "Frames": converted_frames
    }
    
    # Save to file
    output_path = Path(output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to: {output_txt}")
    with open(output_txt, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    file_size = output_path.stat().st_size
    duration = num_frames * frame_duration
    
    print(f"\n✅ Conversion complete!")
    print(f"   Total frames: {num_frames}")
    print(f"   Duration: {duration:.2f}s")
    print(f"   File size: {file_size / 1024:.2f} KB")
    print(f"   Joints mapped: 20 (TienKung) → 27 (Kuavo5)")
    print(f"   Unmapped Kuavo5 joints (set to 0):")
    print(f"     - waist_yaw_joint")
    print(f"     - zarm_l5_joint, zarm_l6_joint, zarm_l7_joint")
    print(f"     - zarm_r5_joint, zarm_r6_joint, zarm_r7_joint")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert TienKung AMP motion to Kuavo5 format")
    parser.add_argument("--input_txt", type=str, required=True, help="Input TienKung AMP txt file")
    parser.add_argument("--output_txt", type=str, required=True, help="Output Kuavo5 AMP txt file")
    
    args = parser.parse_args()
    
    convert_tienkung_to_kuavo5(args.input_txt, args.output_txt)
