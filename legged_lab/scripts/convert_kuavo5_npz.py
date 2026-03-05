# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
#
# This file contains code derived from the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""
Kuavo5 NPZ 运动数据转换为 AMP 格式脚本

专门处理 Kuavo5 的 NPZ 数据格式:
- dof_positions: (frames, joints)
- dof_velocities: (frames, joints)  
- body_positions: (frames, bodies, 3)
- body_rotations: (frames, bodies, 4) - 四元数 xyzw
- root_quat: (frames, 4) - 根连杆四元数
- fps: scalar

使用方法:
    python legged_lab/scripts/convert_kuavo5_npz.py \
        --input_npz legged_lab/envs/kuavo5/datasets/npz_0207/静止站立_LiuKe_Skeleton_retargeted.npz \
        --output_txt legged_lab/envs/kuavo5/datasets/motion_visualization/stand.txt
    
    # 批量转换
    python legged_lab/scripts/convert_kuavo5_npz.py \
        --input_npz "legged_lab/envs/kuavo5/datasets/npz_0207/*.npz" \
        --output_dir legged_lab/envs/kuavo5/datasets/motion_visualization/ \
        --batch
"""

import argparse
import glob
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def convert_kuavo5_npz_to_amp(npz_file, output_txt, root_body_idx=0):
    """
    将 Kuavo5 NPZ 格式转换为 AMP 格式
    
    Args:
        npz_file: 输入 NPZ 文件路径
        output_txt: 输出 TXT 文件路径
        root_body_idx: 根连杆在 body_positions 中的索引（默认 0，通常是 torso 或 pelvis）
    """
    print(f"\n{'='*80}")
    print(f"转换文件：{npz_file}")
    print(f"{'='*80}")
    
    data = np.load(npz_file)
    
    # 验证必需的键
    required_keys = ['dof_positions', 'fps']
    for key in required_keys:
        if key not in data.keys():
            print(f"❌ 错误：缺少必需的键 '{key}'")
            data.close()
            return False
    
    # 提取数据
    dof_pos = data['dof_positions']  # (frames, joints)
    dof_vel = data.get('dof_velocities', None)
    body_pos = data.get('body_positions', None)  # (frames, bodies, 3)
    body_rot = data.get('body_rotations', None)  # (frames, bodies, 4)
    root_quat = data.get('root_quat', None)  # (frames, 4)
    body_lin_vel = data.get('body_linear_velocities', None)
    body_ang_vel = data.get('body_angular_velocities', None)
    
    fps = float(data['fps'])
    dt = 1.0 / fps
    
    num_frames, num_joints = dof_pos.shape
    print(f"\n📊 数据信息:")
    print(f"  帧数：{num_frames}")
    print(f"  关节数：{num_joints}")
    print(f"  FPS: {fps}")
    print(f"  DT: {dt:.6f}")
    
    if body_pos is not None:
        print(f"  身体连杆数：{body_pos.shape[1]}")
    
    # 提取根位置（使用第一个 body 的位置，通常是 torso/pelvis）
    if body_pos is not None:
        root_pos = body_pos[:, root_body_idx, :]  # (frames, 3)
        print(f"  使用 body[{root_body_idx}] 作为根位置")
    else:
        print("  ⚠️  未找到 body_positions，使用零填充根位置")
        root_pos = np.zeros((num_frames, 3))
    
    # Apply coordinate transformation: rotate around Z-axis by 90 degrees
    # This transforms from Kuavo5 mocap frame (-Y front) to IsaacSim frame (+X front)
    print(f"\n🔄 应用坐标变换：绕 Z 轴旋转 90°")
    rot_z_transform = Rotation.from_euler('Z', 90, degrees=True)
    
    # ====== 手动可调的额外旋转配置 ======
    # 在这里直接修改额外的 XYZ 旋转角度（单位：度）
    # 格式：(roll, pitch, yaw)
    # 例如：(0, 0, 0) 表示不额外旋转
    #      (0, 0, 90) 表示额外绕 Z 轴转 90°
    #      (0, 0, 180) 表示额外绕 Z 轴转 180°
    MANUAL_ROOT_ROT_XYZ = (90, 0, -90)  # <--- 修改这里的角度值
    # ===================================
    
    manual_rot_transform = Rotation.from_euler('XYZ', MANUAL_ROOT_ROT_XYZ, degrees=True)
    if MANUAL_ROOT_ROT_XYZ != (0, 0, 0):
        print(f"\n🎯 应用手动 XYZ 旋转变换：{MANUAL_ROOT_ROT_XYZ} 度 (roll, pitch, yaw)")
    
    # Rotate root position (X, Y coordinates)
    root_pos_original = root_pos.copy()
    root_pos_rotated = rot_z_transform.apply(root_pos)
    print(f"  根位置变换前范围：X=[{root_pos_original[:, 0].min():.3f}, {root_pos_original[:, 0].max():.3f}], "
          f"Y=[{root_pos_original[:, 1].min():.3f}, {root_pos_original[:, 1].max():.3f}]")
    print(f"  根位置变换后范围：X=[{root_pos_rotated[:, 0].min():.3f}, {root_pos_rotated[:, 0].max():.3f}], "
          f"Y=[{root_pos_rotated[:, 1].min():.3f}, {root_pos_rotated[:, 1].max():.3f}]")
    root_pos = root_pos_rotated
    
    # Combine base rotation and manual rotation
    combined_rot = manual_rot_transform * rot_z_transform
    
    # 提取根旋转 - 使用四元数乘法正确组合旋转
    if root_quat is not None:
        # NPZ 数据中 root_quat 已经是 wxyz 格式，直接使用
        print(f"  使用 root_quat 作为根旋转（wxyz 格式）")
        # Combine rotations using quaternion multiplication: R_total = R_combined * R_original
        quat_combined = (combined_rot * Rotation.from_quat(root_quat)).as_quat()
        # Convert final quaternion to euler angles
        euler_angles = Rotation.from_quat(quat_combined).as_euler('XYZ', degrees=False)
        euler_angles = np.unwrap(euler_angles, axis=0)
    elif body_rot is not None:
        # 使用指定 body 的旋转
        print(f"  使用 body[{root_body_idx}] 的旋转作为根旋转（wxyz 格式）")
        body_rot_wxyz = body_rot[:, root_body_idx, :]  # 已经是 wxyz 格式
        # Combine rotations using quaternion multiplication
        quat_combined = (combined_rot * Rotation.from_quat(body_rot_wxyz)).as_quat()
        # Convert final quaternion to euler angles
        euler_angles = Rotation.from_quat(quat_combined).as_euler('XYZ', degrees=False)
        euler_angles = np.unwrap(euler_angles, axis=0)
    else:
        print("  ⚠️  未找到旋转数据，使用零填充")
        euler_angles = np.zeros((num_frames, 3))
    
    # 计算或获取线速度
    if body_lin_vel is not None:
        root_lin_vel_original = body_lin_vel[:, root_body_idx, :]
        print(f"  使用 body_linear_velocities")
        # Rotate linear velocity using the same transformation
        root_lin_vel = rot_z_transform.apply(root_lin_vel_original)
    else:
        # 通过微分计算
        print("  ⚠️  通过微分计算根线速度")
        root_lin_vel = np.diff(root_pos, axis=0) / dt
        root_lin_vel = np.vstack([root_lin_vel, root_lin_vel[-1:]])
    
    # 计算或获取角速度
    if body_ang_vel is not None:
        root_ang_vel = body_ang_vel[:, root_body_idx, :]
        print(f"  使用 body_angular_velocities")
    else:
        # 通过欧拉角微分近似（简化处理）
        print("  ⚠️  使用零填充根角速度")
        root_ang_vel = np.zeros((num_frames, 3))
    
    # 如果关节速度不存在，通过微分计算
    if dof_vel is None:
        print("  ⚠️  通过微分计算关节速度")
        dof_vel = np.diff(dof_pos, axis=0) / dt
        dof_vel = np.vstack([dof_vel, dof_vel[-1:]])
    
    # 组合 AMP 格式数据
    # 格式：[root_pos(3), root_euler(3), dof_pos(N), root_lin_vel(3), root_ang_vel(3), dof_vel(N)]
    frames_data = []
    for i in range(num_frames):
        frame = np.concatenate([
            root_pos[i],           # 3
            euler_angles[i],       # 3
            dof_pos[i],            # N joints
            root_lin_vel[i],       # 3
            root_ang_vel[i],       # 3
            dof_vel[i],            # N joints
        ])
        frames_data.append(frame)
    
    # 创建输出目录
    output_path = Path(output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为 JSON 格式
    print(f"\n💾 保存到：{output_txt}")
    with open(output_txt, 'w') as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {dt:.6f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')
        
        for i, frame in enumerate(frames_data):
            line_start_str = '  ['
            frame_str = ', '.join([f'{v:.16e}' for v in frame])
            if i == len(frames_data) - 1:
                f.write(f'{line_start_str}{frame_str}]\n')
            else:
                f.write(f'{line_start_str}{frame_str}],\n')
        
        f.write(']\n}')
    
    duration = num_frames * dt
    file_size = output_path.stat().st_size
    print(f"\n✅ 转换成功!")
    print(f"   总帧数：{num_frames}")
    print(f"   时长：{duration:.2f} 秒")
    print(f"   文件大小：{file_size / 1024:.2f} KB")
    
    data.close()
    return True


def batch_convert(input_pattern, output_dir):
    """批量转换 NPZ 文件"""
    npz_files = sorted(glob.glob(input_pattern))
    
    if not npz_files:
        print(f"❌ 未找到匹配的文件：{input_pattern}")
        return []
    
    print(f"🔍 找到 {len(npz_files)} 个 NPZ 文件")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    converted_files = []
    failed_files = []
    
    for npz_file in npz_files:
        input_name = Path(npz_file).stem
        output_txt = output_dir / f"{input_name}.txt"
        
        try:
            success = convert_kuavo5_npz_to_amp(npz_file, str(output_txt))
            if success:
                converted_files.append(str(output_txt))
            else:
                failed_files.append(npz_file)
        except Exception as e:
            print(f"\n❌ 转换失败 {npz_file}: {e}")
            failed_files.append(npz_file)
    
    print(f"\n{'='*80}")
    print(f"批量转换完成")
    print(f"  成功：{len(converted_files)} 个文件")
    print(f"  失败：{len(failed_files)} 个文件")
    if failed_files:
        print(f"  失败文件：{', '.join([Path(f).name for f in failed_files])}")
    print(f"{'='*80}")
    
    return converted_files


def main():
    parser = argparse.ArgumentParser(description="Convert Kuavo5 NPZ motion data to AMP format")
    parser.add_argument("--input_npz", type=str, required=True, 
                       help="Path to NPZ file or glob pattern")
    parser.add_argument("--output_txt", type=str, default=None,
                       help="Output TXT file path (for single file)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (for batch mode)")
    parser.add_argument("--batch", action="store_true",
                       help="Enable batch conversion mode")
    parser.add_argument("--root_body_idx", type=int, default=0,
                       help="Index of root body in body_positions (default: 0)")
    
    args = parser.parse_args()
    
    # 自动检测批量模式
    if '*' in args.input_npz or '?' in args.input_npz:
        args.batch = True
    
    if args.batch:
        if args.output_dir is None:
            print("❌ 错误：批量转换需要指定 --output_dir")
            return
        
        batch_convert(args.input_npz, args.output_dir)
    else:
        if args.output_txt is None:
            print("❌ 错误：单文件转换需要指定 --output_txt")
            return
        
        convert_kuavo5_npz_to_amp(args.input_npz, args.output_txt, args.root_body_idx)


if __name__ == "__main__":
    main()
