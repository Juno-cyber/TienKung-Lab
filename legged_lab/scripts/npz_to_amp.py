# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
#
# This file contains code derived from the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""
NPZ 运动数据转换为 AMP 格式脚本

功能:
1. 将 NPZ 格式的运动数据转换为 AMP 所需的 TXT/JSON 格式
2. 支持可视化验证
3. 自动检测数据结构并适配

使用方法:
    # 转换单个文件
    python legged_lab/scripts/npz_to_amp.py \
        --input_npz legged_lab/envs/tienkung/datasets/kuavo5/npz_0207/静止站立_LiuKe_Skeleton_retargeted.npz \
        --output_txt legged_lab/envs/tienkung/datasets/motion_visualization/stand.txt
    
    # 批量转换
    python legged_lab/scripts/npz_to_amp.py \
        --input_npz "legged_lab/envs/tienkung/datasets/kuavo5/npz_0207/*.npz" \
        --output_dir legged_lab/envs/tienkung/datasets/motion_visualization/ \
        --batch
    
    # 转换并播放验证
    python legged_lab/scripts/npz_to_amp.py \
        --input_npz legged_lab/envs/tienkung/datasets/kuavo5/npz_0207/walk.npz \
        --output_txt legged_lab/envs/tienkung/datasets/motion_visualization/walk.txt \
        --playback --task=walk
"""

import argparse
import glob
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def detect_data_structure(data):
    """自动检测 NPZ 数据结构"""
    print("\n🔍 检测数据结构...")
    
    keys = list(data.keys())
    info = {
        'has_root_pos': 'root_pos' in keys,
        'has_root_rot': 'root_rot' in keys,
        'has_dof_pos': 'dof_pos' in keys,
        'has_root_lin_vel': 'root_lin_vel' in keys,
        'has_root_ang_vel': 'root_ang_vel' in keys,
        'has_dof_vel': 'dof_vel' in keys,
        'num_frames': 0,
        'num_joints': 0,
        'fps': 30.0,
    }
    
    # 推断帧数
    for key in ['root_pos', 'dof_pos', 'root_rot']:
        if key in keys:
            arr = data[key]
            if len(arr.shape) == 2:
                info['num_frames'] = arr.shape[0]
                if info['num_joints'] == 0 and key != 'root_pos':
                    info['num_joints'] = arr.shape[1]
                elif key == 'root_pos':
                    info['root_dim'] = arr.shape[1]
                break
    
    # 推断 FPS
    if 'fps' in keys:
        info['fps'] = float(data['fps'])
    elif 'dt' in keys:
        info['fps'] = 1.0 / float(data['dt'])
    
    print(f"  帧数：{info['num_frames']}")
    print(f"  关节维度：{info['num_joints']}")
    print(f"  FPS: {info['fps']}")
    print(f"  包含根位置：{info['has_root_pos']}")
    print(f"  包含根旋转：{info['has_root_rot']}")
    print(f"  包含关节位置：{info['has_dof_pos']}")
    print(f"  包含根线速度：{info['has_root_lin_vel']}")
    print(f"  包含根角速度：{info['has_root_ang_vel']}")
    print(f"  包含关节速度：{info['has_dof_vel']}")
    
    return info


def convert_npz_to_amp_format(npz_file, output_txt, structure_info=None):
    """将 NPZ 转换为 AMP 格式"""
    print(f"\n{'='*80}")
    print(f"转换文件：{npz_file}")
    print(f"{'='*80}")
    
    data = np.load(npz_file)
    
    # 自动检测数据结构
    if structure_info is None:
        structure_info = detect_data_structure(data)
    
    num_frames = structure_info['num_frames']
    fps = structure_info['fps']
    dt = 1.0 / fps
    
    # 提取数据
    root_pos = data.get('root_pos', None)
    root_rot = data.get('root_rot', None)
    dof_pos = data.get('dof_pos', None)
    root_lin_vel = data.get('root_lin_vel', None)
    root_ang_vel = data.get('root_ang_vel', None)
    dof_vel = data.get('dof_vel', None)
    
    # 验证必需数据
    if root_pos is None or dof_pos is None:
        print("❌ 错误：缺少必需的数据 (root_pos 或 dof_pos)")
        data.close()
        return False
    
    # 如果速度数据不存在，计算微分近似
    if root_lin_vel is None and root_pos is not None:
        print("  ⚠️  未找到 root_lin_vel，通过微分计算...")
        root_lin_vel = np.diff(root_pos, axis=0) / dt
        root_lin_vel = np.vstack([root_lin_vel, root_lin_vel[-1:]])
    
    if dof_vel is None and dof_pos is not None:
        print("  ⚠️  未找到 dof_vel，通过微分计算...")
        dof_vel = np.diff(dof_pos, axis=0) / dt
        dof_vel = np.vstack([dof_vel, dof_vel[-1:]])
    
    if root_ang_vel is None:
        print("  ⚠️  未找到 root_ang_vel，使用零填充...")
        root_ang_vel = np.zeros((num_frames, 3))
    
    # 处理根旋转（四元数转欧拉角）
    if root_rot is not None:
        print("  将四元数转换为欧拉角...")
        # 假设输入是 xyzw 格式，转换为 wxyz
        if root_rot.shape[1] == 4:
            # 检测是否需要转换格式
            # 检查最后一列是否接近 1（w 分量通常较大）
            if np.abs(root_rot[:, 3]).mean() < 0.9:
                print("    检测到 xyzw 格式，转换为 wxyz...")
                root_rot_wxyz = root_rot[:, [3, 0, 1, 2]]
            else:
                root_rot_wxyz = root_rot
            
            # 转换为欧拉角并展开
            euler_angles = Rotation.from_quat(root_rot_wxyz).as_euler('XYZ', degrees=False)
            euler_angles = np.unwrap(euler_angles, axis=0)
        else:
            print(f"    ⚠️  意外的 root_rot 维度：{root_rot.shape}，使用零填充")
            euler_angles = np.zeros((num_frames, 3))
    else:
        print("  ⚠️  未找到 root_rot，使用零填充...")
        euler_angles = np.zeros((num_frames, 3))
    
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
    print(f"\n✅ 转换成功!")
    print(f"   总帧数：{num_frames}")
    print(f"   时长：{duration:.2f} 秒")
    print(f"   文件大小：{output_path.stat().st_size / 1024:.2f} KB")
    
    data.close()
    return True


def batch_convert_npz_files(input_pattern, output_dir):
    """批量转换 NPZ 文件"""
    npz_files = sorted(glob.glob(input_pattern))
    
    if not npz_files:
        print(f"❌ 未找到匹配的文件：{input_pattern}")
        return []
    
    print(f"🔍 找到 {len(npz_files)} 个 NPZ 文件")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    converted_files = []
    for npz_file in npz_files:
        input_name = Path(npz_file).stem
        output_txt = output_dir / f"{input_name}.txt"
        
        try:
            success = convert_npz_to_amp_format(npz_file, str(output_txt))
            if success:
                converted_files.append(str(output_txt))
        except Exception as e:
            print(f"\n❌ 转换失败 {npz_file}: {e}")
    
    print(f"\n{'='*80}")
    print(f"批量转换完成：{len(converted_files)}/{len(npz_files)} 个文件成功")
    print(f"{'='*80}")
    
    return converted_files


def play_converted_motion(txt_file, task_name="walk"):
    """播放转换后的运动文件进行验证"""
    import subprocess
    import sys
    
    print(f"\n🎬 播放运动数据进行验证...")
    print(f"   文件：{txt_file}")
    print(f"   任务：{task_name}")
    print(f"\n提示：按 Ctrl+C 停止播放\n")
    
    try:
        cmd = [
            sys.executable,
            "legged_lab/scripts/play_amp_animation.py",
            f"--task={task_name}",
            "--num_envs=1",
            f"--save_path={txt_file.replace('.txt', '_verified.txt')}",
            "--fps=30.0"
        ]
        
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print(f"\n✅ 播放完成")
        else:
            print(f"\n⚠️  播放过程中断")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断播放")
    except Exception as e:
        print(f"❌ 播放失败：{e}")


def main():
    parser = argparse.ArgumentParser(description="Convert NPZ motion data to AMP format")
    parser.add_argument("--input_npz", type=str, required=True, 
                       help="Path to NPZ file or glob pattern")
    parser.add_argument("--output_txt", type=str, default=None,
                       help="Output TXT file path (for single file conversion)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (for batch conversion)")
    parser.add_argument("--batch", action="store_true",
                       help="Enable batch conversion mode")
    parser.add_argument("--playback", action="store_true",
                       help="Play converted motion for verification")
    parser.add_argument("--task", type=str, default="walk",
                       help="Task name for playback (default: walk)")
    
    args = parser.parse_args()
    
    # 自动检测批量模式
    if '*' in args.input_npz or '?' in args.input_npz:
        args.batch = True
    
    if args.batch:
        # 批量转换
        if args.output_dir is None:
            print("❌ 错误：批量转换需要指定 --output_dir")
            return
        
        converted_files = batch_convert_npz_files(args.input_npz, args.output_dir)
        
        # 播放第一个文件进行验证
        if args.playback and converted_files:
            play_converted_motion(converted_files[0], args.task)
    else:
        # 单文件转换
        if args.output_txt is None:
            print("❌ 错误：单文件转换需要指定 --output_txt")
            return
        
        success = convert_npz_to_amp_format(args.input_npz, args.output_txt)
        
        # 播放验证
        if args.playback and success:
            play_converted_motion(args.output_txt, args.task)


if __name__ == "__main__":
    main()
