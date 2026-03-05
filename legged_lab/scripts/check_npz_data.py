# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
#
# This file contains code derived from the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""
NPZ 运动数据检查脚本

功能:
1. 检查 NPZ 文件的数据结构
2. 显示关键统计信息
3. 可视化关节角度轨迹
4. 验证数据完整性

使用方法:
    python legged_lab/scripts/check_npz_data.py \
        --input_npz legged_lab/envs/tienkung/datasets/kuavo5/npz_0207/静止站立_LiuKe_Skeleton_retargeted.npz
    
    # 或者批量检查多个文件
    python legged_lab/scripts/check_npz_data.py \
        --input_npz "legged_lab/envs/tienkung/datasets/kuavo5/npz_0207/*.npz"
"""

import argparse
import glob
import numpy as np
from pathlib import Path


def check_npz_structure(npz_file):
    """检查单个 NPZ 文件的结构"""
    print(f"\n{'='*80}")
    print(f"检查文件：{npz_file}")
    print(f"{'='*80}")
    
    try:
        data = np.load(npz_file)
    except Exception as e:
        print(f"❌ 错误：无法加载文件 - {e}")
        return None
    
    keys = list(data.keys())
    print(f"\n📦 包含 {len(keys)} 个数组:")
    for i, key in enumerate(keys, 1):
        arr = data[key]
        print(f"  {i}. {key}:")
        print(f"     形状：{arr.shape}")
        print(f"     数据类型：{arr.dtype}")
        print(f"     内存大小：{arr.nbytes / 1024:.2f} KB")
        
        if len(arr.shape) == 2:
            print(f"     维度说明：{arr.shape[0]} 帧 × {arr.shape[1]} 特征")
        elif len(arr.shape) == 1:
            print(f"     维度说明：{arr.shape[0]} 个元素")
    
    # 显示关键数据的统计信息
    print(f"\n📊 数据统计:")
    # 支持多种键名约定
    possible_keys = {
        'root_pos': ['root_pos', 'body_positions'],
        'root_rot': ['root_rot', 'root_quat', 'body_rotations'],
        'dof_pos': ['dof_pos', 'dof_positions'],
        'dof_vel': ['dof_vel', 'dof_velocities']
    }
    
    for display_name, key_options in possible_keys.items():
        for key in key_options:
            if key in keys:
                arr = data[key]
                print(f"\n  {display_name} ({key}):")
                print(f"    形状：{arr.shape}")
                if arr.ndim <= 2:
                    print(f"    最小值：{arr.min():.6f}")
                    print(f"    最大值：{arr.max():.6f}")
                    print(f"    平均值：{arr.mean():.6f}")
                    print(f"    标准差：{arr.std():.6f}")
                    
                    if len(arr.shape) == 2 and arr.shape[0] > 0:
                        print(f"    第 1 帧示例：{arr[0, :min(5, arr.shape[1])]}...")
                        print(f"    最后 1 帧示例：{arr[-1, :min(5, arr.shape[1])]}...")
                break
    
    # 计算运动时长
    if 'fps' in keys:
        fps = float(data['fps'])
    else:
        fps = 30.0  # 默认 30 FPS
    
    # 检测帧数（支持多种键名）
    num_frames = 0
    for key in ['dof_positions', 'body_positions', 'root_pos', 'dof_pos']:
        if key in keys and len(data[key].shape) >= 2:
            num_frames = data[key].shape[0]
            break
    
    if num_frames > 0:
        duration = num_frames / fps
        print(f"\n⏱️  运动信息:")
        print(f"    总帧数：{num_frames}")
        print(f"    FPS: {fps}")
        print(f"    时长：{duration:.2f} 秒")
    
    data.close()
    return data


def visualize_trajectory(npz_file, plot_keys=None):
    """使用 matplotlib 可视化轨迹数据"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  未安装 matplotlib，跳过可视化")
        print("   安装命令：pip install matplotlib")
        return
    
    data = np.load(npz_file)
    
    # 确定要绘制的键
    if plot_keys is None:
        plot_keys = []
        if 'dof_pos' in data.keys():
            plot_keys.append('dof_pos')
        if 'root_pos' in data.keys():
            plot_keys.append('root_pos')
    
    if not plot_keys:
        print("⚠️  没有可可视化的数据")
        return
    
    # 创建图形
    fig, axes = plt.subplots(len(plot_keys), 1, figsize=(12, 4 * len(plot_keys)))
    if len(plot_keys) == 1:
        axes = [axes]
    
    for idx, key in enumerate(plot_keys):
        if key not in data.keys():
            continue
            
        arr = data[key]
        ax = axes[idx]
        
        if len(arr.shape) == 2:
            # 绘制所有维度
            num_dims = arr.shape[1]
            for i in range(min(num_dims, 20)):  # 最多显示 20 个维度
                ax.plot(arr[:, i], label=f'{key}[{i}]', alpha=0.7)
            
            ax.set_xlabel('Frame')
            ax.set_ylabel(key)
            ax.set_title(f'{key} Trajectory ({num_dims} dimensions)')
            ax.legend(loc='upper right', fontsize='small', ncol=4)
            ax.grid(True, alpha=0.3)
        elif len(arr.shape) == 1:
            ax.plot(arr)
            ax.set_xlabel('Index')
            ax.set_ylabel(key)
            ax.set_title(f'{key}')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_png = Path(npz_file).with_suffix('.png')
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"\n✅ 轨迹图已保存到：{output_png}")
    
    plt.close()
    data.close()


def batch_check_npz_files(pattern):
    """批量检查多个 NPZ 文件"""
    npz_files = sorted(glob.glob(pattern))
    
    if not npz_files:
        print(f"❌ 未找到匹配的文件：{pattern}")
        return
    
    print(f"🔍 找到 {len(npz_files)} 个 NPZ 文件")
    
    summary = []
    for npz_file in npz_files:
        try:
            data = np.load(npz_file)
            
            # 提取基本信息 - 支持多种键名
            info = {
                'file': Path(npz_file).name,
                'frames': 0,
                'dof_dim': 0,
                'duration': 0.0,
            }
            
            # 检测帧数（支持多种键名）
            for key in ['dof_positions', 'dof_pos', 'body_positions', 'root_pos']:
                if key in data.keys():
                    arr = data[key]
                    if len(arr.shape) >= 2:
                        info['frames'] = arr.shape[0]
                        if key in ['dof_positions', 'dof_pos']:
                            info['dof_dim'] = arr.shape[1]
                        elif key == 'body_positions':
                            info['body_dim'] = arr.shape[1]
                        break
            
            if 'fps' in data.keys():
                fps = float(data['fps'])
            else:
                fps = 30.0
            
            if info['frames'] > 0:
                info['duration'] = info['frames'] / fps
            
            summary.append(info)
            data.close()
            
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ {Path(npz_file).name}: 加载失败 - {type(e).__name__}")
            summary.append({
                'file': Path(npz_file).name,
                'frames': -1,
                'dof_dim': -1,
                'duration': -1,
                'error': error_msg
            })
    
    # 打印汇总表
    print(f"\n{'='*80}")
    print("文件汇总")
    print(f"{'='*80}")
    print(f"{'文件名':<50} {'帧数':>8} {'关节/身体':>10} {'时长 (s)':>10}")
    print(f"{'-'*80}")
    
    for info in summary:
        if info['frames'] > 0:
            dim_info = f"J:{info['dof_dim']}" if info['dof_dim'] > 0 else f"B:{info.get('body_dim', '?')}"
            print(f"{info['file']:<50} {info['frames']:>8} {dim_info:>10} {info['duration']:>10.2f}")
        else:
            error_mark = "❌" if 'error' in info else ""
            print(f"{error_mark} {info['file']:<48} {'ERROR':>8} {'-':>10} {'-':>10}")
    
    total_files = len(summary)
    valid_files = sum(1 for s in summary if s['frames'] > 0)
    total_frames = sum(s['frames'] for s in summary if s['frames'] > 0)
    total_duration = sum(s['duration'] for s in summary if s['duration'] > 0)
    
    print(f"{'-'*80}")
    print(f"总计：{valid_files}/{total_files} 个有效文件，共 {total_frames} 帧，{total_duration:.2f} 秒")


def main():
    parser = argparse.ArgumentParser(description="Check NPZ motion data structure")
    parser.add_argument("--input_npz", type=str, required=True, 
                       help="Path to NPZ file or glob pattern (e.g., 'datasets/*.npz')")
    parser.add_argument("--visualize", action="store_true", 
                       help="Visualize trajectory with matplotlib")
    parser.add_argument("--batch", action="store_true", 
                       help="Process multiple files matching the pattern")
    parser.add_argument("--plot_keys", type=str, nargs="+", default=None,
                       help="Keys to plot (e.g., dof_pos root_pos)")
    
    args = parser.parse_args()
    
    # 检查是否包含通配符
    if '*' in args.input_npz or '?' in args.input_npz:
        args.batch = True
    
    if args.batch:
        batch_check_npz_files(args.input_npz)
    else:
        # 检查单个文件
        if not Path(args.input_npz).exists():
            print(f"❌ 文件不存在：{args.input_npz}")
            return
        
        check_npz_structure(args.input_npz)
        
        if args.visualize:
            visualize_trajectory(args.input_npz, args.plot_keys)


if __name__ == "__main__":
    main()
