#!/bin/bash
# NPZ 数据处理工作流示例脚本

echo "=========================================="
echo "TienKung NPZ 数据处理工具"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "\n${GREEN}可用工具:${NC}"
echo "1. check_npz_data.py    - 检查 NPZ 文件结构"
echo "2. npz_to_amp.py        - 转换为 AMP 格式"
echo "3. play_amp_animation.py - 播放运动数据"

echo -e "\n${YELLOW}示例 1: 检查单个 NPZ 文件${NC}"
echo "----------------------------------------"
cat << 'EOF'
python legged_lab/scripts/check_npz_data.py \
    --input_npz legged_lab/envs/tienkung/datasets/kuavo5/npz_0207/静止站立_LiuKe_Skeleton_retargeted.npz \
    --visualize
EOF

echo -e "\n${YELLOW}示例 2: 批量检查所有 NPZ 文件${NC}"
echo "----------------------------------------"
cat << 'EOF'
python legged_lab/scripts/check_npz_data.py \
    --input_npz "legged_lab/envs/tienkung/datasets/kuavo5/npz_0207/*.npz" \
    --batch
EOF

echo -e "\n${YELLOW}示例 3: 转换单个文件为 AMP 格式${NC}"
echo "----------------------------------------"
cat << 'EOF'
python legged_lab/scripts/npz_to_amp.py \
    --input_npz legged_lab/envs/tienkung/datasets/kuavo5/npz_0207/静止站立_LiuKe_Skeleton_retargeted.npz \
    --output_txt legged_lab/envs/tienkung/datasets/motion_visualization/stand.txt
EOF

echo -e "\n${YELLOW}示例 4: 批量转换并播放验证${NC}"
echo "----------------------------------------"
cat << 'EOF'
python legged_lab/scripts/npz_to_amp.py \
    --input_npz "legged_lab/envs/tienkung/datasets/kuavo5/npz_0207/*.npz" \
    --output_dir legged_lab/envs/tienkung/datasets/motion_visualization/ \
    --batch \
    --playback \
    --task=walk
EOF

echo -e "\n${YELLOW}示例 5: 播放已转换的运动数据${NC}"
echo "----------------------------------------"
cat << 'EOF'
python legged_lab/scripts/play_amp_animation.py \
    --task=walk \
    --num_envs=1 \
    --save_path legged_lab/envs/tienkung/datasets/motion_amp_expert/walk.txt \
    --fps=30.0
EOF

echo -e "\n${GREEN}完整工作流程:${NC}"
echo "----------------------------------------"
echo "Step 1: 检查 NPZ 数据结构"
echo "  → python legged_lab/scripts/check_npz_data.py --input_npz <file.npz>"
echo ""
echo "Step 2: 转换为 visualization 格式 (用于播放验证)"
echo "  → python legged_lab/scripts/npz_to_amp.py --input_npz <file.npz> --output_txt datasets/motion_visualization/<name>.txt"
echo ""
echo "Step 3: 播放验证运动质量"
echo "  → python legged_lab/scripts/play_amp_animation.py --task=<task> --num_envs=1"
echo ""
echo "Step 4: 转换为 expert 格式 (用于 AMP 训练)"
echo "  → python legged_lab/scripts/npz_to_amp.py --input_npz <file.npz> --output_txt datasets/motion_amp_expert/<name>.txt"
echo ""
echo "Step 5: 开始训练"
echo "  → python legged_lab/scripts/train.py --task=<task> --run_name=<name>"

echo -e "\n${GREEN}提示:${NC}"
echo "- 如果 NPZ 文件包含中文字符，建议使用引号包裹文件名"
echo "- 批量处理时会自动跳过损坏的文件"
echo "- 转换后的 .png 图表会显示关节角度轨迹"
echo "- 播放时按 Ctrl+C 可以停止"

echo -e "\n=========================================="
echo "需要帮助？运行：<script_name> --help"
echo "==========================================\n"
