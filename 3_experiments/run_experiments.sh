#!/bin/bash
# 单一变量控制实验 - 快速启动脚本

cd /home/kyrie/毕设

echo "=================================="
echo "单一变量控制实验自动化"
echo "=================================="
echo ""
echo "实验设计："
echo "  A. K30_r8 baseline (无正则化) - 基线"
echo "  B. K50_r16 capacity only - 仅提升容量"
echo "  C. K30_r8 + regularization - 仅增加正则化"
echo "  D. K50_r16 + regularization - 容量+正则化"
echo ""
echo "每个实验包含：训练 → 评估 → benchmark"
echo ""

# 激活虚拟环境
source venv/bin/activate

# 运行实验
python 3_experiments/run_controlled_experiments.py "$@"
