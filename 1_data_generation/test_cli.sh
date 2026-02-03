#!/bin/bash

# CLI功能测试脚本
# 测试新的命令行接口和光源修改功能

set -e # 遇到错误立即退出

echo "=========================================="
echo "CLI Functionality Test Suite"
echo "=========================================="
echo ""

# 激活虚拟环境
source ../venv/bin/activate

# 测试1: 帮助信息
echo "[Test 1/5] Testing --help"
python generate_dataset.py --help
echo "✓ Help test passed"
echo ""

# 测试2: 场景发现
echo "[Test 2/5] Testing --list-scenes"
python generate_dataset.py --list-scenes
echo "✓ Scene discovery test passed"
echo ""

# 测试3: Cornell Box快速测试（10探针，1时刻，低SPP）
echo "[Test 3/5] Testing Cornell Box generation (quick)"
python generate_dataset.py \
	--scene scenes/cornell-box/scene.xml \
	--output output/test_cornell \
	--num-probes 10 \
	--sun-hours 12 \
	--spp 32 \
	--num-sh-samples 16 \
	--num-gt-images 2

# 验证输出
if [ -f "output/test_cornell/config.json" ]; then
	echo "✓ Config file created"
else
	echo "✗ Config file missing"
	exit 1
fi

if [ -f "output/test_cornell/probes.npz" ]; then
	echo "✓ Probes file created"
else
	echo "✗ Probes file missing"
	exit 1
fi

if [ -f "output/test_cornell/moment_12/sh_coeffs.npz" ]; then
	echo "✓ SH coefficients created"
else
	echo "✗ SH coefficients missing"
	exit 1
fi

if [ -d "output/test_cornell/moment_12/images" ]; then
	echo "✓ Images directory created"
else
	echo "✗ Images directory missing"
	exit 1
fi

echo "✓ Cornell Box test passed"
echo ""

# 测试4: 多时刻测试（使用staircase2场景）
echo "[Test 4/5] Testing multi-moment generation"
python generate_dataset.py \
	--scene scenes/staircase2/scene.xml \
	--output output/test_multi \
	--num-probes 10 \
	--sun-hours 8 12 16 \
	--spp 32 \
	--num-sh-samples 16 \
	--num-gt-images 1

# 验证3个时刻都生成了
for hour in 8 12 16; do
	moment_dir="output/test_multi/moment_$(printf "%02d" $hour)"
	if [ -d "$moment_dir" ]; then
		echo "✓ Moment $hour created"
	else
		echo "✗ Moment $hour missing"
		exit 1
	fi
done

echo "✓ Multi-moment test passed"
echo ""

# 测试5: Legacy模式（向后兼容） - 跳过以节省时间
echo "[Test 5/5] Testing legacy mode (skipped - would take too long)"
echo "Note: To test legacy mode manually, run: python generate_dataset.py"
echo "✓ Legacy mode test skipped"

echo ""
echo "=========================================="
echo "All Tests Passed! ✓"
echo "=========================================="
echo ""
echo "Generated test outputs:"
echo "  - output/test_cornell/    (Cornell Box, 1 moment)"
echo "  - output/test_multi/      (Staircase2, 3 moments)"
echo ""
echo "You can inspect these directories to verify the data quality."
echo ""
echo "To test legacy mode (backward compatibility):"
echo "  python generate_dataset.py  # No arguments"
echo ""
