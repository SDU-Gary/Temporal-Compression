# TPE Validation Dataset Generation Guide

This guide explains how to generate and use datasets for validating the TPE (Temporal Perturbation Embedding) hypothesis.

## Overview

TPE验证数据集分为两个级别：

- **Level 1**: 玩具场景（简单可控场景）- 快速验证TPE基本原理
- **Level 2**: 单探针验证（真实复杂场景）- 关键决策点，决定TPE是否可行

## Quick Start

### 1. 环境准备

确保已安装Mitsuba 3和必要的Python包：

```bash
# 激活虚拟环境
source ../venv/bin/activate

# 验证Mitsuba可用
python -c "import mitsuba as mi; print('✓ Mitsuba ready')"
```

### 2. 生成Level 1数据集（玩具场景）

#### 场景A：平面阴影移动

```bash
python generate_level1_toy_scenes.py \
    --scene shadow_plane \
    --output output/level1_tpe/shadow_plane \
    --spp 256 \
    --num-sh-samples 64
```

**预期渲染时间**: ~2小时（25探针 × 12时刻）

#### 场景B：球体旋转光照

```bash
python generate_level1_toy_scenes.py \
    --scene rotating_sphere \
    --output output/level1_tpe/rotating_sphere \
    --spp 256 \
    --num-sh-samples 64
```

**预期渲染时间**: ~2小时

#### 场景C：Cornell Box变体

```bash
python generate_level1_toy_scenes.py \
    --scene cornell_moving_light \
    --output output/level1_tpe/cornell_moving_light \
    --spp 256 \
    --num-sh-samples 64
```

**预期渲染时间**: ~5小时（64探针 × 12时刻）

### 3. 生成Level 2数据集（单探针高质量验证）

#### Cornell Box

```bash
python generate_level2_single_probe.py \
    --scene cornell-box \
    --output output/level2_tpe/cornell-box \
    --spp 512 \
    --num-sh-samples 128
```

**预期渲染时间**: ~20分钟

#### Classroom

```bash
python generate_level2_single_probe.py \
    --scene classroom \
    --output output/level2_tpe/classroom \
    --spp 512 \
    --num-sh-samples 128
```

**预期渲染时间**: ~30分钟

#### House

```bash
python generate_level2_single_probe.py \
    --scene house \
    --output output/level2_tpe/house \
    --spp 512 \
    --num-sh-samples 128
```

**预期渲染时间**: ~40分钟

### 4. 验证TPE假设（Level 2）

#### 单个数据集验证

```bash
cd ../multi_time_compression/scripts

python validate_tpe_level2.py \
    --dataset ../../data_generation/output/level2_tpe/cornell-box
```

#### 所有Level 2数据集验证

```bash
python validate_tpe_level2.py --all
```

## 数据集结构

### Level 1 输出

```
output/level1_tpe/shadow_plane/
├── probes.npz              # [25, 3] 探针位置
├── config.json             # 数据集配置
├── moment_06/
│   ├── sh_coeffs.npz       # [25, 27] SH系数
│   └── sun_direction.txt   # 太阳方向
├── moment_07/
...
└── moment_18/
```

### Level 2 输出

```
output/level2_tpe/cornell-box/
├── probes.npz              # [1, 3] 单个探针
├── metadata.json           # 详细元数据
├── sun_trajectory.npz      # [12, 3] 完整太阳轨迹
├── moment_06/
│   └── sh_coeffs.npz       # [1, 27]
...
└── moment_18/
```

## 验证标准

TPE Level 2验证需要满足以下**全部3个**标准：

### 标准1：扰动幅度 < 1.0m
```
||ε(t)||_max < 1.0 meters
```
**物理意义**: Taylor展开有效性范围

### 标准2：时间平滑性 < 0.5m
```
avg(||ε(t+1) - ε(t)||) < 0.5 meters
```
**物理意义**: 扰动随时间平滑变化，符合物理规律

### 标准3：太阳相关性 > 0.5
```
|correlation(||ε(t)||, ||Δsun_dir(t)||)| > 0.5
```
**物理意义**: 扰动与太阳方向变化相关，具有物理解释

## 决策流程

```
生成Level 1数据集
         ↓
初步观察（可选）
         ↓
生成Level 2数据集
         ↓
   运行验证脚本
         ↓
    ┌─────────┐
    │ 3个标准 │
    │全部通过？│
    └─────────┘
       ↓    ↓
     是     否
       ↓    ↓
  实施Level 3  放弃TPE
  完整系统     寻找其他方法
```

## 常见问题

### Q: 渲染速度太慢怎么办？

**A**: 降低质量参数（仅用于测试）：

```bash
# Level 1 快速测试
--spp 128 --num-sh-samples 32

# Level 2 快速测试（不推荐）
--spp 256 --num-sh-samples 64
```

### Q: CUDA不可用怎么办？

**A**: Mitsuba会自动回退到LLVM（CPU模式），但速度会慢很多。考虑使用更低的SPP参数。

### Q: 如何测试数据集是否生成正确？

**A**: 使用数据加载器测试：

```bash
cd ../multi_time_compression/utils

python tpe_dataset_loader.py \
    ../../data_generation/output/level2_tpe/cornell-box
```

### Q: 验证脚本显示"NOT YET IMPLEMENTED"？

**A**: 当前是框架版本。完整实现需要：
1. 实现静态场F_static(x) → SH系数的映射
2. 优化扰动向量ε(t)以最小化重建误差

参见: `/home/kyrie/毕设/TPE_Temporal_Perturbation_Embedding.md` 第5.2节

## 下一步

1. **如果Level 2全部通过** → 实施完整的TPE系统（Level 3）
2. **如果Level 2部分通过** → 分析通过的场景类型，界定TPE适用范围
3. **如果Level 2全部失败** → 考虑其他时间编码方法

## 文件清单

新创建的文件：
- `scenes/toy_level1/shadow_plane.xml` - 平面阴影场景
- `scenes/toy_level1/rotating_sphere.xml` - 球体光照场景
- `generate_level1_toy_scenes.py` - Level 1生成脚本
- `generate_level2_single_probe.py` - Level 2生成脚本
- `../multi_time_compression/utils/tpe_dataset_loader.py` - 数据加载器
- `../multi_time_compression/scripts/validate_tpe_level2.py` - 验证脚本

## 存储需求

- **Level 1**: ~6 GB（3个场景）
- **Level 2**: ~50 MB（3个场景）
- **总计**: ~6.5 GB

## 参考文档

- 完整TPE理论: `/home/kyrie/毕设/TPE_Temporal_Perturbation_Embedding.md`
- 实施计划: `/home/kyrie/.claude/plans/abstract-forging-mitten.md`
