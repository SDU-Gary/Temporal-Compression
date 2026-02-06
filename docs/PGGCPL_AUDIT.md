# PG-GCPL 项目审计报告

**生成日期**: 2026-02-06  
**审计阶段**: 阶段1 - 调研与诊断  
**审计范围**: 多时刻光照压缩项目 (Physics-Guided, GMM-constrained Probe Learning)

## 1. 项目概览与目录结构

### 1.1 总体架构
```
Temporal-Compression/
├── 1_data_generation/          # 阶段1: 数据集生成 (Mitsuba 3 / Falcor)
├── 2_src/                      # 阶段2: 核心算法实现
│   ├── models/                 # 神经网络模型
│   ├── training/               # 训练基础设施
│   ├── data/                   # 数据集加载器
│   ├── utils/                  # 共享工具
│   └── tests/                  # 单元测试
├── 3_experiments/              # 阶段3: 实验与评估
│   ├── configs/                # YAML配置文件
│   ├── scripts/                # 训练/评估/分析脚本
│   └── results/                # 实验结果输出
├── 4_thesis/                   # 阶段4: 论文材料 (暂空)
├── tools/                      # 项目工具
├── docs/                       # 文档
├── archive/                    # 历史/已弃用方法
├── metadata/                   # 项目数据库
└── pipelines/                  # 工作流管道定义
```

### 1.2 工作流阶段划分
- **数据生成**: `1_data_generation/` - 光照探针数据集生成
- **核心算法**: `2_src/` - 神经网络压缩模型实现
- **实验验证**: `3_experiments/` - 训练、评估、可视化
- **文档产出**: `docs/` - 论文、报告、设计文档

## 2. 时间建模路线分析

### 2.1 当前主线 PG-GCPL (Physics-Guided, GMM-constrained Probe Learning)

**核心思想**: FiLM型线性调制统一架构 + 高斯混合空间约束

**技术特点**:
- **FiLM动态基调制**: 通过γ/β网络动态调制低秩基 U_j(Z) = U_j * (1+γ) + β
- **统一描述符系统**: 12D光照描述符 [Type, RGB(3), Pos(3), Dir(3), Meta1, Meta2]
- **光照集合编码**: LightSetEncoder支持任意数量光源，保持强度解耦线性性
- **低秩分解**: Tucker分解结构: SH ≈ U(Z) @ (coeffs @ Z)
- **高斯约束**: K个3D高斯表示空间分布
- **线性调制**: 时间权重wₖ(Z) = coeffs_k @ Z

**目标指标**:
- 压缩比: 1:17 (目标) / 16.59× (当前K30_r8)
- PSNR: >38dB (目标) / 22.38dB (当前基线)
- 解压延迟: <0.5ms

### 2.2 历史时间建模路线

#### 2.2.1 TemporalMLP (已弃用)
- **位置**: `archive/TemporalMLP/`
- **状态**: **legacy_experiment** - 历史基线，已弃用
- **架构**: 三层MLP (GaussianMixture → TemporalMLP → DecoderMLP)
- **弃用原因**:
  - 物理可解释性差 (黑盒MLP vs 物理基函数)
  - 参数效率低 (6.9× 压缩比 vs 16.59×)
  - 被PG-GCPL物理引导方法超越

#### 2.2.2 TPE方法 (已失败)
- **位置**: `archive/TPE_Method_Archived/`
- **状态**: **legacy_experiment** - 失败实验
- **问题**: 84.9%误差贡献，网格插值优于隐式神经表示

#### 2.2.3 硬编码物理基函数模型 (历史)
- **位置**: `physics_low_rank.py`、`gaussian_physics_compression.py`等
- **状态**: **legacy_experiment** - 已被FiLM统一架构替代
- **技术**: 硬编码三角函数基 [cos(θ), sin(θ), cos(φ), sin(φ), intensity, (temp-5500)/2000, exp(-cloud)]

#### 2.2.4 FiLM统一架构 (当前主线)
- **位置**: `gaussian_physics_unified.py`、`light_descriptor.py`等
- **状态**: **core** - 当前活跃开发主线
- **技术**: FiLM动态基调制 + 统一12D光照描述符 + 光照集合编码

## 3. 模块分类与初步标签

### 3.1 核心模块 (core)

| 模块 | 文件位置 | 功能描述 | 时间建模角色 |
|------|----------|----------|--------------|
| **GaussianPhysicsCompressionUnified** | `2_src/models/gaussian_physics_unified.py` | FiLM统一架构模型 | FiLM动态基调制，统一光照描述符编码 |
| **LightSetEncoder** | `2_src/models/gaussian_physics_unified.py` | 光照集合编码器 | 编码任意数量光源为单一潜变量Z |
| **LightDescriptor系统** | `2_src/utils/light_descriptor.py` | 统一光照描述符构建 | 1D/5D参数 → 12D统一描述符 |
| **GaussianPhysicsTrainer** | `2_src/training/gaussian_physics_trainer.py` | 多损失训练器 | 时间正则化、物理约束 |
| **统一训练入口** | `3_experiments/scripts/train.py` | 训练调度器 | 支持所有变体模型，包括FiLM统一架构 |
| **FiLM统一配置** | `3_experiments/configs/bistro_clean_train.yaml` | 统一架构配置 | K30_r8 + FiLM动态调制参数 |

### 3.2 历史实验模块 (legacy_experiment)

| 模块 | 文件位置 | 功能描述 | 状态说明 |
|------|----------|----------|----------|
| **硬编码物理基函数模型** | `2_src/models/physics_low_rank.py` | 三角函数物理基函数 | 历史方法，已被FiLM统一架构替代 |
| **GaussianPhysicsCompression** | `2_src/models/gaussian_physics_compression.py` | 基础高斯-物理混合模型 | 历史基线，使用硬编码三角函数基 |
| **GaussianPhysics5D** | `2_src/models/gaussian_physics_5D.py` | 5D高斯-物理模型 | 扩展物理基函数，仍使用硬编码基 |
| **GaussianPhysics1D** | `2_src/models/gaussian_physics_1D.py` | 1D强度调制模型 | 简化版硬编码基函数 |
| **TemporalMLP完整实现** | `archive/TemporalMLP/src/` | 完整的TemporalMLP三层架构 | 历史基线，保留参考价值 |
| **TPE方法存档** | `archive/TPE_Method_Archived/` | 失败的TPE方法实现 | 失败分析案例 |
| **旧训练脚本** | `3_experiments/scripts/legacy_training/` | 早期实验脚本 | 仍使用硬编码物理基函数 |

### 3.3 候选删除项 (removal_candidate)

| 文件/模块 | 位置 | 判断依据 | 建议动作 |
|-----------|------|----------|----------|
| **重复实验输出** | `3_experiments/results/`中部分早期结果 | 已被优化版本替代，无引用 | 可归档压缩 |
| **临时调试脚本** | `tools/`中未文档化的脚本 | 单次使用，无文档说明 | 可删除或添加文档 |
| **实验配置的过时副本** | `3_experiments/configs/`中已弃用配置 | 已被baseline.yaml替代 | 可删除或标记为历史 |

## 4. 时间建模入口点定位

### 4.1 主线FiLM统一架构训练入口

**主要入口**: `python 3_experiments/scripts/train.py --config 3_experiments/configs/bistro_clean_train.yaml`

**核心配置**: K30_r8 + FiLM (30个高斯，秩8，启用FiLM)
- 高斯数量: 30
- 秩: 8  
- 光照描述符维度: 12 (统一12D描述符)
- FiLM调制: 启用 (动态基调制)
- 强度维度: 3 (RGB强度解耦)
- 总参数: ~7,140 + FiLM网络参数
- 压缩比: 16.59× (343探针×12时刻)

### 4.2 FiLM动态基调制流程

```
# 在 gaussian_physics_unified.py 中
class GaussianPhysicsCompressionUnified(nn.Module):
    def forward(self, positions, light_params, light_mask=None):
        # 1) 编码光照集合为单一潜变量Z
        z = self.light_encoder(light_params, light_mask)  # [B, embed_dim]
        
        # 2) 计算FiLM调制参数
        gamma = self.gamma(z).view(-1, 1, 1, self.rank)  # [B, 1, 1, rank]
        beta = self.beta(z).view(-1, 1, 1, self.rank)   # [B, 1, 1, rank]
        
        # 3) 动态基调制
        selected_U = selected_U * (1.0 + gamma) + beta
        
        # 4) 时间权重计算 (基于光照编码Z)
        time_weights = torch.einsum('bkrl,bl->bkr', selected_coeffs, z)
        
        # 5) SH贡献计算
        sh_contributions = torch.einsum('bkdr,bkr->bkd', selected_U, time_weights)
        
        # 6) 高斯加权聚合
        sh = torch.einsum('bk,bkd->bd', gaussian_weights, sh_contributions)
        return sh
```

### 4.3 统一光照描述符构建

```
# 在 light_descriptor.py 中
def build_descriptor_5d(params_norm, param_min, param_max, light_type=0.0):
    """5D参数 → 12D统一描述符 [Type, RGB(3), Pos(3), Dir(3), Meta1, Meta2]"""
    raw = _denormalize(params_norm, param_min, param_max)
    
    rgb = _kelvin_to_rgb_torch(color_temp) * intensity.unsqueeze(-1)
    dir_vec = _zenith_azimuth_to_dir_torch(zenith, azimuth)
    
    desc = torch.cat([type_val, rgb, pos, dir_vec, meta1, meta2], dim=-1)
    return desc  # [B, 12]
```

### 4.4 历史入口点 (仅供参考)

**TemporalMLP训练**:
- `python archive/TemporalMLP/src/scripts/train.py` (已弃用)

**物理低秩基础训练**:
- `python 3_experiments/scripts/legacy_training/train_physics_low_rank_proper.py`

**高斯-物理混合训练**:
- `python 3_experiments/scripts/legacy_training/train_gaussian_physics_5D.py`

## 5. 技术债与问题清单

### 5.1 命名与结构混乱点

| 问题描述 | 涉及文件/模块 | 影响程度 | 建议修复方向 |
|----------|---------------|----------|--------------|
| **legacy_training/ 目录混杂** | `3_experiments/scripts/legacy_training/` | 中 | 部分脚本仍被引用，但目录名暗示已过时 |
| **TemporalMLP存档位置不明确** | `archive/TemporalMLP/` vs 其他引用 | 低 | 位置合理，但需在文档中明确状态 |
| **physics_low_rank.py 功能过载** | 包含多个类: PhysicsLowRank, PhysicsLowRank5D, ExtendedPhysicsBasis5D | 中 | 考虑按功能拆分或标记为legacy |
| **新旧架构混杂** | `__init__.py`中同时导出旧模型和新FiLM模型 | 中 | 明确导出策略，或标记旧模型为deprecated |
| **配置版本混杂** | `baseline.yaml` (旧MLP) vs `bistro_clean_train.yaml` (FiLM统一) | 中 | 创建配置模板，明确区分新旧架构 |
| **FiLM参数标准化** | FiLM网络的初始化策略和训练稳定性 | 低 | 需要更多消融实验验证 |

### 5.2 耦合点与依赖问题

| 耦合点 | 涉及模块 | 问题描述 | 风险等级 |
|--------|----------|----------|----------|
| **旧模型与新架构并存** | `physics_low_rank.py`和`gaussian_physics_unified.py`并存 | 可能导致混淆，不利于代码维护 | 中 |
| **光照描述符系统耦合** | `light_descriptor.py`与训练脚本紧耦合 | 描述符构建逻辑分散在多个位置 | 中 |
| **高斯参数初始化依赖sklearn** | `gaussian_physics_5D.py`中使用KMeans | 外部依赖，可能影响可复现性 | 中 |
| **训练器与模型紧耦合** | `gaussian_physics_trainer.py`特定于高斯物理模型 | 扩展新模型需要修改训练器 | 中 |

### 5.3 不确定性与待澄清问题

1. **架构演进澄清**:
   - 需要明确文档说明从硬编码物理基函数到FiLM统一架构的演进历程
   - 确认哪些旧模型仍被使用，哪些可以安全标记为legacy

2. **FiLM架构标准化**:
   - FiLM调制网络的最佳实践（层数、激活函数、初始化）
   - 光照描述符维度与FiML网络容量的关系

3. **配置系统演进**:
   - 旧`baseline.yaml` (MLP架构) 与新`bistro_clean_train.yaml` (FiLM统一架构)的关系
   - 是否需要统一的配置模板支持所有变体

3. **实验数据库整合**:
   - 历史TemporalMLP实验是否已导入数据库？
   - 如何确保所有实验可追溯？

4. **文档一致性**:
   - `CLAUDE.md`中的目录结构与实际略有不同
   - 需要更新文档以反映当前结构

### 5.4 时间建模相关待改进点

| 改进点 | 当前状态 | 目标改进 |
|--------|----------|----------|
| **FiLM架构优化** | 基础FiLM实现 | 探索更复杂的调制策略（如FiLM++, AdaIN等） |
| **光照描述符扩展** | 固定12D描述符 | 支持动态描述符维度和可扩展元数据 |
| **多尺度时间建模** | 单尺度FiLM调制 | 实现层次化FiLM调制（4×3级联时间层次） |
| **实时解压优化** | 基础推理 | 实现FiLM参数缓存、CUDA融合核优化 |
| **跨架构对比** | 新旧架构并存 | 系统性的FiLM vs 硬编码物理基函数对比实验 |

## 6. 分类依据说明

### 6.1 core 分类标准
满足以下任一条件:
- 当前FiLM统一架构正在使用 (GaussianPhysicsCompressionUnified)
- 主训练脚本`train.py`默认调用或依赖，特别是`unified_set`和`unified_5d`变体
- 与"FiLM动态基调制+统一光照描述符+高斯空间约束"核心思想直接相关
- 是实现目标指标(1:17压缩比, >38dB PSNR)的关键组件
- 支持光照集合输入和强度解耦线性性

### 6.2 legacy_experiment 分类标准
满足以下条件:
- 完整的历史实验路线，有潜在对比/参考价值
- 目前不在主pipeline中，但保留完整实现
- 有文档说明其历史地位和弃用原因
- 示例: TemporalMLP, TPE方法

### 6.3 removal_candidate 分类标准
满足以下条件:
- 显然属于临时脚本、一次性调试代码
- 不被任何主训练入口或重要模块引用
- 功能已由更优实现替代
- **注意**: 仅标记，不直接删除，需逐项确认

## 7. 后续行动建议

### 7.1 阶段1已完成工作 (FiLM架构审计)
- [x] 全局目录结构扫描与FiLM架构识别
- [x] 关键词搜索：FiLM、物理基函数、统一模型、光照描述符
- [x] 时间建模路线梳理：从硬编码物理基函数到FiLM统一架构的演进
- [x] 模块重新分类：FiLM统一架构(core) vs 硬编码基函数(legacy_experiment)
- [x] FiLM相关技术债识别：新旧架构混杂、配置不一致性
- [x] 光照描述符系统分析：统一12D描述符构建与使用

### 7.2 阶段2待执行工作 (FiLM架构清理，需授权)
1. **架构清理方案**: 设计从硬编码基函数到FiLM统一架构的迁移路径
2. **导出策略优化**: 更新`__init__.py`，明确标记旧模型为deprecated
3. **配置系统统一**: 创建支持FiLM统一架构的配置模板，淘汰旧配置
4. **文档标准化**: 更新所有相关文档，明确架构演进历程
5. **遗留代码处理**: 检查并标记所有仍使用硬编码基函数的训练脚本
6. **实验对比设计**: 设计FiLM vs 硬编码基函数的系统对比实验

### 7.3 潜在风险提示
1. **历史代码引用**: 某些legacy代码可能仍被引用，需仔细验证
2. **数据库一致性**: 重构可能影响实验数据库的记录一致性
3. **训练可复现性**: 文件移动可能影响现有训练脚本的路径假设

---

**审计结论**: 项目已完成从硬编码物理基函数到FiLM型线性调制统一架构的演进。当前主线为`GaussianPhysicsCompressionUnified`，支持动态基调制、统一光照描述符和光照集合编码。存在新旧架构混杂、配置不一致等问题。建议在阶段2设计FiLM架构清理方案，重点关注架构迁移、导出策略优化和配置系统统一。

**关键发现**:
1. ✅ FiLM统一架构已实现并成为主线 (`gaussian_physics_unified.py`)
2. ✅ 统一光照描述符系统完善 (`light_descriptor.py` + `LightSetEncoder`)
3. ✅ 训练链路支持FiLM变体 (`train.py`中的`unified_set`和`unified_5d`)
4. ⚠️ 新旧架构并存，可能导致混淆
5. ⚠️ 配置系统需要统一 (旧`baseline.yaml` vs 新`bistro_clean_train.yaml`)

**下一步**: 等待授权进入阶段2 - FiLM架构清理与标准化。