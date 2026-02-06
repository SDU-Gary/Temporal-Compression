# PG-GCPL 时间建模路线分析

**生成日期**: 2026-02-06  
**分析范围**: 多时刻光照压缩中的时间建模方法  
**分类标准**: core / legacy_experiment / removal_candidate

---

## 1. 当前主线 FiLM统一架构时间建模实现 (core)

### 1.1 FiLM动态基调制统一架构 (FiLM-Based Unified Architecture)

**核心文件**: `2_src/models/gaussian_physics_unified.py`

**主要类**:
- `GaussianPhysicsCompressionUnified`: FiLM统一架构模型
- `LightSetEncoder`: 光照集合编码器，支持任意数量光源

**时间建模原理**:
```
SH(p, L) = Σ_j G_j(p) × [U_j(Z) @ (coeffs_j @ Z)]
```
其中：
- `L`: 光照集合，每个光源由固定长度描述符表示
- `Z`: 池化的光照嵌入（求和池化，强度解耦线性性）
- `U_j(Z)`: FiLM调制的低秩基（动态基）

**FiLM动态基调制**:
```
U_j(Z) = U_j × (1 + γ(Z)) + β(Z)
γ, β = FiLM_Networks(Z)
```

**关键实现**:
```python
# FiLM调制网络 (GaussianPhysicsCompressionUnified)
self.gamma = nn.Sequential(
    nn.Linear(embed_dim, film_hidden), nn.ReLU(), nn.Linear(film_hidden, rank)
)
self.beta = nn.Sequential(
    nn.Linear(embed_dim, film_hidden), nn.ReLU(), nn.Linear(film_hidden, rank)
)

# FiLM动态基调制
def forward(self, positions, light_params, light_mask=None):
    # 1) 编码光照集合为单一潜变量Z
    z = self.light_encoder(light_params, light_mask)  # [B, embed_dim]
    
    # 2) FiLM调制
    gamma = self.gamma(z).view(-1, 1, 1, self.rank)
    beta = self.beta(z).view(-1, 1, 1, self.rank)
    selected_U = selected_U * (1.0 + gamma) + beta
    
    # 3) 时间权重计算 (基于光照编码Z)
    time_weights = torch.einsum('bkrl,bl->bkr', selected_coeffs, z)
    
    # 4) SH贡献计算
    sh_contributions = torch.einsum('bkdr,bkr->bkd', selected_U, time_weights)
    
    # 5) 高斯加权聚合
    sh = torch.einsum('bk,bkd->bd', gaussian_weights, sh_contributions)
    return sh
```

**训练损失**:
- **数据项**: Charbonnier损失 (稳健L2) - `ρ(z) = √(z² + ε²)`
- **时间正则化**: L1 temporal regularization - 鼓励时间平滑
- **线性性约束**: 增强物理一致性（光照强度解耦）
- **空间正则化**: 保持高斯分布连续性
- **FiLM初始化**: 近恒等初始化，确保训练稳定性

### 1.2 统一光照描述符系统 (Unified Light Descriptor System)

**核心文件**: 
- `2_src/utils/light_descriptor.py`: 统一描述符构建
- `2_src/models/gaussian_physics_unified.py`: 统一架构实现

**描述符设计**: 12D统一描述符 `[Type, RGB(3), Pos(3), Dir(3), Meta1, Meta2]`
- **类型标识**: 区分光源类型（0=太阳，1=点光源等）
- **RGB强度**: 3通道强度，支持开尔文温度到RGB转换
- **位置**: 3D位置坐标（太阳光源通常为0）
- **方向**: 3D方向向量（天顶/方位角转换）
- **元数据**: 云量、自定义参数等

**关键实现**:
```python
# 5D参数 → 12D描述符 (light_descriptor.py)
def build_descriptor_5d(params_norm, param_min, param_max, light_type=0.0):
    """5D参数 → 12D统一描述符"""
    raw = _denormalize(params_norm, param_min, param_max)
    
    rgb = _kelvin_to_rgb_torch(color_temp) * intensity.unsqueeze(-1)
    dir_vec = _zenith_azimuth_to_dir_torch(zenith, azimuth)
    
    desc = torch.cat([type_val, rgb, pos, dir_vec, meta1, meta2], dim=-1)
    return desc  # [B, 12]

# 1D强度 → 12D描述符
def build_descriptor_1d(intensity: torch.Tensor, light_type: float = 1.0) -> torch.Tensor:
    """1D强度 → 12D描述符（兼容格式）"""
    rgb = intensity.repeat(1, 3)
    pos = torch.zeros_like(rgb)
    dir_vec = torch.zeros_like(rgb)
    return torch.cat([type_val, rgb, pos, dir_vec, meta1, meta2], dim=-1)
```

**参数配置** (K30_r8 + FiLM最优):
- 高斯数量: 30 (K=30)
- 低秩秩数: 8 (r=8)
- 光照描述符维度: 12 (统一12D描述符)
- FiLM调制: 启用（动态基调制）
- 强度维度: 3 (RGB强度解耦)
- 总参数: ~7,140 + FiLM网络参数
- 压缩比: 16.59× (343探针×12时刻)

**训练入口**: `3_experiments/scripts/train.py --config configs/bistro_clean_train.yaml`

### 1.3 光照集合编码与强度解耦 (Light Set Encoding & Intensity Decoupling)

**核心文件**: 
- `2_src/models/gaussian_physics_unified.py`: `LightSetEncoder`类
- `1_data_generation/falcor/generate_multilight_falcor.py`: 数据生成描述符构建

**光照集合编码原理**:
```
Z = Σ_i intensity_i × MLP(features_i)  # 求和池化，保持线性性
```
- **强度解耦**: 确保光照强度线性影响最终输出
- **求和池化**: 支持任意数量光源输入
- **特征提取**: 非强度特征通过MLP编码

**LightSetEncoder关键实现**:
```python
class LightSetEncoder(nn.Module):
    def forward(self, light_desc: torch.Tensor, light_mask: torch.Tensor | None = None):
        # 强度解耦：分离强度与特征
        intensity = light_desc[..., start:end]
        features = torch.cat([light_desc[..., :start], light_desc[..., end:]], dim=-1)
        
        # 特征编码（可选MLP）
        if self.mlp is not None:
            embed = self.mlp(features)
        
        # 强度加权
        embed = embed * intensity
        
        # 掩码应用与求和池化
        mask = light_mask.unsqueeze(-1)
        embed = embed * mask
        z = embed.sum(dim=1)  # [B, embed_dim]
        return z
```

**设计优势**:
1. **线性性保持**: 强度变化线性影响输出，符合物理规律
2. **集合不变性**: 支持任意数量、任意顺序的光源输入
3. **零强度处理**: 自动处理强度为0的光源（如夜间）
4. **可扩展性**: 容易添加新的光源类型和特征

---

## 2. 历史/备用时间建模路线 (legacy_experiment)

### 2.1 硬编码物理基函数模型 (Hard-coded Physics Basis Functions)

**位置**: 
- `2_src/models/physics_low_rank.py`: 三角函数基函数实现
- `2_src/models/gaussian_physics_compression.py`: 基础高斯-物理混合模型
- `2_src/models/gaussian_physics_5D.py`: 5D扩展版本
- `2_src/models/gaussian_physics_1D.py`: 1D简化版本

**状态**: **历史方法** - 已被FiLM统一架构替代，保留作为演进参考

**技术原理**:
```
SH(t) ≈ U @ Φ(t)
Φ(t) = coeffs @ [cos(θ(t)), sin(θ(t)), 1]  # 1D三角函数基

扩展至5D参数化光源：
Φ = [cos(θ), sin(θ), cos(φ), sin(φ), intensity, (temp-5500)/2000, exp(-cloud)]
```

**关键实现**:
```python
# 7维物理基编码 (ExtendedPhysicsBasis5D)
def forward(self, light_params):
    zenith, azimuth, intensity, temp, cloud = light_params
    basis = torch.stack([
        torch.cos(zenith),
        torch.sin(zenith),
        torch.cos(azimuth),
        torch.sin(azimuth),
        intensity,
        (temp - 5500) / 2000,  # 归一化色温
        torch.exp(-cloud)      # 云量衰减
    ], dim=-1)
    return basis  # [B, 7]
```

**历史贡献**:
1. **物理可解释性**: 明确的三角函数基函数，物理意义清晰
2. **参数效率**: 相比TemporalMLP，压缩比从6.9×提升到16.59×
3. **演进桥梁**: 为FiLM统一架构提供了重要的技术基础

**被替代原因**:
1. **基函数固定**: 三角函数形式硬编码，难以适应复杂光照变化
2. **扩展性有限**: 难以支持多光源、多类型光照场景
3. **学习能力弱**: 无法从数据中学习最优的基函数形式
4. **统一性不足**: 1D和5D需要独立模型实现

### 2.2 TemporalMLP (任务书基线方法)

**位置**: `archive/TemporalMLP/`

**状态**: **已弃用** - 保留作为历史参考和性能基线

**完整架构**:
```
MultiTimeCompressionModel:
    └── GaussianMixture (空间压缩)
        └── TemporalMLP (时间编码: [F_base, sun_dir] → F_time)
            └── DecoderMLP (解码: [F_time, position] → SH系数)
```

**核心文件**:
- `src/models/temporal_mlp.py`: TemporalMLP类定义
- `src/models/decoder_mlp.py`: DecoderMLP类定义  
- `src/models/full_model.py`: 完整集成模型
- `src/scripts/train.py`: 训练脚本
- `src/scripts/knn_baseline.py`: KNN对比基线

**技术特点**:
- **输入维度**: 9 (6维基础潜在码 + 3维太阳方向)
- **网络结构**: 2层×32神经元 + ReLU激活
- **压缩比**: 约6.9× (相比原始方法)
- **实验阶段**: Phase1_TemporalMLP (2025年12月)

**弃用原因**:
1. **物理可解释性差**: MLP黑盒 vs 物理基函数的明确物理意义
2. **参数效率低**: MLP层增加参数开销，压缩比仅6.9× vs PG-GCPL的16.59×
3. **性能不足**: 被PG-GCPL的物理引导方法全面超越
4. **泛化能力弱**: 对未见时刻插值依赖网络泛化能力

### 2.3 TPE方法 (Temporal Perturbation Embedding)

**位置**: `archive/TPE_Method_Archived/`

**状态**: **失败实验** - 重要失败案例分析

**核心思想**: 在隐式神经表示中嵌入时间扰动

**失败原因分析**:
- **误差贡献**: 84.9%的误差来自TPE假设 (Cornell Box场景)
- **关键发现**: 对于平滑的时间变化，网格插值优于隐式神经表示
- **实验结论**: 物理引导的显式方法 (如三角函数基) 更适合室外光照变化

**学习价值**: 验证了物理先验在时间建模中的重要性

### 2.4 旧物理低秩训练脚本

**位置**: `3_experiments/scripts/legacy_training/`

**包含脚本**:
- `train_physics_low_rank_proper.py`: 基础物理低秩训练
- `train_gaussian_physics_5D.py`: 5D高斯-物理训练
- `train_gaussian_physics_1D.py`: 1D简化版训练
- `train_dual_gaussian_fbt.py`: 双高斯因子化基训练
- `train_pg_rglt.py`: 物理引导正则化训练

**状态**: 功能已集成到统一训练器，但保留作为参考实现

---

## 3. 候选删除项 (removal_candidate)

### 3.1 重复或过时实验输出

**位置**: `3_experiments/results/` 中的早期实验目录

**判断依据**:
- 已被优化版本 (K30_r8) 的实验结果替代
- 无脚本或文档引用这些早期结果
- 数据库已记录关键实验结果

**示例**:
- `group1_5D_parametric/01_initial_exploration/` 中的早期探索结果
- `group2_intensity_modulation/` 中非最优配置的结果

**建议动作**: 可归档压缩，保留元数据记录

### 3.2 临时调试与一次性脚本

**位置**: 分散在 `tools/` 和脚本目录中

**特征**:
- 文件名包含 `debug_`、`test_`、`temp_` 前缀
- 无文档说明，单次使用目的
- 功能已被更稳健的实现替代

**判断标准**: 检查文件最后修改时间和代码中的临时性注释

### 3.3 实验配置的过时副本

**位置**: `3_experiments/configs/`

**示例**:
- `baseline_2k.yaml` 的早期版本 (v1, v2)
- 未被任何当前训练脚本引用的配置

**建议**: 保留最新版本 (`baseline.yaml`)，其他可删除或标记为历史

### 3.4 未文档化的数据生成工具

**位置**: `tools/` 中的特定脚本

**判断**:
- 功能单一，未被工作流集成
- 无使用说明或示例
- 可由现有工具替代

---

## 4. 时间建模路线对比分析

### 4.1 方法演进时间线

```
时间线: 2025年12月 → 2026年1月 → 2026年1月-2月 → 2026年2月 (当前)
方法:   TPE (失败) → TemporalMLP (基线) → 硬编码物理基函数 → FiLM统一架构
压缩比: 不可用     → 6.9×             → 16.59×            → 16.59× (优化中)
PSNR:   不可用     → 基线             → 22.38dB           → 目标38dB
状态:   失败存档    → 弃用保留         → legacy_experiment   → 活跃开发 (主线)
```

### 4.2 技术路线对比表

| 特征维度 | TemporalMLP (历史) | 硬编码物理基函数 (历史) | FiLM统一架构 (当前) | TPE (失败) |
|----------|-------------------|-------------------------|---------------------|------------|
| **时间编码** | 学习型MLP (黑盒) | 三角函数基函数 (可解释) | FiLM动态基调制 (学习型) | 隐式扰动嵌入 |
| **物理引导** | 隐式 (通过数据学习) | 显式 (硬编码三角函数) | 混合 (FiLM网络学习) | 无物理引导 |
| **光照输入** | 单光源、单时刻 | 单光源、参数化 | 多光源集合、统一描述符 | 单时刻扰动 |
| **参数效率** | 较低 (MLP开销) | 较高 (低秩分解) | 中等 (FiLM+低秩) | 中等 |
| **压缩比** | 6.9× | 16.59× (K30_r8) | 16.59× (优化中) | 未达到可用 |
| **泛化能力** | 依赖网络能力 | 物理规律保证 | 数据驱动+物理先验 | 差 (84.9%误差) |
| **可解释性** | 低 | 高 (物理意义明确) | 中等 (可学习调制) | 低 |
| **实现复杂度** | 中等 | 中等 | 较高 (FiLM网络) | 高 |
| **当前状态** | 弃用保留 | legacy_experiment | 活跃开发 (主线) | 失败分析案例 |

### 4.3 关键技术转折点

1. **从TPE失败到物理引导的转变** (2025年12月)
   - 关键发现: 平滑时间变化需要物理先验，隐式表示不适用于室外光照
   - 决策: 放弃TPE隐式扰动，转向显式物理基函数方法

2. **从TemporalMLP到硬编码物理基函数** (2026年1月)
   - 改进点: 用低秩分解替代MLP，提高参数效率
   - 压缩比提升: 6.9× → 16.59× (K30_r8配置)
   - 物理一致性: 从学习到硬编码三角函数基函数

3. **从硬编码基函数到FiLM统一架构** (2026年2月)
   - 关键洞察: 硬编码基函数限制扩展性，无法适应复杂光照场景
   - 技术创新: 引入FiLM动态基调制，支持多光源集合输入
   - 架构统一: 统一1D/5D接口，12D光照描述符系统

4. **配置演进历程**
   - 实验优化: K30_r8 (30高斯，秩8) 为最优基础配置
   - 架构演进: `baseline.yaml` (MLP) → `bistro_clean_train.yaml` (FiLM统一)
   - 性能目标: 1:17压缩比，>38dB PSNR，<0.5ms解压

---

## 5. 未来时间建模扩展方向

### 5.1 短期改进 (基于当前架构)

| 方向 | 当前状态 | 目标改进 |
|------|----------|----------|
| **多尺度时间层次** | 单尺度FiLM调制 | 实现层次化FiLM调制 (4×3级联时间体积) |
| **FiLM架构优化** | 基础FiLM实现 | 探索更复杂的调制策略 (FiLM++, AdaIN等) |
| **光照描述符扩展** | 固定12D描述符 | 支持动态描述符维度和可扩展元数据 |
| **实时解压加速** | 基础推理 | 实现FiLM参数缓存、CUDA融合内核优化 |

### 5.2 中期探索 (新研究方向)

| 方向 | 技术思路 | 潜在价值 |
|------|----------|----------|
| **自适应时间基** | 根据变化率动态选择基函数 | 更好的时间分辨率分配 |
| **非线性物理耦合** | 在物理基基础上增加非线性变换 | 捕捉复杂光照相互作用 |
| **多物理场集成** | 结合温度、湿度等环境因素 | 更真实的环境光照建模 |

### 5.3 长期愿景 (理论研究)

| 方向 | 研究问题 | 学术价值 |
|------|----------|----------|
| **物理引导的神经表示理论** | 物理先验如何影响表示学习 | 连接物理与学习的理论框架 |
| **可微分物理模拟集成** | 将物理模拟作为可微分层 | 端到端的物理一致性学习 |
| **跨场景时间建模迁移** | 学习通用的时间变化模式 | 减少对新场景的数据需求 |

---

## 6. 关键文件参考

### 6.1 核心实现文件 (必读)

1. **FiLM统一架构核心**: `2_src/models/gaussian_physics_unified.py`
   - `GaussianPhysicsCompressionUnified`: FiLM统一架构模型
   - `LightSetEncoder`: 光照集合编码器，支持强度解耦

2. **统一光照描述符系统**: `2_src/utils/light_descriptor.py`
   - `build_descriptor_5d()`: 5D参数 → 12D统一描述符
   - `build_descriptor_1d()`: 1D强度 → 12D描述符（兼容格式）

3. **训练基础设施**: `2_src/training/gaussian_physics_trainer.py`
   - `GaussianPhysicsTrainer`: 多损失训练器，支持时间正则化

4. **统一训练入口**: `3_experiments/scripts/train.py`
   - 支持所有变体模型，包括FiLM统一架构 (`unified_set`, `unified_5d`)
   - 数据集适配器与光照描述符转换

### 6.2 历史参考文件 (了解演进)

1. **TemporalMLP完整实现**: `archive/TemporalMLP/src/`
   - 完整的基线方法实现

2. **TPE失败分析**: `archive/TPE_Method_Archived/docs/`
   - 详细的失败原因分析

3. **实验演进记录**: `tools/logexp.py` 数据库
   - Phase0_TPE → Phase1_TemporalMLP → Phase2_PGCPL

### 6.3 配置与文档

1. **最优配置**: `3_experiments/configs/baseline.yaml`
   - K30_r8配置，当前最优参数

2. **任务书规范**: `docs/thesis/多时刻光照压缩任务书.md`
   - 原始设计要求和技术指标

3. **项目指南**: `docs/dev_notes/CLAUDE.md`
   - 全面的开发指导文档

---

**文档状态**: 阶段1调研完成  
**更新建议**: 当新增时间建模方法或现有方法有重大修改时更新此文档  
**关联文档**: `PGGCPL_AUDIT.md` (项目整体审计报告)