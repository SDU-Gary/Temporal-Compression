# 动态光照探针隐式压缩实验指导（Falcor 全链路）

**目的**  
在现有 PG‑GCPL 框架基础上，完成“通用动态光照”方向的**物理一致性**与**可控泛化**验证。本文档将理论策略与当前仓库路径绑定，形成可执行的实验闭环。

---

## 0. 前置约定（必须统一）

### 0.1 线性空间
- **禁止** Tone Mapping / Gamma / Auto-Exposure
- 输出 SH 必须保持线性空间系数

### 0.2 线性叠加原则
- 任何多光源组合必须满足：
  ```
  SH(A+B) ≈ SH(A) + SH(B)
  ```
- 若不成立，先修正数据生成管线（Falcor 端）。

### 0.3 SH 归一化策略
- **不改变输出空间**（保持线性 SH）
- 可在 loss 内部做权重：
  ```
  w = 1 / (||SH_gt|| + eps)
  ```
- 可选：**L0 单独分支预测**（见 §3.4）。

---

## 1. Sanity Tests（准入测试）

### 1.1 线性叠加测试（必做）
**目标**：验证数据本身满足线性叠加。

**流程**：
1) 在 Falcor 中渲染光源 A、光源 B、光源 A+B
2) 生成 SH_A、SH_B、SH_AB
3) 计算误差：
```
ε_lin = mean(||SH_AB - (SH_A + SH_B)||)
```

**建议实现位置**：
- 新增脚本建议：`1_data_generation/falcor/linearity_check.py`
- 可复用 `1_data_generation/falcor/utils/falcor_render.py`

### 1.2 方向一致性测试
**目标**：验证 SH 方向与光源物理方向对齐。

**流程**：
- 用远场光方向 `(0, 1, 0)` 或 `(0, -1, 0)`
- 检查重建后的主亮斑方向

**建议实现位置**：
- 新增脚本建议：`1_data_generation/falcor/direction_alignment_check.py`

---

## 2. 数据生成（Falcor Pipeline）

### 2.1 基础脚本（已存在）
- 5D 参数化数据生成：
  - `1_data_generation/falcor/generate_5d_parametric_falcor.py`
- 1D 强度调制数据生成：
  - `1_data_generation/falcor/generate_intensity_modulation_falcor.py`

### 2.2 需新增的多光源数据生成
**最小目标**：N=2 光源组合
- 太阳（远场）+ 点光源（近场）
- 输出统一 Light Descriptor + SH

建议新增脚本：
- `1_data_generation/falcor/generate_multilight_falcor.py`

输出格式（建议）:
```
parametric_tensor.npz
  - tensor: [P, M, 27]
  - probe_positions: [P, 3]
  - light_configs: [M, N, D]  # N 个光源，每个 D 维描述
metadata.json
```

---

## 3. 模型结构（统一输入 + 物理一致性）

### 3.1 统一 Light Descriptor
建议固定维度 D（16~32）：
```
[ type, pos(3), dir(3), intensity(3), size(1), color_temp(1), flags(...) ]
```
- N 个光源输入 → padding + mask

### 3.2 强度解耦编码（必须）
保证线性叠加：
```
Embed(L_i) = intensity_i * MLP(type, pos, dir, ...)
```

### 3.3 线性聚合（Sum Pooling）
```
Z = Σ mask_i * Embed(L_i)
```

### 3.4 动态基调制（FiLM）
```
U'_j = U_j ⊙ γ(Z) + β(Z)
```

### 3.5 L0 分支（可选）
- 直接预测 SH 的 L0（亮度）分量
- 其余系数走 PG‑GCPL 主干

**改造位置建议**：
- 模型：`2_src/models/gaussian_physics_5D.py` 或新建通用版
- Trainer：`2_src/training/gaussian_physics_trainer.py`
- 统一入口：`3_experiments/scripts/train.py`

---

## 4. Loss 设计（物理一致性）

```
L = L_recon + λ_lin * L_lin + λ_temp * L_temp
```

- **L_recon**：Charbonnier / L1
- **L_lin**：线性叠加约束（随机取 A,B）
  ```
  L_lin = ||f(A+B) - (f(A)+f(B))||
  ```
- **L_temp**：时间平滑

**注意**：L_lin 应建立在**线性空间 SH**输出上。

---

## 5. 评价指标（必须覆盖物理一致性）

### 5.1 物理一致性
- Superposition Error:
  ```
  ε_lin = ||Model(A+B) - (Model(A)+Model(B))|| / ||Model(A+B)||
  ```

### 5.2 精度
- SH RMSE / MAE
- 渲染 PSNR / SSIM（Falcor 中复渲染）

### 5.3 动态特性
- Temporal Flicker Ratio:
  ```
  Flicker = ΔPred / ΔGT
  ```

---

## 6. Baseline 对比（必做）

| Baseline | 名称 | 说明 | 目的 |
|---|---|---|---|
| B1 | PCA + Linear Interp | 离散网格 + PCA | 证明隐式优于查表 |
| B2 | Fixed Physical Basis | 旧 V1.0 | 证明学习编码必要 |
| B3 | Component-wise Sum | 单光源再叠加 | 物理正确性基线 |

**注意**：B3 是“物理上限”，模型至少应逼近该结果。

---

## 7. 分阶段 Roadmap（建议执行顺序）

### Week 1
- Linearity Check + Direction Alignment
- 输出单光源移动数据（Corridor）

### Week 2
- 加入 FiLM 动态调制
- 单光源数据训练，验证阴影拟合能力

### Week 3
- 多光源数据（N=2）生成
- Set Input + Sum Pooling + Mask 训练

### Week 4
- Sponza/Bistro 混合验证
- Baseline 对比 + Flicker 指标

---

## 8. 立即可执行的下一步

1) **实现线性叠加测试脚本**（Falcor）
2) **生成 Corridor 单光源移动数据**
3) **在现有训练代码里加 FiLM 调制最小实现**

---

## 9. 关联代码位置索引（当前项目）

- Falcor 渲染工具：`1_data_generation/falcor/utils/falcor_render.py`
- 5D 数据生成：`1_data_generation/falcor/generate_5d_parametric_falcor.py`
- 1D 数据生成：`1_data_generation/falcor/generate_intensity_modulation_falcor.py`
- Dataset (1D)：`2_src/data/intensity_modulation_dataset.py`
- Dataset (5D)：`2_src/data/transfer_tensor_dataset.py`
- Trainer：`2_src/training/gaussian_physics_trainer.py`
- 统一训练入口：`3_experiments/scripts/train.py`

---

## 10. 备注
- 所有实验必须记录到 `metadata/project.db`
- 每次实验结束后执行：
  - `python tools/logexp.py log ...`
  - `python tools/logexp.py query --limit 10`
