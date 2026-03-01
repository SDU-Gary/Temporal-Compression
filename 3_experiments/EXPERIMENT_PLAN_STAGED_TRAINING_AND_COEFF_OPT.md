# 分阶段训练与 Coeff 架构优化实验计划（PG-GCPL / Bistro Clean v2）

## 1. 背景与目的

基于最新 Oracle 诊断（`oracle_diag_cpu_verify_fix`）可得：

- 现有 baseline（K30_r8）在 SH 域表现受限（all27 SH-PSNR 约 11.8 dB）。
- 在固定路由与基底（`U`）条件下，Oracle 可将 SH-PSNR 拉升到约 23 dB。
- 诊断标签为 **coeff-limited**，说明当前主瓶颈在 `z -> time_weights` 映射（coeff 链路），不是 `U` 完全失效。

因此后续实验聚焦两条主线：

1. **分阶段训练**：改善优化路径，让 coeff 分支先对齐可达上界，再联合微调。
2. **Coeff 架构优化**：在保持 `U * coeff` 可解释性的前提下，提高 coeff 表达能力与数值稳定性。

---

## 2. 统一评估口径（必须固定）

所有实验统一使用当前 unified metrics 框架，并记录：

1. SH 系数域：`SH-PSNR/SH-SSIM`（all27）
2. SH 非 L0：`SH(non-L0)-PSNR/SSIM`
3. Per-band：`L0/L1/L2` 各自 PSNR/SSIM
4. Benchmark 图像域：`HDR-PSNR/SSIM`、`FixedTM-PSNR/SSIM`、`Benchmark-PSNR/SSIM`
5. Oracle 诊断（关键里程碑实验）：
   - `all27_psnr_gain_db`
   - `non_l0_psnr_gain_db`
   - `oracle_consistency.improved_ratio_*`

> 说明：`linear joint-max` 指标仅作参考，不作为主决策依据。

---

## 3. 实验主线 A：分阶段训练（Staged Training）

## A0. 对照基线（Control）

- 目的：作为所有新策略对照。
- 方案：现有联合训练（joint）流程，不加冻结。

## A1. Coeff-first（推荐主线）

- 假设：先让 coeff 分支收敛，可显著缩小与 Oracle 上界差距。
- 阶段：
  - Stage-1（Coeff）：冻结 `mu/log_scale/U/U_l0/(可选 gamma,beta)`，仅训 `light_encoder + coeffs + coeffs_l0`。
  - Stage-2（Joint）：全参数解冻，小学习率联合训练。
- 预期：SH all27 与 non-L0 指标显著提升，且训练更稳定。

## A2. U-first（对照验证）

- 假设：验证“先训 U 再训 coeff”是否有效。
- 阶段：
  - Stage-1（U）：冻结 coeff 与 encoder，仅训 `U/U_l0`（可选 `mu/log_scale`）。
  - Stage-2（Coeff）：冻结 U，仅训 coeff/encoder。
  - Stage-3（Joint）：全解冻微调。
- 目的：检验该路径是否劣于 A1（预计不如 A1 稳定，但需数据证实）。

## A3. 交替训练（Alternating C/U）

- 假设：分块优化可缓解病态耦合。
- 方案：周期性交替（示例：5 epoch coeff-step + 1 epoch U-step）。
- 目的：在不增加太多模型复杂度下提升收敛质量。

---

## 4. 实验主线 B：Coeff 架构优化（保持可解释性）

## B1. Per-band Coeff（最高优先）

- 动机：当前误差主要集中在方向项，需将 `L1/L2` 从共享 coeff 中解耦。
- 方案：
  - 保留 L0 专分支。
  - 对 non-L0 改为至少两支：`coeff_l1`、`coeff_l2`（可同 rank 或分配不同 rank）。
- 预期：L1/L2 PSNR 提升最明显，整体 SH-PSNR 上升。

## B2. Z 预处理与数值稳定增强

- 动机：light descriptor 变化维度少，回归系统条件数差。
- 方案：
  - 对 `z` 加标准化/白化（训练统计）。
  - 对 coeff 增加轻量正则（L2/Frobenius）。
- 预期：训练波动降低、收敛更稳定，对 A1/A3 均有增益。

## B3. Mixture-of-Linear Coeff（次优先）

- 动机：单线性映射可能不足以覆盖多模式光照响应。
- 方案：
  - 每高斯使用小规模模式混合（如 `m=2`），仍保持“线性块可解释”。
- 风险：参数增量与过拟合风险上升，放在 B1/B2 之后验证。

---

## 5. 组合实验矩阵（执行顺序）

### 第一批（低风险高收益）

1. A1（Coeff-first）
2. A3（交替训练）
3. A2（U-first 对照）

### 第二批（结构增强）

4. A1 + B1
5. A1 + B1 + B2
6. A3 + B1 + B2

### 第三批（复杂结构）

7. A1 + B1 + B2 + B3(m=2)

---

## 6. 每组实验固定记录项

- 实验ID、配置文件路径、checkpoint 路径
- 冻结策略（参数组 + epoch 范围）
- 学习率与调度策略（含 warmup）
- 训练/验证曲线（recon、image loss、SH metrics）
- 统一评估结果（第2节全部指标）
- Oracle 复核结果（是否缩小上界差距）

---

## 7. 验收标准（阶段性）

- 阶段一（训练策略）：
  - 相比 A0，`SH all27` 和 `non-L0` 至少有稳定提升。
  - 训练稳定性改善（波动降低，复现实验更一致）。

- 阶段二（架构优化）：
  - `L1/L2` band 指标显著提升。
  - 图像域指标（FixedTM/Benchmark）同步提升或不退化。

- 阶段三（最终候选）：
  - 选出在“指标提升 / 复杂度开销 / 可解释性”三者平衡最优的方案，作为主线模型。

---

## 8. 当前结论与下一步

- 当前证据支持：优先推进 **A1（Coeff-first）+ B1（Per-band coeff）+ B2（z 稳定化）**。
- `U-first` 保留为对照，不作为默认主线。
- B3（Mixture-of-Linear）在前述方案达瓶颈后再引入。

