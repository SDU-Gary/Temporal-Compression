# Week 3-4 MLP混合方法失败归档

**日期**: 2026-01-03
**状态**: ❌ FAILED
**原因**: 数据量不足(41configs)导致MLP(5125参数)严重过拟合, Test MAE退化14.97%

---

## 实验结果

| 模型 | Val MAE | Test MAE | 参数量 | 结论 |
|------|---------|----------|--------|------|
| Stage 1 (Physics-only) | 0.0391 | **0.0876** | 170 | ✓ 成功 |
| Stage 2 (Hybrid) | 0.0372 | 0.1007 | 5,296 | ✗ 过拟合 |

**关键发现**:
- MLP学习到α=0.0126 (目标0.1的12.6%), 说明优化器认为"MLP有害"
- Physics-only已接近性能上限(MAE=0.088, 基于99.61% SVD能量捕获)
- 170参数physics > 5296参数hybrid

---

## 归档内容

### `/code`
- `physics_mlp_hybrid.py`: 混合架构 (SH = U @ (Φ_physics + α·MLP))
- `train_stage1_physics_only.py`: Stage 1训练脚本
- `train_stage2_hybrid.py`: Stage 2训练脚本

### `/checkpoints`
- `stage1_best.pt`: Stage 1最佳checkpoint (Val MAE=0.0391)
- `stage2_best.pt`: Stage 2最佳checkpoint (过拟合, 不可用)

### `/logs`
- `stage1_train.log`: Stage 1训练日志
- `stage2_training.log`: Stage 2训练日志

---

## 决策

**后续路线**: 放弃MLP混合, 继续**Physics-only多探针路线** (Week 5-6计划)

**理由**:
1. Physics-only参数效率高 (170 vs 5296)
2. 已接近数据噪声下限
3. 符合任务目标(17×压缩比, 非精度极致优化)
4. GPU资源有限, 无法渲染足够数据(需100+ configs)

详见: `/home/kyrie/毕设/multi_time_compression/docs/experiment_reports/Week3-4_MLP_Hybrid_FAILED.md`
