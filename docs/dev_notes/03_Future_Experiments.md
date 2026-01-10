# 高斯-物理混合压缩后续实验设计

**日期**: 2025-12-12
**基于**: 09_Gaussian_Physics_Hybrid 初步实验结果

---

## 🎯 实验目标

基于初步实验（17.8x压缩，MAE=0.1141），系统性改进方法以达到：
1. **压缩比目标**: 保持 >15x 的同时
2. **精度目标**: MAE降至接近Spline水平（<0.05）
3. **泛化目标**: 24小时全天候验证

---

## 📊 当前问题诊断

### 初步实验结果回顾
```
数据集: method_test_v1 (343探针×6时刻)
配置: K=20, rank=5, 3,120参数
训练: 3个时刻（奇数小时），测试: 3个时刻（偶数小时）

结果:
✅ 压缩比: 17.81x (优秀)
✅ vs PhysicsOnly: MAE降低34% (证明空间建模有效)
❌ vs Spline: MAE高2.87倍 (精度不足)
```

### 关键限制因素

**1. 训练数据不足**
- 只有3个训练时刻 → SVD只能提取rank=3（而非rank=5）
- 每个高斯有效秩不足 → 时间建模能力受限

**2. 高斯覆盖密度**
- K=20覆盖343探针 → 每个高斯平均17个探针
- 空间分辨率有限 → 局部细节丢失

**3. 训练不充分**
- 只训练1000 epochs
- 最终loss仍在下降 → 未收敛到最优

**4. 物理先验不完整**
- 只用太阳高度角θ
- 缺少方位角φ → 无法区分相同高度但不同方位的情况

---

## 🔬 实验设计（分4个阶段）

### 阶段A：单变量优化（找到最优超参数）

#### Experiment A1: 高斯数量对比
**目标**: 找到精度-压缩比的最优平衡点

**配置变量**: K ∈ {20, 30, 50, 75, 100}
**固定参数**: rank=5, 训练时刻=3, epochs=2000

**预期**:
- K↑ → 精度↑，压缩比↓
- 存在拐点K*使得精度/压缩比比值最优

**评估指标**:
- 测试MAE vs K
- 参数量 vs K
- 压缩比 vs K
- 效率指标: MAE/(参数量/1000)

**实验矩阵**:
```
| K  | 参数量 | 预期MAE | 预期压缩比 |
|----|--------|---------|-----------|
| 20 | 3,120  | 0.114   | 17.8x     |
| 30 | 4,680  | 0.090   | 11.9x     |
| 50 | 7,800  | 0.065   | 7.1x      |
| 75 | 11,700 | 0.050   | 4.7x      |
| 100| 15,600 | 0.045   | 3.6x      |
```

**脚本**: `experiments/a1_gaussian_count.py`

---

#### Experiment A2: 秩数对比
**目标**: 验证低秩假设的有效性

**配置变量**: rank ∈ {3, 5, 7, 10}
**固定参数**: K=50, 训练时刻=3, epochs=2000

**理由**:
- 当前rank=5被限制为rank=3（训练数据不足）
- 使用rank=3/5/7可验证秩的影响
- rank=10作为高秩对照组

**预期**:
- rank=3: 参数少但拟合能力弱
- rank=5: 平衡点
- rank=7/10: 参数多但可能过拟合

**评估指标**:
- 测试MAE vs rank
- 训练MAE vs rank（检测过拟合）
- SVD能量占比 vs rank

**实验矩阵**:
```
| rank | 参数量 | 预期MAE | 过拟合风险 |
|------|--------|---------|-----------|
| 3    | 6,600  | 0.080   | 低        |
| 5    | 7,800  | 0.065   | 中        |
| 7    | 9,000  | 0.062   | 中-高     |
| 10   | 11,100 | 0.060   | 高        |
```

**脚本**: `experiments/a2_rank_comparison.py`

---

#### Experiment A3: 训练时长对比
**目标**: 确认是否充分训练

**配置变量**: epochs ∈ {500, 1000, 2000, 5000}
**固定参数**: K=50, rank=5, 训练时刻=3

**评估指标**:
- 训练曲线收敛情况
- 测试MAE vs epochs
- 早停点识别

**脚本**: `experiments/a3_training_length.py`

---

### 阶段B：数据增强（解决训练数据不足）

#### Experiment B1: 增加训练时刻
**目标**: 提供更多训练数据，让SVD能提取完整rank

**配置变量**: 训练时刻数 ∈ {3, 6}
**固定参数**: K=50, rank=5, epochs=2000

**数据分割**:
- 3时刻训练: 奇数小时（9, 15, 21）
- 6时刻训练: 全部6个时刻

**关键变化**:
```python
# 之前: 3训练/3测试
train_hours = [9, 15, 21]  # 奇数
test_hours = [6, 12, 18]   # 偶数

# 现在: 6训练（无测试集，用交叉验证）
train_hours = [6, 9, 12, 15, 18, 21]  # 全部
# 使用 K-fold 交叉验证评估泛化能力
```

**预期**:
- SVD能提取完整rank=5
- 训练MAE↓
- 但可能牺牲泛化能力（无独立测试集）

**评估策略**:
- 3-fold交叉验证
- Leave-one-out时刻验证

**脚本**: `experiments/b1_more_training_moments.py`

---

#### Experiment B2: 数据增强技术
**目标**: 在不生成新数据的情况下扩充训练集

**增强方法**:
1. **时间插值**: 在已有时刻间插值生成伪时刻
   ```python
   # 例如：在9点和12点间插值生成10点、11点
   sh_10 = (2*sh_09 + sh_12) / 3
   sh_11 = (sh_09 + 2*sh_12) / 3
   ```

2. **空间邻域**: 利用空间相关性增强
   ```python
   # 对每个探针，使用其K近邻的加权平均作为增强样本
   sh_augmented = Σ w_i * sh_neighbor_i
   ```

3. **物理约束增强**: 基于太阳轨迹的对称性
   ```python
   # 早晨和傍晚的光照应该对称（相同高度角）
   # 可以镜像增强
   ```

**配置**:
- 基础: 3训练时刻
- 插值: 生成6个伪时刻 → 9训练样本
- 邻域: 每个探针生成3个邻域样本
- 物理: 利用对称性2x增强

**评估**:
- 增强前后MAE对比
- 检查生成样本的物理合理性

**脚本**: `experiments/b2_data_augmentation.py`

---

### 阶段C：模型改进（增强表达能力）

#### Experiment C1: 加入方位角
**目标**: 增强时间建模的完整性

**当前**: 只用太阳高度角θ
```python
basis = [cos(θ), sin(θ), 1]  # 3维
time_coeffs: [rank, 3]
```

**改进**: 加入太阳方位角φ
```python
basis = [cos(θ), sin(θ), cos(φ), sin(φ), 1]  # 5维
time_coeffs: [rank, 5]  # 参数增加67%
```

**预期**:
- 能区分相同高度但不同方位的光照
- 对于复杂几何场景（如House）改善明显
- 参数增加: 50*5*5 = 1,250 (之前750)

**配置**:
- K=50, rank=5
- 太阳位置计算: (θ, φ) = compute_sun_position(hour, lat, lon, date)

**评估**:
- 对比3维基函数 vs 5维基函数
- 可视化不同方位角下的预测差异

**脚本**: `experiments/c1_azimuth_angle.py`

---

#### Experiment C2: 层次化秩
**目标**: 不同空间位置使用不同的秩

**动机**:
- 光照复杂的区域（阴影边界）需要高秩
- 光照平滑的区域（天空）只需低秩

**方法**: 自适应秩分配
```python
class AdaptiveRankGaussian:
    def __init__(self, K, max_rank=10, min_rank=3):
        self.ranks = nn.Parameter(torch.ones(K) * 5)  # 可学习
        self.U = []
        for k in range(K):
            rank_k = int(self.ranks[k].item())
            self.U.append(nn.Parameter(torch.randn(27, rank_k)))
```

**优化目标**:
- L_total = L_recon + λ_sparse * Σ|rank_k - min_rank|
- 鼓励使用低秩，但允许必要时提升秩

**预期**:
- 总参数量降低
- 精度保持或提升

**脚本**: `experiments/c2_adaptive_rank.py`

---

#### Experiment C3: 多尺度高斯
**目标**: 类似图像金字塔，使用不同尺度的高斯

**当前**: 单一尺度，每个高斯覆盖半径相似

**改进**: 3个尺度层
```
粗尺度: K_coarse=10, 覆盖半径大, rank=3
中尺度: K_mid=30, 覆盖半径中等, rank=5
细尺度: K_fine=60, 覆盖半径小, rank=7
总计: K=100
```

**查询策略**:
```python
def query(position, time):
    # 粗尺度：全局光照趋势
    F_coarse = query_coarse_gaussians(position, time)

    # 中尺度：区域光照变化
    F_mid = query_mid_gaussians(position, time)

    # 细尺度：局部细节
    F_fine = query_fine_gaussians(position, time)

    # 加权融合
    return w_c*F_coarse + w_m*F_mid + w_f*F_fine
```

**预期**:
- 更好的空间适应性
- 参数分配更合理

**脚本**: `experiments/c3_multiscale_gaussian.py`

---

### 阶段D：泛化验证（扩展到24小时）

#### Experiment D1: 24小时全天实验
**目标**: 验证方法在完整日循环的表现

**数据需求**:
- 生成24小时数据（每小时1个时刻）
- 包括夜间时刻（太阳在地平线下）

**挑战**:
1. **夜间处理**: 太阳高度角为负
   ```python
   # 方案1: 截断为0
   θ = max(0, compute_elevation(hour))

   # 方案2: 引入夜间标志位
   basis = [cos(θ), sin(θ), is_night, 1]
   ```

2. **跨昼夜边界**: 日出/日落时刻光照剧烈变化
   - 可能需要更高的秩
   - 需要时间平滑正则化

**训练策略**:
- 训练: 0, 2, 4, ..., 22点（12个偶数小时）
- 测试: 1, 3, 5, ..., 23点（12个奇数小时）

**评估**:
- 昼间MAE vs 夜间MAE
- 日出/日落时段MAE（预期最高）
- 完整24小时重建视频可视化

**脚本**: `experiments/d1_24hour_full_cycle.py`

---

#### Experiment D2: 跨场景泛化
**目标**: 测试方法在不同场景的泛化能力

**实验设置**:
- 训练: Cornell Box场景
- 测试: House场景（zero-shot）

**假设**:
- 物理基函数（cos/sin太阳角度）是通用的
- 空间高斯需要针对每个场景重新训练

**方法**: 迁移学习
```python
# 1. 在Cornell Box上训练
model_cornell = GaussianPhysicsCompression(K=50, rank=5)
train(model_cornell, cornell_data)

# 2. 迁移到House
model_house = GaussianPhysicsCompression(K=50, rank=5)
# 冻结time_coeffs（物理先验），只训练空间高斯
for param in model_house.time_coeffs.parameters():
    param.requires_grad = False

train(model_house, house_data)
```

**预期**:
- 时间系数可迁移 → 加速House场景训练
- 空间高斯需要重新学习

**评估**:
- 从头训练 vs 迁移学习的收敛速度
- 最终精度对比

**脚本**: `experiments/d2_cross_scene_transfer.py`

---

## 📅 实验执行计划

### 第一周：阶段A（单变量优化）
```
Day 1-2: A1 高斯数量对比（5个配置）
Day 3-4: A2 秩数对比（4个配置）
Day 5:   A3 训练时长（确定最优epochs）
Day 6-7: 分析结果，确定最优超参数组合
```

**预期输出**:
- 最优配置: K*, rank*, epochs*
- 性能曲线图
- Pareto前沿（精度-压缩比）

---

### 第二周：阶段B（数据增强）
```
Day 8-9:  B1 增加训练时刻（3→6）
Day 10-12: B2 数据增强技术
Day 13-14: 对比分析，选择最佳增强策略
```

**预期输出**:
- 数据增强有效性验证
- 最佳增强方案
- SVD秩完整性验证

---

### 第三周：阶段C（模型改进）
```
Day 15-16: C1 加入方位角
Day 17-19: C2 层次化秩
Day 20-21: C3 多尺度高斯
```

**预期输出**:
- 改进模型实现
- 精度提升报告
- 消融实验结果

---

### 第四周：阶段D（泛化验证）
```
Day 22-24: D1 24小时全天实验
Day 25-26: D2 跨场景泛化
Day 27-28: 整理所有实验结果，撰写报告
```

**预期输出**:
- 完整24小时演示视频
- 跨场景迁移能力验证
- 最终实验报告

---

## 📊 预期最终性能

基于实验设计，预期最终达到：

### 保守估计
```
配置: K=50, rank=5, 加入方位角
数据: 6训练时刻 + 轻量增强
训练: 2000 epochs

结果:
- 压缩比: 10-12x
- 测试MAE: 0.05-0.06
- vs Spline: MAE比值 1.3-1.5x
```

### 理想情况
```
配置: K=75, rank=5, 加入方位角, 多尺度
数据: 12训练时刻（24小时数据）
训练: 5000 epochs

结果:
- 压缩比: 8-10x
- 测试MAE: 0.04-0.045
- vs Spline: MAE比值 ~1.1x (接近)
```

---

## 🎯 成功标准

### 必须达到（毕设通过）
- ✅ 压缩比 > 10x
- ✅ 测试MAE < 0.06
- ✅ 训练收敛稳定
- ✅ 24小时完整演示

### 期望达到（优秀毕设）
- ✅ 压缩比 > 15x 且 MAE < 0.05
- ✅ 接近Spline精度（MAE比值 < 1.2x）
- ✅ 跨场景泛化有效
- ✅ 理论分析完整

### 冲刺目标（发论文）
- ✅ 压缩比 > 15x 且 MAE < Spline
- ✅ 多场景验证（5+场景）
- ✅ 与SOTA方法对比（Gaussian Compression, K-Planes）
- ✅ 实时解压缩实现（CUDA）

---

## 🔧 实现工具

### 实验管理框架
```python
# experiments/experiment_manager.py
class ExperimentManager:
    """统一管理所有实验的执行、记录、对比"""

    def __init__(self, base_config):
        self.base_config = base_config
        self.results = {}

    def run_experiment(self, name, config_overrides):
        """运行单个实验"""
        config = {**self.base_config, **config_overrides}
        result = train_and_evaluate(config)
        self.results[name] = result
        self.save_result(name, result)

    def run_grid_search(self, param_grid):
        """网格搜索"""
        for params in itertools.product(*param_grid.values()):
            config = dict(zip(param_grid.keys(), params))
            name = self.generate_name(config)
            self.run_experiment(name, config)

    def compare_results(self, metric='test_mae'):
        """对比所有实验结果"""
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1][metric]
        )
        return sorted_results

    def plot_pareto_frontier(self, x='params', y='test_mae'):
        """绘制Pareto前沿"""
        # ...
```

### 可视化工具
```python
# utils/visualization.py
def plot_gaussian_distribution(model, probe_positions):
    """可视化高斯分布"""
    # 3D散点图显示高斯中心和探针位置

def plot_temporal_evolution(model, position, hours):
    """可视化时间演化"""
    # 显示某一位置的SH系数随时间变化

def create_24hour_video(model, dataset):
    """生成24小时演示视频"""
    # 渲染完整一天的光照变化

def plot_compression_quality_tradeoff(results):
    """压缩比-质量权衡曲线"""
    # Pareto前沿可视化
```

---

## 📝 实验记录模板

每个实验完成后，记录以下内容：

```markdown
# Experiment [ID]: [Name]

## 配置
- K: 50
- rank: 5
- epochs: 2000
- 训练时刻: [6, 9, 12, 15, 18, 21]

## 结果
- 训练MAE: 0.045
- 测试MAE: 0.052
- 参数量: 7,800
- 压缩比: 7.1x
- 训练时间: 15分钟

## 可视化
[插入图片]

## 分析
- 优点: ...
- 缺点: ...
- 下一步: ...

## 对比
与Experiment [上一个ID]相比：
- MAE: ↓ 8.5%
- 参数量: ↑ 12%
```

---

## 🚀 立即开始

**今晚就可以开始的实验**:

1. **Experiment A1** (高斯数量对比)
   - 最简单，只改一个参数
   - 运行5个配置约2-3小时
   - 立刻能看到效果

2. **快速原型验证**
   - K=30试一次，看MAE能否降到0.09
   - 如果有效，继续A1全流程

**脚本位置**:
- `/home/kyrie/毕设/multi_time_compression/experiments/`
- 复制`train_gaussian_physics.py`为`a1_gaussian_count.py`
- 修改参数循环部分

---

**接下来做什么？**
1. 先跑A1确认方向
2. 根据结果决定B/C/D的优先级
3. 每周末总结，调整计划

---

[← 返回索引](00_README.md) | [09 高斯-物理混合 ←](09_Gaussian_Physics_Hybrid.md)
