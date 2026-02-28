# PSNR/SSIM差异问题分析报告

**日期**: 2026-02-10
**问题**: Benchmark测试中PSNR/SSIM为20/0.90，但Rerun记录中达到40/0.98

## 问题根源：两个路径测量的是不同的东西

### 1. Rerun路径（训练可视化）

**测试内容**: 单个探针的SH系数预测准确性

**流程**:
```
单个探针SH系数 (27维)
    ↓
转换为环境贴图 (128×256 equirectangular)
    ↓
使用环境贴图渲染场景 (Falcor path tracer, SPP=32)
    ↓
比较 GT探针 vs 预测探针 的渲染结果
```

**关键代码位置**:
- `2_src/utils/rerun_logger.py:303-490` - `log_rendered_comparison()`
- `2_src/utils/rendering_utils.py:51-85` - `sh_to_envmap()`
- `2_src/utils/rendering_utils.py:444-524` - `render_with_envmap_external()`

**渲染参数**:
- SPP: 32 (路径追踪采样数)
- 分辨率: 256×256
- 输出通道: `AccumulatePass.output` (累积的路径追踪结果)
- Tone mapping: Reinhard (在Python中应用)
- PSNR/SSIM计算: scikit-image库

**测量的是**: "模型能否准确预测单个空间位置的光照SH系数"

---

### 2. Benchmark路径（实时渲染管线）

**测试内容**: 整个场景的空间变化光照渲染质量

**流程**:
```
所有探针SH系数 (N×27)
    ↓
构建3D探针场 (32³网格 + KNN插值, k=8)
    ↓
GPU compute shader实时渲染 (单次采样)
    ↓
比较 GT场 vs 模型场 的整个场景渲染结果
```

**关键代码位置**:
- `tools/benchmark_realtime_pipeline_worker.py:296-304` - 探针场构建
- `tools/benchmark_realtime_pipeline_worker.py:496-527` - 场构建和上传
- `tools/sh_probe_volume.cs.slang` - GPU着色器

**渲染参数**:
- 场分辨率: 32³ (field_res=32)
- KNN邻居数: k=8 (field_knn=8)
- 权重epsilon: 0.1 (weight_eps=0.1)
- 渲染: 单次compute shader pass (无路径追踪)
- 插值: 三线性插值 (shader中第96-146行)
- Tone mapping: Reinhard (在Python中应用)
- PSNR/SSIM计算: 自定义实现

**测量的是**: "整个实时渲染管线（模型+空间离散化+插值+渲染）的端到端质量"

---

## 关键差异分析

### 差异1: 测试范围

| 方面 | Rerun | Benchmark |
|------|-------|-----------|
| 输入 | 单个探针 (1×27) | 所有探针 (N×27) |
| 空间表示 | 环境贴图 (全局光照) | 3D探针场 (空间变化光照) |
| 测试对象 | 单点预测准确性 | 整个场景渲染质量 |

### 差异2: 误差来源

**Rerun路径的误差**:
- 仅模型预测误差

**Benchmark路径的误差**:
1. 模型预测误差
2. **空间离散化误差** (连续探针 → 32³网格)
3. **KNN插值误差** (k=8邻居加权平均)
4. **三线性插值误差** (shader中的网格插值)
5. **渲染方法差异** (compute shader vs 路径追踪)

### 差异3: 渲染质量

| 方面 | Rerun | Benchmark |
|------|-------|-----------|
| 渲染方法 | 路径追踪 (SPP=32) | Compute shader (单次) |
| 光照表示 | 环境贴图 (高分辨率) | 探针场 (32³网格) |
| 采样质量 | 多次采样累积 | 单次采样 |

---

## 问题诊断

### Benchmark PSNR/SSIM低的主要原因

1. **空间离散化过粗** (32³网格)
   - 场景可能需要更高分辨率的网格
   - 当前设置: `field_res=32`
   - 建议测试: 64³或128³

2. **KNN插值参数不优**
   - k=8可能导致过度平滑
   - weight_eps=0.1可能不合适
   - 建议测试不同的k值和epsilon

3. **双重插值累积误差**
   - 第一次: KNN加权平均 (Python)
   - 第二次: 三线性插值 (Shader)
   - 两次插值会累积误差

4. **渲染方法差异**
   - Compute shader是单次采样
   - 没有路径追踪的多次采样平滑效果

---

## 验证方法

### 方法1: 检查GT路径的质量

在benchmark中，GT路径使用的是ground truth SH系数，如果GT路径本身的PSNR就不高，说明问题在于**探针场构建和渲染管线**，而不是模型预测。

**检查位置**: `tools/benchmark_realtime_pipeline_worker.py:482-490`
```python
if route == "gt":
    sh_probe = tensor[:, frame, :]  # 使用GT SH系数
```

### 方法2: 对比单探针渲染

修改benchmark，让它也渲染单个探针的环境贴图（类似rerun），看PSNR/SSIM是否提升到40/0.98。

### 方法3: 提高场分辨率

测试不同的field_res值:
- 当前: 32³ (32,768个单元)
- 测试: 64³ (262,144个单元)
- 测试: 128³ (2,097,152个单元)

### 方法4: 调整KNN参数

测试不同的插值参数:
- field_knn: 4, 8, 16, 32
- weight_eps: 0.01, 0.1, 1.0

---

## 结论

**两个路径的PSNR/SSIM差异是正常的**，因为它们测量的是不同的东西：

- **Rerun (40/0.98)**: 测量模型对单个探针的预测准确性
- **Benchmark (20/0.90)**: 测量整个实时渲染管线的端到端质量

Benchmark的低PSNR/SSIM主要由以下因素导致：
1. 空间离散化误差（32³网格太粗）
2. KNN插值平滑效应
3. 三线性插值累积误差
4. 单次采样vs多次采样的渲染质量差异

**这不是bug，而是两个不同测试场景的预期结果差异。**

---

## 建议的改进方向

### 短期改进（提升benchmark PSNR/SSIM）

1. **提高场分辨率**: field_res从32增加到64或128
2. **优化KNN参数**: 测试不同的k和weight_eps
3. **减少插值次数**: 考虑直接在shader中做KNN查询，避免预先构建网格

### 长期改进（统一测试标准）

1. **添加单探针benchmark**: 让benchmark也支持单探针环境贴图渲染模式
2. **记录GT路径质量**: 在benchmark中单独报告GT路径的PSNR/SSIM
3. **分离误差来源**: 分别测量模型误差、空间离散化误差、渲染误差

### 文档改进

1. 在benchmark输出中明确说明测试的是"端到端渲染质量"
2. 在rerun日志中明确说明测试的是"单探针预测准确性"
3. 添加说明：两个指标不应直接比较

---

## 附录：代码位置索引

### Rerun路径关键代码
- SH转环境贴图: `2_src/utils/rendering_utils.py:51-85`
- 环境贴图渲染: `2_src/utils/rendering_utils.py:244-424`
- 外部进程渲染: `2_src/utils/rendering_utils.py:444-524`
- PSNR/SSIM计算: `2_src/utils/rendering_utils.py:529-555` (scikit-image)
- Rerun日志记录: `2_src/utils/rerun_logger.py:303-490`

### Benchmark路径关键代码
- 探针场构建: `tools/benchmark_realtime_pipeline_worker.py:29-63`
- 场打包: `tools/benchmark_realtime_pipeline_worker.py:66-73`
- GPU渲染循环: `tools/benchmark_realtime_pipeline_worker.py:424-598`
- PSNR/SSIM计算: `tools/benchmark_realtime_pipeline_worker.py:141-175` (自定义)
- Compute shader: `tools/sh_probe_volume.cs.slang`

### 配置参数
- Benchmark参数: `tools/run_benchmark_matrix.py:104-111`
  - field_res: 32 (默认)
  - field_knn: 8 (默认)
  - weight_eps: 0.1 (默认)
- Rerun参数: `2_src/utils/rerun_logger.py:380-420`
  - SPP: 32 (默认)
  - 分辨率: 256×256 (默认)
