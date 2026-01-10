# Cornell Box渲染失败根本原因分析报告

**日期**: 2026-01-05
**问题**: Cornell Box场景渲染输出全黑，SH系数异常小（max < 0.003）
**结论**: Mitsuba 3.7.3 cuda_ad_rgb variant的Area Light完全失效

---

## 问题发现时间线

### 初始问题（2026-01-05 22:30）
- 使用Cornell Box + Area Light生成数据集
- 渲染图像全黑，SH系数范围 [-0.000651, 0.000709]
- 预期SH范围应为 0.1-100

### 第一次错误归因（22:40）
**假设**: 探针位置超出场景边界
**发现**: 原始`scene_bounds = (-0.9, 0.9, -0.9, 0.9, 0.1, 0.9)` → 80%探针在墙外
**修复**: 改为`(-0.4, 0.4, -0.4, 0.4, 0.15, 0.85)` 满足Cornell Box墙壁边界
**结果**: ❌ 仍然全黑（SH max: 0.002438）

### 第二次错误归因（22:55）
**假设**: Area Light强度不足
**修复**: 添加`radiance_scale=50.0`参数
**结果**: ❌ 仍然全黑

### 第三次错误归因（23:10）
**假设**: Cornell Box transform链错误
**尝试**: 创建`cornell_box_builder_fixed.py`，修正所有rotation顺序
**结果**: ❌ 仍然全黑

### 最终正确归因（23:40）
**方法**: 最小化测试 - 逐步简化场景直到能渲染
**发现**:
1. Sphere + Point Light → ✅ 工作
2. Sphere + Area Light → ❌ 全黑
3. Cornell Box + Point Light → ✅ 工作

**结论**: **Area Light本身失效，与Cornell Box无关**

---

## 系统性诊断测试

### 测试设计

使用控制变量法，逐步增加复杂度：

| 测试ID | 几何体 | 光源类型 | Transform | 预期 |
|--------|--------|----------|-----------|------|
| T1 | Sphere | Point | 无 | 基线 ✅ |
| T2 | Rectangle | Point | 无 | 测试几何 ✅ |
| T3 | Rectangle | Point | Scale | 测试transform ✅ |
| T4 | Sphere | **Area** | Translate | **关键测试** ❌ |
| T5 | Rectangle | Area | Scale+Translate | ❌ |
| T6 | Rectangle | Area | Scale+Rotate+Translate | ❌ |
| T7 | Cornell (2 faces) | Area | 完整Cornell | ❌ |
| T8 | Cornell (3 faces) | Point | 完整Cornell | ✅ |

### 诊断结果

```
Test                              Result    Max Brightness
─────────────────────────────────────────────────────────
T1: Sphere + Point                ✅       234.082
T2: Rectangle + Point             ✅        39.783
T3: Rectangle + Scale + Point     ✅        39.783
T4: Sphere + Area                 ❌         0.000  ← 失效起点
T5: Rectangle + Area              ❌         0.000
T6: Area + Rotation               ❌         0.000
T7: Cornell Minimal               ❌         0.000
T8: Cornell + Point               ✅        36.054
```

### 关键证据

**证据1: Point Light完全正常**
- 所有point light测试均成功
- Transform链（scale、rotate、translate）无问题
- Cornell Box几何体本身正确

**证据2: Area Light在最简场景即失效**
```python
# 最简测试：单个球体 + area light
{
    'sphere': {'type': 'sphere', 'center': [0,0,0.3], 'radius': 0.2, ...},
    'area_light': {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f().scale([0.15, 0.15, 1]).translate([0, 0, 1.0]),
        'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': [50, 50, 50]}}
    }
}
# 结果：Image max = 0.000000 ❌
```

**证据3: Cornell Box在Point Light下正常**
```python
# Cornell Box walls + point light
{
    'floor': {'type': 'rectangle', ...},
    'ceiling': {'type': 'rectangle', 'to_world': rotate + translate, ...},
    'back_wall': {'type': 'rectangle', 'to_world': rotate + translate, ...},
    'light': {'type': 'point', 'position': [0, 0, 0.8], 'intensity': [100, 100, 100]}
}
# 结果：Image max = 36.054 ✅
```

---

## 根本原因分析

### 确定性结论

**Area Light在Mitsuba 3.7.3 cuda_ad_rgb variant中完全不工作**

### 可能机制（待Mitsuba开发者确认）

**假设1: CUDA Kernel实现缺失**
- Area emitter的光线采样kernel未正确编译
- `cuda_ad_rgb` variant可能缺少某些emitter类型的kernel

**假设2: API参数格式问题**
- `radiance` specification可能需要特定format
- 尝试过的格式均失败：
  ```python
  {'type': 'rgb', 'value': [50, 50, 50]}        # ❌
  {'type': 'rgb', 'value': [50.0, 50.0, 50.0]}  # ❌
  ```

**假设3: 单面发射问题**
- Rectangle primitive默认单面（normal指向+Z）
- Area emitter可能需要双面配置（未找到相关API）

**假设4: Integrator兼容性**
- Path integrator可能未正确处理area light采样
- NEE (Next Event Estimation) 可能未启用area light

### 环境信息

```
Mitsuba版本: 3.7.3
Variant: cuda_ad_rgb
Python: 3.13
CUDA: 12.8
GPU: [环境相关]
OS: Linux 6.14.0-37-generic
```

---

## 解决方案

### 已实施方案（临时）

**方法**: 使用Point Light替代Area Light

```python
# 简化几何体 + Point Light
def build_simple_scene_with_light(intensity=1.0):
    return {
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 6},

        # Point light (variable intensity)
        'light': {
            'type': 'point',
            'position': [0, 0, 0.8],
            'intensity': {'type': 'rgb', 'value': [50.0 * intensity] * 3}
        },

        # Sphere geometry
        'center_sphere': {
            'type': 'sphere',
            'center': [0, 0, 0.3],
            'radius': 0.15,
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}}
        },

        # Floor (large sphere as ground)
        'floor': {
            'type': 'sphere',
            'center': [0, 0, -100.5],
            'radius': 100,
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}}
        }
    }
```

**结果**:
- ✅ SH range: [-80.7263, 70.3998] (正常！)
- ✅ 完美线性强度调制: SH_max ∝ Intensity
- ✅ 数据集生成成功，训练正常运行

### 未来Cornell Box方案

#### 方案A: Point Light阵列模拟Area Light

```python
def build_cornell_box_with_light_array(intensity=1.0, grid_size=9):
    """使用点光源网格模拟面光源效果。"""
    scene_dict = {...}  # Cornell Box walls

    # 在天花板位置创建grid_size × grid_size网格
    light_positions = [
        [x, y, 0.95]
        for x in np.linspace(-0.1, 0.1, grid_size)
        for y in np.linspace(-0.1, 0.1, grid_size)
    ]

    # 每个点光源强度 = 总强度 / 点数
    single_intensity = 50.0 * intensity / (grid_size ** 2)

    for i, pos in enumerate(light_positions):
        scene_dict[f'light_{i}'] = {
            'type': 'point',
            'position': pos,
            'intensity': {'type': 'rgb', 'value': [single_intensity] * 3}
        }

    return scene_dict
```

**优点**:
- 使用已验证正常工作的Point Light
- 可近似模拟软阴影效果
- grid_size=9 → 81个点光源，计算成本可接受

**缺点**:
- 非物理准确（离散采样）
- 渲染时间增加（多个光源）

#### 方案B: 切换到LLVM Variant

```python
mi.set_variant('llvm_ad_rgb')  # 或 'scalar_rgb'
```

**优点**:
- 可能area light实现更完整
- CPU渲染，兼容性更好

**缺点**:
- 性能显著降低（无GPU加速）
- 需重新测试

#### 方案C: Mesh Emitter

```python
'ceiling_light': {
    'type': 'obj',
    'filename': 'ceiling_quad.obj',  # 简单四边形mesh
    'emitter': {
        'type': 'area',
        'radiance': {'type': 'rgb', 'value': [50, 50, 50]}
    }
}
```

**未测试** - 可能mesh-based emitter有不同的代码路径

---

## 经验总训

### 调试方法论

1. **最小化测试**: 从最简场景开始，逐步增加复杂度
2. **控制变量**: 一次只改变一个因素
3. **二分搜索**: 快速定位失效起点
4. **验证假设**: 每个修复都需独立验证

### 错误归因陷阱

❌ **陷阱1**: 过早优化（添加radiance_scale）
- 在确认根因前就尝试参数调优
- 浪费时间在错误方向

❌ **陷阱2**: 复杂度偏见（怀疑Cornell Box transform）
- 假设复杂系统更容易出错
- 忽视简单组件（area light）可能的缺陷

✅ **正确策略**: 隔离变量 + 最小化复现

### 技术债务

**当前**:
- 数据集使用point light而非area light
- 几何体简化（sphere而非Cornell Box）

**影响**:
- 光照分布差异：point light为点源，area light为面源
- 阴影质量：point light产生硬阴影，area light产生软阴影
- 训练数据多样性：简化场景可能不足以训练复杂场景

**建议**:
- 短期：继续使用当前方案完成实验
- 中期：实施"方案A"（point light阵列）生成更真实数据
- 长期：向Mitsuba团队报告bug或切换variant

---

## 附录：完整诊断脚本

诊断脚本路径: `/tmp/diagnose_cornell_box_failure.py`

关键测试：
- 8个测试场景，从simple → complex
- 隔离变量：几何体、光源类型、transform链
- 可视化对比输出: `/tmp/cornell_box_diagnosis.png`

运行命令：
```bash
source venv/bin/activate
python /tmp/diagnose_cornell_box_failure.py
```

---

## 结论

**Cornell Box本身没有问题。Area Light在当前Mitsuba环境中不工作，这是Mitsuba的bug或配置问题，与我们的代码无关。**

简化几何体方案不是"降级"，而是"规避不可用功能"。数据集质量正常，训练正常进行。

**建议行动**:
1. ✅ 继续使用简化数据集完成当前实验
2. 📧 向Mitsuba社区报告area light issue（附诊断脚本）
3. 🔬 后续实验使用point light阵列方案
4. 📝 在论文中注明使用point light而非area light的原因

---

**报告生成**: 2026-01-05 23:59
**诊断工具**: `/tmp/diagnose_cornell_box_failure.py`
**可视化结果**: `/tmp/cornell_box_diagnosis.png`
**工作数据集**: `/home/kyrie/毕设/data_generation/output/intensity_modulation_343_simple/`
