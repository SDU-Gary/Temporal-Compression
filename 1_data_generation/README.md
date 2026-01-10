# 多时刻光照数据集生成

本目录包含使用Mitsuba 3生成多时刻光照数据集的完整工具链。

## 目录结构

```
data_generation/
├── generate_dataset.py      # 主数据生成脚本
├── utils/
│   ├── spherical_harmonics.py  # 球谐系数拟合工具
│   └── sun_position.py         # 太阳位置计算
├── scenes/                    # 场景文件（可选）
└── output/                    # 输出数据目录
    └── prototype_dataset/
        ├── probes.npz         # 探针位置 [N, 3]
        ├── moment_06/         # 早晨6:00数据
        │   ├── sh_coeffs.npz     # 球谐系数 [N, 27]
        │   ├── sun_direction.txt # 太阳方向 [3]
        │   ├── camera_params.json
        │   └── images/           # Ground Truth图像
        │       ├── 000.exr
        │       └── 000.png
        ├── moment_12/         # 正午12:00数据
        └── moment_18/         # 傍晚18:00数据
```

## 环境准备

确保已激活虚拟环境：

```bash
source ../venv/bin/activate
```

## 快速开始

### 1. 测试球谐工具

```bash
cd utils
python spherical_harmonics.py
```

预期输出：
```
Testing Spherical Harmonics utilities...
Sampled 64 directions
SH basis matrix shape: (64, 9)
Fitted SH coefficients shape: (27,)
Reconstruction MSE: 0.000xxx
Relative error: x.xx%
All tests passed! ✓
```

### 2. 测试太阳位置计算

```bash
cd utils
python sun_position.py
```

预期输出：
```
Testing Sun Position calculations...
Generated 24 sun positions

Sample positions:
  Hour 00:00 - Direction: [xxx, xxx, xxx], Altitude: xx.x°
  Hour 06:00 - Direction: [xxx, xxx, xxx], Altitude: xx.x°
  ...
Saved sun directions to: sun_directions_24h.txt
Saved sun trajectory plot to: sun_trajectory.png
All tests passed! ✓
```

### 3. 生成原型数据集

**重要：首次运行需要较长时间（~30-60分钟），因为需要为每个探针烘焙球谐系数。**

```bash
python generate_dataset.py
```

配置参数（在`generate_dataset.py`的`main()`函数中）：

```python
config = {
    'num_probes': 500,      # 探针数量（初期少量）
    'num_moments': 3,       # 时刻数（早、午、晚）
    'num_gt_images': 10,    # 每时刻的GT图像数
    'num_sh_samples': 64,   # 球谐采样方向数
    'spp': 256,             # 每像素采样数
}
```

## 数据格式说明

### 探针位置 (probes.npz)

```python
import numpy as np

data = np.load('output/prototype_dataset/probes.npz')
positions = data['positions']  # Shape: [N, 3]
# positions[i] = [x, y, z] 第i个探针的世界坐标
```

### 球谐系数 (sh_coeffs.npz)

```python
data = np.load('output/prototype_dataset/moment_12/sh_coeffs.npz')
coeffs = data['coeffs']      # Shape: [N, 27]
sun_dir = data['sun_dir']    # Shape: [3]

# coeffs[i] = 第i个探针的27维球谐系数
# 组织方式：9个基函数 × RGB 3通道
# coeffs[i, :9]   - R通道的9个系数
# coeffs[i, 9:18] - G通道的9个系数
# coeffs[i, 18:]  - B通道的9个系数
```

### 重建辐射度

```python
from utils.spherical_harmonics import reconstruct_from_sh, fibonacci_sphere

# 1. 加载球谐系数
sh_coeffs = coeffs[0]  # 第0个探针

# 2. 定义查询方向
directions = fibonacci_sphere(64)

# 3. 重建辐射度
radiances = reconstruct_from_sh(sh_coeffs, directions, max_order=2)
# radiances.shape = [64, 3] RGB辐射度

# 4. 可视化为天空盒
# ... (需要额外的可视化代码)
```

## 扩展到完整数据集

当原型验证通过后，可逐步扩展：

### 阶段1：增加时刻数

```python
# 在generate_dataset.py中修改
selected_hours = [0, 3, 6, 9, 12, 15, 18, 21]  # 8个时刻
```

### 阶段2：增加探针数

```python
'num_probes': 2000,  # 从500增加到2000
```

### 阶段3：完整24时刻

```python
selected_hours = list(range(24))  # 完整24小时
'num_probes': 5000,
```

### 阶段4：生产级数据集

```python
'num_probes': 15000,     # 任务书目标
'num_moments': 24,
'num_gt_images': 150,    # 每时刻150张图像
'spp': 512,              # 更高质量
```

## 性能优化建议

### 1. 使用GPU加速

确保CUDA variant可用：
```python
mi.set_variant('cuda_ad_rgb')  # 而非llvm_ad_rgb
```

### 2. 并行生成多个时刻

可以修改脚本，使用`multiprocessing`并行处理不同时刻。

### 3. 降低初期质量

原型阶段可降低SPP：
```python
'spp': 128,  # 从256降低到128（速度提升2×）
```

### 4. 批量烘焙

修改`bake_sh_at_probe`以一次渲染多个探针。

## 数据验证

### 检查球谐重建质量

```bash
python -c "
import numpy as np
from utils.spherical_harmonics import compute_sh_reconstruction_error, fibonacci_sphere

# 加载数据
data = np.load('output/prototype_dataset/moment_12/sh_coeffs.npz')
sh_coeffs = data['coeffs'][0]  # 第一个探针

# 测试方向
directions = fibonacci_sphere(64)

# 需要Ground Truth辐射度来计算误差
# （这里需要额外的渲染来获取GT）
"
```

### 可视化探针分布

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

probes = np.load('output/prototype_dataset/probes.npz')['positions']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(probes[:, 0], probes[:, 1], probes[:, 2], s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title(f'{len(probes)} Probe Positions')
plt.savefig('probe_distribution.png')
```

## 常见问题

### Q: 渲染速度很慢怎么办？

A:
1. 确保使用GPU variant (`cuda_ad_rgb`)
2. 降低SPP（从256降到128）
3. 减少探针数量（500 → 200）
4. 减少球谐采样数（64 → 32）

### Q: 内存不足？

A:
1. 分批处理探针
2. 降低图像分辨率（512 → 256）
3. 减少同时加载的场景数

### Q: 如何验证数据正确性？

A:
1. 检查球谐重建误差（<5%为优）
2. 可视化GT图像（应看到清晰的光影变化）
3. 比对不同时刻的太阳方向

## 下一步

完成原型数据集生成后：

1. **训练Milestone 1模型**：单时刻4级空间层次
2. **验证数据质量**：PSNR > 35dB
3. **扩展到完整数据集**：15000探针 × 24时刻
4. **复杂场景**：添加更多几何体、材质变化

## 参考

- Mitsuba 3文档：`../mitsuba-docs/`
- 任务书：`../多时刻光照压缩任务书.md`
- 球谐基础：[Spherical Harmonics Lighting](https://www.ppsloan.org/publications/StupidSH36.pdf)
