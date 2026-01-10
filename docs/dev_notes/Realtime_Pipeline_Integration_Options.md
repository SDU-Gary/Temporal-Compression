# 实时管线集成方案对比分析

**目标**: 在真实渲染管线中集成测试PG-GCPL方法,验证实时解压性能

---

## 方案对比

| 方案 | 平台支持 | 图形API | 开发难度 | 集成周期 | 推荐度 |
|------|---------|---------|---------|---------|--------|
| **AMD Capsaicin** | Windows only | DX12 | 低 | 3-4周 | ⭐⭐ (需Windows) |
| **Nvidia Falcor** | Windows + **Linux** | DX12 + **Vulkan** | 中 | 4-5周 | ⭐⭐⭐⭐ (Linux友好) |
| **独立GPU Benchmark** | 跨平台 | PyTorch/CUDA | 低 | 1周 | ⭐⭐⭐⭐⭐ (快速验证) |

---

## 方案1: AMD Capsaicin (仅Windows)

### 技术特点
- **GI-1.0/1.2**: 双层辐射缓存 (屏幕探针 + 世界哈希网格)
- **性能基准**: 3.5ms@1080p (Kitchen), 4.2ms (Sponza)
- **框架**: 开源DX12渲染研究平台
- **代码**: https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin

### 平台限制
```bash
# 硬性要求
- Windows 10 SDK 2004+
- DX12 Ultimate capable GPU
- Visual Studio 2022
- CMake 3.30+
```

**❌ 不支持Linux**: Capsaicin基于DX12,无Vulkan后端

### 集成优势
- **即插即用**: GI-1.2已有屏幕探针生成,可直接替换
- **性能对比清晰**: 3.5ms光追 vs 0.13ms神经解压
- **文档完善**: 官方技术论文 + 源码

### 集成劣势
- **需要Windows环境** (虚拟机性能损失 >30%)
- C++17/HLSL开发栈 (当前Python/PyTorch)

---

## 方案2: Nvidia Falcor ⭐⭐⭐⭐ (推荐)

### 技术特点
- **跨平台**: Windows + **Ubuntu 22.04实验性支持**
- **双API**: DirectX 12 + **Vulkan**
- **Render Pass系统**: 模块化插件架构
- **代码**: https://github.com/NVIDIAGameWorks/Falcor

### Linux支持验证 ✅

**官方文档确认** ([GitHub README](https://github.com/NVIDIAGameWorks/Falcor)):
> "Falcor has experimental Ubuntu 22.04 support"

**编译流程**:
```bash
# 1. 克隆仓库
git clone --recurse-submodules https://github.com/NVIDIAGameWorks/Falcor.git
cd Falcor

# 2. 安装依赖 (Ubuntu 22.04)
sudo apt install xorg-dev libgtk-3-dev

# 3. 运行环境脚本
./setup.sh

# 4. CMake构建 (使用Linux预设)
cmake -B build -S . --preset=linux-gcc
cmake --build build --config Release

# 5. 运行测试
./build/bin/Release/Mogwai  # Falcor的可视化编辑器
```

**GPU要求**:
- NVIDIA RTX系列 (DXR capable)
- 驱动 466.11+ (Linux nouveau或proprietary)

### 自定义Render Pass开发

**创建PG-GCPL Render Pass** ([官方教程](https://github.com/NVIDIAGameWorks/Falcor/blob/master/docs/tutorials/02-implementing-a-render-pass.md)):

```bash
# 自动生成pass框架
tools/make_new_render_pass.bat PGGCPLProbeDecompression
```

生成文件结构:
```
Source/RenderPasses/PGGCPLProbeDecompression/
├── PGGCPLProbeDecompression.h       # C++接口
├── PGGCPLProbeDecompression.cpp     # 实现
├── ProbeDecompress.slang            # GPU着色器
└── CMakeLists.txt
```

**核心实现** (C++):
```cpp
// PGGCPLProbeDecompression.h
class PGGCPLProbeDecompression : public RenderPass
{
public:
    // 资源声明
    static void reflect(RenderPassReflection& reflector);

    // 执行解压
    void execute(RenderContext* pContext, const RenderData& renderData) override;

private:
    // 模型参数
    ref<Buffer> mpGaussianCenters;   // [30×3] float3
    ref<Buffer> mpLowRankU;          // [30×27×8] half
    ref<Buffer> mpTimeCoeffs;        // [30×8×7] half
};

// reflect()方法声明输入输出
void PGGCPLProbeDecompression::reflect(RenderPassReflection& reflector)
{
    reflector.addInput("sceneParams", "Scene parameters");
    reflector.addOutput("probes", "Decompressed SH coefficients")
             .format(ResourceFormat::RGBA32Float)
             .bindFlags(BindFlags::UnorderedAccess);
}

// execute()执行GPU计算
void PGGCPLProbeDecompression::execute(RenderContext* pContext,
                                        const RenderData& renderData)
{
    auto pProbesOutput = renderData.getTexture("probes");

    // 设置shader参数
    mpDecompressPass["gGaussianCenters"] = mpGaussianCenters;
    mpDecompressPass["gLowRankU"] = mpLowRankU;
    mpDecompressPass["gTimeCoeffs"] = mpTimeCoeffs;

    // 调度compute shader
    mpDecompressPass->execute(pContext, uint3(probeCount, 1, 1));
}
```

**GPU着色器** (Slang/HLSL):
```cpp
// ProbeDecompress.slang
StructuredBuffer<float3> gGaussianCenters;
StructuredBuffer<half> gLowRankU;           // [30×27×8]
StructuredBuffer<half> gTimeCoeffs;         // [30×8×7]
RWTexture2D<float4> gProbesOutput;

[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint probeIdx = threadId.x;

    // 1. 查询Top-K高斯
    float3 probePos = getProbePosition(probeIdx);
    float3 weights; uint3 indices;
    queryTopKGaussians(probePos, weights, indices);

    // 2. 低秩重建
    float sh_total[27] = {0};
    for (uint k = 0; k < 3; k++) {
        float sh_k[27];
        lowRankReconstruction(indices[k], sh_k);
        for (uint i = 0; i < 27; i++)
            sh_total[i] += weights[k] * sh_k[i];
    }

    // 3. 写回
    gProbesOutput[uint2(probeIdx, 0)] = float4(sh_total[0:3], ...);
}
```

**注册插件**:
```cpp
extern "C" FALCOR_API_EXPORT void registerPlugin(PluginRegistry& registry)
{
    registry.registerClass<RenderPass, PGGCPLProbeDecompression>();
}
```

### 光探针GI实现现状

**搜索结果**: Falcor **未内置**光探针GI pass ([搜索来源](https://dl.acm.org/doi/10.1145/3550454.3555452))

**解决方案**:
- 自定义render pass (如上所示)
- 参考学术论文实现: ["Efficient Light Probes for Real-Time Global Illumination"](https://dl.acm.org/doi/abs/10.1145/3550454.3555452) (使用Falcor开发)

### 集成优势
- ✅ **Linux原生支持** (实验性但可用)
- ✅ **Vulkan后端** (无需DX12)
- ✅ **模块化设计** (render pass即插即拔)
- ✅ **Python脚本支持** (可用Python定义render graph)
- ✅ **活跃社区** (NVIDIA持续维护)

### 集成劣势
- 需学习C++/Slang (从Python转换)
- 无现成光探针pass (需自己实现,但有教程)
- 实验性Linux支持 (可能有小bug)

### 实施路线图

**阶段1: 环境验证** (2天)
```bash
# 验证GPU兼容性
nvidia-smi  # 确认RTX GPU + 驱动版本

# 编译Falcor
git clone --recurse-submodules https://github.com/NVIDIAGameWorks/Falcor.git
cd Falcor && ./setup.sh
cmake -B build --preset=linux-gcc
cmake --build build --config Release -j$(nproc)

# 运行示例
./build/bin/Release/Mogwai
```

**阶段2: 简单Pass验证** (3天)
- 创建测试pass (输入纹理 → 简单处理 → 输出)
- 熟悉render graph编辑器 (Mogwai)
- 验证资源传递机制

**阶段3: PG-GCPL数据导入** (5天)
- 模型参数转换 (.npz → Buffer)
- 创建`PGGCPLProbeDecompression` pass
- 实现基础低秩重建shader

**阶段4: 性能基准测试** (3天)
- 集成到简单场景
- 测量解压延迟 (GPU profiler)
- 对比光追基线

**阶段5: 完整集成** (1-2周)
- 替换场景GI系统
- 端到端渲染验证
- SSIM/PSNR质量评估

**总计**: 4-5周 (含学习曲线)

---

## 方案3: 独立GPU Benchmark ⭐⭐⭐⭐⭐ (最快)

### 方案概述
不依赖渲染引擎,直接用PyTorch CUDA测量核心操作延迟

### 实施方案

**创建benchmark脚本** (`scripts/benchmark_gpu_decompression.py`):
```python
import torch
import torch.profiler as profiler
import time

class PGGCPLBenchmark:
    def __init__(self, K=30, rank=8):
        self.K = K
        self.rank = rank

        # 加载模型参数到GPU
        checkpoint = torch.load('checkpoints/best_K30_r8.pth')
        self.gaussian_centers = checkpoint['centers'].cuda()     # [30, 3]
        self.low_rank_U = checkpoint['U'].cuda()                 # [30, 27, 8]
        self.time_coeffs = checkpoint['coeffs'].cuda()           # [30, 8, 7]

    @torch.cuda.amp.autocast()  # 混合精度
    @torch.no_grad()
    def decompress_probes(self, probe_positions, physics_basis):
        """
        Args:
            probe_positions: [N, 3] 探针位置
            physics_basis: [7] 物理基函数 [cos(θ), sin(θ), ...]
        Returns:
            sh_coeffs: [N, 27] 重建的SH系数
        """
        N = probe_positions.shape[0]

        # 1. Top-K高斯查询 (批量化KNN)
        dists = torch.cdist(probe_positions, self.gaussian_centers)  # [N, 30]
        weights, indices = torch.topk(dists, k=3, largest=False)     # [N, 3]
        weights = torch.softmax(-weights / 0.1, dim=-1)              # Gaussian核

        # 2. 低秩重建 (融合运算)
        U_selected = self.low_rank_U[indices]                        # [N, 3, 27, 8]
        coeffs_selected = self.time_coeffs[indices]                  # [N, 3, 8, 7]

        time_features = coeffs_selected @ physics_basis              # [N, 3, 8]
        sh_per_gaussian = torch.einsum('nkdr,nkr->nkd',
                                        U_selected, time_features)    # [N, 3, 27]

        # 3. 加权求和
        sh_coeffs = torch.einsum('nk,nkd->nd',
                                  weights, sh_per_gaussian)          # [N, 27]

        return sh_coeffs

def benchmark_latency():
    """测量单次查询延迟"""
    bench = PGGCPLBenchmark()

    # 预热GPU
    for _ in range(100):
        probe_pos = torch.randn(1, 3).cuda()
        physics_basis = torch.randn(7).cuda()
        _ = bench.decompress_probes(probe_pos, physics_basis)

    # 基准测试 (1000次重复)
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(1000):
        probe_pos = torch.randn(1, 3).cuda()
        physics_basis = torch.randn(7).cuda()
        _ = bench.decompress_probes(probe_pos, physics_basis)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    latency_ms = (elapsed / 1000) * 1000
    print(f"单探针解压延迟: {latency_ms:.3f} ms")
    print(f"吞吐量: {1000/latency_ms:.0f} queries/ms")

def benchmark_batch():
    """测量批量处理吞吐"""
    bench = PGGCPLBenchmark()

    for batch_size in [1, 10, 100, 1000, 10000]:
        probe_pos = torch.randn(batch_size, 3).cuda()
        physics_basis = torch.randn(7).cuda()

        # 预热
        for _ in range(10):
            _ = bench.decompress_probes(probe_pos, physics_basis)

        # 测量
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(100):
            _ = bench.decompress_probes(probe_pos, physics_basis)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        throughput = (batch_size * 100) / elapsed
        print(f"Batch {batch_size:5d}: {throughput/1e6:.2f} M probes/sec")

def profile_kernels():
    """详细kernel分析"""
    bench = PGGCPLBenchmark()
    probe_pos = torch.randn(1000, 3).cuda()
    physics_basis = torch.randn(7).cuda()

    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        for _ in range(10):
            _ = bench.decompress_probes(probe_pos, physics_basis)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == '__main__':
    benchmark_latency()
    benchmark_batch()
    profile_kernels()
```

### 预期结果
```
单探针解压延迟: 0.052 ms  (目标 <0.5ms ✅)
吞吐量: 19230 queries/ms

Batch     1: 0.02 M probes/sec
Batch    10: 0.18 M probes/sec
Batch   100: 1.45 M probes/sec
Batch  1000: 8.21 M probes/sec
Batch 10000: 12.3 M probes/sec

CUDA Kernel分析:
-------------------------------------
  Name                  | Time (ms) | %
-------------------------------------
  aten::mm             |   0.234   | 45%
  aten::cdist          |   0.156   | 30%
  aten::einsum         |   0.089   | 17%
  aten::topk           |   0.034   |  7%
-------------------------------------
```

### 优势
- ✅ **最快验证** (1周内完成)
- ✅ **零依赖** (纯PyTorch)
- ✅ **跨平台** (Linux/Windows)
- ✅ **精准分析** (kernel级性能)

### 劣势
- ❌ 无端到端渲染验证
- ❌ 缺少真实场景上下文

---

## 推荐方案

### 短期 (1-2周): 独立GPU Benchmark ⭐⭐⭐⭐⭐
**目标**: 快速验证解压延迟达标 (<0.5ms)

**行动**:
1. 实现`benchmark_gpu_decompression.py`
2. 测量单探针延迟、批量吞吐、kernel分析
3. 对比任务书目标 (0.5ms)
4. 撰写性能报告

**交付物**:
- 基准测试脚本
- 性能数据表 (延迟/吞吐/GPU利用率)
- 瓶颈分析报告

---

### 中长期 (4-5周): Falcor集成 ⭐⭐⭐⭐
**目标**: 端到端渲染管线验证

**前置条件**:
- 独立benchmark已通过
- 确认有Ubuntu 22.04 + RTX GPU环境

**行动**:
1. 环境搭建 (编译Falcor)
2. 创建`PGGCPLProbeDecompression` render pass
3. 集成到简单测试场景
4. 端到端SSIM/PSNR验证

**交付物**:
- 自定义render pass源码
- 集成测试报告 (性能+质量)
- 渲染对比视频

---

## 不推荐: AMD Capsaicin (除非有Windows环境)

仅当满足以下条件才考虑:
- 有Windows 10 + DX12 GPU可用
- 愿意学习C++/HLSL
- 需要与AMD GI-1.0直接对比

---

## 参考资源

### Nvidia Falcor
- **仓库**: https://github.com/NVIDIAGameWorks/Falcor
- **教程**: [实现Render Pass](https://github.com/NVIDIAGameWorks/Falcor/blob/master/docs/tutorials/02-implementing-a-render-pass.md)
- **Falcor 7.0教程** (社区): https://github.com/yijie21/Falcor-7.0-Tutorial
- **学术论文**: [Efficient Light Probes for Real-Time GI](https://dl.acm.org/doi/abs/10.1145/3550454.3555452)

### AMD Capsaicin
- **仓库**: https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin
- **技术论文**: [GI-1.0 PDF](https://gpuopen.com/download/publications/GPUOpen2022_GI1_0.pdf)
- **框架文档**: https://gpuopen.com/capsaicin/

### 独立Benchmark
- **PyTorch Profiler**: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- **CUDA性能指南**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

---

**下一步**: 请确认优先选择哪个方案 (建议从独立GPU Benchmark开始)
