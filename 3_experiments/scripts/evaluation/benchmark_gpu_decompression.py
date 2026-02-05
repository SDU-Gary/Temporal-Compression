"""
GPU性能基准测试: PG-GCPL实时解压性能验证

目标:
  - 单探针延迟 P95 < 0.5ms
  - 批量吞吐 > 1M probes/sec
  - Kernel级性能分析
  - GPU内存占用评估

Author: Claude Code
Date: 2026-01-04
"""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.profiler
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple

from models.gaussian_physics_5D import GaussianPhysicsCompression5D


# ============================================================================
# 1. 单探针延迟测试
# ============================================================================

@torch.no_grad()
def benchmark_single_probe_latency(
    model: GaussianPhysicsCompression5D,
    device: torch.device,
    n_samples: int = 1000,
    warmup_runs: int = 100
) -> Dict[str, float]:
    """
    单探针查询延迟测试

    Args:
        model: 训练好的模型
        device: 计算设备
        n_samples: 测试样本数
        warmup_runs: GPU预热次数

    Returns:
        metrics: {p50, p90, p95, p99, mean, std, min, max, latencies}
    """
    model.eval()

    print(f"  GPU预热 ({warmup_runs} runs)...", end='', flush=True)
    for _ in range(warmup_runs):
        probe_pos = torch.randn(1, 3, device=device)
        light_params = torch.randn(1, 5, device=device)
        _ = model(probe_pos, light_params)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(" 完成")

    print(f"  测量 {n_samples} 个单探针查询...", end='', flush=True)
    latencies = []

    for _ in range(n_samples):
        probe_pos = torch.randn(1, 3, device=device) * 2 - 1  # [-1, 1]
        light_params = torch.tensor([[
            np.random.uniform(0, 90),      # zenith
            np.random.uniform(0, 360),     # azimuth
            np.random.uniform(0.5, 1.5),   # intensity
            np.random.uniform(3000, 7000), # color_temp
            np.random.uniform(0, 0.8)      # cloud_cover
        ]], device=device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(probe_pos, light_params)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # ms

    print(" 完成")

    latencies = np.array(latencies)

    metrics = {
        'latencies': latencies.tolist(),
        'mean': float(np.mean(latencies)),
        'std': float(np.std(latencies)),
        'min': float(np.min(latencies)),
        'max': float(np.max(latencies)),
        'p50': float(np.percentile(latencies, 50)),
        'p90': float(np.percentile(latencies, 90)),
        'p95': float(np.percentile(latencies, 95)),
        'p99': float(np.percentile(latencies, 99)),
        'throughput_qps': 1000.0 / float(np.mean(latencies))
    }

    return metrics


# ============================================================================
# 2. 批量吞吐测试
# ============================================================================

@torch.no_grad()
def benchmark_batch_throughput(
    model: GaussianPhysicsCompression5D,
    device: torch.device
) -> Dict[int, float]:
    """
    批量处理吞吐测试

    Args:
        model: 训练好的模型
        device: 计算设备

    Returns:
        throughput_results: {batch_size: throughput_M_per_sec}
    """
    model.eval()

    batch_sizes = [1, 10, 100, 1000, 10000]
    results = {}

    print(f"  测试批量吞吐 (batch sizes: {batch_sizes})...")

    for batch_size in batch_sizes:
        probe_pos = torch.randn(batch_size, 3, device=device) * 2 - 1
        light_params = torch.randn(batch_size, 5, device=device)

        # 预热
        for _ in range(10):
            _ = model(probe_pos, light_params)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # 测量
        n_repeats = max(100, 10000 // batch_size)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_repeats):
            _ = model(probe_pos, light_params)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        throughput = (batch_size * n_repeats) / elapsed / 1e6  # M/sec
        results[batch_size] = throughput

        print(f"    Batch {batch_size:5d}: {throughput:.2f} M probes/sec")

    return results


# ============================================================================
# 3. Kernel级性能分析
# ============================================================================

def benchmark_kernel_profiling(
    model: GaussianPhysicsCompression5D,
    device: torch.device
) -> List[Dict[str, float]]:
    """
    PyTorch Profiler分析

    Args:
        model: 训练好的模型
        device: 计算设备

    Returns:
        top_kernels: List[{name, cuda_time_ms, percentage}]
    """
    model.eval()

    print(f"  运行PyTorch Profiler...", end='', flush=True)

    probe_pos = torch.randn(1000, 3, device=device) * 2 - 1
    light_params = torch.randn(1000, 5, device=device)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA] if device.type == 'cuda'
                   else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=False
    ) as prof:
        with torch.no_grad():
            for _ in range(10):
                _ = model(probe_pos, light_params)

    print(" 完成")

    # 解析top-10 kernels
    events = prof.key_averages()

    if device.type == 'cuda':
        sorted_events = sorted(events, key=lambda x: x.cuda_time_total, reverse=True)
        total_time = sum(e.cuda_time_total for e in events)

        top_kernels = [
            {
                'name': e.key,
                'cuda_time_ms': e.cuda_time_total / 1000.0,
                'percentage': (e.cuda_time_total / total_time * 100) if total_time > 0 else 0
            }
            for e in sorted_events[:10]
        ]
    else:
        sorted_events = sorted(events, key=lambda x: x.cpu_time_total, reverse=True)
        total_time = sum(e.cpu_time_total for e in events)

        top_kernels = [
            {
                'name': e.key,
                'cpu_time_ms': e.cpu_time_total / 1000.0,
                'percentage': (e.cpu_time_total / total_time * 100) if total_time > 0 else 0
            }
            for e in sorted_events[:10]
        ]

    return top_kernels


# ============================================================================
# 4. 内存占用分析
# ============================================================================

@torch.no_grad()
def benchmark_memory_usage(
    model: GaussianPhysicsCompression5D,
    device: torch.device
) -> Dict[str, float]:
    """
    GPU内存占用分析

    Args:
        model: 训练好的模型
        device: 计算设备

    Returns:
        memory_stats: {model_mb, peak_mb, batch_1k_mb, batch_10k_mb}
    """
    if device.type != 'cuda':
        print("  跳过内存分析 (CPU模式)")
        return {'model_mb': 0, 'peak_mb': 0, 'batch_1k_mb': 0, 'batch_10k_mb': 0}

    model.eval()

    print(f"  分析GPU内存占用...", end='', flush=True)

    # 模型参数内存
    model_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    # Batch 1000测试
    torch.cuda.reset_peak_memory_stats(device)
    probe_pos = torch.randn(1000, 3, device=device)
    light_params = torch.randn(1000, 5, device=device)
    _ = model(probe_pos, light_params)
    batch_1k_mem = torch.cuda.max_memory_allocated(device) / 1e6

    # Batch 10000测试
    torch.cuda.reset_peak_memory_stats(device)
    probe_pos = torch.randn(10000, 3, device=device)
    light_params = torch.randn(10000, 5, device=device)
    _ = model(probe_pos, light_params)
    batch_10k_mem = torch.cuda.max_memory_allocated(device) / 1e6

    print(" 完成")

    return {
        'model_mb': model_mem,
        'peak_mb': torch.cuda.max_memory_allocated(device) / 1e6,
        'batch_1k_mb': batch_1k_mem,
        'batch_10k_mb': batch_10k_mem
    }


# ============================================================================
# 5. 可视化
# ============================================================================

def visualize_results(
    latency_metrics: Dict,
    throughput_results: Dict[int, float],
    kernel_results: List[Dict],
    memory_results: Dict,
    output_dir: Path,
    device_name: str
):
    """
    生成可视化图表

    Args:
        latency_metrics: 延迟测试结果
        throughput_results: 吞吐测试结果
        kernel_results: Kernel分析结果
        memory_results: 内存分析结果
        output_dir: 输出目录
        device_name: 设备名称
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))

    # 1. 延迟分布直方图
    ax1 = plt.subplot(2, 3, 1)
    latencies = np.array(latency_metrics['latencies'])
    ax1.hist(latencies, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(0.5, color='r', linestyle='--', linewidth=2, label='Target P95=0.5ms')
    ax1.axvline(latency_metrics['p95'], color='g', linestyle='--', linewidth=2,
                label=f'Actual P95={latency_metrics["p95"]:.3f}ms')
    ax1.set_xlabel('Latency (ms)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Single Probe Latency Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # 2. 延迟Box Plot
    ax2 = plt.subplot(2, 3, 2)
    ax2.boxplot([latencies], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.axhline(0.5, color='r', linestyle='--', linewidth=2, label='Target=0.5ms')
    ax2.set_ylabel('Latency (ms)', fontsize=12)
    ax2.set_title('Latency Statistics', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(['All Queries'])
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='y')

    # 3. 吞吐扩展性
    ax3 = plt.subplot(2, 3, 3)
    batch_sizes = sorted(throughput_results.keys())
    throughputs = [throughput_results[bs] for bs in batch_sizes]
    ax3.plot(batch_sizes, throughputs, marker='o', linewidth=2, markersize=8, color='green')
    ax3.axhline(1.0, color='r', linestyle='--', linewidth=2, label='Target=1M/sec')
    ax3.set_xlabel('Batch Size', fontsize=12)
    ax3.set_ylabel('Throughput (M probes/sec)', fontsize=12)
    ax3.set_title('Batch Throughput Scaling', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3, which='both')

    # 4. Kernel时间占比
    ax4 = plt.subplot(2, 3, 4)
    if kernel_results:
        top_5 = kernel_results[:5]
        names = [k['name'].split('::')[-1][:20] for k in top_5]  # 简化名称
        percentages = [k['percentage'] for k in top_5]
        colors = plt.cm.Spectral(np.linspace(0.2, 0.8, len(names)))
        ax4.barh(names, percentages, color=colors, alpha=0.8)
        ax4.set_xlabel('Time Percentage (%)', fontsize=12)
        ax4.set_title('Top-5 GPU Kernels', fontsize=14, fontweight='bold')
        ax4.grid(alpha=0.3, axis='x')
    else:
        ax4.text(0.5, 0.5, 'No kernel data', ha='center', va='center', fontsize=14)
        ax4.axis('off')

    # 5. 内存占用
    ax5 = plt.subplot(2, 3, 5)
    if memory_results['model_mb'] > 0:
        mem_labels = ['Model\nParams', 'Batch\n1K', 'Batch\n10K']
        mem_values = [
            memory_results['model_mb'],
            memory_results['batch_1k_mb'],
            memory_results['batch_10k_mb']
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax5.bar(mem_labels, mem_values, color=colors, alpha=0.8)
        ax5.axhline(500, color='r', linestyle='--', linewidth=2, label='Target=500MB')
        ax5.set_ylabel('Memory (MB)', fontsize=12)
        ax5.set_title('GPU Memory Usage', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(alpha=0.3, axis='y')

        # 添加数值标签
        for bar, value in zip(bars, mem_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10)
    else:
        ax5.text(0.5, 0.5, 'CPU mode\n(No GPU memory)', ha='center', va='center', fontsize=14)
        ax5.axis('off')

    # 6. 统计摘要
    ax6 = plt.subplot(2, 3, 6)
    success_latency = latency_metrics['p95'] < 0.5
    success_throughput = max(throughput_results.values()) > 1.0
    success_memory = memory_results['batch_10k_mb'] < 500 if memory_results['model_mb'] > 0 else True

    summary_text = f"""GPU Benchmark Summary
{'='*40}

Device: {device_name}

[1/4] Latency (Single Probe)
  P50:  {latency_metrics['p50']:.4f} ms
  P90:  {latency_metrics['p90']:.4f} ms
  P95:  {latency_metrics['p95']:.4f} ms  {'✓' if success_latency else '✗'}
  P99:  {latency_metrics['p99']:.4f} ms
  Mean: {latency_metrics['mean']:.4f} ± {latency_metrics['std']:.4f} ms
  QPS:  {latency_metrics['throughput_qps']:.0f} queries/sec

[2/4] Throughput (Batch)
  Max:  {max(throughput_results.values()):.2f} M/sec  {'✓' if success_throughput else '✗'}
  Target: >1.0 M/sec

[3/4] Memory (GPU)
  Peak: {memory_results['batch_10k_mb']:.1f} MB  {'✓' if success_memory else '✗'}
  Target: <500 MB

{'='*40}
VERDICT: {'✓ ALL TARGETS MET' if all([success_latency, success_throughput, success_memory]) else '✗ SOME TARGETS MISSED'}
"""

    ax6.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax6.transAxes)
    ax6.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_results.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存: {output_dir / 'benchmark_results.png'}")
    plt.close()


# ============================================================================
# 6. 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GPU decompression benchmark")
    parser.add_argument("--checkpoint", default="PG-GCPL/experiments/04_ablation_and_visualization/ablation/K30_r8/best_model.pt")
    parser.add_argument("--output", default="experiments/gpu_benchmark")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold-latency-ms", type=float, default=0.5)
    parser.add_argument("--threshold-throughput-m", type=float, default=1.0)
    parser.add_argument("--latency-samples", type=int, default=1000)
    parser.add_argument("--warmup-runs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output)
    threshold_latency_ms = args.threshold_latency_ms
    threshold_throughput_M = args.threshold_throughput_m

    print("=" * 80)
    print("GPU性能基准测试: PG-GCPL实时解压性能验证")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print()

    # 1. 加载模型
    print("[1/5] 加载模型...")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint不存在: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 从checkpoint推断K和rank
    K = checkpoint['model_state_dict']['mu'].shape[0]
    rank = checkpoint['model_state_dict']['U'].shape[2]

    model = GaussianPhysicsCompression5D(num_gaussians=K, rank=rank, sh_dim=27)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    print(f"  Model: GaussianPhysicsCompression5D")
    print(f"  K={K}, rank={rank}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Compression: {checkpoint.get('compression_ratio', 'N/A'):.2f}×")
    print()

    # 2. 单探针延迟测试
    print("[2/5] 单探针延迟测试...")
    latency_metrics = benchmark_single_probe_latency(
        model,
        device,
        n_samples=args.latency_samples,
        warmup_runs=args.warmup_runs,
    )
    print(f"  P95: {latency_metrics['p95']:.4f} ms (目标: <{threshold_latency_ms}ms)")
    print()

    # 3. 批量吞吐测试
    print("[3/5] 批量吞吐测试...")
    throughput_results = benchmark_batch_throughput(model, device)
    print()

    # 4. Kernel性能分析
    print("[4/5] Kernel性能分析...")
    kernel_results = benchmark_kernel_profiling(model, device)
    print()

    # 5. 内存占用分析
    print("[5/5] 内存占用分析...")
    memory_results = benchmark_memory_usage(model, device)
    print()

    # 生成报告
    print("=" * 80)
    print("基准测试结果摘要")
    print("=" * 80)

    # 延迟
    success_latency = latency_metrics['p95'] < threshold_latency_ms
    print(f"\n[1/4] 单探针延迟")
    print(f"  P50:  {latency_metrics['p50']:.4f} ms")
    print(f"  P90:  {latency_metrics['p90']:.4f} ms")
    print(f"  P95:  {latency_metrics['p95']:.4f} ms  {'✓' if success_latency else '✗'} (target <{threshold_latency_ms}ms)")
    print(f"  P99:  {latency_metrics['p99']:.4f} ms")
    print(f"  Mean: {latency_metrics['mean']:.4f} ± {latency_metrics['std']:.4f} ms")
    print(f"  Throughput: {latency_metrics['throughput_qps']:.0f} queries/sec")

    # 吞吐
    success_throughput = max(throughput_results.values()) > threshold_throughput_M
    print(f"\n[2/4] 批量吞吐")
    for batch_size, throughput in sorted(throughput_results.items()):
        marker = '✓' if throughput > threshold_throughput_M else ' '
        print(f"  Batch {batch_size:5d}: {throughput:6.2f} M probes/sec  {marker}")
    print(f"  Max: {max(throughput_results.values()):.2f} M/sec  {'✓' if success_throughput else '✗'} (target >{threshold_throughput_M}M/sec)")

    # Kernel
    print(f"\n[3/4] Kernel性能分析 (Top-5)")
    for i, kernel in enumerate(kernel_results[:5], 1):
        time_key = 'cuda_time_ms' if 'cuda_time_ms' in kernel else 'cpu_time_ms'
        print(f"  {i}. {kernel['name'][:50]:50s}  {kernel['percentage']:5.1f}%  ({kernel[time_key]:.3f} ms)")

    # 内存
    success_memory = memory_results['batch_10k_mb'] < 500 if memory_results['model_mb'] > 0 else True
    print(f"\n[4/4] 内存占用")
    if memory_results['model_mb'] > 0:
        print(f"  Model parameters:  {memory_results['model_mb']:6.1f} MB")
        print(f"  Peak (batch 1k):   {memory_results['batch_1k_mb']:6.1f} MB")
        print(f"  Peak (batch 10k):  {memory_results['batch_10k_mb']:6.1f} MB  {'✓' if success_memory else '✗'} (target <500MB)")
    else:
        print(f"  CPU模式 (无GPU内存分析)")

    # 总结
    print("\n" + "=" * 80)
    if all([success_latency, success_throughput, success_memory]):
        print("✓ ALL TARGETS MET - Ready for real-time integration")
    else:
        print("✗ SOME TARGETS MISSED - Review results for optimization opportunities")
    print("=" * 80)

    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'model': {
            'name': 'GaussianPhysicsCompression5D',
            'K': K,
            'rank': rank,
            'total_params': total_params,
            'compression_ratio': float(checkpoint.get('compression_ratio', 0))
        },
        'latency': latency_metrics,
        'throughput': {str(k): v for k, v in throughput_results.items()},
        'kernels': kernel_results,
        'memory': memory_results,
        'success': {
            'latency': success_latency,
            'throughput': success_throughput,
            'memory': success_memory,
            'overall': all([success_latency, success_throughput, success_memory])
        }
    }

    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 可视化
    visualize_results(
        latency_metrics,
        throughput_results,
        kernel_results,
        memory_results,
        output_dir,
        str(device)
    )

    print(f"\n结果已保存至: {output_dir}")
    print(f"  - benchmark_results.json")
    print(f"  - benchmark_results.png")


if __name__ == '__main__':
    main()
