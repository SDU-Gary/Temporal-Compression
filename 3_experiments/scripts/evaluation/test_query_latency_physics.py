"""
Week 4.3: Physics-only查询延迟测试

目标：验证Physics-only模型的实时查询性能
成功标准：P95延迟 < 0.5ms
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import time
from typing import Dict, List
import json
import matplotlib.pyplot as plt
from datetime import datetime

from models.physics_low_rank import PhysicsLowRank5D


def sample_random_configs(n_samples: int = 100, seed: int = 42) -> np.ndarray:
    """
    生成随机5D光源配置（用于延迟测试）

    Args:
        n_samples: 采样数量
        seed: 随机种子

    Returns:
        configs [N, 5]: [zenith, azimuth, intensity, temp, cloud]
    """
    np.random.seed(seed)

    configs = []
    for _ in range(n_samples):
        config = [
            np.random.uniform(10.0, 80.0),      # zenith
            np.random.uniform(0.0, 360.0),      # azimuth
            np.random.uniform(0.5, 1.5),        # intensity
            np.random.uniform(3000, 7000),      # color_temp
            np.random.uniform(0.0, 0.8)         # cloud_cover
        ]
        configs.append(config)

    return np.array(configs)


def measure_query_latency(
    model: PhysicsLowRank5D,
    test_configs: np.ndarray,
    device: torch.device,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """
    测量查询延迟

    Args:
        model: 训练好的Physics模型
        test_configs: [N, 5] 测试配置
        device: 计算设备
        warmup_runs: GPU预热次数

    Returns:
        metrics: 延迟统计 (单位: ms)
    """
    model.eval()

    # 预热GPU
    print(f"  GPU预热 ({warmup_runs} runs)...", end='', flush=True)
    with torch.no_grad():
        for i in range(warmup_runs):
            dummy_config = torch.randn(1, 5).to(device)
            _ = model(dummy_config)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(" 完成")

    # 实际测量
    print(f"  测量 {len(test_configs)} 个查询...", end='', flush=True)
    latencies = []

    with torch.no_grad():
        for config in test_configs:
            config_t = torch.from_numpy(config).float().unsqueeze(0).to(device)

            # 计时开始
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # 查询
            _ = model(config_t)

            # 计时结束
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000.0
            latencies.append(latency_ms)

    print(" 完成")

    latencies = np.array(latencies)

    # 统计
    metrics = {
        'latencies_ms': latencies.tolist(),
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p90_ms': float(np.percentile(latencies, 90)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'device': str(device)
    }

    return metrics


def visualize_latency_results(
    metrics: Dict[str, float],
    output_dir: Path,
    threshold_ms: float = 0.5
):
    """可视化延迟测试结果"""
    output_dir.mkdir(parents=True, exist_ok=True)

    latencies = np.array(metrics['latencies_ms'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 延迟分布直方图
    axes[0, 0].hist(latencies, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(threshold_ms, color='r', linestyle='--',
                       label=f'Target P95={threshold_ms:.2f}ms')
    axes[0, 0].axvline(metrics['p95_ms'], color='g', linestyle='--',
                       label=f'Actual P95={metrics["p95_ms"]:.3f}ms')
    axes[0, 0].set_xlabel('Latency (ms)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Query Latency Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. 时间序列
    axes[0, 1].plot(latencies, alpha=0.6, linewidth=0.8)
    axes[0, 1].axhline(threshold_ms, color='r', linestyle='--',
                       label=f'Target={threshold_ms:.2f}ms')
    axes[0, 1].axhline(metrics['mean_ms'], color='g', linestyle='--',
                       label=f'Mean={metrics["mean_ms"]:.3f}ms')
    axes[0, 1].set_xlabel('Query Index')
    axes[0, 1].set_ylabel('Latency (ms)')
    axes[0, 1].set_title('Latency Time Series')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. Box plot
    box_data = [latencies]
    axes[1, 0].boxplot(box_data, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1, 0].axhline(threshold_ms, color='r', linestyle='--',
                       label=f'Target={threshold_ms:.2f}ms')
    axes[1, 0].set_ylabel('Latency (ms)')
    axes[1, 0].set_title('Latency Distribution (Box Plot)')
    axes[1, 0].set_xticklabels(['All Queries'])
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis='y')

    # 4. 统计信息
    success = metrics['p95_ms'] < threshold_ms
    stats_text = f"""延迟统计 ({metrics['device']}):

Percentiles:
  P50: {metrics['p50_ms']:.4f} ms
  P90: {metrics['p90_ms']:.4f} ms
  P95: {metrics['p95_ms']:.4f} ms
  P99: {metrics['p99_ms']:.4f} ms

Statistics:
  Mean: {metrics['mean_ms']:.4f} ms
  Std:  {metrics['std_ms']:.4f} ms
  Min:  {metrics['min_ms']:.4f} ms
  Max:  {metrics['max_ms']:.4f} ms

Target: P95 < {threshold_ms:.2f} ms
Result: {'PASS ✓' if success else 'FAIL ✗'}

吞吐量: {1000.0/metrics['mean_ms']:.0f} queries/sec
"""

    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'query_latency_analysis.png', dpi=150)
    print(f"可视化结果已保存至: {output_dir / 'query_latency_analysis.png'}")
    plt.close()


def main():
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path('archive/Week3-4_MLP_Hybrid_Failed/checkpoints/stage1_best.pt')
    output_dir = Path('experiments/week4_query_latency')
    n_queries = 100
    threshold_ms = 0.5

    print("=" * 80)
    print("Week 4.3: Physics-only查询延迟测试")
    print("=" * 80)

    # 1. 生成随机查询配置
    print(f"\n[1/4] 生成{n_queries}个随机查询配置...")
    test_configs = sample_random_configs(n_samples=n_queries, seed=42)
    print(f"配置范围:")
    print(f"  Zenith: [{test_configs[:, 0].min():.1f}, {test_configs[:, 0].max():.1f}] deg")
    print(f"  Azimuth: [{test_configs[:, 1].min():.1f}, {test_configs[:, 1].max():.1f}] deg")
    print(f"  Intensity: [{test_configs[:, 2].min():.2f}, {test_configs[:, 2].max():.2f}]")
    print(f"  Color Temp: [{test_configs[:, 3].min():.0f}, {test_configs[:, 3].max():.0f}] K")
    print(f"  Cloud: [{test_configs[:, 4].min():.2f}, {test_configs[:, 4].max():.2f}]")

    # 2. 加载模型
    print(f"\n[2/4] 加载Physics-only模型...")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint不存在: {checkpoint_path}")
        return

    model = PhysicsLowRank5D(rank=5)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract only physics components from hybrid checkpoint
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                      if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)

    print(f"模型已加载: {checkpoint_path}")
    print(f"  Device: {device}")
    print(f"  参数量: 170")

    # 3. 测量延迟
    print(f"\n[3/4] 测量查询延迟 (目标: P95 < {threshold_ms}ms)...")
    metrics = measure_query_latency(
        model=model,
        test_configs=test_configs,
        device=device,
        warmup_runs=10
    )

    # 结果
    print(f"\n延迟统计 ({metrics['device']}):")
    print(f"  P50: {metrics['p50_ms']:.4f} ms")
    print(f"  P90: {metrics['p90_ms']:.4f} ms")
    print(f"  P95: {metrics['p95_ms']:.4f} ms (目标: <{threshold_ms}ms)")
    print(f"  P99: {metrics['p99_ms']:.4f} ms")
    print(f"\n统计:")
    print(f"  Mean: {metrics['mean_ms']:.4f} ms")
    print(f"  Std: {metrics['std_ms']:.4f} ms")
    print(f"  Min: {metrics['min_ms']:.4f} ms")
    print(f"  Max: {metrics['max_ms']:.4f} ms")
    print(f"\n吞吐量: {1000.0/metrics['mean_ms']:.0f} queries/sec")

    # 4. 保存结果
    print(f"\n[4/4] 保存结果...")
    output_dir.mkdir(parents=True, exist_ok=True)

    success = metrics['p95_ms'] < threshold_ms

    results = {
        'timestamp': datetime.now().isoformat(),
        'device': metrics['device'],
        'n_queries': n_queries,
        'threshold_ms': threshold_ms,
        'p50_ms': metrics['p50_ms'],
        'p90_ms': metrics['p90_ms'],
        'p95_ms': metrics['p95_ms'],
        'p99_ms': metrics['p99_ms'],
        'mean_ms': metrics['mean_ms'],
        'std_ms': metrics['std_ms'],
        'min_ms': metrics['min_ms'],
        'max_ms': metrics['max_ms'],
        'throughput_qps': 1000.0 / metrics['mean_ms'],
        'success': success
    }

    with open(output_dir / 'latency_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 可视化
    visualize_latency_results(metrics, output_dir, threshold_ms)

    # Go/No-Go决策
    print("\n" + "=" * 80)
    if success:
        print(f"✓ 延迟测试通过 (P95 {metrics['p95_ms']:.4f}ms < {threshold_ms}ms)")
        print(f"  吞吐量: {1000.0/metrics['mean_ms']:.0f} queries/sec")
        print("  建议: Week 4验证完成, 可进入Week 5多探针架构")
    else:
        print(f"✗ 延迟测试失败 (P95 {metrics['p95_ms']:.4f}ms >= {threshold_ms}ms)")
        print("  建议: 优化推理路径或使用GPU加速")
    print("=" * 80)

    print(f"\n结果已保存至: {output_dir}")


if __name__ == '__main__':
    main()
