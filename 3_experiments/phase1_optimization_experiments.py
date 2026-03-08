#!/usr/bin/env python3
"""
Phase 1 优化实验：K30_r8 基线提升至 35+ dB sRGB PSNR

实验设计：
Stage 1: 色彩空间对齐 (4 experiments, sRGB loss)
Stage 2: 学习率与调度优化 (3 experiments, warmup + cosine)
Stage 3: 数据增强 (3 experiments, intensity/position/dropout)
Stage 4: 架构微调 (3 experiments, attention/residual/deeper FiLM)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_driver_common import modify_config, read_json_if_exists, run_command

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# Stage 1: 色彩空间对齐
STAGE1_EXPERIMENTS = {
    "E1a_srgb_lambda01": {
        "name": "E1a: sRGB λ=0.1",
        "base_config": "3_experiments/configs/bistro_clean_train.yaml",
        "modifications": {
            "model.num_gaussians": 30,
            "model.rank": 8,
            "training.lambda_image": 0.1,
            "training.image_samples": 512,
            "training.image_loss_space": "srgb",
            "training.image_loss_type": "charbonnier",
            "training.enable_weighted_sh_loss": True,
            "training.sh_loss_weights": [1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0],
            "training.sh_weight_mode": "basis",
            "experiment.output_dir": "3_experiments/results/bistro_clean_v2/phase1_E1a_srgb_lambda01",
        },
        "description": "sRGB image loss, λ=0.1, 512 samples"
    },
    "E1b_srgb_lambda05": {
        "name": "E1b: sRGB λ=0.5",
        "base_config": "3_experiments/configs/bistro_clean_train.yaml",
        "modifications": {
            "model.num_gaussians": 30,
            "model.rank": 8,
            "training.lambda_image": 0.5,
            "training.image_samples": 1024,
            "training.image_loss_space": "srgb",
            "training.image_loss_type": "charbonnier",
            "training.enable_weighted_sh_loss": True,
            "training.sh_loss_weights": [1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0],
            "training.sh_weight_mode": "basis",
            "experiment.output_dir": "3_experiments/results/bistro_clean_v2/phase1_E1b_srgb_lambda05",
        },
        "description": "sRGB image loss, λ=0.5, 1024 samples"
    },
    "E1c_srgb_lambda10": {
        "name": "E1c: sRGB λ=1.0",
        "base_config": "3_experiments/configs/bistro_clean_train.yaml",
        "modifications": {
            "model.num_gaussians": 30,
            "model.rank": 8,
            "training.lambda_image": 1.0,
            "training.image_samples": 1024,
            "training.image_loss_space": "srgb",
            "training.image_loss_type": "charbonnier",
            "training.enable_weighted_sh_loss": True,
            "training.sh_loss_weights": [1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0],
            "training.sh_weight_mode": "basis",
            "experiment.output_dir": "3_experiments/results/bistro_clean_v2/phase1_E1c_srgb_lambda10",
        },
        "description": "sRGB image loss, λ=1.0, 1024 samples"
    },
    "E1d_srgb_lambda20": {
        "name": "E1d: sRGB λ=2.0",
        "base_config": "3_experiments/configs/bistro_clean_train.yaml",
        "modifications": {
            "model.num_gaussians": 30,
            "model.rank": 8,
            "training.lambda_image": 2.0,
            "training.image_samples": 1024,
            "training.image_loss_space": "srgb",
            "training.image_loss_type": "charbonnier",
            "training.enable_weighted_sh_loss": True,
            "training.sh_loss_weights": [1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0],
            "training.sh_weight_mode": "basis",
            "experiment.output_dir": "3_experiments/results/bistro_clean_v2/phase1_E1d_srgb_lambda20",
        },
        "description": "sRGB image loss, λ=2.0, 1024 samples"
    },
}

# Stage 2: 学习率与调度优化 (需要 warmup 实现)
STAGE2_EXPERIMENTS = {
    "E2a_lr5e4_warmup100": {
        "name": "E2a: lr=5e-4, warmup=100",
        "base_config": None,  # 从 Stage 1 最佳配置继承
        "modifications": {
            "training.lr": 5e-4,
            "training.lr_scheduler": "cosine",
            "training.lr_min": 1e-5,
            "training.warmup_epochs": 100,
            "training.epochs": 3000,
            "experiment.output_dir": "3_experiments/results/bistro_clean_v2/phase1_E2a_lr5e4_warmup100",
        },
        "description": "lr=5e-4, cosine scheduler, warmup=100, epochs=3000"
    },
    "E2b_lr1e3_warmup100": {
        "name": "E2b: lr=1e-3, warmup=100",
        "base_config": None,
        "modifications": {
            "training.lr": 1e-3,
            "training.lr_scheduler": "cosine",
            "training.lr_min": 1e-5,
            "training.warmup_epochs": 100,
            "training.epochs": 3000,
            "experiment.output_dir": "3_experiments/results/bistro_clean_v2/phase1_E2b_lr1e3_warmup100",
        },
        "description": "lr=1e-3, cosine scheduler, warmup=100, epochs=3000"
    },
    "E2c_lr2e3_warmup200": {
        "name": "E2c: lr=2e-3, warmup=200",
        "base_config": None,
        "modifications": {
            "training.lr": 2e-3,
            "training.lr_scheduler": "cosine",
            "training.lr_min": 1e-5,
            "training.warmup_epochs": 200,
            "training.epochs": 3000,
            "experiment.output_dir": "3_experiments/results/bistro_clean_v2/phase1_E2c_lr2e3_warmup200",
        },
        "description": "lr=2e-3, cosine scheduler, warmup=200, epochs=3000"
    },
}

# Stage 3: 数据增强
STAGE3_EXPERIMENTS = {
    "E3a_aug_intensity": {
        "name": "E3a: Intensity augmentation",
        "base_config": None,
        "modifications": {
            "training.augment_intensity": True,
            "training.augment_position": False,
            "training.augment_dropout": 0.0,
            "experiment.output_dir": "3_experiments/results/bistro_clean_v2/phase1_E3a_aug_intensity",
        },
        "description": "仅强度抖动 (RGB × [0.9, 1.1])"
    },
    "E3b_aug_intensity_position": {
        "name": "E3b: Intensity + Position",
        "base_config": None,
        "modifications": {
            "training.augment_intensity": True,
            "training.augment_position": True,
            "training.augment_dropout": 0.0,
            "experiment.output_dir": "3_experiments/results/bistro_clean_v2/phase1_E3b_aug_intensity_position",
        },
        "description": "强度抖动 + 位置噪声 (σ=0.02)"
    },
    "E3c_aug_full": {
        "name": "E3c: Full augmentation",
        "base_config": None,
        "modifications": {
            "training.augment_intensity": True,
            "training.augment_position": True,
            "training.augment_dropout": 0.15,
            "experiment.output_dir": "3_experiments/results/bistro_clean_v2/phase1_E3c_aug_full",
        },
        "description": "全部增强 (强度 + 位置 + dropout 15%)"
    },
}

FALCOR_CONFIG = {
    "python_path": "/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python",
    "python_bin": "/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10",
}


def run_training(exp_id: str, config_file: str, output_dir: str) -> bool:
    """运行训练"""
    log_file = f"{output_dir}/training.log"

    cmd = [
        "python", "3_experiments/scripts/train.py",
        "--config", config_file
    ]

    returncode = run_command(cmd, f"Training {exp_id}", log_file)
    return returncode == 0


def run_evaluation(exp_id: str, output_dir: str) -> Dict[str, Any]:
    """运行评估并返回指标"""
    checkpoint = f"{output_dir}/best_model.pt"

    if not os.path.exists(checkpoint):
        print(f"⚠️  Checkpoint not found: {checkpoint}")
        return {}

    log_file = f"{output_dir}/evaluation.log"

    cmd = [
        "python", "3_experiments/scripts/eval.py",
        "--data-root", "1_data_generation/output/bistro_clean_v2",
        "--checkpoint", checkpoint,
        "--split", "test"
    ]

    returncode = run_command(cmd, f"Evaluation {exp_id}", log_file)

    if returncode != 0:
        print(f"⚠️  Evaluation failed for {exp_id}")
        return {}

    eval_json = f"{output_dir}/eval.json"
    return read_json_if_exists(eval_json, default={})


def run_benchmark(exp_id: str, output_dir: str) -> Dict[str, Any]:
    """运行 benchmark 并返回图像指标"""
    checkpoint = f"{output_dir}/best_model.pt"
    benchmark_dir = f"{output_dir}/benchmark"

    if not os.path.exists(checkpoint):
        print(f"⚠️  Checkpoint not found: {checkpoint}")
        return {}

    os.makedirs(benchmark_dir, exist_ok=True)
    log_file = f"{benchmark_dir}/benchmark.log"

    cmd = [
        "python", "tools/benchmark_realtime_pipeline.py",
        "--dataset", "1_data_generation/output/bistro_clean_v2",
        "--scene", "1_data_generation/falcor/scenes/bistro_exterior_emissive_spheres.pyscene",
        "--route", "both",
        "--checkpoint", checkpoint,
        "--output-dir", benchmark_dir,
        "--falcor-python-path", FALCOR_CONFIG["python_path"],
        "--falcor-python-bin", FALCOR_CONFIG["python_bin"],
        "--width", "1280",
        "--height", "720",
        "--warmup-frames", "4",
        "--benchmark-frames", "24",
        "--sync-every", "1",
        "--compute-image-metrics",
        "--save-frame-metrics-every", "1",
        "--cosine-mode", "irradiance"
    ]

    returncode = run_command(cmd, f"Benchmark {exp_id}", log_file)

    if returncode != 0:
        print(f"⚠️  Benchmark failed for {exp_id}")
        return {}

    summary_json = f"{benchmark_dir}/benchmark_summary.json"
    data = read_json_if_exists(summary_json, default={})
    return data.get("image_metrics", {}) if isinstance(data, dict) else {}


def run_single_experiment(exp_id: str, exp_config: Dict[str, Any]) -> Dict[str, Any]:
    """运行单个实验的完整流程"""
    print(f"\n{'='*80}")
    print(f"开始实验: {exp_config['name']}")
    print(f"描述: {exp_config['description']}")
    print(f"{'='*80}\n")

    # 准备配置文件
    temp_config = f"3_experiments/configs/temp_{exp_id}.yaml"
    config = modify_config(
        exp_config['base_config'],
        exp_config['modifications'],
        temp_config
    )

    output_dir = config['experiment']['output_dir']

    results = {
        "exp_id": exp_id,
        "name": exp_config['name'],
        "description": exp_config['description'],
        "config": exp_config['modifications'],
        "status": "started",
        "start_time": datetime.now().isoformat(),
    }

    # 训练
    print(f"\n[1/3] 开始训练...")
    training_success = run_training(exp_id, temp_config, output_dir)

    if not training_success:
        results["status"] = "training_failed"
        results["end_time"] = datetime.now().isoformat()
        return results

    # 评估
    print(f"\n[2/3] 开始评估...")
    eval_metrics = run_evaluation(exp_id, output_dir)
    results["eval_metrics"] = eval_metrics

    # Benchmark
    print(f"\n[3/3] 开始 benchmark...")
    benchmark_metrics = run_benchmark(exp_id, output_dir)
    results["benchmark_metrics"] = benchmark_metrics

    results["status"] = "completed"
    results["end_time"] = datetime.now().isoformat()

    # 清理临时配置文件
    if os.path.exists(temp_config):
        os.remove(temp_config)

    return results


def select_best_config(results: List[Dict[str, Any]], metric: str = "mean_psnr") -> Optional[str]:
    """选择最佳配置"""
    best_exp_id = None
    best_value = -float('inf')

    for result in results:
        if result['status'] != 'completed':
            continue
        value = result.get('benchmark_metrics', {}).get(metric, -float('inf'))
        if value > best_value:
            best_value = value
            best_exp_id = result['exp_id']

    return best_exp_id


def generate_stage_report(stage: int, results: List[Dict[str, Any]], output_file: str):
    """生成阶段报告"""
    print(f"\n{'='*80}")
    print(f"Stage {stage} 结果报告")
    print(f"{'='*80}\n")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"完整结果已保存到: {output_file}\n")

    # 表格
    print("="*100)
    print(f"Stage {stage} 实验结果")
    print("="*100)
    print()

    print(f"{'实验':<30} {'状态':<12} {'SH MAE':<12} {'sRGB PSNR':<12} {'SSIM':<8}")
    print("-"*100)

    for result in results:
        exp_name = result['name']
        status = result['status']

        if status != 'completed':
            print(f"{exp_name:<30} {status:<12} - - -")
            continue

        mae = result.get('eval_metrics', {}).get('mae', 0)
        psnr = result.get('benchmark_metrics', {}).get('mean_psnr', 0)
        ssim = result.get('benchmark_metrics', {}).get('mean_ssim', 0)

        print(f"{exp_name:<30} {status:<12} {mae:<12.6f} {psnr:<12.2f} {ssim:<8.4f}")

    print()
    print("="*100)
    print()

    # 选择最佳
    best_exp_id = select_best_config(results)
    if best_exp_id:
        best_result = next(r for r in results if r['exp_id'] == best_exp_id)
        best_psnr = best_result['benchmark_metrics']['mean_psnr']
        print(f"✓ 最佳配置: {best_result['name']} (PSNR: {best_psnr:.2f} dB)")
        print(f"  实验 ID: {best_exp_id}")
    else:
        print("⚠️  未找到成功的实验")

    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1 优化实验")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], required=True,
                       help="要运行的 Stage (1-4)")
    parser.add_argument("--base-config", type=str,
                       help="Stage 2/3/4 的基础配置 (从 Stage 1 最佳配置继承)")
    parser.add_argument("--parallel", action="store_true",
                       help="并行运行实验 (需要多 GPU 或 tmux)")

    args = parser.parse_args()

    # 选择实验集
    if args.stage == 1:
        experiments = STAGE1_EXPERIMENTS
    elif args.stage == 2:
        if not args.base_config:
            print("错误: Stage 2 需要 --base-config 参数 (从 Stage 1 选择最佳配置)")
            sys.exit(1)
        experiments = STAGE2_EXPERIMENTS
        # 更新 base_config
        for exp_config in experiments.values():
            exp_config['base_config'] = args.base_config
    elif args.stage == 3:
        if not args.base_config:
            print("错误: Stage 3 需要 --base-config 参数 (从 Stage 2 选择最佳配置)")
            sys.exit(1)
        experiments = STAGE3_EXPERIMENTS
        for exp_config in experiments.values():
            exp_config['base_config'] = args.base_config
    else:
        print("Stage 4 尚未实现")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"Phase 1 优化实验 - Stage {args.stage}")
    print(f"{'='*80}")
    print(f"\n将运行 {len(experiments)} 个实验:")
    for exp_id, exp_config in experiments.items():
        print(f"  - {exp_config['name']}: {exp_config['description']}")
    print()

    if args.parallel:
        print("⚠️  并行模式需要手动启动多个 tmux 会话")
        print("   建议: 为每个实验创建独立会话 (tmux new -s train_E1a)")
        sys.exit(0)

    # 运行所有实验
    all_results = []

    for i, (exp_id, exp_config) in enumerate(experiments.items(), 1):
        print(f"\n{'#'*80}")
        print(f"实验 {i}/{len(experiments)}: {exp_id}")
        print(f"{'#'*80}\n")

        result = run_single_experiment(exp_id, exp_config)
        all_results.append(result)

        # 保存中间结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        interim_file = f"3_experiments/results/phase1_stage{args.stage}_interim_{timestamp}.json"
        with open(interim_file, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 生成最终报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = f"3_experiments/results/phase1_stage{args.stage}_final_{timestamp}.json"
    generate_stage_report(args.stage, all_results, final_file)

    print(f"\n✓ Stage {args.stage} 完成!")
    print(f"最终结果: {final_file}\n")


if __name__ == "__main__":
    main()
