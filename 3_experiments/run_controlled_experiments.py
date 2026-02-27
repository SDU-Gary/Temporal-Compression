#!/usr/bin/env python3
"""
单一变量控制实验自动化脚本

实验设计：
A. K30_r8 baseline (无正则化) - 基线
B. K50_r16 capacity only (无正则化) - 仅提升容量
C. K30_r8 + regularization - 仅增加正则化
D. K50_r16 + regularization - 容量 + 正则化

每个实验包含：训练 → 评估 → benchmark → 指标收集
"""

import os
import sys
import json
import yaml
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# 实验配置
EXPERIMENTS = {
    "A_K30_r8_baseline": {
        "name": "A. K30_r8 Baseline (无正则化)",
        "config_file": "3_experiments/configs/bistro_clean_train.yaml",
        "modifications": {
            "model.num_gaussians": 30,
            "model.rank": 8,
            "training.enable_sh_scaler": False,
            "training.enable_weighted_sh_loss": False,
            "training.lambda_image": 0.0,
            "experiment.output_dir": "3_experiments/results/bistro_clean_v2/exp_A_K30_r8_baseline",
        },
        "description": "基线：小容量，无正则化"
    },
    "B_K50_r16_capacity": {
        "name": "B. K50_r16 Capacity Only (无正则化)",
        "config_file": "3_experiments/configs/bistro_clean_train_k50_r16.yaml",
        "modifications": {
            "model.num_gaussians": 50,
            "model.rank": 16,
            "training.enable_sh_scaler": False,
            "training.enable_weighted_sh_loss": False,
            "training.lambda_image": 0.0,
            "experiment.output_dir": "3_experiments/results/bistro_clean_v2/exp_B_K50_r16_capacity",
        },
        "description": "单一变量：仅提升容量 (K30→K50, r8→r16)"
    },
    "C_K30_r8_regularized": {
        "name": "C. K30_r8 + Regularization",
        "config_file": "3_experiments/configs/bistro_clean_train.yaml",
        "modifications": {
            "model.num_gaussians": 30,
            "model.rank": 8,
            "training.enable_sh_scaler": False,
            "training.enable_weighted_sh_loss": True,
            "training.sh_loss_weights": [1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0],
            "training.sh_weight_mode": "basis",
            "training.lambda_image": 0.01,
            "training.image_loss_type": "mse",
            "training.image_samples": 256,
            "training.image_sample_seed": 42,
            "training.image_loss_space": "srgb",
            "experiment.output_dir": "3_experiments/results/bistro_clean_v2/exp_C_K30_r8_regularized",
        },
        "description": "单一变量：仅增加正则化（修正的权重 + sRGB image loss）"
    },
    "D_K50_r16_full": {
        "name": "D. K50_r16 + Regularization",
        "config_file": "3_experiments/configs/bistro_clean_train_k50_r16.yaml",
        "modifications": {
            "model.num_gaussians": 50,
            "model.rank": 16,
            "training.enable_sh_scaler": False,
            "training.enable_weighted_sh_loss": True,
            "training.sh_loss_weights": [1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0],
            "training.sh_weight_mode": "basis",
            "training.lambda_image": 0.01,
            "training.image_loss_type": "mse",
            "training.image_samples": 256,
            "training.image_sample_seed": 42,
            "training.image_loss_space": "srgb",
            "experiment.output_dir": "3_experiments/results/bistro_clean_v2/exp_D_K50_r16_full",
        },
        "description": "组合：容量提升 + 正则化"
    }
}

# Falcor 路径配置
FALCOR_CONFIG = {
    "python_path": "/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python",
    "python_bin": "/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10",
}


def set_nested_value(d: Dict, key_path: str, value: Any):
    """设置嵌套字典的值，如 'model.num_gaussians' -> d['model']['num_gaussians']"""
    keys = key_path.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def modify_config(config_file: str, modifications: Dict[str, Any], output_file: str):
    """修改配置文件并保存"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    for key_path, value in modifications.items():
        set_nested_value(config, key_path, value)

    # 确保输出目录存在
    output_dir = config['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config


def run_command(cmd: List[str], description: str, log_file: str = None) -> int:
    """运行命令并记录输出"""
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    if log_file:
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Started: {datetime.now()}\n\n")

        with open(log_file, 'a') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

        with open(log_file, 'a') as f:
            f.write(f"\nFinished: {datetime.now()}\n")
            f.write(f"Exit code: {result.returncode}\n")
    else:
        result = subprocess.run(cmd)

    return result.returncode


def run_training(exp_id: str, config_file: str, output_dir: str) -> bool:
    """运行训练"""
    log_file = f"{output_dir}/training.log"
    
    cmd = [
        "python", "3_experiments/scripts/train.py",
        "--config", config_file
    ]
    
    returncode = run_command(
        cmd,
        f"Training {exp_id}",
        log_file
    )
    
    return returncode == 0


def run_evaluation(exp_id: str, output_dir: str, rank: int) -> Dict[str, Any]:
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
    
    returncode = run_command(
        cmd,
        f"Evaluation {exp_id}",
        log_file
    )
    
    if returncode != 0:
        print(f"⚠️  Evaluation failed for {exp_id}")
        return {}
    
    # 读取评估结果
    eval_json = f"{output_dir}/eval.json"
    if os.path.exists(eval_json):
        with open(eval_json, 'r') as f:
            return json.load(f)
    
    return {}


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
    
    returncode = run_command(
        cmd,
        f"Benchmark {exp_id}",
        log_file
    )
    
    if returncode != 0:
        print(f"⚠️  Benchmark failed for {exp_id}")
        return {}
    
    # 读取 benchmark 结果
    summary_json = f"{benchmark_dir}/benchmark_summary.json"
    if os.path.exists(summary_json):
        with open(summary_json, 'r') as f:
            data = json.load(f)
            return data.get("image_metrics", {})
    
    return {}


def run_single_experiment(exp_id: str, exp_config: Dict[str, Any]) -> Dict[str, Any]:
    """运行单个实验的完整流程"""
    print(f"\n{'='*80}")
    print(f"开始实验: {exp_config['name']}")
    print(f"描述: {exp_config['description']}")
    print(f"{'='*80}\n")
    
    # 1. 准备配置文件
    temp_config = f"3_experiments/configs/temp_{exp_id}.yaml"
    config = modify_config(
        exp_config['config_file'],
        exp_config['modifications'],
        temp_config
    )
    
    output_dir = config['experiment']['output_dir']
    rank = config['model']['rank']
    
    results = {
        "exp_id": exp_id,
        "name": exp_config['name'],
        "description": exp_config['description'],
        "config": {
            "num_gaussians": config['model']['num_gaussians'],
            "rank": config['model']['rank'],
            "enable_sh_scaler": config['training'].get('enable_sh_scaler', False),
            "enable_weighted_sh_loss": config['training'].get('enable_weighted_sh_loss', False),
            "lambda_image": config['training'].get('lambda_image', 0.0),
        },
        "status": "started",
        "start_time": datetime.now().isoformat(),
    }
    
    # 2. 训练
    print(f"\n[1/3] 开始训练...")
    training_success = run_training(exp_id, temp_config, output_dir)
    
    if not training_success:
        results["status"] = "training_failed"
        results["end_time"] = datetime.now().isoformat()
        return results
    
    # 3. 评估（SH 系数指标）
    print(f"\n[2/3] 开始评估...")
    eval_metrics = run_evaluation(exp_id, output_dir, rank)
    results["eval_metrics"] = eval_metrics
    
    # 4. Benchmark（渲染图像指标）
    print(f"\n[3/3] 开始 benchmark...")
    benchmark_metrics = run_benchmark(exp_id, output_dir)
    results["benchmark_metrics"] = benchmark_metrics
    
    results["status"] = "completed"
    results["end_time"] = datetime.now().isoformat()
    
    # 清理临时配置文件
    if os.path.exists(temp_config):
        os.remove(temp_config)
    
    return results


def generate_comparison_report(all_results: List[Dict[str, Any]], output_file: str):
    """生成对比报告"""
    print(f"\n{'='*80}")
    print("生成对比报告")
    print(f"{'='*80}\n")
    
    # 保存完整结果
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"完整结果已保存到: {output_file}\n")
    
    # 生成对比表格
    print("="*120)
    print("实验结果对比")
    print("="*120)
    print()
    
    # 表头
    print(f"{'实验':<25} {'K':<5} {'r':<5} {'Scaler':<8} {'Weighted':<10} {'λ_img':<8} "
          f"{'MAE':<10} {'RMSE':<10} {'sRGB PSNR':<12} {'SSIM':<8}")
    print("-"*120)
    
    # 数据行
    for result in all_results:
        if result['status'] != 'completed':
            print(f"{result['name']:<25} - FAILED -")
            continue
        
        cfg = result['config']
        eval_m = result.get('eval_metrics', {})
        bench_m = result.get('benchmark_metrics', {})
        
        exp_name = result['exp_id'].replace('_', ' ')
        K = cfg['num_gaussians']
        r = cfg['rank']
        scaler = '✓' if cfg['enable_sh_scaler'] else '✗'
        weighted = '✓' if cfg['enable_weighted_sh_loss'] else '✗'
        lambda_img = cfg['lambda_image']
        
        mae = eval_m.get('mae', 0)
        rmse = eval_m.get('rmse', 0)
        psnr = bench_m.get('mean_psnr', 0)
        ssim = bench_m.get('mean_ssim', 0)
        
        print(f"{exp_name:<25} {K:<5} {r:<5} {scaler:<8} {weighted:<10} {lambda_img:<8.3f} "
              f"{mae:<10.6f} {rmse:<10.6f} {psnr:<12.2f} {ssim:<8.4f}")
    
    print()
    print("="*120)
    print()
    
    # 分析
    print("关键发现:")
    print("-"*120)
    
    if len(all_results) >= 4:
        A = all_results[0]  # K30_r8 baseline
        B = all_results[1]  # K50_r16 capacity
        C = all_results[2]  # K30_r8 regularized
        D = all_results[3]  # K50_r16 full
        
        if all(r['status'] == 'completed' for r in [A, B, C, D]):
            A_psnr = A['benchmark_metrics'].get('mean_psnr', 0)
            B_psnr = B['benchmark_metrics'].get('mean_psnr', 0)
            C_psnr = C['benchmark_metrics'].get('mean_psnr', 0)
            D_psnr = D['benchmark_metrics'].get('mean_psnr', 0)
            
            print(f"1. 容量提升效果 (A→B): {B_psnr - A_psnr:+.2f} dB")
            print(f"   {'提升' if B_psnr > A_psnr else '下降'}")
            print()
            
            print(f"2. 正则化效果 (A→C): {C_psnr - A_psnr:+.2f} dB")
            print(f"   {'提升' if C_psnr > A_psnr else '下降'}")
            print()
            
            print(f"3. 组合效果 (A→D): {D_psnr - A_psnr:+.2f} dB")
            print(f"   {'提升' if D_psnr > A_psnr else '下降'}")
            print()
            
            print(f"4. 协同效应 (B+C vs D):")
            expected = (B_psnr - A_psnr) + (C_psnr - A_psnr)
            actual = D_psnr - A_psnr
            print(f"   预期: {expected:+.2f} dB (独立效应之和)")
            print(f"   实际: {actual:+.2f} dB")
            print(f"   协同: {actual - expected:+.2f} dB ({'正' if actual > expected else '负'}协同)")
    
    print()
    print("="*120)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="运行单一变量控制实验")
    parser.add_argument("--experiments", nargs="+", 
                       choices=list(EXPERIMENTS.keys()) + ["all"],
                       default=["all"],
                       help="要运行的实验 (默认: all)")
    parser.add_argument("--skip-training", action="store_true",
                       help="跳过训练，仅运行评估和benchmark")
    parser.add_argument("--skip-benchmark", action="store_true",
                       help="跳过benchmark，仅运行训练和评估")
    
    args = parser.parse_args()
    
    # 确定要运行的实验
    if "all" in args.experiments:
        exp_ids = list(EXPERIMENTS.keys())
    else:
        exp_ids = args.experiments
    
    print(f"\n{'='*80}")
    print("单一变量控制实验")
    print(f"{'='*80}")
    print(f"\n将运行 {len(exp_ids)} 个实验:")
    for exp_id in exp_ids:
        print(f"  - {EXPERIMENTS[exp_id]['name']}")
    print()
    
    # 运行所有实验
    all_results = []
    
    for i, exp_id in enumerate(exp_ids, 1):
        print(f"\n{'#'*80}")
        print(f"实验 {i}/{len(exp_ids)}: {exp_id}")
        print(f"{'#'*80}\n")
        
        result = run_single_experiment(exp_id, EXPERIMENTS[exp_id])
        all_results.append(result)
        
        # 保存中间结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        interim_file = f"3_experiments/results/experiment_results_interim_{timestamp}.json"
        with open(interim_file, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 生成最终报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = f"3_experiments/results/experiment_results_final_{timestamp}.json"
    generate_comparison_report(all_results, final_file)
    
    print(f"\n✓ 所有实验完成!")
    print(f"最终结果: {final_file}\n")


if __name__ == "__main__":
    main()
