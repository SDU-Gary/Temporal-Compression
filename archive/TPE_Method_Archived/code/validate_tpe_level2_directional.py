"""
实验G: 方向约束TPE验证

基于实验C的发现，使用方向约束优化扰动，大幅减少自由度。

扰动形式：ε(t) = α(t) · d
- d: 固定方向（从PCA提取或使用纯Y轴）
- α(t): 标量幅度（T个自由度，而非3T个）

预期：通过方向约束，||ε(t)||_max 降至 < 1.0m

Usage:
    python validate_tpe_level2_directional.py \
        --dataset ../../data_generation/output/level2_tpe/cornell-box_test \
        --direction-mode pca
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
from typing import Dict, Tuple

from utils.tpe_dataset_loader import TPEValidationDataset
from models.gaussian_mixture import GaussianMixture
from models.decoder_mlp import DecoderMLP


def validate_directional_tpe(
    dataset_path: Path,
    direction_mode: str = 'pca',  # 'pca', 'y_axis', or 'custom'
    custom_direction: np.ndarray = None,
    reference_hour: int = 12,
    num_optimization_steps: int = 500,
    learning_rate: float = 0.01
) -> Tuple[bool, Dict]:
    """
    使用方向约束验证TPE假设
    """
    # Load dataset
    print(f"\n{'='*70}")
    print(f"EXPERIMENT G: DIRECTION-CONSTRAINED TPE VALIDATION")
    print(f"{'='*70}")

    dataset = TPEValidationDataset(dataset_path, level=2)
    data = dataset.get_probe_data(0)

    scene_name = data['metadata']['scene_name']
    probe_position = data['position']

    print(f"\nDataset: {scene_name}")
    print(f"Probe position: [{probe_position[0]:.2f}, {probe_position[1]:.2f}, {probe_position[2]:.2f}]")
    print(f"Time steps: {len(data['hours'])}")

    # Convert to tensors
    sh_gt = torch.tensor(data['sh_gt'], dtype=torch.float32)
    sun_dirs = torch.tensor(data['sun_dirs'], dtype=torch.float32)
    T = len(sh_gt)

    # Find reference index
    try:
        reference_idx = list(data['hours']).index(reference_hour)
    except ValueError:
        reference_idx = T // 2

    # ==========================================================================
    # 确定方向约束
    # ==========================================================================

    print(f"\n{'='*70}")
    print(f"DETERMINING PERTURBATION DIRECTION")
    print(f"{'='*70}")

    if direction_mode == 'pca':
        # 从之前的分析结果加载主方向
        analysis_file = dataset_path / 'validation_results' / 'direction_analysis_results.npz'
        if analysis_file.exists():
            analysis = np.load(analysis_file)
            direction = analysis['principal_direction']
            print(f"\nUsing PCA principal direction from previous analysis:")
            print(f"  Direction: [{direction[0]:+.3f}, {direction[1]:+.3f}, {direction[2]:+.3f}]")
            print(f"  Explained variance: {analysis['explained_variance_ratio']:.1%}")
        else:
            print(f"\nWARNING: PCA analysis not found, falling back to Y-axis")
            direction = np.array([0.0, 1.0, 0.0])
    elif direction_mode == 'y_axis':
        direction = np.array([0.0, 1.0, 0.0])
        print(f"\nUsing pure Y-axis direction: [0, 1, 0]")
        print(f"  Rationale: Cornell Box vertical structure")
    elif direction_mode == 'custom' and custom_direction is not None:
        direction = custom_direction / np.linalg.norm(custom_direction)
        print(f"\nUsing custom direction: [{direction[0]:+.3f}, {direction[1]:+.3f}, {direction[2]:+.3f}]")
    else:
        raise ValueError(f"Invalid direction_mode: {direction_mode}")

    direction_tensor = torch.tensor(direction, dtype=torch.float32)

    # ==========================================================================
    # STAGE 1: 训练静态场（同baseline）
    # ==========================================================================

    print(f"\n{'='*70}")
    print(f"STAGE 1: Training Static Field F_static(x)")
    print(f"{'='*70}")

    probe_pos_tensor = torch.tensor(probe_position, dtype=torch.float32)
    sun_dir_ref = sun_dirs[reference_idx]
    sh_ref = sh_gt[reference_idx]

    gaussian_mixture = GaussianMixture(num_gaussians=1, latent_dim=9, init_scale=0.5)
    with torch.no_grad():
        gaussian_mixture.means.data[0] = probe_pos_tensor
        gaussian_mixture.scales.data = torch.log(torch.ones(1, 3) * 0.5)

    decoder = DecoderMLP(input_dim=15, hidden_dim=64, num_layers=2, output_dim=27)

    optimizer_static = torch.optim.Adam(
        list(gaussian_mixture.parameters()) + list(decoder.parameters()),
        lr=0.001
    )

    print(f"\nTraining static field (1000 steps)...")
    for step in range(1000):
        latent = gaussian_mixture(probe_pos_tensor.unsqueeze(0))
        sh_pred = decoder(latent, probe_pos_tensor.unsqueeze(0), sun_dir_ref)
        loss = F.mse_loss(sh_pred, sh_ref.unsqueeze(0))

        optimizer_static.zero_grad()
        loss.backward()
        optimizer_static.step()

        if (step + 1) % 200 == 0:
            print(f"  Step {step+1:4d}: Loss = {loss.item():.6f}")

    final_static_loss = loss.item()
    print(f"\n✓ Static field trained. Final loss: {final_static_loss:.6f}")

    # ==========================================================================
    # STAGE 2: 优化方向约束的扰动
    # ==========================================================================

    print(f"\n{'='*70}")
    print(f"STAGE 2: Optimizing Direction-Constrained Perturbations")
    print(f"{'='*70}")
    print(f"\nOptimization strategy:")
    print(f"  ε(t) = α(t) · d")
    print(f"  Free parameters: T scalar amplitudes (vs. 3T in baseline)")
    print(f"  Constraint: α[reference] = 0")

    # Freeze static field
    for param in gaussian_mixture.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    # Initialize scalar amplitudes
    alphas = torch.zeros(T, requires_grad=True)

    # Stronger L2 regularization for constrained case
    reg_weight = 0.002  # 2x baseline
    optimizer_pert = torch.optim.Adam([alphas], lr=learning_rate)

    print(f"\nOptimizing {T} scalar amplitudes ({num_optimization_steps} steps)...")
    for step in range(num_optimization_steps):
        # Compute perturbations: ε(t) = α(t) · d
        perturbations = alphas.unsqueeze(1) * direction_tensor.unsqueeze(0)  # [T, 3]

        # Forward pass
        sh_pred_list = []
        for t in range(T):
            pos_t = probe_pos_tensor.unsqueeze(0) + perturbations[t:t+1]
            latent_t = gaussian_mixture(pos_t)
            sh_t = decoder(latent_t, pos_t, sun_dir_ref)
            sh_pred_list.append(sh_t)
        sh_pred = torch.cat(sh_pred_list, dim=0)

        # Loss
        loss_recon = F.mse_loss(sh_pred, sh_gt)
        loss_reg = reg_weight * torch.mean(alphas**2)
        loss = loss_recon + loss_reg

        # Backward
        optimizer_pert.zero_grad()
        loss.backward()
        optimizer_pert.step()

        # Constraint: α[reference_idx] = 0
        with torch.no_grad():
            alphas[reference_idx] = 0.0

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1:4d}: Recon Loss = {loss_recon.item():.6f}, "
                  f"Reg Loss = {loss_reg.item():.6f}, "
                  f"Max |α| = {torch.abs(alphas).max().item():.3f}m")

    final_recon_loss = loss_recon.item()
    print(f"\n✓ Direction-constrained optimization complete.")
    print(f"  Final reconstruction loss: {final_recon_loss:.6f}")

    # Compute final perturbations
    perturbations = (alphas.unsqueeze(1) * direction_tensor.unsqueeze(0)).detach()

    # ==========================================================================
    # VALIDATION CRITERIA
    # ==========================================================================

    print(f"\n{'='*70}")
    print(f"COMPUTING VALIDATION CRITERIA")
    print(f"{'='*70}")

    # Criterion 1: Max norm
    perturb_norms = torch.norm(perturbations, dim=1).numpy()
    max_norm = perturb_norms.max()
    criterion_1 = max_norm < 1.0

    print(f"\n[Criterion 1] Maximum Perturbation Norm")
    print(f"  Threshold: ||ε(t)||_max < 1.0 meters")
    print(f"  Measured:  ||ε(t)||_max = {max_norm:.3f} meters")
    print(f"  Status:    {'✓ PASS' if criterion_1 else '✗ FAIL'}")

    # Compare with baseline
    baseline_file = dataset_path / 'validation_results' / 'tpe_validation_results.npz'
    if baseline_file.exists():
        baseline_data = np.load(baseline_file)
        baseline_max_norm = float(baseline_data['max_norm'])
        improvement_pct = (baseline_max_norm - max_norm) / baseline_max_norm * 100
        print(f"\n  Comparison with baseline (unconstrained):")
        print(f"    Baseline ||ε||_max:  {baseline_max_norm:.3f}m")
        print(f"    Directional ||ε||_max: {max_norm:.3f}m")
        print(f"    Improvement: {improvement_pct:+.1f}%")

        if improvement_pct > 0:
            print(f"    ✓ Direction constraint REDUCED perturbation magnitude!")
        else:
            print(f"    ⚠ Direction constraint did NOT reduce magnitude")

    # Criterion 2: Smoothness
    perturb_diffs = torch.norm(torch.diff(perturbations, dim=0), dim=1).numpy()
    avg_smoothness = perturb_diffs.mean()
    criterion_2 = avg_smoothness < 0.5

    print(f"\n[Criterion 2] Temporal Smoothness")
    print(f"  Threshold: avg(||ε(t+1) - ε(t)||) < 0.5 meters")
    print(f"  Measured:  avg = {avg_smoothness:.3f} meters")
    print(f"  Status:    {'✓ PASS' if criterion_2 else '✗ FAIL'}")

    # Criterion 3: Correlation
    sun_diffs = sun_dirs - sun_dirs[reference_idx:reference_idx+1]
    sun_diff_norms = torch.norm(sun_diffs, dim=1).numpy()

    if perturb_norms.std() < 1e-6 or sun_diff_norms.std() < 1e-6:
        correlation = 0.0
        print(f"\n[Criterion 3] Sun Direction Correlation")
        print(f"  WARNING: Cannot compute correlation")
        criterion_3 = False
    else:
        correlation = np.corrcoef(perturb_norms, sun_diff_norms)[0, 1]
        criterion_3 = abs(correlation) > 0.5

        print(f"\n[Criterion 3] Sun Direction Correlation")
        print(f"  Threshold: |correlation(||ε||, ||Δsun||)| > 0.5")
        print(f"  Measured:  correlation = {correlation:.3f}")
        print(f"  Status:    {'✓ PASS' if criterion_3 else '✗ FAIL'}")

    # ==========================================================================
    # FINAL RESULT
    # ==========================================================================

    all_pass = criterion_1 and criterion_2 and criterion_3

    print(f"\n{'='*70}")
    print(f"EXPERIMENT G FINAL RESULT")
    print(f"{'='*70}")

    print(f"\n  Criterion 1 (||ε||_max < 1.0m):     {'✓ PASS' if criterion_1 else '✗ FAIL'}")
    print(f"  Criterion 2 (smoothness < 0.5m):     {'✓ PASS' if criterion_2 else '✗ FAIL'}")
    print(f"  Criterion 3 (correlation > 0.5):     {'✓ PASS' if criterion_3 else '✗ FAIL'}")

    print(f"\n  Overall: {'✓✓✓ ALL PASS ✓✓✓' if all_pass else '✗✗✗ FAILED ✗✗✗'}")

    if all_pass:
        print(f"\n  🎉🎉🎉 BREAKTHROUGH! 🎉🎉🎉")
        print(f"  → Direction-constrained TPE is VALID for '{scene_name}'")
        print(f"  → Reducing free parameters from 3T to T enabled success")
        print(f"  → This proves TPE works with proper geometric constraints")
    elif criterion_1:
        print(f"\n  ✓ Direction constraint FIXED criterion 1!")
        print(f"  → ||ε||_max now within threshold")
        if not all_pass:
            print(f"  ⚠ Other criteria still need work")
    else:
        print(f"\n  ✗ Direction constraint insufficient")
        print(f"  → May need multi-anchor TPE (Experiment D)")

    print(f"\n{'='*70}\n")

    # Save results
    output_dir = dataset_path / 'validation_results'
    results_file = output_dir / 'tpe_directional_validation_results.npz'
    np.savez_compressed(
        results_file,
        perturbations=perturbations.numpy(),
        alphas=alphas.detach().numpy(),
        direction=direction,
        sun_dirs=sun_dirs.numpy(),
        hours=data['hours'],
        probe_position=probe_position,
        reference_idx=reference_idx,
        scene_name=scene_name,
        max_norm=max_norm,
        avg_smoothness=avg_smoothness,
        correlation=correlation,
        direction_mode=direction_mode
    )

    print(f"✓ Results saved to: {results_file}")

    results = {
        'scene_name': scene_name,
        'direction_mode': direction_mode,
        'all_pass': all_pass,
        'criterion_1_pass': criterion_1,
        'criterion_2_pass': criterion_2,
        'criterion_3_pass': criterion_3,
        'max_perturbation_norm': max_norm,
        'avg_temporal_smoothness': avg_smoothness,
        'sun_correlation': correlation,
        'final_recon_loss': final_recon_loss,
        'perturbations': perturbations.numpy(),
        'alphas': alphas.detach().numpy()
    }

    return all_pass, results


def main():
    parser = argparse.ArgumentParser(
        description='Experiment G: Direction-constrained TPE validation'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to Level 2 dataset directory'
    )

    parser.add_argument(
        '--direction-mode',
        type=str,
        default='pca',
        choices=['pca', 'y_axis', 'custom'],
        help='Direction constraint mode (default: pca)'
    )

    parser.add_argument(
        '--reference-hour',
        type=int,
        default=12,
        help='Reference hour (default: 12)'
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    all_pass, results = validate_directional_tpe(
        dataset_path,
        direction_mode=args.direction_mode,
        reference_hour=args.reference_hour
    )

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
