"""
TPE Level 2 Validation Script

Validates the TPE (Temporal Perturbation Embedding) hypothesis using Level 2 datasets.

This is the CRITICAL TEST for TPE viability. Given a single probe's temporal
lighting data, this script optimizes perturbation vectors ε(t) and checks if
they satisfy three criteria:

1. ||ε(t)||_max < 1.0m  (Taylor expansion validity)
2. avg(||ε(t+1) - ε(t)||) < 0.5m  (temporal smoothness)
3. correlation(||ε||, ||Δsun_dir||) > 0.5  (physical correlation)

If ALL THREE criteria pass → TPE is VALID, proceed to full system
If ANY criterion fails → TPE is INVALID, abandon approach

Usage:
    python validate_tpe_level2.py --dataset ../data_generation/output/level2_tpe/cornell-box
    python validate_tpe_level2.py --all  # Test all Level 2 datasets
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
from typing import Dict, Tuple

from utils.tpe_dataset_loader import TPEValidationDataset
from models.gaussian_mixture import GaussianMixture
from models.decoder_mlp import DecoderMLP


def validate_perturbation_hypothesis(
    dataset_path: Path,
    reference_hour: int = 12,
    num_optimization_steps: int = 500,
    learning_rate: float = 0.01
) -> Tuple[bool, Dict]:
    """
    Validate TPE core hypothesis through perturbation optimization.

    Given ground truth SH coefficients across time, optimize perturbation
    vectors ε(t) to minimize reconstruction error, then check if they
    satisfy the three TPE criteria.

    Args:
        dataset_path: Path to Level 2 dataset
        reference_hour: Reference time step (default 12 = noon)
        num_optimization_steps: Number of optimization iterations
        learning_rate: Learning rate for Adam optimizer

    Returns:
        (all_pass, results): Tuple of boolean success flag and results dict
    """
    # Load dataset
    print(f"\n{'='*70}")
    print(f"Loading dataset...")
    print(f"{'='*70}")

    dataset = TPEValidationDataset(dataset_path, level=2)
    data = dataset.get_probe_data(0)

    scene_name = data['metadata']['scene_name']
    probe_position = data['position']

    print(f"\nDataset: {scene_name}")
    print(f"Probe position: [{probe_position[0]:.2f}, {probe_position[1]:.2f}, {probe_position[2]:.2f}]")
    print(f"Time steps: {len(data['hours'])}")
    print(f"Hours: {data['hours'].tolist()}")

    # Convert to torch tensors
    sh_gt = torch.tensor(data['sh_gt'], dtype=torch.float32)      # [T, 27]
    sun_dirs = torch.tensor(data['sun_dirs'], dtype=torch.float32) # [T, 3]
    T = len(sh_gt)

    # Find reference index
    try:
        reference_idx = list(data['hours']).index(reference_hour)
    except ValueError:
        reference_idx = T // 2  # Fallback to middle
        print(f"WARNING: Reference hour {reference_hour} not found, using index {reference_idx}")

    print(f"\n{'='*70}")
    print(f"TPE LEVEL 2 VALIDATION")
    print(f"{'='*70}")
    print(f"\nThis test validates whether lighting changes can be modeled")
    print(f"as coordinate perturbations ε(t) that satisfy physical constraints.")
    print(f"\nReference time: Hour {data['hours'][reference_idx]} (index {reference_idx})")

    # ==========================================================================
    # STAGE 1: TRAIN STATIC FIELD (using Gaussian Compression)
    # ==========================================================================

    print(f"\n{'='*70}")
    print(f"STAGE 1: Training Static Field F_static(x)")
    print(f"{'='*70}")
    print(f"\nUsing Gaussian Compression architecture:")
    print(f"  - GaussianMixture(K=1) for spatial representation")
    print(f"  - DecoderMLP for converting latent → SH coefficients")
    print(f"  - Trained on reference time step (Hour {data['hours'][reference_idx]})")

    # Initialize static field components
    probe_pos_tensor = torch.tensor(probe_position, dtype=torch.float32)
    sun_dir_ref = sun_dirs[reference_idx]  # [3] reference sun direction
    sh_ref = sh_gt[reference_idx]  # [27] reference SH coefficients

    # Initialize Gaussian Mixture (K=1 for single probe)
    gaussian_mixture = GaussianMixture(
        num_gaussians=1,
        latent_dim=9,
        init_scale=0.5
    )

    # Manually set Gaussian center to probe position
    with torch.no_grad():
        gaussian_mixture.means.data[0] = probe_pos_tensor
        gaussian_mixture.scales.data = torch.log(torch.ones(1, 3) * 0.5)

    # Initialize Decoder
    decoder = DecoderMLP(
        input_dim=15,  # latent(9) + position(3) + sun_dir(3)
        hidden_dim=64,
        num_layers=2,
        output_dim=27
    )

    # Train static field on reference time step
    optimizer_static = torch.optim.Adam(
        list(gaussian_mixture.parameters()) + list(decoder.parameters()),
        lr=0.001
    )

    print(f"\nTraining static field (1000 steps)...")
    for step in range(1000):
        # Forward pass
        latent = gaussian_mixture(probe_pos_tensor.unsqueeze(0))  # [1, 9]
        sh_pred = decoder(latent, probe_pos_tensor.unsqueeze(0), sun_dir_ref)  # [1, 27]

        # Loss
        loss = F.mse_loss(sh_pred, sh_ref.unsqueeze(0))

        # Backward pass
        optimizer_static.zero_grad()
        loss.backward()
        optimizer_static.step()

        if (step + 1) % 200 == 0:
            print(f"  Step {step+1:4d}: Loss = {loss.item():.6f}")

    final_static_loss = loss.item()
    print(f"\n✓ Static field trained. Final loss: {final_static_loss:.6f}")

    if final_static_loss > 0.01:
        print(f"  WARNING: Static field fitting error is high!")
        print(f"  This may indicate the Gaussian+Decoder architecture is insufficient.")

    # ==========================================================================
    # STAGE 2: OPTIMIZE PERTURBATIONS ε(t)
    # ==========================================================================

    print(f"\n{'='*70}")
    print(f"STAGE 2: Optimizing Perturbations ε(t)")
    print(f"{'='*70}")
    print(f"\nOptimizing {T} perturbation vectors to minimize:")
    print(f"  loss = Σ_t || F_static(x + ε(t)) - SH_gt(t) ||^2")
    print(f"  subject to: ε[{reference_idx}] = 0")

    # Freeze static field parameters
    for param in gaussian_mixture.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    # Initialize perturbations
    perturbations = torch.zeros(T, 3, requires_grad=True)

    optimizer_pert = torch.optim.Adam([perturbations], lr=0.01)

    print(f"\nOptimizing perturbations ({num_optimization_steps} steps)...")
    for step in range(num_optimization_steps):
        # Compute perturbed positions
        perturbed_positions = probe_pos_tensor.unsqueeze(0) + perturbations  # [T, 3]

        # Forward pass through static field (loop over time steps)
        sh_pred_list = []
        for t in range(T):
            pos_t = perturbed_positions[t:t+1]  # [1, 3]
            latent_t = gaussian_mixture(pos_t)  # [1, 9]
            sh_t = decoder(latent_t, pos_t, sun_dir_ref)  # [1, 27]
            sh_pred_list.append(sh_t)
        sh_pred = torch.cat(sh_pred_list, dim=0)  # [T, 27]

        # Reconstruction loss
        loss_recon = F.mse_loss(sh_pred, sh_gt)

        # L2 regularization on perturbations (encourage small ε)
        loss_reg = 0.001 * torch.mean(torch.norm(perturbations, dim=1)**2)

        loss = loss_recon + loss_reg

        # Backward pass
        optimizer_pert.zero_grad()
        loss.backward()
        optimizer_pert.step()

        # Enforce constraint: ε[reference_idx] = 0
        with torch.no_grad():
            perturbations[reference_idx] = 0.0

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1:4d}: Recon Loss = {loss_recon.item():.6f}, "
                  f"Reg Loss = {loss_reg.item():.6f}")

    final_recon_loss = loss_recon.item()
    print(f"\n✓ Perturbation optimization complete.")
    print(f"  Final reconstruction loss: {final_recon_loss:.6f}")

    if final_recon_loss > 0.05:
        print(f"  WARNING: Reconstruction error is high!")
        print(f"  This suggests TPE hypothesis may not hold for this scene.")

    # Detach perturbations for validation
    perturbations = perturbations.detach()

    # ==========================================================================
    # VALIDATION CRITERIA COMPUTATION
    # ==========================================================================

    print(f"\n{'='*70}")
    print(f"COMPUTING VALIDATION CRITERIA")
    print(f"{'='*70}")

    # Criterion 1: Maximum perturbation norm < 1.0m
    perturb_norms = torch.norm(perturbations, dim=1).numpy()
    max_norm = perturb_norms.max()
    criterion_1 = max_norm < 1.0

    print(f"\n[Criterion 1] Maximum Perturbation Norm")
    print(f"  Threshold: ||ε(t)||_max < 1.0 meters")
    print(f"  Measured:  ||ε(t)||_max = {max_norm:.3f} meters")
    print(f"  Status:    {'✓ PASS' if criterion_1 else '✗ FAIL'}")

    if criterion_1:
        print(f"  → Taylor expansion approximation is valid")
    else:
        print(f"  → Perturbations too large, F(x+ε) ≈ F(x) + ∇F·ε breaks down")

    # Criterion 2: Temporal smoothness < 0.5m
    perturb_diffs = torch.norm(torch.diff(perturbations, dim=0), dim=1).numpy()
    avg_smoothness = perturb_diffs.mean()
    criterion_2 = avg_smoothness < 0.5

    print(f"\n[Criterion 2] Temporal Smoothness")
    print(f"  Threshold: avg(||ε(t+1) - ε(t)||) < 0.5 meters")
    print(f"  Measured:  avg = {avg_smoothness:.3f} meters")
    print(f"  Status:    {'✓ PASS' if criterion_2 else '✗ FAIL'}")

    if criterion_2:
        print(f"  → Perturbations change smoothly over time (physically plausible)")
    else:
        print(f"  → Perturbations jump erratically (unphysical)")

    # Criterion 3: Correlation with sun direction > 0.5
    sun_diffs = sun_dirs - sun_dirs[reference_idx:reference_idx+1]
    sun_diff_norms = torch.norm(sun_diffs, dim=1).numpy()

    if perturb_norms.std() < 1e-6 or sun_diff_norms.std() < 1e-6:
        correlation = 0.0  # Undefined correlation if no variation
        print(f"\n[Criterion 3] Sun Direction Correlation")
        print(f"  Threshold: |correlation| > 0.5")
        print(f"  WARNING: Cannot compute correlation (no perturbation variation)")
        criterion_3 = False
    else:
        correlation = np.corrcoef(perturb_norms, sun_diff_norms)[0, 1]
        criterion_3 = abs(correlation) > 0.5

        print(f"\n[Criterion 3] Sun Direction Correlation")
        print(f"  Threshold: |correlation(||ε||, ||Δsun||)| > 0.5")
        print(f"  Measured:  correlation = {correlation:.3f}")
        print(f"  Status:    {'✓ PASS' if criterion_3 else '✗ FAIL'}")

        if criterion_3:
            print(f"  → Perturbations correlate with sun movement (physical meaning)")
        else:
            print(f"  → Perturbations don't follow sun (lacks physical interpretation)")

    # ==========================================================================
    # FINAL DECISION
    # ==========================================================================

    all_pass = criterion_1 and criterion_2 and criterion_3

    print(f"\n{'='*70}")
    print(f"FINAL RESULT")
    print(f"{'='*70}")

    print(f"\n  Criterion 1 (||ε||_max < 1.0m):     {'✓ PASS' if criterion_1 else '✗ FAIL'}")
    print(f"  Criterion 2 (smoothness < 0.5m):     {'✓ PASS' if criterion_2 else '✗ FAIL'}")
    print(f"  Criterion 3 (correlation > 0.5):     {'✓ PASS' if criterion_3 else '✗ FAIL'}")

    print(f"\n  Overall: {'✓✓✓ ALL PASS ✓✓✓' if all_pass else '✗✗✗ FAILED ✗✗✗'}")

    if all_pass:
        print(f"\n  → TPE hypothesis is VALID for scene '{scene_name}'")
        print(f"  → Lighting changes CAN be approximated by coordinate perturbations")
        print(f"  → RECOMMENDATION: Proceed to Level 3 (full system implementation)")
    else:
        print(f"\n  → TPE hypothesis is INVALID for scene '{scene_name}'")
        print(f"  → Lighting changes CANNOT be modeled as simple perturbations")
        print(f"  → RECOMMENDATION: Abandon TPE approach for this scene type")

    print(f"\n{'='*70}\n")

    # Save results for further analysis
    output_dir = dataset_path / 'validation_results'
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / 'tpe_validation_results.npz'
    np.savez_compressed(
        results_file,
        perturbations=perturbations.numpy(),
        sun_dirs=sun_dirs.numpy(),
        hours=data['hours'],
        probe_position=probe_position,
        sh_gt=sh_gt.numpy(),
        reference_idx=reference_idx,
        scene_name=scene_name,
        max_norm=max_norm,
        avg_smoothness=avg_smoothness,
        correlation=correlation
    )

    print(f"\n✓ Validation results saved to: {results_file}")
    print(f"  Use this file for further analysis (e.g., direction analysis)")

    # Return results
    results = {
        'scene_name': scene_name,
        'all_pass': all_pass,
        'criterion_1_pass': criterion_1,
        'criterion_2_pass': criterion_2,
        'criterion_3_pass': criterion_3,
        'max_perturbation_norm': max_norm,
        'avg_temporal_smoothness': avg_smoothness,
        'sun_correlation': correlation,
        'perturbations': perturbations.numpy(),
        'results_file': str(results_file)
    }

    return all_pass, results


def main():
    parser = argparse.ArgumentParser(
        description='Validate TPE hypothesis using Level 2 datasets'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to Level 2 dataset directory'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Validate all Level 2 datasets'
    )

    parser.add_argument(
        '--reference-hour',
        type=int,
        default=12,
        help='Reference hour for ε(t_ref) = 0 constraint (default: 12 = noon)'
    )

    args = parser.parse_args()

    # Determine which datasets to test
    if args.all:
        # Test all Level 2 datasets
        level2_base = Path('../data_generation/output/level2_tpe')

        if not level2_base.exists():
            print(f"ERROR: Level 2 output directory not found: {level2_base}")
            sys.exit(1)

        dataset_dirs = [d for d in level2_base.iterdir() if d.is_dir()]

        if not dataset_dirs:
            print(f"ERROR: No Level 2 datasets found in {level2_base}")
            sys.exit(1)

        print(f"Found {len(dataset_dirs)} Level 2 datasets to validate")

        results_summary = []

        for dataset_dir in dataset_dirs:
            all_pass, results = validate_perturbation_hypothesis(
                dataset_dir,
                reference_hour=args.reference_hour
            )
            results_summary.append(results)

        # Print summary
        print(f"\n{'='*70}")
        print(f"SUMMARY: ALL LEVEL 2 DATASETS")
        print(f"{'='*70}\n")

        for res in results_summary:
            status = '✓ PASS' if res['all_pass'] else '✗ FAIL'
            print(f"  {res['scene_name']:20s}  {status}")

        total_pass = sum(1 for r in results_summary if r['all_pass'])
        print(f"\n  Total: {total_pass}/{len(results_summary)} datasets passed")

        if total_pass > 0:
            print(f"\n  → TPE is viable for {total_pass} scene type(s)")
            print(f"  → Consider full implementation")
        else:
            print(f"\n  → TPE failed on all scenes")
            print(f"  → Strongly recommend abandoning TPE approach")

    elif args.dataset:
        # Test single dataset
        dataset_path = Path(args.dataset)

        if not dataset_path.exists():
            print(f"ERROR: Dataset not found: {dataset_path}")
            sys.exit(1)

        all_pass, results = validate_perturbation_hypothesis(
            dataset_path,
            reference_hour=args.reference_hour
        )

        sys.exit(0 if all_pass else 1)

    else:
        parser.print_help()
        print("\nERROR: Must specify --dataset or --all")
        sys.exit(1)


if __name__ == "__main__":
    main()
