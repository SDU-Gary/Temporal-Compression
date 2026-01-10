"""
TPE Level 1 Validation Script (for multi-probe toy scenes)

Validates TPE hypothesis using Level 1 toy scene datasets.
Since Level 1 has multiple probes, this script tests a selected probe (typically the center probe).

Usage:
    python validate_tpe_level1.py --dataset ../data_generation/output/level1_tpe/shadow_plane_test --probe 12
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

from models.gaussian_mixture import GaussianMixture
from models.decoder_mlp import DecoderMLP


def load_level1_dataset(dataset_path: Path, probe_idx: int = 12) -> Dict:
    """
    Load Level 1 dataset and extract data for a specific probe.

    Args:
        dataset_path: Path to Level 1 dataset
        probe_idx: Index of probe to test

    Returns:
        data: Dict containing probe data
    """
    dataset_path = Path(dataset_path)

    # Load probe positions
    probes = np.load(dataset_path / "probes.npz")
    positions = probes['positions']  # [N, 3]

    if probe_idx >= len(positions):
        raise ValueError(f"Probe index {probe_idx} out of range (0-{len(positions)-1})")

    probe_position = positions[probe_idx]

    # Find all moment directories
    moment_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('moment_')])

    if not moment_dirs:
        raise ValueError(f"No moment directories found in {dataset_path}")

    # Load SH coefficients and sun directions for all time steps
    sh_coeffs_list = []
    sun_dirs_list = []
    hours_list = []

    for moment_dir in moment_dirs:
        # Extract hour from directory name (e.g., "moment_06" -> 6)
        hour = int(moment_dir.name.split('_')[1])

        # Load SH coefficients
        sh_file = moment_dir / "sh_coeffs.npz"
        if not sh_file.exists():
            print(f"WARNING: Missing {sh_file}, skipping")
            continue

        sh_data = np.load(sh_file)
        sh_coeffs_all = sh_data['coeffs']  # [N_probes, 27]
        sun_dir = sh_data['sun_dir']  # [3]

        # Extract data for specific probe
        sh_coeffs_list.append(sh_coeffs_all[probe_idx])
        sun_dirs_list.append(sun_dir)
        hours_list.append(hour)

    # Convert to arrays
    sh_gt = np.array(sh_coeffs_list)  # [T, 27]
    sun_dirs = np.array(sun_dirs_list)  # [T, 3]
    hours = np.array(hours_list)  # [T]

    # Extract scene name from path
    scene_name = dataset_path.name.replace('_test', '')

    return {
        'scene_name': scene_name,
        'position': probe_position,
        'sh_gt': sh_gt,
        'sun_dirs': sun_dirs,
        'hours': hours,
        'probe_idx': probe_idx,
        'total_probes': len(positions)
    }


def validate_tpe_level1(
    dataset_path: Path,
    probe_idx: int = 12,
    reference_hour: int = 12,
    num_optimization_steps: int = 500,
    learning_rate: float = 0.01
) -> Tuple[bool, Dict]:
    """
    Validate TPE hypothesis on Level 1 dataset.

    Args:
        dataset_path: Path to Level 1 dataset
        probe_idx: Index of probe to test (default: 12 = center)
        reference_hour: Reference time step
        num_optimization_steps: Number of optimization iterations
        learning_rate: Learning rate

    Returns:
        (all_pass, results): Tuple of boolean and results dict
    """
    # Load dataset
    print(f"\n{'='*70}")
    print(f"Loading Level 1 dataset...")
    print(f"{'='*70}")

    data = load_level1_dataset(dataset_path, probe_idx)

    scene_name = data['scene_name']
    probe_position = data['position']

    print(f"\n✓ Loaded Level 1 dataset: {scene_name}")
    print(f"  Total probes: {data['total_probes']}")
    print(f"  Testing probe: {probe_idx}")
    print(f"  Probe position: [{probe_position[0]:.2f}, {probe_position[1]:.2f}, {probe_position[2]:.2f}]")
    print(f"  Time steps: {len(data['hours'])}")
    print(f"  Hours: {data['hours'].tolist()}")

    # Convert to torch tensors
    sh_gt = torch.tensor(data['sh_gt'], dtype=torch.float32)      # [T, 27]
    sun_dirs = torch.tensor(data['sun_dirs'], dtype=torch.float32) # [T, 3]
    T = len(sh_gt)

    # Find reference index
    try:
        reference_idx = list(data['hours']).index(reference_hour)
    except ValueError:
        reference_idx = T // 2
        print(f"WARNING: Reference hour {reference_hour} not found, using index {reference_idx}")

    print(f"\n{'='*70}")
    print(f"TPE LEVEL 1 VALIDATION")
    print(f"{'='*70}")
    print(f"\nScene: {scene_name} (TOY SCENE - simpler than Level 2)")
    print(f"Testing whether lighting changes can be modeled as perturbations ε(t)")
    print(f"\nReference time: Hour {data['hours'][reference_idx]} (index {reference_idx})")

    # ==========================================================================
    # STAGE 1: TRAIN STATIC FIELD
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
    sun_dir_ref = sun_dirs[reference_idx]  # [3]
    sh_ref = sh_gt[reference_idx]  # [27]

    # Initialize Gaussian Mixture
    gaussian_mixture = GaussianMixture(
        num_gaussians=1,
        latent_dim=9,
        init_scale=0.5
    )

    with torch.no_grad():
        gaussian_mixture.means.data[0] = probe_pos_tensor
        gaussian_mixture.scales.data = torch.log(torch.ones(1, 3) * 0.5)

    # Initialize Decoder
    decoder = DecoderMLP(
        input_dim=15,
        hidden_dim=64,
        num_layers=2,
        output_dim=27
    )

    # Train static field
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

    if final_static_loss > 0.01:
        print(f"  WARNING: Static field fitting error is high!")

    # ==========================================================================
    # STAGE 2: OPTIMIZE PERTURBATIONS
    # ==========================================================================

    print(f"\n{'='*70}")
    print(f"STAGE 2: Optimizing Perturbations ε(t)")
    print(f"{'='*70}")
    print(f"\nOptimizing {T} perturbation vectors to minimize:")
    print(f"  loss = Σ_t || F_static(x + ε(t)) - SH_gt(t) ||^2")
    print(f"  subject to: ε[{reference_idx}] = 0")

    # Freeze static field
    for param in gaussian_mixture.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    # Initialize perturbations
    perturbations = torch.zeros(T, 3, requires_grad=True)
    optimizer_pert = torch.optim.Adam([perturbations], lr=learning_rate)

    print(f"\nOptimizing perturbations ({num_optimization_steps} steps)...")
    for step in range(num_optimization_steps):
        perturbed_positions = probe_pos_tensor.unsqueeze(0) + perturbations

        # Forward pass
        sh_pred_list = []
        for t in range(T):
            pos_t = perturbed_positions[t:t+1]
            latent_t = gaussian_mixture(pos_t)
            sh_t = decoder(latent_t, pos_t, sun_dir_ref)
            sh_pred_list.append(sh_t)
        sh_pred = torch.cat(sh_pred_list, dim=0)

        # Loss
        loss_recon = F.mse_loss(sh_pred, sh_gt)
        loss_reg = 0.001 * torch.mean(torch.norm(perturbations, dim=1)**2)
        loss = loss_recon + loss_reg

        # Backward
        optimizer_pert.zero_grad()
        loss.backward()
        optimizer_pert.step()

        # Constraint
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

    perturbations = perturbations.detach()

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

    if criterion_1:
        print(f"  → Taylor expansion approximation is valid")
    else:
        print(f"  → Perturbations too large, F(x+ε) ≈ F(x) + ∇F·ε breaks down")

    # Criterion 2: Smoothness
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

    # Criterion 3: Correlation
    sun_diffs = sun_dirs - sun_dirs[reference_idx:reference_idx+1]
    sun_diff_norms = torch.norm(sun_diffs, dim=1).numpy()

    if perturb_norms.std() < 1e-6 or sun_diff_norms.std() < 1e-6:
        correlation = 0.0
        print(f"\n[Criterion 3] Sun Direction Correlation")
        print(f"  Threshold: |correlation| > 0.5")
        print(f"  WARNING: Cannot compute correlation (no variation)")
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
    # FINAL RESULT
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
        print(f"\n  → TPE hypothesis is VALID for Level 1 toy scene '{scene_name}'")
        print(f"  → Lighting changes CAN be approximated by coordinate perturbations")
        print(f"  → This supports the TPE approach for simple scenarios")
    else:
        print(f"\n  → TPE hypothesis is INVALID for Level 1 toy scene '{scene_name}'")
        print(f"  → Even in simple scenes, perturbations don't satisfy criteria")
        print(f"  → This is concerning for the TPE approach")

    print(f"\n{'='*70}\n")

    results = {
        'scene_name': scene_name,
        'level': 1,
        'probe_idx': probe_idx,
        'all_pass': all_pass,
        'criterion_1_pass': criterion_1,
        'criterion_2_pass': criterion_2,
        'criterion_3_pass': criterion_3,
        'max_perturbation_norm': max_norm,
        'avg_temporal_smoothness': avg_smoothness,
        'sun_correlation': correlation,
        'final_recon_loss': final_recon_loss,
        'perturbations': perturbations.numpy()
    }

    return all_pass, results


def main():
    parser = argparse.ArgumentParser(
        description='Validate TPE hypothesis on Level 1 toy scenes'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to Level 1 dataset directory'
    )

    parser.add_argument(
        '--probe',
        type=int,
        default=12,
        help='Probe index to test (default: 12 = center probe in 5×5 grid)'
    )

    parser.add_argument(
        '--reference-hour',
        type=int,
        default=12,
        help='Reference hour for ε(t_ref) = 0 constraint (default: 12)'
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    all_pass, results = validate_tpe_level1(
        dataset_path,
        probe_idx=args.probe,
        reference_hour=args.reference_hour
    )

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
