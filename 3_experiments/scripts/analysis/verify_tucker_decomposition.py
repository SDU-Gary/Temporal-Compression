"""Tucker decomposition verification for dual-gaussian FBT validation.

Analyzes transmission tensor T[probe, light, sh_coeffs] using:
1. Mode-wise SVD analysis (probe, light, SH modes)
2. Tucker decomposition with tensorly
3. Phase gate decision (GO/NO-GO/ADJUST)

Phase Gate Criteria:
- Probe mode: rank-20 captures ≥90% energy
- Light mode: rank-10 captures ≥90% energy
- SH mode: rank-5 captures ≥95% energy
- Tucker reconstruction error: <5%

Usage:
    python verify_tucker_decomposition.py --data ../data_generation/output/transfer_tensor_validation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

try:
    import tensorly as tl
    from tensorly.decomposition import tucker
except ImportError:
    print("Error: tensorly not installed. Run: pip install tensorly")
    sys.exit(1)


def load_transfer_tensor(data_dir: Path):
    """Load transmission tensor from data directory.

    Args:
        data_dir: Directory containing transfer_tensor.npz

    Returns:
        tensor: [P, L, C] transmission tensor
        probe_positions: [P, 3] probe positions
        light_positions: [L, 3] light positions
        metadata: dict with generation parameters
    """
    data_path = data_dir / 'transfer_tensor.npz'

    if not data_path.exists():
        raise FileNotFoundError(f"Tensor file not found: {data_path}")

    data = np.load(data_path, allow_pickle=True)

    tensor = data['tensor']  # [P, L, C]
    probe_positions = data['probe_positions']  # [P, 3]
    light_positions = data['light_positions']  # [L, 3]
    metadata = data['metadata'].item() if 'metadata' in data else {}

    print(f"Loaded tensor from: {data_path}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Probe positions: {probe_positions.shape[0]}")
    print(f"  Light positions: {light_positions.shape[0]}")
    print(f"  SH coefficients: {tensor.shape[2]}")

    return tensor, probe_positions, light_positions, metadata


def analyze_mode_unfolding(
    tensor: np.ndarray,
    mode: int,
    mode_name: str,
    target_rank: int,
    energy_threshold: float = 0.90
):
    """Analyze low-rank structure of specific tensor mode via SVD.

    Args:
        tensor: [P, L, C] transmission tensor
        mode: Which mode to unfold (0=probe, 1=light, 2=sh)
        mode_name: Human-readable name for reporting
        target_rank: Expected rank for phase gate
        energy_threshold: Energy threshold for phase gate (default 0.90)

    Returns:
        analysis: dict with SVD results and phase gate status
    """
    P, L, C = tensor.shape

    # Unfold tensor along specified mode
    if mode == 0:  # Probe mode: [P, L×C]
        matrix = tensor.reshape(P, L * C)
    elif mode == 1:  # Light mode: [L, P×C]
        matrix = np.transpose(tensor, (1, 0, 2)).reshape(L, P * C)
    elif mode == 2:  # SH mode: [C, P×L]
        matrix = np.transpose(tensor, (2, 0, 1)).reshape(C, P * L)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Compute SVD
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    # Compute energy metrics
    energy = S ** 2
    total_energy = energy.sum()
    cumulative_energy = np.cumsum(energy) / total_energy

    # Find rank needed to capture threshold
    actual_rank = int(np.argmax(cumulative_energy >= energy_threshold) + 1)

    # Compute reconstruction error at target rank
    S_truncated = S.copy()
    S_truncated[target_rank:] = 0
    reconstruction_error = np.linalg.norm(S_truncated[target_rank:]) / np.linalg.norm(S)

    # Phase gate decision
    if actual_rank <= target_rank:
        status = "✓ PASS"
        decision = "GO"
    elif actual_rank <= target_rank * 1.5:
        status = "⚠ MARGINAL"
        decision = "ADJUST"
    else:
        status = "✗ FAIL"
        decision = "NO-GO"

    energy_at_target = cumulative_energy[min(target_rank - 1, len(cumulative_energy) - 1)]

    analysis = {
        'mode': mode,
        'mode_name': mode_name,
        'matrix_shape': matrix.shape,
        'singular_values': S,
        'cumulative_energy': cumulative_energy,
        'target_rank': target_rank,
        'actual_rank': actual_rank,
        'energy_threshold': energy_threshold,
        'energy_at_target': float(energy_at_target),
        'reconstruction_error': float(reconstruction_error),
        'status': status,
        'decision': decision
    }

    return analysis


def run_tucker_decomposition(
    tensor: np.ndarray,
    ranks: tuple = (20, 10, 5)
):
    """Run Tucker decomposition using tensorly.

    Args:
        tensor: [P, L, C] transmission tensor
        ranks: (rank_probe, rank_light, rank_sh) target ranks

    Returns:
        decomposition: Tucker decomposition result
        analysis: dict with reconstruction metrics
    """
    print(f"\nRunning Tucker decomposition with ranks {ranks}...")

    # Convert to tensorly tensor
    tensor_tl = tl.tensor(tensor)

    # Tucker decomposition
    core, factors = tucker(tensor_tl, rank=ranks)

    # Reconstruction
    reconstructed = tl.tucker_to_tensor((core, factors))

    # Compute reconstruction error
    error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)
    mae = np.mean(np.abs(tensor - reconstructed))
    max_error = np.max(np.abs(tensor - reconstructed))

    # Compression ratio
    original_size = np.prod(tensor.shape)
    compressed_size = np.prod(core.shape) + sum(f.shape[0] * f.shape[1] for f in factors)
    compression_ratio = original_size / compressed_size

    analysis = {
        'ranks': ranks,
        'core_shape': core.shape,
        'factor_shapes': [f.shape for f in factors],
        'reconstruction_error': float(error),
        'mae': float(mae),
        'max_error': float(max_error),
        'original_size': int(original_size),
        'compressed_size': int(compressed_size),
        'compression_ratio': float(compression_ratio)
    }

    print(f"  Core tensor shape: {core.shape}")
    print(f"  Factor matrices: {[f.shape for f in factors]}")
    print(f"  Reconstruction error: {error:.4f} ({error*100:.2f}%)")
    print(f"  Compression ratio: {compression_ratio:.1f}×")

    return (core, factors), analysis


def make_phase_gate_decision(mode_analyses: list, tucker_analysis: dict):
    """Make final phase gate decision based on all analyses.

    Args:
        mode_analyses: List of analysis dicts from analyze_mode_unfolding()
        tucker_analysis: Analysis dict from run_tucker_decomposition()

    Returns:
        decision: 'GO', 'ADJUST', or 'NO-GO'
        rationale: Explanation string
    """
    # Check mode-wise criteria
    mode_decisions = [a['decision'] for a in mode_analyses]

    # Check Tucker error
    tucker_error = tucker_analysis['reconstruction_error']
    tucker_pass = tucker_error < 0.05  # 5% threshold

    # Decision logic
    if all(d == 'GO' for d in mode_decisions) and tucker_pass:
        decision = 'GO'
        rationale = f"All modes satisfy rank criteria. Tucker error {tucker_error:.3f} < 0.05. Proceed to FBT training."

    elif 'NO-GO' in mode_decisions:
        failed_modes = [a['mode_name'] for a in mode_analyses if a['decision'] == 'NO-GO']
        decision = 'NO-GO'
        rationale = f"Failed modes: {', '.join(failed_modes)}. Low-rank hypothesis invalid. Switch to PG-RGLT fallback."

    elif not tucker_pass:
        decision = 'ADJUST' if tucker_error < 0.10 else 'NO-GO'
        rationale = f"Tucker error {tucker_error:.3f} too high. {'Try higher ranks.' if decision == 'ADJUST' else 'Switch to fallback.'}"

    else:
        decision = 'ADJUST'
        marginal_modes = [a['mode_name'] for a in mode_analyses if a['decision'] == 'ADJUST']
        rationale = f"Marginal performance in: {', '.join(marginal_modes)}. Consider rank adjustment."

    return decision, rationale


def plot_analysis_results(mode_analyses: list, output_dir: Path):
    """Generate visualization plots for SVD analysis.

    Args:
        mode_analyses: List of analysis dicts
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, analysis in enumerate(mode_analyses):
        ax = axes[i]

        # Plot cumulative energy curve
        cumulative_energy = analysis['cumulative_energy']
        target_rank = analysis['target_rank']

        ax.plot(np.arange(1, len(cumulative_energy) + 1), cumulative_energy, 'b-', linewidth=2)
        ax.axhline(y=analysis['energy_threshold'], color='r', linestyle='--', label=f"{analysis['energy_threshold']*100:.0f}% threshold")
        ax.axvline(x=target_rank, color='g', linestyle='--', label=f"Target rank {target_rank}")

        # Mark actual rank
        actual_rank = analysis['actual_rank']
        energy_at_actual = cumulative_energy[actual_rank - 1]
        ax.plot(actual_rank, energy_at_actual, 'ro', markersize=10, label=f"Actual rank {actual_rank}")

        ax.set_xlabel('Rank')
        ax.set_ylabel('Cumulative Energy')
        ax.set_title(f"{analysis['mode_name']} Mode - {analysis['status']}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, min(100, len(cumulative_energy))])
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plot_path = output_dir / 'svd_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved SVD analysis plot: {plot_path}")

    # Plot singular values
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, analysis in enumerate(mode_analyses):
        ax = axes[i]

        S = analysis['singular_values']
        target_rank = analysis['target_rank']

        ax.semilogy(np.arange(1, len(S) + 1), S, 'b-', linewidth=2)
        ax.axvline(x=target_rank, color='g', linestyle='--', label=f"Target rank {target_rank}")
        ax.axvline(x=analysis['actual_rank'], color='r', linestyle='--', label=f"Actual rank {analysis['actual_rank']}")

        ax.set_xlabel('Index')
        ax.set_ylabel('Singular Value (log scale)')
        ax.set_title(f"{analysis['mode_name']} Mode - Singular Values")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, min(100, len(S))])

    plt.tight_layout()
    plot_path = output_dir / 'singular_values.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved singular values plot: {plot_path}")


def generate_report(
    mode_analyses: list,
    tucker_analysis: dict,
    decision: str,
    rationale: str,
    output_dir: Path
):
    """Generate comprehensive phase gate report.

    Args:
        mode_analyses: List of mode analysis dicts
        tucker_analysis: Tucker decomposition analysis
        decision: Final decision ('GO', 'ADJUST', 'NO-GO')
        rationale: Decision rationale string
        output_dir: Directory to save report
    """
    report_path = output_dir / 'phase_gate_report.md'

    with open(report_path, 'w') as f:
        f.write("# Week 1 Phase Gate Report: Tucker Decomposition Validation\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Decision**: **{decision}**\n\n")
        f.write(f"**Rationale**: {rationale}\n\n")

        f.write("---\n\n")
        f.write("## Mode-wise SVD Analysis\n\n")

        for analysis in mode_analyses:
            f.write(f"### {analysis['mode_name']} Mode {analysis['status']}\n\n")
            f.write(f"- **Matrix shape**: {analysis['matrix_shape']}\n")
            f.write(f"- **Target rank**: {analysis['target_rank']}\n")
            f.write(f"- **Actual rank (@ {analysis['energy_threshold']*100:.0f}%)**: {analysis['actual_rank']}\n")
            f.write(f"- **Energy at target rank**: {analysis['energy_at_target']*100:.2f}%\n")
            f.write(f"- **Reconstruction error**: {analysis['reconstruction_error']*100:.2f}%\n")
            f.write(f"- **Decision**: {analysis['decision']}\n\n")

        f.write("---\n\n")
        f.write("## Tucker Decomposition Results\n\n")
        f.write(f"- **Ranks**: {tucker_analysis['ranks']}\n")
        f.write(f"- **Core tensor shape**: {tucker_analysis['core_shape']}\n")
        f.write(f"- **Factor matrices**: {tucker_analysis['factor_shapes']}\n")
        f.write(f"- **Reconstruction error**: {tucker_analysis['reconstruction_error']*100:.2f}%\n")
        f.write(f"- **MAE**: {tucker_analysis['mae']:.4f}\n")
        f.write(f"- **Max error**: {tucker_analysis['max_error']:.4f}\n")
        f.write(f"- **Compression ratio**: {tucker_analysis['compression_ratio']:.1f}×\n\n")

        f.write("---\n\n")
        f.write("## Recommendations\n\n")

        if decision == 'GO':
            f.write("**Proceed with Days 4-5: FBT Prototype Training**\n\n")
            f.write("- Use verified ranks from Tucker decomposition\n")
            f.write("- Initialize factor matrices from SVD (U, V)\n")
            f.write("- Train for 1000 epochs with Adam optimizer\n")
            f.write("- Target validation MAE < 0.15\n")

        elif decision == 'ADJUST':
            f.write("**Attempt rank adjustment before fallback**\n\n")
            marginal_modes = [a for a in mode_analyses if a['decision'] == 'ADJUST']
            for a in marginal_modes:
                suggested_rank = int(a['actual_rank'] * 1.2)
                f.write(f"- **{a['mode_name']} mode**: Try rank {suggested_rank} (current: {a['target_rank']})\n")
            f.write("\nRe-run Tucker decomposition with adjusted ranks. If still marginal, switch to PG-RGLT.\n")

        else:  # NO-GO
            f.write("**Switch to PG-RGLT Fallback Method**\n\n")
            f.write("- Low-rank hypothesis invalid for transmission tensor\n")
            f.write("- Dual-gaussian approach not suitable for this lighting scenario\n")
            f.write("- 4 weeks remaining for PG-RGLT implementation\n")
            f.write("- PG-RGLT provides 79× compression with physical priors\n")

    print(f"\nPhase gate report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Verify Tucker decomposition for FBT validation")
    parser.add_argument('--data', type=str, required=True,
                        help='Data directory containing transfer_tensor.npz')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for analysis results (default: data_dir/tucker_analysis)')
    parser.add_argument('--rank_probe', type=int, default=20,
                        help='Target rank for probe mode (default: 20)')
    parser.add_argument('--rank_light', type=int, default=10,
                        help='Target rank for light mode (default: 10)')
    parser.add_argument('--rank_sh', type=int, default=5,
                        help='Target rank for SH mode (default: 5)')
    parser.add_argument('--energy_probe', type=float, default=0.90,
                        help='Energy threshold for probe mode (default: 0.90)')
    parser.add_argument('--energy_light', type=float, default=0.90,
                        help='Energy threshold for light mode (default: 0.90)')
    parser.add_argument('--energy_sh', type=float, default=0.95,
                        help='Energy threshold for SH mode (default: 0.95)')

    args = parser.parse_args()

    data_dir = Path(args.data)
    output_dir = Path(args.output) if args.output else data_dir / 'tucker_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Tucker Decomposition Verification - Phase Gate Analysis")
    print("=" * 80)

    # Load data
    tensor, probe_positions, light_positions, metadata = load_transfer_tensor(data_dir)

    # Mode-wise SVD analysis
    print("\n" + "=" * 80)
    print("Mode-wise SVD Analysis")
    print("=" * 80)

    mode_configs = [
        (0, 'Probe', args.rank_probe, args.energy_probe),
        (1, 'Light', args.rank_light, args.energy_light),
        (2, 'SH', args.rank_sh, args.energy_sh)
    ]

    mode_analyses = []
    for mode, name, rank, energy in mode_configs:
        print(f"\nAnalyzing {name} mode...")
        analysis = analyze_mode_unfolding(tensor, mode, name, rank, energy)
        mode_analyses.append(analysis)

        print(f"  Status: {analysis['status']}")
        print(f"  Target rank: {analysis['target_rank']}")
        print(f"  Actual rank (@ {energy*100:.0f}%): {analysis['actual_rank']}")
        print(f"  Energy at target: {analysis['energy_at_target']*100:.2f}%")

    # Tucker decomposition
    print("\n" + "=" * 80)
    print("Tucker Decomposition")
    print("=" * 80)

    ranks = (args.rank_probe, args.rank_light, args.rank_sh)
    decomposition, tucker_analysis = run_tucker_decomposition(tensor, ranks)

    # Phase gate decision
    print("\n" + "=" * 80)
    print("Phase Gate Decision")
    print("=" * 80)

    decision, rationale = make_phase_gate_decision(mode_analyses, tucker_analysis)

    print(f"\n{'='*80}")
    print(f"DECISION: {decision}")
    print(f"{'='*80}")
    print(f"{rationale}")
    print(f"{'='*80}\n")

    # Generate visualizations
    plot_analysis_results(mode_analyses, output_dir)

    # Generate report
    generate_report(mode_analyses, tucker_analysis, decision, rationale, output_dir)

    # Save results as JSON
    results = {
        'decision': decision,
        'rationale': rationale,
        'mode_analyses': mode_analyses,
        'tucker_analysis': tucker_analysis,
        'timestamp': datetime.now().isoformat()
    }

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj

    results_native = convert_to_native(results)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_native, f, indent=2)

    print(f"Results saved to: {output_dir / 'results.json'}\n")


if __name__ == '__main__':
    main()
