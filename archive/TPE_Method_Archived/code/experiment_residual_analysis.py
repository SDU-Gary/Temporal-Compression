"""
Experiment 2: Residual Analysis

Analyzes residuals from Experiment F to determine if a hybrid model is needed.

Residual = F_gt(x,t) - F_static(x+ε(t))

If residuals are small (<5% of total error), pure TPE is sufficient.
If residuals are large, a hybrid TPE+residual model may be needed.

Decision: needs_hybrid = (E_tpe_assumption / E_total) > threshold
"""

import numpy as np
from pathlib import Path
import json


class ExperimentResidualAnalysis:
    """Experiment 2: Residual Analysis"""

    def __init__(self, config):
        self.residual_threshold = config['experiment_2']['residual_threshold_percent']
        self.output_dir = Path(config['output_dir'])

    def run(self, experiment_f_results):
        """
        Analyze residuals from Experiment F

        Args:
            experiment_f_results: Results from Experiment F

        Returns:
            dict with residual statistics and decision
        """
        print("\nAnalyzing residuals from Experiment F...")

        configs = experiment_f_results['configurations']

        # Compute residuals: F_gt(x,t) - F_static(x+ε)
        residuals = []
        for config in configs:
            sh_gt = np.array(config['sh_config_b'])
            sh_static = np.array(config['sh_config_a'])
            residual = sh_gt - sh_static
            residuals.append(residual)

        residuals = np.array(residuals)  # [N, 27]

        # Statistical analysis
        results = {
            'mean_abs_residual': float(np.mean(np.abs(residuals))),
            'max_abs_residual': float(np.max(np.abs(residuals))),
            'std_residual': float(np.std(residuals)),
            'min_abs_residual': float(np.min(np.abs(residuals))),
            'residual_percentage': float(experiment_f_results['tpe_contribution_pct']),
            'threshold_percent': float(self.residual_threshold),
            'needs_hybrid': bool(experiment_f_results['tpe_contribution_pct'] > self.residual_threshold)
        }

        # Per-hour residual norms
        results['residual_norms'] = [
            float(np.linalg.norm(residuals[i]))
            for i in range(len(residuals))
        ]

        results['hours_analyzed'] = [config['hour'] for config in configs]

        print(f"\n{'='*70}")
        print(f"RESIDUAL ANALYSIS RESULTS")
        print(f"{'='*70}")

        print(f"\nResidual Statistics:")
        print(f"  Mean |residual|:  {results['mean_abs_residual']:.4f}")
        print(f"  Max |residual|:   {results['max_abs_residual']:.4f}")
        print(f"  Min |residual|:   {results['min_abs_residual']:.4f}")
        print(f"  Std residual:     {results['std_residual']:.4f}")

        print(f"\nTPE Error Contribution:")
        print(f"  TPE contribution to total error: {results['residual_percentage']:.1f}%")
        print(f"  Threshold for hybrid model:      {results['threshold_percent']:.1f}%")

        print(f"\nPer-Hour Residual Norms:")
        for hour, norm in zip(results['hours_analyzed'], results['residual_norms']):
            print(f"  Hour {hour:2d}: ||residual|| = {norm:.4f}")

        print(f"\n{'='*70}")
        print(f"DECISION")
        print(f"{'='*70}")

        if results['needs_hybrid']:
            print(f"\n✗ Hybrid Model Recommended")
            print(f"  TPE error ({results['residual_percentage']:.1f}%) exceeds threshold ({results['threshold_percent']:.1f}%)")
            print(f"  Consider: TPE + small residual network")
        else:
            print(f"\n✓ Pure TPE Sufficient")
            print(f"  TPE error ({results['residual_percentage']:.1f}%) below threshold ({results['threshold_percent']:.1f}%)")
            print(f"  No residual correction needed")

        # Save results
        output_file = self.output_dir / 'experiment_2_results.json'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

        return results
