"""
Experiment 3: Spatial Generalization Validation

Tests TPE on multiple spatially distributed probes to verify spatial generalization.

Goal: Confirm TPE works across different spatial locations, not just the single
probe used in initial validation.

Note: Requires dataset_2k with per-probe data. If not available, experiment is skipped.
"""

import numpy as np
from pathlib import Path
import json
import sys
from concurrent.futures import ProcessPoolExecutor

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ExperimentSpatialValidation:
    """Experiment 3: Spatial Generalization Validation"""

    def __init__(self, config):
        self.dataset_2k_path = Path(config['dataset_2k_path'])
        self.num_probes = config['experiment_3']['num_probes']
        self.max_workers = config['experiment_3']['max_workers']
        self.direction_mode = config['experiment_3']['direction_mode']
        self.output_dir = Path(config['output_dir'])

    def check_dataset_availability(self):
        """Check if dataset_2k exists and has required structure"""
        if not self.dataset_2k_path.exists():
            return False, "Dataset not found"

        # Check for probes.npz
        probes_file = self.dataset_2k_path / 'probes.npz'
        if not probes_file.exists():
            return False, "probes.npz not found"

        return True, "Dataset available"

    def select_probes(self):
        """Select spatially distributed probes from dataset_2k"""
        # Load probe positions from dataset_2k
        probes_file = self.dataset_2k_path / 'probes.npz'
        probe_data = np.load(probes_file)

        if 'positions' not in probe_data:
            raise KeyError("'positions' key not found in probes.npz")

        all_probes = probe_data['positions']

        print(f"  Total probes available: {len(all_probes)}")

        # Grid-based selection for spatial coverage
        selected_indices = np.linspace(0, len(all_probes)-1, self.num_probes, dtype=int)

        return [(i, all_probes[i]) for i in selected_indices]

    def validate_single_probe(self, probe_idx, probe_position):
        """
        Run directional TPE validation on one probe

        Note: This assumes dataset_2k has per-probe subdirectories.
        If not, this function will return an error.
        """
        # Import here to avoid multiprocessing issues
        from validate_tpe_level2_directional import validate_directional_tpe

        # Create probe dataset path
        probe_dataset = self.dataset_2k_path / f'probe_{probe_idx}'

        if not probe_dataset.exists():
            return {
                'probe_idx': int(probe_idx),
                'probe_position': probe_position.tolist(),
                'success': False,
                'error': f'Dataset not found: {probe_dataset}'
            }

        try:
            success, results = validate_directional_tpe(
                dataset_path=probe_dataset,
                direction_mode=self.direction_mode,
                reference_hour=12
            )

            return {
                'probe_idx': int(probe_idx),
                'probe_position': probe_position.tolist(),
                'success': True,
                'all_pass': results['all_pass'],
                'max_norm': results['max_perturbation_norm'],
                'avg_smoothness': results['avg_temporal_smoothness'],
                'correlation': results['sun_correlation']
            }
        except Exception as e:
            return {
                'probe_idx': int(probe_idx),
                'probe_position': probe_position.tolist(),
                'success': False,
                'error': str(e)
            }

    def run(self):
        """Execute Experiment 3"""
        print("\nChecking dataset_2k availability...")
        available, message = self.check_dataset_availability()

        if not available:
            print(f"  ✗ {message}")
            print(f"  Skipping Experiment 3 (dataset_2k not available)")

            results = {
                'skipped': True,
                'reason': message,
                'num_probes': 0,
                'num_successful': 0,
                'num_passed': 0,
                'success_rate': 0.0,
                'pass_rate': 0.0
            }

            # Save results
            output_file = self.output_dir / 'experiment_3_results.json'
            self.output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            return results

        print(f"  ✓ Dataset available: {self.dataset_2k_path}")

        print(f"\nSelecting {self.num_probes} spatially distributed probes...")
        try:
            selected_probes = self.select_probes()
        except Exception as e:
            print(f"  ✗ Error selecting probes: {e}")
            results = {
                'skipped': True,
                'reason': f'Error selecting probes: {e}',
                'num_probes': 0
            }

            output_file = self.output_dir / 'experiment_3_results.json'
            self.output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            return results

        print(f"  Selected probe indices: {[idx for idx, _ in selected_probes]}")

        print(f"\nRunning TPE validation on {self.num_probes} probes with {self.max_workers} workers...")
        print(f"  (This may take 30-60 minutes)")

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.validate_single_probe, idx, pos)
                for idx, pos in selected_probes
            ]
            results_list = [f.result() for f in futures]

        # Aggregate statistics
        successful = [r for r in results_list if r.get('success', False)]
        passed = [r for r in successful if r.get('all_pass', False)]

        results = {
            'skipped': False,
            'num_probes': self.num_probes,
            'num_successful': len(successful),
            'num_passed': len(passed),
            'success_rate': len(successful) / self.num_probes if self.num_probes > 0 else 0.0,
            'pass_rate': len(passed) / self.num_probes if self.num_probes > 0 else 0.0,
            'avg_max_norm': float(np.mean([r['max_norm'] for r in successful])) if successful else None,
            'individual_results': results_list
        }

        print(f"\n{'='*70}")
        print(f"SPATIAL VALIDATION RESULTS")
        print(f"{'='*70}")

        print(f"\nSummary:")
        print(f"  Probes tested:            {results['num_probes']}")
        print(f"  Successful validations:   {results['num_successful']}/{results['num_probes']}")
        print(f"  Passed all criteria:      {results['num_passed']}/{results['num_probes']}")
        print(f"  Success rate:             {results['success_rate']*100:.1f}%")
        print(f"  Pass rate:                {results['pass_rate']*100:.1f}%")

        if results['avg_max_norm'] is not None:
            print(f"  Average ||ε||_max:        {results['avg_max_norm']:.3f}m")

        # Decision criterion
        success_criterion = results['pass_rate'] >= 0.7  # 70% pass rate

        print(f"\n{'='*70}")
        print(f"DECISION")
        print(f"{'='*70}")

        if success_criterion:
            print(f"\n✓ Spatial Generalization Confirmed")
            print(f"  Pass rate ({results['pass_rate']*100:.1f}%) meets threshold (≥70%)")
        else:
            print(f"\n✗ Spatial Generalization Not Confirmed")
            print(f"  Pass rate ({results['pass_rate']*100:.1f}%) below threshold (≥70%)")

        # Save results
        output_file = self.output_dir / 'experiment_3_results.json'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

        return results
