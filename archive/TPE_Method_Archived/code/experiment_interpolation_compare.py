"""
Experiment 4: Time Interpolation Baseline Comparison

Compares TPE with traditional baseline methods (Spline, DirectMLP, TDML)
on time interpolation task using odd/even hour cross-validation.

Goal: Demonstrate TPE's advantage over simpler time-based methods.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.tpe_dataset_loader import TPEValidationDataset
from models.gaussian_mixture import GaussianMixture
from models.decoder_mlp import DecoderMLP


class ExperimentInterpolationCompare:
    """Experiment 4: Time Interpolation Baseline Comparison"""

    def __init__(self, config):
        self.dataset_path = Path(config['dataset_path'])
        self.baselines = config['experiment_4']['baselines']
        self.cv_split = config['experiment_4']['cross_validation']
        self.output_dir = Path(config['output_dir'])

    def load_dataset(self):
        """Load cornell-box dataset"""
        print("  Loading dataset...")
        dataset = TPEValidationDataset(self.dataset_path, level=2)
        data = dataset.get_probe_data(0)
        print(f"    Loaded {len(data['hours'])} time steps")
        return data

    def split_train_test(self, hours, sh_gt):
        """
        Split into train (odd hours) and test (even hours)

        Args:
            hours: [T] array of hours
            sh_gt: [T, 27] array of SH coefficients

        Returns:
            dict with train/test splits
        """
        hours_arr = np.array(hours)
        train_mask = hours_arr % 2 == 1
        test_mask = ~train_mask

        return {
            'train_hours': hours_arr[train_mask],
            'train_sh': sh_gt[train_mask],
            'test_hours': hours_arr[test_mask],
            'test_sh': sh_gt[test_mask]
        }

    def evaluate_baseline(self, name, baseline, split):
        """
        Train and evaluate a baseline

        Args:
            name: Baseline name
            baseline: Baseline object
            split: Train/test split dict

        Returns:
            dict with mae and rmse
        """
        print(f"\n  Evaluating {name}...")

        # Train
        print(f"    Training on {len(split['train_hours'])} odd hours...")

        if hasattr(baseline, 'train_model'):
            baseline.train_model(split['train_hours'], split['train_sh'])
        else:
            baseline.train(split['train_hours'], split['train_sh'])

        # Test
        print(f"    Testing on {len(split['test_hours'])} even hours...")
        sh_pred = baseline.predict(split['test_hours'])

        # Metrics
        mae = np.mean(np.abs(sh_pred - split['test_sh']))
        rmse = np.sqrt(np.mean((sh_pred - split['test_sh'])**2))

        print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        return {'mae': float(mae), 'rmse': float(rmse)}

    def evaluate_tpe(self, split, data):
        """
        Evaluate TPE on test set

        Args:
            split: Train/test split dict
            data: Full dataset

        Returns:
            dict with mae and rmse
        """
        print(f"\n  Evaluating TPE...")

        # Load TPE results
        tpe_file = self.dataset_path / 'validation_results' / 'tpe_directional_validation_results.npz'
        if not tpe_file.exists():
            print(f"    Warning: TPE results not found at {tpe_file}")
            return {'mae': float('inf'), 'rmse': float('inf'), 'error': 'TPE results not found'}

        tpe_data = np.load(tpe_file)

        # Load or train TPE model
        weights_file = self.dataset_path / 'validation_results' / 'tpe_model_weights.pt'

        if weights_file.exists():
            print(f"    Loading TPE model weights...")
            weights = torch.load(weights_file)
            gaussian_mixture = GaussianMixture(num_gaussians=1, latent_dim=9)
            decoder = DecoderMLP(input_dim=15, hidden_dim=64, num_layers=2, output_dim=27)

            gaussian_mixture.load_state_dict(weights['gaussian_mixture'])
            decoder.load_state_dict(weights['decoder'])
        else:
            print(f"    Warning: TPE model weights not found")
            return {'mae': float('inf'), 'rmse': float('inf'), 'error': 'Model weights not found'}

        # Get predictions for test hours
        probe_pos = torch.tensor(tpe_data['probe_position'], dtype=torch.float32)
        sun_ref = torch.tensor(tpe_data['sun_dirs'][tpe_data['reference_idx']], dtype=torch.float32)
        perturbations = torch.tensor(tpe_data['perturbations'], dtype=torch.float32)
        hours_all = tpe_data['hours']

        sh_pred_list = []
        sh_gt_list = []

        for test_hour in split['test_hours']:
            # Find hour index
            hour_idx = list(hours_all).index(test_hour)

            # Compute TPE prediction
            pos_perturbed = probe_pos + perturbations[hour_idx]
            latent = gaussian_mixture(pos_perturbed.unsqueeze(0))
            sh_pred = decoder(latent, pos_perturbed.unsqueeze(0), sun_ref)

            sh_pred_list.append(sh_pred.squeeze(0).detach().numpy())

            # Get ground truth
            test_idx = list(split['test_hours']).index(test_hour)
            sh_gt_list.append(split['test_sh'][test_idx])

        sh_pred_arr = np.array(sh_pred_list)
        sh_gt_arr = np.array(sh_gt_list)

        # Metrics
        mae = np.mean(np.abs(sh_pred_arr - sh_gt_arr))
        rmse = np.sqrt(np.mean((sh_pred_arr - sh_gt_arr)**2))

        print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        return {'mae': float(mae), 'rmse': float(rmse)}

    def run(self):
        """
        Execute Experiment 4

        Returns:
            dict with results for all baselines and TPE
        """
        print("\nLoading dataset...")
        data = self.load_dataset()

        print("\nSplitting train/test (odd/even hours)...")
        split = self.split_train_test(data['hours'], data['sh_gt'])

        print(f"  Train: {split['train_hours']} ({len(split['train_hours'])} hours)")
        print(f"  Test: {split['test_hours']} ({len(split['test_hours'])} hours)")

        results = {}

        # Evaluate baselines
        for baseline_name in self.baselines:
            if baseline_name == 'spline':
                from experiments.baselines.spline_baseline import SplineBaseline
                baseline = SplineBaseline()
            elif baseline_name == 'direct_mlp':
                from experiments.baselines.direct_mlp_baseline import DirectMLPBaseline
                baseline = DirectMLPBaseline()
            elif baseline_name == 'tdml':
                from experiments.baselines.tdml_baseline import TDMLBaseline
                baseline = TDMLBaseline()
            else:
                print(f"  Unknown baseline: {baseline_name}")
                continue

            results[baseline_name] = self.evaluate_baseline(baseline_name, baseline, split)

        # Evaluate TPE
        results['tpe'] = self.evaluate_tpe(split, data)

        # Ranking
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        sorted_by_mae = sorted(valid_results.items(), key=lambda x: x[1]['mae'])

        print(f"\n{'='*70}")
        print("BASELINE COMPARISON RESULTS")
        print(f"{'='*70}")
        print(f"\n  Ranking by MAE:")
        for i, (name, metrics) in enumerate(sorted_by_mae):
            print(f"    {i+1}. {name:12s}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

        results['ranking_mae'] = [name for name, _ in sorted_by_mae]
        results['split_info'] = {
            'train_hours': split['train_hours'].tolist(),
            'test_hours': split['test_hours'].tolist()
        }

        # Save results
        output_file = self.output_dir / 'experiment_4_results.json'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

        return results
