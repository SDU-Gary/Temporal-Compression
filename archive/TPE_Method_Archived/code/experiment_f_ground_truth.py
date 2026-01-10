"""
Experiment F: Ground Truth Validation

Verifies TPE physical assumption by rendering ground truth SH at perturbed positions.

Critical test: Does F_static(x + ε(t)) ≈ F_gt(x, t) hold?

Renders two configurations for each hour:
- Config A: F_static(x + ε(t)) with reference sun
- Config B: F_gt(x, t) with hour-specific sun

Computes error decomposition:
- E_generalization: ||F_static(x+ε) - F_tpe_pred(x+ε)||
- E_tpe_assumption: ||F_gt(x,t) - F_static(x+ε)||
- E_total: ||F_gt(x,t) - F_tpe_pred(x+ε)||
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import importlib.util

# Add parent directories to path
multi_time_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(multi_time_path))

from utils.tpe_dataset_loader import TPEValidationDataset
from models.gaussian_mixture import GaussianMixture
from models.decoder_mlp import DecoderMLP

# Import from data_generation package using absolute path
# multi_time_path is /home/kyrie/毕设/multi_time_compression
# We need /home/kyrie/毕设/data_generation
data_gen_path = multi_time_path.parent / 'data_generation'
if not data_gen_path.exists():
    raise FileNotFoundError(f"data_generation path not found: {data_gen_path}")

sys.path.insert(0, str(data_gen_path))

# Try importing with error handling
try:
    from core.sh_baker import bake_sh_at_probe
    from core.lighting_modifier import LightingModifier
except ImportError as e:
    print(f"Import error: {e}")
    print(f"data_gen_path: {data_gen_path}")
    print(f"sys.path: {sys.path[:5]}")
    raise


class ExperimentF:
    """Experiment F: Ground Truth Validation"""

    def __init__(self, config):
        self.dataset_path = Path(config['dataset_path'])
        self.scene_path = Path(config['scene_path'])
        self.selected_hours = config['experiment_f']['selected_hours']
        self.spp = config['experiment_f']['spp']
        self.num_samples = config['experiment_f']['num_samples']

    def load_tpe_results(self):
        """Load perturbations from directional TPE validation"""
        results_file = self.dataset_path / 'validation_results' / 'tpe_directional_validation_results.npz'

        if not results_file.exists():
            raise FileNotFoundError(f"TPE validation results not found: {results_file}")

        print(f"  Loading TPE validation results from: {results_file}")
        data = np.load(results_file)

        return {
            'perturbations': data['perturbations'],  # [T, 3]
            'alphas': data['alphas'],  # [T]
            'direction': data['direction'],  # [3]
            'probe_position': data['probe_position'],  # [3]
            'sun_dirs': data['sun_dirs'],  # [T, 3]
            'hours': data['hours'],  # [T]
            'reference_idx': int(data['reference_idx'])
        }

    def load_or_train_tpe_model(self, tpe_data):
        """Load TPE model weights or re-train if not available"""
        weights_file = self.dataset_path / 'validation_results' / 'tpe_model_weights.pt'

        if weights_file.exists():
            # Load existing weights
            print(f"  Loading TPE model weights from: {weights_file}")
            weights = torch.load(weights_file)

            gaussian_mixture = GaussianMixture(num_gaussians=1, latent_dim=9)
            decoder = DecoderMLP(input_dim=15, hidden_dim=64, num_layers=2, output_dim=27)

            gaussian_mixture.load_state_dict(weights['gaussian_mixture'])
            decoder.load_state_dict(weights['decoder'])

            print(f"  ✓ TPE model weights loaded successfully")

        else:
            # Re-train static field (same as validate_tpe_level2_directional.py:114-140)
            print(f"  TPE model weights not found, re-training static field...")

            probe_pos = torch.tensor(tpe_data['probe_position'], dtype=torch.float32)
            sun_ref = torch.tensor(tpe_data['sun_dirs'][tpe_data['reference_idx']], dtype=torch.float32)

            # Load ground truth SH for reference hour
            dataset = TPEValidationDataset(self.dataset_path, level=2)
            data = dataset.get_probe_data(0)
            sh_ref = torch.tensor(data['sh_gt'][tpe_data['reference_idx']], dtype=torch.float32)

            gaussian_mixture = GaussianMixture(num_gaussians=1, latent_dim=9, init_scale=0.5)
            with torch.no_grad():
                gaussian_mixture.means.data[0] = probe_pos
                gaussian_mixture.scales.data = torch.log(torch.ones(1, 3) * 0.5)

            decoder = DecoderMLP(input_dim=15, hidden_dim=64, num_layers=2, output_dim=27)

            optimizer = torch.optim.Adam(
                list(gaussian_mixture.parameters()) + list(decoder.parameters()),
                lr=0.001
            )

            print(f"    Training for 1000 steps...")
            for step in range(1000):
                latent = gaussian_mixture(probe_pos.unsqueeze(0))
                sh_pred = decoder(latent, probe_pos.unsqueeze(0), sun_ref)
                loss = F.mse_loss(sh_pred, sh_ref.unsqueeze(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step + 1) % 200 == 0:
                    print(f"      Step {step+1}/1000: Loss = {loss.item():.6f}")

            # Save weights for future use
            torch.save({
                'gaussian_mixture': gaussian_mixture.state_dict(),
                'decoder': decoder.state_dict()
            }, weights_file)

            print(f"  ✓ Static field trained and saved to {weights_file}")

        return gaussian_mixture, decoder

    def get_tpe_prediction(self, gaussian_mixture, decoder, probe_pos, perturbation, sun_ref):
        """Get TPE prediction: F_static(x + ε(t))"""
        pos_perturbed = probe_pos + perturbation
        latent = gaussian_mixture(pos_perturbed.unsqueeze(0))
        sh_pred = decoder(latent, pos_perturbed.unsqueeze(0), sun_ref)
        return sh_pred.squeeze(0).detach().numpy()

    def render_sh_at_position(self, position, sun_direction):
        """
        Render SH coefficients at given position and sun direction

        Uses LightingModifier to change sun direction dynamically.
        """
        # Set Mitsuba variant before loading scene
        import mitsuba as mi
        try:
            mi.set_variant('cuda_ad_rgb')
        except:
            mi.set_variant('llvm_ad_rgb')

        # Use LightingModifier to change sun direction
        scene = LightingModifier.modify_scene_lighting(self.scene_path, sun_direction)

        # Bake SH
        sh_coeffs = bake_sh_at_probe(
            scene,
            probe_position=position,
            num_samples=self.num_samples,
            spp=self.spp
        )

        return sh_coeffs  # [27]

    def run(self):
        """Execute Experiment F"""
        print("\nLoading TPE validation results...")
        tpe_data = self.load_tpe_results()

        print("\nLoading TPE model...")
        gaussian_mixture, decoder = self.load_or_train_tpe_model(tpe_data)

        probe_pos = torch.tensor(tpe_data['probe_position'], dtype=torch.float32)
        sun_ref = torch.tensor(tpe_data['sun_dirs'][tpe_data['reference_idx']], dtype=torch.float32)

        results = {
            'scene_name': 'cornell-box',
            'scene_path': str(self.scene_path),
            'selected_hours': self.selected_hours,
            'probe_position': tpe_data['probe_position'].tolist(),
            'reference_hour': int(tpe_data['hours'][tpe_data['reference_idx']]),
            'configurations': []
        }

        print(f"\n{'='*70}")
        print(f"RENDERING GROUND TRUTH CONFIGURATIONS")
        print(f"{'='*70}")
        print(f"\nScene: {self.scene_path}")
        print(f"Probe position: {tpe_data['probe_position']}")
        print(f"Reference hour: {results['reference_hour']}")
        print(f"Selected hours for validation: {self.selected_hours}")
        print(f"Rendering quality: spp={self.spp}, sh_samples={self.num_samples}")
        print(f"\nThis will take approximately {len(self.selected_hours) * 2 * 5} minutes...")

        for hour in self.selected_hours:
            hour_idx = list(tpe_data['hours']).index(hour)

            print(f"\n{'-'*70}")
            print(f"Hour {hour}:00")
            print(f"{'-'*70}")

            # Configuration A: F_static(x + ε(t)) with reference sun
            pos_perturbed = tpe_data['probe_position'] + tpe_data['perturbations'][hour_idx]
            sun_ref_np = tpe_data['sun_dirs'][tpe_data['reference_idx']]

            print(f"  Config A: Rendering at perturbed position with reference sun...")
            print(f"    Position: {pos_perturbed}")
            print(f"    Perturbation: {tpe_data['perturbations'][hour_idx]}")
            print(f"    Perturbation norm: {np.linalg.norm(tpe_data['perturbations'][hour_idx]):.3f}m")
            print(f"    Sun direction (reference): {sun_ref_np}")

            sh_config_a = self.render_sh_at_position(pos_perturbed, sun_ref_np)

            print(f"    ✓ Rendered Config A")

            # Configuration B: F_gt(x, t) with hour sun
            sun_hour = tpe_data['sun_dirs'][hour_idx]

            print(f"  Config B: Rendering at original position with hour sun...")
            print(f"    Position: {tpe_data['probe_position']}")
            print(f"    Sun direction (hour {hour}): {sun_hour}")

            sh_config_b = self.render_sh_at_position(tpe_data['probe_position'], sun_hour)

            print(f"    ✓ Rendered Config B")

            # TPE prediction
            print(f"  Computing TPE prediction...")
            perturbation = torch.tensor(tpe_data['perturbations'][hour_idx], dtype=torch.float32)
            sh_tpe_pred = self.get_tpe_prediction(gaussian_mixture, decoder, probe_pos, perturbation, sun_ref)

            # Compute errors
            E_generalization = np.linalg.norm(sh_config_a - sh_tpe_pred)
            E_tpe_assumption = np.linalg.norm(sh_config_b - sh_config_a)
            E_total = np.linalg.norm(sh_config_b - sh_tpe_pred)

            print(f"\n  Error Analysis:")
            print(f"    E_generalization (model error):    {E_generalization:.4f}")
            print(f"    E_tpe_assumption (TPE error):      {E_tpe_assumption:.4f}")
            print(f"    E_total (total error):             {E_total:.4f}")
            print(f"    TPE contribution: {E_tpe_assumption/E_total*100:.1f}% of total error")

            results['configurations'].append({
                'hour': int(hour),
                'sh_config_a': sh_config_a.tolist(),
                'sh_config_b': sh_config_b.tolist(),
                'sh_tpe_pred': sh_tpe_pred.tolist(),
                'E_generalization': float(E_generalization),
                'E_tpe_assumption': float(E_tpe_assumption),
                'E_total': float(E_total),
                'perturbation': tpe_data['perturbations'][hour_idx].tolist(),
                'perturbation_norm': float(np.linalg.norm(tpe_data['perturbations'][hour_idx]))
            })

        # Aggregate statistics
        results['avg_E_gen'] = np.mean([c['E_generalization'] for c in results['configurations']])
        results['avg_E_tpe'] = np.mean([c['E_tpe_assumption'] for c in results['configurations']])
        results['avg_E_total'] = np.mean([c['E_total'] for c in results['configurations']])
        results['tpe_contribution_pct'] = results['avg_E_tpe'] / results['avg_E_total'] * 100

        print(f"\n{'='*70}")
        print(f"EXPERIMENT F SUMMARY")
        print(f"{'='*70}")
        print(f"\nAverage Errors:")
        print(f"  E_generalization (model error):    {results['avg_E_gen']:.4f}")
        print(f"  E_tpe_assumption (TPE error):      {results['avg_E_tpe']:.4f}")
        print(f"  E_total (total error):             {results['avg_E_total']:.4f}")
        print(f"\nTPE Contribution to Total Error: {results['tpe_contribution_pct']:.1f}%")

        # Decision criterion
        success = results['tpe_contribution_pct'] < 50.0

        if success:
            print(f"\n✓✓✓ SUCCESS ✓✓✓")
            print(f"TPE assumption is VALID (contributes <50% of error)")
        else:
            print(f"\n✗✗✗ FAILURE ✗✗✗")
            print(f"TPE assumption is INVALID (contributes ≥50% of error)")

        # Save results
        output_dir = self.dataset_path / 'validation_results'
        output_file = output_dir / 'experiment_f_results.npz'

        np.savez_compressed(
            output_file,
            **results
        )

        print(f"\n✓ Results saved to: {output_file}")

        return results
