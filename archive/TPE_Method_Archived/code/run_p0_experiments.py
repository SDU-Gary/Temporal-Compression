"""
P0 Experiments Pipeline Main Orchestrator

One-click execution of all 4 P0 experiments:
- Experiment F: Ground Truth Validation
- Experiment 2: Residual Analysis
- Experiment 3: Spatial Generalization
- Experiment 4: Baseline Comparison

Features:
- Sequential execution with dependency handling
- Checkpoint/resume capability
- Error handling and logging
- Final report generation

Usage:
    python run_p0_experiments.py --config config/p0_experiments_config.yaml
    python run_p0_experiments.py --config config/p0_experiments_config.yaml --resume
"""

import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
import traceback
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_file):
    """Load checkpoint if exists"""
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_file, results):
    """Save checkpoint"""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def log_error(error_log, experiment_name, error):
    """Log error to file"""
    error_log.parent.mkdir(parents=True, exist_ok=True)
    with open(error_log, 'a') as f:
        timestamp = datetime.now().isoformat()
        f.write(f"\n{'='*70}\n")
        f.write(f"[{timestamp}] {experiment_name} ERROR\n")
        f.write(f"{'='*70}\n")
        f.write(f"{error}\n")
        f.write(f"{traceback.format_exc()}\n")


def generate_final_report(results, config, output_path):
    """Generate markdown final report"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# P0 Experiments Final Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n\n")

        # Check which experiments completed
        completed = [key for key in results.keys() if key.startswith('experiment_')]
        f.write(f"**Completed Experiments**: {len(completed)}/4\n\n")

        # Experiment F
        if 'experiment_f' in results:
            f.write("### Experiment F: Ground Truth Validation\n\n")
            exp_f = results['experiment_f']
            f.write(f"- **Status**: ✓ Completed\n")
            f.write(f"- **Configurations Rendered**: {len(exp_f.get('configurations', []))}\n")
            f.write(f"- **Average E_generalization**: {exp_f.get('avg_E_gen', 'N/A'):.4f}\n")
            f.write(f"- **Average E_tpe_assumption**: {exp_f.get('avg_E_tpe', 'N/A'):.4f}\n")
            f.write(f"- **Average E_total**: {exp_f.get('avg_E_total', 'N/A'):.4f}\n")
            f.write(f"- **TPE Contribution**: {exp_f.get('tpe_contribution_pct', 'N/A'):.1f}%\n")
            f.write(f"- **Decision**: {'✓ VALID' if exp_f.get('tpe_contribution_pct', 100) < 50 else '✗ INVALID'}\n\n")
        else:
            f.write("### Experiment F: Ground Truth Validation\n\n")
            f.write("- **Status**: ✗ Not completed\n\n")

        # Experiment 2
        if 'experiment_2' in results:
            f.write("### Experiment 2: Residual Analysis\n\n")
            exp_2 = results['experiment_2']
            f.write(f"- **Status**: ✓ Completed\n")
            f.write(f"- **Mean |residual|**: {exp_2.get('mean_abs_residual', 'N/A'):.4f}\n")
            f.write(f"- **TPE Contribution**: {exp_2.get('residual_percentage', 'N/A'):.1f}%\n")
            f.write(f"- **Threshold**: {exp_2.get('threshold_percent', 'N/A'):.1f}%\n")
            f.write(f"- **Hybrid Model Needed**: {'YES' if exp_2.get('needs_hybrid', False) else 'NO'}\n\n")
        else:
            f.write("### Experiment 2: Residual Analysis\n\n")
            f.write("- **Status**: ✗ Not completed\n\n")

        # Experiment 3
        if 'experiment_3' in results:
            f.write("### Experiment 3: Spatial Generalization\n\n")
            exp_3 = results['experiment_3']

            if exp_3.get('skipped', False):
                f.write(f"- **Status**: ⊘ Skipped\n")
                f.write(f"- **Reason**: {exp_3.get('reason', 'Unknown')}\n\n")
            else:
                f.write(f"- **Status**: ✓ Completed\n")
                f.write(f"- **Probes Tested**: {exp_3.get('num_probes', 0)}\n")
                f.write(f"- **Successful**: {exp_3.get('num_successful', 0)}/{exp_3.get('num_probes', 0)}\n")
                f.write(f"- **Passed All Criteria**: {exp_3.get('num_passed', 0)}/{exp_3.get('num_probes', 0)}\n")
                f.write(f"- **Pass Rate**: {exp_3.get('pass_rate', 0)*100:.1f}%\n")
                f.write(f"- **Decision**: {'✓ Confirmed' if exp_3.get('pass_rate', 0) >= 0.7 else '✗ Not Confirmed'}\n\n")
        else:
            f.write("### Experiment 3: Spatial Generalization\n\n")
            f.write("- **Status**: ✗ Not completed\n\n")

        # Experiment 4
        if 'experiment_4' in results:
            f.write("### Experiment 4: Baseline Comparison\n\n")
            exp_4 = results['experiment_4']
            f.write(f"- **Status**: ✓ Completed\n")
            f.write(f"- **Baselines Tested**: {len([k for k in exp_4.keys() if k not in ['ranking_mae', 'split_info']])}\n")

            if 'ranking_mae' in exp_4:
                f.write(f"- **Ranking by MAE**:\n")
                for i, method in enumerate(exp_4['ranking_mae']):
                    mae = exp_4.get(method, {}).get('mae', 'N/A')
                    f.write(f"  {i+1}. {method}: {mae:.4f}\n")
            f.write("\n")
        else:
            f.write("### Experiment 4: Baseline Comparison\n\n")
            f.write("- **Status**: ✗ Not completed\n\n")

        # Overall conclusion
        f.write("## Overall Conclusion\n\n")

        if len(completed) == 4:
            f.write("✓ All P0 experiments completed successfully.\n\n")
        elif len(completed) >= 2:
            f.write(f"⚠ Partial completion: {len(completed)}/4 experiments completed.\n\n")
        else:
            f.write(f"✗ Minimal completion: Only {len(completed)}/4 experiments completed.\n\n")

        # Next steps
        f.write("## Next Steps\n\n")
        if 'experiment_f' in results and results['experiment_f'].get('tpe_contribution_pct', 100) < 50:
            f.write("- ✓ TPE assumption validated\n")
        else:
            f.write("- Consider alternative approaches or TPE modifications\n")

        if 'experiment_2' in results and not results['experiment_2'].get('needs_hybrid', True):
            f.write("- ✓ Pure TPE sufficient, no hybrid model needed\n")
        else:
            f.write("- Consider hybrid TPE + residual model\n")

    print(f"\n✓ Final report generated: {output_path}")


def main():
    """Main orchestrator"""
    parser = argparse.ArgumentParser(
        description='P0 Experiments Pipeline - One-click execution of all 4 P0 experiments'
    )

    parser.add_argument(
        '--config',
        default='config/p0_experiments_config.yaml',
        help='Path to configuration file (default: config/p0_experiments_config.yaml)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint'
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"P0 EXPERIMENTS PIPELINE")
    print(f"{'='*70}")
    print(f"\nConfiguration: {config_path}")

    config = load_config(config_path)

    # Setup directories
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = output_dir / config['checkpoint_file']
    error_log = output_dir / 'p0_errors.log'

    # Load checkpoint if resuming
    if args.resume:
        results = load_checkpoint(checkpoint_file)
        print(f"Resuming from checkpoint: {checkpoint_file}")
        print(f"  Already completed: {list(results.keys())}")
    else:
        results = {}
        print(f"Starting fresh run")

    start_time = datetime.now()

    # ==================================================================
    # EXPERIMENT F: GROUND TRUTH VALIDATION
    # ==================================================================

    if 'experiment_f' not in results:
        try:
            print(f"\n{'='*70}")
            print("EXPERIMENT F: GROUND TRUTH VALIDATION")
            print(f"{'='*70}")

            from experiments.experiment_f_ground_truth import ExperimentF
            exp_f = ExperimentF(config)
            results['experiment_f'] = exp_f.run()

            save_checkpoint(checkpoint_file, results)
            print(f"\n✓ Experiment F completed successfully")

        except Exception as e:
            log_error(error_log, 'experiment_f', e)
            print(f"\n✗ Experiment F failed: {e}")
            print(f"  Error logged to: {error_log}")
            print(f"  Continuing to next experiment...")
    else:
        print(f"\n⊙ Experiment F already completed (skipping)")

    # ==================================================================
    # EXPERIMENT 2: RESIDUAL ANALYSIS (depends on F)
    # ==================================================================

    if 'experiment_2' not in results and 'experiment_f' in results:
        try:
            print(f"\n{'='*70}")
            print("EXPERIMENT 2: RESIDUAL ANALYSIS")
            print(f"{'='*70}")

            from experiments.experiment_residual_analysis import ExperimentResidualAnalysis
            exp_2 = ExperimentResidualAnalysis(config)
            results['experiment_2'] = exp_2.run(results['experiment_f'])

            save_checkpoint(checkpoint_file, results)
            print(f"\n✓ Experiment 2 completed successfully")

        except Exception as e:
            log_error(error_log, 'experiment_2', e)
            print(f"\n✗ Experiment 2 failed: {e}")
            print(f"  Error logged to: {error_log}")
            print(f"  Continuing to next experiment...")
    elif 'experiment_2' in results:
        print(f"\n⊙ Experiment 2 already completed (skipping)")
    else:
        print(f"\n⊘ Experiment 2 skipped (depends on Experiment F)")

    # ==================================================================
    # EXPERIMENT 3: SPATIAL GENERALIZATION (independent)
    # ==================================================================

    if 'experiment_3' not in results:
        try:
            print(f"\n{'='*70}")
            print("EXPERIMENT 3: SPATIAL GENERALIZATION")
            print(f"{'='*70}")

            from experiments.experiment_spatial_validation import ExperimentSpatialValidation
            exp_3 = ExperimentSpatialValidation(config)
            results['experiment_3'] = exp_3.run()

            save_checkpoint(checkpoint_file, results)
            print(f"\n✓ Experiment 3 completed successfully")

        except Exception as e:
            log_error(error_log, 'experiment_3', e)
            print(f"\n✗ Experiment 3 failed: {e}")
            print(f"  Error logged to: {error_log}")
            print(f"  Continuing to next experiment...")
    else:
        print(f"\n⊙ Experiment 3 already completed (skipping)")

    # ==================================================================
    # EXPERIMENT 4: BASELINE COMPARISON (independent)
    # ==================================================================

    if 'experiment_4' not in results:
        try:
            print(f"\n{'='*70}")
            print("EXPERIMENT 4: BASELINE COMPARISON")
            print(f"{'='*70}")

            from experiments.experiment_interpolation_compare import ExperimentInterpolationCompare
            exp_4 = ExperimentInterpolationCompare(config)
            results['experiment_4'] = exp_4.run()

            save_checkpoint(checkpoint_file, results)
            print(f"\n✓ Experiment 4 completed successfully")

        except Exception as e:
            log_error(error_log, 'experiment_4', e)
            print(f"\n✗ Experiment 4 failed: {e}")
            print(f"  Error logged to: {error_log}")
    else:
        print(f"\n⊙ Experiment 4 already completed (skipping)")

    # ==================================================================
    # GENERATE FINAL REPORT
    # ==================================================================

    print(f"\n{'='*70}")
    print("GENERATING FINAL REPORT")
    print(f"{'='*70}")

    generate_final_report(results, config, output_dir / 'p0_final_report.md')

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*70}")
    print("P0 EXPERIMENTS PIPELINE COMPLETE")
    print(f"{'='*70}")

    print(f"\nExecution Summary:")
    print(f"  Start time:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End time:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration:    {duration}")

    completed = [key for key in results.keys() if key.startswith('experiment_')]
    print(f"\nExperiments Completed: {len(completed)}/4")
    for exp in completed:
        print(f"  ✓ {exp}")

    print(f"\nResults Directory: {output_dir}")
    print(f"  - Checkpoint: {checkpoint_file}")
    print(f"  - Error log: {error_log}")
    print(f"  - Final report: {output_dir / 'p0_final_report.md'}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
