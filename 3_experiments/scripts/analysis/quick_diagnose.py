"""
Quick diagnostic analysis from training logs
"""
import re
import numpy as np
from pathlib import Path

def parse_training_log(log_file):
    """Parse training log and extract loss components"""

    train_steps = []
    train_mse = []
    train_psnr = []
    train_total = []
    train_recon = []
    train_temporal = []

    val_steps = []
    val_mse = []
    val_psnr = []

    with open(log_file, 'r') as f:
        for line in f:
            # Parse training lines
            if 'Train Step' in line and 'total:' in line:
                match = re.search(r'Step\s+(\d+)', line)
                if match:
                    step = int(match.group(1))

                    # Extract metrics
                    mse_match = re.search(r'mse:\s+([\d.]+)', line)
                    psnr_match = re.search(r'psnr:\s+([\d.]+)', line)
                    total_match = re.search(r'total:\s+([\d.]+)', line)
                    recon_match = re.search(r'reconstruction:\s+([\d.]+)', line)
                    temporal_match = re.search(r'temporal_smooth:\s+([\d.]+)', line)

                    if all([mse_match, psnr_match, total_match, recon_match, temporal_match]):
                        train_steps.append(step)
                        train_mse.append(float(mse_match.group(1)))
                        train_psnr.append(float(psnr_match.group(1)))
                        train_total.append(float(total_match.group(1)))
                        train_recon.append(float(recon_match.group(1)))
                        train_temporal.append(float(temporal_match.group(1)))

            # Parse validation lines
            elif 'Val Step' in line and 'mse:' in line:
                match = re.search(r'Step\s+(\d+)', line)
                if match:
                    step = int(match.group(1))

                    mse_match = re.search(r'mse:\s+([\d.]+)', line)
                    psnr_match = re.search(r'psnr:\s+([\d.]+)', line)

                    if mse_match and psnr_match:
                        val_steps.append(step)
                        val_mse.append(float(mse_match.group(1)))
                        val_psnr.append(float(psnr_match.group(1)))

    return {
        'train': {
            'steps': np.array(train_steps),
            'mse': np.array(train_mse),
            'psnr': np.array(train_psnr),
            'total_loss': np.array(train_total),
            'recon_loss': np.array(train_recon),
            'temporal_loss': np.array(train_temporal)
        },
        'val': {
            'steps': np.array(val_steps),
            'mse': np.array(val_mse),
            'psnr': np.array(val_psnr)
        }
    }

def main():
    log_file = Path("experiments/baseline_2k/logs/training_20251208_155115.log")

    print("="*70)
    print("QUICK DIAGNOSTIC ANALYSIS FROM TRAINING LOG")
    print("="*70)

    # Parse log
    print("\nParsing training log...")
    data = parse_training_log(log_file)

    # Get temporal smooth weight from config
    import yaml
    with open("experiments/baseline_2k/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    temporal_weight = config['training']['loss']['temporal_smooth']

    # Analyze final stats (last 100 training steps)
    print("\n" + "="*70)
    print("FINAL TRAINING STATISTICS (last 100 steps)")
    print("="*70)

    train_data = data['train']
    last_100_idx = -100

    final_recon = np.mean(train_data['recon_loss'][last_100_idx:])
    final_temporal = np.mean(train_data['temporal_loss'][last_100_idx:])
    final_total = np.mean(train_data['total_loss'][last_100_idx:])
    final_psnr = np.mean(train_data['psnr'][last_100_idx:])

    print(f"\nReconstruction Loss: {final_recon:.6f}")
    print(f"Temporal Smooth Loss (raw): {final_temporal:.6f}")
    print(f"Total Loss: {final_total:.6f}")
    print(f"Training PSNR: {final_psnr:.2f} dB")

    # Calculate temporal contribution
    temporal_contribution = temporal_weight * final_temporal
    temporal_percentage = (temporal_contribution / final_total) * 100

    print(f"\n--- Temporal Regularization Analysis ---")
    print(f"Temporal weight: {temporal_weight}")
    print(f"Temporal contribution to total loss: {temporal_contribution:.6f}")
    print(f"Temporal percentage: {temporal_percentage:.2f}%")

    if temporal_percentage > 20:
        print(f"\n🔴 CRITICAL: Temporal regularization is TOO STRONG!")
        print(f"   Recommended: <15%, Current: {temporal_percentage:.2f}%")
        rec_weight = temporal_weight * (12.0 / temporal_percentage)
        print(f"   Suggestion: Reduce temporal_smooth from {temporal_weight} to {rec_weight:.4f}")
    elif temporal_percentage > 15:
        print(f"\n🟡 WARNING: Temporal regularization is strong")
        rec_weight = temporal_weight * (12.0 / temporal_percentage)
        print(f"   Recommended: <15%, Current: {temporal_percentage:.2f}%")
        print(f"   Suggestion: Reduce temporal_smooth from {temporal_weight} to {rec_weight:.4f}")
    else:
        print(f"\n🟢 Temporal regularization is reasonable")

    # Analyze validation behavior
    print("\n" + "="*70)
    print("VALIDATION BEHAVIOR ANALYSIS")
    print("="*70)

    val_data = data['val']
    best_val_idx = np.argmax(val_data['psnr'])
    best_val_step = val_data['steps'][best_val_idx]
    best_val_psnr = val_data['psnr'][best_val_idx]
    final_val_psnr = val_data['psnr'][-1]

    print(f"\nBest validation PSNR: {best_val_psnr:.2f} dB at step {best_val_step}")
    print(f"Final validation PSNR: {final_val_psnr:.2f} dB at step {val_data['steps'][-1]}")
    print(f"Best validation step: {best_val_step} / {train_data['steps'][-1]} total steps")
    print(f"Percentage: {best_val_step / train_data['steps'][-1] * 100:.1f}%")

    # Check if validation was still improving
    last_1000_val = val_data['psnr'][val_data['steps'] > (train_data['steps'][-1] - 1000)]
    if len(last_1000_val) > 1:
        val_trend = last_1000_val[-1] - last_1000_val[0]
        print(f"\nValidation trend in last 1000 steps: {val_trend:+.3f} dB")
        if abs(val_trend) < 0.1:
            print("   → Validation has plateaued")
        elif val_trend > 0:
            print("   → Validation still improving slightly")
        else:
            print("   → Validation degrading (possible overfit)")

    # Training trend
    print("\n" + "="*70)
    print("TRAINING CONVERGENCE ANALYSIS")
    print("="*70)

    # Compare early vs late training
    early_psnr = np.mean(train_data['psnr'][:100])
    mid_psnr = np.mean(train_data['psnr'][len(train_data['psnr'])//2:len(train_data['psnr'])//2+100])
    late_psnr = np.mean(train_data['psnr'][-100:])

    print(f"\nTraining PSNR progression:")
    print(f"  Early (step 0-1000):     {early_psnr:.2f} dB")
    print(f"  Mid (around step 4000):  {mid_psnr:.2f} dB")
    print(f"  Late (last 100 steps):   {late_psnr:.2f} dB")
    print(f"  Total improvement:       {late_psnr - early_psnr:.2f} dB")

    # Check if training was still improving
    last_1000_train = train_data['psnr'][-1000:]
    train_slope = (last_1000_train[-100:].mean() - last_1000_train[:100].mean())
    print(f"\nTraining trend in last 1000 steps: {train_slope:+.3f} dB")
    if abs(train_slope) < 0.1:
        print("   → Training has converged")
    else:
        print("   → Training still improving (early stop may be premature)")

    # Final recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if temporal_percentage > 20:
        print("\n🔴 PRIMARY ISSUE: Temporal regularization is too strong")
        print(f"\n   1. Create new config with temporal_smooth={temporal_weight * 0.5:.4f}")
        print("   2. Also try temporal_smooth=0.01 (baseline value)")
        print("   3. Expected improvement: +3-5 dB in both train and val PSNR")

    if best_val_step < train_data['steps'][-1] * 0.4:
        print("\n🟡 SECONDARY ISSUE: Best validation occurred very early")
        print(f"   Consider: Increase patience or reduce min_delta")

    if abs(train_slope) > 0.1:
        print("\n🟡 Training was still improving when stopped")
        print("   Consider: Train longer or adjust early stopping")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
