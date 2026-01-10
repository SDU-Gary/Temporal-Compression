"""
K-NN Baseline to estimate data quality upper bound
Uses nearest neighbor interpolation to predict SH coefficients
"""
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from data.dataset import MultiTimeLightingDataset
from training.metrics import psnr
from tqdm import tqdm

def knn_interpolation(query_pos, train_positions, train_sh_coeffs, k=5, p=2):
    """
    K-NN interpolation for SH coefficients

    Args:
        query_pos: [3] query position
        train_positions: [N, 3] training positions
        train_sh_coeffs: [N, T, 27] training SH coefficients
        k: number of nearest neighbors
        p: power for distance weighting (p=2 for inverse square)

    Returns:
        pred_sh: [T, 27] predicted SH coefficients
    """
    # Compute distances
    distances = torch.norm(train_positions - query_pos.unsqueeze(0), dim=1)  # [N]

    # Find k nearest neighbors
    k_actual = min(k, len(distances))
    knn_distances, knn_indices = torch.topk(distances, k_actual, largest=False)

    # Compute weights (inverse distance weighting)
    # Add small epsilon to avoid division by zero
    weights = 1.0 / (knn_distances ** p + 1e-8)  # [k]
    weights = weights / weights.sum()  # Normalize

    # Weighted average of SH coefficients
    knn_sh_coeffs = train_sh_coeffs[knn_indices]  # [k, T, 27]
    pred_sh = torch.sum(knn_sh_coeffs * weights.view(-1, 1, 1), dim=0)  # [T, 27]

    return pred_sh

def evaluate_knn_baseline(dataset_path, k=5, p=2):
    """Evaluate K-NN baseline on validation set"""

    print("="*70)
    print("K-NN BASELINE EVALUATION")
    print("="*70)

    # Load datasets
    print(f"\nLoading datasets from {dataset_path}...")
    train_dataset = MultiTimeLightingDataset(
        data_root=dataset_path,
        split='train',
        train_ratio=0.6,
        val_ratio=0.2
    )

    val_dataset = MultiTimeLightingDataset(
        data_root=dataset_path,
        split='val',
        train_ratio=0.6,
        val_ratio=0.2
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Get all training data (use normalized positions)
    print("\nLoading training data into memory...")
    train_positions = torch.from_numpy(train_dataset.positions_normalized).float()  # [N_train, 3], normalized

    # Load all SH coefficients
    train_sh_list = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        train_sh_list.append(sample['sh_gt'].float())  # [T, 27], already a tensor
    train_sh_coeffs = torch.stack(train_sh_list, dim=0)  # [N_train, T, 27]

    print(f"Train positions shape: {train_positions.shape}")
    print(f"Train SH coeffs shape: {train_sh_coeffs.shape}")

    # Evaluate on validation set
    print(f"\nEvaluating K-NN (k={k}, p={p}) on validation set...")

    all_mse = []
    all_psnr = []
    all_distances_to_nearest = []

    for i in tqdm(range(len(val_dataset)), desc="K-NN prediction"):
        sample = val_dataset[i]

        query_pos = sample['position'].float()  # [3], already a tensor
        gt_sh_coeffs = sample['sh_gt'].float()  # [T, 27], already a tensor

        # K-NN prediction
        pred_sh_coeffs = knn_interpolation(
            query_pos, train_positions, train_sh_coeffs, k=k, p=p
        )

        # Compute metrics
        mse = torch.mean((pred_sh_coeffs - gt_sh_coeffs) ** 2).item()
        sample_psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100.0

        all_mse.append(mse)
        all_psnr.append(sample_psnr)

        # Record distance to nearest neighbor
        distances = torch.norm(train_positions - query_pos.unsqueeze(0), dim=1)
        nearest_dist = distances.min().item()
        all_distances_to_nearest.append(nearest_dist)

    # Statistics
    all_mse = np.array(all_mse)
    all_psnr = np.array(all_psnr)
    all_distances = np.array(all_distances_to_nearest)

    print("\n" + "="*70)
    print("K-NN BASELINE RESULTS")
    print("="*70)

    print(f"\nConfiguration:")
    print(f"  k (neighbors): {k}")
    print(f"  p (distance power): {p}")

    print(f"\nOverall Performance:")
    print(f"  Mean PSNR: {all_psnr.mean():.2f} dB")
    print(f"  Median PSNR: {np.median(all_psnr):.2f} dB")
    print(f"  Std PSNR: {all_psnr.std():.2f} dB")
    print(f"  Min PSNR: {all_psnr.min():.2f} dB")
    print(f"  Max PSNR: {all_psnr.max():.2f} dB")

    print(f"\nMSE Statistics:")
    print(f"  Mean MSE: {all_mse.mean():.6f}")
    print(f"  Median MSE: {np.median(all_mse):.6f}")

    print(f"\nDistance to Nearest Neighbor:")
    print(f"  Mean distance: {all_distances.mean():.6f}")
    print(f"  Median distance: {np.median(all_distances):.6f}")
    print(f"  Max distance: {all_distances.max():.6f}")

    # Percentile analysis
    print(f"\nPSNR Percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p}th percentile: {np.percentile(all_psnr, p):.2f} dB")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    mean_psnr = all_psnr.mean()

    if mean_psnr >= 38:
        print("\n🟢 K-NN PSNR ≥ 38dB: DATA QUALITY IS GOOD")
        print("   → Problem is in the MODEL")
        print("   → Recommendations:")
        print("     1. Increase MLP capacity (hidden_dim 32/64 → 128/256)")
        print("     2. Increase latent dimensions (base 6→12, time 9→18)")
        print("     3. Check for architecture issues (shortcut learning)")
    elif mean_psnr >= 35:
        print("\n🟡 K-NN PSNR 35-38dB: DATA QUALITY IS ACCEPTABLE")
        print("   → Some room for model improvement")
        print("   → Current model (31dB) has ~4-7dB gap to data upper bound")
        print("   → Try moderate model capacity increase")
    elif mean_psnr >= 32:
        print("\n🟠 K-NN PSNR 32-35dB: DATA QUALITY IS LIMITING")
        print("   → Model is approaching data quality upper bound")
        print("   → Current model (31dB) is only 1-4dB below data limit")
        print("   → Recommendations:")
        print("     1. Increase rendering quality (spp 128 → 512)")
        print("     2. Improve SH fitting (num_sh_samples 16 → 64)")
        print("     3. Check for data generation issues")
    else:
        print("\n🔴 K-NN PSNR < 32dB: DATA QUALITY IS POOR")
        print("   → Data quality is the PRIMARY bottleneck")
        print("   → Model performance (31dB) is AT or ABOVE data quality!")
        print("   → MUST improve data generation:")
        print("     1. Significantly increase spp (128 → 1024)")
        print("     2. Increase SH fitting samples (16 → 128)")
        print("     3. Check for bugs in data generation pipeline")

    # Model comparison
    print("\n" + "="*70)
    print("MODEL VS K-NN COMPARISON")
    print("="*70)

    model_train_psnr = 31.37  # V2
    model_val_psnr = 28.53    # V2
    knn_psnr = mean_psnr

    print(f"\nCurrent Model (V2):")
    print(f"  Training PSNR: {model_train_psnr:.2f} dB")
    print(f"  Validation PSNR: {model_val_psnr:.2f} dB")

    print(f"\nK-NN Baseline:")
    print(f"  Validation PSNR: {knn_psnr:.2f} dB")

    gap = knn_psnr - model_val_psnr
    print(f"\nGap (K-NN - Model): {gap:+.2f} dB")

    if gap > 5:
        print("  → LARGE GAP: Model has significant room for improvement")
    elif gap > 2:
        print("  → MODERATE GAP: Model can be improved")
    elif gap > 0:
        print("  → SMALL GAP: Model is approaching data quality limit")
    else:
        print("  → NEGATIVE GAP: Model is at or above simple interpolation!")
        print("     (This suggests the model is learning good representations)")

    print("\n" + "="*70)

    return {
        'mean_psnr': mean_psnr,
        'median_psnr': np.median(all_psnr),
        'std_psnr': all_psnr.std(),
        'all_psnr': all_psnr,
        'all_mse': all_mse,
        'all_distances': all_distances
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                       default='../data_generation/output/dataset_2k',
                       help='Path to dataset')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of nearest neighbors')
    parser.add_argument('--p', type=float, default=2.0,
                       help='Power for distance weighting')
    args = parser.parse_args()

    dataset_path = Path(__file__).parent.parent / args.dataset

    results = evaluate_knn_baseline(dataset_path, k=args.k, p=args.p)

    # Save results
    output_file = Path("experiments/knn_baseline_results.npz")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file,
             mean_psnr=results['mean_psnr'],
             median_psnr=results['median_psnr'],
             std_psnr=results['std_psnr'],
             all_psnr=results['all_psnr'],
             all_mse=results['all_mse'],
             all_distances=results['all_distances'])
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
