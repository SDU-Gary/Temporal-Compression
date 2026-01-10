"""
Analyze spatial smoothness of SH coefficients in the dataset
"""
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from data.dataset import MultiTimeLightingDataset
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def analyze_spatial_smoothness(dataset_path):
    """Analyze how SH coefficients vary with spatial distance"""

    print("="*70)
    print("SPATIAL SMOOTHNESS ANALYSIS")
    print("="*70)

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    dataset = MultiTimeLightingDataset(
        data_root=dataset_path,
        split='train',
        train_ratio=0.6,
        val_ratio=0.2
    )

    print(f"Total samples: {len(dataset)}")

    # Load all data
    positions = torch.from_numpy(dataset.positions_normalized).float().numpy()  # [N, 3]

    sh_list = []
    for i in range(len(dataset)):
        sample = dataset[i]
        sh_list.append(sample['sh_gt'].numpy())  # [T, 27]
    sh_coeffs = np.array(sh_list)  # [N, T, 27]

    print(f"Positions shape: {positions.shape}")
    print(f"SH coeffs shape: {sh_coeffs.shape}")

    # Build nearest neighbors
    print("\nBuilding nearest neighbors index...")
    nbrs = NearestNeighbors(n_neighbors=min(20, len(dataset)), algorithm='ball_tree')
    nbrs.fit(positions)

    # For each sample, find its neighbors and compute SH difference
    print("\nAnalyzing spatial vs SH difference...")

    distances_list = []
    sh_diff_list = []
    sh_relative_diff_list = []

    n_samples = min(len(dataset), 500)  # Sample for speed

    for i in range(n_samples):
        dists, indices = nbrs.kneighbors([positions[i]])
        dists = dists[0][1:]  # Exclude self
        indices = indices[0][1:]

        for j, (dist, idx) in enumerate(zip(dists, indices)):
            # Compute SH difference
            sh_diff = np.linalg.norm(sh_coeffs[i] - sh_coeffs[idx])
            sh_norm = (np.linalg.norm(sh_coeffs[i]) + np.linalg.norm(sh_coeffs[idx])) / 2
            sh_relative_diff = sh_diff / (sh_norm + 1e-8)

            distances_list.append(dist)
            sh_diff_list.append(sh_diff)
            sh_relative_diff_list.append(sh_relative_diff)

    distances = np.array(distances_list)
    sh_diffs = np.array(sh_diff_list)
    sh_relative_diffs = np.array(sh_relative_diff_list)

    # Statistics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\nSpatial Distance Statistics:")
    print(f"  Mean: {distances.mean():.6f}")
    print(f"  Median: {np.median(distances):.6f}")
    print(f"  Min: {distances.min():.6f}")
    print(f"  Max: {distances.max():.6f}")

    print(f"\nSH Coefficient Difference (L2 norm):")
    print(f"  Mean: {sh_diffs.mean():.6f}")
    print(f"  Median: {np.median(sh_diffs):.6f}")
    print(f"  Min: {sh_diffs.min():.6f}")
    print(f"  Max: {sh_diffs.max():.6f}")

    print(f"\nRelative SH Difference:")
    print(f"  Mean: {sh_relative_diffs.mean():.4f}")
    print(f"  Median: {np.median(sh_relative_diffs):.4f}")

    # Correlation analysis
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)

    # Group by distance bins
    bins = [0, 0.05, 0.1, 0.2, 0.4, 1.0]
    bin_labels = ['<0.05', '0.05-0.1', '0.1-0.2', '0.2-0.4', '0.4-1.0']

    print(f"\nSH Difference vs Spatial Distance:")
    print(f"{'Distance Bin':<15} {'Count':<10} {'Mean SH Diff':<15} {'Median SH Diff':<15}")
    print("-" * 60)

    for i in range(len(bins) - 1):
        mask = (distances >= bins[i]) & (distances < bins[i+1])
        if mask.sum() > 0:
            bin_sh_diffs = sh_diffs[mask]
            print(f"{bin_labels[i]:<15} {mask.sum():<10} {bin_sh_diffs.mean():<15.4f} {np.median(bin_sh_diffs):<15.4f}")

    # Correlation coefficient
    corr = np.corrcoef(distances, sh_diffs)[0, 1]
    print(f"\nPearson correlation (distance vs SH diff): {corr:.4f}")

    if corr < 0.3:
        print("  → WEAK correlation: Spatial distance poorly predicts SH similarity!")
        print("  → This explains why K-NN performs poorly (11 dB)")
    elif corr < 0.6:
        print("  → MODERATE correlation: Some spatial structure exists")
    else:
        print("  → STRONG correlation: Good spatial structure")

    # Check if nearby points have similar SH
    nearby_mask = distances < 0.1
    if nearby_mask.sum() > 0:
        nearby_sh_diffs = sh_diffs[nearby_mask]
        print(f"\nFor nearby points (distance < 0.1):")
        print(f"  Count: {nearby_mask.sum()}")
        print(f"  Mean SH diff: {nearby_sh_diffs.mean():.4f}")
        print(f"  Median SH diff: {np.median(nearby_sh_diffs):.4f}")

        if nearby_sh_diffs.mean() > 2.0:
            print("  → Even nearby points have LARGE SH differences!")
            print("  → Possible causes:")
            print("     1. High-frequency lighting (shadows, occlusion)")
            print("     2. Data generation issues (probe placement)")
            print("     3. SH fitting errors")

    # Estimate theoretical K-NN performance
    print("\n" + "="*70)
    print("K-NN PERFORMANCE ESTIMATION")
    print("="*70)

    # For k=5 neighbors
    k = 5
    estimated_mse_list = []

    for i in range(n_samples):
        dists, indices = nbrs.kneighbors([positions[i]])
        dists = dists[0][1:k+1]
        indices = indices[0][1:k+1]

        # Weighted average
        weights = 1.0 / (dists ** 2 + 1e-8)
        weights = weights / weights.sum()

        pred_sh = np.sum(sh_coeffs[indices] * weights[:, None, None], axis=0)
        mse = np.mean((pred_sh - sh_coeffs[i]) ** 2)
        estimated_mse_list.append(mse)

    estimated_mse = np.mean(estimated_mse_list)
    estimated_psnr = 10 * np.log10(1.0 / estimated_mse)

    print(f"\nEstimated K-NN (k={k}) performance:")
    print(f"  MSE: {estimated_mse:.6f}")
    print(f"  PSNR: {estimated_psnr:.2f} dB")

    # Compare with model
    model_psnr = 28.53
    gap = estimated_psnr - model_psnr

    print(f"\nModel performance: {model_psnr:.2f} dB")
    print(f"Gap (K-NN - Model): {gap:+.2f} dB")

    if gap < -15:
        print("\n🔴 CRITICAL: Model vastly outperforms spatial interpolation!")
        print("   → This indicates:")
        print("     1. Data has poor spatial smoothness")
        print("     2. Model learns structured latent representations")
        print("     3. Problem is NOT in model capacity")
        print("   → Conclusion: PSNR ~28-31dB may be near optimal for this data")

    # Save visualization
    print("\n" + "="*70)
    print("Generating visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax = axes[0]
    scatter = ax.scatter(distances, sh_diffs, alpha=0.1, s=1)
    ax.set_xlabel('Spatial Distance')
    ax.set_ylabel('SH Coefficient Difference (L2 norm)')
    ax.set_title(f'Spatial Distance vs SH Difference\n(Correlation: {corr:.3f})')
    ax.grid(True, alpha=0.3)

    # Histogram of relative differences
    ax = axes[1]
    ax.hist(sh_relative_diffs, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Relative SH Difference')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Relative SH Differences')
    ax.axvline(sh_relative_diffs.mean(), color='r', linestyle='--', label=f'Mean: {sh_relative_diffs.mean():.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path("experiments/data_smoothness_analysis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to {output_path}")

    plt.close()

    print("\n" + "="*70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                       default='../data_generation/output/dataset_2k',
                       help='Path to dataset')
    args = parser.parse_args()

    dataset_path = Path(__file__).parent.parent / args.dataset
    analyze_spatial_smoothness(dataset_path)
