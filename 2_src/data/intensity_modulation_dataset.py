"""PyTorch dataset for 1D intensity modulation data.

Loads probe positions, time moments with intensities, and SH coefficients.
Designed for training GaussianPhysicsCompression1D model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import json


class IntensityModulationDataset(Dataset):
    """Dataset for 1D intensity modulation experiments.

    Data format:
        - probes.npz: probe positions [P, 3]
        - moment_XX/sh_coeffs.npz: SH coefficients [P, 27] + intensity metadata
        - metadata.json: dataset configuration

    Split strategy:
        - Split by TIME MOMENTS (not probes) for temporal generalization
        - Use all probes for each split
        - This tests ability to interpolate to unseen time moments
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_seed: int = 42,
        normalize_positions: bool = True
    ):
        """Initialize dataset.

        Args:
            data_root: Root directory (e.g., output/intensity_modulation_343/)
            split: 'train', 'val', or 'test'
            train_ratio: Fraction of moments for training (default 0.7)
            val_ratio: Fraction of moments for validation (default 0.15)
            random_seed: Random seed for splits
            normalize_positions: Whether to normalize probe positions to [-1, 1]
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split
        self.normalize_positions = normalize_positions

        # Load metadata
        metadata_path = self.data_root / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load probe positions
        probes_path = self.data_root / 'probes.npz'
        if not probes_path.exists():
            raise FileNotFoundError(f"Probes file not found: {probes_path}")

        probes_data = np.load(probes_path)
        self.probe_positions = probes_data['positions']  # [P, 3]
        P = len(self.probe_positions)

        # Load all moments
        num_moments = self.metadata['num_moments']
        moment_dirs = [self.data_root / f'moment_{i:02d}' for i in range(num_moments)]

        # Validate all moments exist
        for moment_dir in moment_dirs:
            if not moment_dir.exists():
                raise FileNotFoundError(f"Moment directory not found: {moment_dir}")

        # Load SH coefficients and intensities for all moments
        self.sh_coeffs_all = []  # List of [P, 27] arrays
        self.intensities_all = []  # List of scalar intensities
        self.times_all = []  # List of scalar times

        for moment_dir in moment_dirs:
            sh_path = moment_dir / 'sh_coeffs.npz'
            sh_data = np.load(sh_path)

            self.sh_coeffs_all.append(sh_data['sh_coeffs'])  # [P, 27]
            self.intensities_all.append(float(sh_data['intensity']))
            self.times_all.append(float(sh_data['time']))

        # Convert to arrays
        self.sh_coeffs_all = np.stack(self.sh_coeffs_all, axis=1)  # [P, M, 27]
        self.intensities_all = np.array(self.intensities_all)  # [M]
        self.times_all = np.array(self.times_all)  # [M]

        # Split by time moments
        np.random.seed(random_seed)
        moment_indices = np.random.permutation(num_moments)

        n_train = int(num_moments * train_ratio)
        n_val = int(num_moments * val_ratio)

        if split == 'train':
            self.moment_indices = moment_indices[:n_train]
        elif split == 'val':
            self.moment_indices = moment_indices[n_train:n_train + n_val]
        elif split == 'test':
            self.moment_indices = moment_indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Extract subset
        self.sh_coeffs_subset = self.sh_coeffs_all[:, self.moment_indices, :]  # [P, M_split, 27]
        self.intensities_subset = self.intensities_all[self.moment_indices]  # [M_split]
        self.times_subset = self.times_all[self.moment_indices]  # [M_split]

        # Normalize positions if requested
        if normalize_positions:
            self.probe_min = self.probe_positions.min(axis=0)
            self.probe_max = self.probe_positions.max(axis=0)
            self.probe_positions_normalized = self._normalize_positions(
                self.probe_positions, self.probe_min, self.probe_max
            )
        else:
            self.probe_positions_normalized = self.probe_positions

        # Flatten to (probe, moment) pairs
        self.num_probes = P
        self.num_moments = len(self.moment_indices)

        print(f"IntensityModulationDataset ({split}):")
        print(f"  Probes: {self.num_probes}")
        print(f"  Moments (split): {self.num_moments}/{num_moments}")
        print(f"  Intensity range: [{self.intensities_subset.min():.3f}, {self.intensities_subset.max():.3f}]")
        print(f"  Time range: [{self.times_subset.min():.3f}, {self.times_subset.max():.3f}]s")
        print(f"  Total samples: {len(self)} ({self.num_probes} × {self.num_moments})")

    def _normalize_positions(self, positions: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
        """Normalize positions to [-1, 1]."""
        return 2.0 * (positions - min_vals) / (max_vals - min_vals + 1e-8) - 1.0

    def __len__(self) -> int:
        """Total number of (probe, moment) pairs."""
        return self.num_probes * self.num_moments

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get single (probe, intensity, sh_coeffs) sample.

        Args:
            idx: Flattened index (probe_idx * num_moments + moment_idx)

        Returns:
            sample: dict with keys:
                - probe_position: [3] probe position
                - intensity: [1] scalar intensity value
                - sh_coeffs: [27] target SH coefficients
                - time: [1] scalar time value (for visualization)
                - probe_idx: scalar probe index (for debugging)
                - moment_idx: scalar moment index (for debugging)
        """
        # Unflatten index
        probe_idx = idx // self.num_moments
        moment_idx = idx % self.num_moments

        # Get data
        probe_pos = self.probe_positions_normalized[probe_idx]  # [3]
        intensity = self.intensities_subset[moment_idx]  # scalar
        time = self.times_subset[moment_idx]  # scalar
        sh_coeffs = self.sh_coeffs_subset[probe_idx, moment_idx, :]  # [27]

        sample = {
            'probe_position': torch.from_numpy(probe_pos).float(),
            'intensity': torch.tensor([intensity]).float(),  # [1]
            'sh_coeffs': torch.from_numpy(sh_coeffs).float(),
            'time': torch.tensor([time]).float(),  # [1]
            'probe_idx': torch.tensor(probe_idx, dtype=torch.long),
            'moment_idx': torch.tensor(moment_idx, dtype=torch.long)
        }

        return sample

    def get_full_probe_positions(self, normalized: bool = True) -> np.ndarray:
        """Get all probe positions.

        Args:
            normalized: Return normalized positions

        Returns:
            positions: [P, 3] probe positions
        """
        if normalized and self.normalize_positions:
            return self.probe_positions_normalized
        else:
            return self.probe_positions

    def get_full_intensities(self) -> np.ndarray:
        """Get all intensity values for this split.

        Returns:
            intensities: [M_split] intensity values
        """
        return self.intensities_subset

    def get_full_times(self) -> np.ndarray:
        """Get all time values for this split.

        Returns:
            times: [M_split] time values
        """
        return self.times_subset


def create_dataloaders(
    data_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders.

    Args:
        data_root: Root directory containing dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        train_ratio: Fraction of moments for training
        val_ratio: Fraction of moments for validation
        random_seed: Random seed for splits

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = IntensityModulationDataset(
        data_root=data_root,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed
    )

    val_dataset = IntensityModulationDataset(
        data_root=data_root,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed
    )

    test_dataset = IntensityModulationDataset(
        data_root=data_root,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':  # pragma: no cover
    """Test dataset loader."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python intensity_modulation_dataset.py <data_root>")
        sys.exit(1)

    data_root = sys.argv[1]

    print("Testing IntensityModulationDataset...\n")

    # Create datasets
    train_dataset = IntensityModulationDataset(data_root, split='train')
    val_dataset = IntensityModulationDataset(data_root, split='val')
    test_dataset = IntensityModulationDataset(data_root, split='test')

    print("\n" + "=" * 60)
    print("Testing data loading:")
    print("=" * 60)

    # Test single sample
    sample = train_dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"  probe_position: {sample['probe_position'].shape} {sample['probe_position']}")
    print(f"  intensity: {sample['intensity'].shape} {sample['intensity']}")
    print(f"  sh_coeffs: {sample['sh_coeffs'].shape}")
    print(f"  time: {sample['time'].shape} {sample['time']}")

    # Test batch loading
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=32,
        num_workers=0
    )

    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  probe_position: {batch['probe_position'].shape}")
    print(f"  intensity: {batch['intensity'].shape}")
    print(f"  sh_coeffs: {batch['sh_coeffs'].shape}")
    print(f"  time: {batch['time'].shape}")

    print("\n✅ Dataset test passed!")
