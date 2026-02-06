"""PyTorch dataset for transmission tensor data.

Loads T[probe, light, sh_coeffs] tensor and provides train/val/test splits
for dual-gaussian FBT training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict


class TransferTensorDataset(Dataset):
    """Dataset for transmission tensor.

    Data format:
        - tensor: [P, L, C] transmission tensor
        - probe_positions: [P, 3] probe world positions
        - light_positions: [L, 3] light world positions

    Split strategy:
        - Split probes into train/val/test sets
        - Use all light positions for each split
        - This tests generalization to new spatial locations
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_seed: int = 42,
        normalize: bool = True
    ):
        """Initialize dataset.

        Args:
            data_root: Root directory containing transfer_tensor.npz
            split: 'train', 'val', or 'test'
            train_ratio: Fraction of probes for training (default 0.7)
            val_ratio: Fraction of probes for validation (default 0.15)
            random_seed: Random seed for splits
            normalize: Whether to normalize probe/light positions to [-1, 1]
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split
        self.normalize = normalize

        # Load data
        data_path = self.data_root / 'transfer_tensor.npz'
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        data = np.load(data_path, allow_pickle=True)

        self.tensor = data['tensor']  # [P, L, C]
        self.probe_positions = data['probe_positions']  # [P, 3]
        self.light_positions = data['light_positions']  # [L, 3]
        self.metadata = data['metadata'].item() if 'metadata' in data else {}

        P, L, C = self.tensor.shape

        # Create probe splits
        np.random.seed(random_seed)
        probe_indices = np.random.permutation(P)

        n_train = int(P * train_ratio)
        n_val = int(P * val_ratio)

        if split == 'train':
            self.probe_indices = probe_indices[:n_train]
        elif split == 'val':
            self.probe_indices = probe_indices[n_train:n_train + n_val]
        elif split == 'test':
            self.probe_indices = probe_indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Extract subset
        self.tensor_subset = self.tensor[self.probe_indices]  # [P_split, L, C]
        self.probe_positions_subset = self.probe_positions[self.probe_indices]  # [P_split, 3]

        # Normalize positions if requested
        if normalize:
            self.probe_min = self.probe_positions.min(axis=0)
            self.probe_max = self.probe_positions.max(axis=0)
            self.probe_positions_subset = self._normalize_positions(
                self.probe_positions_subset, self.probe_min, self.probe_max
            )

            self.light_min = self.light_positions.min(axis=0)
            self.light_max = self.light_positions.max(axis=0)
            self.light_positions = self._normalize_positions(
                self.light_positions, self.light_min, self.light_max
            )

        # Flatten to (probe, light) pairs
        self.num_probes = len(self.probe_indices)
        self.num_lights = L

        print(f"TransferTensorDataset ({split}):")
        print(f"  Total probes: {P}, split probes: {self.num_probes}")
        print(f"  Light positions: {self.num_lights}")
        print(f"  Total samples: {len(self)} ({self.num_probes} probes × {self.num_lights} lights)")

    def _normalize_positions(self, positions: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
        """Normalize positions to [-1, 1]."""
        return 2.0 * (positions - min_vals) / (max_vals - min_vals + 1e-8) - 1.0

    def __len__(self) -> int:
        """Total number of (probe, light) pairs."""
        return self.num_probes * self.num_lights

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get single (probe, light, sh_coeffs) sample.

        Args:
            idx: Flattened index (probe_idx * num_lights + light_idx)

        Returns:
            sample: dict with keys:
                - probe_position: [3] probe position
                - light_position: [3] light position
                - sh_coeffs: [27] target SH coefficients
                - probe_idx: scalar probe index (for debugging)
                - light_idx: scalar light index (for debugging)
        """
        # Unflatten index
        probe_idx = idx // self.num_lights
        light_idx = idx % self.num_lights

        # Get data
        probe_pos = self.probe_positions_subset[probe_idx]  # [3]
        light_pos = self.light_positions[light_idx]  # [3]
        sh_coeffs = self.tensor_subset[probe_idx, light_idx, :]  # [27]

        sample = {
            'probe_position': torch.from_numpy(probe_pos).float(),
            'light_position': torch.from_numpy(light_pos).float(),
            'sh_coeffs': torch.from_numpy(sh_coeffs).float(),
            'probe_idx': torch.tensor(probe_idx, dtype=torch.long),
            'light_idx': torch.tensor(light_idx, dtype=torch.long)
        }

        return sample

    def get_all_probe_positions(self) -> np.ndarray:
        """Get all probe positions (for Gaussian initialization).

        Returns:
            positions: [P_total, 3] all probe positions (not just this split)
        """
        if self.normalize:
            return self._normalize_positions(self.probe_positions, self.probe_min, self.probe_max)
        return self.probe_positions

    def get_all_light_positions(self) -> np.ndarray:
        """Get all light positions (for Gaussian initialization).

        Returns:
            positions: [L, 3] all light positions
        """
        # Already normalized if self.normalize=True
        return self.light_positions


def create_dataloaders(
    data_root: str | Path,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    normalize: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders.

    Args:
        data_root: Root directory with transfer_tensor.npz
        batch_size: Batch size for training
        train_ratio: Fraction of probes for training
        val_ratio: Fraction of probes for validation
        num_workers: Number of dataloader workers
        normalize: Whether to normalize positions

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = TransferTensorDataset(
        data_root=data_root,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize=normalize
    )

    val_dataset = TransferTensorDataset(
        data_root=data_root,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize=normalize
    )

    test_dataset = TransferTensorDataset(
        data_root=data_root,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize=normalize
    )

    # Create dataloaders
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
    from pathlib import Path

    # Test on quick test data
    data_root = Path(__file__).parent.parent.parent / 'data_generation' / 'output' / 'test_quick'

    if not data_root.exists():
        print(f"Test data not found at: {data_root}")
        print("Run generate_transfer_tensor.py first")
        sys.exit(1)

    # Create datasets
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=str(data_root),
        batch_size=4,
        num_workers=0
    )

    print(f"\nDataloader statistics:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Test iteration
    print(f"\nTesting train loader iteration...")
    batch = next(iter(train_loader))

    print(f"  Batch keys: {batch.keys()}")
    print(f"  probe_position shape: {batch['probe_position'].shape}")
    print(f"  light_position shape: {batch['light_position'].shape}")
    print(f"  sh_coeffs shape: {batch['sh_coeffs'].shape}")
    print(f"  SH coeffs range: [{batch['sh_coeffs'].min():.4f}, {batch['sh_coeffs'].max():.4f}]")

    print("\nDataset test passed!")


class TransferTensorDataset5D(Dataset):
    """Dataset for 5D parametric light transfer tensor.

    Data format:
        - tensor: [P, M, C] transmission tensor
        - probe_positions: [P, 3] probe world positions
        - light_configs: [M, 5] light configurations [zenith, azimuth, intensity, temp, cloud]

    Split strategy:
        - Split light configs into train/val/test sets
        - Use all probe positions for each split
        - This tests generalization to new lighting conditions
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_seed: int = 42,
        normalize_probes: bool = True,
        normalize_params: bool = True
    ):
        """Initialize 5D parametric dataset.

        Args:
            data_root: Root directory containing parametric_tensor.npz
            split: 'train', 'val', or 'test'
            train_ratio: Fraction of light configs for training (default 0.7)
            val_ratio: Fraction of light configs for validation (default 0.15)
            random_seed: Random seed for splits
            normalize_probes: Whether to normalize probe positions to [-1, 1]
            normalize_params: Whether to normalize light parameters
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split
        self.normalize_probes = normalize_probes
        self.normalize_params = normalize_params

        # Load data
        data_path = self.data_root / 'parametric_tensor.npz'
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        data = np.load(data_path, allow_pickle=True)

        self.tensor = data['tensor']  # [P, M, C]
        self.probe_positions = data['probe_positions']  # [P, 3]
        self.light_configs = data['light_configs']  # [M, 5]

        # Load metadata
        metadata_path = self.data_root / 'metadata.json'
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        P, M, C = self.tensor.shape

        # Create light config splits (different from original: split configs not probes)
        np.random.seed(random_seed)
        config_indices = np.random.permutation(M)

        n_train = int(M * train_ratio)
        n_val = int(M * val_ratio)

        if split == 'train':
            self.config_indices = config_indices[:n_train]
        elif split == 'val':
            self.config_indices = config_indices[n_train:n_train + n_val]
        elif split == 'test':
            self.config_indices = config_indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Extract subset: [P, M_split, C]
        self.tensor_subset = self.tensor[:, self.config_indices, :]
        self.light_configs_subset = self.light_configs[self.config_indices]

        # Normalize probe positions if requested
        if normalize_probes:
            self.probe_min = self.probe_positions.min(axis=0)
            self.probe_max = self.probe_positions.max(axis=0)
            self.probe_positions_norm = self._normalize_positions(
                self.probe_positions, self.probe_min, self.probe_max
            )
        else:
            self.probe_positions_norm = self.probe_positions

        # Normalize light parameters if requested
        if normalize_params:
            # Define parameter ranges (from ParametricLightBuilder)
            # [zenith, azimuth, intensity, color_temp, cloud_cover]
            self.param_min = np.array([0.0, 0.0, 0.1, 2500.0, 0.0])
            self.param_max = np.array([90.0, 360.0, 2.0, 10000.0, 0.9])

            self.light_configs_subset_norm = self._normalize_params(
                self.light_configs_subset, self.param_min, self.param_max
            )
        else:
            self.light_configs_subset_norm = self.light_configs_subset

        # Flatten to (probe, config) pairs
        self.num_probes = P
        self.num_configs = len(self.config_indices)

        print(f"TransferTensorDataset5D ({split}):")
        print(f"  Probes: {self.num_probes}")
        print(f"  Total light configs: {M}, split configs: {self.num_configs}")
        print(f"  Total samples: {len(self)} ({self.num_probes} probes × {self.num_configs} configs)")

    def _normalize_positions(self, positions: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
        """Normalize positions to [-1, 1]."""
        return 2.0 * (positions - min_vals) / (max_vals - min_vals + 1e-8) - 1.0

    def _normalize_params(self, params: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
        """Normalize parameters to [-1, 1]."""
        return 2.0 * (params - min_vals) / (max_vals - min_vals + 1e-8) - 1.0

    def __len__(self) -> int:
        """Total number of (probe, config) pairs."""
        return self.num_probes * self.num_configs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get single (probe, light_params, sh_coeffs) sample.

        Args:
            idx: Flattened index (probe_idx * num_configs + config_idx)

        Returns:
            sample: dict with keys:
                - probe_position: [3] probe position
                - light_params: [5] light parameters [zenith, azimuth, intensity, temp, cloud]
                - sh_coeffs: [27] target SH coefficients
                - probe_idx: scalar probe index (for debugging)
                - config_idx: scalar config index (for debugging)
        """
        # Unflatten index
        probe_idx = idx // self.num_configs
        config_idx = idx % self.num_configs

        # Get data
        probe_pos = self.probe_positions_norm[probe_idx]  # [3]
        light_params = self.light_configs_subset_norm[config_idx]  # [5]
        sh_coeffs = self.tensor_subset[probe_idx, config_idx, :]  # [27]

        sample = {
            'probe_position': torch.from_numpy(probe_pos).float(),
            'light_params': torch.from_numpy(light_params).float(),
            'sh_coeffs': torch.from_numpy(sh_coeffs).float(),
            'probe_idx': torch.tensor(probe_idx, dtype=torch.long),
            'config_idx': torch.tensor(config_idx, dtype=torch.long)
        }

        return sample

    def get_all_probe_positions(self) -> np.ndarray:
        """Get all probe positions (for Gaussian initialization).

        Returns:
            positions: [P, 3] all probe positions
        """
        return self.probe_positions_norm

    def get_all_light_configs(self) -> np.ndarray:
        """Get all light configs in this split.

        Returns:
            configs: [M_split, 5] light configurations
        """
        return self.light_configs_subset_norm

    def get_unnormalized_light_configs(self) -> np.ndarray:
        """Get unnormalized light configs (for SVD initialization).

        Returns:
            configs: [M_split, 5] raw light configurations
        """
        return self.light_configs_subset


def create_dataloaders_5D(
    data_root: str | Path,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    normalize_probes: bool = True,
    normalize_params: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders for 5D parametric dataset.

    Args:
        data_root: Root directory with parametric_tensor.npz
        batch_size: Batch size for training
        train_ratio: Fraction of light configs for training
        val_ratio: Fraction of light configs for validation
        num_workers: Number of dataloader workers
        normalize_probes: Whether to normalize probe positions
        normalize_params: Whether to normalize light parameters

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = TransferTensorDataset5D(
        data_root=data_root,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize_probes=normalize_probes,
        normalize_params=normalize_params
    )

    val_dataset = TransferTensorDataset5D(
        data_root=data_root,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize_probes=normalize_probes,
        normalize_params=normalize_params
    )

    test_dataset = TransferTensorDataset5D(
        data_root=data_root,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize_probes=normalize_probes,
        normalize_params=normalize_params
    )

    # Create dataloaders
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
