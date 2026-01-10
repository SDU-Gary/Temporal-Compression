"""
TPE Validation Dataset Loader

Loads datasets generated for TPE (Temporal Perturbation Embedding) validation.
Supports both Level 1 (toy scenes) and Level 2 (single probe critical validation).

Usage:
    from utils.tpe_dataset_loader import TPEValidationDataset

    # Load Level 2 dataset
    dataset = TPEValidationDataset(
        dataset_path='../data_generation/output/level2_tpe/cornell-box',
        level=2
    )

    # Get single probe data for validation
    data = dataset.get_probe_data(0)
    # Returns: {'position': [3], 'sh_gt': [T, 27], 'sun_dirs': [T, 3], ...}
"""

import numpy as np
from pathlib import Path
import json
from typing import Dict, Any


class TPEValidationDataset:
    """
    Dataset loader for TPE validation experiments.

    Loads pre-generated datasets containing:
    - Probe positions
    - Spherical harmonics coefficients across time
    - Sun directions for each time step

    Supports Level 1 (multiple probes) and Level 2 (single probe) datasets.
    """

    def __init__(self, dataset_path: Path, level: int):
        """
        Initialize TPE validation dataset loader.

        Args:
            dataset_path: Path to dataset directory
            level: Dataset level (1 or 2)

        Raises:
            FileNotFoundError: If dataset files are missing
            ValueError: If level is not 1 or 2
        """
        if level not in [1, 2]:
            raise ValueError(f"Level must be 1 or 2, got {level}")

        self.path = Path(dataset_path)
        self.level = level

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.path}")

        # Load metadata
        metadata_file = 'config.json' if level == 1 else 'metadata.json'
        metadata_path = self.path / metadata_file

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # Load probe positions
        probe_file = self.path / 'probes.npz'
        if not probe_file.exists():
            raise FileNotFoundError(f"Probe file not found: {probe_file}")

        probe_data = np.load(probe_file)
        self.probe_positions = probe_data['positions']

        # Get time steps from metadata
        self.hours = self.metadata['time_steps']
        self.T = len(self.hours)
        self.N = len(self.probe_positions)

        # Load all time step data
        self._load_temporal_data()

        print(f"✓ Loaded {self.metadata.get('scene_name', 'unknown')} Level {level} dataset")
        print(f"  Probes: {self.N}")
        print(f"  Time steps: {self.T}")
        print(f"  Hours: {self.hours}")

    def _load_temporal_data(self):
        """Load SH coefficients and sun directions for all time steps."""
        sh_list = []
        sun_list = []

        for hour in self.hours:
            moment_dir = self.path / f"moment_{hour:02d}"

            if not moment_dir.exists():
                raise FileNotFoundError(f"Moment directory not found: {moment_dir}")

            sh_file = moment_dir / "sh_coeffs.npz"
            if not sh_file.exists():
                raise FileNotFoundError(f"SH coefficients file not found: {sh_file}")

            data = np.load(sh_file)
            sh_list.append(data['coeffs'])
            sun_list.append(data['sun_dir'])

        self.sh_coeffs = np.stack(sh_list)  # [T, N, 27]
        self.sun_dirs = np.stack(sun_list)   # [T, 3]

        # Validate shapes
        assert self.sh_coeffs.shape == (self.T, self.N, 27), \
            f"Unexpected SH shape: {self.sh_coeffs.shape}, expected ({self.T}, {self.N}, 27)"
        assert self.sun_dirs.shape == (self.T, 3), \
            f"Unexpected sun_dirs shape: {self.sun_dirs.shape}, expected ({self.T}, 3)"

    def get_probe_data(self, probe_idx: int = 0) -> Dict[str, Any]:
        """
        Get complete temporal data for a single probe.

        This is the primary interface for TPE validation, providing all
        necessary data to test the perturbation hypothesis.

        Args:
            probe_idx: Probe index (0 to N-1)

        Returns:
            Dictionary containing:
                - 'position': [3] probe 3D coordinates
                - 'sh_gt': [T, 27] ground truth SH coefficients across time
                - 'sun_dirs': [T, 3] sun directions for each time step
                - 'hours': [T] hour values for each time step
                - 'metadata': Full dataset metadata dictionary

        Example:
            >>> data = dataset.get_probe_data(0)
            >>> sh_gt = data['sh_gt']  # [12, 27]
            >>> sun_dirs = data['sun_dirs']  # [12, 3]
        """
        if probe_idx < 0 or probe_idx >= self.N:
            raise IndexError(f"Probe index {probe_idx} out of range [0, {self.N-1}]")

        return {
            'position': self.probe_positions[probe_idx],      # [3]
            'sh_gt': self.sh_coeffs[:, probe_idx, :],         # [T, 27]
            'sun_dirs': self.sun_dirs,                        # [T, 3]
            'hours': np.array(self.hours),                    # [T]
            'metadata': self.metadata
        }

    def get_all_probes_data(self) -> Dict[str, Any]:
        """
        Get data for all probes simultaneously.

        Useful for Level 1 datasets with multiple probes.

        Returns:
            Dictionary containing:
                - 'positions': [N, 3] all probe positions
                - 'sh_gt': [T, N, 27] SH coefficients for all probes
                - 'sun_dirs': [T, 3] sun directions
                - 'hours': [T] hour values
                - 'metadata': Dataset metadata
        """
        return {
            'positions': self.probe_positions,    # [N, 3]
            'sh_gt': self.sh_coeffs,              # [T, N, 27]
            'sun_dirs': self.sun_dirs,            # [T, 3]
            'hours': np.array(self.hours),        # [T]
            'metadata': self.metadata
        }

    def __len__(self) -> int:
        """Return number of probes."""
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get data for probe at index.

        Args:
            idx: Probe index

        Returns:
            Probe data dictionary (same as get_probe_data)
        """
        return self.get_probe_data(idx)

    def __repr__(self) -> str:
        return (
            f"TPEValidationDataset(level={self.level}, "
            f"scene='{self.metadata.get('scene_name', 'unknown')}', "
            f"probes={self.N}, time_steps={self.T})"
        )


def load_level2_for_validation(dataset_path: Path) -> Dict[str, Any]:
    """
    Convenience function to load Level 2 dataset for validation.

    Specifically designed for Level 2 datasets which have a single probe.

    Args:
        dataset_path: Path to Level 2 dataset

    Returns:
        Single probe data dictionary ready for validation

    Example:
        >>> data = load_level2_for_validation('output/level2_tpe/cornell-box')
        >>> sh_gt = data['sh_gt']  # [12, 27]
    """
    dataset = TPEValidationDataset(dataset_path, level=2)

    if len(dataset) != 1:
        print(f"WARNING: Level 2 dataset should have 1 probe, found {len(dataset)}")

    return dataset.get_probe_data(0)


def test_loader():
    """Test function to verify dataset loading."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tpe_dataset_loader.py <dataset_path>")
        print("Example: python tpe_dataset_loader.py ../data_generation/output/level2_tpe/cornell-box")
        sys.exit(1)

    dataset_path = Path(sys.argv[1])

    # Detect level from metadata
    if (dataset_path / 'config.json').exists():
        level = 1
    elif (dataset_path / 'metadata.json').exists():
        level = 2
    else:
        print("ERROR: Could not detect dataset level (missing config/metadata file)")
        sys.exit(1)

    print(f"\nLoading dataset from: {dataset_path}")
    print(f"Detected level: {level}")

    dataset = TPEValidationDataset(dataset_path, level=level)

    print(f"\n{dataset}")
    print(f"\nDataset info:")
    print(f"  Scene: {dataset.metadata.get('scene_name', 'unknown')}")
    print(f"  Purpose: {dataset.metadata.get('purpose', 'N/A')}")
    print(f"  Probes: {len(dataset)}")
    print(f"  Time steps: {dataset.T}")

    # Test getting probe data
    probe_data = dataset.get_probe_data(0)

    print(f"\nProbe 0 data shapes:")
    print(f"  position: {probe_data['position'].shape}")
    print(f"  sh_gt: {probe_data['sh_gt'].shape}")
    print(f"  sun_dirs: {probe_data['sun_dirs'].shape}")
    print(f"  hours: {probe_data['hours'].shape}")

    print(f"\nProbe 0 position: {probe_data['position']}")
    print(f"Hour range: {probe_data['hours'][0]} - {probe_data['hours'][-1]}")

    print("\n✓ Dataset loading test passed!")


if __name__ == "__main__":
    test_loader()
