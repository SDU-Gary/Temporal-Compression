"""Dataset loader for multi-temporal lighting compression.

Loads precomputed SH coefficients from data_generation output.
"""

from pathlib import Path
from typing import Literal, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiTimeLightingDataset(Dataset):
    """Dataset for multi-temporal lighting compression.

    Loads probe positions and SH coefficients across multiple time moments.
    Each sample contains a probe position and its SH coefficients for all moments.

    Args:
        data_root: Path to dataset directory (e.g., 'output/method_test_v1')
        split: Dataset split ('train', 'val', or 'test')
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
    """

    def __init__(
        self,
        data_root: Path | str,
        split: Literal['train', 'val', 'test'] = 'train',
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ):
        self.data_root = Path(data_root)
        self.split = split

        # Load probe positions
        probes_path = self.data_root / 'probes.npz'
        if not probes_path.exists():
            raise FileNotFoundError(f"Probes file not found: {probes_path}")

        probes_data = np.load(probes_path)
        all_positions = probes_data['positions']  # [N, 3]

        # Split dataset
        N = len(all_positions)
        train_end = int(N * train_ratio)
        val_end = train_end + int(N * val_ratio)

        if split == 'train':
            self.positions = all_positions[:train_end]
            self.indices = np.arange(train_end)
        elif split == 'val':
            self.positions = all_positions[train_end:val_end]
            self.indices = np.arange(train_end, val_end)
        else:  # test
            self.positions = all_positions[val_end:]
            self.indices = np.arange(val_end, N)

        # Find all moment directories
        moment_dirs = sorted(self.data_root.glob('moment_*'))
        if not moment_dirs:
            raise FileNotFoundError(f"No moment directories found in {self.data_root}")

        # Load SH coefficients and sun directions for all moments
        self.moments: List[str] = []
        self.sh_coeffs: Dict[str, np.ndarray] = {}
        self.sun_dirs: Dict[str, np.ndarray] = {}

        for moment_dir in moment_dirs:
            moment_id = moment_dir.name.split('_')[1]  # Extract '06', '09', etc.

            # Load SH coefficients
            sh_path = moment_dir / 'sh_coeffs.npz'
            if not sh_path.exists():
                continue

            sh_data = np.load(sh_path)
            coeffs = sh_data['coeffs']  # [N_probes, 27]
            sun_dir = sh_data['sun_dir']  # [3]

            # Store only the subset for this split
            self.sh_coeffs[moment_id] = coeffs[self.indices]
            self.sun_dirs[moment_id] = sun_dir
            self.moments.append(moment_id)

        self.num_moments = len(self.moments)

        # Normalize probe positions to [-1, 1] for better training
        self.position_mean = self.positions.mean(axis=0)
        self.position_std = self.positions.std(axis=0) + 1e-6
        self.positions_normalized = (self.positions - self.position_mean) / self.position_std

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Returns:
            dict with keys:
                - position: [3] probe position (normalized)
                - sh_gt: [T, 27] ground truth SH coefficients for all moments
                - sun_dirs: [T, 3] sun directions for all moments
                - moment_ids: [T] moment IDs (06, 09, 12, ...)
        """
        position = self.positions_normalized[idx]

        # Collect SH coefficients and sun directions for all moments
        sh_gt_list = []
        sun_dir_list = []
        moment_id_list = []

        for moment_id in self.moments:
            sh_gt_list.append(self.sh_coeffs[moment_id][idx])
            sun_dir_list.append(self.sun_dirs[moment_id])
            moment_id_list.append(int(moment_id))

        return {
            'position': torch.from_numpy(position).float(),
            'sh_gt': torch.from_numpy(np.stack(sh_gt_list)).float(),  # [T, 27]
            'sun_dirs': torch.from_numpy(np.stack(sun_dir_list)).float(),  # [T, 3]
            'moment_ids': torch.tensor(moment_id_list, dtype=torch.long),  # [T]
        }

    def get_normalization_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get position normalization parameters for inference."""
        return self.position_mean, self.position_std

    def denormalize_position(self, position: np.ndarray) -> np.ndarray:
        """Convert normalized position back to world coordinates."""
        return position * self.position_std + self.position_mean

    def __repr__(self) -> str:
        return (
            f"MultiTimeLightingDataset(\n"
            f"  split={self.split},\n"
            f"  num_samples={len(self)},\n"
            f"  num_moments={self.num_moments},\n"
            f"  moments={self.moments}\n"
            f")"
        )
