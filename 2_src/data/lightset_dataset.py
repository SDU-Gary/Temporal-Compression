"""Dataset for multi-light (light-set) SH compression.

Expected data format (npz):
  - tensor: [P, M, 27]
  - probe_positions: [P, 3]
  - light_configs: [M, N, 12]  (LightDescriptor)
  - light_mask: [M, N] (optional, defaults to ones)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LightSetDataset(Dataset):
    """Dataset for light-set inputs with mask."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_seed: int = 42,
        normalize_probes: bool = True,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split
        self.normalize_probes = normalize_probes

        data_path = self.data_root / "parametric_tensor.npz"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # TODO(perf): train/val/test 会各自实例化并重复 np.load 同一大文件。
        # 可考虑共享内存映射（mmap_mode）或上层复用已加载数组，降低启动与内存压力。
        data = np.load(data_path, allow_pickle=True)
        self.tensor = data["tensor"]  # [P, M, 27]
        self.probe_positions = data["probe_positions"]  # [P, 3]
        self.light_configs = data["light_configs"]  # [M, N, 12]
        if "light_mask" in data:
            self.light_mask = data["light_mask"]
        else:
            self.light_mask = np.ones(self.light_configs.shape[:2], dtype=np.float32)
        if "valid_mask" in data:
            self.valid_mask = np.array(data["valid_mask"], dtype=np.float32).reshape(-1)
        else:
            self.valid_mask = np.ones((self.tensor.shape[0],), dtype=np.float32)

        # Filter invalid probes
        valid_indices = np.where(self.valid_mask > 0.5)[0]
        if valid_indices.size < self.tensor.shape[0]:
            self.tensor = self.tensor[valid_indices]
            self.probe_positions = self.probe_positions[valid_indices]
            self.valid_mask = self.valid_mask[valid_indices]

        P, M, _ = self.tensor.shape

        # Split by config index
        rng = np.random.RandomState(random_seed)
        config_indices = rng.permutation(M)

        n_train = int(M * train_ratio)
        n_val = int(M * val_ratio)

        if split == "train":
            self.config_indices = config_indices[:n_train]
        elif split == "val":
            self.config_indices = config_indices[n_train : n_train + n_val]
        elif split == "test":
            self.config_indices = config_indices[n_train + n_val :]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Subset by configs
        self.tensor_subset = self.tensor[:, self.config_indices, :]
        self.light_configs_subset = self.light_configs[self.config_indices]
        self.light_mask_subset = self.light_mask[self.config_indices]

        # Normalize probe positions
        if normalize_probes:
            self.probe_min = self.probe_positions.min(axis=0)
            self.probe_max = self.probe_positions.max(axis=0)
            self.probe_positions_norm = self._normalize_positions(
                self.probe_positions, self.probe_min, self.probe_max
            )
        else:
            self.probe_positions_norm = self.probe_positions

        self.num_probes = P
        self.num_configs = len(self.config_indices)

        print(f"LightSetDataset ({split}):")
        print(f"  Probes: {self.num_probes}")
        print(f"  Configs (split): {self.num_configs}/{M}")
        print(f"  Light slots: {self.light_configs.shape[1]}")
        print(f"  Total samples: {len(self)} ({self.num_probes} × {self.num_configs})")

    def _normalize_positions(self, positions: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
        return 2.0 * (positions - min_vals) / (max_vals - min_vals + 1e-8) - 1.0

    def __len__(self) -> int:
        return self.num_probes * self.num_configs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        probe_idx = idx // self.num_configs
        config_idx = idx % self.num_configs

        probe_pos = self.probe_positions_norm[probe_idx]
        light_params = self.light_configs_subset[config_idx]
        light_mask = self.light_mask_subset[config_idx]
        sh_coeffs = self.tensor_subset[probe_idx, config_idx, :]

        # TODO(perf): 每个样本都执行 from_numpy(...).float()，会产生大量小对象转换开销。
        # 可考虑在初始化阶段预构建 torch tensor（或使用自定义 collate 批量转换）。
        sample = {
            "probe_position": torch.from_numpy(probe_pos).float(),
            "light_params": torch.from_numpy(light_params).float(),
            "light_mask": torch.from_numpy(light_mask).float(),
            "sh_coeffs": torch.from_numpy(sh_coeffs).float(),
            "probe_idx": torch.tensor(probe_idx, dtype=torch.long),
            "config_idx": torch.tensor(config_idx, dtype=torch.long),
        }
        return sample

    def get_full_probe_positions(self, normalized: bool = True) -> np.ndarray:
        if normalized and self.normalize_probes:
            return self.probe_positions_norm
        return self.probe_positions


def create_dataloaders_lightset(
    data_root: str | Path,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    normalize_probes: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = LightSetDataset(
        data_root=data_root,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize_probes=normalize_probes,
    )

    val_dataset = LightSetDataset(
        data_root=data_root,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize_probes=normalize_probes,
    )

    test_dataset = LightSetDataset(
        data_root=data_root,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize_probes=normalize_probes,
    )

    # TODO(perf): 可按平台/任务调优 persistent_workers、prefetch_factor、pin_memory_device。
    # 当前仅 pin_memory=True，仍有进一步提升输入管线吞吐空间。
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
