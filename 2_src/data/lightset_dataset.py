"""Dataset for multi-light (light-set) SH compression.

Expected data format (npz):
  - tensor: [P, M, 27]
  - probe_positions: [P, 3]
  - light_configs: [M, N, 12]  (LightDescriptor)
  - light_mask: [M, N] (optional, defaults to ones)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


_NPZ_CACHE: Dict[Path, tuple[int, int, Dict[str, np.ndarray]]] = {}


def clear_lightset_npz_cache() -> None:
    _NPZ_CACHE.clear()


def _load_lightset_npz(data_path: Path, use_cache: bool = True) -> Dict[str, np.ndarray]:
    resolved = data_path.resolve()
    stat = resolved.stat()
    stamp = (int(stat.st_mtime_ns), int(stat.st_size))

    if use_cache:
        cached = _NPZ_CACHE.get(resolved)
        if cached is not None and cached[0] == stamp[0] and cached[1] == stamp[1]:
            return cached[2]

    with np.load(resolved, allow_pickle=True) as data:
        payload = {k: data[k] for k in data.files}

    if use_cache:
        _NPZ_CACHE[resolved] = (stamp[0], stamp[1], payload)
    return payload


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
        use_npz_cache: bool = True,
        use_torch_views: bool = True,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split
        self.normalize_probes = normalize_probes
        self.use_torch_views = bool(use_torch_views)

        data_path = self.data_root / "parametric_tensor.npz"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        data = _load_lightset_npz(data_path, use_cache=use_npz_cache)
        self.tensor = np.asarray(data["tensor"], dtype=np.float32)  # [P, M, 27]
        self.probe_positions = np.asarray(data["probe_positions"], dtype=np.float32)  # [P, 3]
        self.light_configs = np.asarray(data["light_configs"], dtype=np.float32)  # [M, N, 12]
        if "light_mask" in data:
            self.light_mask = np.asarray(data["light_mask"], dtype=np.float32)
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
        self.tensor_subset = np.ascontiguousarray(self.tensor[:, self.config_indices, :], dtype=np.float32)
        self.light_configs_subset = np.ascontiguousarray(self.light_configs[self.config_indices], dtype=np.float32)
        self.light_mask_subset = np.ascontiguousarray(self.light_mask[self.config_indices], dtype=np.float32)

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

        self._probe_positions_tensor: Optional[torch.Tensor] = None
        self._light_configs_tensor: Optional[torch.Tensor] = None
        self._light_mask_tensor: Optional[torch.Tensor] = None
        self._tensor_subset_tensor: Optional[torch.Tensor] = None
        if self.use_torch_views:
            self._probe_positions_tensor = torch.from_numpy(
                np.ascontiguousarray(self.probe_positions_norm, dtype=np.float32)
            )
            self._light_configs_tensor = torch.from_numpy(self.light_configs_subset)
            self._light_mask_tensor = torch.from_numpy(self.light_mask_subset)
            self._tensor_subset_tensor = torch.from_numpy(self.tensor_subset)

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

        if self._probe_positions_tensor is not None:
            probe_pos = self._probe_positions_tensor[probe_idx]
            light_params = self._light_configs_tensor[config_idx]
            light_mask = self._light_mask_tensor[config_idx]
            sh_coeffs = self._tensor_subset_tensor[probe_idx, config_idx, :]
        else:
            probe_pos = torch.from_numpy(self.probe_positions_norm[probe_idx])
            light_params = torch.from_numpy(self.light_configs_subset[config_idx])
            light_mask = torch.from_numpy(self.light_mask_subset[config_idx])
            sh_coeffs = torch.from_numpy(self.tensor_subset[probe_idx, config_idx, :])

        sample = {
            "probe_position": probe_pos,
            "light_params": light_params,
            "light_mask": light_mask,
            "sh_coeffs": sh_coeffs,
            "probe_idx": torch.tensor(probe_idx, dtype=torch.long),
            "config_idx": torch.tensor(config_idx, dtype=torch.long),
        }
        return sample

    def __getitems__(self, indices) -> list[Dict[str, torch.Tensor]]:
        if not indices:
            return []
        idx_tensor = torch.as_tensor(indices, dtype=torch.long)
        probe_idx = torch.div(idx_tensor, self.num_configs, rounding_mode="floor")
        config_idx = torch.remainder(idx_tensor, self.num_configs)

        if self._probe_positions_tensor is not None:
            probe_pos_b = self._probe_positions_tensor[probe_idx]
            light_params_b = self._light_configs_tensor[config_idx]
            light_mask_b = self._light_mask_tensor[config_idx]
            sh_coeffs_b = self._tensor_subset_tensor[probe_idx, config_idx, :]
        else:
            probe_pos_b = torch.from_numpy(self.probe_positions_norm[probe_idx.numpy()])
            light_params_b = torch.from_numpy(self.light_configs_subset[config_idx.numpy()])
            light_mask_b = torch.from_numpy(self.light_mask_subset[config_idx.numpy()])
            sh_coeffs_b = torch.from_numpy(self.tensor_subset[probe_idx.numpy(), config_idx.numpy(), :])

        samples: list[Dict[str, torch.Tensor]] = []
        for i in range(int(idx_tensor.numel())):
            samples.append(
                {
                    "probe_position": probe_pos_b[i],
                    "light_params": light_params_b[i],
                    "light_mask": light_mask_b[i],
                    "sh_coeffs": sh_coeffs_b[i],
                    "probe_idx": probe_idx[i].to(dtype=torch.long),
                    "config_idx": config_idx[i].to(dtype=torch.long),
                }
            )
        return samples

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
    dataloader_timeout_seconds: int = 0,
    dataloader_prefetch_factor: Optional[int] = None,
    dataloader_persistent_workers: Optional[bool] = None,
    dataloader_multiprocessing_context: Optional[str] = None,
    use_npz_cache: bool = True,
    use_torch_views: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = LightSetDataset(
        data_root=data_root,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize_probes=normalize_probes,
        use_npz_cache=use_npz_cache,
        use_torch_views=use_torch_views,
    )

    val_dataset = LightSetDataset(
        data_root=data_root,
        split="val",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize_probes=normalize_probes,
        use_npz_cache=use_npz_cache,
        use_torch_views=use_torch_views,
    )

    test_dataset = LightSetDataset(
        data_root=data_root,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize_probes=normalize_probes,
        use_npz_cache=use_npz_cache,
        use_torch_views=use_torch_views,
    )

    num_workers_i = int(num_workers)
    timeout_i = max(0, int(dataloader_timeout_seconds)) if num_workers_i > 0 else 0

    loader_kwargs = {
        "num_workers": num_workers_i,
        "pin_memory": True,
        "timeout": timeout_i,
    }

    if num_workers_i > 0:
        if dataloader_prefetch_factor is not None and int(dataloader_prefetch_factor) > 0:
            loader_kwargs["prefetch_factor"] = int(dataloader_prefetch_factor)
        if dataloader_persistent_workers is not None:
            loader_kwargs["persistent_workers"] = bool(dataloader_persistent_workers)
        context = None
        if dataloader_multiprocessing_context is not None:
            context = str(dataloader_multiprocessing_context).strip().lower()
        if context and context != "none":
            loader_kwargs["multiprocessing_context"] = context

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
