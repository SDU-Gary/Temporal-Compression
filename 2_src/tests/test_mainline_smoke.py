from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data.lightset_dataset import LightSetDataset
from models.gaussian_physics_unified import GaussianPhysicsCompressionUnified


def _write_lightset_npz(root: Path, num_probes: int = 4, num_configs: int = 3, num_slots: int = 2) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tensor = np.random.rand(num_probes, num_configs, 27).astype(np.float32)
    probe_positions = np.random.rand(num_probes, 3).astype(np.float32)
    light_configs = np.random.rand(num_configs, num_slots, 12).astype(np.float32)
    light_mask = np.ones((num_configs, num_slots), dtype=np.float32)
    np.savez(
        root / "parametric_tensor.npz",
        tensor=tensor,
        probe_positions=probe_positions,
        light_configs=light_configs,
        light_mask=light_mask,
    )


def test_lightset_dataset_and_model_forward(tmp_path: Path) -> None:
    data_root = tmp_path / "lightset"
    _write_lightset_npz(data_root)

    ds = LightSetDataset(data_root, split="train", train_ratio=0.7, val_ratio=0.15)
    sample = ds[0]

    model = GaussianPhysicsCompressionUnified(
        num_gaussians=4,
        rank=2,
        sh_dim=27,
        light_dim=12,
        embed_dim=8,
        intensity_dim=3,
        intensity_offset=1,
        enable_film=True,
    )

    positions = sample["probe_position"].unsqueeze(0)
    light_params = sample["light_params"].unsqueeze(0)
    light_mask = sample["light_mask"].unsqueeze(0)
    out = model(positions, light_params, top_k=2, light_mask=light_mask)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 27)

