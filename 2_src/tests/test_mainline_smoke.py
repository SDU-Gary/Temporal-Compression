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


def test_forward_with_routing_dense_masked_matches_gather() -> None:
    torch.manual_seed(7)
    model = GaussianPhysicsCompressionUnified(
        num_gaussians=6,
        rank=3,
        sh_dim=27,
        light_dim=12,
        embed_dim=8,
        intensity_dim=3,
        intensity_offset=1,
        enable_film=True,
    )

    batch_size = 4
    num_slots = 3
    positions = torch.randn(batch_size, 3, dtype=torch.float32)
    light_params = torch.randn(batch_size, num_slots, 12, dtype=torch.float32)
    light_mask = torch.ones(batch_size, num_slots, dtype=torch.float32)
    routing = model.compute_gaussian_routing(positions, top_k=3)

    pred_gather = model.forward_with_routing(
        routing,
        light_params,
        light_mask=light_mask,
        param_select_mode="gather",
    )
    pred_dense = model.forward_with_routing(
        routing,
        light_params,
        light_mask=light_mask,
        param_select_mode="dense_masked",
    )

    assert pred_gather.shape == pred_dense.shape
    assert torch.allclose(pred_gather, pred_dense, atol=1e-5, rtol=1e-4)


def test_forward_with_routing_fused_matches_legacy() -> None:
    torch.manual_seed(11)
    model = GaussianPhysicsCompressionUnified(
        num_gaussians=6,
        rank=3,
        sh_dim=27,
        light_dim=12,
        embed_dim=8,
        intensity_dim=3,
        intensity_offset=1,
        enable_film=True,
    )

    batch_size = 5
    num_slots = 3
    positions = torch.randn(batch_size, 3, dtype=torch.float32)
    light_params = torch.randn(batch_size, num_slots, 12, dtype=torch.float32)
    light_mask = torch.ones(batch_size, num_slots, dtype=torch.float32)
    routing = model.compute_gaussian_routing(positions, top_k=3)

    pred_legacy = model.forward_with_routing(
        routing,
        light_params,
        light_mask=light_mask,
        param_select_mode="dense_masked",
        contraction_mode="legacy",
    )
    pred_fused = model.forward_with_routing(
        routing,
        light_params,
        light_mask=light_mask,
        param_select_mode="dense_masked",
        contraction_mode="fused",
    )

    assert pred_legacy.shape == pred_fused.shape
    assert torch.allclose(pred_legacy, pred_fused, atol=1e-5, rtol=1e-4)
