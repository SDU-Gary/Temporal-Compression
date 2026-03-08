from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import data.lightset_dataset as lightset_dataset


def _write_lightset_npz(root: Path, seed: int = 0, num_probes: int = 6, num_configs: int = 8, num_slots: int = 2) -> None:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    tensor = rng.random((num_probes, num_configs, 27), dtype=np.float32)
    probe_positions = rng.random((num_probes, 3), dtype=np.float32)
    light_configs = rng.random((num_configs, num_slots, 12), dtype=np.float32)
    light_mask = np.ones((num_configs, num_slots), dtype=np.float32)
    np.savez(
        root / "parametric_tensor.npz",
        tensor=tensor,
        probe_positions=probe_positions,
        light_configs=light_configs,
        light_mask=light_mask,
    )


def test_lightset_dataset_uses_npz_cache(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "lightset"
    _write_lightset_npz(data_root, seed=1)
    lightset_dataset.clear_lightset_npz_cache()

    calls = {"count": 0}
    original_np_load = lightset_dataset.np.load

    def _counted_np_load(*args, **kwargs):
        calls["count"] += 1
        return original_np_load(*args, **kwargs)

    monkeypatch.setattr(lightset_dataset.np, "load", _counted_np_load)

    lightset_dataset.LightSetDataset(data_root, split="train")
    lightset_dataset.LightSetDataset(data_root, split="val")
    lightset_dataset.LightSetDataset(data_root, split="test")

    assert calls["count"] == 1


def test_lightset_dataset_cache_invalidation_on_file_change(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "lightset"
    _write_lightset_npz(data_root, seed=2)
    lightset_dataset.clear_lightset_npz_cache()

    calls = {"count": 0}
    original_np_load = lightset_dataset.np.load

    def _counted_np_load(*args, **kwargs):
        calls["count"] += 1
        return original_np_load(*args, **kwargs)

    monkeypatch.setattr(lightset_dataset.np, "load", _counted_np_load)

    lightset_dataset.LightSetDataset(data_root, split="train")
    _write_lightset_npz(data_root, seed=3)
    lightset_dataset.LightSetDataset(data_root, split="train")

    assert calls["count"] == 2


def test_lightset_dataset_sample_dtype_is_float32(tmp_path: Path) -> None:
    data_root = tmp_path / "lightset"
    _write_lightset_npz(data_root, seed=5)
    ds = lightset_dataset.LightSetDataset(data_root, split="train")
    sample = ds[0]
    assert sample["probe_position"].dtype == torch.float32
    assert sample["light_params"].dtype == torch.float32
    assert sample["light_mask"].dtype == torch.float32
    assert sample["sh_coeffs"].dtype == torch.float32


def test_lightset_dataset_getitem_avoids_from_numpy_when_views_enabled(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "lightset"
    _write_lightset_npz(data_root, seed=6)
    ds = lightset_dataset.LightSetDataset(data_root, split="train", use_torch_views=True)

    def _blocked_from_numpy(*args, **kwargs):
        raise AssertionError("from_numpy should not be called in __getitem__ when torch views are enabled")

    monkeypatch.setattr(lightset_dataset.torch, "from_numpy", _blocked_from_numpy)
    _ = ds[0]


def test_lightset_dataset_getitem_fallback_without_views(tmp_path: Path) -> None:
    data_root = tmp_path / "lightset"
    _write_lightset_npz(data_root, seed=7)
    ds = lightset_dataset.LightSetDataset(data_root, split="train", use_torch_views=False)
    sample = ds[0]
    assert sample["probe_position"].dtype == torch.float32
