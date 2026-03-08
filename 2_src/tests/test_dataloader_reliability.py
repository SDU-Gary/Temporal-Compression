from __future__ import annotations

from typing import Any, Dict, List

import torch

from data import lightset_dataset as dataset_mod


class _DummyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"x": torch.tensor(float(idx), dtype=torch.float32)}


def test_loader_kwargs_without_workers(monkeypatch):
    captured: List[Dict[str, Any]] = []

    def _fake_dataset(*args, **kwargs):
        return _DummyDataset()

    def _fake_loader(dataset, batch_size, shuffle, **kwargs):
        captured.append(dict(kwargs))
        return {"dataset": dataset, "batch_size": batch_size, "shuffle": shuffle, "kwargs": kwargs}

    monkeypatch.setattr(dataset_mod, "LightSetDataset", _fake_dataset)
    monkeypatch.setattr(dataset_mod, "DataLoader", _fake_loader)

    dataset_mod.create_dataloaders_lightset(
        data_root="dummy",
        batch_size=2,
        train_ratio=0.7,
        val_ratio=0.15,
        num_workers=0,
        normalize_probes=True,
        dataloader_timeout_seconds=77,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=True,
        dataloader_multiprocessing_context="spawn",
    )

    assert len(captured) == 3
    for kwargs in captured:
        assert kwargs["num_workers"] == 0
        assert kwargs["timeout"] == 0
        assert "prefetch_factor" not in kwargs
        assert "persistent_workers" not in kwargs
        assert "multiprocessing_context" not in kwargs


def test_loader_kwargs_with_workers(monkeypatch):
    captured: List[Dict[str, Any]] = []

    def _fake_dataset(*args, **kwargs):
        return _DummyDataset()

    def _fake_loader(dataset, batch_size, shuffle, **kwargs):
        captured.append(dict(kwargs))
        return {"dataset": dataset, "batch_size": batch_size, "shuffle": shuffle, "kwargs": kwargs}

    monkeypatch.setattr(dataset_mod, "LightSetDataset", _fake_dataset)
    monkeypatch.setattr(dataset_mod, "DataLoader", _fake_loader)

    dataset_mod.create_dataloaders_lightset(
        data_root="dummy",
        batch_size=2,
        train_ratio=0.7,
        val_ratio=0.15,
        num_workers=2,
        normalize_probes=True,
        dataloader_timeout_seconds=15,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=False,
        dataloader_multiprocessing_context="spawn",
    )

    assert len(captured) == 3
    for kwargs in captured:
        assert kwargs["num_workers"] == 2
        assert kwargs["timeout"] == 15
        assert kwargs["prefetch_factor"] == 4
        assert kwargs["persistent_workers"] is False
        assert kwargs["multiprocessing_context"] == "spawn"

