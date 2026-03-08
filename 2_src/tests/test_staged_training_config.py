from __future__ import annotations

from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import torch
import torch.nn as nn


def _load_train_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "3_experiments" / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_entry", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load train.py module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(4, 3))
        self.log_scale = nn.Parameter(torch.randn(4, 3))
        self.U = nn.Parameter(torch.randn(4, 27, 8))
        self.U_l0 = nn.Parameter(torch.randn(4, 3, 8))
        self.coeffs = nn.Parameter(torch.randn(4, 8, 16))
        self.coeffs_l0 = nn.Parameter(torch.randn(4, 8, 16))
        self.light_encoder = nn.Linear(12, 16)
        self.gamma = nn.Sequential(nn.Linear(16, 8))
        self.beta = nn.Sequential(nn.Linear(16, 8))
        self.extra = nn.Parameter(torch.randn(1))


def test_resolve_staged_specs_from_config():
    mod = _load_train_module()

    args = Namespace(
        lr=1e-3,
        weight_decay=0.0,
        lr_scheduler="none",
        lr_min=1e-4,
        warmup_epochs=0,
        _config_obj={
            "training": {
                "staged": {
                    "enabled": True,
                    "stages": [
                        {
                            "name": "coeff",
                            "epochs": 100,
                            "lr": 1e-3,
                            "trainable_groups": ["coeff", "encoder"],
                            "group_lrs": {"coeff": 1e-3, "encoder": 8e-4, "all": 1e-4},
                        },
                        {"name": "joint", "epochs": 50, "trainable_groups": ["all"]},
                    ],
                }
            }
        },
    )

    specs = mod._resolve_staged_specs(args)
    assert len(specs) == 2
    assert specs[0]["name"] == "coeff"
    assert specs[0]["epochs"] == 100
    assert specs[0]["trainable_groups"] == ["coeff", "encoder"]
    assert specs[0]["group_lrs"]["coeff"] == 1e-3
    assert specs[0]["group_lrs"]["encoder"] == 8e-4
    assert specs[0]["group_lrs"]["all"] == 1e-4
    assert specs[1]["name"] == "joint"
    assert specs[1]["trainable_groups"] == ["all"]


def test_set_trainable_groups_coeff_encoder_only():
    mod = _load_train_module()
    model = _DummyModel()

    summary = mod._set_trainable_groups(model, ["coeff", "encoder"])
    assert summary["trainable_param_count"] > 0

    named = dict(model.named_parameters())
    assert named["coeffs"].requires_grad
    assert named["coeffs_l0"].requires_grad
    assert named["light_encoder.weight"].requires_grad
    assert named["light_encoder.bias"].requires_grad

    assert not named["U"].requires_grad
    assert not named["U_l0"].requires_grad
    assert not named["mu"].requires_grad
    assert not named["log_scale"].requires_grad
    assert not named["gamma.0.weight"].requires_grad
    assert not named["beta.0.weight"].requires_grad


def test_resolve_ema_and_oracle_monitor_config_from_yaml_dict():
    mod = _load_train_module()
    args = Namespace(
        seed=7,
        batch_size=256,
        num_workers=2,
        top_k=3,
        _config_obj={
            "training": {
                "ema": {
                    "enabled": True,
                    "decay": 0.9999,
                    "eval_on_ema": True,
                    "save_best_with_ema": True,
                },
                "oracle_monitor": {
                    "enabled": True,
                    "split": "test",
                    "period_epochs": 20,
                    "lightweight_max_samples": 1024,
                    "stage_end_full": True,
                    "full_max_samples": 0,
                    "timeout_seconds": 321,
                    "fail_on_timeout": True,
                },
            }
        },
    )

    ema_cfg = mod._resolve_ema_config(args)
    oracle_cfg = mod._resolve_oracle_monitor_config(args)

    assert ema_cfg["enabled"] is True
    assert ema_cfg["decay"] == 0.9999
    assert ema_cfg["eval_on_ema"] is True
    assert ema_cfg["save_best_with_ema"] is True

    assert oracle_cfg["enabled"] is True
    assert oracle_cfg["split"] == "test"
    assert oracle_cfg["period_epochs"] == 20
    assert oracle_cfg["lightweight_max_samples"] == 1024
    assert oracle_cfg["batch_size"] == 256
    assert oracle_cfg["timeout_seconds"] == 321
    assert oracle_cfg["fail_on_timeout"] is True


def test_update_heartbeat_writes_status_and_history(tmp_path: Path):
    mod = _load_train_module()
    heartbeat_path = tmp_path / "runtime" / "heartbeat.json"
    history_path = tmp_path / "runtime" / "heartbeat.history.jsonl"
    run_context = {"run_id": "unit_test"}

    mod._update_heartbeat(
        heartbeat_path=heartbeat_path,
        history_path=history_path,
        enabled=True,
        save_history=True,
        run_context=run_context,
        state="running",
        last_event="epoch_end",
        stage_index=1,
        stage_name="single_stage",
        epoch=10,
        num_epochs=100,
        metrics={"val_mae": 0.12},
        extra={"note": "ok"},
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload["state"] == "running"
    assert payload["last_event"] == "epoch_end"
    assert payload["metrics"]["val_mae"] == 0.12
    assert payload["run_context"]["run_id"] == "unit_test"

    lines = history_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["state"] == "running"


def test_run_oracle_monitor_propagates_timeout(tmp_path: Path, monkeypatch):
    mod = _load_train_module()
    called: dict = {}

    def _fake_run(cmd, check, timeout):
        called["timeout"] = timeout
        called["check"] = check
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    args = Namespace(
        data_root="dummy",
        batch_size=8,
        seed=42,
        top_k=3,
        device=None,
    )
    oracle_cfg = {
        "split": "test",
        "batch_size": 8,
        "num_workers": 0,
        "top_k": 3,
    }

    with pytest.raises(subprocess.TimeoutExpired):
        mod._run_oracle_monitor(
            checkpoint_path=tmp_path / "best.pt",
            output_json_path=tmp_path / "oracle.json",
            args=args,
            oracle_cfg=oracle_cfg,
            max_samples=16,
            sample_seed=0,
            timeout_seconds=7,
        )

    assert called["check"] is True
    assert called["timeout"] == 7
