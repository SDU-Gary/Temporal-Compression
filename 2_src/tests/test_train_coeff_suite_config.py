from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


train_script = _load_module("train_script_coeff_suite_test", ROOT / "3_experiments" / "scripts" / "train.py")


def test_resolve_coeff_suite_config_enabled() -> None:
    args = Namespace(
        seed=7,
        batch_size=256,
        top_k=3,
        _config_obj={
            "training": {
                "coeff_suite": {
                    "enabled": True,
                    "run_after_training": True,
                    "split": "test",
                    "device": "cpu",
                    "max_samples": 4096,
                    "sample_seed": 17,
                    "mlp_hidden": 64,
                }
            }
        },
    )
    cfg = train_script._resolve_coeff_suite_config(args)
    assert cfg["enabled"] is True
    assert cfg["max_samples"] == 4096
    assert cfg["sample_seed"] == 17
    assert cfg["mlp_hidden"] == 64


def test_phase0_metrics_extraction() -> None:
    payload = {
        "train_metrics": {
            "total": 1.2,
            "recon": 0.9,
            "grad_post_clip/total": 10.0,
            "grad_post_clip/routing": 1.0,
            "grad_post_clip/basis": 5.0,
            "grad_post_clip/coeff": 2.0,
            "grad_post_clip/encoder": 1.5,
            "grad_post_clip/film": 0.5,
        },
        "val_metrics": {"mae": 0.123},
    }
    m = train_script._phase0_metrics_from_payload(payload)
    assert abs(m["train_total"] - 1.2) < 1e-12
    assert abs(m["val_mae"] - 0.123) < 1e-12
    assert abs(m["grad_ratio_coeff_over_basis"] - 0.4) < 1e-12
    assert abs(m["grad_ratio_coeff_over_total"] - 0.2) < 1e-12


def test_coeff_suite_metrics_from_summary() -> None:
    summary = {
        "phase1": {
            "coeff_alignment": {
                "pred_vs_oracle": {
                    "mse": 0.01,
                    "cosine_mean": 0.9,
                    "r2": 0.8,
                }
            },
            "sh_reconstruction": {
                "pred": {"all27": {"sh_psnr": 10.0}},
                "oracle": {"all27": {"sh_psnr": 30.0}},
                "gains_over_pred_db": {"oracle_all27": 20.0},
            },
        }
    }
    m = train_script._coeff_suite_metrics_from_summary(summary)
    assert abs(m["coeff_pred_oracle_mse"] - 0.01) < 1e-12
    assert abs(m["coeff_pred_oracle_cos"] - 0.9) < 1e-12
    assert abs(m["coeff_gap_all27_db"] - 20.0) < 1e-12

