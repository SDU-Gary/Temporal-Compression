from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


suite_mod = _load_module(
    "coeff_suite_mod_test",
    ROOT / "3_experiments" / "scripts" / "analysis" / "run_coeff_learnability_suite.py",
)


def test_vector_metrics_perfect_prediction() -> None:
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y_pred = y_true.copy()
    m = suite_mod._vector_metrics(y_true, y_pred)
    assert abs(m["mse"]) < 1e-12
    assert abs(m["mae"]) < 1e-12
    assert abs(m["cosine_mean"] - 1.0) < 1e-12


def test_linear_ridge_reconstructs_linear_mapping() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(128, 4)).astype(np.float32)
    w_gt = np.array(
        [
            [1.0, -2.0, 0.5],
            [0.2, 0.3, -0.1],
            [1.2, 0.0, 0.4],
            [-0.7, 0.9, 0.2],
            [0.1, -0.2, 0.3],
        ],
        dtype=np.float32,
    )
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    y = x_aug @ w_gt

    w_fit = suite_mod._fit_linear_ridge(x, y, alpha=1e-8)
    y_hat = suite_mod._predict_linear_ridge(x, w_fit)
    mse = float(np.mean((y_hat - y) ** 2))
    assert mse < 1e-8


def test_split_indices_cover_all_samples() -> None:
    train_idx, val_idx, test_idx = suite_mod._split_indices(100, 0.8, seed=42)
    all_idx = np.concatenate([train_idx, val_idx, test_idx], axis=0)
    assert np.unique(all_idx).size == 100
    assert set(all_idx.tolist()) == set(range(100))

