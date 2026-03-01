from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from utils.unified_metrics import compute_sh_metrics_with_rms_normalization


ROOT = Path(__file__).resolve().parents[2]
DIAG_PATH = ROOT / "3_experiments" / "scripts" / "analysis" / "run_sh_oracle_diagnostics.py"
_spec = importlib.util.spec_from_file_location("run_sh_oracle_diagnostics", str(DIAG_PATH))
diag = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(diag)


def test_oracle_non_l0_upper_bound_on_synthetic_case() -> None:
    torch.manual_seed(42)
    bsz, top_k, rank = 16, 3, 2
    n_coeff = top_k * rank

    weights = torch.rand(bsz, top_k)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    selected_u = torch.randn(bsz, top_k, 27, rank) * 0.2
    selected_u_l0 = torch.randn(bsz, top_k, 3, rank) * 0.2

    # Make L0 rows strong to mimic real-world L0 dominance.
    l0_cols = list(diag.SH_L0_INDICES)
    selected_u[:, :, l0_cols, :] *= 10.0

    # Build synthetic target from hidden oracle weights.
    x_true_non_l0 = torch.randn(bsz, n_coeff)
    x_true_l0 = torch.randn(bsz, n_coeff)

    a_main = (selected_u * weights[:, :, None, None]).permute(0, 2, 1, 3).reshape(bsz, 27, n_coeff)
    a_l0 = (selected_u_l0 * weights[:, :, None, None]).permute(0, 2, 1, 3).reshape(bsz, 3, n_coeff)

    target = torch.bmm(a_main, x_true_non_l0.unsqueeze(-1)).squeeze(-1)
    l0_target = torch.nn.functional.softplus(torch.bmm(a_l0, x_true_l0.unsqueeze(-1)).squeeze(-1))
    target[:, l0_cols[0]] = l0_target[:, 0]
    target[:, l0_cols[1]] = l0_target[:, 1]
    target[:, l0_cols[2]] = l0_target[:, 2]

    # Baseline from random weights.
    x_base_non_l0 = torch.randn(bsz, n_coeff)
    x_base_l0 = torch.randn(bsz, n_coeff)
    baseline = torch.bmm(a_main, x_base_non_l0.unsqueeze(-1)).squeeze(-1)
    base_l0 = torch.nn.functional.softplus(torch.bmm(a_l0, x_base_l0.unsqueeze(-1)).squeeze(-1))
    baseline[:, l0_cols[0]] = base_l0[:, 0]
    baseline[:, l0_cols[1]] = base_l0[:, 1]
    baseline[:, l0_cols[2]] = base_l0[:, 2]

    comp = {
        "weights": weights,
        "selected_u": selected_u,
        "selected_u_l0": selected_u_l0,
    }
    oracle = diag._oracle_timeweights_only(target, comp)

    non_l0 = [i for i in range(27) if i not in diag.SH_L0_INDICES]
    mse_base_non_l0 = torch.mean((baseline[:, non_l0] - target[:, non_l0]) ** 2, dim=-1)
    mse_oracle_non_l0 = torch.mean((oracle[:, non_l0] - target[:, non_l0]) ** 2, dim=-1)

    assert torch.all(mse_oracle_non_l0 <= mse_base_non_l0 + 1e-8)


def test_sh_rmsnorm_robust_floor_avoids_extreme_blowup() -> None:
    gt = torch.zeros(64, 27)
    pred = torch.zeros(64, 27)
    gt[0, 0] = 1.0
    pred[0, 0] = 0.8

    m = compute_sh_metrics_with_rms_normalization(gt, pred, max_i=1.0)
    assert m["sh_rmsnorm_scale_floor"] > 0.0
    assert m["sh_rmsnorm_psnr"] > -80.0
