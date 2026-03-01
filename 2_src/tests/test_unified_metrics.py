from __future__ import annotations

import numpy as np

from utils.unified_metrics import (
    SH_L0_INDICES,
    SH_L1_INDICES,
    SH_L2_INDICES,
    compute_benchmark_metrics,
    compute_fixed_tm_metrics,
    compute_hdr_radiance_metrics,
    compute_linear_joint_metrics,
    compute_real_render_hdr_scene_metrics,
    compute_sh_band_metrics,
    compute_sh_metrics,
    compute_sh_metrics_with_rms_normalization,
    split_sh_bands,
)


def test_sh_metrics_identical() -> None:
    sh = np.ones((4, 27), dtype=np.float32) * 0.25
    m = compute_sh_metrics(sh, sh, max_i=1.0)
    assert np.isinf(m["sh_psnr"])
    assert abs(m["sh_ssim"] - 1.0) < 1e-8


def test_hdr_metrics_and_real_hdr_metrics_consistent() -> None:
    gt = np.ones((8, 8, 3), dtype=np.float32) * 0.7
    pred = np.ones((8, 8, 3), dtype=np.float32) * 0.6

    hdr = compute_hdr_radiance_metrics(gt, pred, max_i=1.0)
    real = compute_real_render_hdr_scene_metrics(gt, pred, max_i=1.0)

    assert hdr["hdr_psnr"] > 0.0
    assert abs(hdr["hdr_psnr"] - real["real_render_hdr_psnr"]) < 1e-9
    assert abs(hdr["hdr_ssim"] - real["real_render_hdr_ssim"]) < 1e-9


def test_fixed_tm_metrics_keys() -> None:
    gt = np.ones((8, 8, 3), dtype=np.float32) * 4.0
    pred = np.ones((8, 8, 3), dtype=np.float32) * 2.0
    m = compute_fixed_tm_metrics(gt, pred, exposure=1.0, max_i=1.0)
    assert "fixedtm_psnr" in m
    assert "fixedtm_ssim" in m
    assert m["fixedtm_psnr"] > 0.0


def test_benchmark_metrics_and_joint_linear_scale() -> None:
    gt = np.ones((4, 4, 3), dtype=np.float32) * 2.0
    pred = np.ones((4, 4, 3), dtype=np.float32) * 1.0

    b = compute_benchmark_metrics(
        gt,
        pred,
        metric_align_scale_gt=1.0,
        metric_align_scale_pred=1.0,
        max_i=1.0,
    )
    jl = compute_linear_joint_metrics(gt, pred, max_i=1.0)

    assert "benchmark_psnr" in b
    assert "benchmark_linear_psnr" in b
    assert b["benchmark_linear_scale"] == jl["linear_scale"]
    assert b["benchmark_linear_psnr"] == jl["linear_psnr"]


def test_split_sh_bands_shapes_and_indices() -> None:
    sh = np.arange(54, dtype=np.float32).reshape(2, 27)
    bands = split_sh_bands(sh)

    assert bands["all"].shape == (2, 27)
    assert bands["l0"].shape == (2, 3)
    assert bands["l1"].shape == (2, 9)
    assert bands["l2"].shape == (2, 15)
    assert bands["non_l0"].shape == (2, 24)

    assert np.allclose(bands["l0"], sh[:, list(SH_L0_INDICES)])
    assert np.allclose(bands["l1"], sh[:, list(SH_L1_INDICES)])
    assert np.allclose(bands["l2"], sh[:, list(SH_L2_INDICES)])


def test_sh_band_metrics_keys_exist() -> None:
    gt = np.ones((8, 27), dtype=np.float32)
    pred = gt * 0.9
    m = compute_sh_band_metrics(gt, pred, max_i=1.0)

    for band in ("l0", "l1", "l2"):
        assert f"sh_{band}_psnr" in m
        assert f"sh_{band}_ssim" in m
        assert f"sh_{band}_mae" in m
        assert f"sh_{band}_rmse" in m


def test_sh_rmsnorm_psnr_scale_invariant() -> None:
    rng = np.random.default_rng(123)
    gt = rng.normal(0.0, 0.3, size=(32, 27)).astype(np.float32)
    pred = gt + rng.normal(0.0, 0.05, size=(32, 27)).astype(np.float32)

    m1 = compute_sh_metrics_with_rms_normalization(gt, pred, max_i=1.0)
    m2 = compute_sh_metrics_with_rms_normalization(gt * 7.0, pred * 7.0, max_i=1.0)

    assert abs(m1["sh_rmsnorm_psnr"] - m2["sh_rmsnorm_psnr"]) < 1e-6
    assert abs(m1["sh_rmsnorm_ssim"] - m2["sh_rmsnorm_ssim"]) < 1e-6
