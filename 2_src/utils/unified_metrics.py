"""Unified PSNR/SSIM metrics framework.

This module centralizes all project metric calculations so benchmark/eval/
rendering pipelines share the exact same implementation.

Metric domains supported:
1) SH domain metrics (direct SH coefficients)
2) Linear HDR radiance metrics (Falcor linear output)
3) Fixed tonemap sRGB metrics (fixed exposure + Reinhard + sRGB)
4) Benchmark display metrics (current benchmark behavior)
5) Real rendered HDR scene metrics (model-SH render vs GT-SH render)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


SH_L0_INDICES: Tuple[int, int, int] = (0, 9, 18)
SH_L1_INDICES: Tuple[int, ...] = (1, 2, 3, 10, 11, 12, 19, 20, 21)
SH_L2_INDICES: Tuple[int, ...] = tuple(i for i in range(27) if i not in SH_L0_INDICES and i not in SH_L1_INDICES)


def _as_np64(arr: Any) -> np.ndarray:
    if hasattr(arr, "detach") and hasattr(arr, "cpu") and hasattr(arr, "numpy"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=np.float64)


def tone_map_reinhard(image: Any) -> np.ndarray:
    img = np.maximum(_as_np64(image), 0.0)
    return img / (1.0 + img)


def srgb_encode(image: Any) -> np.ndarray:
    img = np.maximum(_as_np64(image), 0.0)
    threshold = 0.0031308
    low = 12.92 * img
    high = 1.055 * np.power(img, 1.0 / 2.4) - 0.055
    return np.where(img <= threshold, low, high)


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (ax / max(sigma, 1e-8)) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def _conv1d_reflect(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    arr_pad = np.pad(arr, pad_width, mode="reflect")
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis, arr_pad)


def _gaussian_blur_2d_or_3d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    out = _conv1d_reflect(arr, kernel, axis=0)
    out = _conv1d_reflect(out, kernel, axis=1)
    return out


def compute_psnr(gt: Any, pred: Any, *, max_i: float = 1.0, clip_unit: bool = False) -> float:
    x = _as_np64(gt)
    y = _as_np64(pred)
    if clip_unit:
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
    mse = float(np.mean((x - y) ** 2))
    if mse <= 1e-16:
        return float("inf")
    return float(10.0 * np.log10(max(max_i, 1e-12) ** 2 / mse))


def compute_ssim_image(gt: Any, pred: Any, *, clip_unit: bool = True) -> float:
    x = _as_np64(gt)
    y = _as_np64(pred)
    if clip_unit:
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)

    kernel = _gaussian_kernel(size=11, sigma=1.5)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = _gaussian_blur_2d_or_3d(x, kernel)
    mu_y = _gaussian_blur_2d_or_3d(y, kernel)

    sigma_x2 = _gaussian_blur_2d_or_3d(x * x, kernel) - mu_x * mu_x
    sigma_y2 = _gaussian_blur_2d_or_3d(y * y, kernel) - mu_y * mu_y
    sigma_xy = _gaussian_blur_2d_or_3d(x * y, kernel) - mu_x * mu_y

    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / np.maximum(den, 1e-12)
    return float(np.mean(ssim_map))


def compute_ssim_global(gt: Any, pred: Any, *, max_i: float = 1.0) -> float:
    x = _as_np64(gt).reshape(-1)
    y = _as_np64(pred).reshape(-1)
    if x.size == 0:
        return float("nan")

    c1 = (0.01 * max(max_i, 1e-12)) ** 2
    c2 = (0.03 * max(max_i, 1e-12)) ** 2

    mu_x = float(np.mean(x))
    mu_y = float(np.mean(y))
    sigma_x2 = float(np.mean((x - mu_x) ** 2))
    sigma_y2 = float(np.mean((y - mu_y) ** 2))
    sigma_xy = float(np.mean((x - mu_x) * (y - mu_y)))

    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x2 + sigma_y2 + c2)
    return float(num / max(den, 1e-12))


def compute_pair_metrics(
    gt: Any,
    pred: Any,
    *,
    max_i: float = 1.0,
    clip_unit: bool = True,
    ssim_mode: str = "image",
) -> Dict[str, float]:
    x = _as_np64(gt)
    y = _as_np64(pred)
    if clip_unit:
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)

    mae = float(np.mean(np.abs(x - y)))
    mse = float(np.mean((x - y) ** 2))
    rmse = float(np.sqrt(mse))
    psnr = float("inf") if mse <= 1e-16 else float(10.0 * np.log10(max(max_i, 1e-12) ** 2 / mse))
    if ssim_mode == "global":
        ssim = compute_ssim_global(x, y, max_i=max_i)
    else:
        ssim = compute_ssim_image(x, y, clip_unit=False)
    return {"psnr": psnr, "ssim": float(ssim), "mae": mae, "rmse": rmse}


def compute_sh_metrics(sh_gt: Any, sh_pred: Any, *, max_i: float = 1.0) -> Dict[str, float]:
    m = compute_pair_metrics(sh_gt, sh_pred, max_i=max_i, clip_unit=False, ssim_mode="global")
    return {
        "sh_psnr": float(m["psnr"]),
        "sh_ssim": float(m["ssim"]),
        "sh_mae": float(m["mae"]),
        "sh_rmse": float(m["rmse"]),
    }


def split_sh_bands(sh: Any) -> Dict[str, np.ndarray]:
    arr = _as_np64(sh)
    if arr.shape[-1] != 27:
        raise ValueError(f"Expected SH last dim = 27, got {arr.shape[-1]}")
    return {
        "all": arr,
        "l0": arr[..., list(SH_L0_INDICES)],
        "l1": arr[..., list(SH_L1_INDICES)],
        "l2": arr[..., list(SH_L2_INDICES)],
        "non_l0": arr[..., [i for i in range(27) if i not in SH_L0_INDICES]],
    }


def compute_sh_band_metrics(sh_gt: Any, sh_pred: Any, *, max_i: float = 1.0) -> Dict[str, float]:
    gt_bands = split_sh_bands(sh_gt)
    pred_bands = split_sh_bands(sh_pred)

    out: Dict[str, float] = {}
    for band in ("l0", "l1", "l2"):
        m = compute_pair_metrics(
            gt_bands[band],
            pred_bands[band],
            max_i=max_i,
            clip_unit=False,
            ssim_mode="global",
        )
        out[f"sh_{band}_psnr"] = float(m["psnr"])
        out[f"sh_{band}_ssim"] = float(m["ssim"])
        out[f"sh_{band}_mae"] = float(m["mae"])
        out[f"sh_{band}_rmse"] = float(m["rmse"])
    return out


def compute_sh_metrics_with_rms_normalization(
    sh_gt: Any,
    sh_pred: Any,
    *,
    max_i: float = 1.0,
    eps: float = 1e-8,
    floor_quantile: float = 0.05,
    floor_ratio: float = 0.1,
) -> Dict[str, float]:
    gt = _as_np64(sh_gt)
    pred = _as_np64(sh_pred)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, pred={pred.shape}")
    if gt.shape[-1] != 27:
        raise ValueError(f"Expected SH last dim = 27, got {gt.shape[-1]}")

    gt_2d = gt.reshape(-1, 27)
    pred_2d = pred.reshape(-1, 27)
    rms = np.sqrt(np.mean(gt_2d * gt_2d, axis=-1, keepdims=True))
    q = float(np.quantile(rms, np.clip(floor_quantile, 0.0, 1.0)))
    floor = max(float(eps), float(q * max(floor_ratio, 0.0)))
    scale = np.maximum(rms, floor)
    gt_n = gt_2d / scale
    pred_n = pred_2d / scale

    m = compute_pair_metrics(gt_n, pred_n, max_i=max_i, clip_unit=False, ssim_mode="global")
    return {
        "sh_rmsnorm_psnr": float(m["psnr"]),
        "sh_rmsnorm_ssim": float(m["ssim"]),
        "sh_rmsnorm_mae": float(m["mae"]),
        "sh_rmsnorm_rmse": float(m["rmse"]),
        "sh_rmsnorm_scale_mean": float(np.mean(scale)),
        "sh_rmsnorm_scale_median": float(np.median(scale)),
        "sh_rmsnorm_scale_floor": float(floor),
        "sh_rmsnorm_floor_quantile": float(floor_quantile),
        "sh_rmsnorm_floor_ratio": float(floor_ratio),
    }


def compute_hdr_radiance_metrics(hdr_gt: Any, hdr_pred: Any, *, max_i: float = 1.0) -> Dict[str, float]:
    gt = np.maximum(_as_np64(hdr_gt), 0.0)
    pred = np.maximum(_as_np64(hdr_pred), 0.0)
    m = compute_pair_metrics(gt, pred, max_i=max_i, clip_unit=True, ssim_mode="image")
    return {
        "hdr_psnr": float(m["psnr"]),
        "hdr_ssim": float(m["ssim"]),
        "hdr_mae": float(m["mae"]),
        "hdr_rmse": float(m["rmse"]),
    }


def compute_fixed_tm_metrics(
    hdr_gt: Any,
    hdr_pred: Any,
    *,
    exposure: float = 1.0,
    max_i: float = 1.0,
) -> Dict[str, float]:
    gt = tone_map_reinhard(np.maximum(_as_np64(hdr_gt), 0.0) * float(exposure))
    pred = tone_map_reinhard(np.maximum(_as_np64(hdr_pred), 0.0) * float(exposure))
    gt_srgb = srgb_encode(gt)
    pred_srgb = srgb_encode(pred)
    m = compute_pair_metrics(gt_srgb, pred_srgb, max_i=max_i, clip_unit=True, ssim_mode="image")
    return {
        "fixedtm_psnr": float(m["psnr"]),
        "fixedtm_ssim": float(m["ssim"]),
        "fixedtm_mae": float(m["mae"]),
        "fixedtm_rmse": float(m["rmse"]),
    }


def compute_linear_joint_metrics(gt_linear: Any, pred_linear: Any, *, max_i: float = 1.0) -> Dict[str, float]:
    gt = _as_np64(gt_linear)
    pred = _as_np64(pred_linear)
    linear_scale = float(max(1e-6, float(np.max(gt)), float(np.max(pred))))
    gt_n = np.clip(gt / linear_scale, 0.0, 1.0)
    pred_n = np.clip(pred / linear_scale, 0.0, 1.0)
    m = compute_pair_metrics(gt_n, pred_n, max_i=max_i, clip_unit=True, ssim_mode="image")
    return {
        "linear_psnr": float(m["psnr"]),
        "linear_ssim": float(m["ssim"]),
        "linear_mae": float(m["mae"]),
        "linear_rmse": float(m["rmse"]),
        "linear_scale": linear_scale,
    }


def compute_benchmark_metrics(
    gt_linear_raw: Any,
    pred_linear_raw: Any,
    *,
    metric_align_scale_gt: float = 1.0,
    metric_align_scale_pred: float = 1.0,
    max_i: float = 1.0,
) -> Dict[str, float]:
    gt_linear = _as_np64(gt_linear_raw) * float(metric_align_scale_gt)
    pred_linear = _as_np64(pred_linear_raw) * float(metric_align_scale_pred)

    gt_img = srgb_encode(tone_map_reinhard(gt_linear))
    pred_img = srgb_encode(tone_map_reinhard(pred_linear))
    display = compute_pair_metrics(gt_img, pred_img, max_i=max_i, clip_unit=True, ssim_mode="image")
    linear = compute_linear_joint_metrics(gt_linear, pred_linear, max_i=max_i)

    return {
        "benchmark_psnr": float(display["psnr"]),
        "benchmark_ssim": float(display["ssim"]),
        "benchmark_mae": float(display["mae"]),
        "benchmark_rmse": float(display["rmse"]),
        "benchmark_linear_psnr": float(linear["linear_psnr"]),
        "benchmark_linear_ssim": float(linear["linear_ssim"]),
        "benchmark_linear_mae": float(linear["linear_mae"]),
        "benchmark_linear_rmse": float(linear["linear_rmse"]),
        "benchmark_linear_scale": float(linear["linear_scale"]),
        "metric_align_scale_gt": float(metric_align_scale_gt),
        "metric_align_scale_model": float(metric_align_scale_pred),
    }


def compute_real_render_hdr_scene_metrics(
    gt_render_hdr: Any,
    model_render_hdr: Any,
    *,
    max_i: float = 1.0,
) -> Dict[str, float]:
    m = compute_hdr_radiance_metrics(gt_render_hdr, model_render_hdr, max_i=max_i)
    return {
        "real_render_hdr_psnr": float(m["hdr_psnr"]),
        "real_render_hdr_ssim": float(m["hdr_ssim"]),
        "real_render_hdr_mae": float(m["hdr_mae"]),
        "real_render_hdr_rmse": float(m["hdr_rmse"]),
    }


def compute_benchmark_srgb_metrics(pt_srgb: Any, route_srgb: Any, *, max_i: float = 1.0) -> Dict[str, float]:
    m = compute_pair_metrics(pt_srgb, route_srgb, max_i=max_i, clip_unit=True, ssim_mode="image")
    return {
        "psnr": float(m["psnr"]),
        "ssim": float(m["ssim"]),
        "mae": float(m["mae"]),
        "rmse": float(m["rmse"]),
    }
