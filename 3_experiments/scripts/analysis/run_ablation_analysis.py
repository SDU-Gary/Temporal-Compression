#!/usr/bin/env python3
"""Run ablation analysis for unified_set models on lightset datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "2_src"
if str(_SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SRC))

from data.lightset_dataset import LightSetDataset
from data.sh_scaler import AdaptiveSHScaler, L0_INDICES
from models.gaussian_physics_unified import GaussianPhysicsCompressionUnified
from training.gaussian_physics_trainer import BatchAdapter


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _aabb_distance(point: np.ndarray, aabb: Tuple[np.ndarray, np.ndarray]) -> float:
    mn, mx = aabb
    dx = max(mn[0] - point[0], 0.0, point[0] - mx[0])
    dy = max(mn[1] - point[1], 0.0, point[1] - mx[1])
    dz = max(mn[2] - point[2], 0.0, point[2] - mx[2])
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def _nearest_aabb_distance(point: np.ndarray, obstacles: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    if not obstacles:
        return float("inf")
    return min(_aabb_distance(point, aabb) for aabb in obstacles)


def _build_model(args: argparse.Namespace, device: torch.device) -> GaussianPhysicsCompressionUnified:
    model = GaussianPhysicsCompressionUnified(
        num_gaussians=args.num_gaussians,
        rank=args.rank,
        sh_dim=27,
        light_dim=args.light_dim,
        embed_dim=args.embed_dim,
        intensity_dim=args.intensity_dim,
        intensity_offset=args.intensity_offset,
        enable_film=not args.disable_film,
    ).to(device)
    return model


def _load_scaler(path: Path | None) -> AdaptiveSHScaler | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return AdaptiveSHScaler.load(path)


def _compute_l0_luma(sh: np.ndarray) -> np.ndarray:
    l0 = sh[..., list(L0_INDICES)]  # [..., 3]
    return l0 @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _l1_direction(sh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # SH layout: [R0..R8, G0..G8, B0..B8]
    cR = sh[..., 1:4]
    cG = sh[..., 10:13]
    cB = sh[..., 19:22]
    c = 0.2126 * cR + 0.7152 * cG + 0.0722 * cB
    # Basis order: Y1-1 (y), Y10 (z), Y11 (x)
    v = np.stack([c[..., 2], c[..., 0], c[..., 1]], axis=-1)
    norm = np.linalg.norm(v, axis=-1)
    v = v / (norm[..., None] + 1e-8)
    return v, norm


def _pairwise_knn(positions: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    P = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    idx = np.argpartition(dist, kth=min(k, P - 1), axis=1)[:, :k]
    dist_k = np.take_along_axis(dist, idx, axis=1)
    return idx, dist_k


def _gradient_magnitude(values: np.ndarray, neighbors: np.ndarray, dist: np.ndarray) -> np.ndarray:
    # values: [P], neighbors: [P, k], dist: [P, k]
    v = values[:, None]
    diffs = np.abs(values[neighbors] - v) / (dist + 1e-8)
    return np.mean(diffs, axis=1)


def run_analysis(args: argparse.Namespace) -> Dict[str, float]:
    device = _device_from_arg(args.device)
    data_root = Path(args.data_root)
    meta_path = data_root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {meta_path}")
    metadata = json.loads(meta_path.read_text())
    obstacles = [
        (np.array(o["min"], dtype=np.float32), np.array(o["max"], dtype=np.float32))
        for o in metadata.get("obstacles", [])
    ]

    dataset = LightSetDataset(
        data_root=data_root,
        split=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
        normalize_probes=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = _build_model(args, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = _load_scaler(Path(args.sh_scaler) if args.sh_scaler else None)
    adapter = BatchAdapter(params_key="light_params", mask_key="light_mask")
    if scaler is not None:
        adapter.target_transform = scaler.transform
        adapter.target_inverse = scaler.inverse

    P = dataset.num_probes
    M = dataset.num_configs
    pred_tensor = np.zeros((P, M, 27), dtype=np.float32)
    gt_tensor = np.zeros((P, M, 27), dtype=np.float32)

    with torch.no_grad():
        for batch in loader:
            unpacked = adapter.unpack(batch, device)
            if len(unpacked) == 4:
                positions, params, targets, mask = unpacked
            else:
                positions, params, targets = unpacked
                mask = None
            preds = model(positions, params, top_k=args.top_k, light_mask=mask)
            preds_eval, targets_eval = adapter.inverse_targets(preds, targets)
            pred_np = preds_eval.detach().cpu().numpy()
            gt_np = targets_eval.detach().cpu().numpy()
            probe_idx = batch["probe_idx"].cpu().numpy().astype(int)
            config_idx = batch["config_idx"].cpu().numpy().astype(int)
            pred_tensor[probe_idx, config_idx, :] = pred_np
            gt_tensor[probe_idx, config_idx, :] = gt_np

    err = pred_tensor - gt_tensor
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    l0_idx = list(L0_INDICES)
    ho_idx = [i for i in range(27) if i not in l0_idx]
    l0_rmse = float(np.sqrt(np.mean(err[..., l0_idx] ** 2)))
    ho_rmse = float(np.sqrt(np.mean(err[..., ho_idx] ** 2)))

    # Near/Open groups
    probe_positions = dataset.probe_positions
    dist = np.array([_nearest_aabb_distance(p, obstacles) for p in probe_positions], dtype=np.float32)
    near_mask = dist < args.near_threshold
    near_err = err[near_mask]
    open_err = err[~near_mask]
    near_rmse = float(np.sqrt(np.mean(near_err ** 2))) if near_err.size else float("nan")
    open_rmse = float(np.sqrt(np.mean(open_err ** 2))) if open_err.size else float("nan")

    # Angular error (L1)
    pred_dirs, pred_norm = _l1_direction(pred_tensor.reshape(-1, 27))
    gt_dirs, gt_norm = _l1_direction(gt_tensor.reshape(-1, 27))
    valid = (pred_norm > 1e-6) & (gt_norm > 1e-6)
    dot = np.sum(pred_dirs * gt_dirs, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    if np.any(valid):
        ang_mean = float(np.mean(ang[valid]))
    else:
        ang_mean = float("nan")

    # Gradient index (L0 luma)
    l0_pred = _compute_l0_luma(pred_tensor)
    l0_gt = _compute_l0_luma(gt_tensor)
    neighbors, dist_k = _pairwise_knn(probe_positions, args.k_neighbors)
    grad_errors = []
    for m in range(M):
        g_pred = _gradient_magnitude(l0_pred[:, m], neighbors, dist_k)
        g_gt = _gradient_magnitude(l0_gt[:, m], neighbors, dist_k)
        grad_errors.append(g_pred - g_gt)
    grad_errors = np.stack(grad_errors, axis=0)
    grad_rmse = float(np.sqrt(np.mean(grad_errors ** 2)))

    # Distance histogram stats
    dist_stats = {
        "min": float(dist.min()),
        "max": float(dist.max()),
        "mean": float(dist.mean()),
        "median": float(np.median(dist)),
        "p10": float(np.quantile(dist, 0.10)),
        "p25": float(np.quantile(dist, 0.25)),
        "p50": float(np.quantile(dist, 0.50)),
        "p75": float(np.quantile(dist, 0.75)),
        "p90": float(np.quantile(dist, 0.90)),
    }

    results = {
        "mae": mae,
        "rmse": rmse,
        "l0_rmse": l0_rmse,
        "ho_rmse": ho_rmse,
        "near_rmse": near_rmse,
        "open_rmse": open_rmse,
        "angular_error_deg": ang_mean,
        "shadow_gradient_rmse": grad_rmse,
        "distance_stats": dist_stats,
        "num_probes": int(P),
        "num_configs": int(M),
        "near_threshold": float(args.near_threshold),
        "k_neighbors": int(args.k_neighbors),
    }

    if args.save_pred_config is not None:
        cfg = int(args.save_pred_config)
        if cfg < 0 or cfg >= M:
            raise ValueError(f"save_pred_config out of range: 0..{M-1}")
        pred_out = np.ascontiguousarray(pred_tensor[:, cfg, :])
        out_path = Path(args.save_pred_out) if args.save_pred_out else (Path(args.output) if args.output else Path.cwd()) / f"pred_sh_config{cfg}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, pred_out)
        results["saved_pred_sh"] = str(out_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation analysis for unified_set models")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-gaussians", type=int, default=20)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--light-dim", type=int, default=12)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--intensity-dim", type=int, default=3)
    parser.add_argument("--intensity-offset", type=int, default=1)
    parser.add_argument("--disable-film", action="store_true")
    parser.add_argument("--sh-scaler", type=str, default=None)
    parser.add_argument("--near-threshold", type=float, default=0.2)
    parser.add_argument("--k-neighbors", type=int, default=6)
    parser.add_argument("--save-pred-config", type=int, default=None)
    parser.add_argument("--save-pred-out", type=str, default=None)
    args = parser.parse_args()

    results = run_analysis(args)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved analysis to {out_path}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
