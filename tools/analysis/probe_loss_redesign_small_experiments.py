#!/usr/bin/env python3
"""Small offline experiments for loss-redesign validation.

This script intentionally avoids any multi-epoch retraining. It probes existing
checkpoints with three lightweight analyses:

1. Checkpoint ranking / objective alignment
2. Route-profile consistency gap
3. Single-batch gradient conflict

Outputs are written as JSON/CSV files under the chosen output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


_SCRIPT_DIR = Path(__file__).resolve().parent
_EXP_SCRIPT_DIR = _SCRIPT_DIR.parent.parent / "3_experiments" / "scripts"
if str(_EXP_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_SCRIPT_DIR))

from _path_setup import ensure_repo_paths


ROOT, SRC = ensure_repo_paths(__file__, root_levels=2)

from data.lightset_dataset import LightSetDataset
from data.sh_scaler import AdaptiveSHScaler
from models.gaussian_physics_unified import GaussianPhysicsCompressionUnified
from training.gaussian_physics_trainer import _param_group_from_name
from utils.unified_metrics import SH_L0_INDICES, SH_L1_INDICES


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + epsilon * epsilon))


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    if xa.size < 2:
        return float("nan")
    if np.allclose(xa.std(), 0.0) or np.allclose(ya.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    if xa.size < 2:
        return float("nan")
    return pearson(_rankdata(xa), _rankdata(ya))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe loss-redesign hypotheses with small offline experiments.")
    p.add_argument(
        "--checkpoint-glob",
        default="3_experiments/results/bistro_clean_v2/exp_*/global_best_falcor.pt",
        help="Glob for checkpoints used in ranking experiment.",
    )
    p.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Optional explicit checkpoint list. Overrides --checkpoint-glob when provided.",
    )
    p.add_argument(
        "--data-root",
        default="1_data_generation/output/bistro_clean_v2",
        help="Dataset root containing parametric_tensor.npz",
    )
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--grad-split", default="train", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-samples", type=int, default=2048)
    p.add_argument("--grad-batch-size", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--radiance-samples", type=int, default=64)
    p.add_argument("--irradiance-samples", type=int, default=64)
    p.add_argument("--teacher-profile", default="soft8_t018")
    p.add_argument(
        "--deploy-profiles",
        nargs="*",
        default=["soft3_t018", "soft3_t022", "hard3"],
        help="Profiles compared against teacher for route consistency.",
    )
    p.add_argument(
        "--gradient-checkpoint",
        default="3_experiments/results/bistro_clean_v2/exp_A2_3_3_disable_loss_eff_seed19/global_best_falcor.pt",
        help="Checkpoint used for single-batch gradient conflict analysis.",
    )
    p.add_argument(
        "--output-dir",
        default="3_experiments/results/analysis/loss_redesign_small_experiments/latest",
        help="Directory to store JSON/CSV reports.",
    )
    return p.parse_args()


@dataclass
class Profile:
    name: str
    top_k: int
    training_soft_routing: bool
    routing_soft_topk: Optional[int]
    routing_temperature: Optional[float]


def parse_profile(spec: str) -> Profile:
    s = str(spec).strip().lower()
    if s == "hard3":
        return Profile(name=s, top_k=3, training_soft_routing=False, routing_soft_topk=None, routing_temperature=None)
    if s.startswith("soft"):
        head, temp_part = s.split("_t", 1)
        soft_topk = int(head.replace("soft", ""))
        if "." in temp_part:
            temp = float(temp_part)
        else:
            temp_digits = temp_part.lstrip("0")
            if len(temp_part) >= 2:
                temp = float(int(temp_part)) / 100.0
            else:
                temp = float(temp_digits or "0") / 10.0
        return Profile(
            name=s,
            top_k=3,
            training_soft_routing=True,
            routing_soft_topk=soft_topk,
            routing_temperature=temp,
        )
    raise ValueError(f"Unsupported profile spec: {spec}")


def discover_checkpoints(args: argparse.Namespace) -> List[Path]:
    if args.checkpoints:
        paths = [Path(p).resolve() for p in args.checkpoints]
    else:
        paths = sorted(ROOT.glob(args.checkpoint_glob))
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError("No checkpoints found for experiment.")
    return paths


def build_scaler_from_meta(meta: Dict[str, Any]) -> Optional[AdaptiveSHScaler]:
    scaler_meta = meta.get("sh_scaler")
    if not isinstance(scaler_meta, dict):
        return None
    return AdaptiveSHScaler(
        l0_mean=np.array(scaler_meta.get("l0_mean", [0.0, 0.0, 0.0]), dtype=np.float32),
        l0_std=np.array(scaler_meta.get("l0_std", [1.0, 1.0, 1.0]), dtype=np.float32),
        ho_rms=np.array(scaler_meta.get("ho_rms", [1.0, 1.0, 1.0]), dtype=np.float32),
        eps=float(scaler_meta.get("eps", 1e-6)),
    )


def infer_split_meta(meta: Dict[str, Any]) -> Tuple[float, float]:
    split_meta = meta.get("split", {})
    if isinstance(split_meta, dict):
        return float(split_meta.get("train_ratio", 0.7)), float(split_meta.get("val_ratio", 0.15))
    return 0.7, 0.15


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[GaussianPhysicsCompressionUnified, Dict[str, Any], Optional[AdaptiveSHScaler]]:
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    K = int(state["mu"].shape[0])
    rank = int(state["U"].shape[-1])
    embed_dim = int(state["coeffs"].shape[-1])
    sh_dim = int(state["U"].shape[1])

    model_meta = meta.get("model", {}) if isinstance(meta, dict) else {}
    model = GaussianPhysicsCompressionUnified(
        num_gaussians=K,
        rank=rank,
        sh_dim=sh_dim,
        light_dim=int(model_meta.get("light_dim", 12)),
        embed_dim=embed_dim,
        intensity_dim=int(model_meta.get("intensity_dim", 3)),
        intensity_offset=int(model_meta.get("intensity_offset", 1)),
        enable_film=bool(model_meta.get("enable_film", True)),
        light_encoder_mode=str(model_meta.get("light_encoder_mode", "normal")),
        bypass_feature_pairs=model_meta.get("bypass_feature_pairs", None),
        bypass_feature_norm_mean=model_meta.get("bypass_feature_norm_mean", None),
        bypass_feature_norm_std=model_meta.get("bypass_feature_norm_std", None),
        contraction_mode=str(model_meta.get("contraction_mode", "fused")),
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State mismatch for {checkpoint_path}: missing={missing}, unexpected={unexpected}")
    model.eval()
    scaler = build_scaler_from_meta(meta if isinstance(meta, dict) else {})
    return model, meta, scaler


def build_dataset(
    data_root: str | Path,
    split: str,
    train_ratio: float,
    val_ratio: float,
    *,
    max_samples: int,
    seed: int,
) -> LightSetDataset | Subset:
    dataset = LightSetDataset(
        data_root=data_root,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize_probes=True,
    )
    if max_samples > 0 and len(dataset) > max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(dataset), size=max_samples, replace=False))
        return Subset(dataset, indices.tolist())
    return dataset


def collate_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
    return out


def build_random_dirs(count: int, seed: int, ref: torch.Tensor) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    u = torch.rand((count,), generator=gen, dtype=torch.float32)
    v = torch.rand((count,), generator=gen, dtype=torch.float32)
    z = 2.0 * u - 1.0
    phi = 2.0 * math.pi * v
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    return torch.stack([x, y, z], dim=-1).to(device=ref.device, dtype=ref.dtype)


def build_seeded_rotation_matrix(seed: int, ref: torch.Tensor) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    u1, u2, u3 = torch.rand((3,), generator=gen, dtype=torch.float32)
    two_pi = float(2.0 * np.pi)
    qx = torch.sqrt(1.0 - u1) * torch.sin(two_pi * u2)
    qy = torch.sqrt(1.0 - u1) * torch.cos(two_pi * u2)
    qz = torch.sqrt(u1) * torch.sin(two_pi * u3)
    qw = torch.sqrt(u1) * torch.cos(two_pi * u3)
    x, y, z, w = qx, qy, qz, qw
    row0 = torch.stack(
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        dim=0,
    )
    row1 = torch.stack(
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        dim=0,
    )
    row2 = torch.stack(
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        dim=0,
    )
    rot = torch.stack([row0, row1, row2], dim=0).to(dtype=torch.float32)
    return rot.to(device=ref.device, dtype=ref.dtype)


def build_fibonacci_dirs(count: int, seed: int, ref: torch.Tensor) -> torch.Tensor:
    i = torch.arange(int(count), dtype=torch.float32) + 0.5
    y = 1.0 - 2.0 * i / float(max(1, count))
    r = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
    golden = float(np.pi * (3.0 - np.sqrt(5.0)))
    theta = golden * i
    x = r * torch.cos(theta)
    z = r * torch.sin(theta)
    dirs = torch.stack([x, y, z], dim=-1).to(device=ref.device, dtype=ref.dtype)
    rot = build_seeded_rotation_matrix(seed, ref)
    return torch.matmul(dirs, rot.transpose(0, 1))


def build_sh_basis_from_dirs(dirs: torch.Tensor) -> torch.Tensor:
    xb, yb, zb = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    return torch.stack(
        [
            torch.full_like(xb, 0.282095),
            0.488603 * yb,
            0.488603 * zb,
            0.488603 * xb,
            1.092548 * xb * yb,
            1.092548 * yb * zb,
            0.315392 * (3.0 * zb * zb - 1.0),
            1.092548 * xb * zb,
            0.546274 * (xb * xb - yb * yb),
        ],
        dim=-1,
    )


def render_radiance_samples(sh: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    sh_rgb = torch.stack([sh[..., 0:9], sh[..., 9:18], sh[..., 18:27]], dim=-2)
    image = torch.einsum("...cn,sn->...sc", sh_rgb, basis)
    return torch.clamp(image, min=0.0)


def render_irradiance_samples(sh: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    sh_rgb = torch.stack([sh[..., 0:9], sh[..., 9:18], sh[..., 18:27]], dim=-2)
    w = torch.tensor(
        [
            float(np.pi),
            float(2.0 * np.pi / 3.0),
            float(2.0 * np.pi / 3.0),
            float(2.0 * np.pi / 3.0),
            float(np.pi / 4.0),
            float(np.pi / 4.0),
            float(np.pi / 4.0),
            float(np.pi / 4.0),
            float(np.pi / 4.0),
        ],
        device=sh.device,
        dtype=sh.dtype,
    )
    view_shape = [1] * (sh_rgb.dim() - 1) + [9]
    sh_rgb_weighted = sh_rgb * w.view(*view_shape)
    image = torch.einsum("...cn,sn->...sc", sh_rgb_weighted, basis)
    image = torch.nn.functional.softplus(image, beta=8.0)
    return image


def render_radiance_mix_loss(
    pred_sh: torch.Tensor,
    target_sh: torch.Tensor,
    basis: torch.Tensor,
    *,
    epsilon: float,
    log_mix_weight: float = 0.3,
) -> torch.Tensor:
    pred_img = render_radiance_samples(pred_sh, basis)
    target_img = render_radiance_samples(target_sh, basis)
    linear = charbonnier_loss(pred_img, target_img, epsilon=epsilon)
    pred_log = torch.log1p(torch.clamp(pred_img, min=0.0))
    target_log = torch.log1p(torch.clamp(target_img, min=0.0))
    log = charbonnier_loss(pred_log, target_log, epsilon=epsilon)
    w = float(np.clip(log_mix_weight, 0.0, 1.0))
    return (1.0 - w) * linear + w * log


def profile_forward(
    model: GaussianPhysicsCompressionUnified,
    positions: torch.Tensor,
    light_params: torch.Tensor,
    light_mask: torch.Tensor,
    profile: Profile,
) -> torch.Tensor:
    return model(
        positions,
        light_params,
        top_k=profile.top_k,
        light_mask=light_mask,
        training_soft_routing=profile.training_soft_routing,
        routing_temperature=1.0 if profile.routing_temperature is None else profile.routing_temperature,
        routing_soft_topk=profile.routing_soft_topk,
    )


def maybe_inverse_scaler(sh: torch.Tensor, scaler: Optional[AdaptiveSHScaler]) -> torch.Tensor:
    if scaler is None:
        return sh
    return scaler.inverse(sh)


def summary_path_for_checkpoint(checkpoint: Path) -> Optional[Path]:
    stem = checkpoint.stem
    candidate = checkpoint.with_name(f"{stem}_summary.json")
    if candidate.exists():
        return candidate
    if checkpoint.name == "global_best_falcor.pt":
        candidate = checkpoint.with_name("global_best_falcor_summary.json")
        if candidate.exists():
            return candidate
    return None


def falcor_target_for_checkpoint(checkpoint: Path) -> float:
    p = summary_path_for_checkpoint(checkpoint)
    if p is None:
        return float("nan")
    obj = json.loads(p.read_text(encoding="utf-8"))
    return float(obj.get("value", float("nan")))


def experiment_name_from_checkpoint(checkpoint: Path) -> str:
    return checkpoint.parent.name


def compute_checkpoint_metrics(
    checkpoint_path: Path,
    dataset: LightSetDataset | Subset,
    device: torch.device,
    teacher: Profile,
    deploy_profiles: Sequence[Profile],
    *,
    batch_size: int,
    radiance_basis: torch.Tensor,
    irradiance_basis: torch.Tensor,
    epsilon: float,
) -> Dict[str, float | str]:
    model, meta, scaler = load_checkpoint_model(checkpoint_path, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    totals: Dict[str, float] = {
        "sample_count": 0.0,
        "recon_full": 0.0,
        "recon_l0": 0.0,
        "recon_l0l1": 0.0,
        "recon_l2": 0.0,
        "radiance_mix": 0.0,
        "irradiance": 0.0,
    }
    for prof in deploy_profiles:
        for suffix in ("sh_mae", "radiance_mae", "irradiance_mae"):
            totals[f"route_gap__{teacher.name}__{prof.name}__{suffix}"] = 0.0

    l0l1_idx = list(SH_L0_INDICES) + list(SH_L1_INDICES)
    l2_idx = [i for i in range(27) if i not in set(l0l1_idx)]

    with torch.inference_mode():
        for batch in loader:
            batch = collate_to_device(batch, device)
            positions = batch["probe_position"]
            light_params = batch["light_params"]
            light_mask = batch["light_mask"]
            target = batch["sh_coeffs"]

            pred_teacher = profile_forward(model, positions, light_params, light_mask, teacher)
            pred_teacher = maybe_inverse_scaler(pred_teacher, scaler)
            target_phys = target

            bsz = int(positions.shape[0])
            totals["sample_count"] += float(bsz)
            totals["recon_full"] += float(charbonnier_loss(pred_teacher, target_phys, epsilon=epsilon).item()) * bsz
            totals["recon_l0"] += float(
                charbonnier_loss(
                    pred_teacher[..., list(SH_L0_INDICES)],
                    target_phys[..., list(SH_L0_INDICES)],
                    epsilon=epsilon,
                ).item()
            ) * bsz
            totals["recon_l0l1"] += float(
                charbonnier_loss(pred_teacher[..., l0l1_idx], target_phys[..., l0l1_idx], epsilon=epsilon).item()
            ) * bsz
            totals["recon_l2"] += float(
                charbonnier_loss(pred_teacher[..., l2_idx], target_phys[..., l2_idx], epsilon=epsilon).item()
            ) * bsz
            totals["radiance_mix"] += float(
                render_radiance_mix_loss(
                    pred_teacher,
                    target_phys,
                    radiance_basis,
                    epsilon=epsilon,
                    log_mix_weight=0.3,
                ).item()
            ) * bsz
            totals["irradiance"] += float(
                charbonnier_loss(
                    render_irradiance_samples(pred_teacher, irradiance_basis),
                    render_irradiance_samples(target_phys, irradiance_basis),
                    epsilon=epsilon,
                ).item()
            ) * bsz

            teacher_radiance = render_radiance_samples(pred_teacher, radiance_basis)
            teacher_irradiance = render_irradiance_samples(pred_teacher, irradiance_basis)
            for prof in deploy_profiles:
                pred_other = profile_forward(model, positions, light_params, light_mask, prof)
                pred_other = maybe_inverse_scaler(pred_other, scaler)
                key_prefix = f"route_gap__{teacher.name}__{prof.name}"
                totals[f"{key_prefix}__sh_mae"] += float(torch.mean(torch.abs(pred_teacher - pred_other)).item()) * bsz
                totals[f"{key_prefix}__radiance_mae"] += float(
                    torch.mean(torch.abs(teacher_radiance - render_radiance_samples(pred_other, radiance_basis))).item()
                ) * bsz
                totals[f"{key_prefix}__irradiance_mae"] += float(
                    torch.mean(torch.abs(teacher_irradiance - render_irradiance_samples(pred_other, irradiance_basis))).item()
                ) * bsz

    denom = max(1.0, totals.pop("sample_count"))
    out: Dict[str, float | str] = {
        "experiment": experiment_name_from_checkpoint(checkpoint_path),
        "checkpoint": str(checkpoint_path),
        "falcor_hdr_psnr": falcor_target_for_checkpoint(checkpoint_path),
    }
    for k, v in totals.items():
        out[k] = float(v / denom)
    return out


def flatten_grads(
    grads: Sequence[Optional[torch.Tensor]],
    params: Sequence[Tuple[str, torch.nn.Parameter]],
) -> Dict[str, np.ndarray]:
    buckets: Dict[str, List[np.ndarray]] = {"all": []}
    for (name, param), grad in zip(params, grads):
        if grad is None:
            g = np.zeros(param.numel(), dtype=np.float32)
        else:
            g = grad.detach().reshape(-1).to(device="cpu", dtype=torch.float32).numpy()
        buckets["all"].append(g)
        group = _param_group_from_name(name)
        buckets.setdefault(group, []).append(g)
    return {k: np.concatenate(v, axis=0) if v else np.zeros((0,), dtype=np.float32) for k, v in buckets.items()}


def compute_gradient_conflicts(
    checkpoint_path: Path,
    dataset: LightSetDataset | Subset,
    device: torch.device,
    teacher: Profile,
    deploy_profiles: Sequence[Profile],
    *,
    batch_size: int,
    radiance_basis: torch.Tensor,
    irradiance_basis: torch.Tensor,
    epsilon: float,
) -> Dict[str, Any]:
    model, meta, scaler = load_checkpoint_model(checkpoint_path, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    batch = next(iter(loader))
    batch = collate_to_device(batch, device)

    positions = batch["probe_position"]
    light_params = batch["light_params"]
    light_mask = batch["light_mask"]
    target = batch["sh_coeffs"]

    pred_teacher_scaled = profile_forward(model, positions, light_params, light_mask, teacher)
    pred_teacher = maybe_inverse_scaler(pred_teacher_scaled, scaler)
    target_phys = target

    soft3_t018 = next((p for p in deploy_profiles if p.name == "soft3_t018"), None)
    hard3 = next((p for p in deploy_profiles if p.name == "hard3"), None)
    pred_soft3 = None
    pred_hard3 = None
    if soft3_t018 is not None:
        pred_soft3 = maybe_inverse_scaler(
            profile_forward(model, positions, light_params, light_mask, soft3_t018),
            scaler,
        )
    if hard3 is not None:
        pred_hard3 = maybe_inverse_scaler(
            profile_forward(model, positions, light_params, light_mask, hard3),
            scaler,
        )

    l0l1_idx = list(SH_L0_INDICES) + list(SH_L1_INDICES)
    l2_idx = [i for i in range(27) if i not in set(l0l1_idx)]
    loss_items: List[Tuple[str, torch.Tensor]] = [
        ("recon_full", charbonnier_loss(pred_teacher, target_phys, epsilon=epsilon)),
        (
            "recon_l0",
            charbonnier_loss(
                pred_teacher[..., list(SH_L0_INDICES)],
                target_phys[..., list(SH_L0_INDICES)],
                epsilon=epsilon,
            ),
        ),
        ("recon_l0l1", charbonnier_loss(pred_teacher[..., l0l1_idx], target_phys[..., l0l1_idx], epsilon=epsilon)),
        ("recon_l2", charbonnier_loss(pred_teacher[..., l2_idx], target_phys[..., l2_idx], epsilon=epsilon)),
        (
            "radiance_mix",
            render_radiance_mix_loss(pred_teacher, target_phys, radiance_basis, epsilon=epsilon, log_mix_weight=0.3),
        ),
        (
            "irradiance",
            charbonnier_loss(
                render_irradiance_samples(pred_teacher, irradiance_basis),
                render_irradiance_samples(target_phys, irradiance_basis),
                epsilon=epsilon,
            ),
        ),
    ]
    if pred_soft3 is not None:
        loss_items.append(("route_consistency_soft3", charbonnier_loss(pred_teacher, pred_soft3, epsilon=epsilon)))
    if pred_hard3 is not None:
        loss_items.append(("route_consistency_hard3", charbonnier_loss(pred_teacher, pred_hard3, epsilon=epsilon)))

    named_params: List[Tuple[str, torch.nn.Parameter]] = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    gradients: Dict[str, Dict[str, np.ndarray]] = {}
    loss_scalars: Dict[str, float] = {}
    for i, (loss_name, loss_val) in enumerate(loss_items):
        grads = torch.autograd.grad(
            loss_val,
            [p for _, p in named_params],
            retain_graph=(i < len(loss_items) - 1),
            allow_unused=True,
        )
        gradients[loss_name] = flatten_grads(grads, named_params)
        loss_scalars[loss_name] = float(loss_val.detach().item())

    groups = sorted(set(g for d in gradients.values() for g in d.keys()))
    cosine_rows: List[Dict[str, Any]] = []
    norm_rows: List[Dict[str, Any]] = []
    for loss_name, bucket in gradients.items():
        for group, vec in bucket.items():
            norm_rows.append(
                {
                    "loss": loss_name,
                    "group": group,
                    "grad_norm": float(np.linalg.norm(vec)),
                }
            )
    loss_names = list(gradients.keys())
    for group in groups:
        for a in loss_names:
            for b in loss_names:
                va = gradients[a].get(group, np.zeros((0,), dtype=np.float32))
                vb = gradients[b].get(group, np.zeros((0,), dtype=np.float32))
                cosine_rows.append(
                    {
                        "group": group,
                        "loss_a": a,
                        "loss_b": b,
                        "cosine": cosine(va, vb),
                    }
                )

    return {
        "checkpoint": str(checkpoint_path),
        "experiment": experiment_name_from_checkpoint(checkpoint_path),
        "teacher_profile": teacher.name,
        "deploy_profiles": [p.name for p in deploy_profiles],
        "loss_values": loss_scalars,
        "gradient_norms": norm_rows,
        "gradient_cosines": cosine_rows,
    }


def write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_correlations(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not rows:
        return [], {}
    numeric_keys = []
    sample = rows[0]
    for k, v in sample.items():
        if k in {"experiment", "checkpoint"}:
            continue
        if isinstance(v, (int, float)):
            numeric_keys.append(k)
    target = [float(r["falcor_hdr_psnr"]) for r in rows]
    corr_rows: List[Dict[str, Any]] = []
    for key in numeric_keys:
        if key == "falcor_hdr_psnr":
            continue
        vals = [float(r[key]) for r in rows]
        corr_rows.append(
            {
                "metric": key,
                "pearson_with_falcor": pearson(vals, target),
                "spearman_with_falcor": spearman(vals, target),
                "expected_direction": "negative",
            }
        )
    corr_rows = sorted(
        corr_rows,
        key=lambda r: abs(float(r["spearman_with_falcor"])) if math.isfinite(float(r["spearman_with_falcor"])) else -1.0,
        reverse=True,
    )
    summary = {
        "num_checkpoints": len(rows),
        "best_by_abs_spearman": corr_rows[:10],
        "best_by_abs_pearson": sorted(
            corr_rows,
            key=lambda r: abs(float(r["pearson_with_falcor"])) if math.isfinite(float(r["pearson_with_falcor"])) else -1.0,
            reverse=True,
        )[:10],
    }
    return corr_rows, summary


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher = parse_profile(args.teacher_profile)
    deploy_profiles = [parse_profile(x) for x in args.deploy_profiles]
    checkpoints = discover_checkpoints(args)

    # Use metadata from the gradient checkpoint to define split ratios consistently.
    _, meta_for_split, _ = load_checkpoint_model(Path(args.gradient_checkpoint), device=device)
    train_ratio, val_ratio = infer_split_meta(meta_for_split)

    dataset = build_dataset(
        args.data_root,
        args.split,
        train_ratio,
        val_ratio,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    grad_dataset = build_dataset(
        args.data_root,
        args.grad_split,
        train_ratio,
        val_ratio,
        max_samples=max(args.grad_batch_size, args.grad_batch_size * 2),
        seed=args.seed + 17,
    )

    ref = torch.empty((1, 27), device=device, dtype=torch.float32)
    radiance_basis = build_sh_basis_from_dirs(build_random_dirs(args.radiance_samples, args.seed, ref))
    irradiance_basis = build_sh_basis_from_dirs(build_fibonacci_dirs(args.irradiance_samples, args.seed + 1, ref))

    metric_rows: List[Dict[str, Any]] = []
    for checkpoint in checkpoints:
        row = compute_checkpoint_metrics(
            checkpoint,
            dataset,
            device,
            teacher,
            deploy_profiles,
            batch_size=args.batch_size,
            radiance_basis=radiance_basis,
            irradiance_basis=irradiance_basis,
            epsilon=args.epsilon,
        )
        metric_rows.append(row)

    corr_rows, corr_summary = summarize_correlations(metric_rows)
    gradient_report = compute_gradient_conflicts(
        Path(args.gradient_checkpoint),
        grad_dataset,
        device,
        teacher,
        deploy_profiles,
        batch_size=args.grad_batch_size,
        radiance_basis=radiance_basis,
        irradiance_basis=irradiance_basis,
        epsilon=args.epsilon,
    )

    metric_rows = sorted(metric_rows, key=lambda r: float(r["falcor_hdr_psnr"]), reverse=True)
    write_csv(metric_rows, out_dir / "checkpoint_metric_table.csv")
    write_csv(corr_rows, out_dir / "metric_falcor_correlations.csv")
    write_csv(gradient_report["gradient_norms"], out_dir / "gradient_norms.csv")
    write_csv(gradient_report["gradient_cosines"], out_dir / "gradient_cosines.csv")

    (out_dir / "checkpoint_metric_table.json").write_text(json.dumps(metric_rows, indent=2), encoding="utf-8")
    (out_dir / "metric_falcor_correlations.json").write_text(json.dumps(corr_rows, indent=2), encoding="utf-8")
    (out_dir / "gradient_conflicts.json").write_text(json.dumps(gradient_report, indent=2), encoding="utf-8")

    summary = {
        "device": str(device),
        "num_checkpoints": len(metric_rows),
        "teacher_profile": teacher.name,
        "deploy_profiles": [p.name for p in deploy_profiles],
        "dataset_split": args.split,
        "grad_split": args.grad_split,
        "max_samples": int(args.max_samples),
        "grad_batch_size": int(args.grad_batch_size),
        "top_checkpoint_by_falcor": metric_rows[0] if metric_rows else None,
        "correlation_summary": corr_summary,
        "gradient_checkpoint": str(Path(args.gradient_checkpoint).resolve()),
        "gradient_loss_values": gradient_report["loss_values"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Small offline experiment reports written to: {out_dir}")


if __name__ == "__main__":
    main()
