#!/usr/bin/env python3
"""Audit global expert utilization under one or more routing profiles.

This script answers the "effective experts" question with global counters:
- per-gaussian sample reference counts
- per-gaussian unique probe/config coverage
- per-gaussian soft routing mass

It supports both hard and soft routing profiles so train/infer route mismatch
does not silently bias utilization conclusions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "2_src"
if str(_SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SRC))

from data.lightset_dataset import LightSetDataset
from training.gaussian_physics_trainer import BatchAdapter


def _load_diag_module():
    diag_path = _ROOT / "3_experiments" / "scripts" / "analysis" / "run_sh_oracle_diagnostics.py"
    spec = importlib.util.spec_from_file_location("run_sh_oracle_diagnostics", str(diag_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load diagnostics module from {diag_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _device_from_arg(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sanitize_name(name: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in str(name).strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "profile"


def _distribution_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(np.sum(arr))
    if total <= 0.0:
        return {
            "sum": 0.0,
            "entropy": 0.0,
            "entropy_norm": 0.0,
            "effective_k": 0.0,
            "hhi": 0.0,
            "gini": 0.0,
        }
    p = arr / total
    p = np.clip(p, 0.0, 1.0)
    nz = p[p > 0.0]
    entropy = float(-np.sum(nz * np.log(nz)))
    n = int(arr.size)
    entropy_norm = float(entropy / np.log(float(max(2, n))))
    effective_k = float(np.exp(entropy))
    hhi = float(np.sum(p * p))

    xs = np.sort(arr)
    n_f = float(max(1, xs.size))
    gini_num = float(np.sum((2.0 * np.arange(1, xs.size + 1) - n_f - 1.0) * xs))
    gini_den = n_f * float(np.sum(xs)) + 1e-12
    gini = float(gini_num / gini_den)
    return {
        "sum": total,
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "effective_k": effective_k,
        "hhi": hhi,
        "gini": gini,
    }


def _coverage_stats(values: np.ndarray, denom: float) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr / max(1e-12, denom))),
        "median": float(np.median(arr / max(1e-12, denom))),
        "p90": float(np.quantile(arr / max(1e-12, denom), 0.90)),
        "max": float(np.max(arr / max(1e-12, denom))),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run global expert utilization audit")
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=None)
    p.add_argument("--val-ratio", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=8192)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument(
        "--profiles-json",
        default="",
        help="JSON string/list for route profiles. If empty, use one hard profile from --top-k.",
    )
    return p


def _normalize_profiles(raw: Any, default_top_k: int) -> List[Dict[str, Any]]:
    if not raw:
        return [
            {
                "name": f"hard{int(default_top_k)}",
                "top_k": int(default_top_k),
                "training_soft_routing": False,
                "routing_soft_topk": None,
                "routing_temperature": None,
            }
        ]

    if isinstance(raw, dict):
        raw_profiles = raw.get("profiles", [])
    elif isinstance(raw, list):
        raw_profiles = raw
    else:
        raise ValueError("profiles-json must be a JSON list or a mapping with key 'profiles'")

    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_profiles):
        if not isinstance(item, dict):
            raise ValueError(f"profiles[{idx}] must be mapping")
        name = str(item.get("name", f"profile_{idx+1}")).strip()
        cfg = {
            "name": name or f"profile_{idx+1}",
            "top_k": int(item.get("top_k", default_top_k)),
            "training_soft_routing": bool(item.get("training_soft_routing", False)),
            "routing_soft_topk": item.get("routing_soft_topk", None),
            "routing_temperature": item.get("routing_temperature", None),
        }
        if cfg["routing_soft_topk"] is not None:
            cfg["routing_soft_topk"] = int(cfg["routing_soft_topk"])
        if cfg["routing_temperature"] is not None:
            cfg["routing_temperature"] = float(cfg["routing_temperature"])
        out.append(cfg)

    if not out:
        return _normalize_profiles(None, default_top_k)
    return out


def _maybe_sample_dataset(dataset, max_samples: int, sample_seed: int):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset, None
    rng = np.random.default_rng(int(sample_seed))
    indices = np.sort(rng.choice(len(dataset), size=int(max_samples), replace=False).astype(np.int64))
    return Subset(dataset, indices.tolist()), indices


def _to_numpy_int(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.int64, copy=False)


def _to_numpy_float(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float64, copy=False)


def _summarize_profile(
    *,
    sample_counts: np.ndarray,
    top1_counts: np.ndarray,
    mass_sums: np.ndarray,
    unique_probe_counts: np.ndarray,
    unique_config_counts: np.ndarray,
    num_samples: int,
    num_unique_probes: int,
    num_unique_configs: int,
) -> Dict[str, Any]:
    K = int(sample_counts.size)
    nonzero_sample_ratio = float(np.mean(sample_counts > 0))
    nonzero_mass_ratio = float(np.mean(mass_sums > 0.0))
    nonzero_probe_ratio = float(np.mean(unique_probe_counts > 0))
    nonzero_config_ratio = float(np.mean(unique_config_counts > 0))

    sample_dist = _distribution_stats(sample_counts.astype(np.float64))
    mass_dist = _distribution_stats(mass_sums.astype(np.float64))
    top1_dist = _distribution_stats(top1_counts.astype(np.float64))

    per_gaussian = []
    for g in range(K):
        per_gaussian.append(
            {
                "gaussian": int(g),
                "sample_count": int(sample_counts[g]),
                "top1_count": int(top1_counts[g]),
                "mass_sum": float(mass_sums[g]),
                "mean_mass_if_referenced": float(mass_sums[g] / max(1, sample_counts[g])),
                "unique_probe_count": int(unique_probe_counts[g]),
                "unique_probe_ratio": float(unique_probe_counts[g] / max(1, num_unique_probes)),
                "unique_config_count": int(unique_config_counts[g]),
                "unique_config_ratio": float(unique_config_counts[g] / max(1, num_unique_configs)),
                "sample_ratio": float(sample_counts[g] / max(1, num_samples)),
            }
        )

    return {
        "num_gaussians": int(K),
        "nonzero_ratio": {
            "sample_count": nonzero_sample_ratio,
            "mass_sum": nonzero_mass_ratio,
            "probe_coverage": nonzero_probe_ratio,
            "config_coverage": nonzero_config_ratio,
        },
        "distribution": {
            "sample_count": sample_dist,
            "mass_sum": mass_dist,
            "top1_count": top1_dist,
        },
        "coverage": {
            "probe_ratio": _coverage_stats(unique_probe_counts, float(max(1, num_unique_probes))),
            "config_ratio": _coverage_stats(unique_config_counts, float(max(1, num_unique_configs))),
        },
        "top_gaussians_by_mass": sorted(
            (
                {
                    "gaussian": int(g),
                    "mass_sum": float(mass_sums[g]),
                    "sample_count": int(sample_counts[g]),
                    "top1_count": int(top1_counts[g]),
                }
                for g in range(K)
            ),
            key=lambda x: x["mass_sum"],
            reverse=True,
        )[:10],
        "per_gaussian": per_gaussian,
    }


def main() -> None:
    args = _build_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = output_path.parent

    diag_mod = _load_diag_module()
    device = _device_from_arg(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    meta = ckpt.get("meta") if isinstance(ckpt, dict) else {}
    if meta is None:
        meta = {}

    model, model_info = diag_mod._build_model_from_checkpoint(state, meta, device)
    default_top_k = int(args.top_k) if args.top_k is not None else int(model_info["top_k"])
    train_ratio, val_ratio = diag_mod._resolve_split_ratios(args, meta)

    profiles = _normalize_profiles(
        json.loads(args.profiles_json) if str(args.profiles_json).strip() else None,
        default_top_k=default_top_k,
    )

    dataset = LightSetDataset(
        data_root=args.data_root,
        split=args.split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=args.seed,
        normalize_probes=True,
    )
    sampled_dataset, sampled_indices = _maybe_sample_dataset(
        dataset,
        max_samples=int(args.max_samples),
        sample_seed=int(args.sample_seed),
    )

    loader = DataLoader(
        sampled_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )

    adapter = BatchAdapter(params_key="light_params", mask_key="light_mask")
    K = int(model_info["num_gaussians"])
    profile_results: Dict[str, Any] = {}

    for profile in profiles:
        name = str(profile["name"])
        safe = _sanitize_name(name)
        sample_counts = np.zeros((K,), dtype=np.int64)
        top1_counts = np.zeros((K,), dtype=np.int64)
        mass_sums = np.zeros((K,), dtype=np.float64)
        probe_sets = [set() for _ in range(K)]
        config_sets = [set() for _ in range(K)]
        all_probe_ids: set[int] = set()
        all_config_ids: set[int] = set()

        with torch.no_grad():
            for batch in loader:
                unpacked = adapter.unpack(batch, device)
                if len(unpacked) == 4:
                    positions, params, _, mask = unpacked
                else:
                    positions, params, _ = unpacked
                    mask = None

                weights, indices = model.compute_gaussian_weights(
                    positions,
                    top_k=int(profile.get("top_k", default_top_k)),
                    training_soft_routing=bool(profile.get("training_soft_routing", False)),
                    routing_soft_topk=profile.get("routing_soft_topk", None),
                    routing_temperature=float(profile.get("routing_temperature", 1.0))
                    if profile.get("routing_temperature", None) is not None
                    else 1.0,
                )

                idx_np = _to_numpy_int(indices)
                w_np = _to_numpy_float(weights)
                probe_np = _to_numpy_int(batch["probe_idx"])
                cfg_np = _to_numpy_int(batch["config_idx"])
                all_probe_ids.update(int(x) for x in probe_np.tolist())
                all_config_ids.update(int(x) for x in cfg_np.tolist())
                top1_slot = np.argmax(w_np, axis=1)

                for b in range(idx_np.shape[0]):
                    p = int(probe_np[b])
                    c = int(cfg_np[b])
                    g_top1 = int(idx_np[b, int(top1_slot[b])])
                    top1_counts[g_top1] += 1
                    for j in range(idx_np.shape[1]):
                        g = int(idx_np[b, j])
                        w = float(w_np[b, j])
                        sample_counts[g] += 1
                        mass_sums[g] += w
                        probe_sets[g].add(p)
                        config_sets[g].add(c)

        unique_probe_counts = np.array([len(s) for s in probe_sets], dtype=np.int64)
        unique_config_counts = np.array([len(s) for s in config_sets], dtype=np.int64)
        profile_summary = _summarize_profile(
            sample_counts=sample_counts,
            top1_counts=top1_counts,
            mass_sums=mass_sums,
            unique_probe_counts=unique_probe_counts,
            unique_config_counts=unique_config_counts,
            num_samples=int(len(sampled_dataset)),
            num_unique_probes=int(max(1, len(all_probe_ids))),
            num_unique_configs=int(max(1, len(all_config_ids))),
        )

        csv_path = out_dir / f"{safe}_per_gaussian.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write(
                "gaussian,sample_count,top1_count,mass_sum,mean_mass_if_referenced,"
                "unique_probe_count,unique_probe_ratio,unique_config_count,unique_config_ratio,sample_ratio\n"
            )
            for row in profile_summary["per_gaussian"]:
                f.write(
                    f"{row['gaussian']},{row['sample_count']},{row['top1_count']},"
                    f"{row['mass_sum']:.10f},{row['mean_mass_if_referenced']:.10f},"
                    f"{row['unique_probe_count']},{row['unique_probe_ratio']:.10f},"
                    f"{row['unique_config_count']},{row['unique_config_ratio']:.10f},"
                    f"{row['sample_ratio']:.10f}\n"
                )

        profile_summary["per_gaussian_csv"] = str(csv_path)
        profile_results[name] = {
            "profile": profile,
            "summary": profile_summary,
        }

    summary = {
        "meta": {
            "checkpoint": str(args.checkpoint),
            "data_root": str(args.data_root),
            "split": str(args.split),
            "device": str(device),
            "seed": int(args.seed),
            "model": model_info,
        },
        "counts": {
            "dataset_size_before_sampling": int(len(dataset)),
            "dataset_size_after_sampling": int(len(sampled_dataset)),
            "sampled": bool(sampled_indices is not None),
            "num_profiles": int(len(profiles)),
        },
        "profiles": profile_results,
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved expert utilization summary: {output_path}")


if __name__ == "__main__":
    main()
