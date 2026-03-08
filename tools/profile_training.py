#!/usr/bin/env python3
"""Profile PG-GCPL training loop with torch profiler and optional Nsight wrappers."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable

if __package__ is None or __package__ == "":
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from tools.profiler_common import (
    build_ncu_command,
    build_nsys_command,
    build_run_dir,
    command_exists,
    ensure_repo_import_paths,
    export_csv,
    export_json,
    recommend_cuda,
    summarize_latency,
    top3_concentration_pct,
    topk_ops_by_cuda_time,
)


def _load_train_module(root: Path):
    train_path = root / "3_experiments" / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script_for_profiler", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load training module from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Profile training loop (torch profiler + optional Nsight)")
    p.add_argument("--config", required=True, help="Training YAML config path")
    p.add_argument("--variant", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--checkpoint", default=None, help="Optional checkpoint to load")
    p.add_argument("--device", default=None)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stack", choices=["torch", "nsys", "both"], default="both")
    p.add_argument("--enable-extra-losses", action="store_true")
    p.add_argument("--run-nsys", action="store_true", help="Actually run nsys command when stack includes nsys")
    p.add_argument("--run-ncu", action="store_true", help="Generate and run one-shot ncu command")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--output-base", default="3_experiments/results/profiling")
    p.add_argument("--dry-run", action="store_true")
    return p


def _prepare_train_args(args: argparse.Namespace, train_module: Any):
    parser = train_module.build_arg_parser()
    defaults = parser.parse_args([])
    run_args = parser.parse_args([])

    cfg = train_module._load_yaml(args.config)
    run_args._config_obj = cfg
    if "train" in cfg and isinstance(cfg["train"], dict):
        cfg = train_module.merge_configs(cfg, cfg["train"])
    train_module._apply_config(run_args, defaults, cfg)

    if args.variant is not None:
        run_args.variant = args.variant
    if args.data_root is not None:
        run_args.data_root = args.data_root
    if args.device is not None:
        run_args.device = args.device
    if args.batch_size is not None:
        run_args.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        run_args.num_workers = int(args.num_workers)
    if args.checkpoint is not None:
        run_args.load_model = args.checkpoint

    run_args.seed = int(args.seed)
    run_args.steps = int(args.steps)
    run_args.warmup_steps = int(args.warmup_steps)
    run_args.stack = str(args.stack)
    run_args.no_init = True
    run_args.show_progress = False
    run_args.enable_rerun = False
    run_args.no_auto_log = True
    run_args.enable_sh_scaler = False

    if not args.enable_extra_losses:
        run_args.lambda_linearity = 0.0
        run_args.lambda_spatial = 0.0
        run_args.lambda_temporal = 0.0
        run_args.lambda_image = 0.0

    if run_args.variant is None:
        raise ValueError("variant must be set via config or --variant")

    run_args.data_root = train_module.resolve_data_root(run_args.data_root, run_args.manifest)
    train_module.apply_variant_defaults(run_args)
    run_args = train_module.normalize_train_args(run_args)
    return run_args


def _iter_batches(loader) -> Iterable[Dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def _sync_if_cuda(device: Any) -> None:
    try:
        import torch

        if getattr(device, "type", "cpu") == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception:
        pass


def _profile_training_steps(train_module: Any, run_args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    train_module._lazy_imports()
    torch = train_module.torch

    if run_args.seed is not None:
        torch.manual_seed(int(run_args.seed))

    device = train_module._device_from_arg(run_args.device)
    model, adapter, loaders, helpers = train_module.build_variant(run_args.variant, run_args, device)
    train_loader = loaders[0]

    if not run_args.no_init:
        helpers["init_fn"](model, train_loader)
        model.to(device)

    trainer = train_module.GaussianPhysicsTrainer(
        model=model,
        device=device,
        adapter=adapter,
        lr=run_args.lr,
        weight_decay=run_args.weight_decay,
        lr_scheduler=run_args.lr_scheduler,
        lr_min=run_args.lr_min,
        warmup_epochs=int(getattr(run_args, "warmup_epochs", 0)),
        recon_loss=run_args.recon_loss,
        charbonnier_eps=run_args.charbonnier_eps,
        temporal_weight=run_args.lambda_temporal,
        temporal_loss_fn=helpers.get("temporal_loss_fn"),
        top_k=run_args.top_k,
        grad_clip=run_args.grad_clip,
        linearity_weight=run_args.lambda_linearity,
        linearity_aug_pairs=run_args.linearity_aug_pairs,
        spatial_weight=run_args.lambda_spatial,
        spatial_k=run_args.spatial_k,
        image_loss_weight=run_args.lambda_image,
        image_loss_type=run_args.image_loss_type,
        image_samples=run_args.image_samples,
        image_sample_seed=run_args.image_sample_seed,
        image_loss_space=run_args.image_loss_space,
        enable_weighted_sh_loss=run_args.enable_weighted_sh_loss,
        sh_loss_weights=run_args.sh_loss_weights,
        sh_weight_mode=run_args.sh_weight_mode,
        compute_img_metrics_in_val=False,
        compute_superposition_in_val=False,
        show_progress=False,
    )
    trainer._init_spatial_neighbors(train_loader)

    total_steps = int(max(1, run_args.steps))
    warmup_steps = int(max(0, run_args.warmup_steps))
    loops = total_steps + warmup_steps
    batch_iter = _iter_batches(train_loader)

    use_torch_stack = run_args.stack in ("torch", "both")
    prof_ctx = nullcontext()
    prof = None
    if use_torch_stack:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda" and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        prof_ctx = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=warmup_steps, active=total_steps, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )

    rows: list[Dict[str, Any]] = []
    with prof_ctx as prof_obj:
        prof = prof_obj if use_torch_stack else None
        for i in range(loops):
            t_fetch0 = time.perf_counter()
            batch = next(batch_iter)
            t_fetch1 = time.perf_counter()

            unpacked = trainer.adapter.unpack(batch, device)
            if len(unpacked) == 4:
                positions, params, targets, mask = unpacked
            else:
                positions, params, targets = unpacked
                mask = None

            t_step0 = time.perf_counter()
            trainer.optimizer.zero_grad(set_to_none=True)

            t_fwd0 = time.perf_counter()
            routing = trainer._compute_routing(positions)
            preds = trainer._forward(positions, params, mask, routing=routing)
            _sync_if_cuda(device)
            t_fwd1 = time.perf_counter()

            t_loss0 = time.perf_counter()
            loss_recon = trainer._recon_loss(preds, targets)
            preds_eval, targets_eval = preds, targets
            if trainer.adapter.target_inverse is not None:
                preds_eval, targets_eval = trainer.adapter.inverse_targets(preds, targets)

            loss_image = trainer._compute_image_loss(preds_eval, targets_eval)
            loss_temporal = trainer._compute_temporal_loss()
            loss_linearity = trainer._compute_linearity_loss(positions, params, mask, preds, routing_full=routing)
            loss_linearity_aug = trainer._compute_linearity_aug_loss(positions, params, mask, routing_full=routing)
            probe_idx = batch.get("probe_idx") if isinstance(batch, dict) else None
            loss_spatial = trainer._compute_spatial_loss(preds, params, mask, probe_idx)

            loss_total = (
                loss_recon
                + trainer.image_loss_weight * loss_image
                + trainer.temporal_weight * loss_temporal
                + trainer.linearity_weight * (loss_linearity + loss_linearity_aug)
                + trainer.spatial_weight * loss_spatial
            )
            _sync_if_cuda(device)
            t_loss1 = time.perf_counter()

            t_bwd0 = time.perf_counter()
            loss_total.backward()
            if trainer.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.grad_clip)
            _sync_if_cuda(device)
            t_bwd1 = time.perf_counter()

            t_opt0 = time.perf_counter()
            trainer.optimizer.step()
            trainer._update_ema()
            _sync_if_cuda(device)
            t_opt1 = time.perf_counter()

            t_step1 = time.perf_counter()

            if prof is not None:
                prof.step()

            if i >= warmup_steps:
                row = {
                    "step": int(i - warmup_steps + 1),
                    "dataloader_wait_ms": float((t_fetch1 - t_fetch0) * 1000.0),
                    "forward_ms": float((t_fwd1 - t_fwd0) * 1000.0),
                    "loss_ms": float((t_loss1 - t_loss0) * 1000.0),
                    "backward_ms": float((t_bwd1 - t_bwd0) * 1000.0),
                    "optimizer_ms": float((t_opt1 - t_opt0) * 1000.0),
                    "step_ms": float((t_step1 - t_step0) * 1000.0),
                    "loss_total": float(loss_total.detach().item()),
                }
                accounted = row["forward_ms"] + row["loss_ms"] + row["backward_ms"] + row["optimizer_ms"]
                row["python_overhead_ms"] = max(0.0, row["step_ms"] - accounted)
                rows.append(row)

    export_csv(out_dir / "training_step_breakdown.csv", rows)

    summary = {
        "steps": int(total_steps),
        "warmup_steps": int(warmup_steps),
        "device": str(device),
        "latency": {
            "dataloader_wait": summarize_latency([r["dataloader_wait_ms"] for r in rows]),
            "step": summarize_latency([r["step_ms"] for r in rows]),
            "forward": summarize_latency([r["forward_ms"] for r in rows]),
            "loss": summarize_latency([r["loss_ms"] for r in rows]),
            "backward": summarize_latency([r["backward_ms"] for r in rows]),
            "optimizer": summarize_latency([r["optimizer_ms"] for r in rows]),
            "python_overhead": summarize_latency([r["python_overhead_ms"] for r in rows]),
        },
    }

    step_total = sum(r["step_ms"] for r in rows)
    fetch_total = sum(r["dataloader_wait_ms"] for r in rows)
    overhead_total = sum(r["python_overhead_ms"] for r in rows)
    summary["percentages"] = {
        "dataloader_wait_pct": float(100.0 * fetch_total / max(1e-6, fetch_total + step_total)),
        "python_overhead_pct": float(100.0 * overhead_total / max(1e-6, step_total)),
    }

    op_rows: list[Dict[str, Any]] = []
    if prof is not None:
        trace_path = out_dir / "training_torch_trace.json"
        prof.export_chrome_trace(str(trace_path))
        key_avg = prof.key_averages()
        op_rows = topk_ops_by_cuda_time(key_avg, limit=50)
        export_csv(out_dir / "training_torch_ops.csv", op_rows)
        table = key_avg.table(sort_by="self_cuda_time_total", row_limit=50)
        (out_dir / "training_torch_key_averages.txt").write_text(table, encoding="utf-8")

    summary["top3_ops_pct"] = top3_concentration_pct(op_rows)
    export_json(out_dir / "training_profile_summary.json", summary)

    cuda_rec = recommend_cuda({"training": {**summary.get("percentages", {}), "top3_ops_pct": summary["top3_ops_pct"]}})
    export_json(out_dir / "cuda_recommendation_training.json", cuda_rec)

    return {
        "summary": summary,
        "op_rows": op_rows,
        "cuda_recommendation": cuda_rec,
    }


def _build_self_torch_cmd(args: argparse.Namespace, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--stack",
        "torch",
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(out_dir / "torch_stack_run"),
    ]
    if args.variant is not None:
        cmd.extend(["--variant", str(args.variant)])
    if args.data_root is not None:
        cmd.extend(["--data-root", str(args.data_root)])
    if args.device is not None:
        cmd.extend(["--device", str(args.device)])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.num_workers is not None:
        cmd.extend(["--num-workers", str(args.num_workers)])
    if args.checkpoint is not None:
        cmd.extend(["--checkpoint", str(args.checkpoint)])
    if args.enable_extra_losses:
        cmd.append("--enable-extra-losses")
    return cmd


def run_profile(args: argparse.Namespace) -> Dict[str, Any]:
    root, _, _ = ensure_repo_import_paths()
    out_dir = Path(args.output_dir) if args.output_dir else build_run_dir(args.output_base, "training_profile")
    out_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "run_dir": str(out_dir),
        "stack": str(args.stack),
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        export_json(out_dir / "training_profile_summary.json", {"status": "dry_run", **result})
    else:
        train_module = _load_train_module(root)
        run_args = _prepare_train_args(args, train_module)
        result.update(_profile_training_steps(train_module, run_args, out_dir))

    torch_cmd = _build_self_torch_cmd(args, out_dir)
    nsys_cmd = build_nsys_command(torch_cmd, out_dir / "nsys_training")
    ncu_cmd = build_ncu_command(torch_cmd, out_dir / "ncu_training")
    (out_dir / "nsys_command.sh").write_text(" ".join(nsys_cmd) + "\n", encoding="utf-8")
    (out_dir / "ncu_command.sh").write_text(" ".join(ncu_cmd) + "\n", encoding="utf-8")

    if args.stack in ("nsys", "both") and args.run_nsys and command_exists("nsys"):
        subprocess.run(nsys_cmd, check=True)
        result["nsys_ran"] = True
    else:
        result["nsys_ran"] = False

    if args.run_ncu and command_exists("ncu"):
        subprocess.run(ncu_cmd, check=True)
        result["ncu_ran"] = True
    else:
        result["ncu_ran"] = False

    export_json(out_dir / "training_profile_report.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    report = run_profile(args)
    print(f"[training-profiler] run_dir={report['run_dir']}")
    print(f"[training-profiler] nsys_ran={report.get('nsys_ran', False)} ncu_ran={report.get('ncu_ran', False)}")


if __name__ == "__main__":
    main()

