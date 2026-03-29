#!/usr/bin/env python3
"""Profile model-level and pipeline-level inference performance."""

from __future__ import annotations

import argparse
import json
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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Profile inference (model + realtime pipeline)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--amp-mode", choices=["off", "bf16", "fp16"], default="off")
    p.add_argument("--torch-compile", action="store_true", dest="torch_compile")
    p.add_argument("--no-torch-compile", action="store_false", dest="torch_compile")
    p.set_defaults(torch_compile=False)
    p.add_argument("--torch-compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="reduce-overhead")
    p.add_argument("--torch-compile-dynamic", action="store_true", dest="torch_compile_dynamic")
    p.add_argument("--no-torch-compile-dynamic", action="store_false", dest="torch_compile_dynamic")
    p.set_defaults(torch_compile_dynamic=False)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--profile-level", choices=["model", "pipeline", "both"], default="both")
    p.add_argument("--stack", choices=["torch", "nsys", "both"], default="both")

    p.add_argument("--benchmark-script", default="tools/benchmark_realtime_pipeline.py")
    p.add_argument("--scene", default=None)
    p.add_argument("--pipeline-route", choices=["model", "gt", "both"], default="both")
    p.add_argument("--pipeline-warmup-frames", type=int, default=30)
    p.add_argument("--pipeline-benchmark-frames", type=int, default=300)
    p.add_argument("--pipeline-run", action="store_true", help="Actually execute pipeline benchmark")
    p.add_argument("--falcor-python-path", default=None)
    p.add_argument("--falcor-python-bin", default=None)

    p.add_argument("--run-nsys", action="store_true")
    p.add_argument("--run-ncu", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--output-base", default="3_experiments/results/profiling")
    p.add_argument("--dry-run", action="store_true")
    return p


def _sync_if_cuda(torch_mod: Any, device: Any) -> None:
    if getattr(device, "type", "cpu") == "cuda" and torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize(device)


def _iter_batches(loader) -> Iterable[Dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def _load_model_and_data(args: argparse.Namespace):
    ensure_repo_import_paths()

    import torch
    from data.lightset_dataset import LightSetDataset
    from data.sh_scaler import AdaptiveSHScaler
    from models.gaussian_physics_unified import GaussianPhysicsCompressionUnified

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    meta = ckpt.get("meta") if isinstance(ckpt, dict) else {}

    K = int(state["mu"].shape[0])
    rank = int(state["U"].shape[-1])
    embed_dim = int(state["coeffs"].shape[-1])
    sh_dim = int(state["U"].shape[1])

    train_ratio = float(meta.get("split", {}).get("train_ratio", 0.7)) if isinstance(meta, dict) else 0.7
    val_ratio = float(meta.get("split", {}).get("val_ratio", 0.15)) if isinstance(meta, dict) else 0.15

    model_meta = meta.get("model", {}) if isinstance(meta, dict) else {}
    top_k = int(args.top_k) if args.top_k is not None else int(model_meta.get("top_k", 3))
    light_dim = int(model_meta.get("light_dim", 12))
    intensity_dim = int(model_meta.get("intensity_dim", 3))
    intensity_offset = int(model_meta.get("intensity_offset", 1))
    enable_film = bool(model_meta.get("enable_film", True))
    light_encoder_mode = str(model_meta.get("light_encoder_mode", "normal"))
    bypass_feature_pairs = model_meta.get("bypass_feature_pairs", None)
    bypass_feature_norm_mean = model_meta.get("bypass_feature_norm_mean", None)
    bypass_feature_norm_std = model_meta.get("bypass_feature_norm_std", None)

    model = GaussianPhysicsCompressionUnified(
        num_gaussians=K,
        rank=rank,
        sh_dim=sh_dim,
        light_dim=light_dim,
        embed_dim=embed_dim,
        intensity_dim=intensity_dim,
        intensity_offset=intensity_offset,
        enable_film=enable_film,
        light_encoder_mode=light_encoder_mode,
        bypass_feature_pairs=bypass_feature_pairs,
        bypass_feature_norm_mean=bypass_feature_norm_mean,
        bypass_feature_norm_std=bypass_feature_norm_std,
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state mismatch. missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    compile_requested = bool(getattr(args, "torch_compile", False))
    compile_applied = False
    if compile_requested:
        try:
            model = torch.compile(
                model,
                mode=str(getattr(args, "torch_compile_mode", "reduce-overhead")),
                dynamic=bool(getattr(args, "torch_compile_dynamic", False)),
            )
            model.eval()
            compile_applied = True
        except Exception:
            compile_applied = False

    scaler = None
    scaler_meta = meta.get("sh_scaler") if isinstance(meta, dict) else None
    if isinstance(scaler_meta, dict):
        import numpy as np

        scaler = AdaptiveSHScaler(
            l0_mean=np.array(scaler_meta.get("l0_mean", [0.0, 0.0, 0.0]), dtype=np.float32),
            l0_std=np.array(scaler_meta.get("l0_std", [1.0, 1.0, 1.0]), dtype=np.float32),
            ho_rms=np.array(scaler_meta.get("ho_rms", [1.0, 1.0, 1.0]), dtype=np.float32),
            eps=float(scaler_meta.get("eps", 1e-6)),
        )

    dataset = LightSetDataset(
        data_root=args.data_root,
        split=args.split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize_probes=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(max(1, args.batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return {
        "torch": torch,
        "device": device,
        "model": model,
        "loader": loader,
        "top_k": top_k,
        "scaler": scaler,
        "compile_requested": bool(compile_requested),
        "compile_applied": bool(compile_applied),
    }


def _profile_model_level(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    ctx = _load_model_and_data(args)
    torch = ctx["torch"]
    device = ctx["device"]
    model = ctx["model"]
    loader = ctx["loader"]
    top_k = ctx["top_k"]
    scaler = ctx.get("scaler")
    compile_requested = bool(ctx.get("compile_requested", False))
    compile_applied = bool(ctx.get("compile_applied", False))

    amp_mode = str(getattr(args, "amp_mode", "off")).strip().lower()
    if device.type != "cuda":
        amp_mode = "off"
    use_amp = amp_mode in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else (torch.float16 if amp_mode == "fp16" else None)

    def _autocast_ctx():
        if use_amp and amp_dtype is not None:
            return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)
        return nullcontext()

    total_frames = int(max(1, args.frames))
    warmup = int(max(0, args.warmup_frames))
    loops = total_frames + warmup

    use_torch_stack = args.stack in ("torch", "both")
    prof_ctx = nullcontext()
    prof = None
    if use_torch_stack:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda" and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        prof_ctx = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=warmup, active=total_frames, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )

    rows: list[Dict[str, Any]] = []
    batch_iter = _iter_batches(loader)

    with prof_ctx as prof_obj:
        prof = prof_obj if use_torch_stack else None
        with torch.inference_mode():
            for i in range(loops):
                t_fetch0 = time.perf_counter()
                batch = next(batch_iter)
                t_fetch1 = time.perf_counter()

                positions = batch["probe_position"].to(device, non_blocking=True)
                params = batch["light_params"].to(device, non_blocking=True)
                mask = batch.get("light_mask")
                if mask is not None:
                    mask = mask.to(device, non_blocking=True)

                t_step0 = time.perf_counter()
                t_route0 = time.perf_counter()
                with _autocast_ctx():
                    if hasattr(model, "compute_gaussian_routing"):
                        routing = model.compute_gaussian_routing(positions, top_k=top_k)
                    else:
                        routing = None
                _sync_if_cuda(torch, device)
                t_route1 = time.perf_counter()

                t_fwd0 = time.perf_counter()
                with _autocast_ctx():
                    if routing is not None and hasattr(model, "forward_with_routing"):
                        preds = model.forward_with_routing(routing, params, light_mask=mask)
                    else:
                        preds = model(positions, params, top_k=top_k, light_mask=mask)
                if scaler is not None:
                    preds = scaler.inverse(preds)
                _sync_if_cuda(torch, device)
                t_fwd1 = time.perf_counter()
                t_step1 = time.perf_counter()

                if prof is not None:
                    prof.step()

                if i >= warmup:
                    batch_n = int(preds.shape[0])
                    row = {
                        "frame": int(i - warmup + 1),
                        "batch_size": batch_n,
                        "dataloader_wait_ms": float((t_fetch1 - t_fetch0) * 1000.0),
                        "routing_ms": float((t_route1 - t_route0) * 1000.0),
                        "forward_ms": float((t_fwd1 - t_fwd0) * 1000.0),
                        "step_ms": float((t_step1 - t_step0) * 1000.0),
                        "per_sample_ms": float(((t_step1 - t_step0) * 1000.0) / max(1, batch_n)),
                    }
                    rows.append(row)

    export_csv(out_dir / "inference_model_step_breakdown.csv", rows)

    summary = {
        "frames": int(total_frames),
        "warmup_frames": int(warmup),
        "device": str(device),
        "amp_mode": amp_mode,
        "torch_compile_requested": bool(compile_requested),
        "torch_compile_applied": bool(compile_applied),
        "latency": {
            "step": summarize_latency([r["step_ms"] for r in rows]),
            "forward": summarize_latency([r["forward_ms"] for r in rows]),
            "routing": summarize_latency([r["routing_ms"] for r in rows]),
            "dataloader_wait": summarize_latency([r["dataloader_wait_ms"] for r in rows]),
            "per_sample": summarize_latency([r["per_sample_ms"] for r in rows]),
        },
    }

    op_rows: list[Dict[str, Any]] = []
    if prof is not None:
        trace_path = out_dir / "inference_model_torch_trace.json"
        prof.export_chrome_trace(str(trace_path))
        key_avg = prof.key_averages()
        op_rows = topk_ops_by_cuda_time(key_avg, limit=50)
        export_csv(out_dir / "inference_model_torch_ops.csv", op_rows)
        (out_dir / "inference_model_torch_key_averages.txt").write_text(
            key_avg.table(sort_by="self_cuda_time_total", row_limit=50),
            encoding="utf-8",
        )

    summary["top3_ops_pct"] = top3_concentration_pct(op_rows)
    export_json(out_dir / "inference_model_summary.json", summary)
    return {"summary": summary, "op_rows": op_rows}


def _build_pipeline_cmd(args: argparse.Namespace, out_dir: Path) -> tuple[list[str] | None, str | None, Path]:
    bench_script = Path(args.benchmark_script)
    if not bench_script.is_absolute():
        bench_script = Path(__file__).resolve().parents[1] / bench_script

    pipeline_out = out_dir / "pipeline_benchmark"
    if args.scene is None:
        return None, "missing --scene for pipeline-level profiling", pipeline_out

    cmd = [
        sys.executable,
        str(bench_script),
        "--dataset",
        str(args.data_root),
        "--scene",
        str(args.scene),
        "--output-dir",
        str(pipeline_out),
        "--route",
        str(args.pipeline_route),
        "--warmup-frames",
        str(args.pipeline_warmup_frames),
        "--benchmark-frames",
        str(args.pipeline_benchmark_frames),
    ]
    if args.pipeline_route in ("model", "both"):
        cmd.extend(["--checkpoint", str(args.checkpoint)])
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.device is not None:
        cmd.extend(["--device", str(args.device)])
    if args.falcor_python_path:
        cmd.extend(["--falcor-python-path", str(args.falcor_python_path)])
    if args.falcor_python_bin:
        cmd.extend(["--falcor-python-bin", str(args.falcor_python_bin)])

    return cmd, None, pipeline_out


def _pipeline_model_share_pct(summary: Dict[str, Any]) -> float:
    if not isinstance(summary, dict):
        return 0.0
    model = summary.get("results", {}).get("model", {})
    infer_ms = float(model.get("inference_ms", {}).get("mean", 0.0) or 0.0)
    full_ms = float(model.get("with_gbuffer_full_ms", {}).get("mean", 0.0) or 0.0)
    if full_ms <= 0:
        return 0.0
    return float(100.0 * infer_ms / full_ms)


def run_profile(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir) if args.output_dir else build_run_dir(args.output_base, "inference_profile")
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "run_dir": str(out_dir),
        "stack": str(args.stack),
        "profile_level": str(args.profile_level),
        "dry_run": bool(args.dry_run),
    }

    model_summary: Dict[str, Any] | None = None
    pipeline_summary: Dict[str, Any] | None = None

    if args.profile_level in ("model", "both"):
        if args.dry_run:
            model_summary = {"status": "dry_run", "frames": int(args.frames), "top3_ops_pct": 0.0}
            export_json(out_dir / "inference_model_summary.json", model_summary)
        else:
            model_result = _profile_model_level(args, out_dir)
            model_summary = model_result["summary"]

    if args.profile_level in ("pipeline", "both"):
        cmd, reason, pipeline_out = _build_pipeline_cmd(args, out_dir)
        if cmd is None:
            pipeline_summary = {"status": "not_run", "reason": reason}
            export_json(out_dir / "inference_pipeline_summary.json", pipeline_summary)
        else:
            (out_dir / "pipeline_command.sh").write_text(" ".join(cmd) + "\n", encoding="utf-8")
            if args.dry_run or not args.pipeline_run:
                pipeline_summary = {
                    "status": "not_run",
                    "reason": "dry_run or --pipeline-run not enabled",
                    "command": cmd,
                }
                export_json(out_dir / "inference_pipeline_summary.json", pipeline_summary)
            else:
                subprocess.run(cmd, check=True)
                summary_path = pipeline_out / "benchmark_summary.json"
                if summary_path.exists():
                    pipeline_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                else:
                    pipeline_summary = {"status": "missing_summary", "path": str(summary_path)}
                export_json(out_dir / "inference_pipeline_summary.json", pipeline_summary)

    torch_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint",
        str(args.checkpoint),
        "--data-root",
        str(args.data_root),
        "--profile-level",
        "model",
        "--stack",
        "torch",
        "--frames",
        str(args.frames),
        "--warmup-frames",
        str(args.warmup_frames),
        "--output-dir",
        str(out_dir / "torch_stack_run"),
    ]
    if args.top_k is not None:
        torch_cmd.extend(["--top-k", str(args.top_k)])
    if args.device is not None:
        torch_cmd.extend(["--device", str(args.device)])
    if args.amp_mode is not None:
        torch_cmd.extend(["--amp-mode", str(args.amp_mode)])
    if args.torch_compile:
        torch_cmd.append("--torch-compile")
    else:
        torch_cmd.append("--no-torch-compile")
    if args.torch_compile_mode is not None:
        torch_cmd.extend(["--torch-compile-mode", str(args.torch_compile_mode)])
    if args.torch_compile_dynamic:
        torch_cmd.append("--torch-compile-dynamic")
    else:
        torch_cmd.append("--no-torch-compile-dynamic")

    nsys_cmd = build_nsys_command(torch_cmd, out_dir / "nsys_inference")
    ncu_cmd = build_ncu_command(torch_cmd, out_dir / "ncu_inference")
    (out_dir / "nsys_command.sh").write_text(" ".join(nsys_cmd) + "\n", encoding="utf-8")
    (out_dir / "ncu_command.sh").write_text(" ".join(ncu_cmd) + "\n", encoding="utf-8")

    if args.stack in ("nsys", "both") and args.run_nsys and command_exists("nsys"):
        subprocess.run(nsys_cmd, check=True)
        report["nsys_ran"] = True
    else:
        report["nsys_ran"] = False

    if args.run_ncu and command_exists("ncu"):
        subprocess.run(ncu_cmd, check=True)
        report["ncu_ran"] = True
    else:
        report["ncu_ran"] = False

    infer_model_bundle = {}
    if isinstance(model_summary, dict):
        infer_model_bundle = {
            "top3_ops_pct": float(model_summary.get("top3_ops_pct", 0.0) or 0.0),
            "p95_ms": float(model_summary.get("latency", {}).get("per_sample", {}).get("p95", 0.0) or 0.0),
        }
    pipeline_bundle = {}
    if isinstance(pipeline_summary, dict):
        pipeline_bundle = {
            "model_share_pct": _pipeline_model_share_pct(pipeline_summary),
        }

    cuda_rec = recommend_cuda(
        {
            "inference_model": infer_model_bundle,
            "pipeline": pipeline_bundle,
        }
    )
    export_json(out_dir / "cuda_recommendation_inference.json", cuda_rec)

    report["cuda_recommendation"] = cuda_rec
    report["model_summary_path"] = str(out_dir / "inference_model_summary.json")
    report["pipeline_summary_path"] = str(out_dir / "inference_pipeline_summary.json")
    export_json(out_dir / "inference_profile_report.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    report = run_profile(args)
    print(f"[inference-profiler] run_dir={report['run_dir']}")
    print(f"[inference-profiler] nsys_ran={report.get('nsys_ran', False)} ncu_ran={report.get('ncu_ran', False)}")


if __name__ == "__main__":
    main()
