#!/usr/bin/env python3
"""Falcor-side worker for realtime benchmark (Python 3.10 environment)."""

from __future__ import annotations

import argparse
import ctypes
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def _compute_bounds(metadata: Any, probes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(metadata, dict) and "bounds" in metadata:
        b = metadata["bounds"]
        return np.array(b["min"], dtype=np.float32), np.array(b["max"], dtype=np.float32)
    min_pt = probes.min(axis=0)
    max_pt = probes.max(axis=0)
    pad = 0.05 * (max_pt - min_pt + 1e-6)
    return min_pt - pad, max_pt + pad


def _build_probe_field_weights(
    probes: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    grid_res: int,
    k: int,
    chunk: int,
    weight_eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    grid_res = int(grid_res)
    k = int(k)
    lin = np.linspace(0.5 / grid_res, 1.0 - 0.5 / grid_res, grid_res, dtype=np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    centers = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    centers = bounds_min[None, :] + centers * (bounds_max - bounds_min)[None, :]

    out_idx = np.zeros((centers.shape[0], k), dtype=np.int32)
    out_w = np.zeros((centers.shape[0], k), dtype=np.float32)
    eps2 = float(weight_eps) ** 2

    for i in range(0, centers.shape[0], chunk):
        c = centers[i : i + chunk]
        d2 = ((c[:, None, :] - probes[None, :, :]) ** 2).sum(axis=2)
        k_eff = min(k, d2.shape[1])
        idx = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
        sel_d2 = np.take_along_axis(d2, idx, axis=1)
        w = 1.0 / (sel_d2 + eps2)
        w = w / (w.sum(axis=1, keepdims=True) + 1e-6)
        out_idx[i : i + chunk, :k_eff] = idx.astype(np.int32)
        out_w[i : i + chunk, :k_eff] = w.astype(np.float32)

    return (
        out_idx.reshape(grid_res, grid_res, grid_res, k),
        out_w.reshape(grid_res, grid_res, grid_res, k),
    )


def _pack_field(field: np.ndarray) -> np.ndarray:
    cells = field.reshape(-1, 27).astype(np.float32)
    packed = np.zeros((cells.shape[0] * 9, 4), dtype=np.float32)
    for i in range(9):
        packed[i::9, 0] = cells[:, i]
        packed[i::9, 1] = cells[:, 9 + i]
        packed[i::9, 2] = cells[:, 18 + i]
    return packed


def _summarize_ms(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "count": 0,
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "count": int(arr.size),
    }


def _fps_from_ms(ms_mean: float) -> float:
    if not np.isfinite(ms_mean) or ms_mean <= 0.0:
        return float("nan")
    return float(1000.0 / ms_mean)


def _prepare_falcor_runtime(falcor_python_path: str | None) -> None:
    """Ensure libFalcor and its companion libs are discoverable before import."""
    if not falcor_python_path:
        return

    py_path = Path(falcor_python_path)
    cand_dirs = [py_path.parent, py_path.parent.parent, py_path]

    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [str(d) for d in cand_dirs if d.exists()] + [p for p in old_ld.split(":") if p]
    seen = set()
    merged = []
    for p in parts:
        if p not in seen:
            merged.append(p)
            seen.add(p)
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)

    for d in cand_dirs:
        lib_path = d / "libFalcor.so"
        if lib_path.exists():
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass



def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Falcor runtime worker for benchmark")
    p.add_argument("--dataset-npz", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--routes", default="gt,model", help="comma separated: gt,model")
    p.add_argument("--model-sh-npz", default=None, help="npz with keys: frames, sh_model")
    p.add_argument("--frames-json", required=True, help="json with all_frames and warmup_count")

    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--render-scale", type=float, default=1.0)

    p.add_argument("--field-res", type=int, default=32)
    p.add_argument("--field-knn", type=int, default=8)
    p.add_argument("--weight-eps", type=float, default=0.1)
    p.add_argument("--grid-chunk", type=int, default=4096)
    p.add_argument("--field-builder", choices=["cpu", "gpu"], default="cpu",
                   help="Field construction backend (commit1 metadata only; logic unchanged)")
    p.add_argument("--gbuffer-mode", choices=["realtime", "reuse_first"], default="realtime",
                   help="GBuffer mode (commit1 metadata only; logic unchanged)")

    p.add_argument("--falcor-python-path", default=None)
    p.add_argument("--sync-gpu", action="store_true", default=False,
                   help="Force per-frame GPU synchronization/readback (slow)")
    p.add_argument("--no-sync-gpu", action="store_false", dest="sync_gpu")
    p.add_argument("--sync-every", type=int, default=0,
                   help="If >0, synchronize/readback every N benchmark frames")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.falcor_python_path and args.falcor_python_path not in sys.path:
        sys.path.insert(0, args.falcor_python_path)

    _prepare_falcor_runtime(args.falcor_python_path)

    if "BISTRO_FBX" not in os.environ or not os.environ.get("BISTRO_FBX", ""):
        root = Path(__file__).resolve().parents[1]
        os.environ["BISTRO_FBX"] = str(
            root / "1_data_generation" / "scenes" / "Bistro_v5_2" / "BistroExterior.fbx"
        )

    import falcor as fc

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset_npz, allow_pickle=True)
    tensor = np.asarray(data["tensor"], dtype=np.float32)
    probes = np.asarray(data["probe_positions"], dtype=np.float32)
    metadata = data["metadata"].item() if "metadata" in data else {}

    if "valid_mask" in data:
        valid_mask = np.asarray(data["valid_mask"], dtype=np.float32).reshape(-1)
        valid_idx = np.where(valid_mask > 0.5)[0]
        if valid_idx.size > 0 and valid_idx.size < probes.shape[0]:
            probes = probes[valid_idx]
            tensor = tensor[valid_idx]

    num_probes = int(tensor.shape[0])

    with open(args.frames_json, "r", encoding="utf-8") as f:
        fs = json.load(f)
    all_frames = [int(x) for x in fs["all_frames"]]
    warmup_count = int(fs["warmup_count"])
    warmup_frames = set(all_frames[:warmup_count])

    routes = [x.strip() for x in args.routes.split(",") if x.strip()]

    model_sh = None
    model_map = None
    if "model" in routes:
        if not args.model_sh_npz:
            raise ValueError("route includes model but --model-sh-npz not provided")
        ms = np.load(args.model_sh_npz, allow_pickle=False)
        model_frames = np.asarray(ms["frames"], dtype=np.int64)
        model_sh = np.asarray(ms["sh_model"], dtype=np.float32)
        if model_sh.shape[0] != model_frames.shape[0]:
            raise ValueError("model_sh first dim must match model frames length")
        model_map = {int(fr): i for i, fr in enumerate(model_frames.tolist())}

    bounds_min, bounds_max = _compute_bounds(metadata, probes)
    field_knn, field_w = _build_probe_field_weights(
        probes,
        bounds_min,
        bounds_max,
        grid_res=args.field_res,
        k=args.field_knn,
        chunk=args.grid_chunk,
        weight_eps=args.weight_eps,
    )

    render_scale = max(0.1, float(args.render_scale))
    out_w = int(round(args.width * render_scale))
    out_h = int(round(args.height * render_scale))

    testbed_cls = getattr(fc, "Testbed", None)
    if testbed_cls is None and hasattr(fc, "falcor_ext"):
        testbed_cls = getattr(fc.falcor_ext, "Testbed", None)
    if testbed_cls is None:
        raise RuntimeError("Falcor Testbed not found in worker")

    testbed = testbed_cls(width=out_w, height=out_h, create_window=False)
    graph = testbed.create_render_graph("BenchmarkProbeVolumeWorker")
    graph.create_pass("GBufferRT", "GBufferRT", {"samplePattern": "Center", "sampleCount": 1})
    for out_name in (
        "GBufferRT.posW",
        "GBufferRT.normW",
        "GBufferRT.diffuseOpacity",
        "GBufferRT.emissive",
        "GBufferRT.depth",
        "GBufferRT.linearZ",
    ):
        graph.mark_output(out_name)
    testbed.render_graph = graph
    testbed.load_scene(args.scene)

    device = testbed.device
    field_buf = device.create_structured_buffer(
        struct_size=16,
        element_count=(args.field_res ** 3) * 9,
        bind_flags=fc.ResourceBindFlags.ShaderResource | fc.ResourceBindFlags.UnorderedAccess,
    )
    output_tex = device.create_texture(
        format=fc.ResourceFormat.RGBA32Float,
        width=out_w,
        height=out_h,
        mip_levels=1,
        bind_flags=fc.ResourceBindFlags.UnorderedAccess | fc.ResourceBindFlags.ShaderResource,
    )

    shader_path = Path(__file__).parent / "sh_probe_volume.cs.slang"
    compute = fc.ComputePass(device, file=shader_path, cs_entry="main")
    compute.globals.gFieldSH = field_buf
    compute.globals.gOutput = output_tex
    compute.globals.gBoundsMin = fc.float3(float(bounds_min[0]), float(bounds_min[1]), float(bounds_min[2]))
    compute.globals.gBoundsMax = fc.float3(float(bounds_max[0]), float(bounds_max[1]), float(bounds_max[2]))
    compute.globals.gFieldRes = fc.uint3(int(args.field_res), int(args.field_res), int(args.field_res))
    compute.globals.gFrameDim = fc.uint2(int(out_w), int(out_h))
    compute.globals.gExposure = 1.0
    compute.globals.gAOStrength = 0.0

    field_builder = str(args.field_builder)
    if field_builder not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported field builder: {field_builder}")

    num_cells = int(args.field_res) ** 3
    k_field = int(field_knn.shape[-1])
    probe_sh_buf = None
    field_build = None
    if field_builder == "gpu":
        probe_sh_buf = device.create_structured_buffer(
            struct_size=4,
            element_count=num_probes * 27,
            bind_flags=fc.ResourceBindFlags.ShaderResource,
        )
        knn_idx_buf = device.create_structured_buffer(
            struct_size=4,
            element_count=num_cells * k_field,
            bind_flags=fc.ResourceBindFlags.ShaderResource,
        )
        knn_w_buf = device.create_structured_buffer(
            struct_size=4,
            element_count=num_cells * k_field,
            bind_flags=fc.ResourceBindFlags.ShaderResource,
        )

        knn_idx_flat = field_knn.reshape(-1).astype(np.uint32, copy=False)
        knn_w_flat = field_w.reshape(-1).astype(np.float32, copy=False)
        knn_idx_buf.from_numpy(knn_idx_flat)
        knn_w_buf.from_numpy(knn_w_flat)

        field_build_path = Path(__file__).parent / "sh_probe_field_build.cs.slang"
        field_build = fc.ComputePass(device, file=field_build_path, cs_entry="main")
        field_build.globals.gProbeSH = probe_sh_buf
        field_build.globals.gKnnIdx = knn_idx_buf
        field_build.globals.gKnnW = knn_w_buf
        field_build.globals.gFieldSH = field_buf
        field_build.globals.gFieldRes = fc.uint3(int(args.field_res), int(args.field_res), int(args.field_res))
        field_build.globals.gNumProbes = int(num_probes)
        field_build.globals.gK = int(k_field)

    phase_times: Dict[str, Dict[str, List[float]]] = {
        "gbuffer": {"ms": []},
        "gt": {
            "source": [], "field": [], "pack": [], "upload": [],
            "field_build_submit": [], "field_build_sync": [],
            "shade": [], "shade_submit": [], "shade_sync": [],
            "total": [], "with_gbuffer": []
        },
        "model": {
            "source": [], "field": [], "pack": [], "upload": [],
            "field_build_submit": [], "field_build_sync": [],
            "shade": [], "shade_submit": [], "shade_sync": [],
            "total": [], "with_gbuffer": []
        },
    }
    frame_rows: List[Dict[str, Any]] = []
    bench_counter = 0

    gbuffer_mode = str(args.gbuffer_mode)
    if gbuffer_mode not in {"realtime", "reuse_first"}:
        raise ValueError(f"Unsupported gbuffer mode: {gbuffer_mode}")
    gbuffer_cache: Tuple[Any, Any, Any, Any, Any, Any] | None = None
    gbuffer_frames_rendered = 0

    for i, frame in enumerate(all_frames):
        rendered_gbuffer_this_frame = False
        if gbuffer_mode == "reuse_first" and gbuffer_cache is not None:
            gbuffer_ms = 0.0
            pos_tex, norm_tex, alb_tex, emi_tex, dep_tex, lin_tex = gbuffer_cache
        else:
            t_frame = float(frame) / float(args.fps)
            try:
                testbed.clock.time = t_frame
            except Exception:
                try:
                    testbed.clock.setTime(t_frame)
                except Exception:
                    pass

            t0 = time.perf_counter()
            testbed.resize_frame_buffer(out_w, out_h)
            testbed.frame()
            t1 = time.perf_counter()
            gbuffer_ms = (t1 - t0) * 1000.0
            rendered_gbuffer_this_frame = True
            gbuffer_frames_rendered += 1

            pos_tex = graph.get_output("GBufferRT.posW")
            norm_tex = graph.get_output("GBufferRT.normW")
            alb_tex = graph.get_output("GBufferRT.diffuseOpacity")
            emi_tex = graph.get_output("GBufferRT.emissive")
            dep_tex = graph.get_output("GBufferRT.depth")
            lin_tex = graph.get_output("GBufferRT.linearZ")

            if gbuffer_mode == "reuse_first" and gbuffer_cache is None:
                gbuffer_cache = (pos_tex, norm_tex, alb_tex, emi_tex, dep_tex, lin_tex)

        compute.globals.gPosW = pos_tex
        compute.globals.gNormW = norm_tex
        compute.globals.gAlbedo = alb_tex
        compute.globals.gEmissive = emi_tex
        compute.globals.gDepth = dep_tex
        compute.globals.gLinearZ = lin_tex

        is_bench = frame not in warmup_frames
        if is_bench:
            bench_counter += 1

        row: Dict[str, Any] = {
            "frame": int(frame),
            "is_warmup": not is_bench,
            "gbuffer_ms": gbuffer_ms,
            "gbuffer_rendered": int(rendered_gbuffer_this_frame),
        }

        for route in routes:
            t_src0 = time.perf_counter()
            if route == "gt":
                sh_probe = tensor[:, frame, :]
            elif route == "model":
                assert model_sh is not None and model_map is not None
                if frame not in model_map:
                    raise KeyError(f"Frame {frame} not present in model_sh npz")
                sh_probe = model_sh[model_map[frame]]
            else:
                raise RuntimeError(f"Unknown route: {route}")
            t_src1 = time.perf_counter()

            field_build_submit_ms = 0.0
            field_build_sync_ms = 0.0

            if field_builder == "cpu":
                t_field0 = time.perf_counter()
                field = (field_w[..., None] * sh_probe[field_knn]).sum(axis=-2)
                t_field1 = time.perf_counter()

                t_pack0 = time.perf_counter()
                packed = _pack_field(field)
                t_pack1 = time.perf_counter()

                t_upload0 = time.perf_counter()
                field_buf.from_numpy(packed)
                t_upload1 = time.perf_counter()
            else:
                assert probe_sh_buf is not None and field_build is not None
                t_field0 = time.perf_counter()
                t_field1 = t_field0

                t_pack0 = time.perf_counter()
                t_pack1 = t_pack0

                t_upload0 = time.perf_counter()
                probe_sh_buf.from_numpy(sh_probe.reshape(-1).astype(np.float32, copy=False))
                t_upload1 = time.perf_counter()

                t_fb0 = time.perf_counter()
                field_build.execute(threads_x=num_cells, threads_y=1, threads_z=1)
                t_fb1 = time.perf_counter()
                field_build_submit_ms = (t_fb1 - t_fb0) * 1000.0

            t_shade_submit0 = time.perf_counter()
            compute.execute(threads_x=out_w, threads_y=out_h)
            t_shade_submit1 = time.perf_counter()

            should_sync = bool(args.sync_gpu)
            if (not should_sync) and int(args.sync_every) > 0 and is_bench and (bench_counter % int(args.sync_every) == 0):
                should_sync = True

            shade_sync_ms = 0.0
            if should_sync:
                t_sync0 = time.perf_counter()
                _ = output_tex.to_numpy()
                t_sync1 = time.perf_counter()
                shade_sync_ms = (t_sync1 - t_sync0) * 1000.0

            source_ms = (t_src1 - t_src0) * 1000.0
            field_ms = (t_field1 - t_field0) * 1000.0
            pack_ms = (t_pack1 - t_pack0) * 1000.0
            upload_ms = (t_upload1 - t_upload0) * 1000.0
            shade_submit_ms = (t_shade_submit1 - t_shade_submit0) * 1000.0
            shade_ms = shade_submit_ms + shade_sync_ms
            total_ms = source_ms + field_ms + pack_ms + upload_ms + shade_ms

            row[f"{route}_source_ms"] = source_ms
            row[f"{route}_field_ms"] = field_ms
            row[f"{route}_pack_ms"] = pack_ms
            row[f"{route}_upload_ms"] = upload_ms
            row[f"{route}_field_build_submit_ms"] = field_build_submit_ms
            row[f"{route}_field_build_sync_ms"] = field_build_sync_ms
            row[f"{route}_shade_submit_ms"] = shade_submit_ms
            row[f"{route}_shade_sync_ms"] = shade_sync_ms
            row[f"{route}_shade_ms"] = shade_ms
            row[f"{route}_sync"] = int(should_sync)
            row[f"{route}_total_ms"] = total_ms
            row[f"{route}_with_gbuffer_ms"] = total_ms + gbuffer_ms

            if is_bench:
                phase_times[route]["source"].append(source_ms)
                phase_times[route]["field"].append(field_ms)
                phase_times[route]["pack"].append(pack_ms)
                phase_times[route]["upload"].append(upload_ms)
                phase_times[route]["field_build_submit"].append(field_build_submit_ms)
                phase_times[route]["field_build_sync"].append(field_build_sync_ms)
                phase_times[route]["shade_submit"].append(shade_submit_ms)
                phase_times[route]["shade"].append(shade_ms)
                if should_sync:
                    phase_times[route]["shade_sync"].append(shade_sync_ms)
                phase_times[route]["total"].append(total_ms)
                phase_times[route]["with_gbuffer"].append(total_ms + gbuffer_ms)

        if is_bench:
            phase_times["gbuffer"]["ms"].append(gbuffer_ms)
            frame_rows.append(row)

        if i % max(1, len(all_frames) // 20) == 0:
            tag = "warmup" if not is_bench else "bench"
            print(f"[{i+1}/{len(all_frames)}] frame={frame} ({tag})")

    summary: Dict[str, Any] = {
        "routes": routes,
        "warmup_frames": int(len(warmup_frames)),
        "benchmark_frames": int(len(frame_rows)),
        "field_builder": str(args.field_builder),
        "gbuffer_mode": str(args.gbuffer_mode),
        "gbuffer_frames_rendered": int(gbuffer_frames_rendered),
        "sync_gpu_forced": bool(args.sync_gpu),
        "sync_every": int(max(0, int(args.sync_every))),
        "gbuffer": _summarize_ms(phase_times["gbuffer"]["ms"]),
        "results": {},
    }

    for route in routes:
        r = {
            "source_ms": _summarize_ms(phase_times[route]["source"]),
            "field_ms": _summarize_ms(phase_times[route]["field"]),
            "pack_ms": _summarize_ms(phase_times[route]["pack"]),
            "upload_ms": _summarize_ms(phase_times[route]["upload"]),
            "field_build_submit_ms": _summarize_ms(phase_times[route]["field_build_submit"]),
            "field_build_sync_ms": _summarize_ms(phase_times[route]["field_build_sync"]),
            "shade_submit_ms": _summarize_ms(phase_times[route]["shade_submit"]),
            "shade_sync_ms": _summarize_ms(phase_times[route]["shade_sync"]),
            "shade_ms": _summarize_ms(phase_times[route]["shade"]),
            "total_ms": _summarize_ms(phase_times[route]["total"]),
            "with_gbuffer_ms": _summarize_ms(phase_times[route]["with_gbuffer"]),
        }
        r["fps_route_only"] = _fps_from_ms(r["total_ms"]["mean"])
        r["fps_with_gbuffer"] = _fps_from_ms(r["with_gbuffer_ms"]["mean"])
        summary["results"][route] = r

    sum_path = out_dir / "falcor_summary.json"
    csv_path = out_dir / "falcor_frames.csv"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if frame_rows:
        fields = sorted({k for row in frame_rows for k in row.keys()})
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(frame_rows)

    print(f"Saved: {sum_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
