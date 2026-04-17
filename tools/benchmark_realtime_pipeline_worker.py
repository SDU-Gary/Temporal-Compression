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


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "2_src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.unified_metrics import (
    compute_benchmark_metrics,
    compute_benchmark_srgb_metrics,
    compute_fixed_tm_metrics,
    compute_hdr_radiance_metrics,
    compute_linear_joint_metrics,
    compute_pair_metrics,
    compute_real_render_hdr_scene_metrics,
    compute_sh_metrics,
    srgb_encode,
    tone_map_reinhard,
)


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


def compute_image_metrics(img_gt: np.ndarray, img_pred: np.ndarray) -> Dict[str, float]:
    metrics = compute_pair_metrics(img_gt, img_pred, max_i=1.0, clip_unit=True, ssim_mode="image")
    return {
        "psnr": float(metrics["psnr"]),
        "ssim": float(metrics["ssim"]),
        "mae": float(metrics["mae"]),
    }


def _parse_roi(roi: str | None) -> Tuple[float, float, float, float] | None:
    if not roi:
        return None
    try:
        vals = [float(x.strip()) for x in str(roi).split(",")]
        if len(vals) != 4:
            return None
        x0, y0, x1, y1 = vals
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0 = max(0.0, min(1.0, x0))
        y0 = max(0.0, min(1.0, y0))
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        if (x1 - x0) <= 1e-6 or (y1 - y0) <= 1e-6:
            return None
        return (x0, y0, x1, y1)
    except Exception:
        return None


def _crop_roi(img: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    ix0 = int(np.floor(x0 * w))
    ix1 = int(np.ceil(x1 * w))
    iy0 = int(np.floor(y0 * h))
    iy1 = int(np.ceil(y1 * h))
    ix0 = max(0, min(w - 1, ix0))
    iy0 = max(0, min(h - 1, iy0))
    ix1 = max(ix0 + 1, min(w, ix1))
    iy1 = max(iy0 + 1, min(h, iy1))
    return img[iy0:iy1, ix0:ix1, ...]


def _linear_to_srgb_u8(img_linear: np.ndarray) -> np.ndarray:
    srgb = srgb_encode(tone_map_reinhard(np.asarray(img_linear, dtype=np.float32)))
    return np.clip(srgb * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


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


def _set_testbed_graph(testbed: Any, graph: Any) -> None:
    if hasattr(testbed, "render_graph"):
        testbed.render_graph = graph
        return
    if hasattr(testbed, "set_active_graph"):
        testbed.set_active_graph(graph)
        return
    if hasattr(testbed, "setActiveGraph"):
        testbed.setActiveGraph(graph)
        return
    raise RuntimeError("Unable to switch Testbed render graph")


def _read_graph_output_rgb(graph: Any, name: str) -> np.ndarray:
    tex = graph.get_output(name)
    img = np.asarray(tex.to_numpy(), dtype=np.float32)
    if img.ndim == 1:
        size = int(np.sqrt(img.size / 4))
        img = img.reshape(size, size, -1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    return img


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
    p.add_argument("--compute-image-metrics", action="store_true", default=False,
                   help="Compute PSNR/SSIM for GT vs Model on sampled benchmark frames")
    p.add_argument("--save-frame-metrics-every", type=int, default=30,
                   help="Compute image metrics every N benchmark frames")
    p.add_argument("--normal-transform",
                   choices=["identity", "swap_yz", "swap_xz", "swap_xy", "flip_x", "flip_y", "flip_z"],
                   default="identity",
                   help="Debug normal-space orientation mismatch in SH shading")
    p.add_argument("--cosine-mode", choices=["irradiance", "radiance"], default="irradiance",
                   help="Irradiance applies cosine convolution A_l; radiance disables A_l")
    p.add_argument("--sh-debug-coeff", type=int, default=-1,
                   help="If in [0,8], keep only this SH band coefficient per RGB (Y_lm debug)")
    p.add_argument("--single-probe-index", type=int, default=-1,
                   help="If >=0, broadcast one probe SH to all probes (disables spatial variation)")
    p.add_argument("--axis-test", choices=["off", "x", "y", "z"], default="off",
                   help="Override SH with one strong first-order axis component")
    p.add_argument("--axis-test-sign", choices=["pos", "neg"], default="pos",
                   help="Sign for axis-test component")
    p.add_argument("--axis-test-strength", type=float, default=1.0,
                   help="Strength for axis-test SH coefficient")
    p.add_argument("--save-sampled-images-dir", default=None,
                   help="Optional output dir for sampled metric frames (npz: linear/srgb/depth)")
    p.add_argument(
        "--dump-sampled-images-every",
        type=int,
        default=0,
        help=(
            "If >0, dump sampled images every N benchmark frames even when "
            "metric computation is disabled."
        ),
    )
    p.add_argument("--output-scale", type=float, default=1.0,
                   help="Scale applied to SH shading output before metrics/output")
    p.add_argument("--metric-align-scale-gt", type=float, default=1.0,
                   help="Scale GT route linear image before metric computation (metrics-only)")
    p.add_argument("--metric-align-scale-model", type=float, default=1.0,
                   help="Scale Model route linear image before metric computation (metrics-only)")
    p.add_argument("--roi", default=None,
                   help="Optional normalized ROI x0,y0,x1,y1 for extra image metrics")
    p.add_argument("--pt-reference", action="store_true", default=False,
                   help="Also render PathTracer reference and compute route-vs-PT metrics")
    p.add_argument("--pt-spp", type=int, default=8,
                   help="PathTracer spp per frame for PT reference")
    p.add_argument("--pt-bounces", type=int, default=4,
                   help="PathTracer max surface bounces for PT reference")
    p.add_argument("--pt-use-nee", action="store_true", default=True,
                   help="Enable next-event estimation for PT reference")
    p.add_argument("--pt-no-nee", action="store_false", dest="pt_use_nee")
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

    pt_graph = None
    if bool(args.pt_reference):
        pt_graph = testbed.create_render_graph("BenchmarkPTReferenceWorker")
        pt_graph.create_pass("PathTracer", "PathTracer", {
            "samplesPerPixel": int(max(1, int(args.pt_spp))),
            "maxSurfaceBounces": int(max(0, int(args.pt_bounces))),
            "useNEE": bool(args.pt_use_nee),
        })
        pt_graph.create_pass("VBufferRT", "VBufferRT", {"samplePattern": "Center", "sampleCount": 1})
        pt_graph.add_edge("VBufferRT.vbuffer", "PathTracer.vbuffer")
        pt_graph.add_edge("VBufferRT.viewW", "PathTracer.viewW")
        try:
            pt_graph.add_edge("VBufferRT.mvec", "PathTracer.mvec")
        except Exception:
            pass
        pt_graph.mark_output("PathTracer.color")
        _set_testbed_graph(testbed, graph)

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
    compute.globals.gExposure = float(args.output_scale)
    compute.globals.gAOStrength = 0.0

    normal_mode_map = {
        "identity": 0,
        "swap_yz": 1,
        "swap_xz": 2,
        "swap_xy": 3,
        "flip_x": 4,
        "flip_y": 5,
        "flip_z": 6,
    }
    cosine_mode_map = {
        "irradiance": 0,
        "radiance": 1,
    }
    compute.globals.gNormalTransform = int(normal_mode_map.get(str(args.normal_transform), 0))
    compute.globals.gCosineMode = int(cosine_mode_map.get(str(args.cosine_mode), 0))

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
        "pt_reference": {"ms": []},
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

    do_image_metrics = bool(args.compute_image_metrics and set(routes) == {"gt", "model"})
    metric_every = max(1, int(args.save_frame_metrics_every))
    dump_every = int(max(0, int(args.dump_sampled_images_every)))
    image_metrics: List[Dict[str, float]] = []
    roi = _parse_roi(args.roi)
    metric_align_scale = {
        "gt": float(args.metric_align_scale_gt),
        "model": float(args.metric_align_scale_model),
    }

    save_sampled_images_dir: Path | None = None
    if args.save_sampled_images_dir:
        save_sampled_images_dir = Path(args.save_sampled_images_dir)
        save_sampled_images_dir.mkdir(parents=True, exist_ok=True)

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

        need_image_metrics_this_frame = bool(do_image_metrics and is_bench and (bench_counter % metric_every == 0))
        need_pt_metrics_this_frame = bool(args.pt_reference and is_bench and (bench_counter % metric_every == 0))
        need_dump_images_this_frame = bool(
            save_sampled_images_dir is not None
            and is_bench
            and dump_every > 0
            and (bench_counter % dump_every == 0)
        )

        row: Dict[str, Any] = {
            "frame": int(frame),
            "is_warmup": not is_bench,
            "gbuffer_ms": gbuffer_ms,
            "gbuffer_rendered": int(rendered_gbuffer_this_frame),
            "metric_sampled": int(
                need_image_metrics_this_frame
                or need_pt_metrics_this_frame
                or need_dump_images_this_frame
            ),
        }

        rendered_images: Dict[str, np.ndarray] = {}
        route_probe_sh: Dict[str, np.ndarray] = {}

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

            if int(args.single_probe_index) >= 0:
                idx = max(0, min(num_probes - 1, int(args.single_probe_index)))
                sh_probe = np.repeat(sh_probe[idx:idx + 1, :], num_probes, axis=0)

            if 0 <= int(args.sh_debug_coeff) <= 8:
                c = int(args.sh_debug_coeff)
                dbg = np.zeros_like(sh_probe)
                dbg[:, c] = sh_probe[:, c]
                dbg[:, 9 + c] = sh_probe[:, 9 + c]
                dbg[:, 18 + c] = sh_probe[:, 18 + c]
                sh_probe = dbg

            if str(args.axis_test) != "off":
                axis_map = {"x": 3, "y": 1, "z": 2}
                c = int(axis_map.get(str(args.axis_test), 3))
                sign = -1.0 if str(args.axis_test_sign) == "neg" else 1.0
                strength = float(args.axis_test_strength) * sign
                axis_probe = np.zeros((num_probes, 27), dtype=np.float32)
                axis_probe[:, c] = strength
                axis_probe[:, 9 + c] = strength
                axis_probe[:, 18 + c] = strength
                sh_probe = axis_probe

            if need_image_metrics_this_frame:
                route_probe_sh[route] = np.asarray(sh_probe, dtype=np.float32)

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
            if need_image_metrics_this_frame or need_pt_metrics_this_frame or need_dump_images_this_frame:
                should_sync = True

            shade_sync_ms = 0.0
            if should_sync:
                t_sync0 = time.perf_counter()
                img = np.asarray(output_tex.to_numpy(), dtype=np.float32)
                t_sync1 = time.perf_counter()
                shade_sync_ms = (t_sync1 - t_sync0) * 1000.0
                if img.shape[-1] > 3:
                    img = img[..., :3]
                if need_image_metrics_this_frame or need_pt_metrics_this_frame or need_dump_images_this_frame:
                    rendered_images[route] = img

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

        if need_pt_metrics_this_frame and pt_graph is not None:
            t_frame = float(frame) / float(args.fps)
            try:
                testbed.clock.time = t_frame
            except Exception:
                try:
                    testbed.clock.setTime(t_frame)
                except Exception:
                    pass
            _set_testbed_graph(testbed, pt_graph)
            t_pt0 = time.perf_counter()
            testbed.resize_frame_buffer(out_w, out_h)
            testbed.frame()
            t_pt1 = time.perf_counter()
            row["pt_reference_ms"] = (t_pt1 - t_pt0) * 1000.0
            phase_times["pt_reference"]["ms"].append(row["pt_reference_ms"])
            rendered_images["pt"] = _read_graph_output_rgb(pt_graph, "PathTracer.color")
            _set_testbed_graph(testbed, graph)

        if is_bench:
            phase_times["gbuffer"]["ms"].append(gbuffer_ms)
            frame_rows.append(row)

            if (need_image_metrics_this_frame or need_pt_metrics_this_frame or need_dump_images_this_frame):
                rec: Dict[str, float] = {"frame": int(frame)}
                has_metric = False

                if set(routes) == {"gt", "model"} and "gt" in rendered_images and "model" in rendered_images and need_image_metrics_this_frame:
                    gt_linear_raw = rendered_images["gt"]
                    model_linear_raw = rendered_images["model"]
                    bench_metric = compute_benchmark_metrics(
                        gt_linear_raw,
                        model_linear_raw,
                        metric_align_scale_gt=float(metric_align_scale["gt"]),
                        metric_align_scale_pred=float(metric_align_scale["model"]),
                        max_i=1.0,
                    )

                    gt_linear = gt_linear_raw * float(metric_align_scale["gt"])
                    model_linear = model_linear_raw * float(metric_align_scale["model"])
                    gt_img = srgb_encode(tone_map_reinhard(gt_linear))
                    model_img = srgb_encode(tone_map_reinhard(model_linear))

                    sh_metric = compute_sh_metrics(route_probe_sh["gt"], route_probe_sh["model"], max_i=1.0)
                    hdr_metric = compute_hdr_radiance_metrics(gt_linear_raw, model_linear_raw, max_i=1.0)
                    fixedtm_metric = compute_fixed_tm_metrics(gt_linear_raw, model_linear_raw, exposure=1.0, max_i=1.0)
                    real_hdr_metric = compute_real_render_hdr_scene_metrics(gt_linear_raw, model_linear_raw, max_i=1.0)

                    rec.update({
                        "sh_psnr": float(sh_metric["sh_psnr"]),
                        "sh_ssim": float(sh_metric["sh_ssim"]),
                        "sh_mae": float(sh_metric["sh_mae"]),
                        "sh_rmse": float(sh_metric["sh_rmse"]),
                        "hdr_psnr": float(hdr_metric["hdr_psnr"]),
                        "hdr_ssim": float(hdr_metric["hdr_ssim"]),
                        "fixedtm_psnr": float(fixedtm_metric["fixedtm_psnr"]),
                        "fixedtm_ssim": float(fixedtm_metric["fixedtm_ssim"]),
                        "real_render_hdr_psnr": float(real_hdr_metric["real_render_hdr_psnr"]),
                        "real_render_hdr_ssim": float(real_hdr_metric["real_render_hdr_ssim"]),
                        "benchmark_psnr": float(bench_metric["benchmark_psnr"]),
                        "benchmark_ssim": float(bench_metric["benchmark_ssim"]),
                        "benchmark_linear_psnr": float(bench_metric["benchmark_linear_psnr"]),
                        "benchmark_linear_ssim": float(bench_metric["benchmark_linear_ssim"]),
                        "psnr": float(bench_metric["benchmark_psnr"]),
                        "ssim": float(bench_metric["benchmark_ssim"]),
                        "linear_psnr": float(bench_metric["benchmark_linear_psnr"]),
                        "linear_ssim": float(bench_metric["benchmark_linear_ssim"]),
                        "linear_scale": float(bench_metric["benchmark_linear_scale"]),
                        "metric_align_scale_gt": float(metric_align_scale["gt"]),
                        "metric_align_scale_model": float(metric_align_scale["model"]),
                        "gt_neg_ratio": float(np.mean(gt_linear_raw < 0.0)),
                        "model_neg_ratio": float(np.mean(model_linear_raw < 0.0)),
                        "gt_p99": float(np.percentile(gt_linear_raw, 99.0)),
                        "model_p99": float(np.percentile(model_linear_raw, 99.0)),
                    })
                    has_metric = True

                    if roi is not None:
                        gt_roi = _crop_roi(gt_img, roi)
                        model_roi = _crop_roi(model_img, roi)
                        roi_metric = compute_image_metrics(gt_roi, model_roi)
                        gt_linear_roi_raw = _crop_roi(gt_linear, roi)
                        model_linear_roi_raw = _crop_roi(model_linear, roi)
                        roi_linear = compute_linear_joint_metrics(gt_linear_roi_raw, model_linear_roi_raw, max_i=1.0)
                        rec["roi_psnr"] = float(roi_metric["psnr"])
                        rec["roi_ssim"] = float(roi_metric["ssim"])
                        rec["roi_linear_psnr"] = float(roi_linear["linear_psnr"])
                        rec["roi_linear_ssim"] = float(roi_linear["linear_ssim"])

                if need_pt_metrics_this_frame and "pt" in rendered_images:
                    pt_linear = rendered_images["pt"]
                    pt_srgb = srgb_encode(tone_map_reinhard(pt_linear))
                    pt_metric_count = 0
                    for route_name in routes:
                        if route_name not in rendered_images:
                            continue
                        route_linear_raw = rendered_images[route_name]
                        route_linear = route_linear_raw * float(metric_align_scale.get(route_name, 1.0))
                        route_srgb = srgb_encode(tone_map_reinhard(route_linear))
                        m_pt_srgb = compute_benchmark_srgb_metrics(pt_srgb, route_srgb, max_i=1.0)
                        m_pt_linear = compute_linear_joint_metrics(pt_linear, route_linear, max_i=1.0)

                        taskh_num = float(np.sum(pt_linear * route_linear))
                        taskh_den_route = float(np.sum(route_linear * route_linear))
                        taskh_den_pt = float(np.sum(pt_linear * pt_linear))
                        taskh_scale_route_to_pt = taskh_num / max(taskh_den_route, 1e-12)
                        taskh_scale_pt_to_route = taskh_num / max(taskh_den_pt, 1e-12)

                        rec[f"pt_psnr_{route_name}"] = float(m_pt_srgb["psnr"])
                        rec[f"pt_ssim_{route_name}"] = float(m_pt_srgb["ssim"])
                        rec[f"pt_mae_{route_name}"] = float(m_pt_srgb["mae"])
                        rec[f"pt_linear_psnr_{route_name}"] = float(m_pt_linear["linear_psnr"])
                        rec[f"pt_linear_ssim_{route_name}"] = float(m_pt_linear["linear_ssim"])
                        rec[f"pt_linear_mae_{route_name}"] = float(m_pt_linear["linear_mae"])
                        rec[f"taskh_scale_{route_name}_to_pt"] = float(taskh_scale_route_to_pt)
                        rec[f"taskh_scale_pt_to_{route_name}"] = float(taskh_scale_pt_to_route)
                        rec[f"metric_align_scale_{route_name}"] = float(metric_align_scale.get(route_name, 1.0))

                        if roi is not None:
                            pt_roi = _crop_roi(pt_srgb, roi)
                            route_roi = _crop_roi(route_srgb, roi)
                            m_pt_roi = compute_image_metrics(pt_roi, route_roi)
                            rec[f"pt_roi_psnr_{route_name}"] = float(m_pt_roi["psnr"])
                            rec[f"pt_roi_ssim_{route_name}"] = float(m_pt_roi["ssim"])
                        pt_metric_count += 1
                    if pt_metric_count > 0:
                        has_metric = True

                if save_sampled_images_dir is not None and (has_metric or need_dump_images_this_frame):
                    save_path = save_sampled_images_dir / f"frame_{int(frame):04d}.npz"
                    if has_metric:
                        depth_np = np.asarray(dep_tex.to_numpy(), dtype=np.float32)
                        if depth_np.ndim == 3 and depth_np.shape[-1] > 1:
                            depth_np = depth_np[..., 0]
                        payload: Dict[str, np.ndarray] = {"depth": depth_np.astype(np.float32)}
                        if "gt" in rendered_images:
                            payload["gt_linear"] = rendered_images["gt"].astype(np.float32)
                        if "model" in rendered_images:
                            payload["model_linear"] = rendered_images["model"].astype(np.float32)
                        if "pt" in rendered_images:
                            payload["pt_linear"] = rendered_images["pt"].astype(np.float32)
                        if "gt" in rendered_images and "model" in rendered_images:
                            gt_img = srgb_encode(tone_map_reinhard(rendered_images["gt"]))
                            model_img = srgb_encode(tone_map_reinhard(rendered_images["model"]))
                            payload["gt_srgb"] = gt_img.astype(np.float32)
                            payload["model_srgb"] = model_img.astype(np.float32)
                            payload["err_srgb"] = np.abs(gt_img - model_img).astype(np.float32)
                            payload["err_linear"] = np.abs(rendered_images["gt"] - rendered_images["model"]).astype(np.float32)
                        if "pt" in rendered_images and "gt" in rendered_images:
                            payload["err_pt_gt_linear"] = np.abs(rendered_images["pt"] - rendered_images["gt"]).astype(np.float32)
                        if "pt" in rendered_images and "model" in rendered_images:
                            payload["err_pt_model_linear"] = np.abs(rendered_images["pt"] - rendered_images["model"]).astype(np.float32)
                        np.savez_compressed(save_path, **payload)
                    else:
                        # Fast export path: lightweight uncompressed u8 sRGB payload.
                        payload_fast: Dict[str, np.ndarray] = {}
                        if "gt" in rendered_images:
                            payload_fast["gt_srgb_u8"] = _linear_to_srgb_u8(rendered_images["gt"])
                        if "model" in rendered_images:
                            payload_fast["model_srgb_u8"] = _linear_to_srgb_u8(rendered_images["model"])
                        if "pt" in rendered_images:
                            payload_fast["pt_srgb_u8"] = _linear_to_srgb_u8(rendered_images["pt"])
                        if payload_fast:
                            np.savez(save_path, **payload_fast)

                if has_metric:
                    image_metrics.append(rec)

        if i % max(1, len(all_frames) // 20) == 0:
            tag = "warmup" if not is_bench else "bench"
            print(f"[{i+1}/{len(all_frames)}] frame={frame} ({tag})")

    summary: Dict[str, Any] = {
        "routes": routes,
        "warmup_frames": int(len(warmup_frames)),
        "benchmark_frames": int(len(frame_rows)),
        "field_builder": str(args.field_builder),
        "gbuffer_mode": str(args.gbuffer_mode),
        "normal_transform": str(args.normal_transform),
        "cosine_mode": str(args.cosine_mode),
        "sh_debug_coeff": int(args.sh_debug_coeff),
        "single_probe_index": int(args.single_probe_index),
        "axis_test": str(args.axis_test),
        "axis_test_sign": str(args.axis_test_sign),
        "axis_test_strength": float(args.axis_test_strength),
        "output_scale": float(args.output_scale),
        "metric_align_scale_gt": float(args.metric_align_scale_gt),
        "metric_align_scale_model": float(args.metric_align_scale_model),
        "dump_sampled_images_every": int(max(0, int(args.dump_sampled_images_every))),
        "roi": str(args.roi) if args.roi else None,
        "pt_reference": bool(args.pt_reference),
        "pt_spp": int(max(1, int(args.pt_spp))),
        "pt_bounces": int(max(0, int(args.pt_bounces))),
        "pt_use_nee": bool(args.pt_use_nee),
        "gbuffer_frames_rendered": int(gbuffer_frames_rendered),
        "sync_gpu_forced": bool(args.sync_gpu),
        "sync_every": int(max(0, int(args.sync_every))),
        "gbuffer": _summarize_ms(phase_times["gbuffer"]["ms"]),
        "pt_reference_ms": _summarize_ms(phase_times["pt_reference"]["ms"]),
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
        sync_count = len(phase_times[route]["shade_sync"])
        total_count = max(1, len(phase_times[route]["total"]))
        r["sync_ratio"] = float(sync_count / total_count)
        summary["results"][route] = r

    summary["measurement_mode"] = (
        "force_every_frame" if bool(args.sync_gpu)
        else ("sampled_every_n" if int(max(0, int(args.sync_every))) > 0 else "async_no_sync")
    )

    if image_metrics:
        mean_fields: Dict[str, float] = {}
        metric_keys = [k for k in image_metrics[0].keys() if k != "frame"]
        for key in metric_keys:
            vals = [float(m[key]) for m in image_metrics if key in m]
            if vals:
                mean_fields[f"mean_{key}"] = float(np.mean(vals))
        summary["image_metrics"] = {
            "samples": int(len(image_metrics)),
            **mean_fields,
            "per_sample": image_metrics,
        }

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
