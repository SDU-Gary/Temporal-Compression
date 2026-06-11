#!/usr/bin/env python3
"""Interactive Falcor window for GT-vs-compressed SH lighting comparison.

Run with Falcor's embedded Python. The model output must be precomputed by
prepare_falcor_demo_sh.py, so this runtime has no PyTorch dependency.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"


def _prepare_falcor_runtime(falcor_python_path: str | None) -> None:
    if not falcor_python_path:
        return
    py_path = Path(falcor_python_path)
    cand_dirs = [py_path.parent, py_path.parent.parent, py_path]
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [str(d) for d in cand_dirs if d.exists()] + [p for p in old_ld.split(":") if p]
    seen = set()
    merged: List[str] = []
    for p in parts:
        if p and p not in seen:
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


def _resolve_dataset_npz(dataset: str) -> Path:
    p = Path(dataset)
    if p.is_dir():
        p = p / "parametric_tensor.npz"
    if not p.exists():
        raise FileNotFoundError(f"Dataset npz not found: {p}")
    return p


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


def _clear_testbed_graph(testbed: Any) -> bool:
    try:
        testbed.render_graph = None
        return True
    except Exception:
        return False


def _set_clock(testbed: Any, t_sec: float) -> None:
    try:
        testbed.clock.time = float(t_sec)
    except Exception:
        try:
            testbed.clock.setTime(float(t_sec))
        except Exception:
            pass


def _as_float_tuple(value: Any, count: int | None = None) -> Tuple[float, ...]:
    vals: List[float] = []
    if hasattr(value, "x"):
        vals.append(float(value.x))
    if hasattr(value, "y"):
        vals.append(float(value.y))
    if hasattr(value, "z"):
        vals.append(float(value.z))
    if hasattr(value, "w"):
        vals.append(float(value.w))
    if not vals:
        try:
            vals = [float(v) for v in value]
        except TypeError:
            vals = [float(value)]
    if count is not None:
        vals = vals[:count]
    return tuple(vals)


def _camera_signature(testbed: Any) -> Tuple[float, ...] | None:
    """Return a compact camera-state signature for cache invalidation."""
    try:
        cam = testbed.scene.camera
    except Exception:
        return None

    vals: List[float] = []
    for attr, count in (
        ("position", 3),
        ("target", 3),
        ("up", 3),
        ("focalLength", 1),
        ("frameHeight", 1),
        ("frameWidth", 1),
        ("nearPlane", 1),
        ("farPlane", 1),
    ):
        try:
            vals.extend(_as_float_tuple(getattr(cam, attr), count))
        except Exception:
            pass
    if not vals:
        return None
    return tuple(round(v, 6) for v in vals)


def _normalize3(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return v / n


def _rotate_vec(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize3(axis)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1.0 - c)


def _to_float3_ctor(fc: Any, vals: np.ndarray) -> Any:
    return fc.float3(float(vals[0]), float(vals[1]), float(vals[2]))


def _key_code(key: Any) -> int | None:
    try:
        return int(key)
    except Exception:
        pass
    value = getattr(key, "value", None)
    if value is not None:
        try:
            return int(value)
        except Exception:
            pass
    try:
        text = str(key)
        if "." in text:
            text = text.rsplit(".", 1)[-1]
        if len(text) == 1:
            return ord(text.upper())
    except Exception:
        pass
    return None


def _install_fps_camera_controls(
    testbed: Any,
    fc: Any,
    look_scale: float,
    look_mode: str,
    move_speed: float,
    fast_multiplier: float,
    slow_multiplier: float,
) -> Callable[[float], bool]:
    """Install Python-side FPS-style mouse look and keyboard movement."""
    # Falcor's key enum uses uppercase ASCII for alpha-numeric keys. Special
    # keys start at 256; Python bindings do not expose every enum value.
    key_space = ord(" ")
    key_w = ord("W")
    key_a = ord("A")
    key_s = ord("S")
    key_d = ord("D")
    key_q = ord("Q")
    key_e = ord("E")
    key_left_shift = 304
    key_left_ctrl = 305
    key_right_shift = 308
    key_right_ctrl = 309

    movement_keys = {
        key_w,
        key_a,
        key_s,
        key_d,
        key_q,
        key_e,
        key_space,
        key_left_shift,
        key_left_ctrl,
        key_right_shift,
        key_right_ctrl,
    }
    state: Dict[str, Any] = {"left": False, "last": None, "keys": set()}

    def reset_last() -> None:
        state["last"] = None

    def rotate_from_delta(delta: np.ndarray) -> bool:
        cam = testbed.scene.camera
        try:
            cam.animated = False
        except Exception:
            pass
        pos = np.array(_as_float_tuple(cam.position, 3), dtype=np.float32)
        target = np.array(_as_float_tuple(cam.target, 3), dtype=np.float32)
        up = _normalize3(np.array(_as_float_tuple(cam.up, 3), dtype=np.float32))
        view = _normalize3(target - pos)
        side = _normalize3(np.cross(view, up))
        if float(np.linalg.norm(side)) < 1e-8:
            return True
        view = _normalize3(_rotate_vec(view, side, float(delta[1])))
        up = _normalize3(_rotate_vec(up, side, float(delta[1])))
        view = _normalize3(_rotate_vec(view, up, float(delta[0])))
        cam.target = _to_float3_ctor(fc, pos + view)
        cam.up = _to_float3_ctor(fc, up)
        return True

    def keyboard_callback(event: Any) -> bool:
        try:
            code = _key_code(getattr(event, "key", None))
            if code is None or code not in movement_keys:
                return False
            event_type = event.type
            if event_type in (fc.KeyboardEvent.Type.KeyPressed, fc.KeyboardEvent.Type.KeyRepeated):
                state["keys"].add(code)
                return True
            if event_type == fc.KeyboardEvent.Type.KeyReleased:
                state["keys"].discard(code)
                return True
        except Exception as exc:
            print(f"[warn] FPS keyboard controls failed: {exc}")
            return False
        return False

    def mouse_callback(event: Any) -> bool:
        try:
            event_type = event.type
            left_button = getattr(fc.MouseButton, "Left")
            if event_type == fc.MouseEvent.Type.ButtonDown and event.button == left_button:
                state["left"] = True
                state["last"] = np.array(_as_float_tuple(event.pos, 2), dtype=np.float32)
                return look_mode == "drag"
            if event_type == fc.MouseEvent.Type.ButtonUp and event.button == left_button:
                state["left"] = False
                reset_last()
                return look_mode == "drag"
            if event_type == fc.MouseEvent.Type.Move:
                cur = np.array(_as_float_tuple(event.pos, 2), dtype=np.float32)
                if bool(np.any(cur < 0.0) or np.any(cur > 1.0)):
                    reset_last()
                    return False
                if look_mode == "drag" and not bool(state["left"]):
                    state["last"] = cur
                    return False
                last = state.get("last")
                state["last"] = cur
                if last is None:
                    return True
                delta = (last - cur) * float(look_scale)
                return rotate_from_delta(delta)
        except Exception as exc:
            print(f"[warn] FPS mouse look failed: {exc}")
            return False
        return False

    def apply_keyboard_movement(dt: float) -> bool:
        keys = state["keys"]
        if not keys:
            return False
        cam = testbed.scene.camera
        try:
            cam.animated = False
        except Exception:
            pass

        pos = np.array(_as_float_tuple(cam.position, 3), dtype=np.float32)
        target = np.array(_as_float_tuple(cam.target, 3), dtype=np.float32)
        up = _normalize3(np.array(_as_float_tuple(cam.up, 3), dtype=np.float32))
        forward = _normalize3(target - pos)
        right = _normalize3(np.cross(forward, up))
        if float(np.linalg.norm(forward)) < 1e-8 or float(np.linalg.norm(right)) < 1e-8:
            return False

        move = np.zeros(3, dtype=np.float32)
        if key_w in keys:
            move += forward
        if key_s in keys:
            move -= forward
        if key_d in keys:
            move += right
        if key_a in keys:
            move -= right
        if key_e in keys or key_space in keys:
            move += up
        if key_q in keys or key_left_ctrl in keys or key_right_ctrl in keys:
            move -= up

        move_norm = float(np.linalg.norm(move))
        if move_norm < 1e-8:
            return False
        move = move / move_norm

        speed = float(move_speed)
        if key_left_shift in keys or key_right_shift in keys:
            speed *= float(fast_multiplier)
        elif key_left_ctrl in keys or key_right_ctrl in keys:
            speed *= float(slow_multiplier)
        step = move * speed * min(max(float(dt), 0.0), 0.05)
        cam.position = _to_float3_ctor(fc, pos + step)
        cam.target = _to_float3_ctor(fc, target + step)
        cam.up = _to_float3_ctor(fc, up)
        return True

    testbed.keyboard_event_callback = keyboard_callback
    testbed.mouse_event_callback = mouse_callback
    return apply_keyboard_movement


def _srgb_u8_from_display_float(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _write_ppm(path: Path, img: np.ndarray) -> None:
    rgb = (_srgb_u8_from_display_float(img) * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii"))
        f.write(rgb.tobytes())


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Falcor interactive GT-vs-Pred lighting demo")
    p.add_argument("--dataset", required=True, help="Dataset root or parametric_tensor.npz")
    p.add_argument("--pred-sh", required=True, help="NPZ produced by prepare_falcor_demo_sh.py")
    p.add_argument("--scene", required=True, help="Falcor .pyscene path")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--mode", choices=["env", "full"], default="env")
    p.add_argument("--full-base", choices=["whitted", "pathtracer", "none"], default="whitted")
    p.add_argument("--pt-spp", type=int, default=1)
    p.add_argument("--pt-bounces", type=int, default=2)
    p.add_argument("--gbuffer-pass", choices=["auto", "raster", "rt"], default="auto")
    p.add_argument(
        "--base-gbuffer-pass",
        choices=["auto", "raster", "rt"],
        default="auto",
        help="GBuffer backend used by the shared direct/shadow base in full mode.",
    )
    p.add_argument(
        "--gbuffer-mode",
        choices=["reuse-first", "realtime"],
        default="reuse-first",
        help="reuse-first is intended for fixed-camera recording; realtime rerenders GBuffer every frame.",
    )
    p.add_argument(
        "--base-mode",
        choices=["reuse-first", "realtime"],
        default="reuse-first",
        help="reuse-first caches the shared direct/shadow base for fixed-camera recording.",
    )
    p.add_argument("--field-res", type=int, default=32)
    p.add_argument("--field-knn", type=int, default=8)
    p.add_argument("--weight-eps", type=float, default=0.1)
    p.add_argument("--grid-chunk", type=int, default=4096)
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--loop-count",
        type=int,
        default=0,
        help="Number of playback loops for window mode; 0 means loop until the window is closed.",
    )
    p.add_argument("--playback-fps", type=float, default=30.0)
    p.add_argument("--no-throttle", action="store_true", default=False)
    p.add_argument("--print-every", type=int, default=30, help="Print one timing line every N frames; 0 disables.")
    p.add_argument("--camera-speed", type=float, default=3.0, help="Python FPS-camera translation speed.")
    p.add_argument("--camera-fast-multiplier", type=float, default=4.0, help="Speed multiplier while holding Shift.")
    p.add_argument("--camera-slow-multiplier", type=float, default=0.25, help="Speed multiplier while holding Ctrl.")
    p.add_argument(
        "--mouse-look-mode",
        choices=["fps", "drag"],
        default="fps",
        help="fps rotates from plain mouse movement; drag requires left-drag.",
    )
    p.add_argument("--mouse-look-scale", type=float, default=1.0, help="Python-side multiplier for camera rotation.")
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--base-scale", type=float, default=0.15)
    p.add_argument(
        "--base-resolution-scale",
        type=float,
        default=1.0,
        help="Render the shared direct/shadow base at a lower resolution in full mode.",
    )
    p.add_argument("--ao-strength", type=float, default=0.0)
    p.add_argument("--display-linear", action="store_true", default=False)
    p.add_argument(
        "--normal-transform",
        choices=["identity", "swap_yz", "swap_xz", "swap_xy", "flip_x", "flip_y", "flip_z"],
        default="identity",
    )
    p.add_argument("--cosine-mode", choices=["irradiance", "radiance"], default="irradiance")
    p.add_argument("--falcor-python-path", default=str(ROOT / "Falcor" / "build" / "linux-gcc" / "bin" / "Debug" / "python"))
    p.add_argument("--headless-smoke", action="store_true", default=False)
    p.add_argument(
        "--headless-timing-only",
        action="store_true",
        default=False,
        help="In headless smoke mode, skip image readback/writes and only save timing JSON.",
    )
    p.add_argument("--smoke-output", default="3_experiments/results/demo/falcor_live_demo/smoke")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    _prepare_falcor_runtime(args.falcor_python_path)
    if args.falcor_python_path and args.falcor_python_path not in sys.path:
        sys.path.insert(0, args.falcor_python_path)

    if "BISTRO_FBX" not in os.environ or not os.environ.get("BISTRO_FBX", ""):
        os.environ["BISTRO_FBX"] = str(
            ROOT / "1_data_generation" / "scenes" / "Bistro_v5_2" / "BistroExterior.fbx"
        )

    import falcor as fc

    dataset_npz = _resolve_dataset_npz(args.dataset)
    scene_path = Path(args.scene)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene not found: {scene_path}")

    data = np.load(dataset_npz, allow_pickle=True)
    tensor = np.asarray(data["tensor"], dtype=np.float32)
    probes = np.asarray(data["probe_positions"], dtype=np.float32)
    metadata = data["metadata"].item() if "metadata" in data else {}
    if "valid_mask" in data:
        valid_mask = np.asarray(data["valid_mask"], dtype=np.float32).reshape(-1)
        valid_idx = np.where(valid_mask > 0.5)[0]
        if valid_idx.size > 0 and valid_idx.size < probes.shape[0]:
            probes = probes[valid_idx]
            tensor = tensor[valid_idx]

    pred_data = np.load(args.pred_sh, allow_pickle=False)
    pred_frames = np.asarray(pred_data["frames"], dtype=np.int64)
    pred_sh = np.asarray(pred_data["sh_model"], dtype=np.float32)
    frame_to_pred = {int(fr): i for i, fr in enumerate(pred_frames.tolist())}
    if pred_sh.shape[1] != probes.shape[0]:
        raise ValueError(
            f"Pred probe count mismatch: pred={pred_sh.shape[1]} dataset={probes.shape[0]}"
        )

    selected = [int(fr) for fr in pred_frames.tolist() if int(fr) >= int(args.frame_start)]
    selected = selected[:: max(1, int(args.frame_step))]
    if args.max_frames is not None:
        selected = selected[: max(0, int(args.max_frames))]
    if not selected:
        raise ValueError("No demo frames selected.")

    bounds_min, bounds_max = _compute_bounds(metadata, probes)
    field_knn, field_w = _build_probe_field_weights(
        probes,
        bounds_min,
        bounds_max,
        grid_res=int(args.field_res),
        k=int(args.field_knn),
        chunk=int(args.grid_chunk),
        weight_eps=float(args.weight_eps),
    )

    testbed_cls = getattr(fc, "Testbed", None)
    if testbed_cls is None and hasattr(fc, "falcor_ext"):
        testbed_cls = getattr(fc.falcor_ext, "Testbed", None)
    if testbed_cls is None:
        raise RuntimeError("Falcor Testbed not found.")

    create_window = not bool(args.headless_smoke)
    testbed = testbed_cls(width=int(args.width), height=int(args.height), create_window=create_window)
    if hasattr(testbed, "show_ui"):
        testbed.show_ui = True
    current_framebuffer_dim = (int(args.width), int(args.height))

    def resize_framebuffer_if_needed(width: int, height: int) -> None:
        nonlocal current_framebuffer_dim
        size = (int(width), int(height))
        if size == current_framebuffer_dim:
            return
        testbed.resize_frame_buffer(size[0], size[1])
        current_framebuffer_dim = size

    if args.gbuffer_pass == "raster":
        gbuffer_candidates = ["GBufferRaster"]
    elif args.gbuffer_pass == "rt":
        gbuffer_candidates = ["GBufferRT"]
    else:
        gbuffer_candidates = ["GBufferRaster", "GBufferRT"]

    gbuffer_graph = None
    gbuffer_pass = ""
    last_gbuffer_exc: Exception | None = None
    for cand in gbuffer_candidates:
        try:
            graph = testbed.create_render_graph(f"FalcorLiveCompareGBuffer_{cand}")
            pass_props = {"samplePattern": "Center", "sampleCount": 1}
            if cand == "GBufferRaster":
                pass_props["allowNonRovUav"] = True
            graph.create_pass(cand, cand, pass_props)
            for out_name in (
                f"{cand}.posW",
                f"{cand}.normW",
                f"{cand}.diffuseOpacity",
                f"{cand}.depth",
                f"{cand}.linearZ",
            ):
                graph.mark_output(out_name)
            gbuffer_graph = graph
            gbuffer_pass = cand
            break
        except Exception as exc:
            last_gbuffer_exc = exc
            if len(gbuffer_candidates) > 1:
                print(f"[warn] {cand} unavailable; falling back to the next GBuffer backend.")
    if gbuffer_graph is None or not gbuffer_pass:
        raise RuntimeError(f"Failed to create a GBuffer graph: {last_gbuffer_exc}") from last_gbuffer_exc
    _set_testbed_graph(testbed, gbuffer_graph)
    testbed.load_scene(str(scene_path))
    try:
        testbed.scene.cameraSpeed = float(args.camera_speed)
    except Exception as exc:
        print(f"[warn] Failed to set cameraSpeed: {exc}")
    apply_camera_controls = _install_fps_camera_controls(
        testbed,
        fc,
        look_scale=float(args.mouse_look_scale),
        look_mode=str(args.mouse_look_mode),
        move_speed=float(args.camera_speed),
        fast_multiplier=float(args.camera_fast_multiplier),
        slow_multiplier=float(args.camera_slow_multiplier),
    )

    base_graph = None
    base_gbuffer_pass = ""
    use_full_base = bool(args.mode == "full" and args.full_base != "none")
    if use_full_base and args.full_base == "pathtracer":
        try:
            base_graph = testbed.create_render_graph("FalcorLiveComparePathTracerBase")
            base_graph.create_pass(
                "PathTracer",
                "PathTracer",
                {
                    "samplesPerPixel": int(max(1, args.pt_spp)),
                    "maxSurfaceBounces": int(max(0, args.pt_bounces)),
                    "useNEE": True,
                },
            )
            base_graph.create_pass("VBufferRT", "VBufferRT", {"samplePattern": "Center", "sampleCount": 1})
            base_graph.add_edge("VBufferRT.vbuffer", "PathTracer.vbuffer")
            base_graph.add_edge("VBufferRT.viewW", "PathTracer.viewW")
            try:
                base_graph.add_edge("VBufferRT.mvec", "PathTracer.mvec")
            except Exception:
                pass
            base_graph.mark_output("PathTracer.color")
        except Exception as exc:
            raise RuntimeError(f"Failed to create PathTracer shared-base graph: {exc}") from exc
    elif use_full_base and args.full_base == "whitted":
        try:
            if args.base_gbuffer_pass == "raster":
                base_gbuffer_candidates = ["GBufferRaster"]
            elif args.base_gbuffer_pass == "rt":
                base_gbuffer_candidates = ["GBufferRT"]
            else:
                # Whitted consumes the packed vbuffer. The local non-ROV
                # GBufferRaster variant is fine for the demo's env pass, but it
                # is not a robust vbuffer producer on this Vulkan stack.
                base_gbuffer_candidates = ["GBufferRT", "GBufferRaster"]
            base_gbuffer_exc: Exception | None = None
            for cand in base_gbuffer_candidates:
                try:
                    probe_graph = testbed.create_render_graph(f"FalcorLiveCompareWhittedProbe_{cand}")
                    probe_graph.create_pass(cand, cand, {"samplePattern": "Center", "sampleCount": 1})
                    base_gbuffer_pass = cand
                    break
                except Exception as exc:
                    base_gbuffer_exc = exc
                    if len(base_gbuffer_candidates) > 1:
                        print(f"[warn] {cand} unavailable for Whitted base; trying the next GBuffer backend.")
            if not base_gbuffer_pass:
                raise RuntimeError(f"Failed to create Whitted base GBuffer pass: {base_gbuffer_exc}")

            base_graph = testbed.create_render_graph("FalcorLiveCompareWhittedBase")
            base_graph.create_pass(
                "WhittedRayTracer",
                "WhittedRayTracer",
                {
                    "maxBounces": int(max(1, args.pt_bounces)),
                    "texLODMode": "RayCones",
                    "rayConeMode": "Unified",
                    "rayConeFilterMode": "AnisotropicWhenRefraction",
                    "useRoughnessToVariance": False,
                },
            )
            base_graph.create_pass(base_gbuffer_pass, base_gbuffer_pass, {"samplePattern": "Center", "sampleCount": 1})
            base_graph.add_edge(f"{base_gbuffer_pass}.posW", "WhittedRayTracer.posW")
            base_graph.add_edge(f"{base_gbuffer_pass}.normW", "WhittedRayTracer.normalW")
            base_graph.add_edge(f"{base_gbuffer_pass}.tangentW", "WhittedRayTracer.tangentW")
            base_graph.add_edge(f"{base_gbuffer_pass}.faceNormalW", "WhittedRayTracer.faceNormalW")
            base_graph.add_edge(f"{base_gbuffer_pass}.texC", "WhittedRayTracer.texC")
            base_graph.add_edge(f"{base_gbuffer_pass}.texGrads", "WhittedRayTracer.texGrads")
            base_graph.add_edge(f"{base_gbuffer_pass}.mtlData", "WhittedRayTracer.mtlData")
            base_graph.add_edge(f"{base_gbuffer_pass}.vbuffer", "WhittedRayTracer.vbuffer")
            base_graph.mark_output("WhittedRayTracer.color")
        except Exception as exc:
            raise RuntimeError(f"Failed to create Whitted shared-base graph: {exc}") from exc

    device = testbed.device
    num_probes = int(probes.shape[0])
    num_cells = int(args.field_res) ** 3
    k_field = int(field_knn.shape[-1])

    def make_field_buffer() -> Any:
        return device.create_structured_buffer(
            struct_size=16,
            element_count=num_cells * 9,
            bind_flags=fc.ResourceBindFlags.ShaderResource | fc.ResourceBindFlags.UnorderedAccess,
        )

    field_gt = make_field_buffer()
    field_pred = make_field_buffer()
    probe_gt = device.create_structured_buffer(
        struct_size=4,
        element_count=num_probes * 27,
        bind_flags=fc.ResourceBindFlags.ShaderResource,
    )
    probe_pred = device.create_structured_buffer(
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
    knn_idx_buf.from_numpy(field_knn.reshape(-1).astype(np.uint32, copy=False))
    knn_w_buf.from_numpy(field_w.reshape(-1).astype(np.float32, copy=False))

    output_tex = device.create_texture(
        format=fc.ResourceFormat.RGBA32Float,
        width=int(args.width),
        height=int(args.height),
        mip_levels=1,
        bind_flags=fc.ResourceBindFlags.UnorderedAccess | fc.ResourceBindFlags.ShaderResource,
    )
    fallback_base = device.create_texture(
        format=fc.ResourceFormat.RGBA32Float,
        width=int(args.width),
        height=int(args.height),
        mip_levels=1,
        bind_flags=fc.ResourceBindFlags.ShaderResource | fc.ResourceBindFlags.UnorderedAccess,
    )
    fallback_base.from_numpy(np.zeros((int(args.height), int(args.width), 4), dtype=np.float32))

    field_build_path = TOOLS / "sh_probe_field_build.cs.slang"
    fb_gt = fc.ComputePass(device, file=field_build_path, cs_entry="main")
    fb_pred = fc.ComputePass(device, file=field_build_path, cs_entry="main")
    for fb, probe_buf, field_buf in ((fb_gt, probe_gt, field_gt), (fb_pred, probe_pred, field_pred)):
        fb.globals.gProbeSH = probe_buf
        fb.globals.gKnnIdx = knn_idx_buf
        fb.globals.gKnnW = knn_w_buf
        fb.globals.gFieldSH = field_buf
        fb.globals.gFieldRes = fc.uint3(int(args.field_res), int(args.field_res), int(args.field_res))
        fb.globals.gNumProbes = int(num_probes)
        fb.globals.gK = int(k_field)

    compare_path = Path(__file__).parent / "sh_probe_compare_live.cs.slang"
    compare = fc.ComputePass(device, file=compare_path, cs_entry="main")
    compare.globals.gFieldSHGT = field_gt
    compare.globals.gFieldSHPred = field_pred
    compare.globals.gOutput = output_tex
    compare.globals.gBoundsMin = fc.float3(float(bounds_min[0]), float(bounds_min[1]), float(bounds_min[2]))
    compare.globals.gBoundsMax = fc.float3(float(bounds_max[0]), float(bounds_max[1]), float(bounds_max[2]))
    compare.globals.gFieldRes = fc.uint3(int(args.field_res), int(args.field_res), int(args.field_res))
    compare.globals.gOutputDim = fc.uint2(int(args.width), int(args.height))
    compare.globals.gSourceDim = fc.uint2(int(args.width), int(args.height))
    compare.globals.gBaseDim = fc.uint2(int(args.width), int(args.height))
    compare.globals.gExposure = float(args.exposure)
    compare.globals.gBaseScale = float(args.base_scale)
    compare.globals.gAOStrength = float(args.ao_strength)
    compare.globals.gMode = 1 if args.mode == "full" else 0
    compare.globals.gToneMap = 0 if bool(args.display_linear) else 1
    compare.globals.gNormalTransform = {
        "identity": 0,
        "swap_yz": 1,
        "swap_xz": 2,
        "swap_xy": 3,
        "flip_x": 4,
        "flip_y": 5,
        "flip_z": 6,
    }[str(args.normal_transform)]
    compare.globals.gCosineMode = {"irradiance": 0, "radiance": 1}[str(args.cosine_mode)]

    hud_text = None
    if create_window:
        try:
            from falcor import ui

            win = ui.Window(
                parent=testbed.screen,
                title="PG-CPL Live Compare",
                position=[12, 12],
                size=[520, 190],
            )
            hud_text = ui.Text(parent=win, text="Initializing...")
        except Exception as exc:
            print(f"[warn] Falcor UI HUD unavailable: {exc}")

    testbed.render_texture = output_tex

    frame_idx = 0
    loops_done = 0
    smoke_dir = Path(args.smoke_output)
    timing_rows: List[Dict[str, float]] = []
    fps_ema = 0.0
    gbuffer_cache: Tuple[Any, Any, Any, Any, Any] | None = None
    base_cache: Any | None = None
    base_dim_cache: Tuple[int, int] | None = None
    cached_camera_sig: Tuple[float, ...] | None = None
    requested_base_res_scale = max(0.05, min(1.0, float(args.base_resolution_scale)))
    base_res_scale = requested_base_res_scale
    if create_window and base_res_scale != 1.0:
        print(
            "[warn] --base-resolution-scale is ignored in window mode because "
            "Falcor Testbed.resize_frame_buffer() resizes the actual window."
        )
        base_res_scale = 1.0
    base_width = max(1, int(round(int(args.width) * base_res_scale)))
    base_height = max(1, int(round(int(args.height) * base_res_scale)))
    last_control_time = time.perf_counter()
    while True:
        if create_window and bool(testbed.should_close):
            break
        if frame_idx >= len(selected):
            loops_done += 1
            if bool(args.headless_smoke):
                break
            if int(args.loop_count) > 0 and loops_done >= int(args.loop_count):
                break
            frame_idx = 0

        frame = int(selected[frame_idx])
        total0 = time.perf_counter()
        control_now = total0
        camera_moved_by_controls = apply_camera_controls(control_now - last_control_time)
        last_control_time = control_now
        t_sec = float(frame) / float(args.fps)
        current_camera_sig = _camera_signature(testbed)
        camera_changed = (
            bool(camera_moved_by_controls)
            or (
                cached_camera_sig is not None
                and current_camera_sig is not None
                and current_camera_sig != cached_camera_sig
            )
        )
        if camera_changed:
            gbuffer_cache = None
            base_cache = None

        render_gbuffer = not (args.gbuffer_mode == "reuse-first" and gbuffer_cache is not None)
        if render_gbuffer:
            _set_clock(testbed, t_sec)
            _set_testbed_graph(testbed, gbuffer_graph)
            t0 = time.perf_counter()
            resize_framebuffer_if_needed(int(args.width), int(args.height))
            testbed.frame()
            gbuffer_ms = (time.perf_counter() - t0) * 1000.0
            pos_tex = gbuffer_graph.get_output(f"{gbuffer_pass}.posW")
            norm_tex = gbuffer_graph.get_output(f"{gbuffer_pass}.normW")
            alb_tex = gbuffer_graph.get_output(f"{gbuffer_pass}.diffuseOpacity")
            dep_tex = gbuffer_graph.get_output(f"{gbuffer_pass}.depth")
            lin_tex = gbuffer_graph.get_output(f"{gbuffer_pass}.linearZ")
            if args.gbuffer_mode == "reuse-first":
                gbuffer_cache = (pos_tex, norm_tex, alb_tex, dep_tex, lin_tex)
            cached_camera_sig = _camera_signature(testbed)
        else:
            gbuffer_ms = 0.0
            assert gbuffer_cache is not None
            pos_tex, norm_tex, alb_tex, dep_tex, lin_tex = gbuffer_cache

        base_tex = fallback_base
        base_dim = (int(args.width), int(args.height))
        base_ms = 0.0
        if use_full_base and base_graph is not None:
            render_base = not (args.base_mode == "reuse-first" and base_cache is not None)
            if render_base:
                _set_clock(testbed, t_sec)
                _set_testbed_graph(testbed, base_graph)
                tb = time.perf_counter()
                resize_framebuffer_if_needed(int(base_width), int(base_height))
                testbed.frame()
                base_ms = (time.perf_counter() - tb) * 1000.0
                if args.full_base == "pathtracer":
                    base_tex = base_graph.get_output("PathTracer.color")
                elif args.full_base == "whitted":
                    base_tex = base_graph.get_output("WhittedRayTracer.color")
                base_dim = (base_width, base_height)
                if args.base_mode == "reuse-first":
                    base_cache = base_tex
                    base_dim_cache = base_dim
            else:
                base_tex = base_cache
                base_dim = base_dim_cache if base_dim_cache is not None else (base_width, base_height)
            resize_framebuffer_if_needed(int(args.width), int(args.height))

        t_upload = time.perf_counter()
        gt_sh = np.asarray(tensor[:, frame, :], dtype=np.float32)
        pred = np.asarray(pred_sh[frame_to_pred[frame]], dtype=np.float32)
        probe_gt.from_numpy(gt_sh.reshape(-1))
        probe_pred.from_numpy(pred.reshape(-1))
        upload_ms = (time.perf_counter() - t_upload) * 1000.0

        t_field = time.perf_counter()
        fb_gt.execute(threads_x=num_cells, threads_y=1, threads_z=1)
        fb_pred.execute(threads_x=num_cells, threads_y=1, threads_z=1)
        field_ms = (time.perf_counter() - t_field) * 1000.0

        compare.globals.gPosW = pos_tex
        compare.globals.gNormW = norm_tex
        compare.globals.gAlbedo = alb_tex
        compare.globals.gDepth = dep_tex
        compare.globals.gLinearZ = lin_tex
        compare.globals.gBaseColor = base_tex
        compare.globals.gBaseDim = fc.uint2(int(base_dim[0]), int(base_dim[1]))
        compare.globals.gMode = 1 if args.mode == "full" else 0

        t_shade = time.perf_counter()
        compare.execute(threads_x=int(args.width), threads_y=int(args.height))
        shade_ms = (time.perf_counter() - t_shade) * 1000.0

        readback_ms = 0.0
        if bool(args.headless_smoke) and not bool(args.headless_timing_only):
            tr = time.perf_counter()
            img = np.asarray(output_tex.to_numpy(), dtype=np.float32)
            readback_ms = (time.perf_counter() - tr) * 1000.0
            smoke_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(smoke_dir / f"frame_{frame:04d}_{args.mode}.npz", image=img)
            _write_ppm(smoke_dir / f"frame_{frame:04d}_{args.mode}.ppm", img)

        present_ms = 0.0
        if create_window:
            graph_cleared = _clear_testbed_graph(testbed)
            if hud_text is not None:
                total_now = (time.perf_counter() - total0) * 1000.0
                fps_now = 1000.0 / max(total_now, 1e-6)
                fps_ema = fps_now if fps_ema <= 0.0 else 0.9 * fps_ema + 0.1 * fps_now
                base_label = "off"
                if use_full_base:
                    base_label = (
                        f"{args.full_base}/{args.base_mode}{' cached' if base_ms <= 0.0 else ''} "
                        f"{base_dim[0]}x{base_dim[1]}"
                    )
                hud_text.text = (
                    "LEFT: GT SH lighting    RIGHT: compressed Pred SH\n"
                    f"Mode: {'SH + shared direct/shadow base' if args.mode == 'full' else 'SH/environment only'}\n"
                    f"GBuffer: {gbuffer_pass}/{args.gbuffer_mode}{' cached' if not render_gbuffer else ''}{' camera update' if camera_changed else ''}\n"
                    f"Base: {base_label}\n"
                    f"Frame: {frame:04d}  Dataset time: {t_sec:6.2f}s\n"
                    f"FPS: {fps_ema:6.1f}\n"
                    f"GBuffer: {gbuffer_ms:6.2f} ms   Base: {base_ms:6.2f} ms\n"
                    f"Upload:  {upload_ms:6.2f} ms   Field: {field_ms:6.2f} ms\n"
                    f"Shade:   {shade_ms:6.2f} ms"
                )
            tp = time.perf_counter()
            testbed.frame()
            present_ms = (time.perf_counter() - tp) * 1000.0
            if not graph_cleared:
                _set_testbed_graph(testbed, gbuffer_graph)

        total_ms = (time.perf_counter() - total0) * 1000.0
        row = {
            "frame": float(frame),
            "gbuffer_ms": float(gbuffer_ms),
            "base_ms": float(base_ms),
            "upload_ms": float(upload_ms),
            "field_ms": float(field_ms),
            "shade_ms": float(shade_ms),
            "readback_ms": float(readback_ms),
            "present_ms": float(present_ms),
            "total_ms": float(total_ms),
            "fps": float(1000.0 / max(total_ms, 1e-6)),
        }
        timing_rows.append(row)
        print_every = max(0, int(args.print_every))
        if print_every > 0 and (len(timing_rows) == 1 or len(timing_rows) % print_every == 0):
            print(
                f"frame={frame:04d} mode={args.mode} fps={row['fps']:.1f} "
                f"gbuf={gbuffer_ms:.2f} base={base_ms:.2f} field={field_ms:.2f} shade={shade_ms:.2f}"
            )

        frame_idx += 1
        if not bool(args.no_throttle) and float(args.playback_fps) > 0.0:
            target = 1.0 / float(args.playback_fps)
            elapsed = time.perf_counter() - total0
            if elapsed < target:
                time.sleep(target - elapsed)

    if timing_rows:
        out_dir = smoke_dir if bool(args.headless_smoke) else Path("3_experiments/results/demo/falcor_live_demo")
        out_dir.mkdir(parents=True, exist_ok=True)
        timings_path = out_dir / f"live_demo_timings_{args.mode}.json"
        with timings_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": str(args.mode),
                    "full_base": str(args.full_base),
                    "base_mode": str(args.base_mode),
                    "gbuffer_pass": str(gbuffer_pass),
                    "gbuffer_mode": str(args.gbuffer_mode),
                    "base_gbuffer_pass": str(base_gbuffer_pass),
                    "base_scale": float(args.base_scale),
                    "base_resolution_scale": float(base_res_scale),
                    "requested_base_resolution_scale": float(requested_base_res_scale),
                    "exposure": float(args.exposure),
                    "frames": timing_rows,
                    "mean_total_ms": float(np.mean([r["total_ms"] for r in timing_rows])),
                    "mean_fps": float(np.mean([r["fps"] for r in timing_rows])),
                },
                f,
                indent=2,
            )
        print(f"Saved timing report: {timings_path}")


if __name__ == "__main__":
    main()
