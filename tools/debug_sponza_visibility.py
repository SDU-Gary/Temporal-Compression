#!/usr/bin/env python3
"""Diagnose Sponza camera visibility through Falcor GBufferRT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FALCOR_UTILS = ROOT / "1_data_generation" / "falcor" / "utils"
if str(FALCOR_UTILS) not in sys.path:
    sys.path.insert(0, str(FALCOR_UTILS))

from falcor_render import get_scene_bounds, load_scene  # noqa: E402
from falcor_render import _ensure_falcor_import  # noqa: E402


def _parse_vec3(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 3:
        raise ValueError(f"Expected x,y,z, got {text!r}")
    return np.asarray(vals, dtype=np.float32)


def _get_output(graph, name: str) -> np.ndarray:
    try:
        tex = graph.get_output(name)
    except Exception:
        tex = graph.getOutput(name)
    if tex is None:
        raise RuntimeError(f"Missing graph output: {name}")
    if hasattr(tex, "to_numpy"):
        return np.asarray(tex.to_numpy(), dtype=np.float32)
    return np.asarray(tex.toNumpy(), dtype=np.float32)


def _make_graph(testbed, falcor):
    try:
        graph = testbed.create_render_graph("SponzaVisibility")
    except Exception:
        graph = testbed.createRenderGraph("SponzaVisibility")

    opts = {"samplePattern": "Center", "sampleCount": 1, "useAlphaTest": True}
    try:
        graph.create_pass("GBufferRT", "GBufferRT", opts)
        graph.mark_output("GBufferRT.posW")
        graph.mark_output("GBufferRT.normW")
        graph.mark_output("GBufferRT.depth")
    except Exception:
        gbuffer = falcor.createPass("GBufferRT", opts)
        graph.addPass(gbuffer, "GBufferRT")
        graph.markOutput("GBufferRT.posW")
        graph.markOutput("GBufferRT.normW")
        graph.markOutput("GBufferRT.depth")

    try:
        testbed.render_graph = graph
    except Exception:
        testbed.setRenderGraph(graph)
    return graph


def _set_camera(scene, falcor, pos: np.ndarray, target: np.ndarray, up: np.ndarray, focal: float) -> None:
    cam = scene.camera
    cam.position = falcor.float3(float(pos[0]), float(pos[1]), float(pos[2]))
    cam.target = falcor.float3(float(target[0]), float(target[1]), float(target[2]))
    cam.up = falcor.float3(float(up[0]), float(up[1]), float(up[2]))
    cam.focalLength = float(focal)
    for attr, val in (("nearPlane", 0.1), ("farPlane", 10000.0), ("aspectRatio", 16.0 / 9.0)):
        try:
            setattr(cam, attr, float(val))
        except Exception:
            pass


def _valid_stats(pos: np.ndarray, norm: np.ndarray, depth: np.ndarray) -> dict[str, float]:
    pos3 = pos[..., :3]
    norm3 = norm[..., :3]
    finite = np.isfinite(pos3).all(axis=-1) & np.isfinite(norm3).all(axis=-1) & np.isfinite(depth)
    nonzero_pos = np.linalg.norm(pos3, axis=-1) > 1e-6
    nonzero_norm = np.linalg.norm(norm3, axis=-1) > 1e-6
    # Falcor depth conventions differ by pass/version. Count either finite
    # nonzero world position or finite normal as a surface hit.
    valid = finite & (nonzero_pos | nonzero_norm)
    if valid.any():
        d = depth[valid]
        p = pos3[valid]
        return {
            "valid_ratio": float(valid.mean()),
            "depth_min": float(np.nanmin(d)),
            "depth_max": float(np.nanmax(d)),
            "pos_min_x": float(np.nanmin(p[:, 0])),
            "pos_max_x": float(np.nanmax(p[:, 0])),
            "pos_min_y": float(np.nanmin(p[:, 1])),
            "pos_max_y": float(np.nanmax(p[:, 1])),
            "pos_min_z": float(np.nanmin(p[:, 2])),
            "pos_max_z": float(np.nanmax(p[:, 2])),
        }
    return {
        "valid_ratio": 0.0,
        "depth_min": float("nan"),
        "depth_max": float("nan"),
        "pos_min_x": float("nan"),
        "pos_max_x": float("nan"),
        "pos_min_y": float("nan"),
        "pos_max_y": float("nan"),
        "pos_min_z": float("nan"),
        "pos_max_z": float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=str(ROOT / "1_data_generation/falcor/scenes/sponza_sun_point.pyscene"))
    parser.add_argument("--falcor-python-path", default=str(ROOT / "Falcor/build/linux-gcc/bin/Debug/python"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--focal", type=float, default=28.0)
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help="Candidate as pos:target[:up], each vec is x,y,z. Can be repeated.",
    )
    args = parser.parse_args()

    falcor = _ensure_falcor_import(args.falcor_python_path)
    testbed = falcor.Testbed(width=int(args.width), height=int(args.height), create_window=False)
    graph = _make_graph(testbed, falcor)
    scene = load_scene(testbed, args.scene)

    try:
        testbed.resize_frame_buffer(int(args.width), int(args.height))
    except Exception:
        testbed.resizeFrameBuffer(int(args.width), int(args.height))
    try:
        testbed.clock.pause()
    except Exception:
        pass

    bounds = get_scene_bounds(scene)
    print(f"scene_bounds min={bounds[0].tolist()} max={bounds[1].tolist()}", flush=True)

    default_cases = [
        ("default_scene", np.array([-2200, 650, 0], np.float32), np.array([0, 550, 0], np.float32), np.array([0, 1, 0], np.float32)),
        ("front_z", np.array([0, 450, 1200], np.float32), np.array([0, 450, 0], np.float32), np.array([0, 1, 0], np.float32)),
        ("back_z", np.array([0, 450, -1200], np.float32), np.array([0, 450, 0], np.float32), np.array([0, 1, 0], np.float32)),
        ("left_x", np.array([-1800, 550, 0], np.float32), np.array([0, 550, 0], np.float32), np.array([0, 1, 0], np.float32)),
        ("right_x", np.array([1800, 550, 0], np.float32), np.array([0, 550, 0], np.float32), np.array([0, 1, 0], np.float32)),
        ("inside_z", np.array([0, 350, 650], np.float32), np.array([0, 350, 0], np.float32), np.array([0, 1, 0], np.float32)),
        ("inside_x", np.array([-900, 350, 0], np.float32), np.array([900, 350, 0], np.float32), np.array([0, 1, 0], np.float32)),
        ("top_down", np.array([0, 2400, 0], np.float32), np.array([0, 500, 0], np.float32), np.array([0, 0, -1], np.float32)),
    ]
    custom_cases = []
    for idx, spec in enumerate(args.camera):
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Bad --camera spec: {spec}")
        pos = _parse_vec3(parts[0])
        target = _parse_vec3(parts[1])
        up = _parse_vec3(parts[2]) if len(parts) == 3 else np.array([0, 1, 0], np.float32)
        custom_cases.append((f"custom_{idx}", pos, target, up))
    cases = custom_cases or default_cases

    for name, pos, target, up in cases:
        _set_camera(scene, falcor, pos, target, up, float(args.focal))
        for _ in range(3):
            testbed.frame()
        pos_img = _get_output(graph, "GBufferRT.posW")
        norm_img = _get_output(graph, "GBufferRT.normW")
        depth_img = _get_output(graph, "GBufferRT.depth")
        stats = _valid_stats(pos_img, norm_img, depth_img)
        print(
            f"{name}: pos={pos.tolist()} target={target.tolist()} up={up.tolist()} "
            f"valid={stats['valid_ratio']:.4f} "
            f"depth=[{stats['depth_min']:.4g},{stats['depth_max']:.4g}] "
            f"pos_x=[{stats['pos_min_x']:.1f},{stats['pos_max_x']:.1f}] "
            f"pos_y=[{stats['pos_min_y']:.1f},{stats['pos_max_y']:.1f}] "
            f"pos_z=[{stats['pos_min_z']:.1f},{stats['pos_max_z']:.1f}]",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
