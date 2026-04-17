#!/usr/bin/env python3
"""Bake offline G-Buffer supervision samples for Bistro (Falcor)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_FALCOR_UTILS = _ROOT / "1_data_generation" / "falcor" / "utils"
for p in (str(_FALCOR_UTILS),):
    if p not in sys.path:
        sys.path.insert(0, p)

from falcor_render import build_testbed, get_scene_bounds, load_scene  # noqa: E402
from gbuffer_baker import (  # noqa: E402
    bake_gbuffer_dataset,
    build_light_config_provider_from_npz,
)
from temporal_dataset_common import parse_frame_indices  # noqa: E402


def _sync_bistro_env(args: argparse.Namespace) -> None:
    os.environ["BISTRO_FBX"] = str(Path(args.bistro_fbx).resolve())
    os.environ["BISTRO_NUM_FRAMES"] = str(int(args.num_frames))
    os.environ["BISTRO_FPS"] = str(float(args.fps))
    os.environ["BISTRO_BALL_INTENSITY"] = str(float(args.ball_intensity))
    os.environ["BISTRO_HEIGHT_BASE_RATIO"] = str(float(args.height_base_ratio))
    os.environ["BISTRO_HEIGHT_AMP_RATIO"] = str(float(args.height_amp_ratio))
    os.environ["BISTRO_AXIS_MARGIN_RATIO"] = str(float(args.axis_margin_ratio))
    os.environ["BISTRO_ORTH_OFFSET_RATIO"] = str(float(args.orth_offset_ratio))

    if args.ball_radius is not None:
        os.environ["BISTRO_BALL_RADIUS"] = str(float(args.ball_radius))
    elif "BISTRO_BALL_RADIUS" in os.environ and os.environ["BISTRO_BALL_RADIUS"] == "":
        del os.environ["BISTRO_BALL_RADIUS"]


def _maybe_set_bistro_bounds(args: argparse.Namespace) -> None:
    if not bool(args.auto_bounds):
        return
    if os.environ.get("BISTRO_BOUNDS_MIN") and os.environ.get("BISTRO_BOUNDS_MAX"):
        return

    base_scene = str(Path(args.base_scene).resolve())
    testbed, _graph, _falcor = build_testbed(
        width=1,
        height=1,
        spp=1,
        falcor_python_path=args.falcor_python_path,
        fixed_seed=1,
        use_russian_roulette=False,
    )
    scene = load_scene(testbed, base_scene)
    bounds = get_scene_bounds(scene)
    os.environ["BISTRO_BOUNDS_MIN"] = ",".join([f"{float(v):.6f}" for v in bounds[0].tolist()])
    os.environ["BISTRO_BOUNDS_MAX"] = ",".join([f"{float(v):.6f}" for v in bounds[1].tolist()])


def _scene_setup_hook(args: argparse.Namespace):
    def _hook(scene, _testbed) -> None:
        scene.renderSettings.useEnvLight = bool(args.use_env_light)
        if bool(args.use_analytic_lights):
            scene.renderSettings.useAnalyticLights = True
            scene.renderSettings.useEmissiveLights = False
        else:
            # Bistro emissive sphere pipeline default.
            scene.renderSettings.useAnalyticLights = False
            scene.renderSettings.useEmissiveLights = True
        scene.animated = True
        scene.loopAnimations = False

    return _hook


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bake Bistro G-Buffer dataset for differentiable image-loss supervision."
    )
    default_scene = _ROOT / "1_data_generation" / "falcor" / "scenes" / "bistro_exterior_emissive_spheres.pyscene"
    default_base_scene = _ROOT / "1_data_generation" / "scenes" / "Bistro_v5_2" / "BistroExterior.pyscene"
    default_bistro_fbx = _ROOT / "1_data_generation" / "scenes" / "Bistro_v5_2" / "BistroExterior.fbx"
    default_light_npz = _ROOT / "1_data_generation" / "output" / "bistro_clean_v2" / "parametric_tensor.npz"
    default_output = _ROOT / "1_data_generation" / "output" / "bistro_clean_v2_gbuffer"

    parser.add_argument("--output-dir", type=str, default=str(default_output))
    parser.add_argument("--scene", type=str, default=str(default_scene))
    parser.add_argument("--base-scene", type=str, default=str(default_base_scene))
    parser.add_argument("--bistro-fbx", type=str, default=str(default_bistro_fbx))
    parser.add_argument("--falcor-python-path", type=str, default=None)

    parser.add_argument("--num-frames", type=int, default=600)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frame-indices", type=str, default="")
    parser.add_argument("--frame-step", type=int, default=None)

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--accum-frames", type=int, default=1)
    parser.add_argument("--fixed-seed", type=int, default=1)

    parser.add_argument("--camera-mode", type=str, default="scene", choices=["scene", "orbit"])
    parser.add_argument("--num-cameras", type=int, default=1)
    parser.add_argument("--include-depth", action="store_true")
    parser.add_argument("--strict-albedo", action="store_true", default=True)
    parser.add_argument("--no-strict-albedo", action="store_false", dest="strict_albedo")

    parser.add_argument("--light-config-npz", type=str, default=str(default_light_npz))
    parser.add_argument("--light-config-key", type=str, default="light_configs")
    parser.add_argument("--light-mask-key", type=str, default="light_mask")
    parser.add_argument("--strict-light-frame-map", action="store_true", default=True)
    parser.add_argument("--no-strict-light-frame-map", action="store_false", dest="strict_light_frame_map")
    parser.add_argument("--default-light-slots", type=int, default=3)
    parser.add_argument("--default-light-dim", type=int, default=12)

    parser.add_argument("--output-format", type=str, default="npz", choices=["pt", "npz"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--auto-bounds", action="store_true", default=True)
    parser.add_argument("--no-auto-bounds", action="store_false", dest="auto_bounds")

    parser.add_argument("--ball-radius", type=float, default=None)
    parser.add_argument("--ball-intensity", type=float, default=20.0)
    parser.add_argument("--height-base-ratio", type=float, default=0.08)
    parser.add_argument("--height-amp-ratio", type=float, default=0.06)
    parser.add_argument("--axis-margin-ratio", type=float, default=0.15)
    parser.add_argument("--orth-offset-ratio", type=float, default=0.12)

    parser.add_argument("--use-env-light", action="store_true")
    parser.add_argument("--use-analytic-lights", action="store_true")
    args = parser.parse_args()

    frame_indices = parse_frame_indices(
        num_frames=int(args.num_frames),
        frame_indices=str(args.frame_indices),
        frame_step=args.frame_step,
    )
    if len(frame_indices) == 0:
        raise ValueError("No frame indices selected.")

    _sync_bistro_env(args)
    _maybe_set_bistro_bounds(args)

    light_provider = None
    light_npz = str(args.light_config_npz).strip()
    if light_npz:
        npz_path = Path(light_npz)
        if not npz_path.exists():
            raise FileNotFoundError(f"--light-config-npz not found: {npz_path}")
        light_provider = build_light_config_provider_from_npz(
            npz_path=npz_path,
            light_configs_key=str(args.light_config_key),
            light_mask_key=str(args.light_mask_key),
            strict_frame_map=bool(args.strict_light_frame_map),
        )

    default_light_params = np.zeros(
        (max(1, int(args.default_light_slots)), max(1, int(args.default_light_dim))),
        dtype=np.float32,
    )
    default_light_mask = np.ones((default_light_params.shape[0],), dtype=np.float32)

    metadata = bake_gbuffer_dataset(
        output_dir=args.output_dir,
        scene_path=args.scene,
        frame_indices=frame_indices,
        fps=float(args.fps),
        width=int(args.width),
        height=int(args.height),
        spp=int(args.spp),
        accum_frames=int(args.accum_frames),
        fixed_seed=int(args.fixed_seed),
        falcor_python_path=args.falcor_python_path,
        camera_mode=str(args.camera_mode),
        num_cameras=int(args.num_cameras),
        light_provider=light_provider,
        scene_setup_hook=_scene_setup_hook(args),
        output_format=str(args.output_format),
        include_depth=bool(args.include_depth),
        strict_albedo=bool(args.strict_albedo),
        default_light_params=default_light_params,
        default_light_mask=default_light_mask,
        overwrite=bool(args.overwrite),
        progress=(not bool(args.no_progress)),
    )

    print(f"[bistro-gbuffer] done. output_dir={metadata['output_dir']}", flush=True)
    print(f"[bistro-gbuffer] samples={metadata['num_samples']}", flush=True)
    print(f"[bistro-gbuffer] metadata={Path(args.output_dir).resolve() / 'metadata.json'}", flush=True)
    print(f"[bistro-gbuffer] manifest={Path(args.output_dir).resolve() / 'samples_manifest.txt'}", flush=True)


if __name__ == "__main__":
    main()
