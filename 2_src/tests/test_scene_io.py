from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCENE_UTILS = ROOT / "1_data_generation" / "falcor" / "utils"
if str(SCENE_UTILS) not in sys.path:
    sys.path.insert(0, str(SCENE_UTILS))

from scene_io import apply_scene_overrides, load_emissive_objects, load_scene_json


def test_load_scene_json_resolves_paths(tmp_path: Path):
    traj = np.zeros((2, 3), dtype=np.float32)
    traj_path = tmp_path / "traj.npy"
    np.save(traj_path, traj)

    probes_path = tmp_path / "probes.npy"
    np.save(probes_path, traj)

    json_path = tmp_path / "scene.json"
    json_path.write_text(
        """
{
  "scene": "scene.pyscene",
  "emissive_objects": [{"name": "BallRed", "trajectory": "traj.npy"}],
  "probes": {"file": "probes.npy"}
}
"""
    )

    cfg = load_scene_json(json_path)
    assert cfg["scene"].endswith("scene.pyscene")
    assert Path(cfg["emissive_objects"][0]["trajectory"]).is_absolute()
    assert Path(cfg["probes"]["file"]).is_absolute()


def test_apply_scene_overrides_sets_defaults(tmp_path: Path):
    cfg = {
        "scene": str(tmp_path / "scene.pyscene"),
        "frames": 10,
        "fps": 24.0,
        "render": {"spp": 16, "cube_res": 8},
        "probes": {"file": str(tmp_path / "probes.npy")},
        "lighting_mode": "analytic_lights",
        "emissive_objects": [],
    }
    args = SimpleNamespace(
        scene="default.pyscene",
        num_frames=600,
        fps=30.0,
        sh_mode="cubemap",
        cube_res=16,
        num_sh_samples=64,
        spp=128,
        accum_frames=1,
        probe_file=None,
        use_analytic_lights=False,
        use_emissive_lights=False,
    )
    defaults = {"scene": "default.pyscene", "num_frames": 600, "fps": 30.0}
    apply_scene_overrides(args, cfg, defaults)
    assert args.scene == cfg["scene"]
    assert args.num_frames == 10
    assert args.fps == 24.0
    assert args.spp == 16
    assert args.cube_res == 8
    assert args.probe_file == cfg["probes"]["file"]
    assert args.use_analytic_lights is True
    assert args.use_emissive_lights is False


def test_apply_scene_overrides_defaults_to_emissive(tmp_path: Path):
    cfg = {
        "scene": str(tmp_path / "scene.pyscene"),
        "frames": 5,
        "fps": 30.0,
        "emissive_objects": [{"name": "BallRed", "trajectory": "traj.npy"}],
    }
    args = SimpleNamespace(
        scene="default.pyscene",
        num_frames=600,
        fps=30.0,
        sh_mode="cubemap",
        cube_res=16,
        num_sh_samples=64,
        spp=128,
        accum_frames=1,
        probe_file=None,
        use_analytic_lights=False,
        use_emissive_lights=False,
    )
    defaults = {"scene": "default.pyscene", "num_frames": 600, "fps": 30.0}
    apply_scene_overrides(args, cfg, defaults)
    assert args.use_emissive_lights is True
    assert args.use_analytic_lights is False


def test_load_emissive_objects_truncates_frames(tmp_path: Path):
    traj_short = np.zeros((5, 3), dtype=np.float32)
    traj_long = np.zeros((8, 3), dtype=np.float32)
    short_path = tmp_path / "short.npy"
    long_path = tmp_path / "long.npy"
    np.save(short_path, traj_short)
    np.save(long_path, traj_long)

    cfg = {
        "emissive_objects": [
            {"name": "A", "trajectory": str(short_path), "color": [1, 0, 0], "radius": 0.2},
            {"name": "B", "trajectory": str(long_path), "color": [0, 1, 0], "radius": 0.3},
        ]
    }
    light_trajs, colors, radii, frames = load_emissive_objects(cfg, 10.0, 0.5, num_frames=10)
    assert frames == 5
    assert len(light_trajs) == 2
    assert light_trajs[0].shape[0] == 5
    assert radii == [0.2, 0.3]
    assert colors[0].tolist() == [1.0, 0.0, 0.0]


def test_load_emissive_objects_missing_raises():
    with pytest.raises(ValueError):
        load_emissive_objects({}, 10.0, 0.5, num_frames=5)
