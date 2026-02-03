import json
from pathlib import Path

import numpy as np
import pytest

from tools.blender.scene_export import (
    build_emissive_object,
    build_scene_payload,
    infer_color_from_name,
    write_scene_json,
    write_trajectory_npy,
)


def test_infer_color_from_name():
    assert infer_color_from_name("BallRed") == [1.0, 0.0, 0.0]
    assert infer_color_from_name("green_light") == [0.0, 1.0, 0.0]
    assert infer_color_from_name("Blue") == [0.0, 0.0, 1.0]
    assert infer_color_from_name("neutral") == [1.0, 1.0, 1.0]


def test_build_emissive_object_defaults():
    obj = build_emissive_object("BallRed", "red.npy")
    assert obj["name"] == "BallRed"
    assert obj["trajectory"] == "red.npy"
    assert obj["color"] == [20.0, 0.0, 0.0]
    assert obj["radius"] > 0.0


def test_build_emissive_object_intensity_override():
    obj = build_emissive_object("BallRed", "red.npy", color=[1.0, 0.0, 0.0], intensity=2.5, radius=0.1)
    assert obj["color"] == [2.5, 0.0, 0.0]
    assert obj["radius"] == 0.1


def test_build_emissive_object_invalid_color():
    with pytest.raises(ValueError):
        build_emissive_object("BallRed", "red.npy", color=[1.0, 0.0])


def test_build_scene_payload_with_probes():
    emissive = [build_emissive_object("BallRed", "red.npy")]
    payload = build_scene_payload(
        scene_path="scene.pyscene",
        frames=10,
        fps=24.0,
        emissive_objects=emissive,
        probes_file="probes.npy",
        render={"spp": 16},
        lighting_mode="analytic_lights",
    )
    assert payload["scene"] == "scene.pyscene"
    assert payload["frames"] == 10
    assert payload["fps"] == 24.0
    assert payload["probes"]["file"] == "probes.npy"
    assert payload["render"]["spp"] == 16
    assert payload["lighting_mode"] == "analytic_lights"


def test_write_scene_json_and_trajectory(tmp_path: Path):
    traj = np.zeros((3, 3), dtype=np.float32)
    traj_path = write_trajectory_npy(traj, tmp_path / "traj.npy")
    assert traj_path.exists()

    payload = build_scene_payload(
        scene_path="scene.pyscene",
        frames=3,
        fps=30.0,
        emissive_objects=[build_emissive_object("BallRed", traj_path.name)],
    )
    json_path = write_scene_json(payload, tmp_path / "scene.json")
    assert json_path.exists()

    data = json.loads(json_path.read_text())
    assert data["frames"] == 3
    assert data["emissive_objects"][0]["trajectory"] == "traj.npy"


def test_write_trajectory_invalid_shape(tmp_path: Path):
    bad_traj = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        write_trajectory_npy(bad_traj, tmp_path / "bad.npy")
