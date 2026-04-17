from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = _load_module(
    "temporal_dataset_common_test",
    ROOT / "1_data_generation" / "falcor" / "utils" / "temporal_dataset_common.py",
)


def test_parse_frame_indices_prefers_explicit_list() -> None:
    out = common.parse_frame_indices(10, "5, 2, 20, -1, 2", frame_step=3)
    assert out == [2, 5]


def test_parse_frame_indices_uses_step_when_no_list() -> None:
    out = common.parse_frame_indices(9, "", frame_step=4)
    assert out == [0, 4, 8]


def test_parse_frame_indices_defaults_full_range() -> None:
    out = common.parse_frame_indices(4, "", frame_step=None)
    assert out == [0, 1, 2, 3]


def test_write_temporal_metadata(tmp_path: Path) -> None:
    payload = {"num_frames": 10, "frame_indices": [0, 2, 4]}
    out_path = common.write_temporal_metadata(tmp_path / "metadata.json", payload)
    assert out_path.exists()
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["num_frames"] == 10
    assert loaded["frame_indices"] == [0, 2, 4]


def test_resolve_num_samples() -> None:
    assert common.resolve_num_samples("dir", cube_res=16, num_sh_samples=512) == 512
    assert common.resolve_num_samples("cubemap", cube_res=16, num_sh_samples=512) == 1536


def test_build_base_temporal_metadata() -> None:
    meta = common.build_base_temporal_metadata(
        scene="dummy_scene.pyscene",
        num_probes=10,
        num_configs=5,
        num_frames=100,
        fps=25.0,
        frame_indices=[0, 2, 4],
        num_lights=3,
        light_descriptor_dim=12,
        sh_mode="cubemap",
        cube_res=8,
        num_sh_samples=128,
        num_samples=384,
        spp=64,
        spp_per_frame=16,
        accum_frames=4,
        radiance_clamp=0.0,
        bounds_min=[0.0, 1.0, 2.0],
        bounds_max=[3.0, 4.0, 5.0],
        duration_sec=3.96,
    )
    assert meta["engine"] == "Falcor"
    assert meta["num_probes"] == 10
    assert meta["frame_indices"] == [0, 2, 4]
    assert meta["num_samples"] == 384
    assert "generation_date" in meta


def test_bake_temporal_tensor_subset() -> None:
    probes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    out = common.bake_temporal_tensor(
        probes=probes,
        frame_indices=[0, 2],
        num_lights=1,
        light_descriptor_dim=12,
        render_probe_sh=lambda probe: np.full((27,), float(probe[0] + 1.0), dtype=np.float32),
        per_frame_setup=lambda frame_idx, out_idx: np.array(
            [[float(frame_idx)] + [0.0] * 11],
            dtype=np.float32,
        ),
        valid_indices=[0, 2],
        probe_progress_desc_fn=lambda out_idx, total: f"F{out_idx + 1}/{total}",
    )

    tensor = out["tensor"]
    light_configs = out["light_configs"]
    light_mask = out["light_mask"]
    assert tensor.shape == (3, 2, 27)
    assert light_configs.shape == (2, 1, 12)
    assert light_mask.shape == (2, 1)

    # probe index 1 is excluded by valid_indices, so it remains zero.
    assert float(tensor[1, 0, 0]) == 0.0
    assert float(tensor[1, 1, 0]) == 0.0
    assert float(tensor[0, 0, 0]) == 1.0
    assert float(tensor[2, 1, 0]) == 3.0
    assert float(light_configs[0, 0, 0]) == 0.0
    assert float(light_configs[1, 0, 0]) == 2.0
    assert np.allclose(light_mask, 1.0)
    assert float(out["bake_seconds"]) >= 0.0


def test_save_temporal_dataset(tmp_path: Path) -> None:
    out_path = common.save_temporal_dataset(
        output_dir=tmp_path,
        tensor=np.zeros((2, 3, 27), dtype=np.float32),
        probe_positions=np.zeros((2, 3), dtype=np.float32),
        light_configs=np.zeros((3, 1, 12), dtype=np.float32),
        light_mask=np.ones((3, 1), dtype=np.float32),
        metadata={"k": "v"},
        extra_arrays={"valid_mask": np.array([1.0, 0.0], dtype=np.float32)},
    )
    assert out_path.exists()
    assert (tmp_path / "metadata.json").exists()
    loaded = np.load(out_path, allow_pickle=True)
    assert "tensor" in loaded.files
    assert "probe_positions" in loaded.files
    assert "light_configs" in loaded.files
    assert "light_mask" in loaded.files
    assert "metadata" in loaded.files
    assert "valid_mask" in loaded.files
