import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from data.intensity_modulation_dataset import (
    IntensityModulationDataset,
    create_dataloaders as create_intensity_loaders,
)
from data.transfer_tensor_dataset import (
    TransferTensorDataset,
    TransferTensorDataset5D,
    create_dataloaders as create_transfer_loaders,
    create_dataloaders_5D,
)
from data.lightset_dataset import LightSetDataset, create_dataloaders_lightset
from data.sh_scaler import AdaptiveSHScaler, _ensure_numpy, _ensure_torch
from utils import config as config_utils
from utils.coords import (
    normalize,
    falcor_to_sh,
    sh_to_falcor,
    zenith_azimuth_to_direction,
    normalize_pos,
)
from utils.light_descriptor import (
    _denormalize,
    _kelvin_to_rgb_torch,
    _zenith_azimuth_to_dir_torch,
    build_descriptor_5d,
    build_descriptor_1d,
)
from utils.spherical_harmonics import (
    fibonacci_sphere,
    evaluate_sh_basis,
    fit_sh_coefficients,
    reconstruct_from_sh,
    compute_sh_reconstruction_error,
)
from utils.probe_field_slicer import (
    l0_luminance,
    directional_luminance,
    slice_probe_field,
)
from utils.sun_position import (
    calculate_sun_direction_simple,
    generate_24hour_sun_trajectory,
    plot_sun_trajectory,
)


def _write_intensity_dataset(root: Path, num_moments: int = 3, num_probes: int = 2) -> None:
    root.mkdir(parents=True, exist_ok=True)
    metadata = {"num_moments": num_moments}
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    positions = np.linspace(0.0, 1.0, num=num_probes * 3, dtype=np.float32).reshape(num_probes, 3)
    np.savez(root / "probes.npz", positions=positions)
    for i in range(num_moments):
        moment_dir = root / f"moment_{i:02d}"
        moment_dir.mkdir(parents=True, exist_ok=True)
        sh_coeffs = np.full((num_probes, 27), i + 0.1, dtype=np.float32)
        np.savez(moment_dir / "sh_coeffs.npz", sh_coeffs=sh_coeffs, intensity=float(i + 1), time=float(i))


def _write_transfer_dataset(root: Path, num_probes: int = 3, num_lights: int = 2) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tensor = np.random.rand(num_probes, num_lights, 27).astype(np.float32)
    probe_positions = np.random.rand(num_probes, 3).astype(np.float32)
    light_positions = np.random.rand(num_lights, 3).astype(np.float32)
    np.savez(
        root / "transfer_tensor.npz",
        tensor=tensor,
        probe_positions=probe_positions,
        light_positions=light_positions,
        metadata={"foo": "bar"},
    )


def _write_parametric_dataset(root: Path, num_probes: int = 3, num_configs: int = 4, num_slots: int = 2) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tensor = np.random.rand(num_probes, num_configs, 27).astype(np.float32)
    probe_positions = np.random.rand(num_probes, 3).astype(np.float32)
    light_configs = np.random.rand(num_configs, num_slots, 12).astype(np.float32)
    light_mask = np.ones((num_configs, num_slots), dtype=np.float32)
    valid_mask = np.ones((num_probes,), dtype=np.float32)
    valid_mask[-1] = 0.0
    np.savez(
        root / "parametric_tensor.npz",
        tensor=tensor,
        probe_positions=probe_positions,
        light_configs=light_configs,
        light_mask=light_mask,
        valid_mask=valid_mask,
    )
    (root / "metadata.json").write_text(json.dumps({"note": "ok"}), encoding="utf-8")


def test_intensity_modulation_dataset(tmp_path: Path) -> None:
    data_root = tmp_path / "intensity"
    _write_intensity_dataset(data_root, num_moments=4, num_probes=3)

    ds = IntensityModulationDataset(data_root, split="train", train_ratio=0.5, val_ratio=0.25)
    assert len(ds) == ds.num_probes * ds.num_moments
    sample = ds[0]
    assert sample["probe_position"].shape == (3,)
    assert sample["intensity"].shape == (1,)
    assert sample["sh_coeffs"].shape == (27,)
    assert sample["time"].shape == (1,)

    assert ds.get_full_probe_positions(normalized=True).shape == (3, 3)
    assert ds.get_full_probe_positions(normalized=False).shape == (3, 3)
    assert ds.get_full_intensities().shape[0] == ds.num_moments
    assert ds.get_full_times().shape[0] == ds.num_moments

    loaders = create_intensity_loaders(data_root, batch_size=2, num_workers=0, train_ratio=0.5, val_ratio=0.25)
    batch = next(iter(loaders[0]))
    assert batch["probe_position"].shape[-1] == 3

    with pytest.raises(ValueError):
        IntensityModulationDataset(data_root, split="bad")

    ds_no_norm = IntensityModulationDataset(
        data_root,
        split="train",
        train_ratio=0.5,
        val_ratio=0.25,
        normalize_positions=False,
    )
    assert np.allclose(ds_no_norm.get_full_probe_positions(normalized=True), ds_no_norm.probe_positions)


def test_intensity_modulation_missing_files(tmp_path: Path) -> None:
    data_root = tmp_path / "intensity_missing"
    data_root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        IntensityModulationDataset(data_root, split="train")

    (data_root / "metadata.json").write_text(json.dumps({"num_moments": 1}), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        IntensityModulationDataset(data_root, split="train")

    np.savez(data_root / "probes.npz", positions=np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(FileNotFoundError):
        IntensityModulationDataset(data_root, split="train")


def test_transfer_tensor_dataset(tmp_path: Path) -> None:
    data_root = tmp_path / "transfer"
    _write_transfer_dataset(data_root)

    ds = TransferTensorDataset(data_root, split="train", train_ratio=0.5, val_ratio=0.25)
    assert len(ds) == ds.num_probes * ds.num_lights
    sample = ds[0]
    assert sample["probe_position"].shape == (3,)
    assert sample["light_position"].shape == (3,)
    assert sample["sh_coeffs"].shape == (27,)
    assert ds.get_all_probe_positions().shape[1] == 3
    assert ds.get_all_light_positions().shape[1] == 3

    loaders = create_transfer_loaders(data_root, batch_size=2, num_workers=0)
    batch = next(iter(loaders[0]))
    assert batch["probe_position"].shape[-1] == 3

    with pytest.raises(ValueError):
        TransferTensorDataset(data_root, split="bad")

    ds_no_norm = TransferTensorDataset(data_root, split="train", normalize=False)
    assert np.allclose(ds_no_norm.get_all_probe_positions(), ds_no_norm.probe_positions)

    with pytest.raises(FileNotFoundError):
        TransferTensorDataset(tmp_path / "missing", split="train")


def test_transfer_tensor_dataset5d(tmp_path: Path) -> None:
    data_root = tmp_path / "parametric"
    data_root.mkdir(parents=True, exist_ok=True)
    tensor = np.random.rand(2, 3, 27).astype(np.float32)
    probes = np.random.rand(2, 3).astype(np.float32)
    configs = np.random.rand(3, 5).astype(np.float32)
    np.savez(data_root / "parametric_tensor.npz", tensor=tensor, probe_positions=probes, light_configs=configs)
    (data_root / "metadata.json").write_text(json.dumps({"a": 1}), encoding="utf-8")

    ds = TransferTensorDataset5D(data_root, split="train", train_ratio=0.5, val_ratio=0.25)
    assert len(ds) == ds.num_probes * ds.num_configs
    sample = ds[0]
    assert sample["probe_position"].shape == (3,)
    assert sample["light_params"].shape == (5,)
    assert sample["sh_coeffs"].shape == (27,)

    ds_raw = TransferTensorDataset5D(data_root, split="val", train_ratio=0.5, val_ratio=0.25, normalize_params=False)
    assert ds_raw.light_configs_subset_norm.shape == ds_raw.light_configs_subset.shape

    ds_test = TransferTensorDataset5D(data_root, split="test", train_ratio=0.5, val_ratio=0.25, normalize_probes=False)
    assert np.allclose(ds_test.get_all_probe_positions(), ds_test.probe_positions_norm)
    ds_test.get_all_light_configs()
    ds_test.get_unnormalized_light_configs()

    loaders = create_dataloaders_5D(data_root, batch_size=2, num_workers=0, train_ratio=0.5, val_ratio=0.25)
    assert len(loaders) == 3

    with pytest.raises(ValueError):
        TransferTensorDataset5D(data_root, split="bad")

    missing_root = tmp_path / "parametric_missing"
    missing_root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        TransferTensorDataset5D(missing_root, split="train")

    no_meta_root = tmp_path / "parametric_no_meta"
    no_meta_root.mkdir(parents=True, exist_ok=True)
    np.savez(no_meta_root / "parametric_tensor.npz", tensor=tensor, probe_positions=probes, light_configs=configs)
    ds_no_meta = TransferTensorDataset5D(no_meta_root, split="train", train_ratio=0.5, val_ratio=0.25)
    assert ds_no_meta.metadata == {}


def test_lightset_dataset(tmp_path: Path) -> None:
    data_root = tmp_path / "lightset"
    _write_parametric_dataset(data_root)
    ds = LightSetDataset(data_root, split="train", train_ratio=0.5, val_ratio=0.25)
    assert ds.tensor.shape[0] == 2  # one invalid probe removed
    sample = ds[0]
    assert sample["light_mask"].shape[-1] == 2
    assert ds.get_full_probe_positions(normalized=True).shape[1] == 3
    assert ds.get_full_probe_positions(normalized=False).shape[1] == 3

    # Branch without light_mask
    data_root2 = tmp_path / "lightset_nomask"
    data_root2.mkdir(parents=True, exist_ok=True)
    tensor = np.random.rand(2, 2, 27).astype(np.float32)
    probes = np.random.rand(2, 3).astype(np.float32)
    light_configs = np.random.rand(2, 1, 12).astype(np.float32)
    np.savez(data_root2 / "parametric_tensor.npz", tensor=tensor, probe_positions=probes, light_configs=light_configs)
    ds2 = LightSetDataset(data_root2, split="test", train_ratio=0.5, val_ratio=0.25)
    assert ds2.light_mask.shape == (2, 1)

    loaders = create_dataloaders_lightset(data_root, batch_size=2, num_workers=0)
    batch = next(iter(loaders[0]))
    assert batch["sh_coeffs"].shape[-1] == 27

    with pytest.raises(ValueError):
        LightSetDataset(data_root, split="bad")

    with pytest.raises(FileNotFoundError):
        LightSetDataset(tmp_path / "missing", split="train")

    ds_no_norm = LightSetDataset(data_root, split="train", normalize_probes=False)
    assert np.allclose(ds_no_norm.get_full_probe_positions(normalized=True), ds_no_norm.probe_positions)
    assert np.allclose(ds_no_norm.get_full_probe_positions(normalized=False), ds_no_norm.probe_positions)


def test_sh_scaler(tmp_path: Path) -> None:
    sh = np.random.rand(4, 27).astype(np.float32)
    scaler = AdaptiveSHScaler.fit(sh)
    scaled = scaler.transform(sh)
    recovered = scaler.inverse(scaled)
    np.testing.assert_allclose(sh, recovered, atol=1e-5)

    tensor = torch.from_numpy(sh)
    scaled_t = scaler.transform(tensor)
    recovered_t = scaler.inverse(scaled_t)
    np.testing.assert_allclose(recovered_t.numpy(), sh, atol=1e-5)

    assert np.allclose(_ensure_numpy(tensor), sh)
    assert isinstance(_ensure_torch(sh), torch.Tensor)
    assert isinstance(_ensure_torch(tensor), torch.Tensor)

    class DummyDataset:
        def __init__(self, tensor_subset):
            self.tensor_subset = tensor_subset

    dataset = DummyDataset(sh.reshape(2, 2, 27))
    scaler2 = AdaptiveSHScaler.fit_from_dataset(dataset, max_samples=2)
    assert scaler2.l0_mean.shape == (3,)

    class DummyDataset2:
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {"sh_coeffs": torch.from_numpy(sh[idx]).float()}

    scaler3 = AdaptiveSHScaler.fit_from_dataset(DummyDataset2(), max_samples=1)
    assert scaler3.ho_rms.shape == (3,)

    class DummyDataset4:
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {"sh_coeffs": torch.from_numpy(sh[idx]).float()}

    scaler4 = AdaptiveSHScaler.fit_from_dataset(DummyDataset4(), max_samples=None)
    assert scaler4.l0_std.shape == (3,)

    class DummyDataset3:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"wrong": 1}

    with pytest.raises(KeyError):
        AdaptiveSHScaler.fit_from_dataset(DummyDataset3(), max_samples=1)

    save_path = tmp_path / "scaler.npz"
    scaler.save(save_path)
    loaded = AdaptiveSHScaler.load(save_path)
    np.testing.assert_allclose(loaded.l0_mean, scaler.l0_mean)


def test_config_utils(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_dict = {"experiment": {"name": "x"}, "data": {"dataset_path": "y"}}
    cfg = config_utils.Config(cfg_dict)
    assert cfg.get("experiment.name") == "x"
    assert cfg["data"]["dataset_path"] == "y"
    assert "Config" in repr(cfg)

    cfg_path = tmp_path / "cfg.yaml"
    cfg.save(cfg_path)
    loaded = config_utils.Config.from_yaml(cfg_path)
    assert loaded.get("experiment.name") == "x"

    with pytest.raises(FileNotFoundError):
        config_utils.Config.from_yaml(tmp_path / "missing.yaml")

    with pytest.raises(ValueError):
        config_utils.validate_config(config_utils.Config({}))

    valid_cfg = config_utils.Config(
        {
            "experiment": {"name": "a", "output_dir": "b", "device": "cpu"},
            "data": {"dataset_path": "x", "batch_size": 1},
            "model": {"num_gaussians": 1, "latent_dim_base": 1, "latent_dim_time": 1},
            "training": {"num_steps": 1},
        }
    )
    assert config_utils.validate_config(valid_cfg) is True

    merged = config_utils.merge_configs({"a": {"b": 1}}, {"a": {"c": 2}, "d": 3})
    assert merged["a"]["b"] == 1 and merged["a"]["c"] == 2 and merged["d"] == 3

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps({"type": "object", "properties": {"a": {"type": "integer"}}}))
    msgs = config_utils.validate_with_schema({"a": "x"}, schema_path, strict=False)
    assert msgs

    class DummyValidator:
        def __init__(self, schema):
            self.schema = schema

        def iter_errors(self, config):
            if not isinstance(config.get("a"), int):
                yield type("Err", (), {"path": ["a"], "message": "a must be int"})()

    class DummyJsonSchema:
        Draft7Validator = DummyValidator

    monkeypatch.setitem(sys.modules, "jsonschema", DummyJsonSchema)
    msgs2 = config_utils.validate_with_schema({"a": "x"}, schema_path, strict=False)
    assert "a must be int" in msgs2[0]
    with pytest.raises(ValueError):
        config_utils.validate_with_schema({"a": "x"}, schema_path, strict=True)

    missing_schema = tmp_path / "missing_schema.json"
    msgs3 = config_utils.validate_with_schema({"a": 1}, missing_schema, strict=False)
    assert "schema not found" in msgs3[0]

    bad_schema = tmp_path / "bad_schema.json"
    bad_schema.write_text("{bad", encoding="utf-8")
    msgs4 = config_utils.validate_with_schema({"a": 1}, bad_schema, strict=False)
    assert "failed to load schema" in msgs4[0]

    class DummyValidatorBoom:
        def __init__(self, schema):
            self.schema = schema

        def iter_errors(self, config):
            raise RuntimeError("boom")

    class DummyJsonSchemaBoom:
        Draft7Validator = DummyValidatorBoom

    monkeypatch.setitem(sys.modules, "jsonschema", DummyJsonSchemaBoom)
    msgs5 = config_utils.validate_with_schema({"a": 1}, schema_path, strict=False)
    assert "schema validation error" in msgs5[0]


def test_coords_and_descriptors() -> None:
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert np.allclose(normalize(vec), vec)
    assert np.allclose(falcor_to_sh(vec), vec)
    assert np.allclose(sh_to_falcor(vec), vec)
    dir_vec = zenith_azimuth_to_direction(60.0, 30.0)
    assert dir_vec.shape == (3,)
    pos = normalize_pos(np.array([1.0, 2.0, 3.0]), np.zeros(3), np.ones(3) * 3)
    assert pos.shape == (3,)

    params = torch.tensor([[0.0, 0.0, 1.0, 6500.0, 0.5]])
    pmin = torch.tensor([0.0, 0.0, 0.1, 2500.0, 0.0])
    pmax = torch.tensor([90.0, 360.0, 2.0, 10000.0, 0.9])
    denorm = _denormalize(torch.zeros_like(params), pmin, pmax)
    assert denorm.shape == params.shape
    rgb = _kelvin_to_rgb_torch(torch.tensor([6500.0]))
    assert rgb.shape == (1, 3)
    dirs = _zenith_azimuth_to_dir_torch(torch.tensor([60.0]), torch.tensor([90.0]))
    assert dirs.shape == (1, 3)
    desc5 = build_descriptor_5d(torch.zeros_like(params), pmin, pmax)
    assert desc5.shape[-1] == 12
    desc1 = build_descriptor_1d(torch.tensor([1.0, 0.5]))
    assert desc1.shape == (2, 12)


def test_spherical_harmonics_and_slicer() -> None:
    dirs = fibonacci_sphere(16)
    Y = evaluate_sh_basis(dirs, max_order=2)
    assert Y.shape == (16, 9)
    with pytest.raises(NotImplementedError):
        evaluate_sh_basis(dirs, max_order=3)
    radiances = np.clip(dirs[:, 2:3], 0.0, 1.0) * np.array([1.0, 0.5, 0.25])
    coeffs = fit_sh_coefficients(dirs, radiances, max_order=2)
    recon = reconstruct_from_sh(coeffs, dirs, max_order=2)
    assert recon.shape == radiances.shape
    mse, rel = compute_sh_reconstruction_error(dirs, radiances, coeffs, max_order=2)
    assert mse >= 0.0 and rel >= 0.0

    sh_coeffs = np.random.rand(10, 27).astype(np.float32)
    l0 = l0_luminance(sh_coeffs)
    assert l0.shape[0] == 10
    lum = directional_luminance(sh_coeffs, np.array([0.0, 1.0, 0.0]))
    assert lum.shape[0] == 10

    probe_positions = np.random.rand(10, 3).astype(np.float32)
    result = slice_probe_field(probe_positions, sh_coeffs, axis="y", value=0.5, tolerance=1.0)
    assert result.heatmap.shape[0] == result.heatmap.shape[1]
    with pytest.raises(ValueError):
        slice_probe_field(probe_positions, sh_coeffs, axis="w")
    with pytest.raises(ValueError):
        slice_probe_field(probe_positions, sh_coeffs, axis="y", value=10.0, tolerance=0.01)
    with pytest.raises(ValueError):
        slice_probe_field(probe_positions, sh_coeffs, axis="y", value=0.5, tolerance=1.0, mode="bad")
    with pytest.raises(ValueError):
        slice_probe_field(probe_positions, sh_coeffs, axis="y", value=0.5, tolerance=1.0, mode="dir")
    slice_probe_field(probe_positions, sh_coeffs, axis="y", value=0.5, tolerance=1.0, mode="dir", direction=np.array([1.0, 0.0, 0.0]))


def test_sun_position(monkeypatch: pytest.MonkeyPatch) -> None:
    direction = calculate_sun_direction_simple(12.0, 40.0, 116.0)
    assert direction.shape == (3,)
    trajectory = generate_24hour_sun_trajectory()
    assert len(trajectory) == 24

    class DummyAxes:
        def plot(self, *args, **kwargs):
            return None

        def scatter(self, *args, **kwargs):
            return None

        def text(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_zlabel(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            return None

        def legend(self, *args, **kwargs):
            return None

        def set_theta_zero_location(self, *args, **kwargs):
            return None

        def set_theta_direction(self, *args, **kwargs):
            return None

        def set_ylim(self, *args, **kwargs):
            return None

        def set_yticks(self, *args, **kwargs):
            return None

        def set_yticklabels(self, *args, **kwargs):
            return None

        def grid(self, *args, **kwargs):
            return None

    class DummyFig:
        def add_subplot(self, *args, **kwargs):
            return DummyAxes()

    class DummyPyplot:
        @staticmethod
        def figure(*args, **kwargs):
            return DummyFig()

        @staticmethod
        def tight_layout():
            return None

        @staticmethod
        def savefig(*args, **kwargs):
            return None

        @staticmethod
        def show():
            return None

    import types

    mpl = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.figure = DummyPyplot.figure
    pyplot.tight_layout = DummyPyplot.tight_layout
    pyplot.savefig = DummyPyplot.savefig
    pyplot.show = DummyPyplot.show
    mpl.pyplot = pyplot
    mpl_toolkits = types.ModuleType("mpl_toolkits")
    mpl_mplot3d = types.ModuleType("mpl_toolkits.mplot3d")
    mpl_mplot3d.Axes3D = type("Axes3D", (), {})
    monkeypatch.setitem(sys.modules, "matplotlib", mpl)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    monkeypatch.setitem(sys.modules, "mpl_toolkits", mpl_toolkits)
    monkeypatch.setitem(sys.modules, "mpl_toolkits.mplot3d", mpl_mplot3d)
    plot_sun_trajectory(trajectory, save_path="dummy.png")
    plot_sun_trajectory(trajectory, save_path=None)

    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    plot_sun_trajectory(trajectory, save_path="dummy.png")


def test_utils_init_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import types
    import utils

    monkeypatch.delitem(sys.modules, "utils.rerun_logger", raising=False)
    monkeypatch.delitem(sys.modules, "utils.light_descriptor", raising=False)
    monkeypatch.delitem(sys.modules, "utils.rendering_utils", raising=False)
    importlib.reload(utils)

    dummy_rerun = types.ModuleType("utils.rerun_logger")
    dummy_rerun.RerunLogger = object
    dummy_render = types.ModuleType("utils.rendering_utils")
    dummy_light = types.ModuleType("utils.light_descriptor")
    monkeypatch.setitem(sys.modules, "utils.rerun_logger", dummy_rerun)
    monkeypatch.setitem(sys.modules, "utils.rendering_utils", dummy_render)
    monkeypatch.setitem(sys.modules, "utils.light_descriptor", dummy_light)
    importlib.reload(utils)
