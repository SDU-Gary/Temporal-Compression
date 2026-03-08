from pathlib import Path
import textwrap
import pytest

from tools.workflow_config import (
    apply_preset,
    build_generator_command,
    ensure_python,
    extend_cli_args,
    load_yaml,
    select_python,
    validate_dataset_config,
)


def test_load_yaml_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "missing.yaml")


def test_load_yaml_not_mapping(tmp_path: Path):
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ValueError):
        load_yaml(path)


def test_apply_preset_missing(tmp_path: Path):
    cfg = {"generator": {"script": "gen.py"}, "output_dir": "out"}
    with pytest.raises(ValueError):
        apply_preset(cfg, "nope")


def test_validate_dataset_config_missing():
    with pytest.raises(ValueError):
        validate_dataset_config({})
    with pytest.raises(ValueError):
        validate_dataset_config({"generator": {}})


def test_build_generator_command(tmp_path: Path):
    cfg = {
        "output_dir": str(tmp_path / "out"),
        "generator": {
            "script": str(tmp_path / "gen.py"),
            "args": {"num_frames": 10, "use_analytic_lights": True, "probe_file": None},
        },
    }
    output_dir, cmd = build_generator_command(cfg)
    assert output_dir.name == "out"
    assert "--num-frames" in cmd
    assert "--use-analytic-lights" in cmd
    assert "--probe-file" not in cmd


def test_ensure_python_injects():
    cmd = ensure_python(["script.py"])
    assert cmd[0].endswith("python") or cmd[0].endswith("python3") or cmd[0].endswith("python3.13")
    assert cmd[1] == "script.py"


def test_ensure_python_no_inject():
    cmd = ensure_python(["/bin/echo"])
    assert cmd[0] == "/bin/echo"


def test_select_python_prefers_config(monkeypatch):
    config = {"python": "/opt/falcor/python"}
    monkeypatch.setenv("FALCOR_PYTHON", "/env/falcor/python")
    assert select_python(config) == "/opt/falcor/python"


def test_select_python_falls_back_env(monkeypatch):
    config = {"generator": {}}
    monkeypatch.setenv("FALCOR_PYTHON", "/env/falcor/python")
    assert select_python(config) == "/env/falcor/python"


def test_extend_cli_args_handles_bool_none_and_lists() -> None:
    cmd = ["script.py"]
    out = extend_cli_args(
        cmd,
        {
            "enabled": True,
            "skip": False,
            "empty": None,
            "items": [1, 2],
            "name": "x",
        },
    )
    assert out is cmd
    assert "--enabled" in out
    assert "--skip" not in out
    assert "--empty" not in out
    assert out.count("--items") == 2
    assert "--name" in out
