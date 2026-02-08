from __future__ import annotations


def test_render_with_envmap_external_quiet_redirects_std(monkeypatch, tmp_path):
    """Ensure quiet mode redirects subprocess stdout/stderr.

    We don't execute Falcor here; just verify subprocess.run is called with
    DEVNULL when quiet=True.
    """

    import subprocess

    from utils import rendering_utils

    # Ensure we don't try to call a real python3.10 interpreter.
    monkeypatch.setattr(rendering_utils, "get_falcor_python_bin", lambda: "/usr/bin/python3")

    calls: list[dict] = []

    def fake_run(cmd, check, env, stdout=None, stderr=None):
        calls.append({"cmd": cmd, "stdout": stdout, "stderr": stderr})
        return None

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Avoid touching the filesystem for output.
    import numpy as np

    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: np.zeros((2, 2, 3), dtype=np.float32))

    img = rendering_utils.render_with_envmap_external(
        envmap_path=str(tmp_path / "env.hdr"),
        python_bin="/usr/bin/python3",
        quiet=True,
    )
    assert img.shape == (2, 2, 3)
    assert len(calls) == 1
    assert calls[0]["stdout"] is subprocess.DEVNULL
    assert calls[0]["stderr"] is subprocess.DEVNULL

