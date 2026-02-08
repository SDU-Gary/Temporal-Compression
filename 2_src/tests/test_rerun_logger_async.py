from __future__ import annotations

import types


def test_rerun_logger_render_async_skips_when_busy(monkeypatch):
    """rendered comparison should not spawn multiple workers."""

    from utils.rerun_logger import RerunLogger

    # Create an instance without invoking rr.init.
    logger = object.__new__(RerunLogger)
    logger._rr_lock = types.SimpleNamespace(__enter__=lambda *_: None, __exit__=lambda *_: None)

    class AliveThread:
        def is_alive(self):
            return True

    logger._render_thread = AliveThread()

    called = {"count": 0}

    def fake_worker(*_args, **_kwargs):
        called["count"] += 1

    logger._render_and_log_comparison = fake_worker  # type: ignore[attr-defined]

    import numpy as np

    logger.log_rendered_comparison(1, np.zeros(27), np.zeros(27), "best")
    assert called["count"] == 0

