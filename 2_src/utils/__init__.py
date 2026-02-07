"""Shared utilities.

Keep this module lightweight: do not import heavy submodules (numpy/torch/matplotlib)
at import time. Import required utilities directly, e.g.:

    from utils.coords import normalize
"""

try:
    from .rerun_logger import RerunLogger  # noqa: F401
except Exception:
    RerunLogger = None

__all__ = []
if RerunLogger is not None:
    __all__.append("RerunLogger")
