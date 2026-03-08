"""Path bootstrap helpers for experiment scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_paths(current_file: str | Path, root_levels: int = 2) -> tuple[Path, Path]:
    current = Path(current_file).resolve()
    root = current.parents[root_levels]
    src = root / "2_src"

    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root, src

