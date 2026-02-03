"""Run summary helpers for dataset/pipeline executions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional


def build_run_summary_md(title: str, items: Dict[str, str], sections: Optional[List[Dict[str, str]]] = None) -> str:
    lines = [f"# {title}", "", "## Summary"]
    for key, value in items.items():
        lines.append(f"- {key}: {value}")
    if sections:
        for section in sections:
            name = section.get("name", "step")
            lines.append("")
            lines.append(f"## {name}")
            for key, value in section.items():
                if key == "name":
                    continue
                lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def write_run_summary(path: str | Path, title: str, items: Dict[str, str], sections: Optional[List[Dict[str, str]]] = None) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = build_run_summary_md(title, items, sections=sections)
    out_path.write_text(content)
    return out_path

