#!/usr/bin/env python3
"""Compose thesis-ready visual panels from sampled benchmark/render NPZ files.

This script is intentionally post-processing only. It does not re-render scenes.
It consumes frame-level `.npz` assets produced by existing export/benchmark tools,
derives GT / Pred / Err visualizations, and composes them into a labeled panel.

Supported input keys (best-effort, in order of preference):

- GT:
  - gt_srgb_u8
  - gt_srgb
  - gt_linear
- Pred:
  - model_srgb_u8 / pred_srgb_u8
  - model_srgb / pred_srgb
  - model_linear / pred_linear
- Err:
  - err_srgb
  - err_linear
  - computed from GT/Pred if missing

Typical usage:

    python tools/analysis/compose_thesis_visual_panels.py \
      --manifest path/to/figure_manifest.json \
      --output 4_thesis/figures/fig5_x.png \
      --export-cells-dir 3_experiments/results/thesis_visuals/cells
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image, ImageColor, ImageDraw, ImageFont
except Exception as exc:  # pragma: no cover - hard requirement for this tool.
    raise RuntimeError("Pillow is required for compose_thesis_visual_panels.py") from exc


RGBColor = Tuple[int, int, int]


def _tone_map_reinhard(image: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(image, dtype=np.float32), 0.0)
    return arr / (1.0 + arr)


def _srgb_encode(image: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(image, dtype=np.float32), 0.0)
    threshold = 0.0031308
    low = 12.92 * arr
    high = 1.055 * np.power(arr, 1.0 / 2.4) - 0.055
    return np.where(arr <= threshold, low, high)


def _to_u8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got shape={arr.shape}")
    if arr.shape[2] < 3:
        raise ValueError(f"Expected at least 3 channels, got shape={arr.shape}")
    if arr.shape[2] > 3:
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr.copy()
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _parse_hex_color(value: str, default: RGBColor) -> RGBColor:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        rgb = ImageColor.getrgb(text)
    except Exception:
        return default
    if len(rgb) == 3:
        return int(rgb[0]), int(rgb[1]), int(rgb[2])
    return default


def _load_font(size: int) -> ImageFont.ImageFont:
    size_i = max(8, int(size))
    candidates = [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for cand in candidates:
        try:
            return ImageFont.truetype(cand, size=size_i)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_bbox(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.ImageFont,
) -> Tuple[int, int]:
    if not text:
        return 0, 0
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def _resize_u8(image_u8: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid target size: {size}")
    if image_u8.shape[1] == width and image_u8.shape[0] == height:
        return image_u8
    pil = Image.fromarray(image_u8)
    resample = getattr(Image, "Resampling", Image).LANCZOS
    return np.asarray(pil.resize((width, height), resample=resample), dtype=np.uint8)


def _crop_image_or_map(arr: np.ndarray, crop: Optional[Sequence[float]]) -> np.ndarray:
    if crop is None:
        return arr
    if len(crop) != 4:
        raise ValueError(f"crop must have 4 values, got {crop}")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    x0, y0, x1, y1 = [float(v) for v in crop]
    normalized = all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1))
    if normalized:
        ix0 = int(round(x0 * w))
        iy0 = int(round(y0 * h))
        ix1 = int(round(x1 * w))
        iy1 = int(round(y1 * h))
    else:
        ix0 = int(round(x0))
        iy0 = int(round(y0))
        ix1 = int(round(x1))
        iy1 = int(round(y1))
    ix0 = max(0, min(w, ix0))
    iy0 = max(0, min(h, iy0))
    ix1 = max(ix0 + 1, min(w, ix1))
    iy1 = max(iy0 + 1, min(h, iy1))
    return arr[iy0:iy1, ix0:ix1, ...]


def _candidate_keys(prefixes: Sequence[str], suffixes: Sequence[str]) -> Iterable[str]:
    for prefix in prefixes:
        for suffix in suffixes:
            yield f"{prefix}{suffix}"


class SampleCache:
    def __init__(self) -> None:
        self._cache: Dict[Path, Dict[str, np.ndarray]] = {}

    def load(self, path: Path) -> Dict[str, np.ndarray]:
        resolved = path.resolve()
        cached = self._cache.get(resolved)
        if cached is not None:
            return cached
        with np.load(resolved, allow_pickle=False) as data:
            payload = {k: np.asarray(data[k]) for k in data.files}
        self._cache[resolved] = payload
        return payload


def _first_present(data: Mapping[str, np.ndarray], keys: Iterable[str]) -> Tuple[str, np.ndarray]:
    for key in keys:
        if key in data:
            return key, np.asarray(data[key])
    raise KeyError(f"None of the candidate keys exist: {list(keys)}")


def _extract_visual_u8(
    data: Mapping[str, np.ndarray],
    *,
    prefixes: Sequence[str],
) -> np.ndarray:
    u8_keys = list(_candidate_keys(prefixes, ("srgb_u8",)))
    try:
        _, arr = _first_present(data, u8_keys)
        return _to_u8_rgb(arr)
    except KeyError:
        pass

    srgb_keys = list(_candidate_keys(prefixes, ("srgb",)))
    try:
        _, arr = _first_present(data, srgb_keys)
        return _to_u8_rgb(arr)
    except KeyError:
        pass

    linear_keys = list(_candidate_keys(prefixes, ("linear",)))
    _, arr = _first_present(data, linear_keys)
    srgb = _srgb_encode(_tone_map_reinhard(np.asarray(arr, dtype=np.float32)))
    return _to_u8_rgb(srgb)


def _extract_linear_or_srgb_float(
    data: Mapping[str, np.ndarray],
    *,
    prefixes: Sequence[str],
    prefer_space: str,
) -> np.ndarray:
    space = str(prefer_space).strip().lower()
    if space == "srgb":
        srgb_keys = list(_candidate_keys(prefixes, ("srgb",)))
        try:
            _, arr = _first_present(data, srgb_keys)
            return np.asarray(arr, dtype=np.float32)
        except KeyError:
            u8_keys = list(_candidate_keys(prefixes, ("srgb_u8",)))
            try:
                _, arr = _first_present(data, u8_keys)
                return np.asarray(arr, dtype=np.float32) / 255.0
            except KeyError:
                linear_keys = list(_candidate_keys(prefixes, ("linear",)))
                _, arr = _first_present(data, linear_keys)
                return _srgb_encode(_tone_map_reinhard(np.asarray(arr, dtype=np.float32)))

    linear_keys = list(_candidate_keys(prefixes, ("linear",)))
    try:
        _, arr = _first_present(data, linear_keys)
        return np.asarray(arr, dtype=np.float32)
    except KeyError:
        srgb_keys = list(_candidate_keys(prefixes, ("srgb",)))
        try:
            _, arr = _first_present(data, srgb_keys)
            return np.asarray(arr, dtype=np.float32)
        except KeyError:
            u8_keys = list(_candidate_keys(prefixes, ("srgb_u8",)))
            _, arr = _first_present(data, u8_keys)
            return np.asarray(arr, dtype=np.float32) / 255.0


def _maybe_load_err_map(
    data: Mapping[str, np.ndarray],
    *,
    prefer_space: str,
) -> Optional[np.ndarray]:
    suffix = "err_srgb" if str(prefer_space).strip().lower() == "srgb" else "err_linear"
    if suffix in data:
        return np.asarray(data[suffix], dtype=np.float32)
    return None


def _compute_error_scalar(
    data: Mapping[str, np.ndarray],
    *,
    prefer_space: str,
) -> np.ndarray:
    err_map = _maybe_load_err_map(data, prefer_space=prefer_space)
    if err_map is None:
        gt = _extract_linear_or_srgb_float(data, prefixes=("gt_",), prefer_space=prefer_space)
        pred = _extract_linear_or_srgb_float(data, prefixes=("model_", "pred_"), prefer_space=prefer_space)
        diff = np.abs(pred - gt)
    else:
        diff = np.asarray(err_map, dtype=np.float32)
    if diff.ndim == 3:
        return np.mean(np.abs(diff[..., :3]), axis=-1)
    if diff.ndim == 2:
        return np.abs(diff)
    raise ValueError(f"Unsupported error map shape: {diff.shape}")


def _simple_colormap_turbo(norm: np.ndarray) -> np.ndarray:
    anchors = np.array(
        [
            [0.0, 0.18995, 0.07176, 0.23217],
            [0.25, 0.25107, 0.25237, 0.63374],
            [0.50, 0.27628, 0.55500, 0.79000],
            [0.75, 0.99300, 0.90600, 0.14400],
            [1.00, 0.47960, 0.01580, 0.01055],
        ],
        dtype=np.float32,
    )
    x = np.clip(np.asarray(norm, dtype=np.float32), 0.0, 1.0)
    out = np.zeros(x.shape + (3,), dtype=np.float32)
    for i in range(len(anchors) - 1):
        left = anchors[i]
        right = anchors[i + 1]
        mask = (x >= left[0]) & (x <= right[0] if i == len(anchors) - 2 else x < right[0])
        if not np.any(mask):
            continue
        t = (x[mask] - left[0]) / max(1e-8, float(right[0] - left[0]))
        out[mask] = (1.0 - t[:, None]) * left[1:] + t[:, None] * right[1:]
    return out


def _colorize_error_map(
    scalar_map: np.ndarray,
    *,
    vmax: float,
    colormap: str,
) -> np.ndarray:
    vmax_f = max(1e-8, float(vmax))
    normalized = np.clip(np.asarray(scalar_map, dtype=np.float32) / vmax_f, 0.0, 1.0)
    cmap_name = str(colormap).strip().lower()
    if cmap_name == "gray":
        rgb = np.repeat(normalized[..., None], 3, axis=-1)
        return _to_u8_rgb(rgb)
    if cmap_name in {"matplotlib_turbo", "turbo_mpl"}:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            rgb = plt.cm.get_cmap("turbo")(normalized)[..., :3]
            return _to_u8_rgb(rgb)
        except Exception:
            pass
    rgb = _simple_colormap_turbo(normalized)
    return _to_u8_rgb(rgb)


@dataclass
class ErrorConfig:
    source_space: str = "srgb"
    percentile: float = 99.0
    vmax: Optional[float] = None
    colormap: str = "turbo"


@dataclass
class LayoutConfig:
    cell_width: Optional[int] = None
    cell_height: Optional[int] = None
    outer_padding: int = 24
    cell_gap: int = 16
    row_title_width: int = 160
    col_title_height: int = 42
    footer_height: int = 0
    background: RGBColor = (255, 255, 255)
    title_color: RGBColor = (20, 20, 20)
    label_color: RGBColor = (20, 20, 20)
    border_color: RGBColor = (220, 220, 220)
    border_width: int = 1
    title_font_size: int = 22
    label_font_size: int = 22


def _resolve_path(base_dir: Path, text: str) -> Path:
    raw = Path(str(text))
    if raw.is_absolute():
        return raw
    cand_manifest = (base_dir / raw).resolve()
    if cand_manifest.exists():
        return cand_manifest
    cand_cwd = (Path.cwd() / raw).resolve()
    if cand_cwd.exists():
        return cand_cwd
    return cand_manifest


def _parse_error_config(raw: Mapping[str, Any]) -> ErrorConfig:
    return ErrorConfig(
        source_space=str(raw.get("source_space", "srgb")),
        percentile=float(raw.get("percentile", 99.0)),
        vmax=(None if raw.get("vmax", None) is None else float(raw.get("vmax"))),
        colormap=str(raw.get("colormap", "turbo")),
    )


def _parse_layout_config(raw: Mapping[str, Any]) -> LayoutConfig:
    size = raw.get("cell_size", None)
    cell_w = cell_h = None
    if isinstance(size, (list, tuple)) and len(size) == 2:
        cell_w, cell_h = int(size[0]), int(size[1])
    return LayoutConfig(
        cell_width=cell_w,
        cell_height=cell_h,
        outer_padding=int(raw.get("outer_padding", 24)),
        cell_gap=int(raw.get("cell_gap", 16)),
        row_title_width=int(raw.get("row_title_width", 160)),
        col_title_height=int(raw.get("col_title_height", 42)),
        footer_height=int(raw.get("footer_height", 0)),
        background=_parse_hex_color(str(raw.get("background", "#ffffff")), (255, 255, 255)),
        title_color=_parse_hex_color(str(raw.get("title_color", "#141414")), (20, 20, 20)),
        label_color=_parse_hex_color(str(raw.get("label_color", "#141414")), (20, 20, 20)),
        border_color=_parse_hex_color(str(raw.get("border_color", "#dcdcdc")), (220, 220, 220)),
        border_width=int(raw.get("border_width", 1)),
        title_font_size=int(raw.get("title_font_size", 22)),
        label_font_size=int(raw.get("label_font_size", 22)),
    )


def _resolve_alias_npz(base_dir: Path, aliases: Mapping[str, str], value: str) -> Path:
    text = str(value)
    if text in aliases:
        return _resolve_path(base_dir, aliases[text])
    return _resolve_path(base_dir, text)


def _infer_cell_size_from_specs(
    base_dir: Path,
    rows: Sequence[Sequence[Mapping[str, Any]]],
    aliases: Mapping[str, str],
    cache: SampleCache,
) -> Tuple[int, int]:
    for row in rows:
        for cell in row:
            if str(cell.get("kind", "")).strip().lower() == "blank":
                continue
            npz_path = _resolve_alias_npz(base_dir, aliases, str(cell["npz"]))
            data = cache.load(npz_path)
            if str(cell.get("kind", "")).strip().lower() == "err":
                scalar = _compute_error_scalar(
                    data,
                    prefer_space=str(cell.get("error_space", "srgb")),
                )
                cropped = _crop_image_or_map(scalar[..., None], cell.get("crop"))
                return int(cropped.shape[1]), int(cropped.shape[0])
            image = _extract_visual_u8(
                data,
                prefixes=("gt_",) if str(cell.get("kind", "")).strip().lower() == "gt" else ("model_", "pred_"),
            )
            cropped = _crop_image_or_map(image, cell.get("crop"))
            return int(cropped.shape[1]), int(cropped.shape[0])
    raise ValueError("Could not infer cell size from manifest rows.")


def _render_cell_image(
    base_dir: Path,
    cell: Mapping[str, Any],
    aliases: Mapping[str, str],
    cache: SampleCache,
    *,
    error_cfg: ErrorConfig,
    shared_err_vmax: float,
    target_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    kind = str(cell.get("kind", "")).strip().lower()
    if kind == "blank":
        return None
    npz_path = _resolve_alias_npz(base_dir, aliases, str(cell["npz"]))
    data = cache.load(npz_path)
    crop = cell.get("crop")

    if kind == "gt":
        image = _extract_visual_u8(data, prefixes=("gt_",))
        image = _crop_image_or_map(image, crop)
        return _resize_u8(image, target_size)

    if kind == "pred":
        image = _extract_visual_u8(data, prefixes=("model_", "pred_"))
        image = _crop_image_or_map(image, crop)
        return _resize_u8(image, target_size)

    if kind == "compare":
        gt = _extract_visual_u8(data, prefixes=("gt_",))
        pred = _extract_visual_u8(data, prefixes=("model_", "pred_"))
        gt = _crop_image_or_map(gt, crop)
        pred = _crop_image_or_map(pred, crop)
        gt = _resize_u8(gt, target_size)
        pred = _resize_u8(pred, target_size)
        return np.concatenate([gt, pred], axis=1)

    if kind == "err":
        err_space = str(cell.get("error_space", error_cfg.source_space))
        scalar = _compute_error_scalar(data, prefer_space=err_space)
        scalar = _crop_image_or_map(scalar[..., None], crop)[..., 0]
        image = _colorize_error_map(
            scalar,
            vmax=(float(cell["error_vmax"]) if "error_vmax" in cell else shared_err_vmax),
            colormap=str(cell.get("colormap", error_cfg.colormap)),
        )
        return _resize_u8(image, target_size)

    raise ValueError(f"Unsupported cell kind: {kind}")


def _collect_error_maps(
    base_dir: Path,
    rows: Sequence[Sequence[Mapping[str, Any]]],
    aliases: Mapping[str, str],
    cache: SampleCache,
    error_cfg: ErrorConfig,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for row in rows:
        for cell in row:
            if str(cell.get("kind", "")).strip().lower() != "err":
                continue
            npz_path = _resolve_alias_npz(base_dir, aliases, str(cell["npz"]))
            data = cache.load(npz_path)
            scalar = _compute_error_scalar(
                data,
                prefer_space=str(cell.get("error_space", error_cfg.source_space)),
            )
            scalar = _crop_image_or_map(scalar[..., None], cell.get("crop"))[..., 0]
            out.append(np.asarray(scalar, dtype=np.float32))
    return out


def _save_png(path: Path, image_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_u8_rgb(image_u8)).save(str(path))


def _compose_canvas(
    *,
    rendered_rows: Sequence[Sequence[Optional[np.ndarray]]],
    column_titles: Sequence[str],
    row_titles: Sequence[str],
    layout: LayoutConfig,
) -> Image.Image:
    num_rows = len(rendered_rows)
    num_cols = max((len(r) for r in rendered_rows), default=0)
    if num_rows <= 0 or num_cols <= 0:
        raise ValueError("rendered_rows must contain at least one non-empty row")

    widths = [0] * num_cols
    heights = [0] * num_rows
    for r_idx, row in enumerate(rendered_rows):
        for c_idx, cell_img in enumerate(row):
            if cell_img is None:
                continue
            heights[r_idx] = max(heights[r_idx], int(cell_img.shape[0]))
            widths[c_idx] = max(widths[c_idx], int(cell_img.shape[1]))

    total_w = (
        layout.outer_padding * 2
        + layout.row_title_width
        + sum(widths)
        + max(0, num_cols - 1) * layout.cell_gap
    )
    total_h = (
        layout.outer_padding * 2
        + layout.col_title_height
        + sum(heights)
        + max(0, num_rows - 1) * layout.cell_gap
        + layout.footer_height
    )

    canvas = Image.new("RGB", (int(total_w), int(total_h)), color=layout.background)
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(layout.title_font_size)
    label_font = _load_font(layout.label_font_size)

    x_base = layout.outer_padding + layout.row_title_width
    y_base = layout.outer_padding + layout.col_title_height

    x_cursor = x_base
    for c_idx in range(num_cols):
        col_title = str(column_titles[c_idx]) if c_idx < len(column_titles) else ""
        w_text, h_text = _text_bbox(draw, col_title, font=title_font)
        title_x = x_cursor + max(0, widths[c_idx] - w_text) // 2
        title_y = layout.outer_padding + max(0, layout.col_title_height - h_text) // 2
        if col_title:
            draw.text((title_x, title_y), col_title, font=title_font, fill=layout.title_color)
        x_cursor += widths[c_idx] + layout.cell_gap

    y_cursor = y_base
    for r_idx in range(num_rows):
        row_title = str(row_titles[r_idx]) if r_idx < len(row_titles) else ""
        w_text, h_text = _text_bbox(draw, row_title, font=label_font)
        title_x = layout.outer_padding + max(0, layout.row_title_width - w_text) // 2
        title_y = y_cursor + max(0, heights[r_idx] - h_text) // 2
        if row_title:
            draw.text((title_x, title_y), row_title, font=label_font, fill=layout.label_color)

        x_cursor = x_base
        row = rendered_rows[r_idx]
        for c_idx in range(num_cols):
            cell_img = row[c_idx] if c_idx < len(row) else None
            if cell_img is not None:
                img = Image.fromarray(cell_img)
                paste_x = x_cursor
                paste_y = y_cursor
                canvas.paste(img, (paste_x, paste_y))
                if layout.border_width > 0:
                    bw = max(1, int(layout.border_width))
                    for i in range(bw):
                        draw.rectangle(
                            [
                                paste_x - i,
                                paste_y - i,
                                paste_x + img.size[0] - 1 + i,
                                paste_y + img.size[1] - 1 + i,
                            ],
                            outline=layout.border_color,
                        )
            x_cursor += widths[c_idx] + layout.cell_gap
        y_cursor += heights[r_idx] + layout.cell_gap

    return canvas


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose thesis-ready GT/Pred/Err panels from sampled NPZ assets.")
    parser.add_argument("--manifest", required=True, help="Manifest JSON describing the panel layout.")
    parser.add_argument("--output", required=True, help="Output PNG path for the composed panel.")
    parser.add_argument(
        "--export-cells-dir",
        default=None,
        help="Optional directory to export derived GT/Pred/Err cell PNGs for inspection.",
    )
    parser.add_argument(
        "--write-metadata",
        default=None,
        help="Optional JSON path for resolved panel metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    base_dir = manifest_path.parent
    rows_raw = manifest.get("rows", [])
    if not isinstance(rows_raw, list) or not rows_raw:
        raise ValueError("Manifest must contain non-empty 'rows'.")
    for idx, row in enumerate(rows_raw):
        if not isinstance(row, list) or not row:
            raise ValueError(f"rows[{idx}] must be a non-empty list.")

    aliases_raw = manifest.get("aliases", {})
    if aliases_raw is None:
        aliases_raw = {}
    if not isinstance(aliases_raw, dict):
        raise ValueError("'aliases' must be a mapping when provided.")
    aliases = {str(k): str(v) for k, v in aliases_raw.items()}

    error_cfg = _parse_error_config(manifest.get("error_map", {}))
    layout = _parse_layout_config(manifest.get("layout", {}))
    column_titles = [str(v) for v in manifest.get("column_titles", [])]
    row_titles = [str(v) for v in manifest.get("row_titles", [])]

    cache = SampleCache()
    if layout.cell_width is None or layout.cell_height is None:
        inferred_w, inferred_h = _infer_cell_size_from_specs(base_dir, rows_raw, aliases, cache)
        if layout.cell_width is None:
            layout.cell_width = inferred_w
        if layout.cell_height is None:
            layout.cell_height = inferred_h
    target_size = (int(layout.cell_width), int(layout.cell_height))

    err_maps = _collect_error_maps(base_dir, rows_raw, aliases, cache, error_cfg)
    if error_cfg.vmax is not None:
        shared_vmax = float(error_cfg.vmax)
    elif err_maps:
        flat = np.concatenate([m.reshape(-1) for m in err_maps]).astype(np.float32)
        shared_vmax = float(np.percentile(flat, float(error_cfg.percentile)))
    else:
        shared_vmax = 1.0
    shared_vmax = max(shared_vmax, 1e-8)

    export_cells_dir = Path(args.export_cells_dir).resolve() if args.export_cells_dir else None
    rendered_rows: List[List[Optional[np.ndarray]]] = []
    metadata_rows: List[List[Dict[str, Any]]] = []

    for r_idx, row in enumerate(rows_raw):
        rendered_row: List[Optional[np.ndarray]] = []
        metadata_row: List[Dict[str, Any]] = []
        for c_idx, cell in enumerate(row):
            if not isinstance(cell, dict):
                raise ValueError(f"rows[{r_idx}][{c_idx}] must be a mapping.")
            kind = str(cell.get("kind", "")).strip().lower()
            if kind not in {"gt", "pred", "err", "compare", "blank"}:
                raise ValueError(f"Unsupported cell kind at rows[{r_idx}][{c_idx}]: {kind}")

            image_u8 = _render_cell_image(
                base_dir,
                cell,
                aliases,
                cache,
                error_cfg=error_cfg,
                shared_err_vmax=shared_vmax,
                target_size=target_size,
            )
            rendered_row.append(image_u8)

            cell_meta: Dict[str, Any] = {
                "kind": kind,
                "row": int(r_idx),
                "col": int(c_idx),
                "crop": list(cell["crop"]) if "crop" in cell and cell.get("crop") is not None else None,
            }
            if "npz" in cell:
                cell_meta["npz"] = str(_resolve_alias_npz(base_dir, aliases, str(cell["npz"])))
            if kind == "err":
                cell_meta["error_space"] = str(cell.get("error_space", error_cfg.source_space))
                cell_meta["error_vmax"] = float(cell.get("error_vmax", shared_vmax))
                cell_meta["colormap"] = str(cell.get("colormap", error_cfg.colormap))
            if export_cells_dir is not None and image_u8 is not None:
                stem = f"r{r_idx:02d}_c{c_idx:02d}_{kind}.png"
                out_path = export_cells_dir / stem
                _save_png(out_path, image_u8)
                cell_meta["exported_png"] = str(out_path)
            metadata_row.append(cell_meta)
        rendered_rows.append(rendered_row)
        metadata_rows.append(metadata_row)

    canvas = _compose_canvas(
        rendered_rows=rendered_rows,
        column_titles=column_titles,
        row_titles=row_titles,
        layout=layout,
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_path))

    metadata = {
        "manifest": str(manifest_path),
        "output": str(output_path),
        "shared_error_vmax": float(shared_vmax),
        "error_config": {
            "source_space": error_cfg.source_space,
            "percentile": float(error_cfg.percentile),
            "vmax": (None if error_cfg.vmax is None else float(error_cfg.vmax)),
            "colormap": error_cfg.colormap,
        },
        "layout": {
            "cell_size": [int(target_size[0]), int(target_size[1])],
            "outer_padding": int(layout.outer_padding),
            "cell_gap": int(layout.cell_gap),
            "row_title_width": int(layout.row_title_width),
            "col_title_height": int(layout.col_title_height),
            "footer_height": int(layout.footer_height),
            "background": list(layout.background),
            "title_color": list(layout.title_color),
            "label_color": list(layout.label_color),
            "border_color": list(layout.border_color),
            "border_width": int(layout.border_width),
        },
        "column_titles": column_titles,
        "row_titles": row_titles,
        "rows": metadata_rows,
    }
    metadata_path = (
        Path(args.write_metadata).resolve()
        if args.write_metadata
        else output_path.with_suffix(".json")
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved panel: {output_path}")
    print(f"Saved metadata: {metadata_path}")
    if export_cells_dir is not None:
        print(f"Exported derived cells: {export_cells_dir}")


if __name__ == "__main__":
    main()
