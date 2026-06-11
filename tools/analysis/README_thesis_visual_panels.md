# Thesis Visual Panels

This directory contains the post-processing step for thesis-ready comparison
figures. It is intentionally separated from rendering/benchmark code:

- Rendering/export stays on the existing Falcor benchmark chain.
- This step only consumes sampled `.npz` assets and composes figure panels.

## Main Script

- [compose_thesis_visual_panels.py](./compose_thesis_visual_panels.py)

## Input Assumptions

The script expects frame-level `.npz` files produced by one of these paths:

- `tools/export_compare_videos.py` benchmark exports
- `tools/benchmark_realtime_pipeline.py --save-sampled-images-dir ...`
- `tools/render_gbuffer_diff_frames.py --save-linear-npz`

Supported keys:

- GT:
  - `gt_srgb_u8`
  - `gt_srgb`
  - `gt_linear`
- Pred:
  - `model_srgb_u8` or `pred_srgb_u8`
  - `model_srgb` or `pred_srgb`
  - `model_linear` or `pred_linear`
- Err:
  - `err_srgb`
  - `err_linear`
  - or computed from GT/Pred if missing

## Manifest Format

Example:

```json
{
  "aliases": {
    "a2_peak": "../../3_experiments/results/thesis_visuals/20260427/a2_peak_f23/sampled_npz/frame_0023.npz",
    "a2_tail": "../../3_experiments/results/thesis_visuals/20260427/a2_tail_f23/sampled_npz/frame_0023.npz",
    "a22a_tail": "../../3_experiments/results/thesis_visuals/20260427/a22a_tail_f23/sampled_npz/frame_0023.npz"
  },
  "column_titles": ["GT", "A2 Peak", "A2 Tail", "A2.2-A Tail"],
  "row_titles": ["Pred", "Err"],
  "layout": {
    "cell_size": [640, 360],
    "outer_padding": 28,
    "cell_gap": 16,
    "row_title_width": 180,
    "col_title_height": 44
  },
  "error_map": {
    "source_space": "srgb",
    "percentile": 99.0,
    "colormap": "turbo"
  },
  "rows": [
    [
      {"kind": "gt", "npz": "a2_peak"},
      {"kind": "pred", "npz": "a2_peak"},
      {"kind": "pred", "npz": "a2_tail"},
      {"kind": "pred", "npz": "a22a_tail"}
    ],
    [
      {"kind": "blank"},
      {"kind": "err", "npz": "a2_peak"},
      {"kind": "err", "npz": "a2_tail"},
      {"kind": "err", "npz": "a22a_tail"}
    ]
  ]
}
```

Notes:

- Paths in `aliases` and cells may be relative to the manifest file.
- `crop: [x0, y0, x1, y1]` is optional on any non-blank cell.
- If all 4 crop values are in `[0, 1]`, they are treated as normalized coords.
- Otherwise they are treated as pixel coords.
- Error normalization is shared across all `err` cells in the same panel unless
  a cell explicitly sets `error_vmax`.

## Typical Usage

```bash
python tools/analysis/compose_thesis_visual_panels.py \
  --manifest tools/analysis/examples/stability_panel_frame23.json \
  --output 4_thesis/figures/fig5_x_stability_frame23.png \
  --export-cells-dir 3_experiments/results/thesis_visuals/20260427/stability_cells
```

## Recommended Workflow

1. Use `tools/export_compare_videos.py` to export single-frame `sampled_npz`.
2. Prepare a small manifest JSON describing the target layout.
3. Run `compose_thesis_visual_panels.py`.
4. Use the generated `*.json` metadata to keep figure provenance.
