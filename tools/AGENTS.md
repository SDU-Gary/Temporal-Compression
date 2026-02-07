# TOOLS DIRECTORY - Project Utilities & CLI Tools

## OVERVIEW
Command-line utilities and workflow orchestration tools for the multi-temporal lighting compression project.

## STRUCTURE
```
tools/
├── logexp.py                    # Experiment logging CLI (most used)
├── run_pipeline.py             # Workflow orchestration
├── run_dataset.py              # Dataset generation orchestration
├── (removed)                  # summarize_state.py (historical)
├── db_utils.py                 # Database connection utilities
├── workflow_config.py          # Python environment management
├── run_summary.py              # Pipeline summary generation
├── fix_database_errors.py      # Database repair utilities
├── (removed)                  # import_existing_data.py (historical)
├── manifest_utils.py           # Dataset manifest handling
├── median_filter_dataset.py    # Data preprocessing
├── ema_smooth_dataset.py       # Temporal smoothing
├── sample_surface_probes.py    # Probe sampling utilities
├── probe_sampling.py           # Advanced probe sampling
├── run_probe_diagnostics.py    # Dataset validation
├── run_parallel_bistro_bakes.py # Parallel rendering
├── run_quick_bistro_experiments.py # Quick experiments
├── render_probe_video.py       # Probe visualization
├── render_probe_video_gpu.py   # GPU-accelerated rendering
├── render_single_envmap_frame.py # Single frame rendering
├── render_bistro_preview.py    # Scene preview rendering
├── sh_probe_volume.cs.slang    # Falcor shader for probe volumes
└── blender/                    # Blender integration
    ├── scene_export.py         # Scene export utilities
    └── export_emissive_trajectories.py # Sun trajectory export
```

## KEY TOOLS

### DATABASE TOOLS
- **logexp.py**: CLI for experiment logging, task management, and baseline setting
- **db_utils.py**: SQLite database connection utilities (project.db)
- **fix_database_errors.py**: Database repair and validation utilities

### PIPELINE TOOLS
- **run_pipeline.py**: Orchestrates dataset → train → eval workflows from YAML configs
- **run_dataset.py**: Dataset generation orchestration with dependency tracking
- **workflow_config.py**: Python environment and dependency management
- **run_summary.py**: Generates pipeline execution summaries

### RENDERING TOOLS
- **render_probe_video.py**: Creates visualization videos from probe data
- **render_bistro_preview.py**: Generates scene preview renders
- **render_single_envmap_frame.py**: Renders individual environment map frames
- **sh_probe_volume.cs.slang**: Falcor shader for probe volume visualization

### DATA PROCESSING
- **median_filter_dataset.py**: Applies median filtering to clean probe data
- **ema_smooth_dataset.py**: Exponential moving average temporal smoothing
- **sample_surface_probes.py**: Surface probe sampling utilities
- **probe_sampling.py**: Advanced probe sampling algorithms
- **run_probe_diagnostics.py**: Validates dataset integrity and quality

## USAGE PATTERNS

### Experiment Logging
```bash
python tools/logexp.py log --phase Phase2_PGCPL --stage training \
  --script train_pgcpl5d --dataset 5D_param_val \
  --hyperparams '{"num_gaussians": 30, "rank": 8}' \
  --results '{"psnr": 32.5, "ssim": 0.951, "compress_ratio": 16.59}'
```

### Pipeline Orchestration
```bash
python tools/run_pipeline.py --config pipelines/pgcpl_bistro_dev.yaml
```

### Dataset Processing
```bash
python tools/median_filter_dataset.py --input probes.npz --output probes_filtered.npz
python tools/ema_smooth_dataset.py --input probes.npz --alpha 0.3
```

## CONVENTIONS
- **Database-first**: All experiments logged to SQLite (project.db)
- **YAML-driven**: Pipelines configured via YAML files in `pipelines/` directory
- **Context persistence**: Prefer writing concise docs under `docs/` and logging experiments via `tools/logexp.py`
- **Blender integration**: Scene export utilities in `blender/` subdirectory
- **Parallel execution**: Tools support parallel processing for large datasets
- **GPU acceleration**: Rendering tools have GPU-accelerated variants

## MOST FREQUENTLY USED
1. `logexp.py` - Daily experiment tracking
2. `run_pipeline.py` - Workflow automation
3. `render_probe_video.py` - Result visualization
4. `median_filter_dataset.py` - Data preprocessing
