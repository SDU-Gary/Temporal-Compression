# 3_EXPERIMENTS KNOWLEDGE BASE

**Generated**: 2026-02-05 19:29:00
**Commit**: c18004a
**Branch**: main

## OVERVIEW
Stage 3: Experimentation and evaluation for hierarchical neural compression of multi-temporal lighting.

## STRUCTURE
```
3_experiments/
├── configs/                    # YAML configuration files
│   ├── bistro_clean_train.yaml # Mainline FiLM unified_set config (K30_r8)
│   └── template_unified_set.yaml
├── scripts/                    # Experiment execution scripts
│   ├── training/               # Training scripts and utilities
│   ├── analysis/               # Diagnostic and analysis tools
│   └── train.py                # Unified training entry point
└── results/                    # Experiment outputs
    └── experiment_groups/      # Hierarchical experiment organization
        ├── group1_5D_parametric/
        └── group2_intensity_modulation/
```

## EXPERIMENT GROUPS
**Hierarchical Organization**: Experiments organized by method and chronological progression:
- `group1_5D_parametric/`: 5D parametric Gaussian-Physics compression
- `group2_intensity_modulation/`: Intensity modulation methods

**Phase Numbering**: Chronological progression with 01_, 02_, 03_, 04_ prefixes:
- `01_initial_exploration/`: Initial method validation
- `02_query_validation/`: Query performance testing
- `03_optimization/`: Hyperparameter optimization
- `04_ablation_and_visualization/`: Ablation studies and visualization

## CONFIGURATION SYSTEM
**YAML-Based**: All experiments configured via YAML files in `configs/`.

**Key Configurations**:
- `bistro_clean_train.yaml`: Mainline config with optimal K30_r8 parameters
- `template_unified_set.yaml`: Template for new experiments

**Modification Workflow**:
1. Copy existing config: `cp configs/bistro_clean_train.yaml configs/my_experiment.yaml`
2. Modify parameters (Gaussian count, rank, learning rates)
3. Run: `python scripts/train.py --config configs/my_experiment.yaml`

## SCRIPTS ORGANIZATION
**Training** (`scripts/training/`):
- Model training with various loss functions
- Checkpoint management and resumption
- Learning rate scheduling

**Analysis** (`scripts/analysis/`):
- Training diagnostics and convergence analysis
- Ablation study execution


## RESULTS STRUCTURE
**Deep Nested Hierarchy**: Results organized for systematic analysis:
```
group1_5D_parametric/
├── 01_initial_exploration/
├── 02_query_validation/
│   ├── latency/
│   ├── interpolation/
│   └── extrapolation/
├── 03_optimization/
└── 04_ablation_and_visualization/
    ├── ablation/
    │   ├── K30_r8/            # Optimal: 30 Gaussians, rank 8
    │   ├── K20_r8/            # 20 Gaussians comparison
    │   └── K30_r5/            # Lower rank comparison
    └── visualization/
        ├── radiance_field/
        ├── scene_rendering/
        └── sample_XXX/        # Individual probe visualizations
```

## CONVENTIONS
**K{r}_r{r} Notation**: Gaussian count (K) and rank (r) specification:
- `K30_r8`: 30 Gaussians, rank 8 (optimal configuration)
- `K20_r8`: 20 Gaussians, rank 8 (reduced capacity)
- `K30_r5`: 30 Gaussians, rank 5 (lower rank)

**Experiment Naming**: Descriptive names indicating method and purpose:
- `visual_validation_K30_r8.py`: Visual validation for optimal configuration
- `run_film_probe_ablation.sh`: Film probe ablation study script
- `diagnose_training.py`: Training convergence diagnostics

**Performance Targets**:
- Compression ratio: 1:17 (16.59× achieved with K30_r8)
- PSNR: >38 dB (22.38 dB baseline, optimization ongoing)
- Decompression latency: <0.5 ms
- Frame rate: 60 FPS @ 1080p

## WORKFLOW
1. **Configure**: Create/modify YAML config in `configs/`
2. **Train**: `python scripts/train.py --config configs/my_config.yaml`
3. **Evaluate**: (TODO) reintroduce mainline `scripts/eval.py`
4. **Analyze**: Use analysis tools to diagnose performance
5. **Visualize**: Generate visualizations for paper/thesis
6. **Organize**: Store results in appropriate hierarchical location

## NOTES
**Optimal Configuration**: K30_r8 yields 16.59× compression ratio with 22.38 dB PSNR.

**Current Baseline**: EXP-20260103-002 (Phase2_PGCPL) in `group1_5D_parametric/`.

**Shell Scripts**: Use `run_*.sh` scripts for complex multi-step experiments.

**Database Integration**: All experiments logged to SQLite database (`project.db`) via `tools/logexp.py`.
