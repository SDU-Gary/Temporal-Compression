# PROJECT KNOWLEDGE BASE

**Generated**: 2026-02-05 19:28:44
**Commit**: c18004a
**Branch**: main

## OVERVIEW
Graduate thesis project on hierarchical neural compression for multi-temporal lighting. Extends Gaussian Compression (SIGGRAPH'25) to multi-moment scenarios using time-varying latent codes, Gaussian mixtures, and spatiotemporal cascaded volumes. Target: 1:17 compression ratio, >38dB PSNR, <0.5ms decompression.

**Core Stack**: Python 3.13, PyTorch 2.9.1, Mitsuba 3.7.3, Falcor rendering.

## STRUCTURE
```
./
├── 1_data_generation/          # Stage 1: Dataset creation (Mitsuba/Falcor)
├── 2_src/                      # Stage 2: Core algorithms (models, training, utils)
│   ├── models/                 # Neural networks (Gaussian-Physics, low-rank)
│   ├── training/               # Loss functions, trainers, metrics
│   ├── data/                   # Dataset loaders and preprocessing
│   ├── utils/                  # Shared utilities (logging, rendering, coords)
│   └── tests/                  # Unit and integration tests
├── 3_experiments/              # Stage 3: Experimentation and evaluation
│   ├── configs/                # YAML configuration files
│   ├── scripts/                # Training/evaluation/analysis scripts
│   └── results/                # Experiment outputs (logs, checkpoints, viz)
├── 4_thesis/                   # Stage 4: Thesis materials (reports, figures)
├── tools/                      # Project utilities and CLI tools
├── docs/                       # Documentation and references
│   ├── dev_notes/              # Development guides (CLAUDE.md, workflows)
│   ├── mitsuba/                # Local Mitsuba documentation
│   └── literature_review/      # Research paper analyses
├── archive/                    # Historical/deprecated methods
├── metadata/                   # Project database (metadata/project.db)
└── pipelines/                  # Workflow pipeline definitions
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Understand project | `docs/dev_notes/CLAUDE.md` | 588-line comprehensive guide |
| Run experiments | `3_experiments/scripts/` | Training/evaluation entry points |
| Configure models | `3_experiments/configs/` | YAML configuration files |
| Core algorithms | `2_src/models/` | Gaussian-Physics, low-rank implementations |
| Data generation | `1_data_generation/` | Mitsuba/Falcor rendering scripts |
| Utility functions | `2_src/utils/` | Logging, rendering, coordinate transforms |
| Experiment tracking | `tools/logexp.py` | Database-driven experiment logging |
| Pipeline workflows | `pipelines/` | YAML pipeline definitions |

## CODE MAP
**Key Models**:
- `gaussian_physics_unified.py`: Current mainline unified model (LightSetEncoder + Gaussian routing + FiLM low-rank)
- `gaussian_physics_trainer.py`: Training infrastructure with multi-loss + EMA/staged/oracle/coeff-suite support
- `lightset_dataset.py`: Mainline dataset loader for `parametric_tensor.npz`
- `train.py`: Unified training entry with YAML + staged training

**Critical Utilities**:
- `config.py`: YAML configuration loading with validation
- `logger.py`: Rich-formatted logging with file/console output
- `rendering_utils.py`: SH reconstruction and cubemap processing
- `spherical_harmonics.py`: SH fitting and reconstruction

**Entry Points**:
- `1_data_generation/generate_dataset.py`: Main dataset generation
- `3_experiments/scripts/train.py`: Unified training entry point
- `tools/run_pipeline.py`: Pipeline orchestration
- `tools/run_dataset.py`: Dataset generation orchestration

## CONVENTIONS
**Workflow-first**: Numbered directories (1_, 2_, 3_, 4_) indicate sequential workflow stages.

**YAML Configuration**: All experiments configured via YAML files in `3_experiments/configs/`.

**Chinese Documentation**: Most documentation in Chinese with bilingual code comments.

**Database-driven Experiment Tracking**: All experiments logged to SQLite database (`metadata/project.db`).

**No Traditional CI/CD**: Custom shell scripts (`run_*.sh`) instead of GitHub Actions/Makefile.

**Virtual Environment**: Use `source venv/bin/activate` but no `requirements.txt`.

## ANTI-PATTERNS (THIS PROJECT)
1. **Never guess dataset paths** - Always query database or check `docs/dev_notes/project_state.md`
2. **Never modify BSDF physical realism parameters** in Mitsuba (specular_reflectance/transmittance)
3. **Never use deprecated TemporalMLP methods** - Use PG-GCPL (Physics-Guided Gaussian-Physics Compression)
4. **Never bypass virtual environment activation**
5. **Never confuse experimental methods with target architecture** - Refer to `4_thesis/report/多时刻光照压缩任务书.md`
6. **Never use deprecated Mitsuba API methods** (put(), props.keys(), del props[key])

## UNIQUE STYLES
**Research Workflow Organization**: Clear separation: data → source → experiments → thesis.

**Deep Experiment Hierarchy**: Nested results structure: `group1_5D_parametric/04_ablation_and_visualization/ablation/K30_r8/`

**K{r}_r{r} Notation**: Gaussian count (K) and rank (r) notation (e.g., K30_r8 = 30 Gaussians, rank 8).

**Physics-Guided Compression**: Integration of astronomical sun position calculations with neural compression.

**Legacy Preservation**: Failed methods archived with detailed failure analysis (TPE: 84.9% error contribution).

## COMMANDS
```bash
# Activate environment
source venv/bin/activate

# Generate dataset
cd 1_data_generation
python generate_dataset.py

# Train model
cd 3_experiments
python scripts/train.py --config configs/bistro_clean_train.yaml

# Run pipeline
python tools/run_pipeline.py --config pipelines/pgcpl_bistro_dev.yaml

# Log experiment
python tools/logexp.py log --phase Phase2_PGCPL --stage training \
  --script train_pgcpl5d --dataset 5D_param_val \
  --hyperparams '{"num_gaussians": 30, "rank": 8}' \
  --results '{"psnr": 32.5, "ssim": 0.951, "compress_ratio": 16.59}'

# Query database
python tools/logexp.py query --phase Phase2_PGCPL --limit 5

# Run tests
cd 2_src
pytest tests/
```

## NOTES
**Optimal Configuration**: K30_r8 (30 Gaussians, rank 8) yields 16.59× compression ratio.

**Current Baseline**: EXP-20260103-002 (Phase2_PGCPL): PSNR 22.38 dB, SSIM 0.951.

**Active Tasks**: 7 pending (energy conservation loss, 4×3 cascaded volumes, 10-bit quantization, etc.).

**Dataset Reality Check**: Config files may declare incorrect probe counts - always trust actual `probes.npz` data.

**Memory System**: SQLite database (`metadata/project.db`) maintains experiment history across conversation compactions. `tools/summarize_state.py` has been removed; use `tools/logexp.py query ...` or open `docs/dev_notes/project_state.md` snapshot.

**Performance Targets**: 1:17 compression ratio, >38dB PSNR, <0.5ms decompression, 60 FPS @ 1080p.

**Critical Files**: 
- `4_thesis/report/多时刻光照压缩任务书.md`: Authoritative task specification
- `docs/dev_notes/CLAUDE.md`: Comprehensive development guide
- `docs/dev_notes/project_state.md`: Historical project context snapshot (manual refresh only)
- `DATABASE_VERIFICATION_REPORT.md`: Database validation methodology
