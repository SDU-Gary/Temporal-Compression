# TPE Method (Temporal Perturbation Embedding) - ARCHIVED

**Status**: Failed Method (Archived for reference)
**Timeline**: Week 1 (Early exploration)
**Reason for Failure**: 84.9% error contribution, implicit neural representation unsuitable for smooth temporal data

## Method Overview

### Core Idea (Failed)
Use Temporal Perturbation Embedding (TPE) to encode time-varying lighting:
1. Learn base latent code for each spatial location
2. Perturb latent code based on time/sun position
3. Decode perturbed latent to SH coefficients

### Why It Failed

**Key Finding**: For smooth temporal lighting data, **explicit grid + interpolation >> implicit neural representations**

**Error Analysis** (Cornell Box):
- Total MAE: 1.0613
- TPE contribution: 0.9005 (84.9%)
- Decoder contribution: 0.1608 (15.1%)
- **Conclusion**: TPE encoding is the primary bottleneck

**Fundamental Issues**:
1. **Smoothness Mismatch**: Temporal lighting changes are inherently smooth (solar cycle), but TPE uses discrete perturbation vectors
2. **Overfitting**: High-frequency perturbations fit noise instead of underlying solar pattern
3. **Lack of Physical Prior**: No knowledge of physics → poor generalization

## Archived Contents

```
archive/TPE_Method_Archived/
├── code/
│   ├── run_p0_experiments.py
│   ├── validate_tpe_level1.py
│   ├── validate_tpe_level2.py
│   ├── validate_tpe_level2_directional.py
│   ├── tpe_dataset_loader.py
│   ├── p0_experiments_config.yaml
│   ├── p0_experiments_config_house.yaml
│   └── baselines/                          # Baseline comparison scripts
├── docs/
│   ├── 01_TPE_Theory_and_Motivation.md
│   ├── 02_P0_Experiment_Design.md
│   ├── 03_Cornell_Box_Results.md
│   ├── 04_House_Scene_Comparison.md
│   ├── 05_Baseline_Methods_Analysis.md
│   ├── 06_Key_Findings_and_Insights.md
│   └── 07_Future_Work_and_Recommendations.md
└── experiments/
    ├── p0_results/                         # Cornell Box results
    └── p0_results_house/                   # House scene results
```

## Key Learnings from Failure

### 1. Smooth Data Needs Explicit Smoothness
**Problem**: TPE uses discrete perturbation vectors with no smoothness constraints
**Solution**: Use trigonometric basis functions [cos(θ), sin(θ), 1] → inherently smooth

### 2. Physics Priors Matter
**Problem**: TPE has no knowledge of solar geometry
**Solution**: Compute sun direction from astronomical equations → PG-GCPL method

### 3. Implicit ≠ Always Better
**Problem**: Implicit neural representations (INR) are popular but not always optimal
**Insight**: For smooth, low-dimensional temporal dynamics, explicit factorization works better
**Evidence**: Spline interpolation (explicit) outperformed TPE (implicit) by 70%

### 4. Error Attribution Is Critical
**Method**: Decompose total error into component contributions
**Result**: Identified TPE as 84.9% of error → focused effort on fixing temporal encoding
**Outcome**: Led to physics-guided low-rank method (PG-GCPL)

## Comparison: TPE vs PG-GCPL

| Aspect | TPE (Failed) | PG-GCPL (Success) |
|--------|-------------|-------------------|
| **Temporal Encoding** | Discrete perturbations | Trigonometric basis |
| **Physical Prior** | None | Solar geometry |
| **Smoothness** | Not enforced | Inherent |
| **Interpretability** | Black box | Clear physics |
| **MAE (Cornell Box)** | 1.0613 | ~0.3 (70% improvement) |
| **Compression** | Moderate | 16.59× |

## Documentation

### Experiment Reports (Archived)
1. **01_TPE_Theory_and_Motivation.md**: Theoretical foundation (flawed)
2. **02_P0_Experiment_Design.md**: Experiment setup
3. **03_Cornell_Box_Results.md**: Failure analysis
4. **04_House_Scene_Comparison.md**: Cross-scene validation
5. **05_Baseline_Methods_Analysis.md**: KNN, spline comparisons
6. **06_Key_Findings_and_Insights.md**: What we learned
7. **07_Future_Work_and_Recommendations.md**: Path to PG-GCPL

### Key Findings Summary (from Report 06)

**Finding 1**: Grid interpolation beats implicit neural representations for smooth data
- Spline interpolation: MAE 0.2985
- TPE method: MAE 1.0613
- **3.6× worse performance**

**Finding 2**: Temporal encoding is the bottleneck (84.9% error)
- Decoder MLP performs well (15.1% error)
- Focus improvement on temporal encoding → led to physics basis

**Finding 3**: Cross-scene generalization requires physics priors
- TPE overfits to specific solar patterns per scene
- Physics-guided method generalizes better (verified in PG-GCPL Phase 04)

## Path Forward (Historical Context)

After TPE failure, we pursued:
1. **Phase 1**: Physics-guided low-rank for single probe (2.34× compression, +33.5% quality)
2. **Phase 2**: Gaussian-physics hybrid for multi-probe (16.59× compression)
3. **Phase 3**: 5D parametric space extension
4. **Phase 4**: Ablation and visual validation (SSIM 0.951)

**Result**: PG-GCPL method successfully addresses all TPE limitations

## Lessons for Future Work

1. **Analyze error sources** before optimizing entire system
2. **Use physics priors** when available (solar geometry, energy conservation)
3. **Question popular paradigms** (implicit neural representations not universal)
4. **Value explicit methods** for smooth, low-dimensional data
5. **Archive failures** with detailed analysis → guides future research

## Related Methods

- **TemporalMLP** (Deprecated): [../../TemporalMLP/README.md](../../TemporalMLP/README.md)
- **PG-GCPL** (Success): [../../PG-GCPL/README.md](../../PG-GCPL/README.md)
- **Week3-4 MLP Hybrid** (Also Failed): [../Week3-4_MLP_Hybrid_Failed/](../Week3-4_MLP_Hybrid_Failed/)

## Citation

If referencing this failure analysis in papers:
```
[Thesis work in progress - shows importance of physics priors over black-box INR]
```

---

**Archive Date**: 2026-01-03
**Reason**: Method fundamentally flawed, replaced by physics-guided approach
**Preservation**: For educational purposes and failure analysis reference
