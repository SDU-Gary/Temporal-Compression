# Phase 04: Ablation and Visualization

**Timeline**: Week 6
**Goal**: Ablation study across hyperparameters and comprehensive visual validation

## Experiment Structure

### ablation/
- **Purpose**: Systematic ablation study on Gaussian count (K) and rank (r)
- **Configurations**: 6 configurations total
  - K20_r5: 20 Gaussians, rank 5
  - K20_r8: 20 Gaussians, rank 8
  - K30_r5: 30 Gaussians, rank 5
  - K30_r8: 30 Gaussians, rank 8 ✓ (Best trade-off)
  - K40_r5: 40 Gaussians, rank 5
  - K40_r8: 40 Gaussians, rank 8
- **Files**:
  - Individual config folders with checkpoints
  - `ablation_summary.json` - Consolidated results
  - `ablation_analysis.png` - Performance visualization

### visualization/
- **Purpose**: Multi-level visual validation of reconstruction quality
- **Subfolders**:
  - `sh_validation/` - SH coefficient reconstruction comparison
  - `radiance_field/` - 3D radiance field visualization
  - `scene_rendering/` - Cornell Box rendering with GT vs Pred lighting
  - `rendering_comparison/` - Environment map rendering

## Ablation Results Summary

| Config | Params | Compression | PSNR | SSIM | MAE | Notes |
|--------|--------|-------------|------|------|-----|-------|
| K20_r5 | 6,900 | 20.05× | - | - | - | Underfitting |
| K20_r8 | 7,920 | 17.47× | - | - | - | - |
| K30_r5 | 7,350 | 18.83× | - | - | - | - |
| **K30_r8** | **8,340** | **16.59×** | **Best** | **Best** | **Best** | **Recommended** |
| K40_r5 | 7,800 | 17.74× | - | - | - | - |
| K40_r8 | 9,360 | 14.78× | - | - | - | Overfitting risk |

## Visual Validation Results

### SH Validation
- Direct comparison of GT vs Predicted SH coefficients
- Per-coefficient error analysis
- Best case: Sample 121 (PSNR 25.28 dB, SSIM 0.986)

### Scene Rendering (Cornell Box)
- **Average PSNR**: 22.38 ± 3.54 dB
- **Average SSIM**: 0.951 ± 0.046
- **Average MAE**: 0.004440
- Demonstrates: Model captures lighting effects for realistic rendering

### Key Findings

1. **K30_r8 Optimal**: Best balance of compression vs quality
2. **Scene-Level Quality**: SSIM >0.95 shows perceptually accurate lighting
3. **Visual Fidelity**: Rendered scenes nearly indistinguishable from GT
4. **Compression Success**: 16.59× with minimal perceptual loss

## Related Documentation

- Report: [PG-GCPL/docs/reports/04_Query_Validation.md](../../docs/reports/04_Query_Validation.md)
- Report: [PG-GCPL/docs/reports/05_Final_Validation.md](../../docs/reports/05_Final_Validation.md)
- Original reports: Week4-6_Query_Validation_and_Ablation.md, Week6_Final_Validation_Report.md

## Scripts

Evaluation:
- `PG-GCPL/src/scripts/evaluation/ablation_gaussian_physics_5D.py`

Visualization:
- `PG-GCPL/src/scripts/visualization/visual_validation_K30_r8.py`
- `PG-GCPL/src/scripts/visualization/render_comparison_K30_r8.py`
- `PG-GCPL/src/scripts/visualization/render_scene_comparison_K30_r8.py`
- `PG-GCPL/src/scripts/visualization/visualize_sh_radiance.py`
- `PG-GCPL/src/scripts/visualization/analyze_rendering_quality.py`
