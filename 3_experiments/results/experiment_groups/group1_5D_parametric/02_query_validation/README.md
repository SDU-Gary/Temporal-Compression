# Phase 02: Query Validation

**Timeline**: Week 4
**Goal**: Validate interpolation/extrapolation performance and query latency

## Experiment Structure

### interpolation/
- **Purpose**: Test model performance on interpolated lighting conditions
- **Test Cases**: Query points between training light configurations
- **Key Metrics**: PSNR, SSIM, MAE on interpolated queries
- **Files**: `interpolation_results.json`, analysis plots

### extrapolation/
- **Purpose**: Test model generalization beyond training distribution
- **Test Cases**: Extreme sun positions, edge conditions
- **Key Metrics**: Degradation analysis, error bounds
- **Files**: `extrapolation_results.json`, error analysis plots

### latency/
- **Purpose**: Measure query latency for real-time rendering viability
- **Test Cases**: Single query, batch query, different model sizes
- **Target**: <0.5ms per query for 60 FPS compatibility
- **Files**: Latency benchmarks, profiling results

## Key Findings

1. **Interpolation**: Smooth performance within training distribution
2. **Extrapolation**: Graceful degradation at distribution edges
3. **Query Latency**: Achieves real-time target (<0.5ms) for K≤30 Gaussians

## Related Documentation

- Report: [PG-GCPL/docs/reports/04_Query_Validation.md](../../docs/reports/04_Query_Validation.md)
- Original report: Week4-6_Query_Validation_and_Ablation.md

## Scripts

Evaluation:
- `PG-GCPL/src/scripts/evaluation/test_interpolation_query_physics.py`
- `PG-GCPL/src/scripts/evaluation/test_extrapolation_query_physics.py`
- `PG-GCPL/src/scripts/evaluation/test_query_latency_physics.py`
