# P0 Experiments Final Report

**Generated**: 2025-12-12 11:22:39

## Summary

**Completed Experiments**: 4/4

### Experiment F: Ground Truth Validation

- **Status**: ✓ Completed
- **Configurations Rendered**: 4
- **Average E_generalization**: 0.2712
- **Average E_tpe_assumption**: 0.3048
- **Average E_total**: 0.3590
- **TPE Contribution**: 84.9%
- **Decision**: ✗ INVALID

### Experiment 2: Residual Analysis

- **Status**: ✓ Completed
- **Mean |residual|**: 0.0461
- **TPE Contribution**: 84.9%
- **Threshold**: 5.0%
- **Hybrid Model Needed**: YES

### Experiment 3: Spatial Generalization

- **Status**: ✓ Completed
- **Probes Tested**: 10
- **Successful**: 0/10
- **Passed All Criteria**: 0/10
- **Pass Rate**: 0.0%
- **Decision**: ✗ Not Confirmed

### Experiment 4: Baseline Comparison

- **Status**: ✓ Completed
- **Baselines Tested**: 4
- **Ranking by MAE**:
  1. spline: 0.0312
  2. direct_mlp: 0.0458
  3. tdml: 0.0725

## Overall Conclusion

✓ All P0 experiments completed successfully.

## Next Steps

- Consider alternative approaches or TPE modifications
- Consider hybrid TPE + residual model
