# P0 Experiments Final Report

**Generated**: 2025-12-12 14:51:29

## Summary

**Completed Experiments**: 4/4

### Experiment F: Ground Truth Validation

- **Status**: ✓ Completed
- **Configurations Rendered**: 4
- **Average E_generalization**: 0.0008
- **Average E_tpe_assumption**: 0.0018
- **Average E_total**: 0.0019
- **TPE Contribution**: 96.5%
- **Decision**: ✗ INVALID

### Experiment 2: Residual Analysis

- **Status**: ✓ Completed
- **Mean |residual|**: 0.0003
- **TPE Contribution**: 96.5%
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
  1. spline: 0.0002
  2. direct_mlp: 0.0003
  3. tdml: 0.0158

## Overall Conclusion

✓ All P0 experiments completed successfully.

## Next Steps

- Consider alternative approaches or TPE modifications
- Consider hybrid TPE + residual model
