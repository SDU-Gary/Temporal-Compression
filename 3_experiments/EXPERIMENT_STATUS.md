# K50_r16_plain Experiment Quick Reference

## Current Status
- **Training**: Epoch 38/2000 (1.9%)
- **ETA**: ~8-10 hours
- **Session**: `train_k50_r16_plain`

## When Training Completes

### Step 1: Auto-Evaluate
```bash
cd /home/kyrie/毕设
bash 3_experiments/auto_eval_k50_r16_plain.sh
```

This will:
- Evaluate SH coefficient errors
- Benchmark rendered image PSNR
- Display comparison with baseline
- Suggest next experiment

### Step 2: Decision Tree

**If PSNR >= 25 dB** (SUCCESS):
```bash
mkdir -p 3_experiments/results/bistro_clean_v2/unified_set_K50_r16_weighted_fixed
tmux new-session -d -s train_weighted_fixed "source venv/bin/activate && python 3_experiments/scripts/train.py --config 3_experiments/configs/bistro_clean_train_k50_r16_weighted_fixed.yaml 2>&1 | tee 3_experiments/results/bistro_clean_v2/unified_set_K50_r16_weighted_fixed/train_$(date +%Y%m%d_%H%M%S).log"
```

**If PSNR 20-25 dB** (MARGINAL):
- Capacity helps slightly
- Consider trying Experiment B cautiously
- Or explore other architectures

**If PSNR < 20 dB** (FAILURE):
- Capacity doesn't help
- Need fundamental architecture changes
- Consider: attention mechanisms, hierarchical structure, or different basis functions

## Monitoring Commands

**Check progress**:
```bash
tmux attach -t train_k50_r16_plain
# Detach: Ctrl+B then D
```

**Quick peek**:
```bash
tmux capture-pane -t train_k50_r16_plain -p | tail -20
```

**Check if training finished**:
```bash
ls -lh 3_experiments/results/bistro_clean_v2/unified_set_K50_r16_plain/best_model.pt
```

## Experiment Pipeline

1. **A: K50_r16_plain** (RUNNING)
   - Capacity only, no optimizations
   - Target: 25-30 dB

2. **B: K50_r16_weighted_fixed** (READY)
   - A + corrected weights [1.0, 1.5, 1.5, 1.5, 2.0×5]
   - Target: 27-35 dB

3. **C: K50_r16_full_fixed** (READY)
   - B + sRGB image loss (256 samples, λ=0.01)
   - Target: 35+ dB ✓

## Key Findings

**K50_r16 with "optimizations" FAILED**:
- Weighted loss [3.0, 1.0, 1.0, 1.0, 0.5×5] was backwards
- Image loss (64 dirs, linear) misaligned with eval
- Result: 15.6 dB (worse than 20-25 dB baseline)

**Root Cause**:
- Under-weighted directional terms (L2)
- Over-weighted DC component (L0)
- Training metric ≠ evaluation metric

## Contact
Training will continue in tmux even if SSH disconnects.
Check back in ~8-10 hours to evaluate results.
