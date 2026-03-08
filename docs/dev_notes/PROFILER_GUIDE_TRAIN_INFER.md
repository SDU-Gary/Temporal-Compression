# Training / Inference Profiler Guide

This guide introduces two profiling entrypoints:

- `tools/profile_training.py` (training-step micro profile)
- `tools/profile_inference.py` (model-level + pipeline-level inference)

Both scripts support:

- `torch.profiler` stack
- Nsight command generation (`nsys`, `ncu`)
- CUDA recommendation report output

---

## 1) Training profiler

### Quick dry-run

```bash
python -m tools.profile_training \
  --config 3_experiments/configs/bistro_clean_train.yaml \
  --stack both \
  --steps 200 \
  --warmup-steps 30 \
  --dry-run
```

### Real run (torch profiler)

```bash
python -m tools.profile_training \
  --config 3_experiments/configs/bistro_clean_train.yaml \
  --stack torch \
  --steps 200 \
  --warmup-steps 30 \
  --output-base 3_experiments/results/profiling
```

### Optional Nsight execution

```bash
python -m tools.profile_training \
  --config 3_experiments/configs/bistro_clean_train.yaml \
  --stack both \
  --run-nsys
```

Outputs include:

- `training_profile_summary.json`
- `training_step_breakdown.csv`
- `training_torch_trace.json` (when torch stack enabled)
- `training_torch_ops.csv`
- `cuda_recommendation_training.json`
- `nsys_command.sh`, `ncu_command.sh`

---

## 2) Inference profiler

### Quick dry-run

```bash
python -m tools.profile_inference \
  --checkpoint path/to/best_model.pt \
  --data-root 1_data_generation/output/bistro_clean_v2 \
  --profile-level both \
  --stack both \
  --dry-run
```

### Model-level real run

```bash
python -m tools.profile_inference \
  --checkpoint path/to/best_model.pt \
  --data-root 1_data_generation/output/bistro_clean_v2 \
  --profile-level model \
  --stack torch \
  --frames 300 \
  --warmup-frames 30
```

### Pipeline-level execution (Falcor)

```bash
python -m tools.profile_inference \
  --checkpoint path/to/best_model.pt \
  --data-root 1_data_generation/output/bistro_clean_v2 \
  --profile-level pipeline \
  --scene 1_data_generation/falcor/scenes/bistro_exterior_emissive_spheres.pyscene \
  --pipeline-run
```

Outputs include:

- `inference_model_summary.json`
- `inference_pipeline_summary.json`
- `inference_profile_report.json`
- `cuda_recommendation_inference.json`
- `nsys_command.sh`, `ncu_command.sh`

---

## 3) CUDA recommendation meaning

Decision field in recommendation JSON:

- `recommend`: bottleneck concentrated in a few CUDA ops, CUDA kernel work is likely high ROI.
- `defer`: first optimize non-kernel overhead (data waiting / Python overhead / flow).
- `not_needed`: end-to-end bottleneck is not model compute dominated.

---

## 4) Notes

- By default Nsight commands are generated but not executed unless `--run-nsys`/`--run-ncu` is provided.
- For stable comparisons, keep dataset/checkpoint/device fixed across runs.
- Use dry-run first to validate command wiring in your environment.
- Pipeline-level profiler now defaults to `--pipeline-route both` for consistency with GT-vs-Model benchmark checks.

