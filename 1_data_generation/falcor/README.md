# Falcor Data Generation (Full Pipeline)

This folder contains Falcor-based dataset generation for the PG-GCPL pipeline.
Unlike the Mitsuba generators, all rendering here uses Falcor's Python bindings.

## Quick Start

1) Build Falcor and enable Python bindings:
   - See: `Falcor/docs/falcor-in-python.md`
   - Ensure your environment can `import falcor`

2) Generate the 5D parametric dataset:

```bash
python 1_data_generation/falcor/generate_5d_parametric_falcor.py \
  --output 1_data_generation/output/5D_parametric_falcor \
  --scene 1_data_generation/falcor/scenes/cornell_box_sun.pyscene \
  --grid-resolution 5 \
  --num-sh-samples 64 \
  --spp 64
```

3) Generate the 1D intensity modulation dataset:

```bash
python 1_data_generation/falcor/generate_intensity_modulation_falcor.py \
  --output-dir 1_data_generation/output/intensity_modulation_falcor \
  --scene 1_data_generation/falcor/scenes/cornell_box_sun.pyscene \
  --grid-resolution 7 \
  --num-moments 12 \
  --num-sh-samples 64 \
  --spp 64
```

4) Generate a multi-light dataset (sun + point light):

```bash
python 1_data_generation/falcor/generate_multilight_falcor.py \
  --output 1_data_generation/output/multilight_falcor \
  --scene 1_data_generation/falcor/scenes/cornell_box_sun_point.pyscene \
  --grid-resolution 5 \
  --num-sh-samples 64 \
  --spp 64
```

For a quick smoke test:

```bash
python 1_data_generation/falcor/generate_5d_parametric_falcor.py \
  --output 1_data_generation/output/5D_parametric_falcor_quick \
  --grid-resolution 2 \
  --num-sh-samples 16 \
  --spp 8 \
  --max-configs 3 \
  --max-probes 8
```

Quick 1D smoke test:

```bash
python 1_data_generation/falcor/generate_intensity_modulation_falcor.py \
  --output-dir 1_data_generation/output/intensity_modulation_falcor_quick \
  --grid-resolution 2 \
  --num-moments 3 \
  --num-sh-samples 16 \
  --spp 8 \
  --max-probes 8
```

Linearity sanity check:

```bash
python 1_data_generation/falcor/linearity_check.py \
  --scene 1_data_generation/falcor/scenes/cornell_box_sun_point.pyscene \
  --num-sh-samples 32 \
  --spp 32
```

## Notes
- The Cornell Box scene used here (`cornell_box_sun.pyscene`) includes a
  `DistantLight('Sun')` whose direction/intensity are updated per config.
- `sun.direction` convention may require a sign flip depending on Falcor
  version; verify with a quick render if the sun appears inverted.
- Output format matches existing PG-GCPL datasets:
  `parametric_tensor.npz` with `tensor`, `probe_positions`, `light_configs`.
