# 1_DATA_GENERATION KNOWLEDGE BASE

**Generated**: 2026-02-05
**Directory**: Stage 1 - Dataset Generation

## OVERVIEW
Multi-temporal lighting dataset generation using Mitsuba 3 and Falcor rendering engines for neural compression research.

## STRUCTURE
```
1_data_generation/
├── core/                    # Core rendering logic
│   ├── sh_baker.py         # SH coefficient baking
│   ├── probe_sampler.py    # Probe position sampling
│   ├── lighting_modifier.py # Sun position integration
│   ├── scene_utils.py      # Scene analysis utilities
│   └── cornell_box_builder.py # Cornell Box scene builder
├── scenes/                 # 3D scene definitions
│   ├── Bistro_v5_2/        # Bistro Exterior (Falcor)
│   ├── cornell-box/        # Cornell Box (Mitsuba)
│   ├── house/              # Victorian House
│   ├── spaceship/          # Spaceship interior
│   ├── classroom/          # Japanese Classroom
│   ├── staircase2/         # Staircase scene
│   └── toy_level1/         # Simple test scenes
├── utils/                  # Utility functions
│   ├── spherical_harmonics.py # SH fitting and reconstruction
│   ├── sun_position.py     # Astronomical sun position calculation
│   └── color_temperature.py # Color temperature utilities
├── configs/                # Generation configurations
│   ├── bistro_dev.yaml     # Bistro development config
│   ├── bistro_prod.yaml    # Bistro production config
│   └── bistro_smoke.yaml   # Bistro smoke test config
├── falcor/                 # Falcor rendering system
│   ├── generate_bistro_temporal_spheres.py # Main Falcor script
│   ├── scenes/             # Falcor scene definitions
│   └── utils/              # Falcor-specific utilities
├── output/                 # Generated datasets
│   ├── dataset_2k/         # 2,000 probe dataset
│   ├── dataset_15k/        # 15,000 probe dataset
│   ├── 5D_parametric_validation/ # 5D parametric validation
│   ├── method_test_v1/     # Method testing dataset
│   └── bistro_temporal/    # Bistro temporal dataset
└── scripts/                # Main generation scripts
    ├── generate_dataset.py # Mitsuba dataset generation
    ├── validate_dataset.py # Dataset validation
    └── generate_5D_parametric.py # 5D parametric generation
```

## DATA GENERATION PIPELINE
1. **Scene Loading**: Load 3D scene (Mitsuba XML or Falcor pyscene)
2. **Probe Sampling**: Sample probe positions in scene volume
3. **Sun Position Calculation**: Compute sun directions for each moment
4. **SH Baking**: Render cubemaps/directional samples at each probe
5. **SH Fitting**: Fit spherical harmonics coefficients to radiance samples
6. **GT Rendering**: Generate ground truth images for validation
7. **Dataset Packaging**: Save probes.npz, sh_coeffs.npz, metadata.json

## RENDERING SYSTEMS
**Mitsuba 3**:
- Primary rendering engine for research datasets
- Supports Cornell Box, House, Spaceship scenes
- Uses `cuda_ad_rgb` variant for GPU acceleration
- SH baking via directional sampling (64-128 samples)

**Falcor**:
- Production rendering for Bistro Exterior scene
- Real-time path tracing with temporal accumulation
- Cubemap-based SH baking (16-64 resolution)
- Supports 600-frame temporal sequences with RGB spheres

## SCENE DEFINITIONS
**Available Scenes**:
- **Bistro Exterior**: Large outdoor scene with complex geometry (Falcor)
- **Cornell Box**: Classic indirect lighting test (Mitsuba)
- **Victorian House**: Complex architectural exterior (Mitsuba)
- **Spaceship**: Metal-rich interior with specular reflections (Mitsuba)
- **Japanese Classroom**: Indoor scene with window lighting (Mitsuba)
- **Staircase2**: Complex geometry with multiple levels (Mitsuba)

## SUN POSITION CALCULATION
- Astronomical algorithm based on latitude/longitude/time
- Generates 24-hour sun trajectory with altitude/azimuth
- Integrated into lighting system via `lighting_modifier.py`
- Outputs sun direction vectors for each temporal moment

## SPHERICAL HARMONICS
- 2nd order SH (9 coefficients per channel, 27 total)
- Fibonacci sphere sampling for directional coverage
- Cubemap projection for faster Falcor rendering
- Reconstruction error validation (<5% target)

## OUTPUT STRUCTURE
```
dataset_name/
├── probes.npz              # Probe positions [N, 3]
├── metadata.json           # Generation parameters
├── moment_00/              # Temporal moment 00:00
│   ├── sh_coeffs.npz       # SH coefficients [N, 27]
│   ├── sun_direction.txt   # Sun direction vector
│   └── images/             # Ground truth images
├── moment_06/              # 06:00 moment
└── moment_12/              # 12:00 moment
```

## CONVENTIONS
**Dataset Naming**:
- `dataset_2k`: 2,000 probes × 3 moments
- `dataset_15k`: 15,000 probes × 24 moments (target)
- `5D_parametric_validation`: 5D parametric space validation
- `bistro_temporal`: Bistro scene with 600 temporal frames

**Probe Sampling**:
- Uniform grid + adaptive surface sampling
- 70% uniform + 30% near-surface distribution
- Depth-guided sampling for Bistro scene

**Moment Organization**:
- 24 moments for full day coverage
- 3 moments (06:00, 12:00, 18:00) for prototyping
- 600 frames @ 30 FPS for temporal sequences

## KEY SCRIPTS
- `generate_dataset.py`: Main Mitsuba dataset generation
- `generate_bistro_temporal_spheres.py`: Falcor Bistro generation
- `validate_dataset.py`: Dataset quality validation
- `generate_5D_parametric.py`: 5D parametric dataset generation
- `analyze_sh_quality.py`: SH reconstruction quality analysis

## COMMANDS
```bash
# Generate Mitsuba dataset
python generate_dataset.py --scene scenes/cornell-box/scene.xml --output output/dataset_2k

# Generate Falcor Bistro dataset
python falcor/generate_bistro_temporal_spheres.py --output output/bistro_temporal

# Validate dataset quality
python validate_dataset.py --dataset output/dataset_2k

# Analyze SH reconstruction
python analyze_sh_quality.py --dataset output/dataset_2k
```

## NOTES
- **Memory Intensive**: 15k probe dataset requires ~10GB storage
- **Rendering Time**: Full 24-moment dataset takes 6-12 hours
- **Quality vs Speed**: Cubemap (fast) vs directional sampling (accurate)
- **Validation Required**: Always validate SH reconstruction error (<5%)
- **Scene Compatibility**: Mitsuba scenes ≠ Falcor scenes (different formats)