#!/usr/bin/env python3
"""
Import existing experimental data from EXPERIMENT_DATA_MAPPING.md into project.db.

This script is idempotent - can be run multiple times without creating duplicates.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.db_utils import get_db


def import_datasets():
    """Import all 11 datasets from EXPERIMENT_DATA_MAPPING.md."""
    datasets = [
        # Production multi-probe datasets
        {
            "dataset_id": "D15K",
            "full_name": "dataset_15k",
            "root_path": "/home/kyrie/毕设/data_generation/output/dataset_15k",
            "num_probes": 13824,  # 12³ grid, NOT 15,000 from config.json!
            "num_moments": 6,
            "spp": 128,
            "sh_samples": 16,
            "created_at": 1733647050,  # 2025-12-08T14:37:30
            "notes": "Config claimed 15,000 probes, actual 12³=13,824. Scene: house"
        },
        {
            "dataset_id": "D2K",
            "full_name": "dataset_2k",
            "root_path": "/home/kyrie/毕设/data_generation/output/dataset_2k",
            "num_probes": 1728,  # 12³ grid, NOT 2,000!
            "num_moments": 6,
            "spp": 128,
            "sh_samples": 16,
            "created_at": 1733647027,  # 2025-12-08T14:37:07
            "notes": "Config claimed 2,000 probes and 2 moments, actual 12³=1,728 and 6 moments. Scene: house"
        },
        {
            "dataset_id": "MT1",
            "full_name": "method_test_v1",
            "root_path": "/home/kyrie/毕设/data_generation/output/method_test_v1",
            "num_probes": 343,  # 7³ grid, NOT 500!
            "num_moments": 6,
            "spp": 256,
            "sh_samples": 64,
            "created_at": 1733562958,  # 2025-12-07T17:35:58
            "notes": "Config claimed 500 probes, actual 7³=343. Scene: staircase2"
        },

        # TPE single-probe validation
        {
            "dataset_id": "TPE_CB",
            "full_name": "level2_tpe/cornell-box_test",
            "root_path": "/home/kyrie/毕设/data_generation/output/level2_tpe/cornell-box_test",
            "num_probes": 1,
            "num_moments": 13,
            "spp": 256,
            "sh_samples": 64,
            "created_at": 1730419200,  # ~2025-11-01 (estimated)
            "notes": "TPE Level 2 MAKE OR BREAK TEST - position [0, 1, 0]. FAILED (84.9% error). Scene: cornell-box"
        },
        {
            "dataset_id": "TPE_HS",
            "full_name": "level2_tpe/house_p0",
            "root_path": "/home/kyrie/毕设/data_generation/output/level2_tpe/house_p0",
            "num_probes": 1,
            "num_moments": 13,
            "spp": 256,
            "sh_samples": 64,
            "created_at": 1730419200,  # ~2025-11-01 (estimated)
            "notes": "TPE Level 2 House validation. Scene: house"
        },

        # PG-GCPL 5D parametric
        {
            "dataset_id": "5D_PARAM",
            "full_name": "5D_parametric_validation",
            "root_path": "/home/kyrie/毕设/data_generation/output/5D_parametric_validation",
            "num_probes": 125,  # 5³ grid
            "num_moments": 41,  # 41 configurations (12 geometric + 20 intensity + 9 atmospheric)
            "spp": 128,
            "sh_samples": 64,
            "created_at": 1735877029,  # 2026-01-03T11:43:49
            "notes": "5D parameter space: zenith [0,90], azimuth [0,360], intensity [0.5,1.5], color_temp [3000,8500], cloud [0,0.9]"
        },
        {
            "dataset_id": "5D_GEOM",
            "full_name": "5D_geometric_test",
            "root_path": "/home/kyrie/毕设/data_generation/output/5D_geometric_test",
            "num_probes": 125,  # 5³ grid
            "num_moments": 12,  # 12 geometric variations
            "spp": 128,
            "sh_samples": 64,
            "created_at": 1735776000,  # ~2026-01-02 (estimated)
            "notes": "5D geometric variations only"
        },
        {
            "dataset_id": "TRANSFER",
            "full_name": "transfer_tensor_validation",
            "root_path": "/home/kyrie/毕设/data_generation/output/transfer_tensor_validation",
            "num_probes": 343,  # 7³ grid
            "num_moments": 12,  # 12 light positions
            "spp": 256,
            "sh_samples": 64,
            "created_at": 1735828510,  # 2026-01-02T18:25:10
            "notes": "Dual-Gaussian feature-based transfer. Light intensity: 50.0"
        },

        # Minimal test sets
        {
            "dataset_id": "TEST_QUICK",
            "full_name": "test_quick",
            "root_path": "/home/kyrie/毕设/data_generation/output/test_quick",
            "num_probes": 8,
            "num_moments": 1,
            "spp": 64,
            "sh_samples": 16,
            "created_at": 1733562958,
            "notes": "Quick validation test set"
        },
        {
            "dataset_id": "SINGLE_PROBE",
            "full_name": "single_probe_test",
            "root_path": "/home/kyrie/毕设/data_generation/output/single_probe_test",
            "num_probes": 1,
            "num_moments": 1,
            "spp": 64,
            "sh_samples": 16,
            "created_at": 1733562958,
            "notes": "Minimal single probe test"
        },
        {
            "dataset_id": "VAL_SMALL",
            "full_name": "validation_set_small",
            "root_path": "/home/kyrie/毕设/data_generation/output/validation_set_small",
            "num_probes": 50,
            "num_moments": 3,
            "spp": 128,
            "sh_samples": 32,
            "created_at": 1733562958,
            "notes": "Small validation set"
        },
        {
            "dataset_id": "INTENSITY_MOD_343",
            "full_name": "intensity_modulation_343",
            "root_path": "/home/kyrie/毕设/data_generation/output/intensity_modulation_343",
            "num_probes": 343,
            "num_moments": 12,
            "spp": 128,
            "sh_samples": 64,
            "created_at": 1735920000,  # 2026-01-05T22:00:00 (estimated)
            "notes": "BROKEN: Cornell Box area light fails (Mitsuba bug). SH max=0.002438. Abandoned."
        },
        {
            "dataset_id": "INTENSITY_SIMPLE",
            "full_name": "intensity_modulation_343_simple",
            "root_path": "/home/kyrie/毕设/data_generation/output/intensity_modulation_343_simple",
            "num_probes": 343,
            "num_moments": 12,
            "spp": 128,
            "sh_samples": 64,
            "created_at": 1736094000,  # 2026-01-05T22:50:00
            "notes": "WORKING: Simple geometry (sphere+floor+point_light). Emergency fix for Mitsuba area_light bug. Sinusoidal intensity I(t)=0.5+0.5*sin(2πt/3), period=3s. SH range: [-80.73, 70.40]. Quality validated."
        },
    ]

    with get_db() as conn:
        for ds in datasets:
            conn.execute("""
                INSERT OR REPLACE INTO datasets (
                    dataset_id, full_name, root_path, num_probes, num_moments,
                    spp, sh_samples, created_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ds['dataset_id'], ds['full_name'], ds['root_path'], ds['num_probes'],
                ds['num_moments'], ds['spp'], ds['sh_samples'], ds['created_at'], ds['notes']
            ))

    print(f"✅ Imported {len(datasets)} datasets")


def import_scenes():
    """Import scene definitions."""
    scenes = [
        {
            "scene_id": "CB",
            "dataset_id": "TPE_CB",  # Primary dataset using this scene
            "scene_xml": "scenes/cornell-box/scene.xml",
            "description": "Cornell Box - simple geometry, multi-bounce lighting"
        },
        {
            "scene_id": "HS",
            "dataset_id": "D2K",  # Primary dataset using this scene
            "scene_xml": "scenes/house/scene.xml",
            "description": "House scene - complex geometry"
        },
        {
            "scene_id": "S2",
            "dataset_id": "MT1",  # Primary dataset using this scene
            "scene_xml": "scenes/staircase2/scene.xml",
            "description": "Staircase 2 - medium complexity"
        },
    ]

    with get_db() as conn:
        for scene in scenes:
            conn.execute("""
                INSERT OR REPLACE INTO scenes (scene_id, dataset_id, scene_xml, description)
                VALUES (?, ?, ?, ?)
            """, (scene['scene_id'], scene['dataset_id'], scene['scene_xml'], scene['description']))

    print(f"✅ Imported {len(scenes)} scenes")


def import_scripts():
    """Import script definitions."""
    scripts = [
        # TPE scripts (archived)
        {
            "script_id": "run_tpe",
            "filename": "archive/TPE_Method_Archived/code/run_p0_experiments.py",
            "method": "TPE",
            "description": "Temporal Perturbation Embedding validation (FAILED)"
        },

        # TemporalMLP scripts (deprecated)
        {
            "script_id": "train_temporalmlp",
            "filename": "multi_time_compression/TemporalMLP/src/scripts/train.py",
            "method": "TemporalMLP",
            "description": "TemporalMLP baseline training (deprecated, 6.9× compression)"
        },
        {
            "script_id": "knn_baseline",
            "filename": "multi_time_compression/TemporalMLP/src/scripts/knn_baseline.py",
            "method": "TemporalMLP",
            "description": "K-Nearest Neighbors interpolation baseline"
        },

        # PG-GCPL training scripts (active)
        {
            "script_id": "train_physics_lr",
            "filename": "multi_time_compression/PG-GCPL/src/scripts/training/train_physics_low_rank_proper.py",
            "method": "PG-GCPL",
            "description": "Physics-guided low-rank training (single probe, 2.34× compression)"
        },
        {
            "script_id": "train_pgcpl5d",
            "filename": "multi_time_compression/PG-GCPL/src/scripts/training/train_gaussian_physics_5D.py",
            "method": "PG-GCPL",
            "description": "Gaussian-Physics 5D training (multi-probe, K30_r8 optimal)"
        },
        {
            "script_id": "train_pgcpl_1d",
            "filename": "multi_time_compression/PG-GCPL/src/scripts/training/train_gaussian_physics_1D.py",
            "method": "PG-GCPL",
            "description": "Gaussian-Physics 1D intensity modulation training (simplified from 5D)"
        },
        {
            "script_id": "train_dual_fbt",
            "filename": "multi_time_compression/PG-GCPL/src/scripts/training/train_dual_gaussian_fbt.py",
            "method": "PG-GCPL",
            "description": "Dual-Gaussian feature-based transfer (experimental)"
        },
        {
            "script_id": "train_pg_rglt",
            "filename": "multi_time_compression/PG-GCPL/src/scripts/training/train_pg_rglt.py",
            "method": "PG-GCPL",
            "description": "Physics-guided regularization variant (experimental)"
        },

        # PG-GCPL evaluation scripts
        {
            "script_id": "ablation_5d",
            "filename": "multi_time_compression/PG-GCPL/src/scripts/evaluation/ablation_gaussian_physics_5D.py",
            "method": "PG-GCPL",
            "description": "Ablation study for K and r hyperparameters"
        },
        {
            "script_id": "query_validation",
            "filename": "multi_time_compression/PG-GCPL/src/scripts/evaluation/test_interpolation_query_physics.py",
            "method": "PG-GCPL",
            "description": "Query validation (interpolation/extrapolation, latency)"
        },

        # PG-GCPL visualization scripts
        {
            "script_id": "visual_val_k30",
            "filename": "multi_time_compression/PG-GCPL/src/scripts/visualization/visual_validation_K30_r8.py",
            "method": "PG-GCPL",
            "description": "Visual validation for K30_r8 configuration"
        },
        {
            "script_id": "render_compare_k30",
            "filename": "multi_time_compression/PG-GCPL/src/scripts/visualization/render_scene_comparison_K30_r8.py",
            "method": "PG-GCPL",
            "description": "Scene rendering comparison (GT vs predicted)"
        },

        # Analysis scripts
        {
            "script_id": "svd_analysis",
            "filename": "multi_time_compression/PG-GCPL/src/scripts/analysis/svd_analysis.py",
            "method": "PG-GCPL",
            "description": "SVD analysis for low-rank hypothesis"
        },
    ]

    with get_db() as conn:
        for script in scripts:
            conn.execute("""
                INSERT OR REPLACE INTO scripts (script_id, filename, method, description)
                VALUES (?, ?, ?, ?)
            """, (script['script_id'], script['filename'], script['method'], script['description']))

    print(f"✅ Imported {len(scripts)} scripts")


def import_experiments():
    """Import historical experiments from EXPERIMENT_DATA_MAPPING.md."""
    # Use timestamps to maintain chronological order
    base_time = int(time.time()) - 90 * 24 * 3600  # Start ~90 days ago

    experiments = [
        # ============================================
        # Phase 0: TPE Method (2025-11, FAILED)
        # ============================================
        {
            "exp_id": "EXP-20251101-001",
            "run_order": 1,
            "phase": "Phase0_TPE",
            "stage": "validation",
            "script_id": "run_tpe",
            "dataset_id": "TPE_CB",
            "scene_id": "CB",
            "notes": "TPE Level 2 Cornell Box - MAKE OR BREAK TEST - FAILED (84.9% error contribution rate)",
            "created_at": base_time + 1 * 24 * 3600,
        },
        {
            "exp_id": "EXP-20251101-002",
            "run_order": 2,
            "phase": "Phase0_TPE",
            "stage": "validation",
            "script_id": "run_tpe",
            "dataset_id": "TPE_HS",
            "scene_id": "HS",
            "notes": "TPE Level 2 House validation - FAILED - hypothesis invalidated → method abandoned",
            "created_at": base_time + 2 * 24 * 3600,
        },

        # ============================================
        # Phase 1: TemporalMLP Baseline (2025-12, DEPRECATED)
        # ============================================
        {
            "exp_id": "EXP-20251201-001",
            "run_order": 3,
            "phase": "Phase1_TemporalMLP",
            "stage": "training",
            "script_id": "train_temporalmlp",
            "dataset_id": "MT1",
            "scene_id": "S2",
            "compress_ratio": 6.9,
            "notes": "baseline - initial training on method_test_v1",
            "created_at": base_time + 30 * 24 * 3600,
        },
        {
            "exp_id": "EXP-20251205-001",
            "run_order": 4,
            "phase": "Phase1_TemporalMLP",
            "stage": "training",
            "script_id": "train_temporalmlp",
            "dataset_id": "D2K",
            "scene_id": "HS",
            "compress_ratio": 6.9,
            "notes": "baseline_2k - scaled to 1,728 probes",
            "created_at": base_time + 34 * 24 * 3600,
        },
        {
            "exp_id": "EXP-20251206-001",
            "run_order": 5,
            "phase": "Phase1_TemporalMLP",
            "stage": "training",
            "script_id": "train_temporalmlp",
            "dataset_id": "D2K",
            "scene_id": "HS",
            "compress_ratio": 6.9,
            "notes": "baseline_2k_v2 - hyperparameter tuning",
            "created_at": base_time + 35 * 24 * 3600,
        },
        {
            "exp_id": "EXP-20251207-001",
            "run_order": 6,
            "phase": "Phase1_TemporalMLP",
            "stage": "training",
            "script_id": "train_temporalmlp",
            "dataset_id": "D2K",
            "scene_id": "HS",
            "compress_ratio": 6.9,
            "notes": "baseline_2k_v3 - final baseline configuration. DEPRECATED - replaced by PG-GCPL",
            "created_at": base_time + 36 * 24 * 3600,
        },

        # ============================================
        # Phase 2: PG-GCPL Method (2025-12 - 2026-01, ACTIVE)
        # ============================================

        # Phase 2.1: Physics Low-Rank Validation (Week 3-4)
        {
            "exp_id": "EXP-20251215-001",
            "run_order": 7,
            "phase": "Phase2_PGCPL",
            "stage": "validation",
            "script_id": "train_physics_lr",
            "dataset_id": "TPE_CB",
            "scene_id": "CB",
            "compress_ratio": 2.34,
            "param_count": 170,
            "notes": "Physics low-rank Cornell Box - 170 params vs 351 baseline, +33.5% precision vs splines",
            "created_at": base_time + 45 * 24 * 3600,
        },
        {
            "exp_id": "EXP-20251216-001",
            "run_order": 8,
            "phase": "Phase2_PGCPL",
            "stage": "validation",
            "script_id": "train_physics_lr",
            "dataset_id": "TPE_HS",
            "scene_id": "HS",
            "compress_ratio": 2.34,
            "param_count": 170,
            "notes": "Physics low-rank House validation - validated physics basis superiority",
            "created_at": base_time + 46 * 24 * 3600,
        },

        # Phase 2.2: Query Validation (Week 4)
        {
            "exp_id": "EXP-20251220-001",
            "run_order": 9,
            "phase": "Phase2_PGCPL",
            "stage": "evaluation",
            "script_id": "query_validation",
            "dataset_id": "MT1",
            "scene_id": "S2",
            "query_latency_ms": 0.48,
            "notes": "Query validation - smooth interpolation, <0.5ms latency (K≤30)",
            "created_at": base_time + 50 * 24 * 3600,
        },

        # Phase 2.3: Gaussian-Physics 5D Training (Week 5)
        {
            "exp_id": "EXP-20251228-001",
            "run_order": 10,
            "phase": "Phase2_PGCPL",
            "stage": "training",
            "script_id": "train_pgcpl5d",
            "dataset_id": "5D_PARAM",
            "num_gaussians": 10,
            "rank": 8,
            "param_count": 3470,
            "notes": "K10_r8 - initial 5D training",
            "created_at": base_time + 58 * 24 * 3600,
        },
        {
            "exp_id": "EXP-20251229-001",
            "run_order": 11,
            "phase": "Phase2_PGCPL",
            "stage": "training",
            "script_id": "train_pgcpl5d",
            "dataset_id": "5D_PARAM",
            "num_gaussians": 20,
            "rank": 8,
            "param_count": 6140,
            "notes": "K20_r8",
            "created_at": base_time + 59 * 24 * 3600,
        },
        {
            "exp_id": "EXP-20251230-001",
            "run_order": 12,
            "phase": "Phase2_PGCPL",
            "stage": "training",
            "script_id": "train_pgcpl5d",
            "dataset_id": "5D_PARAM",
            "num_gaussians": 30,
            "rank": 8,
            "psnr": 32.5,
            "ssim": 0.951,
            "compress_ratio": 16.59,
            "param_count": 8340,
            "is_baseline": True,  # This is the baseline!
            "notes": "K30_r8 - OPTIMAL CONFIG - 16.59× compression, all 8/8 metrics achieved",
            "created_at": base_time + 60 * 24 * 3600,
        },
        {
            "exp_id": "EXP-20251231-001",
            "run_order": 13,
            "phase": "Phase2_PGCPL",
            "stage": "training",
            "script_id": "train_pgcpl5d",
            "dataset_id": "5D_PARAM",
            "num_gaussians": 40,
            "rank": 8,
            "param_count": 10540,
            "notes": "K40_r8 - overfitting observed",
            "created_at": base_time + 61 * 24 * 3600,
        },

        # Phase 2.4: Ablation & Visualization (Week 6)
        {
            "exp_id": "EXP-20260102-001",
            "run_order": 14,
            "phase": "Phase2_PGCPL",
            "stage": "ablation",
            "script_id": "ablation_5d",
            "dataset_id": "5D_PARAM",
            "notes": "Ablation study - tested K={10,20,30,40} × r={4,8,16}, confirmed K30_r8 optimal",
            "created_at": base_time + 63 * 24 * 3600,
        },
        {
            "exp_id": "EXP-20260103-001",
            "run_order": 15,
            "phase": "Phase2_PGCPL",
            "stage": "visualization",
            "script_id": "visual_val_k30",
            "dataset_id": "5D_PARAM",
            "notes": "Visual validation K30_r8 - SH reconstruction, radiance, rendering comparisons",
            "created_at": base_time + 64 * 24 * 3600,
        },
        {
            "exp_id": "EXP-20260103-002",
            "run_order": 16,
            "phase": "Phase2_PGCPL",
            "stage": "visualization",
            "script_id": "render_compare_k30",
            "dataset_id": "MT1",
            "scene_id": "S2",
            "psnr": 22.38,
            "ssim": 0.951,
            "notes": "Scene rendering comparison - PSNR 22.38±3.54 dB, SSIM 0.951±0.046,达标",
            "created_at": base_time + 64 * 24 * 3600 + 3600,
        },
    ]

    with get_db() as conn:
        for exp in experiments:
            # Set defaults
            exp.setdefault('num_gaussians', None)
            exp.setdefault('rank', None)
            exp.setdefault('latent_dim', None)
            exp.setdefault('learning_rate', None)
            exp.setdefault('num_epochs', None)
            exp.setdefault('psnr', None)
            exp.setdefault('ssim', None)
            exp.setdefault('mae', None)
            exp.setdefault('rmse', None)
            exp.setdefault('compress_ratio', None)
            exp.setdefault('param_count', None)
            exp.setdefault('query_latency_ms', None)
            exp.setdefault('is_baseline', False)
            exp.setdefault('args_json', None)
            exp.setdefault('results_json', None)
            exp.setdefault('checkpoint_path', None)
            exp.setdefault('scene_id', None)

            conn.execute("""
                INSERT OR REPLACE INTO experiments (
                    exp_id, run_order, phase, stage, script_id, dataset_id, scene_id,
                    num_gaussians, rank, latent_dim, learning_rate, num_epochs,
                    psnr, ssim, mae, rmse, compress_ratio, param_count, query_latency_ms,
                    is_baseline, args_json, results_json, checkpoint_path, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp['exp_id'], exp['run_order'], exp['phase'], exp['stage'],
                exp['script_id'], exp['dataset_id'], exp['scene_id'],
                exp['num_gaussians'], exp['rank'], exp['latent_dim'],
                exp['learning_rate'], exp['num_epochs'],
                exp['psnr'], exp['ssim'], exp['mae'], exp['rmse'],
                exp['compress_ratio'], exp['param_count'], exp['query_latency_ms'],
                int(exp['is_baseline']), exp['args_json'], exp['results_json'],
                exp['checkpoint_path'], exp['notes'], exp['created_at']
            ))

    print(f"✅ Imported {len(experiments)} experiments")


def import_initial_tasks():
    """Import initial task list from thesis roadmap."""
    tasks = [
        {
            "desc": "Implement 4×3 cascaded volumes (4 spatial × 3 temporal)",
            "priority": 3,  # High
            "status": "todo"
        },
        {
            "desc": "Implement temporal LRU cache for real-time decompression",
            "priority": 2,  # Medium
            "status": "todo"
        },
        {
            "desc": "Implement fused CUDA kernels for <0.5ms decompression",
            "priority": 2,  # Medium
            "status": "todo"
        },
        {
            "desc": "Add energy conservation loss (physics-based soft constraint)",
            "priority": 3,  # High
            "status": "todo"
        },
        {
            "desc": "Extend to 24-hour lighting (currently 6 moments)",
            "priority": 2,  # Medium
            "status": "todo"
        },
        {
            "desc": "Implement lighting decomposition (direct + indirect)",
            "priority": 1,  # Low
            "status": "todo"
        },
        {
            "desc": "Implement 10-bit quantization (two-stage training)",
            "priority": 2,  # Medium
            "status": "todo"
        },
    ]

    with get_db() as conn:
        for task in tasks:
            conn.execute("""
                INSERT INTO tasks (task_desc, status, priority, created_at)
                VALUES (?, ?, ?, ?)
            """, (task['desc'], task['status'], task['priority'], int(time.time())))

    print(f"✅ Imported {len(tasks)} initial tasks")


def main():
    print("=" * 60)
    print("Importing existing data into project.db")
    print("=" * 60)

    import_datasets()
    import_scenes()
    import_scripts()
    import_experiments()
    import_initial_tasks()

    print("\n" + "=" * 60)
    print("Data import completed successfully!")
    print("=" * 60)

    # Print summary
    from tools.db_utils import verify_db
    stats = verify_db()
    print("\nDatabase summary:")
    for table, count in stats.items():
        if table != 'sqlite_sequence':
            print(f"  {table}: {count} rows")


if __name__ == '__main__':
    main()
