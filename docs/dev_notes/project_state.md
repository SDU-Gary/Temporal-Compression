# Project State Snapshot

**Generated**: 2026-01-08 11:49:49
**DB Last Updated**: 2026-01-08 11:48:47
**Status**: Historical snapshot (not auto-generated in current repository state)

---

## [ENV] Environment

- **Working Dir**: /home/kyrie/毕设
- **Python**: /home/kyrie/miniconda3/bin/python
- **Conda Env**: base
- **Virtual Env**: N/A

---

## [TASKS] Active Tasks (7 pending)

| Priority | Status | Task | Created |
|----------|--------|------|----------|
| 🔴 High | ⏳ todo | Add energy conservation loss (physics-based soft c | 2026-01-04 |
| 🔴 High | ⏳ todo | Implement 4×3 cascaded volumes (4 spatial × 3 temp | 2026-01-04 |
| 🟡 Medium | ⏳ todo | Implement 10-bit quantization (two-stage training) | 2026-01-04 |
| 🟡 Medium | ⏳ todo | Extend to 24-hour lighting (currently 6 moments) | 2026-01-04 |
| 🟡 Medium | ⏳ todo | Implement fused CUDA kernels for <0.5ms decompress | 2026-01-04 |
| 🟡 Medium | ⏳ todo | Implement temporal LRU cache for real-time decompr | 2026-01-04 |
| 🟢 Low | ⏳ todo | Implement lighting decomposition (direct + indirec | 2026-01-04 |

---

## [MILESTONES] Key Results

### Current Baseline
*No baseline set*

### Historical Best
- **EXP-20260103-002** (Phase2_PGCPL)
- PSNR: 22.38 dB, SSIM: 0.951

---

## [RECENT] Last 5 Experiments

| Run | Exp ID | Phase | Dataset | PSNR | SSIM | Status |
|-----|--------|-------|---------|------|------|--------|
| 22 | EXP-20260108-003 | Phase2 | FALCOR_P2 | - | - | ⭐ BASELINE |
| 21 | EXP-20260108-002 | Phase1 | FALCOR_P1 | - | - | ✅ |
| 20 | EXP-20260108-001 | Phase1 | TEST_QUICK | - | - | ✅ |
| 19 | EXP-20260106-002 | PGCPL | INTENSITY_SIMPLE | - | - | ✅ |
| 18 | EXP-20260106-001 | PGCPL | INTENSITY_SIMPLE | - | - | ⭐ BASELINE |

---

## [MAPS] Dataset-Scene Mapping (Top 5 Used)

- **5D_PARAM** (5D_parametric_validation): 125 probes × 41 moments, SPP=128 (6 exps)
- **D2K** (dataset_2k): 1,728 probes × 6 moments, SPP=128 (3 exps)
- **MT1** (method_test_v1): 343 probes × 6 moments, SPP=256 (3 exps)
- **INTENSITY_SIMPLE** (intensity_modulation_343_simple): 343 probes × 12 moments, SPP=128 (2 exps)
- **TPE_CB** (level2_tpe/cornell-box_test): 1 probes × 13 moments, SPP=256 (2 exps)

---

## [QUICK_REF] Script Shortcuts

- `ablation_5d`: ablation_gaussian_physics_5D.py [PG-GCPL]
- `query_validation`: test_interpolation_query_physics.py [PG-GCPL]
- `render_compare_k30`: render_scene_comparison_K30_r8.py [PG-GCPL]
- `svd_analysis`: svd_analysis.py [PG-GCPL]
- `train_dual_fbt`: train_dual_gaussian_fbt.py [PG-GCPL]
- `train_pg_rglt`: train_pg_rglt.py [PG-GCPL]

---

## Action Protocol

**After conversation compact**:
1. Query live DB first: `python tools/logexp.py query --limit 10`
2. Optionally read this snapshot: `cat docs/dev_notes/project_state.md`
3. Read [ENV], [TASKS], [MILESTONES], [RECENT] sections
4. **NEVER guess dataset paths** - query DB or check this file

**After completing experiment**:
1. Log immediately: `python tools/logexp.py log --phase ... --results ...`
2. If baseline: `python tools/logexp.py set-baseline <exp_id>`
3. Verify persistence: `python tools/logexp.py query --limit 5`

**For task planning**:
1. Add task: `python tools/logexp.py add-task "..." --priority 2`
2. Complete: `python tools/logexp.py complete-task <id>`
