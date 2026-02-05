# Temporal-Compression Project Folder Structure Analysis Report

**Date**: 2026-02-05  
**Analyzer**: Sisyphus-Junior  
**Project**: Graduate thesis on hierarchical neural compression for multi-temporal lighting

## Executive Summary

The Temporal-Compression project has undergone a significant refactoring to establish a clear workflow-oriented structure (1_data_generation → 2_src → 3_experiments → 4_thesis). This structure effectively separates concerns and follows a logical research workflow. However, several areas could benefit from further refinement to improve maintainability, reduce duplication, and enhance clarity.

## 1. Current Folder Hierarchy Analysis

### Root Directory Structure
```
./
├── 1_data_generation/          # Stage 1: Dataset creation (Mitsuba/Falcor)
├── 2_src/                      # Stage 2: Core algorithms (models, training, utils)
├── 3_experiments/              # Stage 3: Experimentation and evaluation
├── 4_thesis/                   # Stage 4: Thesis materials (reports, figures)
├── archive/                    # Historical/deprecated methods
├── docs/                       # Documentation and references
├── metadata/                   # Project database (project.db)
├── pipelines/                  # Workflow pipeline definitions
├── tools/                      # Project utilities and CLI tools
└── [various config and README files]
```

### Numbered Workflow Directories Detail

#### 1_data_generation/ (Data Generation Stage)
- **Core strength**: Well-organized with clear separation of concerns
- **Subdirectories**: `core/`, `utils/`, `configs/`, `scenes/`, `output/`, `falcor/`
- **Key files**: `generate_dataset.py`, `validate_dataset.py`, `generate_5D_parametric.py`
- **Observation**: Comprehensive data generation pipeline with both Mitsuba and Falcor support

#### 2_src/ (Source Code Stage)
- **Core strength**: Clean separation of models, training, data, and utilities
- **Subdirectories**: `models/`, `training/`, `data/`, `utils/`, `tests/`
- **Key models**: Gaussian-Physics hybrid models (1D, 5D, unified), physics-guided low-rank
- **Observation**: Well-structured Python package with proper `__init__.py` files

#### 3_experiments/ (Experimentation Stage)
- **Core strength**: Clear separation of configs, scripts, and results
- **Subdirectories**: `configs/`, `scripts/`, `results/`
- **Script organization**: Further divided into `training/`, `analysis/`, `visualization/`, `evaluation/`
- **Observation**: Good organization but some script naming inconsistencies

#### 4_thesis/ (Thesis Stage)
- **Current state**: Mostly empty with `figures/` and `report/` subdirectories
- **Observation**: Underutilized - could better organize thesis-related materials

## 2. Strengths of Current Structure

### ✅ **Workflow-First Organization**
- Clear progression: Data → Source → Experiments → Thesis
- Each stage has distinct responsibilities
- Easy to understand project lifecycle

### ✅ **Separation of Concerns**
- Core algorithms (`2_src/`) separate from experiments (`3_experiments/`)
- Data generation (`1_data_generation/`) separate from model training
- Archive (`archive/`) keeps deprecated code without cluttering active development

### ✅ **Comprehensive Documentation**
- Each directory has its own `AGENTS.md` knowledge base
- `docs/` directory contains detailed development notes
- Database-driven experiment tracking (`metadata/project.db`)

### ✅ **Research-Friendly Structure**
- Experiment results organized by configuration and parameters
- Support for multiple rendering engines (Mitsuba, Falcor)
- Clear pipeline definitions (`pipelines/`)

## 3. Weaknesses and Pain Points

### ❌ **Duplicate Utility Functions**
**Critical Issue**: Found duplicate files with identical functionality:
1. `1_data_generation/utils/spherical_harmonics.py` ≈ `2_src/utils/spherical_harmonics.py`
2. `1_data_generation/utils/sun_position.py` ≈ `2_src/utils/sun_position.py`

**Impact**: Code duplication, maintenance burden, potential inconsistency

### ❌ **Inconsistent Script Naming**
**Issue**: Script files in `3_experiments/scripts/` show inconsistent patterns:
- Some include configuration details in names (`visual_validation_K30_r8.py`)
- Others use generic names (`train_gaussian_physics.py`)
- Mixed Chinese/English naming

**Impact**: Difficult to locate specific scripts, unclear what parameters they use

### ❌ **Underutilized 4_thesis Directory**
**Issue**: The thesis stage directory is mostly empty while thesis-related files are scattered:
- Some thesis documents in `docs/`
- Figures potentially scattered across experiment results
- No clear organization for thesis writing process

### ❌ **Deep Nesting in Experiments**
**Issue**: Experiment results create deep directory hierarchies:
```
3_experiments/results/group1_5D_parametric/04_ablation_and_visualization/ablation/K30_r8/
```

**Impact**: Difficult navigation, long file paths, potential for path length issues

### ❌ **Missing Python Package Configuration**
**Issue**: No `setup.py`, `pyproject.toml`, or `requirements.txt` files
**Impact**: Difficult to install as a package, dependency management relies on virtual environment

### ❌ **Cross-Directory Import Patterns**
**Issue**: Some imports suggest tight coupling:
- `tools/` imports from various locations
- Potential circular dependencies if not carefully managed

## 4. Specific Refactoring Suggestions

### **Priority 1: Consolidate Duplicate Utilities**
```bash
# Move shared utilities to a common location
mkdir -p shared/utils/
mv 1_data_generation/utils/spherical_harmonics.py shared/utils/
mv 1_data_generation/utils/sun_position.py shared/utils/

# Update imports in both 1_data_generation and 2_src
# Create symbolic links or update Python path
```

**Alternative**: Create a proper Python package with `src/temporal_compression/utils/`

### **Priority 2: Standardize Script Naming Convention**
**Proposed Convention**: `{purpose}_{model}_{config}_{version}.py`
- Example: `train_gaussian_physics_5D_K30_r8_v1.py`
- Example: `visualize_sh_reconstruction_bistro_v2.py`

**Action**: Rename existing scripts to follow consistent pattern

### **Priority 3: Enhance 4_thesis Organization**
```bash
# Reorganize thesis materials
4_thesis/
├── chapters/           # Individual thesis chapters
├── figures/           # All thesis figures (organized by chapter)
├── data/              # Thesis-specific data/analysis
├── references/        # Thesis bibliography
├── drafts/            # Draft versions
└── final/             # Final submission materials
```

### **Priority 4: Flatten Experiment Results Structure**
**Proposed Structure**:
```
3_experiments/results/
├── {experiment_id}/           # e.g., EXP-20260103-002
│   ├── config.yaml           # Experiment configuration
│   ├── logs/                 # Training logs
│   ├── checkpoints/          # Model checkpoints
│   ├── visualizations/       # Generated figures
│   └── metrics.json          # Evaluation metrics
```

**Use database** (`metadata/project.db`) to track experiment metadata instead of deep directory nesting

### **Priority 5: Create Proper Python Package**
```python
# Create pyproject.toml
[project]
name = "temporal-compression"
version = "0.1.0"
dependencies = [
    "torch>=2.9.1",
    "numpy>=1.26.4",
    # ... other dependencies
]

[project.optional-dependencies]
dev = ["pytest", "black", "mypy"]
data = ["mitsuba>=3.7.3"]
```

### **Priority 6: Improve Import Structure**
**Current**: Relative imports within `2_src/`, absolute imports elsewhere
**Proposed**: Use absolute imports with proper package structure
```python
# Instead of: from . import spherical_harmonics
# Use: from temporal_compression.utils import spherical_harmonics
```

## 5. Impact Assessment

### **What Would Break**

1. **Existing imports** to duplicate utility files would need updating
2. **Script references** in pipeline definitions might need path updates
3. **Experiment tracking** scripts might need adjustment for new directory structure
4. **Database queries** might need updating if experiment ID format changes

### **What Would Need Updating**

1. **All import statements** referencing moved files
2. **Pipeline YAML files** (`pipelines/`) with script paths
3. **Documentation** referencing old file locations
4. **Test files** with hardcoded paths
5. **Database schema** if experiment tracking changes

### **Migration Strategy**

**Phase 1**: Create new structure alongside old (parallel deployment)
**Phase 2**: Update critical paths gradually
**Phase 3**: Run comprehensive tests
**Phase 4**: Remove old structure after verification

## 6. Additional Observations

### **Positive Patterns to Preserve**
1. **Workflow numbering** (1_, 2_, 3_, 4_) is intuitive and effective
2. **AGENTS.md files** in each directory provide excellent context
3. **Database-driven experiment tracking** is sophisticated and valuable
4. **Archive directory** properly preserves historical work

### **Cultural/Project-Specific Considerations**
1. **Chinese documentation** with bilingual code comments
2. **Research-focused** rather than production-focused
3. **Heavy GPU/rendering dependencies** (Mitsuba, Falcor)
4. **Academic timeline constraints** (thesis deadlines)

## 7. Recommended Implementation Order

### **Immediate (Low Risk)**
1. Create `pyproject.toml` and basic package structure
2. Standardize script naming convention for new scripts
3. Enhance `4_thesis/` organization

### **Short-term (Medium Risk)**
1. Consolidate duplicate utilities into shared location
2. Flatten experiment results structure for new experiments
3. Improve import structure gradually

### **Long-term (High Risk)**
1. Complete migration to new package structure
2. Update all existing scripts and pipelines
3. Comprehensive test suite update

## 8. Conclusion

The Temporal-Compression project has a fundamentally sound structure that effectively supports the research workflow. The numbered directory approach (1-4) provides clear separation of concerns and logical progression through the research pipeline.

The main areas for improvement are:
1. **Eliminating code duplication** (spherical_harmonics.py, sun_position.py)
2. **Standardizing naming conventions** across scripts
3. **Creating a proper Python package** structure
4. **Optimizing the experiment results** directory hierarchy
5. **Better utilizing the thesis directory** for academic outputs

These improvements would reduce maintenance burden, improve code clarity, and make the project more accessible to new contributors while preserving the workflow-oriented approach that makes the current structure effective for research.

**Recommendation**: Implement the "Priority 1" and "Priority 5" recommendations first, as they provide the most value with moderate risk. The other improvements can be phased in gradually as the project evolves.