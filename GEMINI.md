# GEMINI.md - Project Guide

This document provides a comprehensive guide for interacting with and developing the "Hierarchical Neural Compression for Multi-Moment Lighting" graduation project.

## 1. Project Overview

This is a research project for a graduate thesis focused on developing a novel method for compressing lighting data from multiple times of day to enable real-time rendering. The goal is to achieve high compression ratios while maintaining visual quality (PSNR > 35dB) and fast decompression (< 0.5ms).

The core technical approach is **Physics-Guided Gaussian Compression for Parametric Lighting (PG-GCPL)**. This method combines:

1.  **Spatial Compression**: A **Gaussian Mixture Model** represents the spatial distribution of lighting probes in the scene.
2.  **Temporal Compression**: A **Physics-Guided Low-Rank Factorization** compresses the time-varying lighting information (represented as Spherical Harmonics coefficients). This factorization uses a physical basis function derived from the sun's position, allowing for efficient and accurate interpolation and extrapolation of lighting conditions.

The project is built on **Python**, **PyTorch**, and the **Mitsuba 3** rendering system. The development process is highly structured, involving rigorous experimentation, and all results are tracked in a dedicated SQLite database (`project.db`).

The authoritative goal of the project is defined in the thesis task document: `docs/thesis/多时刻光照压缩任务书.md`.

## 2. Project Structure

The repository is organized into several key directories:

-   `data_generation/`: Scripts for generating datasets using Mitsuba 3. See `data_generation/README.md`.
-   `multi_time_compression/`: The main development directory containing the source code for the compression models, training scripts, and detailed experiment reports.
    -   `multi_time_compression/PG-GCPL/`: Contains the source for the current, most successful method (PG-GCPL). This is where active development happens.
    -   `multi_time_compression/archive/`: Contains code from previous, unsuccessful experimental approaches (like TPE).
-   `docs/`: General project documentation, including research notes and the official thesis proposal.
-   `tools/`: Utility scripts for managing the project, including `logexp.py` for logging experiments to the database.
-   `analysis/`: In-depth analysis of foundational research papers.
-   `project.db`: An SQLite database that tracks all experiments, datasets, and development tasks.
-   `project_state.md`: An auto-generated summary of the current project status, derived from the database.

## 3. Getting Started

### 3.1. Environment Setup

The project uses a Python virtual environment.

1.  **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```

2.  **Verify dependencies:**
    Ensure key libraries like PyTorch, Mitsuba, and DrJit are installed and correctly configured.
    ```bash
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    python -c "import mitsuba as mi; mi.set_variant('cuda_ad_rgb'); print(f'Mitsuba variants: {mi.variants()}')"
    ```

### 3.2. Data Generation

The training data consists of scenes rendered at multiple times of day with varying sun positions.

1.  **Navigate to the data generation directory:**
    ```bash
    cd data_generation/
    ```

2.  **Run the main generation script:**
    This script uses Mitsuba to render the scenes defined in `data_generation/scenes/` and generates the necessary `.npz` files containing probe positions and SH coefficients.
    ```bash
    python generate_dataset.py
    ```
    Generated data is stored in the `data_generation/output/` directory.

### 3.3. Training the Model

The main training scripts are located within the `multi_time_compression/PG-GCPL/` directory. The training process is highly configurable via `.yaml` files.

1.  **Navigate to the main compression directory:**
    ```bash
    cd multi_time_compression/
    ```

2.  **Train the PG-GCPL model:**
    The current best model is trained using the following script.
    ```bash
    # Navigate to the correct subdirectory for the PG-GCPL method
    cd PG-GCPL/src/scripts/training/

    # Run the training script for the 5D parametric model
    python train_gaussian_physics_5D.py --config ../../../../configs/pg_gcpl/K30_r8.yaml
    ```
    *Note: The exact configuration file might change. Refer to `project_state.md` or the database for the latest optimal configuration.*

## 4. Development Conventions

This project follows a strict, data-driven development methodology.

-   **Experiment Tracking**: All experiments **must** be logged to the `project.db` database using the `tools/logexp.py` script. This ensures reproducibility and provides a clear history of what has been tried.
    ```bash
    # Example of logging a new experiment
    python tools/logexp.py log --phase Phase2_PGCPL --script train_gaussian_physics_5D --dataset 5D_PARAM --results '{"mae": 0.0343, "ssim": 0.951, "compress_ratio": 16.59}'
    ```

-   **State Summary**: Always refer to the `project_state.md` file for a quick overview of active tasks, recent experiments, and key dataset paths. This file is the "single source of truth" for the project's current status. To regenerate it:
    ```bash
    python tools/summarize_state.py
    ```

-   **Configuration Files**: Hyperparameters are managed exclusively through `.yaml` files in the `multi_time_compression/configs/` directory. Do not hardcode parameters in scripts.

-   **Documentation**: The `CLAUDE.md` and `EXPERIMENTAL_WORKFLOW.md` files contain extensive details on the project's architecture and the evolution of the research. They are invaluable resources for understanding the "why" behind the current implementation.
