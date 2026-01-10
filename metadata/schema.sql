-- SQLite Schema for Experiment Memory & State Recovery System
-- Purpose: Persistent storage for experiments, datasets, scripts, tasks, and environment config
-- Solves: Context loss after conversation compaction

PRAGMA foreign_keys = ON;

-- ====================
-- 1. Foundation Tables
-- ====================

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,        -- Short name: D15K, D2K, MT1, TPE_CB, etc.
    full_name TEXT NOT NULL,            -- Descriptive name: dataset_15k, method_test_v1, etc.
    root_path TEXT NOT NULL,            -- Absolute path
    num_probes INTEGER,                 -- Actual probe count (from probes.npz, NOT config.json)
    num_moments INTEGER,                -- Number of time moments
    spp INTEGER,                        -- Samples per pixel
    sh_samples INTEGER,                 -- Spherical harmonics samples
    created_at INTEGER,                 -- Unix timestamp
    notes TEXT                          -- Additional metadata (e.g., config.json inconsistencies)
);

CREATE TABLE IF NOT EXISTS scenes (
    scene_id TEXT PRIMARY KEY,          -- Short name: CB (cornell-box), S2 (staircase2), HS (house)
    dataset_id TEXT NOT NULL,           -- Associated dataset (can be shared across multiple datasets)
    scene_xml TEXT,                     -- Path to scene.xml from project root
    description TEXT,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scripts (
    script_id TEXT PRIMARY KEY,         -- Short name: train_pgcpl5d, run_tpe, ablation_5D, etc.
    filename TEXT NOT NULL,             -- Full path from project root
    method TEXT,                        -- TPE, TemporalMLP, PG-GCPL, etc.
    description TEXT
);

-- ====================
-- 2. Core Experiment Table
-- ====================

CREATE TABLE IF NOT EXISTS experiments (
    exp_id TEXT PRIMARY KEY,            -- Format: EXP-YYYYMMDD-### (e.g., EXP-20260104-001)
    run_order INTEGER UNIQUE NOT NULL,  -- Sequential order across ALL experiments (never resets)
    phase TEXT,                         -- Phase0_TPE, Phase1_TemporalMLP, Phase2_PGCPL
    stage TEXT CHECK(stage IN ('validation','training','ablation','visualization','evaluation')),
    script_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    scene_id TEXT,                      -- Optional (some datasets don't have associated scenes)

    -- Core hyperparameters (extracted for easy aggregation/comparison)
    num_gaussians INTEGER,              -- K value (Gaussian count)
    rank INTEGER,                       -- r value (low-rank factorization)
    latent_dim INTEGER,                 -- Latent code dimension
    learning_rate REAL,
    num_epochs INTEGER,

    -- Results metrics
    psnr REAL,                          -- Peak Signal-to-Noise Ratio (dB)
    ssim REAL,                          -- Structural Similarity Index
    mae REAL,                           -- Mean Absolute Error
    rmse REAL,                          -- Root Mean Squared Error
    compress_ratio REAL,                -- Compression ratio (e.g., 16.59)
    param_count INTEGER,                -- Total parameter count
    query_latency_ms REAL,              -- Query latency in milliseconds

    -- Baseline marking (only one baseline per phase)
    is_baseline BOOLEAN DEFAULT 0,      -- True if this is the reference baseline

    -- Full details (JSON storage for flexibility)
    args_json TEXT,                     -- Complete hyperparameters as JSON
    results_json TEXT,                  -- Complete results as JSON
    checkpoint_path TEXT,               -- Path to checkpoint directory
    notes TEXT,                         -- Free-form notes

    created_at INTEGER,                 -- Unix timestamp

    FOREIGN KEY (script_id) REFERENCES scripts(script_id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
    FOREIGN KEY (scene_id) REFERENCES scenes(scene_id)
);

-- ====================
-- 3. Task Management Table
-- ====================

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_desc TEXT NOT NULL,
    status TEXT CHECK(status IN ('todo', 'doing', 'done')) DEFAULT 'todo',
    priority INTEGER DEFAULT 1,         -- 1=low, 2=medium, 3=high
    related_exp_id TEXT,                -- Link to experiment if applicable
    created_at INTEGER,
    completed_at INTEGER,               -- Unix timestamp when marked done
    FOREIGN KEY (related_exp_id) REFERENCES experiments(exp_id)
);

-- ====================
-- 4. Environment Configuration
-- ====================

CREATE TABLE IF NOT EXISTS env_config (
    key TEXT PRIMARY KEY,               -- e.g., "working_dir", "conda_env", "data_root"
    value TEXT,
    updated_at INTEGER                  -- Unix timestamp
);

-- ====================
-- 5. Useful Views
-- ====================

-- View: Milestones (current baseline + historical best)
CREATE VIEW IF NOT EXISTS v_milestones AS
SELECT
    'BASELINE' as type,
    exp_id,
    phase,
    psnr,
    ssim,
    compress_ratio,
    param_count,
    created_at
FROM experiments
WHERE is_baseline = 1
UNION ALL
SELECT
    'BEST_PSNR' as type,
    exp_id,
    phase,
    psnr,
    ssim,
    compress_ratio,
    param_count,
    created_at
FROM experiments
WHERE psnr IS NOT NULL
ORDER BY psnr DESC
LIMIT 1;

-- View: Recent experiments (last 10)
CREATE VIEW IF NOT EXISTS v_recent_experiments AS
SELECT
    e.exp_id,
    e.run_order,
    e.phase,
    e.stage,
    s.filename as script,
    d.dataset_id,
    e.psnr,
    e.ssim,
    e.is_baseline,
    datetime(e.created_at, 'unixepoch') as created
FROM experiments e
JOIN scripts s ON e.script_id = s.script_id
JOIN datasets d ON e.dataset_id = d.dataset_id
ORDER BY e.run_order DESC
LIMIT 10;

-- View: Active tasks (todo + doing, sorted by priority)
CREATE VIEW IF NOT EXISTS v_active_tasks AS
SELECT
    id,
    task_desc,
    status,
    priority,
    related_exp_id,
    datetime(created_at, 'unixepoch') as created
FROM tasks
WHERE status != 'done'
ORDER BY priority DESC, created_at ASC;

-- View: Dataset usage statistics
CREATE VIEW IF NOT EXISTS v_dataset_usage AS
SELECT
    d.dataset_id,
    d.full_name,
    d.num_probes,
    d.num_moments,
    COUNT(e.exp_id) as experiment_count,
    MAX(e.created_at) as last_used
FROM datasets d
LEFT JOIN experiments e ON d.dataset_id = e.dataset_id
GROUP BY d.dataset_id
ORDER BY experiment_count DESC, last_used DESC;

-- ====================
-- 6. Indexes for Performance
-- ====================

CREATE INDEX IF NOT EXISTS idx_experiments_phase ON experiments(phase);
CREATE INDEX IF NOT EXISTS idx_experiments_is_baseline ON experiments(is_baseline);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);
