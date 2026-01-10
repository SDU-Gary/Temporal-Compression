"""Tests for MultiTimeLightingDataset."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np

from data.dataset import MultiTimeLightingDataset


# Path to test dataset
TEST_DATASET_PATH = Path(__file__).parent.parent.parent / "data_generation" / "output" / "method_test_v1"


@pytest.fixture
def dataset_path():
    """Get dataset path."""
    if not TEST_DATASET_PATH.exists():
        pytest.skip(f"Test dataset not found at {TEST_DATASET_PATH}")
    return TEST_DATASET_PATH


def test_dataset_loading(dataset_path):
    """Test basic dataset loading."""
    dataset = MultiTimeLightingDataset(dataset_path, split='train')

    assert len(dataset) > 0, "Dataset should not be empty"
    assert dataset.num_moments > 0, "Should have at least one moment"
    print(f"✓ Loaded {len(dataset)} samples with {dataset.num_moments} moments")


def test_dataset_splits(dataset_path):
    """Test dataset splitting."""
    train_dataset = MultiTimeLightingDataset(dataset_path, split='train', train_ratio=0.6, val_ratio=0.2)
    val_dataset = MultiTimeLightingDataset(dataset_path, split='val', train_ratio=0.6, val_ratio=0.2)
    test_dataset = MultiTimeLightingDataset(dataset_path, split='test', train_ratio=0.6, val_ratio=0.2)

    total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)

    assert len(train_dataset) > 0, "Train split should not be empty"
    assert len(val_dataset) > 0, "Val split should not be empty"
    assert len(test_dataset) > 0, "Test split should not be empty"

    print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"  Total: {total_samples} samples")


def test_sample_structure(dataset_path):
    """Test sample structure and shapes."""
    dataset = MultiTimeLightingDataset(dataset_path, split='train')
    sample = dataset[0]

    # Check keys
    assert 'position' in sample
    assert 'sh_gt' in sample
    assert 'sun_dirs' in sample
    assert 'moment_ids' in sample

    # Check shapes
    assert sample['position'].shape == (3,), f"Expected position shape (3,), got {sample['position'].shape}"
    assert sample['sh_gt'].shape[0] == dataset.num_moments, "SH coefficients should match number of moments"
    assert sample['sh_gt'].shape[1] == 27, "SH coefficients should be 27-dimensional"
    assert sample['sun_dirs'].shape == (dataset.num_moments, 3), "Sun directions should be (T, 3)"
    assert sample['moment_ids'].shape == (dataset.num_moments,), "Moment IDs should be (T,)"

    # Check types
    assert isinstance(sample['position'], torch.Tensor)
    assert isinstance(sample['sh_gt'], torch.Tensor)
    assert isinstance(sample['sun_dirs'], torch.Tensor)
    assert isinstance(sample['moment_ids'], torch.Tensor)

    print(f"✓ Sample structure valid:")
    print(f"  position: {sample['position'].shape}")
    print(f"  sh_gt: {sample['sh_gt'].shape}")
    print(f"  sun_dirs: {sample['sun_dirs'].shape}")
    print(f"  moment_ids: {sample['moment_ids'].shape}")


def test_normalization(dataset_path):
    """Test position normalization and denormalization."""
    dataset = MultiTimeLightingDataset(dataset_path, split='train')

    # Get normalization params
    mean, std = dataset.get_normalization_params()

    assert mean.shape == (3,), "Mean should be 3D"
    assert std.shape == (3,), "Std should be 3D"
    assert np.all(std > 0), "Standard deviation should be positive"

    # Test denormalization
    sample = dataset[0]
    denormalized = dataset.denormalize_position(sample['position'].numpy())

    assert denormalized.shape == (3,), "Denormalized position should be 3D"

    print(f"✓ Normalization params:")
    print(f"  mean: {mean}")
    print(f"  std: {std}")


def test_batch_loading(dataset_path):
    """Test batch loading with DataLoader."""
    from torch.utils.data import DataLoader

    dataset = MultiTimeLightingDataset(dataset_path, split='train')
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    batch = next(iter(loader))

    assert batch['position'].shape == (4, 3), "Batch position shape should be (batch_size, 3)"
    assert batch['sh_gt'].shape == (4, dataset.num_moments, 27), "Batch SH shape should be (batch_size, T, 27)"
    assert batch['sun_dirs'].shape == (4, dataset.num_moments, 3), "Batch sun_dirs shape should be (batch_size, T, 3)"

    print(f"✓ Batch loading works:")
    print(f"  batch_size: 4")
    print(f"  position: {batch['position'].shape}")
    print(f"  sh_gt: {batch['sh_gt'].shape}")


if __name__ == "__main__":
    # Run tests manually
    print("Running dataset tests...\n")

    if not TEST_DATASET_PATH.exists():
        print(f"Test dataset not found at {TEST_DATASET_PATH}")
        print("Please generate test dataset first using data_generation")
        sys.exit(1)

    print("=" * 70)
    print("Test 1: Dataset Loading")
    print("=" * 70)
    test_dataset_loading(TEST_DATASET_PATH)

    print("\n" + "=" * 70)
    print("Test 2: Dataset Splits")
    print("=" * 70)
    test_dataset_splits(TEST_DATASET_PATH)

    print("\n" + "=" * 70)
    print("Test 3: Sample Structure")
    print("=" * 70)
    test_sample_structure(TEST_DATASET_PATH)

    print("\n" + "=" * 70)
    print("Test 4: Normalization")
    print("=" * 70)
    test_normalization(TEST_DATASET_PATH)

    print("\n" + "=" * 70)
    print("Test 5: Batch Loading")
    print("=" * 70)
    test_batch_loading(TEST_DATASET_PATH)

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
