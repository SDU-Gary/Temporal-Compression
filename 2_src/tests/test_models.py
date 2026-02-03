"""Tests for neural network model components."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np
import yaml

temporal_module = pytest.importorskip("models.temporal_mlp", reason="Legacy TemporalMLP module not available")
decoder_module = pytest.importorskip("models.decoder_mlp", reason="Legacy DecoderMLP module not available")
full_model_module = pytest.importorskip("models.full_model", reason="Legacy full model module not available")

from models.gaussian_mixture import GaussianMixture
TemporalMLP = temporal_module.TemporalMLP
DecoderMLP = decoder_module.DecoderMLP
MultiTimeCompressionModel = full_model_module.MultiTimeCompressionModel


@pytest.fixture
def sample_positions():
    """Generate sample probe positions for testing."""
    np.random.seed(42)
    # Use 1000 positions to accommodate baseline config (750 Gaussians)
    return np.random.randn(1000, 3).astype(np.float32)


@pytest.fixture
def sample_sun_directions():
    """Generate sample sun directions for testing."""
    # 6 time moments
    sun_dirs = np.array([
        [1.0, 0.0, 0.0],  # 06:00
        [0.522, 0.675, -0.522],  # 09:00
        [0.0, 0.866, -0.5],  # 12:00
        [-0.522, 0.675, -0.522],  # 15:00
        [-1.0, 0.0, 0.0],  # 18:00
        [-0.707, 0.0, 0.707]  # 21:00
    ], dtype=np.float32)
    return sun_dirs


@pytest.fixture
def model_config():
    """Load baseline configuration."""
    config_path = Path(__file__).parent.parent / 'configs' / 'baseline.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_gaussian_mixture_init():
    """Test Gaussian mixture initialization."""
    num_gaussians = 50
    latent_dim = 6

    model = GaussianMixture(num_gaussians=num_gaussians, latent_dim=latent_dim)

    assert model.num_gaussians == num_gaussians
    assert model.latent_dim == latent_dim
    assert model.means.shape == (num_gaussians, 3)
    assert model.scales.shape == (num_gaussians, 3)
    assert model.rotations.shape == (num_gaussians, 4)
    assert model.latent_codes.shape == (num_gaussians, latent_dim)

    print(f"✓ GaussianMixture initialized: {num_gaussians} Gaussians, {latent_dim}-D latent")


def test_gaussian_mixture_kmeans(sample_positions):
    """Test K-Means initialization."""
    num_gaussians = 20
    latent_dim = 6

    model = GaussianMixture(num_gaussians=num_gaussians, latent_dim=latent_dim)
    model.initialize_from_positions(sample_positions)

    # Check that means are set
    assert not torch.allclose(model.means, torch.zeros_like(model.means))
    # Check that scales are positive
    assert (torch.exp(model.scales) > 0).all()

    print(f"✓ K-Means initialization successful with {len(sample_positions)} positions")


def test_gaussian_mixture_forward():
    """Test Gaussian mixture forward pass."""
    num_gaussians = 20
    latent_dim = 6

    model = GaussianMixture(num_gaussians=num_gaussians, latent_dim=latent_dim)

    # Test single batch
    positions = torch.randn(10, 3)
    output = model(positions)

    assert output.shape == (10, latent_dim)
    print(f"✓ Gaussian forward pass: {positions.shape} → {output.shape}")


def test_temporal_mlp_forward():
    """Test temporal MLP forward pass."""
    input_dim = 9  # 6 (base latent) + 3 (sun dir)
    hidden_dim = 32
    output_dim = 9

    model = TemporalMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        output_dim=output_dim
    )

    # Test Case 1: single position, single time
    base_latent = torch.randn(10, 6)
    sun_dir = torch.randn(3)
    output = model(base_latent, sun_dir)
    assert output.shape == (10, output_dim)
    print(f"✓ TemporalMLP Case 1: base={base_latent.shape}, sun={sun_dir.shape} → {output.shape}")

    # Test Case 2: single position, multiple times
    sun_dirs = torch.randn(6, 3)
    output = model(base_latent, sun_dirs)
    assert output.shape == (10, 6, output_dim)
    print(f"✓ TemporalMLP Case 2: base={base_latent.shape}, sun={sun_dirs.shape} → {output.shape}")


def test_decoder_mlp_forward():
    """Test decoder MLP forward pass."""
    input_dim = 15  # 9 (time latent) + 3 (position) + 3 (time encoding)
    hidden_dim = 64
    output_dim = 27  # 9 SH bases × 3 RGB

    model = DecoderMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        output_dim=output_dim
    )

    # Test Case 1: single time
    latent_code = torch.randn(10, 9)
    position = torch.randn(10, 3)
    sun_dir = torch.randn(3)
    output = model(latent_code, position, sun_dir)
    assert output.shape == (10, output_dim)
    print(f"✓ DecoderMLP Case 1: latent={latent_code.shape}, pos={position.shape} → {output.shape}")

    # Test Case 2: multiple times
    latent_code = torch.randn(10, 6, 9)
    sun_dirs = torch.randn(6, 3)
    output = model(latent_code, position, sun_dirs)
    assert output.shape == (10, 6, output_dim)
    print(f"✓ DecoderMLP Case 2: latent={latent_code.shape}, sun={sun_dirs.shape} → {output.shape}")


def test_full_model(model_config, sample_positions, sample_sun_directions):
    """Test full integrated model."""
    # Create model
    model = MultiTimeCompressionModel(model_config)

    # Initialize Gaussians
    model.initialize_gaussians(sample_positions)

    # Forward pass
    positions = torch.from_numpy(sample_positions[:10]).float()
    sun_dirs = torch.from_numpy(sample_sun_directions).float()

    sh_coeffs = model(positions, sun_dirs)

    expected_shape = (10, 6, 27)
    assert sh_coeffs.shape == expected_shape, f"Expected {expected_shape}, got {sh_coeffs.shape}"

    print(f"✓ Full model forward: pos={positions.shape}, sun={sun_dirs.shape} → SH={sh_coeffs.shape}")

    # Test model size calculation
    size_bytes, size_mb = model.get_model_size()
    print(f"  Model size: {size_mb:.2f} MB")

    # Test compression ratio
    num_probes = 343
    num_moments = 6
    ratio = model.get_compression_ratio(num_probes, num_moments)
    print(f"  Compression ratio: {ratio:.2f}x")


def test_model_gradients():
    """Test that gradients flow through the model."""
    config = {
        'model': {
            'num_gaussians': 10,
            'latent_dim_base': 6,
            'latent_dim_time': 9,
            'temporal_mlp': {
                'input_dim': 9,
                'hidden_dim': 32,
                'num_layers': 2,
                'activation': 'relu'
            },
            'decoder_mlp': {
                'input_dim': 15,
                'hidden_dim': 64,
                'num_layers': 2,
                'output_dim': 27,
                'activation': 'relu'
            }
        }
    }

    model = MultiTimeCompressionModel(config)

    # Initialize with random positions
    positions_np = np.random.randn(50, 3).astype(np.float32)
    model.initialize_gaussians(positions_np)

    # Forward pass
    positions = torch.randn(5, 3, requires_grad=True)
    sun_dirs = torch.randn(3, 3)

    output = model(positions, sun_dirs)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients exist
    assert model.gaussian_mixture.means.grad is not None
    assert model.gaussian_mixture.latent_codes.grad is not None

    print("✓ Gradients flow through all model components")


if __name__ == "__main__":
    print("Running model tests...\n")

    print("=" * 70)
    print("Test 1: Gaussian Mixture Initialization")
    print("=" * 70)
    test_gaussian_mixture_init()

    print("\n" + "=" * 70)
    print("Test 2: Gaussian Mixture K-Means")
    print("=" * 70)
    sample_pos = np.random.randn(1000, 3).astype(np.float32)
    test_gaussian_mixture_kmeans(sample_pos)

    print("\n" + "=" * 70)
    print("Test 3: Gaussian Mixture Forward")
    print("=" * 70)
    test_gaussian_mixture_forward()

    print("\n" + "=" * 70)
    print("Test 4: Temporal MLP Forward")
    print("=" * 70)
    test_temporal_mlp_forward()

    print("\n" + "=" * 70)
    print("Test 5: Decoder MLP Forward")
    print("=" * 70)
    test_decoder_mlp_forward()

    print("\n" + "=" * 70)
    print("Test 6: Full Model Integration")
    print("=" * 70)
    config_path = Path(__file__).parent.parent / 'configs' / 'baseline.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    sample_pos = np.random.randn(1000, 3).astype(np.float32)
    sun_dirs = np.array([
        [1.0, 0.0, 0.0],
        [0.522, 0.675, -0.522],
        [0.0, 0.866, -0.5],
        [-0.522, 0.675, -0.522],
        [-1.0, 0.0, 0.0],
        [-0.707, 0.0, 0.707]
    ], dtype=np.float32)
    test_full_model(config, sample_pos, sun_dirs)

    print("\n" + "=" * 70)
    print("Test 7: Gradient Flow")
    print("=" * 70)
    test_model_gradients()

    print("\n" + "=" * 70)
    print("All model tests passed!")
    print("=" * 70)
