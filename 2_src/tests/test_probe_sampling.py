import numpy as np
import pytest

from tools.probe_sampling import (
    compute_light_weights,
    farthest_point_sampling,
    load_light_trajectories,
    sample_with_decluster,
)


def test_load_light_trajectories_single(tmp_path):
    traj = np.zeros((4, 3), dtype=np.float32)
    path = tmp_path / "traj.npy"
    np.save(path, traj)
    loaded = load_light_trajectories([str(path)])
    assert len(loaded) == 1
    assert loaded[0].shape == (4, 3)


def test_load_light_trajectories_multi(tmp_path):
    traj = np.zeros((4, 2, 3), dtype=np.float32)
    path = tmp_path / "traj.npy"
    np.save(path, traj)
    loaded = load_light_trajectories([str(path)])
    assert len(loaded) == 2
    assert loaded[0].shape == (4, 3)


def test_load_light_trajectories_invalid(tmp_path):
    traj = np.zeros((4, 4), dtype=np.float32)
    path = tmp_path / "traj.npy"
    np.save(path, traj)
    with pytest.raises(ValueError):
        load_light_trajectories([str(path)])


def test_compute_light_weights_shape():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    traj = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    weights = compute_light_weights(points, [traj], step=1, eps=0.1, temporal_weight=0.5)
    assert weights.shape == (2,)
    assert (weights > 0).all()


def test_compute_light_weights_invalid_points():
    with pytest.raises(ValueError):
        compute_light_weights(np.zeros((3, 2), dtype=np.float32), [], step=1)


def test_farthest_point_sampling():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    sampled = farthest_point_sampling(points, 2, seed=0)
    assert sampled.shape == (2, 3)


def test_farthest_point_sampling_invalid():
    points = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        farthest_point_sampling(points, 0)


def test_sample_with_decluster():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(50, 3)).astype(np.float32)
    weights = np.ones((50,), dtype=np.float32)
    sampled = sample_with_decluster(
        points,
        weights,
        total=10,
        adaptive_ratio=0.5,
        rng=rng,
        decluster=True,
        decluster_factor=2.0,
        decluster_seed=0,
    )
    assert sampled.shape == (10, 3)


def test_sample_with_decluster_invalid_total():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(10, 3)).astype(np.float32)
    weights = np.ones((10,), dtype=np.float32)
    with pytest.raises(ValueError):
        sample_with_decluster(points, weights, total=0, adaptive_ratio=0.5, rng=rng)
