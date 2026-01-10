"""Compatibility wrapper for Gaussian-Physics trainers."""

from .gaussian_physics_trainer import BatchAdapter, GaussianPhysicsTrainer

# Backwards-compatible alias for older imports.
Trainer = GaussianPhysicsTrainer

__all__ = ["BatchAdapter", "GaussianPhysicsTrainer", "Trainer"]
