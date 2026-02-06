"""Model exports.

NOTE: Architecture evolution timeline:
1. GaussianPhysicsCompression1D/5D: Historical hard-coded physics basis (legacy)
2. GaussianPhysicsCompressionUnified: Current FiLM-based unified architecture (mainline)

TODO: Consider marking legacy models as deprecated in future versions.
"""

from .gaussian_physics_1D import GaussianPhysicsCompression1D
from .gaussian_physics_5D import GaussianPhysicsCompression5D
from .gaussian_physics_unified import GaussianPhysicsCompressionUnified

__all__ = [
    "GaussianPhysicsCompression1D",
    "GaussianPhysicsCompression5D",
    "GaussianPhysicsCompressionUnified",
]
