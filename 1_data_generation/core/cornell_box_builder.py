"""Cornell Box scene builder with dynamic point light.

Programmatically constructs Cornell Box scene using Mitsuba 3 Python API.
Supports arbitrary point light positioning for transmission tensor generation.
"""

import numpy as np
import mitsuba as mi
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from mitsuba import Scene


def build_cornell_box_with_point_light(
    light_position: np.ndarray,
    light_intensity: float = 50.0,
    box_size: float = 1.0
) -> 'Scene':
    """Build Cornell Box scene with point light at specified position.

    Args:
        light_position: [3] array of (x, y, z) light position in world coords
        light_intensity: Radiant intensity of point light (default 50.0)
        box_size: Side length of Cornell Box in meters (default 1.0)

    Returns:
        scene: Loaded Mitsuba scene object

    Notes:
        - Box corners: [-0.5, 0.5] in x, y; [0, 1] in z (Y-up convention)
        - Left wall: red diffuse (0.63, 0.065, 0.05)
        - Right wall: green diffuse (0.14, 0.45, 0.091)
        - Other walls: white diffuse (0.725, 0.71, 0.68)
        - Small/tall boxes: white diffuse, positioned at standard locations
    """
    half_size = box_size / 2.0

    # Scene dictionary for mi.load_dict()
    scene_dict = {
        'type': 'scene',

        # Integrator: path tracer
        'integrator': {
            'type': 'path',
            'max_depth': 8
        },

        # Point light source
        'point_light': {
            'type': 'point',
            'position': light_position.tolist(),
            'intensity': {
                'type': 'rgb',
                'value': [light_intensity, light_intensity, light_intensity]
            }
        },

        # Floor (white)
        'floor': {
            'type': 'rectangle',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            },
            'to_world': mi.ScalarTransform4f.scale([half_size, half_size, 1])
                       .rotate([1, 0, 0], -90)  # XY plane at z=0
                       .translate([0, 0, 0])
        },

        # Ceiling (white)
        'ceiling': {
            'type': 'rectangle',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            },
            'to_world': mi.ScalarTransform4f.scale([half_size, half_size, 1])
                       .rotate([1, 0, 0], 90)  # XY plane at z=box_size
                       .translate([0, 0, box_size])
        },

        # Back wall (white)
        'back_wall': {
            'type': 'rectangle',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            },
            'to_world': mi.ScalarTransform4f.scale([half_size, half_size, 1])
                       .translate([0, -half_size, half_size])
        },

        # Left wall (red)
        'left_wall': {
            'type': 'rectangle',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.63, 0.065, 0.05]}
            },
            'to_world': mi.ScalarTransform4f.scale([half_size, half_size, 1])
                       .rotate([0, 1, 0], -90)  # YZ plane at x=-half_size
                       .translate([-half_size, 0, half_size])
        },

        # Right wall (green)
        'right_wall': {
            'type': 'rectangle',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.14, 0.45, 0.091]}
            },
            'to_world': mi.ScalarTransform4f.scale([half_size, half_size, 1])
                       .rotate([0, 1, 0], 90)  # YZ plane at x=half_size
                       .translate([half_size, 0, half_size])
        },

        # Small box (white, positioned front-right)
        'small_box': {
            'type': 'cube',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            },
            'to_world': mi.ScalarTransform4f.scale([0.15, 0.15, 0.15])
                       .rotate([0, 0, 1], -18)  # Slight rotation
                       .translate([0.2, 0.2, 0.15])
        },

        # Tall box (white, positioned back-left)
        'tall_box': {
            'type': 'cube',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            },
            'to_world': mi.ScalarTransform4f.scale([0.15, 0.15, 0.3])
                       .rotate([0, 0, 1], 16)  # Slight rotation
                       .translate([-0.2, -0.2, 0.3])
        }
    }

    # Load scene from dictionary
    scene = mi.load_dict(scene_dict)
    return scene


def build_cornell_box_with_area_light(
    intensity: float = 1.0,
    color: List[float] = [1.0, 1.0, 1.0],
    box_size: float = 1.0,
    light_scale: float = 0.3,
    radiance_scale: float = 50.0
) -> dict:
    """Build Cornell Box scene with area light on ceiling.

    Used for intensity modulation experiments where a ceiling area light
    varies in intensity over time.

    Args:
        intensity: Light intensity multiplier [0, 1]
        color: RGB color of light (default: white)
        box_size: Side length of Cornell Box in meters (default 1.0)
        light_scale: Scale of area light rectangle relative to box size (default 0.3)
        radiance_scale: Scale factor for radiance in W/(sr·m²) (default 50.0)

    Returns:
        scene_dict: Scene dictionary ready for mi.load_dict()

    Notes:
        - Area light positioned at ceiling center, facing downward
        - Light size: light_scale * box_size (e.g., 0.3m × 0.3m for box_size=1.0)
        - Radiance: radiance_scale * intensity * color (modulated by intensity parameter)
        - Same Cornell Box geometry as point light version
    """
    half_size = box_size / 2.0
    light_size = light_scale * box_size

    # Scene dictionary for mi.load_dict()
    scene_dict = {
        'type': 'scene',

        # Integrator: path tracer
        'integrator': {
            'type': 'path',
            'max_depth': 8
        },

        # Area light on ceiling
        'ceiling_light': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.scale([light_size, light_size, 1])
                       .rotate([1, 0, 0], 180)  # Face downward
                       .translate([0, 0, box_size - 0.01]),  # Just below ceiling
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [radiance_scale * intensity * c for c in color]
                }
            }
        },

        # Floor (white)
        'floor': {
            'type': 'rectangle',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            },
            'to_world': mi.ScalarTransform4f.scale([half_size, half_size, 1])
                       .rotate([1, 0, 0], -90)  # XY plane at z=0
                       .translate([0, 0, 0])
        },

        # Ceiling (white) - non-emissive surface
        'ceiling': {
            'type': 'rectangle',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            },
            'to_world': mi.ScalarTransform4f.scale([half_size, half_size, 1])
                       .rotate([1, 0, 0], 90)  # XY plane at z=box_size
                       .translate([0, 0, box_size])
        },

        # Back wall (white)
        'back_wall': {
            'type': 'rectangle',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            },
            'to_world': mi.ScalarTransform4f.scale([half_size, half_size, 1])
                       .translate([0, -half_size, half_size])
        },

        # Left wall (red)
        'left_wall': {
            'type': 'rectangle',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.63, 0.065, 0.05]}
            },
            'to_world': mi.ScalarTransform4f.scale([half_size, half_size, 1])
                       .rotate([0, 1, 0], -90)  # YZ plane at x=-half_size
                       .translate([-half_size, 0, half_size])
        },

        # Right wall (green)
        'right_wall': {
            'type': 'rectangle',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.14, 0.45, 0.091]}
            },
            'to_world': mi.ScalarTransform4f.scale([half_size, half_size, 1])
                       .rotate([0, 1, 0], 90)  # YZ plane at x=half_size
                       .translate([half_size, 0, half_size])
        },

        # Small box (white, positioned front-right)
        'small_box': {
            'type': 'cube',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            },
            'to_world': mi.ScalarTransform4f.scale([0.15, 0.15, 0.15])
                       .rotate([0, 0, 1], -18)  # Slight rotation
                       .translate([0.2, 0.2, 0.15])
        },

        # Tall box (white, positioned back-left)
        'tall_box': {
            'type': 'cube',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            },
            'to_world': mi.ScalarTransform4f.scale([0.15, 0.15, 0.3])
                       .rotate([0, 0, 1], 16)  # Slight rotation
                       .translate([-0.2, -0.2, 0.3])
        }
    }

    return scene_dict


def generate_circular_light_trajectory(
    center: Tuple[float, float, float] = (0.0, 0.0, 1.5),
    radius: float = 1.2,
    num_positions: int = 12,
    start_angle: float = 0.0
) -> np.ndarray:
    """Generate circular trajectory for point light source.

    Args:
        center: (x, y, z) center of circular path
        radius: Radius of circular path in xy-plane
        num_positions: Number of uniformly spaced positions
        start_angle: Starting angle in radians (0 = +x axis)

    Returns:
        positions: [num_positions, 3] array of light positions

    Example:
        >>> positions = generate_circular_light_trajectory(
        ...     center=(0, 0, 1.5),
        ...     radius=1.2,
        ...     num_positions=12
        ... )
        >>> positions.shape
        (12, 3)
    """
    angles = np.linspace(start_angle, start_angle + 2*np.pi, num_positions, endpoint=False)

    positions = np.zeros((num_positions, 3))
    positions[:, 0] = center[0] + radius * np.cos(angles)  # x
    positions[:, 1] = center[1] + radius * np.sin(angles)  # y
    positions[:, 2] = center[2]  # z (constant height)

    return positions


def rebuild_scene_with_new_light(
    base_scene_dict: dict,
    light_position: np.ndarray,
    light_intensity: float = 50.0
) -> 'Scene':
    """Rebuild scene with updated point light position.

    Efficient for batch rendering: keeps scene geometry unchanged,
    only updates light position.

    Args:
        base_scene_dict: Scene dictionary from build_cornell_box_with_point_light()
        light_position: [3] new light position
        light_intensity: Updated light intensity (optional)

    Returns:
        scene: Reloaded scene with new light

    Notes:
        This is more efficient than rebuilding geometry from scratch.
        For transmission tensor generation, call this in a loop over light positions.
    """
    # Deep copy to avoid mutating original
    import copy
    updated_dict = copy.deepcopy(base_scene_dict)

    # Update point light position
    updated_dict['point_light']['position'] = light_position.tolist()
    updated_dict['point_light']['intensity']['value'] = [
        light_intensity, light_intensity, light_intensity
    ]

    scene = mi.load_dict(updated_dict)
    return scene


def validate_light_trajectory(
    positions: np.ndarray,
    box_size: float = 1.0,
    min_height: float = 0.5
) -> Tuple[bool, str]:
    """Validate that light trajectory is physically reasonable.

    Args:
        positions: [N, 3] light positions to validate
        box_size: Cornell Box size (for boundary checking)
        min_height: Minimum allowed height above floor

    Returns:
        is_valid: True if trajectory is valid
        message: Explanation if invalid

    Checks:
        1. Light is inside or near box boundaries
        2. Light height is above floor
        3. No positions are NaN or inf
    """
    # Check for invalid values
    if np.any(~np.isfinite(positions)):
        return False, "Trajectory contains NaN or inf values"

    # Check height constraint
    min_z = np.min(positions[:, 2])
    if min_z < min_height:
        return False, f"Light too close to floor: min_z={min_z:.2f} < {min_height:.2f}"

    # Check box boundaries (with margin for external lighting)
    half_size = box_size / 2.0
    margin = 2.0  # Allow lights outside box for realistic scenarios

    x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
    y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])

    if x_min < -half_size - margin or x_max > half_size + margin:
        return False, f"X range [{x_min:.2f}, {x_max:.2f}] exceeds bounds"

    if y_min < -half_size - margin or y_max > half_size + margin:
        return False, f"Y range [{y_min:.2f}, {y_max:.2f}] exceeds bounds"

    return True, "Trajectory valid"


if __name__ == '__main__':
    """Test Cornell Box builder."""
    # Set Mitsuba variant
    mi.set_variant('cuda_ad_rgb')

    # Build scene with point light at center
    light_pos = np.array([0.0, 0.0, 1.5])
    scene = build_cornell_box_with_point_light(light_pos, light_intensity=50.0)

    print(f"Scene built successfully")
    print(f"Bounding box: {scene.bbox()}")

    # Generate circular trajectory
    trajectory = generate_circular_light_trajectory(
        center=(0.0, 0.0, 1.5),
        radius=1.2,
        num_positions=12
    )

    print(f"\nGenerated {len(trajectory)} light positions:")
    for i, pos in enumerate(trajectory):
        print(f"  Position {i:2d}: ({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f})")

    # Validate trajectory
    is_valid, msg = validate_light_trajectory(trajectory, box_size=1.0)
    print(f"\nTrajectory validation: {msg}")
