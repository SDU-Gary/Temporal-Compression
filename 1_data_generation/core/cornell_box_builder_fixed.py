"""Fixed Cornell Box scene builder with correct geometry.

Z-axis is vertical (up direction).
Box bounds: X ∈ [-0.5, 0.5], Y ∈ [-0.5, 0.5], Z ∈ [0, 1]
"""

import numpy as np
import mitsuba as mi
from typing import List


def build_cornell_box_with_area_light(
    intensity: float = 1.0,
    color: List[float] = [1.0, 1.0, 1.0],
    box_size: float = 1.0,
    light_scale: float = 0.3,
    radiance_scale: float = 50.0
) -> dict:
    """Build Cornell Box with area light on ceiling.

    Args:
        intensity: Light intensity multiplier [0, 1]
        color: RGB color of light
        box_size: Side length of Cornell Box (default 1.0m)
        light_scale: Light size relative to box (default 0.3)
        radiance_scale: Radiance multiplier in W/(sr·m²) (default 50.0)

    Returns:
        scene_dict: Scene dictionary for mi.load_dict()

    Notes:
        - Z-axis vertical: floor at Z=0, ceiling at Z=box_size
        - Walls at X,Y ∈ [-box_size/2, box_size/2]
        - Area light at ceiling center, facing downward
    """
    half = box_size / 2.0
    light_size = light_scale * box_size / 2.0  # Half size for transform

    scene_dict = {
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 8},

        # Area light on ceiling (emissive rectangle)
        'ceiling_light': {
            'type': 'rectangle',
            # Start at XY plane (Z=0), scale, flip normal to -Z, move to ceiling
            'to_world': mi.ScalarTransform4f()
                       .scale([light_size, light_size, 1])
                       .rotate([1, 0, 0], 180)  # Flip normal to face down
                       .translate([0, 0, box_size - 0.01]),  # Just below ceiling
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [radiance_scale * intensity * c for c in color]
                }
            }
        },

        # Floor (Z=0, normal +Z)
        'floor': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().scale([half, half, 1]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            }
        },

        # Ceiling (Z=box_size, normal -Z)
        'ceiling': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f()
                       .scale([half, half, 1])
                       .rotate([1, 0, 0], 180)  # Flip to face down
                       .translate([0, 0, box_size]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            }
        },

        # Back wall (Y=-half, normal +Y)
        'back_wall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f()
                       .scale([half, half, 1])
                       .rotate([1, 0, 0], -90)  # Rotate from XY to XZ plane
                       .translate([0, -half, half]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            }
        },

        # Front wall (Y=+half, normal -Y)
        'front_wall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f()
                       .scale([half, half, 1])
                       .rotate([1, 0, 0], 90)  # Rotate to XZ plane, face inward
                       .translate([0, half, half]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            }
        },

        # Left wall (X=-half, normal +X, RED)
        'left_wall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f()
                       .scale([half, half, 1])
                       .rotate([0, 1, 0], 90)  # Rotate from XY to YZ plane
                       .translate([-half, 0, half]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.63, 0.065, 0.05]}
            }
        },

        # Right wall (X=+half, normal -X, GREEN)
        'right_wall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f()
                       .scale([half, half, 1])
                       .rotate([0, 1, 0], -90)  # Rotate to YZ plane, face inward
                       .translate([half, 0, half]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.14, 0.45, 0.091]}
            }
        },

        # Small box (front-right)
        'small_box': {
            'type': 'cube',
            'to_world': mi.ScalarTransform4f()
                       .scale([0.15, 0.15, 0.15])
                       .rotate([0, 0, 1], -18)
                       .translate([0.2, 0.2, 0.15]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            }
        },

        # Tall box (back-left)
        'tall_box': {
            'type': 'cube',
            'to_world': mi.ScalarTransform4f()
                       .scale([0.15, 0.15, 0.3])
                       .rotate([0, 0, 1], 16)
                       .translate([-0.2, -0.2, 0.3]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            }
        }
    }

    return scene_dict


def build_cornell_box_with_point_light(
    light_position: np.ndarray,
    light_intensity: float = 50.0,
    box_size: float = 1.0
) -> 'mi.Scene':
    """Build Cornell Box with point light (kept for compatibility)."""
    half = box_size / 2.0

    scene_dict = {
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 8},

        # Point light
        'point_light': {
            'type': 'point',
            'position': light_position.tolist(),
            'intensity': {
                'type': 'rgb',
                'value': [light_intensity] * 3
            }
        },

        # Floor
        'floor': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().scale([half, half, 1]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            }
        },

        # Ceiling
        'ceiling': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f()
                       .scale([half, half, 1])
                       .rotate([1, 0, 0], 180)
                       .translate([0, 0, box_size]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            }
        },

        # Walls (same as area light version)
        'back_wall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f()
                       .scale([half, half, 1])
                       .rotate([1, 0, 0], -90)
                       .translate([0, -half, half]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            }
        },

        'front_wall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f()
                       .scale([half, half, 1])
                       .rotate([1, 0, 0], 90)
                       .translate([0, half, half]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            }
        },

        'left_wall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f()
                       .scale([half, half, 1])
                       .rotate([0, 1, 0], 90)
                       .translate([-half, 0, half]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.63, 0.065, 0.05]}
            }
        },

        'right_wall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f()
                       .scale([half, half, 1])
                       .rotate([0, 1, 0], -90)
                       .translate([half, 0, half]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.14, 0.45, 0.091]}
            }
        },

        'small_box': {
            'type': 'cube',
            'to_world': mi.ScalarTransform4f()
                       .scale([0.15, 0.15, 0.15])
                       .rotate([0, 0, 1], -18)
                       .translate([0.2, 0.2, 0.15]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            }
        },

        'tall_box': {
            'type': 'cube',
            'to_world': mi.ScalarTransform4f()
                       .scale([0.15, 0.15, 0.3])
                       .rotate([0, 0, 1], 16)
                       .translate([-0.2, -0.2, 0.3]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.725, 0.71, 0.68]}
            }
        }
    }

    scene = mi.load_dict(scene_dict)
    return scene


if __name__ == '__main__':
    """Test fixed Cornell Box."""
    mi.set_variant('cuda_ad_rgb')

    # Test area light version
    scene_dict = build_cornell_box_with_area_light(
        intensity=1.0,
        radiance_scale=50.0
    )

    # Add camera
    scene_dict['sensor'] = {
        'type': 'perspective',
        'fov': 45.0,
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0, -1.2, 0.5],
            target=[0, 0, 0.5],
            up=[0, 0, 1]
        ),
        'film': {'type': 'hdrfilm', 'width': 512, 'height': 512}
    }

    scene = mi.load_dict(scene_dict)
    print(f"Scene bbox: {scene.bbox()}")

    image = mi.render(scene, spp=128)
    image_np = np.array(image)

    print("=" * 60)
    print("FIXED CORNELL BOX TEST")
    print("=" * 60)
    print(f"Range: [{image_np.min():.6f}, {image_np.max():.6f}]")
    print(f"Mean: {image_np.mean():.6f}")

    if image_np.max() > 0.1:
        print("✅ SUCCESS!")
        mi.util.write_bitmap('/tmp/cornell_box_fixed.png', image)
        print("Saved: /tmp/cornell_box_fixed.png")
    else:
        print("❌ Still broken")
    print("=" * 60)
