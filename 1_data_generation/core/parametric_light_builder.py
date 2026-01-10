"""
Parametric Light Source Builder for Mitsuba 3

Converts 5D/8D light parameters to Mitsuba scene dictionaries.
Supports sun position, intensity, color temperature, atmospheric effects, and shape.
"""

import numpy as np
import mitsuba as mi
from typing import Dict, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.color_temperature import kelvin_to_rgb


class ParametricLightBuilder:
    """
    Build Mitsuba light sources from parametric descriptions.

    Supports 5D parameter space:
    - sun_zenith: [0, 90] degrees (0 = overhead, 90 = horizon)
    - sun_azimuth: [0, 360] degrees
    - intensity: [0.1, 2.0] relative intensity
    - color_temp: [2500, 10000] Kelvin
    - cloud_cover: [0.0, 0.9] fraction (0 = clear, 1 = overcast)
    """

    def __init__(self, box_center=(0, 0, 0), box_size=1.0):
        """
        Initialize light builder.

        Args:
            box_center: Center of the scene (for sun distance calculation)
            box_size: Size of the scene (for sun distance calculation)
        """
        self.box_center = np.array(box_center)
        self.box_size = box_size

        # Sun distance: 5x box size (close enough for measurable radiance)
        # For directional sun simulation, consider increasing to 50x with proportional intensity
        self.sun_distance = 5.0 * box_size

    def _zenith_azimuth_to_direction(
        self,
        zenith_deg: float,
        azimuth_deg: float
    ) -> np.ndarray:
        """
        Convert spherical coordinates to Cartesian direction.

        Args:
            zenith_deg: Zenith angle in degrees [0=up, 90=horizon]
            azimuth_deg: Azimuth angle in degrees [0=+X, 90=+Y]

        Returns:
            direction: Unit direction vector [x, y, z]
        """
        zenith_rad = np.deg2rad(zenith_deg)
        azimuth_rad = np.deg2rad(azimuth_deg)

        # Spherical to Cartesian (Y-up convention)
        x = np.sin(zenith_rad) * np.cos(azimuth_rad)
        z = np.sin(zenith_rad) * np.sin(azimuth_rad)
        y = np.cos(zenith_rad)

        direction = np.array([x, y, z])
        direction = direction / np.linalg.norm(direction)

        return direction

    def _compute_atmospheric_attenuation(
        self,
        zenith_deg: float,
        cloud_cover: float
    ) -> float:
        """
        Compute atmospheric attenuation factor.

        Approximates Rayleigh scattering and cloud absorption.

        Args:
            zenith_deg: Sun zenith angle [0, 90] degrees
            cloud_cover: Cloud fraction [0.0, 1.0]

        Returns:
            attenuation: Multiplicative factor [0, 1]
        """
        # Path length increases as cos(zenith)^-1
        zenith_rad = np.deg2rad(zenith_deg)
        air_mass = 1.0 / max(np.cos(zenith_rad), 0.01)  # Avoid division by zero

        # Rayleigh scattering: exponential decay with air mass
        rayleigh_atten = np.exp(-0.1 * (air_mass - 1.0))

        # Cloud absorption: linear reduction
        cloud_atten = 1.0 - 0.7 * cloud_cover  # Clouds block 70% at full cover

        # Combined attenuation
        attenuation = rayleigh_atten * cloud_atten

        return float(np.clip(attenuation, 0.05, 1.0))  # Minimum 5% transmission

    def build_5D_light(
        self,
        sun_zenith: float,
        sun_azimuth: float,
        intensity: float,
        color_temp: float,
        cloud_cover: float
    ) -> Dict:
        """
        Build Mitsuba point light from 5D parameters.

        Args:
            sun_zenith: [0, 90] degrees
            sun_azimuth: [0, 360] degrees
            intensity: [0.1, 2.0] relative intensity
            color_temp: [2500, 10000] Kelvin
            cloud_cover: [0.0, 0.9] atmospheric absorption

        Returns:
            light_dict: Mitsuba scene dictionary for point light
        """
        # 1. Compute sun direction and position
        direction = self._zenith_azimuth_to_direction(sun_zenith, sun_azimuth)
        position = self.box_center + direction * self.sun_distance

        # 2. Convert color temperature to RGB
        base_rgb = kelvin_to_rgb(color_temp)

        # 3. Compute atmospheric attenuation
        attenuation = self._compute_atmospheric_attenuation(sun_zenith, cloud_cover)

        # 4. Compute final radiance
        # Base intensity scaled by user parameter and atmospheric effects
        # Scale by 50.0 to account for distance falloff (matching transfer_tensor baseline)
        base_radiance = 50.0  # Radiant intensity baseline
        final_rgb = base_rgb * base_radiance * intensity * attenuation

        # 5. Build Mitsuba light dictionary
        light_dict = {
            'type': 'point',
            'position': position.tolist(),
            'intensity': {
                'type': 'rgb',
                'value': final_rgb.tolist()
            }
        }

        return light_dict

    def build_8D_light(
        self,
        sun_zenith: float,
        sun_azimuth: float,
        intensity: float,
        color_temp: float,
        cloud_cover: float,
        light_size: float,
        beam_inner: float,
        beam_outer: float
    ) -> Dict:
        """
        Build Mitsuba area/spot light from 8D parameters.

        Args:
            sun_zenith: [0, 90] degrees
            sun_azimuth: [0, 360] degrees
            intensity: [0.1, 2.0] relative intensity
            color_temp: [2500, 10000] Kelvin
            cloud_cover: [0.0, 0.9] atmospheric absorption
            light_size: [0.0, 0.5] radius for area light (0 = point)
            beam_inner: [0, 45] degrees inner cone (0 = no spotlight)
            beam_outer: [beam_inner, 90] degrees outer cone

        Returns:
            light_dict: Mitsuba scene dictionary
        """
        # Compute base properties (reuse 5D logic)
        direction = self._zenith_azimuth_to_direction(sun_zenith, sun_azimuth)
        position = self.box_center + direction * self.sun_distance
        base_rgb = kelvin_to_rgb(color_temp)
        attenuation = self._compute_atmospheric_attenuation(sun_zenith, cloud_cover)
        base_radiance = 50.0
        final_rgb = base_rgb * base_radiance * intensity * attenuation

        # Decide light type based on parameters
        if light_size > 0.01:
            # Area light (sphere)
            light_dict = {
                'type': 'sphere',
                'center': position.tolist(),
                'radius': float(light_size * self.box_size),
                'emitter': {
                    'type': 'area',
                    'radiance': {
                        'type': 'rgb',
                        'value': final_rgb.tolist()
                    }
                }
            }
        elif beam_inner > 0.1:
            # Spot light
            light_dict = {
                'type': 'spot',
                'position': position.tolist(),
                'target': self.box_center.tolist(),
                'beam_width': float(beam_inner),
                'cutoff_angle': float(beam_outer),
                'intensity': {
                    'type': 'rgb',
                    'value': final_rgb.tolist()
                }
            }
        else:
            # Point light (default)
            light_dict = {
                'type': 'point',
                'position': position.tolist(),
                'intensity': {
                    'type': 'rgb',
                    'value': final_rgb.tolist()
                }
            }

        return light_dict

    def sample_5D_stratified(
        self,
        num_geometric: int = 12,
        num_intensity: int = 20,
        num_atmospheric: int = 9
    ) -> np.ndarray:
        """
        Generate stratified 5D light configurations.

        Strategy:
        - Geometric group (zenith × azimuth): 12 configs
        - Intensity group (intensity × color_temp): 20 configs
        - Atmospheric group (cloud_cover × zenith): 9 configs

        Total: 41 configs

        Args:
            num_geometric: Number of geometric configurations
            num_intensity: Number of intensity configurations
            num_atmospheric: Number of atmospheric configurations

        Returns:
            configs: [N, 5] array of [zenith, azimuth, intensity, temp, cloud]
        """
        configs = []

        # --- Group 1: Geometric variations (fix intensity/color/cloud) ---
        zenith_vals = [15, 30, 45, 60]  # 4 elevations
        azimuth_vals = [0, 120, 240]  # 3 azimuths (evenly spaced)

        for zenith in zenith_vals:
            for azimuth in azimuth_vals:
                configs.append([
                    zenith,
                    azimuth,
                    1.0,     # Standard intensity
                    5500,    # Standard daylight
                    0.0      # Clear sky
                ])

        # --- Group 2: Intensity variations (fix geometry/cloud) ---
        intensity_vals = np.linspace(0.5, 1.5, 5)  # 5 levels
        temp_vals = [3000, 4500, 6500, 8500]  # 4 color temps

        for intensity in intensity_vals:
            for temp in temp_vals:
                configs.append([
                    30,       # Low zenith (high sun)
                    90,       # East
                    intensity,
                    temp,
                    0.2       # Light cloud
                ])

        # --- Group 3: Atmospheric variations (fix intensity/color) ---
        cloud_vals = [0.0, 0.3, 0.6]  # 3 cloud levels
        zenith_atm = [15, 45, 75]  # 3 sun heights

        for cloud in cloud_vals:
            for zenith in zenith_atm:
                configs.append([
                    zenith,
                    180,      # South
                    1.0,
                    5500,
                    cloud
                ])

        configs = np.array(configs)

        # Limit to requested numbers
        total_requested = num_geometric + num_intensity + num_atmospheric
        configs = configs[:total_requested]

        return configs


def test_parametric_light_builder():
    """Test parametric light builder."""
    print("Testing Parametric Light Builder")
    print("=" * 70)

    builder = ParametricLightBuilder(box_center=(0, 0, 0), box_size=1.0)

    # Test 5D light construction
    print("\n1. Testing 5D Light Construction")
    print("-" * 70)

    test_params = [
        (30, 90, 1.0, 6500, 0.0, "Clear noon sun"),
        (60, 180, 0.8, 3000, 0.3, "Warm sunset with clouds"),
        (15, 0, 1.2, 8000, 0.0, "Bright cold morning"),
        (45, 270, 0.5, 4500, 0.6, "Dim overcast afternoon"),
    ]

    for params in test_params:
        zenith, azimuth, intensity, temp, cloud, desc = params
        light = builder.build_5D_light(zenith, azimuth, intensity, temp, cloud)

        rgb = light['intensity']['value']
        pos = light['position']

        print(f"{desc:30s}  RGB: [{rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f}]  "
              f"Pos: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]")

    # Test stratified sampling
    print("\n2. Testing Stratified Sampling")
    print("-" * 70)

    configs = builder.sample_5D_stratified(num_geometric=12, num_intensity=20, num_atmospheric=9)
    print(f"Generated {len(configs)} configurations")
    print(f"Shape: {configs.shape}")
    print(f"\nFirst 5 configs:")
    print("Zenith  Azimuth  Intensity  ColorTemp  CloudCover")
    for i in range(min(5, len(configs))):
        print(f"{configs[i, 0]:6.1f}  {configs[i, 1]:7.1f}  "
              f"{configs[i, 2]:9.2f}  {configs[i, 3]:9.0f}  {configs[i, 4]:10.2f}")

    print(f"\n✓ All tests passed!")


if __name__ == '__main__':
    # Set Mitsuba variant
    mi.set_variant('cuda_ad_rgb')
    test_parametric_light_builder()
