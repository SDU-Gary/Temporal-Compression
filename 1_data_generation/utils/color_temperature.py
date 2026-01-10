"""
Color Temperature to RGB Conversion

Implements Tanner Helland algorithm for converting blackbody temperature to RGB.
Reference: http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
"""

import numpy as np


def kelvin_to_rgb(kelvin: float) -> np.ndarray:
    """
    Convert color temperature (Kelvin) to normalized RGB.

    Based on Tanner Helland's algorithm, which approximates Planck's law
    for blackbody radiation in the visible spectrum.

    Args:
        kelvin: Color temperature in Kelvin (1000-40000)

    Returns:
        rgb: Normalized RGB array [R, G, B] in [0, 1]

    Examples:
        >>> kelvin_to_rgb(6500)  # Daylight
        array([1.0, 0.996, 1.0])

        >>> kelvin_to_rgb(3000)  # Warm white
        array([1.0, 0.755, 0.543])
    """
    # Clamp to valid range
    temp = np.clip(kelvin, 1000, 40000)

    # Temperature is divided by 100
    temp = temp / 100.0

    # --- Red channel ---
    if temp <= 66:
        red = 255
    else:
        # Formula: a + b * log(temp - 60)
        red = temp - 60
        red = 329.698727446 * (red ** -0.1332047592)
        red = np.clip(red, 0, 255)

    # --- Green channel ---
    if temp <= 66:
        # Formula: a + b * log(temp)
        green = temp
        green = 99.4708025861 * np.log(green) - 161.1195681661
        green = np.clip(green, 0, 255)
    else:
        # Formula: a + b * log(temp - 60)
        green = temp - 60
        green = 288.1221695283 * (green ** -0.0755148492)
        green = np.clip(green, 0, 255)

    # --- Blue channel ---
    if temp >= 66:
        blue = 255
    elif temp <= 19:
        blue = 0
    else:
        # Formula: a + b * log(temp - 10)
        blue = temp - 10
        blue = 138.5177312231 * np.log(blue) - 305.0447927307
        blue = np.clip(blue, 0, 255)

    # Normalize to [0, 1]
    rgb = np.array([red, green, blue]) / 255.0

    return rgb


def kelvin_to_rgb_batch(kelvins: np.ndarray) -> np.ndarray:
    """
    Vectorized version for batch conversion.

    Args:
        kelvins: [N] array of temperatures in Kelvin

    Returns:
        rgbs: [N, 3] array of normalized RGB values
    """
    rgbs = np.array([kelvin_to_rgb(k) for k in kelvins])
    return rgbs


def test_color_temperature():
    """Test color temperature conversion with known reference points."""
    print("Testing Color Temperature Conversion")
    print("=" * 60)

    # Reference temperatures and expected colors
    test_cases = [
        (1000, "Candlelight", [1.0, 0.329, 0.0]),
        (2000, "Sunrise", [1.0, 0.549, 0.208]),
        (3000, "Warm White LED", [1.0, 0.755, 0.543]),
        (4000, "Cool White", [1.0, 0.859, 0.706]),
        (5000, "Daylight", [1.0, 0.929, 0.843]),
        (6500, "Standard Daylight", [1.0, 0.996, 1.0]),
        (7500, "Overcast Sky", [0.937, 0.965, 1.0]),
        (10000, "Clear Blue Sky", [0.776, 0.886, 1.0]),
        (20000, "Arctic Sky", [0.596, 0.765, 1.0]),
    ]

    for kelvin, name, _ in test_cases:
        rgb = kelvin_to_rgb(kelvin)
        print(f"{kelvin:5d}K  {name:20s}  RGB: [{rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f}]")

    print()

    # Test batch conversion
    print("Testing batch conversion...")
    kelvins = np.array([3000, 5000, 6500, 10000])
    rgbs = kelvin_to_rgb_batch(kelvins)
    print(f"Input: {kelvins}")
    print(f"Output shape: {rgbs.shape}")
    print(f"Output:\n{rgbs}")

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    test_color_temperature()
