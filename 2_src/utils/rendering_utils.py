"""Rendering utilities for SH coefficient visualization and quality assessment.

This module provides functions to:
1. Convert SH coefficients to environment maps
2. Render scenes using Mitsuba 3 with environment lighting
3. Compute image quality metrics (PSNR, SSIM)
4. Create error visualizations
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Tuple, Optional
import shutil

import numpy as np

# Import SH utilities from data_generation
_DATA_GEN_PATH = Path(__file__).parents[2] / "1_data_generation"
if str(_DATA_GEN_PATH) not in sys.path:
    sys.path.insert(0, str(_DATA_GEN_PATH))

from utils.spherical_harmonics import reconstruct_from_sh


def falcor_available(falcor_python_path: Optional[str] = None) -> bool:
    """Check if Falcor Python bindings with RenderGraph are available."""
    try:
        if falcor_python_path is None:
            falcor_python_path = '/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python'
        if falcor_python_path not in sys.path:
            sys.path.insert(0, falcor_python_path)
        import falcor as fc
        return all(hasattr(fc, name) for name in ("RenderGraph", "Testbed", "createPass", "float3"))
    except Exception:
        return False


def openexr_available() -> bool:
    """Check if OpenEXR Python bindings are available."""
    try:
        import OpenEXR  # noqa: F401
        import Imath  # noqa: F401
        return True
    except Exception:
        return False


def sh_to_envmap(sh_coeffs: np.ndarray, H: int = 128, W: int = 256) -> np.ndarray:
    """Convert SH coefficients to equirectangular environment map.

    Args:
        sh_coeffs: [27] SH coefficients (9 bases × 3 RGB channels)
        H: Environment map height
        W: Environment map width (typically 2*H for equirectangular)

    Returns:
        envmap: [H, W, 3] HDR environment map in range [0, +inf)
    """
    # Generate equirectangular sampling grid
    theta = np.linspace(0, np.pi, H)         # Elevation [0, π]
    phi = np.linspace(0, 2 * np.pi, W)        # Azimuth [0, 2π]

    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    # Convert to Cartesian coordinates
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.cos(theta_grid)
    z = np.sin(theta_grid) * np.sin(phi_grid)

    # Direction vectors [H*W, 3]
    directions = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)

    # Reconstruct radiance from SH [H*W, 3]
    radiances = reconstruct_from_sh(sh_coeffs, directions, max_order=2)

    # Reshape to environment map [H, W, 3]
    envmap = radiances.reshape(H, W, 3)

    # Clamp negative values to zero
    envmap = np.maximum(envmap, 0.0)

    return envmap


def save_exr(filepath: str, image: np.ndarray):
    """Save HDR image in OpenEXR format.

    Args:
        filepath: Output path (should end with .exr)
        image: [H, W, 3] HDR image array
    """
    try:
        import OpenEXR
        import Imath

        H, W, C = image.shape
        assert C == 3, "Image must have 3 channels (RGB)"

        # Convert to float32
        image = image.astype(np.float32)

        # Create header
        header = OpenEXR.Header(W, H)
        header['channels'] = {
            'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        }

        # Create output file
        exr = OpenEXR.OutputFile(filepath, header)

        # Write channels
        r = image[:, :, 0].tobytes()
        g = image[:, :, 1].tobytes()
        b = image[:, :, 2].tobytes()

        exr.writePixels({'R': r, 'G': g, 'B': b})
        exr.close()

    except ImportError:
        # Fallback: save as numpy array if OpenEXR not available
        print(f"Warning: OpenEXR not available, saving as .npy instead")
        np.save(filepath.replace('.exr', '.npy'), image)


def _encode_rgbe(image: np.ndarray) -> np.ndarray:
    """Convert float RGB image to RGBE encoding."""
    img = np.maximum(image, 0.0).astype(np.float32)
    max_rgb = np.max(img, axis=2)
    rgbe = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)

    nonzero = max_rgb > 1e-32
    if np.any(nonzero):
        mantissa, exp = np.frexp(max_rgb[nonzero])
        scale = mantissa * 256.0 / max_rgb[nonzero]
        rgb_scaled = img[nonzero] * scale[:, None]
        rgbe[nonzero, 0:3] = np.clip(rgb_scaled, 0, 255).astype(np.uint8)
        rgbe[nonzero, 3] = np.clip(exp + 128, 0, 255).astype(np.uint8)

    return rgbe


def _rle_encode_scanline(scanline: np.ndarray) -> bytes:
    """Run-length encode one scanline (width x 4 RGBE)."""
    width = scanline.shape[0]
    out = bytearray()
    out.extend([2, 2, (width >> 8) & 0xFF, width & 0xFF])

    for chan in range(4):
        data = scanline[:, chan]
        i = 0
        while i < width:
            run_len = 1
            while i + run_len < width and run_len < 127 and data[i] == data[i + run_len]:
                run_len += 1
            if run_len >= 4:
                out.append(128 + run_len)
                out.append(int(data[i]))
                i += run_len
                continue

            start = i
            i += run_len
            while i < width:
                run_len = 1
                while i + run_len < width and run_len < 127 and data[i] == data[i + run_len]:
                    run_len += 1
                if run_len >= 4 or (i - start) >= 127:
                    break
                i += run_len

            length = i - start
            out.append(length)
            out.extend(data[start:i].tolist())

    return bytes(out)


def save_hdr(filepath: str, image: np.ndarray):
    """Save HDR image in Radiance .hdr (RGBE) format.

    By default, write the legacy non-RLE scanline format for maximum loader compatibility.
    Set env var FALCOR_HDR_RLE=1 to enable RLE encoding.
    """
    H, W, C = image.shape
    assert C == 3, "Image must have 3 channels (RGB)"
    rgbe = _encode_rgbe(image)

    header = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {H} +X {W}\n"
    use_rle = os.environ.get("FALCOR_HDR_RLE", "0") == "1"
    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        if use_rle:
            for y in range(H):
                f.write(_rle_encode_scanline(rgbe[y]))
        else:
            # Legacy non-RLE format: raw RGBE pixels in scanline order.
            f.write(rgbe.tobytes())


def compute_luminance(image: np.ndarray) -> np.ndarray:
    """Compute luminance from RGB image (linear space)."""
    return (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    )


def compute_auto_exposure(
    image: np.ndarray,
    percentile: float = 95.0,
    target: float = 0.6,
    eps: float = 1e-6,
    min_exposure: float = 0.05,
    max_exposure: float = 50.0,
) -> float:
    """Compute exposure scale so a luminance percentile maps to target."""
    lum = compute_luminance(np.maximum(image, 0.0))
    key = float(np.percentile(lum, percentile))
    exposure = float(target / (key + eps))
    return float(np.clip(exposure, min_exposure, max_exposure))


def tone_map_reinhard(image: np.ndarray) -> np.ndarray:
    """Apply Reinhard tone mapping (expects linear HDR)."""
    img = np.maximum(image, 0.0)
    return img / (1.0 + img)


def srgb_encode(image: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB."""
    img = np.maximum(image, 0.0)
    threshold = 0.0031308
    low = 12.92 * img
    high = 1.055 * np.power(img, 1.0 / 2.4) - 0.055
    return np.where(img <= threshold, low, high)


def render_with_envmap(
    envmap_path: str,
    scene_path: Optional[str] = None,
    output_path: Optional[str] = None,
    spp: int = 32,
    resolution: Tuple[int, int] = (256, 256),
    falcor_python_path: Optional[str] = None,
    clamp_output: bool = True,
    output_pass: str = "ToneMapper.dst",
    enable_tonemapper: bool = True,
) -> np.ndarray:
    """Render scene using Falcor with environment map lighting.

    Args:
        envmap_path: Path to environment map (.exr file)
        scene_path: Optional path to Falcor scene (.pyscene file).
                   If None, uses a simple default scene with Cornell Box.
        output_path: Optional path to save rendered image
        spp: Samples per pixel
        resolution: (width, height) of output image
        falcor_python_path: Optional path to Falcor Python bindings
                           (default: /home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python)

    Returns:
        rendered_image: [H, W, 3] rendered image in range [0, 1]

    Raises:
        RuntimeError: If Falcor rendering fails
    """
    try:
        import sys

        # Add Falcor Python path
        if falcor_python_path is None:
            falcor_python_path = '/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python'

        if falcor_python_path not in sys.path:
            sys.path.insert(0, falcor_python_path)

        import falcor as fc

        # Optional device selection via env vars
        device_type_name = os.environ.get("FALCOR_DEVICE_TYPE", "Default")
        device_type = getattr(fc.DeviceType, device_type_name, fc.DeviceType.Default)
        try:
            gpu_index = int(os.environ.get("FALCOR_GPU", "0"))
        except ValueError:
            gpu_index = 0

        # Create testbed and render graph
        testbed = fc.Testbed(
            width=resolution[0],
            height=resolution[1],
            create_window=False,
            device_type=device_type,
            gpu=gpu_index,
        )

        render_graph = testbed.create_render_graph("EnvMapRenderer")

        # Passes (PathTracer pipeline)
        try:
            render_graph.create_pass("PathTracer", "PathTracer", {
                'samplesPerPixel': 1,
                'maxSurfaceBounces': 4,
                'useNEE': True
            })
            render_graph.create_pass("VBufferRT", "VBufferRT", {
                'samplePattern': 'Stratified',
                'sampleCount': 16
            })
            render_graph.create_pass("AccumulatePass", "AccumulatePass", {
                'enabled': True,
                'precisionMode': 'Single',
                'autoReset': False,
                'maxFrameCount': int(spp),
                'overflowMode': 'Stop'
            })
            if enable_tonemapper:
                render_graph.create_pass("ToneMapper", "ToneMapper", {
                    'autoExposure': False,
                    'exposureCompensation': 0.0
                })
        except Exception as e:
            raise RuntimeError(
                f"{e} (Falcor PathTracer/VBufferRT require SM6.5-capable GPU; "
                "if you see llvmpipe, Vulkan is running in software mode.)"
            )

        # Connect passes
        render_graph.add_edge("VBufferRT.vbuffer", "PathTracer.vbuffer")
        render_graph.add_edge("PathTracer.color", "AccumulatePass.input")
        if enable_tonemapper:
            render_graph.add_edge("AccumulatePass.output", "ToneMapper.src")

        render_graph.mark_output(output_pass)
        testbed.render_graph = render_graph

        # Load scene
        if scene_path and Path(scene_path).exists():
            testbed.load_scene(scene_path)
        else:
            # Fall back to a built-in Falcor test scene if available.
            repo_root = Path(__file__).parents[2]
            candidates = []
            env_scene = os.environ.get("FALCOR_DEFAULT_SCENE")
            if env_scene:
                candidates.append(Path(env_scene))
            candidates.extend([
                repo_root / "1_data_generation" / "falcor" / "scenes" / "cornell_box_sun_point.pyscene",
                repo_root / "1_data_generation" / "falcor" / "scenes" / "cornell_box_sun.pyscene",
                repo_root / "Falcor" / "tests" / "image_tests" / "scene" / "scenes" / "SDFSVS.pyscene",
                repo_root / "Falcor" / "tests" / "image_tests" / "scene" / "scenes" / "SDFSBS.pyscene",
                repo_root / "Falcor" / "tests" / "image_tests" / "scene" / "scenes" / "SDFSVO.pyscene",
                repo_root / "Falcor" / "tests" / "image_tests" / "scene" / "scenes" / "NDSDFGrid.pyscene",
                repo_root / "Falcor" / "scripts" / "sdf-editor" / "SDFEditorStartScene.pyscene",
            ])
            default_scene = next((p for p in candidates if p and p.exists()), None)
            if default_scene is None:
                raise RuntimeError(
                    "No scene_path provided and no default Falcor .pyscene found. "
                    "Set FALCOR_DEFAULT_SCENE to a valid .pyscene path."
                )
            testbed.load_scene(str(default_scene))

        # Set environment map
        if not testbed.scene.setEnvMap(envmap_path):
            raise RuntimeError(f"Failed to set envmap: {envmap_path}")
        if hasattr(testbed.scene, "envMap") and testbed.scene.envMap is not None:
            testbed.scene.envMap.intensity = 1.0

        # Enable environment lighting
        testbed.scene.renderSettings.useEnvLight = True
        testbed.scene.renderSettings.useAnalyticLights = False
        testbed.scene.renderSettings.useEmissiveLights = False

        # Set resolution
        testbed.resize_frame_buffer(resolution[0], resolution[1])

        # Render frames for accumulation
        testbed.clock.pause()
        for _ in range(max(spp, 1)):
            testbed.frame()

        # Get output image
        output_texture = testbed.render_graph.get_output(output_pass)
        if output_texture is None:
            raise RuntimeError("Failed to get output texture from render graph")

        # Convert to numpy array
        image_np = output_texture.to_numpy()
        if image_np.ndim == 1:
            h, w = resolution[1], resolution[0]
            pixel_count = h * w
            if image_np.size % pixel_count != 0:
                raise RuntimeError(
                    f"Unexpected output texture size {image_np.size} for resolution {w}x{h}"
                )
            channels = max(1, image_np.size // pixel_count)
            image_np = image_np.reshape(h, w, channels)
        elif image_np.ndim == 2:
            image_np = image_np[:, :, None]
        if image_np.shape[2] > 3:
            image_np = image_np[:, :, :3]

        # Optionally clamp to [0, 1] for LDR output
        if os.environ.get("RERUN_KEEP_HDR", "0") == "1":
            clamp_output = False
        if clamp_output:
            image_np = np.clip(image_np, 0.0, 1.0)

        # Save if requested
        if output_path:
            from PIL import Image
            image_uint8 = (np.clip(image_np, 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(image_uint8).save(output_path)

        return image_np

    except Exception as e:
        raise RuntimeError(f"Falcor rendering failed: {e}")


def get_falcor_python_bin() -> Optional[str]:
    """Return a Python 3.10 interpreter suitable for Falcor bindings if available."""
    candidates = []
    env_bin = os.environ.get("FALCOR_PYTHON_BIN")
    if env_bin:
        candidates.append(env_bin)

    repo_root = Path(__file__).parents[2]
    candidates.append(str(repo_root / "Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10"))
    candidates.append(str(Path.home() / ".pyenv/versions/3.10.12/bin/python"))

    for path in candidates:
        if path and Path(path).exists():
            return path
    return None


def render_with_envmap_external(
    envmap_path: str,
    scene_path: Optional[str] = None,
    spp: int = 32,
    resolution: Tuple[int, int] = (256, 256),
    falcor_python_path: Optional[str] = None,
    python_bin: Optional[str] = None,
    output_pass: str = "ToneMapper.dst",
    enable_tonemapper: bool = True,
) -> np.ndarray:
    """Render using Falcor in a separate Python 3.10 process."""
    import subprocess
    import tempfile

    if falcor_python_path is None:
        falcor_python_path = '/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python'

    if python_bin is None:
        python_bin = get_falcor_python_bin()
    if python_bin is None:
        raise RuntimeError("No suitable Python 3.10 interpreter found for Falcor")

    script_path = Path(__file__).parent / "falcor_render_cli.py"
    if not script_path.exists():
        raise RuntimeError(f"Missing helper script: {script_path}")

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        out_path = tmp.name

    cmd = [
        python_bin,
        str(script_path),
        "--envmap", envmap_path,
        "--out", out_path,
        "--spp", str(spp),
        "--width", str(resolution[0]),
        "--height", str(resolution[1]),
        "--falcor-python-path", falcor_python_path,
        "--output-pass", output_pass,
    ]
    if not enable_tonemapper:
        cmd.append("--no-tonemap")
    if scene_path:
        cmd += ["--scene", scene_path]

    env = os.environ.copy()
    repo_root = Path(__file__).parents[2]
    src_root = repo_root / "2_src"
    env["PYTHONPATH"] = ":".join([str(falcor_python_path), str(src_root), str(repo_root)])

    falcor_bin = Path(falcor_python_path).parent
    env["LD_LIBRARY_PATH"] = f"{falcor_bin}:{env.get('LD_LIBRARY_PATH', '')}"
    vk_icd = os.environ.get("FALCOR_VK_ICD")
    if vk_icd:
        env["VK_ICD_FILENAMES"] = vk_icd
    env["PYTHONNOUSERSITE"] = "1"

    subprocess.run(cmd, check=True, env=env)

    image = np.load(out_path)
    try:
        os.unlink(out_path)
    except Exception:
        pass

    return image


def compute_image_metrics(img_gt: np.ndarray, img_pred: np.ndarray) -> dict[str, float]:
    """Compute PSNR and SSIM between two images.

    Args:
        img_gt: [H, W, 3] ground truth image in range [0, 1]
        img_pred: [H, W, 3] predicted image in range [0, 1]

    Returns:
        Dictionary with 'psnr' and 'ssim' keys
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    # Ensure images are in correct range
    img_gt = np.clip(img_gt, 0.0, 1.0)
    img_pred = np.clip(img_pred, 0.0, 1.0)

    # Compute PSNR
    psnr = peak_signal_noise_ratio(img_gt, img_pred, data_range=1.0)

    # Compute SSIM
    ssim = structural_similarity(
        img_gt, img_pred,
        channel_axis=2,
        data_range=1.0
    )

    return {'psnr': float(psnr), 'ssim': float(ssim)}


def create_error_heatmap(img_gt: np.ndarray, img_pred: np.ndarray) -> np.ndarray:
    """Create error heatmap visualization (blue=low error, red=high error).

    Args:
        img_gt: [H, W, 3] ground truth image
        img_pred: [H, W, 3] predicted image

    Returns:
        heatmap: [H, W, 3] error heatmap in range [0, 1]
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # Compute per-pixel L2 error
    error = np.sqrt(np.sum((img_gt - img_pred) ** 2, axis=-1))  # [H, W]

    # Normalize to [0, 1]
    error_norm = Normalize(vmin=0, vmax=np.percentile(error, 99))(error)

    # Apply colormap (blue to red)
    cmap = plt.cm.get_cmap('jet')  # Or 'turbo' for better perceptual uniformity
    heatmap = cmap(error_norm)[:, :, :3]  # [H, W, 3] (discard alpha)

    return heatmap.astype(np.float32)


def cleanup_temp_files(temp_dir: str):
    """Clean up temporary files in directory.

    Args:
        temp_dir: Directory containing temporary files
    """
    temp_path = Path(temp_dir)
    if temp_path.exists() and temp_path.is_dir():
        # Remove all .exr and .npy files
        for ext in ['*.exr', '*.npy', '*.png', '*.jpg']:
            for file in temp_path.glob(ext):
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete {file}: {e}")
