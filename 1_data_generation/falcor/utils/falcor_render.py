"""Falcor rendering helpers for probe sampling.

This module wraps Falcor Python bindings to:
1) Create a headless Testbed + PathTracer render graph
2) Load a .pyscene scene
3) Update camera + light parameters
4) Read back the PathTracer output as numpy
"""

from __future__ import annotations

import os
import sys
import ctypes
import glob
import importlib.machinery
import importlib.util
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

_MOGWAI_RENDERER = None
_CUBEMAP_CACHE = {}


def set_mogwai_renderer(renderer) -> None:
    """Register Mogwai renderer so helper functions can reuse it."""
    global _MOGWAI_RENDERER
    _MOGWAI_RENDERER = renderer


def _get_mogwai_renderer():
    """Return Mogwai renderer if running inside Mogwai, else None."""
    if _MOGWAI_RENDERER is not None:
        return _MOGWAI_RENDERER
    try:
        import __main__
        renderer = getattr(__main__, "m", None)
        if renderer is not None:
            return renderer
    except Exception:
        pass
    try:
        import builtins
        return getattr(builtins, "m", None)
    except Exception:
        return None


def _ensure_falcor_import(falcor_python_path: Optional[str] = None):
    """Import falcor with optional python path injection."""
    if falcor_python_path:
        if falcor_python_path not in sys.path:
            sys.path.insert(0, falcor_python_path)
    try:
        import falcor  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Failed to import falcor. Make sure Falcor is built and "
            "its python path is on sys.path (see Falcor/docs/falcor-in-python.md)."
        ) from exc
    falcor_mod = sys.modules["falcor"]
    if hasattr(falcor_mod, "Testbed"):
        return falcor_mod

    # Fallback: load extension module directly and inject symbols.
    if not falcor_python_path:
        raise RuntimeError("Falcor import lacks Testbed and falcor_python_path was not provided.")

    ext_candidates = glob.glob(os.path.join(str(falcor_python_path), "falcor", "falcor_ext*.so"))
    if not ext_candidates:
        raise RuntimeError("Falcor extension .so not found under falcor_python_path.")

    ext_path = ext_candidates[0]
    lib_dir = str(Path(falcor_python_path).parent)
    lib_path = os.path.join(lib_dir, "libFalcor.so")
    if os.path.exists(lib_path):
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    # Load extension with its canonical module name so PyInit_falcor_ext resolves.
    module_name = "falcor.falcor_ext"
    if "falcor.falcor_ext" in sys.modules:
        del sys.modules["falcor.falcor_ext"]
    loader = importlib.machinery.ExtensionFileLoader(module_name, ext_path)
    spec = importlib.util.spec_from_file_location(module_name, ext_path, loader=loader)
    if spec is None:
        raise RuntimeError("Failed to create spec for Falcor extension module.")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules[module_name] = module

    for name in dir(module):
        if not name.startswith("_"):
            setattr(falcor_mod, name, getattr(module, name))

    if not hasattr(falcor_mod, "Testbed"):
        raise RuntimeError("Falcor extension loaded but Testbed is still missing.")

    return falcor_mod


def build_testbed(
    width: int = 1,
    height: int = 1,
    spp: int = 64,
    falcor_python_path: Optional[str] = None,
    fixed_seed: Optional[int] = None,
    use_russian_roulette: Optional[bool] = None,
):
    """Create a headless Testbed with a PathTracer graph.

    Returns:
        testbed, graph, falcor module
    """
    falcor = _ensure_falcor_import(falcor_python_path)

    renderer = _get_mogwai_renderer()
    if renderer is not None:
        # Running inside Mogwai; use renderer as testbed.
        testbed = renderer
    else:
        testbed = falcor.Testbed(width=width, height=height, create_window=False)

    # Prefer Testbed.create_render_graph() if available (works outside Mogwai).
    graph = None
    try:
        if hasattr(testbed, "create_render_graph"):
            graph = testbed.create_render_graph("ProbePathTracer")
        elif hasattr(testbed, "createRenderGraph"):
            graph = testbed.createRenderGraph("ProbePathTracer")
    except Exception:
        graph = None

    if graph is None:
        # Fallback to legacy RenderGraph API (requires Mogwai context).
        graph = falcor.RenderGraph("ProbePathTracer")

    pass_options = {
        "samplesPerPixel": int(spp),
        "maxSurfaceBounces": 4,
        "useNEE": True,
    }
    if fixed_seed is not None:
        pass_options["fixedSeed"] = int(fixed_seed)
    if use_russian_roulette is not None:
        pass_options["useRussianRoulette"] = bool(use_russian_roulette)

    def _graph_create_supported(graph_obj) -> bool:
        return hasattr(graph_obj, "create_pass") or hasattr(graph_obj, "createPass")

    def _create_pass(graph_obj, name, pass_type, options):
        if hasattr(graph_obj, "create_pass"):
            return graph_obj.create_pass(name, pass_type, options)
        if hasattr(graph_obj, "createPass"):
            return graph_obj.createPass(name, pass_type, options)
        # Legacy API
        return falcor.createPass(pass_type, options)

    def _add_pass(graph_obj, pass_obj, name):
        if hasattr(graph_obj, "add_pass"):
            graph_obj.add_pass(pass_obj, name)
            return
        if hasattr(graph_obj, "addPass"):
            graph_obj.addPass(pass_obj, name)
            return

    def _add_edge(graph_obj, src, dst):
        if hasattr(graph_obj, "add_edge"):
            graph_obj.add_edge(src, dst)
            return
        if hasattr(graph_obj, "addEdge"):
            graph_obj.addEdge(src, dst)
            return

    def _mark_output(graph_obj, name):
        if hasattr(graph_obj, "mark_output"):
            graph_obj.mark_output(name)
            return
        if hasattr(graph_obj, "markOutput"):
            graph_obj.markOutput(name)
            return

    path_tracer = _create_pass(graph, "PathTracer", "PathTracer", pass_options)
    # Use a deterministic VBuffer to avoid per-frame subpixel jitter in SH baking.
    vbuffer = _create_pass(graph, "VBufferRT", "VBufferRT", {
        "samplePattern": "Center",
        "sampleCount": 1,
        "useAlphaTest": True,
    })

    if not _graph_create_supported(graph):
        _add_pass(graph, path_tracer, "PathTracer")
        _add_pass(graph, vbuffer, "VBufferRT")
    _add_edge(graph, "VBufferRT.vbuffer", "PathTracer.vbuffer")
    _add_edge(graph, "VBufferRT.viewW", "PathTracer.viewW")
    _add_edge(graph, "VBufferRT.mvec", "PathTracer.mvec")
    _mark_output(graph, "PathTracer.color")

    if renderer is not None:
        try:
            testbed.addGraph(graph)
        except Exception:
            try:
                testbed.add_graph(graph)
            except Exception:
                pass

        try:
            testbed.setActiveGraph(graph)
        except Exception:
            pass

        try:
            active_graph = testbed.activeGraph
            if active_graph is not None:
                graph = active_graph
        except Exception:
            pass

        try:
            testbed.resizeFrameBuffer(width, height)
        except Exception:
            try:
                testbed.resize_frame_buffer(width, height)
            except Exception:
                pass
    else:
        # Falcor Testbed path (outside Mogwai)
        try:
            testbed.render_graph = graph
        except Exception:
            try:
                testbed.setRenderGraph(graph)
            except Exception:
                pass
        try:
            testbed.resize_frame_buffer(width, height)
        except Exception:
            try:
                testbed.resizeFrameBuffer(width, height)
            except Exception:
                pass

    # Stabilize time for deterministic renders
    try:
        testbed.clock.pause()
    except Exception:
        pass

    return testbed, graph, falcor


def load_scene(testbed, scene_path: str | Path):
    """Load a .pyscene file into the testbed and return the scene."""
    scene_path = str(scene_path)
    try:
        testbed.load_scene(scene_path)
    except Exception:
        if hasattr(testbed, "loadScene"):
            testbed.loadScene(scene_path)
        else:
            raise
    return testbed.scene


def configure_scene_for_sun(scene):
    """Disable emissive/env lighting; keep analytic lights."""
    scene.renderSettings.useEnvLight = False
    scene.renderSettings.useEmissiveLights = False
    scene.renderSettings.useAnalyticLights = True


def get_light(scene, name: str = "Sun"):
    """Get a light by name, fallback to index 0."""
    light = None
    try:
        light = scene.getLight(name)
    except Exception:
        pass
    if light is None:
        light = scene.getLight(0)
    return light


def set_camera_look(scene, position: np.ndarray, direction: np.ndarray):
    """Set camera position and orientation using a look direction."""
    cam = scene.camera
    pos = position.astype(np.float32)
    dir_norm = direction / (np.linalg.norm(direction) + 1e-8)

    # Choose an up vector that is not parallel to direction
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(np.dot(dir_norm, up)) > 0.99:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    target = pos + dir_norm
    cam.position = falcor_float3(pos)
    cam.target = falcor_float3(target)
    cam.up = falcor_float3(up)


def set_camera_orientation(scene, position: np.ndarray, forward: np.ndarray, up: np.ndarray):
    """Set camera with explicit forward/up vectors."""
    cam = scene.camera
    pos = position.astype(np.float32)
    fwd = forward.astype(np.float32)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    up_vec = up.astype(np.float32)
    up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-8)
    cam.position = falcor_float3(pos)
    cam.target = falcor_float3(pos + fwd)
    cam.up = falcor_float3(up_vec)


def configure_cubemap_camera(scene, near_plane: float = 0.001, far_plane: float = 10.0):
    """Configure camera for 90-degree FOV cubemap faces."""
    cam = scene.camera
    cam.aspectRatio = 1.0
    cam.frameHeight = 24.0
    cam.frameWidth = 24.0
    cam.focalLength = 12.0  # 90 deg FOV with frameHeight=24
    cam.nearPlane = float(near_plane)
    cam.farPlane = float(far_plane)


def falcor_float3(vec: np.ndarray):
    """Create a Falcor float3 from numpy array."""
    import falcor
    return falcor.float3(float(vec[0]), float(vec[1]), float(vec[2]))


def render_single_pixel(
    testbed,
    graph,
    num_frames: int = 4,
    seed_base: Optional[int] = None,
    radiance_clamp: Optional[float] = None,
) -> np.ndarray:
    """Render one pixel (averaged over frames) and return RGB from PathTracer output."""
    graph_obj = None
    for candidate in (graph, getattr(testbed, "activeGraph", None), getattr(testbed, "render_graph", None)):
        if candidate is not None:
            graph_obj = candidate
            break
    if graph_obj is None:
        raise RuntimeError("No render graph available for output readback.")

    path_tracer = None
    try:
        if hasattr(graph_obj, "get_pass"):
            path_tracer = graph_obj.get_pass("PathTracer")
        else:
            path_tracer = graph_obj.getPass("PathTracer")
    except Exception:
        path_tracer = None

    _has_reset = hasattr(path_tracer, "reset") if path_tracer is not None else False

    def _set_fixed_seed(seed_value: int) -> None:
        if path_tracer is None:
            return
        try:
            path_tracer.set_properties({"fixedSeed": int(seed_value)})
        except Exception:
            try:
                path_tracer.setProperties({"fixedSeed": int(seed_value)})
            except Exception:
                pass

    def _render_frame() -> np.ndarray:
        if _has_reset:
            try:
                path_tracer.reset()
            except Exception:
                pass
        rendered = False
        if hasattr(testbed, "frame"):
            try:
                testbed.frame()
                rendered = True
            except Exception:
                rendered = False
        if not rendered and hasattr(testbed, "renderFrame"):
            try:
                testbed.renderFrame()
                rendered = True
            except Exception:
                rendered = False
        if not rendered:
            try:
                import falcor
                falcor.renderFrame()
                rendered = True
            except Exception:
                rendered = False
        if not rendered:
            raise RuntimeError("No renderFrame method available on testbed or falcor module.")

        output = None
        try:
            if hasattr(graph_obj, "get_output"):
                output = graph_obj.get_output("PathTracer.color")
            else:
                output = graph_obj.getOutput("PathTracer.color")
        except Exception:
            output = None

        if output is None:
            raise RuntimeError("Failed to get PathTracer.color output texture")
        if hasattr(output, "to_numpy"):
            img = output.to_numpy()
        elif hasattr(output, "toNumpy"):
            img = output.toNumpy()
        else:
            raise RuntimeError("Output texture does not support to_numpy()")
        img = np.asarray(img, dtype=np.float32).reshape(-1)
        if img.size < 3:
            raise RuntimeError("Unexpected output texture size")
        rgb = img[:3].copy()
        return _apply_radiance_clamp(rgb, radiance_clamp)

    if num_frames <= 1:
        if seed_base is not None:
            _set_fixed_seed(seed_base)
        return _render_frame()

    accum = np.zeros(3, dtype=np.float32)
    for i in range(num_frames):
        if seed_base is not None:
            _set_fixed_seed(seed_base + i)
        accum += _render_frame()
    return accum / float(num_frames)


def _cubemap_face_bases() -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (forward, up) for cubemap faces."""
    return [
        (np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)),   # +X
        (np.array([-1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)),  # -X
        (np.array([0.0, 1.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32)),  # +Y
        (np.array([0.0, -1.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)),  # -Y
        (np.array([0.0, 0.0, 1.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)),   # +Z
        (np.array([0.0, 0.0, -1.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)),  # -Z
    ]


def _get_cubemap_cache(cube_res: int):
    if cube_res in _CUBEMAP_CACHE:
        return _CUBEMAP_CACHE[cube_res]
    from spherical_harmonics import evaluate_sh_basis

    coords = (np.arange(cube_res, dtype=np.float32) + 0.5) / cube_res
    u = 2.0 * coords - 1.0
    v = 1.0 - 2.0 * coords  # top -> +1
    uu, vv = np.meshgrid(u, v)
    weight = (4.0 / (cube_res * cube_res)) / np.power(1.0 + uu * uu + vv * vv, 1.5)
    weight_sum = float(6.0 * np.sum(weight))
    if weight_sum > 1e-12:
        weight = weight * ((4.0 * np.pi) / weight_sum)

    faces = []
    for forward, up in _cubemap_face_bases():
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up_orth = np.cross(right, forward)
        up_orth = up_orth / (np.linalg.norm(up_orth) + 1e-8)

        dirs = (
            forward[None, None, :]
            + uu[..., None] * right[None, None, :]
            + vv[..., None] * up_orth[None, None, :]
        )
        dirs = dirs / (np.linalg.norm(dirs, axis=2, keepdims=True) + 1e-8)
        dirs_flat = dirs.reshape(-1, 3)
        Y = evaluate_sh_basis(dirs_flat, max_order=2).astype(np.float32)
        faces.append(
            {
                "forward": forward,
                "up": up_orth,
                "dirs": dirs_flat,
                "Y": Y,
                "weight": weight.reshape(-1).astype(np.float32),
            }
        )

    _CUBEMAP_CACHE[cube_res] = faces
    return faces


def _read_output_image(graph_obj) -> np.ndarray:
    output = None
    try:
        if hasattr(graph_obj, "get_output"):
            output = graph_obj.get_output("PathTracer.color")
        else:
            output = graph_obj.getOutput("PathTracer.color")
    except Exception:
        output = None
    if output is None:
        raise RuntimeError("Failed to get PathTracer.color output texture")
    if hasattr(output, "to_numpy"):
        img = output.to_numpy()
    elif hasattr(output, "toNumpy"):
        img = output.toNumpy()
    else:
        raise RuntimeError("Output texture does not support to_numpy()")
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 1:
        # fallback: try to infer square image
        size = int(np.sqrt(img.size / 4))
        img = img.reshape(size, size, -1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    return img


def _apply_radiance_clamp(img: np.ndarray, clamp_value: Optional[float]) -> np.ndarray:
    if clamp_value is None:
        return img
    try:
        clamp_f = float(clamp_value)
    except Exception:
        return img
    if clamp_f <= 0.0:
        return img
    return np.minimum(img, clamp_f)


def render_sh_cubemap(
    testbed,
    graph,
    scene,
    probe: np.ndarray,
    cube_res: int = 32,
    num_frames: int = 4,
    seed_base: Optional[int] = None,
    radiance_clamp: Optional[float] = None,
    near_plane: float = 0.001,
    far_plane: Optional[float] = None,
) -> np.ndarray:
    """Render cubemap around probe and project to SH coefficients."""
    graph_obj = None
    for candidate in (graph, getattr(testbed, "activeGraph", None), getattr(testbed, "render_graph", None)):
        if candidate is not None:
            graph_obj = candidate
            break
    if graph_obj is None:
        raise RuntimeError("No render graph available for output readback.")

    path_tracer = None
    try:
        if hasattr(graph_obj, "get_pass"):
            path_tracer = graph_obj.get_pass("PathTracer")
        else:
            path_tracer = graph_obj.getPass("PathTracer")
    except Exception:
        path_tracer = None

    _has_reset = hasattr(path_tracer, "reset") if path_tracer is not None else False

    def _set_fixed_seed(seed_value: int) -> None:
        if path_tracer is None:
            return
        try:
            path_tracer.set_properties({"fixedSeed": int(seed_value)})
        except Exception:
            try:
                path_tracer.setProperties({"fixedSeed": int(seed_value)})
            except Exception:
                pass

    # Configure camera for cubemap. Use scene scale if far plane not provided.
    if far_plane is None or far_plane <= 0.0:
        try:
            bmin, bmax = get_scene_bounds(scene)
            diag = float(np.linalg.norm(bmax - bmin))
            far_plane = max(10.0, diag * 1.2)
        except Exception:
            far_plane = 10.0
    configure_cubemap_camera(scene, near_plane=near_plane, far_plane=float(far_plane))

    faces = _get_cubemap_cache(cube_res)
    coeffs = np.zeros((3, 9), dtype=np.float32)

    try:
        testbed.resizeFrameBuffer(cube_res, cube_res)
    except Exception:
        try:
            testbed.resize_frame_buffer(cube_res, cube_res)
        except Exception:
            pass

    for face_idx, face in enumerate(faces):
        forward = face["forward"]
        up = face["up"]
        set_camera_orientation(scene, probe, forward, up)

        accum_img = None
        frames = max(1, int(num_frames))
        for f in range(frames):
            if seed_base is not None:
                _set_fixed_seed(seed_base + f + face_idx * 7)
            if _has_reset:
                try:
                    path_tracer.reset()
                except Exception:
                    pass

            rendered = False
            if hasattr(testbed, "frame"):
                try:
                    testbed.frame()
                    rendered = True
                except Exception:
                    rendered = False
            if not rendered and hasattr(testbed, "renderFrame"):
                try:
                    testbed.renderFrame()
                    rendered = True
                except Exception:
                    rendered = False
            if not rendered:
                try:
                    import falcor
                    falcor.renderFrame()
                    rendered = True
                except Exception:
                    rendered = False
            if not rendered:
                raise RuntimeError("No renderFrame method available on testbed or falcor module.")

            img = _read_output_image(graph_obj)
            if accum_img is None:
                accum_img = img
            else:
                accum_img = accum_img + img
        img = accum_img / float(frames)
        img = _apply_radiance_clamp(img, radiance_clamp)

        L = img.reshape(-1, 3)
        w = face["weight"]
        Y = face["Y"]
        for c in range(3):
            coeffs[c] += (w[:, None] * Y * L[:, c:c+1]).sum(axis=0)

    return coeffs.reshape(-1)


def get_scene_bounds(scene) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min, max) bounds as numpy arrays."""
    bounds = scene.bounds
    min_pt = np.array([bounds.minPoint.x, bounds.minPoint.y, bounds.minPoint.z], dtype=np.float32)
    max_pt = np.array([bounds.maxPoint.x, bounds.maxPoint.y, bounds.maxPoint.z], dtype=np.float32)
    return min_pt, max_pt
