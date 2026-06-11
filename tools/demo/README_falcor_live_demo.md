# Falcor Live GT-vs-Pred Demo

This demo is for defense screen recording in a real Falcor/Testbed window.
It shows the same camera view split horizontally:

- Left half: the left half of the ground-truth probe SH rendering.
- Right half: the right half of the compressed model-predicted probe SH rendering.
- HUD: FPS and stage timings.

The split is a normal full-frame camera view. It does not squeeze a full GT
image into the left half or a full Pred image into the right half.

The runtime is split into two steps because the project training environment
uses Python 3.13/PyTorch, while Falcor embeds Python 3.10.

## 1. Precompute Pred SH

Run this with the project virtual environment:

```bash
source venv/bin/activate
python tools/demo/prepare_falcor_demo_sh.py \
  --dataset 1_data_generation/output/bistro_clean_v2/parametric_tensor.npz \
  --checkpoint 3_experiments/results/bistro_clean_v2/exp_A2_3_3_disable_loss_eff_seed19/global_best_falcor.pt \
  --output 3_experiments/results/demo/falcor_live_demo/a233_global_best_pred_sh.npz \
  --frame-start 0 \
  --max-frames 600 \
  --device cuda \
  --routing-profile-json '{"top_k":3,"training_soft_routing":true,"routing_soft_topk":8,"routing_temperature":0.18}'
```

## 2. Run Interactive Window

Environment/SH-only comparison:

```bash
LD_LIBRARY_PATH=/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug \
PYTHONPATH=/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python \
/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10 \
  tools/demo/falcor_live_compare_demo.py \
  --dataset 1_data_generation/output/bistro_clean_v2/parametric_tensor.npz \
  --pred-sh 3_experiments/results/demo/falcor_live_demo/a233_global_best_pred_sh.npz \
  --scene 1_data_generation/scenes/Bistro_v5_2/BistroExterior.pyscene \
  --width 1920 \
  --height 1080 \
  --mode env \
  --gbuffer-pass raster \
  --gbuffer-mode realtime \
  --mouse-look-mode fps \
  --mouse-look-scale 3.0 \
  --camera-speed 3.0 \
  --playback-fps 60
```

Shared direct/shadow base plus SH comparison:

```bash
LD_LIBRARY_PATH=/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug \
PYTHONPATH=/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python \
/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10 \
  tools/demo/falcor_live_compare_demo.py \
  --dataset 1_data_generation/output/bistro_clean_v2/parametric_tensor.npz \
  --pred-sh 3_experiments/results/demo/falcor_live_demo/a233_global_best_pred_sh.npz \
  --scene 1_data_generation/scenes/Bistro_v5_2/BistroExterior.pyscene \
  --width 1920 \
  --height 1080 \
  --mode full \
  --full-base whitted \
  --pt-bounces 1 \
  --gbuffer-pass raster \
  --gbuffer-mode realtime \
  --base-gbuffer-pass auto \
  --base-mode realtime \
  --base-scale 0.15 \
  --mouse-look-mode fps \
  --mouse-look-scale 3.0 \
  --camera-speed 3.0 \
  --playback-fps 60
```

For interactive camera motion, use `--gbuffer-pass raster --gbuffer-mode
realtime`. The local Falcor build includes a non-ROV raster GBuffer variant for
the demo's required channels (`posW`, `normW`, `diffuseOpacity`, `depth`,
`linearZ`), so the comparison pass no longer has to fall back to the ray-traced
GBuffer when the camera moves.

For defense recording with a fixed camera, you may still use `--gbuffer-mode
reuse-first` and `--base-mode reuse-first`. The first rendered frame pays the
Falcor warmup, acceleration structure, GBuffer, and base-render costs; following
frames reuse the same geometry buffers and direct/shadow base, so the live loop
is dominated by SH upload, probe-field reconstruction, and the final split
comparison shader.

Camera movement is supported in cached mode too: when the camera position,
target, or lens parameters change, the cached GBuffer and base image are
invalidated and rebuilt on the next frame. For smooth first-person movement,
prefer realtime raster GBuffer instead of relying on cache invalidation.

Useful camera controls:

- `--mouse-look-mode fps`: default. Rotate the camera directly from plain mouse movement over the scene view.
- `--mouse-look-mode drag`: fallback mode. Rotate only while left-dragging.
- `--mouse-look-scale 2.0` or `3.0`: increase look sensitivity.
- `W/S`: move forward/backward along the camera view direction.
- `A/D`: strafe left/right.
- `E` or `Space`: move up.
- `Q` or `Ctrl`: move down. Holding `Ctrl` also applies the slow-speed multiplier.
- `Shift`: sprint.
- `--camera-speed 3.0`: base WASD/QE translation speed.
- `--camera-fast-multiplier 4.0`: Shift speed multiplier.
- `--camera-slow-multiplier 0.25`: Ctrl speed multiplier.

The mouse look and keyboard translation are implemented in Python so the demo
has a consistent FPS-style control scheme. Falcor's Python window API does not
expose cursor locking, so FPS mode behaves like a direct mouse-look area rather
than a fully captured infinite mouse cursor.

In full mode, the shared Whitted direct/shadow base consumes a packed vbuffer.
On the current Vulkan stack, the local non-ROV raster GBuffer is not robust for
that vbuffer path, so `--base-gbuffer-pass auto` intentionally prefers
`GBufferRT` for the Whitted base while the main comparison GBuffer remains
Raster. `--base-resolution-scale` is intentionally ignored in interactive
window mode because Falcor's Python-exposed `resize_frame_buffer()` resizes the
actual OS window. It is still useful for headless timing/smoke runs.

`--base-scale` controls how strongly the shared direct-light/shadow image is
added in full mode. The default is `0.15`; lower it further if the direct-light
version is still too bright, or increase it if the shadow/direct-light cue is too
subtle.

## Smoke Test

```bash
LD_LIBRARY_PATH=/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug \
PYTHONPATH=/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python \
/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10 \
  tools/demo/falcor_live_compare_demo.py \
  --dataset 1_data_generation/output/bistro_clean_v2/parametric_tensor.npz \
  --pred-sh 3_experiments/results/demo/falcor_live_demo/a233_global_best_pred_sh.npz \
  --scene 1_data_generation/scenes/Bistro_v5_2/BistroExterior.pyscene \
  --width 640 \
  --height 360 \
  --mode env \
  --gbuffer-pass raster \
  --gbuffer-mode realtime \
  --headless-smoke \
  --headless-timing-only \
  --max-frames 3
```
