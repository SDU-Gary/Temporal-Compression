# 命令模板（Falcor + Bistro 时序数据）

> 默认使用 cubemap 模式以加速生成。

## 1) 场景预览
```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
FALCOR_DEVICE_TYPE=Vulkan \
LD_LIBRARY_PATH=/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug \
/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10 \
tools/render_bistro_preview.py \
  --scene /home/kyrie/毕设/1_data_generation/scenes/Bistro_v5_2/BistroExterior.pyscene \
  --output-dir /home/kyrie/毕设/metadata/bistro_preview \
  --width 1280 --height 720 --spp 256 \
  --falcor-python-path /home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python
```

## 2) 训练集（600 帧 / cubemap）
```bash
BISTRO_FBX=/home/kyrie/毕设/1_data_generation/scenes/Bistro_v5_2/BistroExterior.fbx \
BISTRO_NUM_FRAMES=600 \
BISTRO_FPS=30 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
FALCOR_DEVICE_TYPE=Vulkan \
LD_LIBRARY_PATH=/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug \
/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10 \
1_data_generation/falcor/generate_bistro_temporal_spheres.py \
  --scene /home/kyrie/毕设/1_data_generation/falcor/scenes/bistro_exterior_emissive_spheres.pyscene \
  --output /home/kyrie/毕设/1_data_generation/output/bistro_temporal/train \
  --num-frames 600 --fps 30 \
  --sh-mode cubemap --cube-res 16 \
  --probe-mode adaptive --max-probes 200 --probe-uniform-ratio 0.7 \
  --spp 32 --accum-frames 1 \
  --falcor-python-path /home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python
```

## 2.1) 深度引导探针采样（可选，推荐）
```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
FALCOR_DEVICE_TYPE=Vulkan \
LD_LIBRARY_PATH=/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug \
/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10 \
tools/sample_surface_probes.py \
  --scene /home/kyrie/毕设/1_data_generation/falcor/scenes/bistro_exterior_emissive_spheres.pyscene \
  --output /home/kyrie/毕设/1_data_generation/output/bistro_temporal/probes_surface.npy \
  --num-probes 800 --adaptive-ratio 0.7 \
  --width 320 --height 180 --num-cameras 3 \
  --camera-offset 2.0 --camera-up-offset 1.5 \
  --light-sample-step 10 --eps 0.1 \
  --falcor-python-path /home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python \
  --save-ply
```

随后在训练命令中加入：
```
--probe-file /home/kyrie/毕设/1_data_generation/output/bistro_temporal/probes_surface.npy
```

## 3) 测试/论文集（10 帧 / 高质量）
```bash
BISTRO_FBX=/home/kyrie/毕设/1_data_generation/scenes/Bistro_v5_2/BistroExterior.fbx \
BISTRO_NUM_FRAMES=600 \
BISTRO_FPS=30 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
FALCOR_DEVICE_TYPE=Vulkan \
LD_LIBRARY_PATH=/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug \
/home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/pythondist/bin/python3.10 \
1_data_generation/falcor/generate_bistro_temporal_spheres.py \
  --scene /home/kyrie/毕设/1_data_generation/falcor/scenes/bistro_exterior_emissive_spheres.pyscene \
  --output /home/kyrie/毕设/1_data_generation/output/bistro_temporal/test_gt \
  --num-frames 600 --fps 30 --frame-step 60 \
  --sh-mode cubemap --cube-res 64 \
  --probe-mode adaptive --max-probes 200 --probe-uniform-ratio 0.7 \
  --spp 512 --accum-frames 1 \
  --falcor-python-path /home/kyrie/毕设/Falcor/build/linux-gcc/bin/Debug/python
```
