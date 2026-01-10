---
url: https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html
crawled_at: 2025-11-13T17:10:01.052877
title: https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Download notebook ](https://raw.githubusercontent.com/mitsuba-renderer/mitsuba-tutorials/master/inverse_rendering/volume_optimization.ipynb) [ Download data ](https://d38rqfq1h7iukm.cloudfront.net/scenes/tutorials/scenes.zip)
# Volumetric inverse rendering[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#Volumetric-inverse-rendering "Link to this heading")
## Overview[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#Overview "Link to this heading")
In this tutorial, we use Mitsuba’s differentiable volumetric path tracer to optimize a scattering volume to match a set of (synthetic) reference images. We will optimize a 3D volume density that’s stored on a regular grid. The optimization will account for both direct and indirect illumination by using [path replay backpropagation](https://rgl.epfl.ch/publications/Vicini2021PathReplay) to compute derivatives of delta tracking and volumetric multiple scattering. The reconstructed volume parameters can then for example be re-rendered using novel illumination conditions.
🚀 **You will learn how to:**
  * Construct a scene with volumes
  * Optimize a volume grid to match a set of reference images
  * Upscale the optimized parameters during the optimization


## Setup[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#Setup "Link to this heading")
As always, we start with the usual imports and set a variant that supports automatic differentation.
Copy to clipboard
```
importmatplotlib.pyplotasplt

importdrjitasdr
importmitsubaasmi

mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')

```
Copy to clipboard
## Creating multiple sensors[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#Creating-multiple-sensors "Link to this heading")
We cannot hope to obtain a robust volumetric reconstruction using only a single reference image. Multiple viewpoints are needed to sufficiently constrain the reconstructed volume density. Using a multi-view optimization we can recover volume parameters that generalize to novel views (and illumination conditions).
In this tutorial, we use 5 sensors placed on a half circle around the origin. For the simple optimization in this tutorial this is sufficient, but more complex scenes may require using significantly more views (e.g., using 50-100 sensors is not unreasonable).
Copy to clipboard
```
frommitsubaimport ScalarTransform4f as T

sensor_count = 5
sensors = []

for i in range(sensor_count):
    angle = 180.0 / sensor_count * i - 90.0
    sensor_rotation = T().rotate([0, 1, 0], angle)
    sensor_to_world = T().look_at(target=[0, 0, 0], origin=[0, 0, 4], up=[0, 1, 0])
    sensors.append(mi.load_dict({
        'type': 'perspective',
        'fov': 45,
        'to_world': sensor_rotation @ sensor_to_world,
        'film': {
            'type': 'hdrfilm',
            'width': 64, 'height': 64,
            'filter': {'type': 'tent'}
        }
    }))
  

```
Copy to clipboard
## Rendering synthetic reference images[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#Rendering-synthetic-reference-images "Link to this heading")
We will now setup a simple scene with a constant environment illumination and a reference volume placed at the origin. The heterogenous volume is instantiated inside of a cube. We assign the `null` BSDF to the cube’s surface, since we do not want the cube’s surface to interact with light in any way (i.e., the surface should be invisible). To learn more about volume rendering in Mitsuba, please refer to the [plugin documentation](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_media.html#heterogeneous-medium-heterogeneous).
We then render this scene using the previously created sensors and store the resulting images in a list for later use.
Copy to clipboard
```
scene_dict = {
    'type': 'scene',
    'integrator': {'type': 'prbvolpath'},
    'object': {
        'type': 'cube',
        'bsdf': {'type': 'null'},
        'interior': {
            'type': 'heterogeneous',
            'sigma_t': {
                'type': 'gridvolume',
                'filename': '../scenes/volume.vol',
                'to_world': T().rotate([1, 0, 0], -90).scale(2).translate(-0.5)
            },
            'scale': 40
        }
    },
    'emitter': {'type': 'constant'}
}

scene_ref = mi.load_dict(scene_dict)

# Number of samples per pixel for reference images
ref_spp = 512

```
Copy to clipboard
Copy to clipboard
```
ref_images = [mi.render(scene_ref, sensor=sensors[i], spp=ref_spp) for i in range(sensor_count)]

```
Copy to clipboard
Copy to clipboard
```
fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
for i in range(sensor_count):
    axs[i].imshow(mi.util.convert_to_bitmap(ref_images[i]))
    axs[i].axis('off')

```
Copy to clipboard
![../../_images/src_inverse_rendering_volume_optimization_10_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_volume_optimization_10_0.png)
## Setting up the optimization scene[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#Setting-up-the-optimization-scene "Link to this heading")
Our goal is now to optimize a 3D volume density (also called extinction) to match the previously generated reference images. For this we create a second scene, where we replace the reference volume by a simple uniform initialization.
To initialize a volume grid from Python, we use the [VolumeGrid](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.VolumeGrid) object in conjunction with [TensorXf](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.TensorXf). The `VolumeGrid` class is responsible for loading and writing volumes from disk, similar to the `Bitmap` class for images. Using the `grid` property of the [gridvolume](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_media.html#grid-based-volume-data-source-gridvolume) plugin, it is possible to pass it directly to the plugin constructor in Python.
We initialize the extinction `sigma_t` to a low constant value, (e.g. `0.002`). This tends to help the optimization process, as it seems to be easier for the optimizer to increase the volume density rather than remove parts of a very dense volume.
Note that we use a fairly small initial volume resolution here. This is done on purpose since we will upsample the volume grid during the actual optimization process. As explained later, this typically improves the convexity of the volume optimization problem.
Copy to clipboard
```
v_res = 16

# Modify the scene dictionary
scene_dict['object'] = {
    'type': 'cube',
    'interior': {
        'type': 'heterogeneous',
        'sigma_t': {
            'type': 'gridvolume',
            'grid': mi.VolumeGrid(dr.full(mi.TensorXf, 0.002, (v_res, v_res, v_res, 1))),
            'to_world': T().translate(-1).scale(2.0)
        },
        'scale': 40.0,
    },
    'bsdf': {'type': 'null'}
}

scene = mi.load_dict(scene_dict)

```
Copy to clipboard
We load the modified scene and render it for all view angles. Those are going to be our initial image in the optimization process.
Copy to clipboard
```
init_images = [mi.render(scene, sensor=sensors[i], spp=ref_spp) for i in range(sensor_count)]

fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
for i in range(sensor_count):
    axs[i].imshow(mi.util.convert_to_bitmap(init_images[i]))
    axs[i].axis('off')

```
Copy to clipboard
![../../_images/src_inverse_rendering_volume_optimization_14_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_volume_optimization_14_0.png)
## Optimization[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#Optimization "Link to this heading")
We instantiate an `Adam` optimizer and load the `sigma_t` grid data as parameter to be optimized.
Copy to clipboard
```
params = mi.traverse(scene)

key = 'object.interior_medium.sigma_t.data'

opt = mi.ad.Adam(lr=0.02)
opt[key] = params[key]
params.update(opt);

iteration_count = 20
spp = 8

```
Copy to clipboard
We then run the optimization loop for a few iterations, similar to the other tutorials.
Copy to clipboard
```
for it in range(iteration_count):
    total_loss = 0.0
    for sensor_idx in range(sensor_count):
        # Perform the differentiable light transport simulation
        img = mi.render(scene, params, sensor=sensors[sensor_idx], spp=spp, seed=it)

        # L2 loss function
        loss = dr.mean(dr.square(img - ref_images[sensor_idx]))

        # Backpropagate gradients
        dr.backward(loss)

        # Take a gradient step
        opt.step()

        # Clamp the optimized density values. Since we used the `scale` parameter
        # when instantiating the volume, we are in fact optimizing extinction
        # in a range from [1e-6 * scale, scale].
        opt[key] = dr.clip(opt[key], 1e-6, 1.0)

        # Propagate changes to the scene
        params.update(opt)

        total_loss += loss
    print(f"Iteration {it:02d}: error={total_loss}", end='\r')

```
Copy to clipboard
```
Iteration 19: error=0.0446167

```
Copy to clipboard
## Intermediate results[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#Intermediate-results "Link to this heading")
We have only performed a few iterations so far and can take a look at the current results.
Copy to clipboard
```
intermediate_images = [mi.render(scene, sensor=sensors[i], spp=ref_spp) for i in range(sensor_count)]

fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
for i in range(sensor_count):
    axs[i].imshow(mi.util.convert_to_bitmap(intermediate_images[i]))
    axs[i].axis('off')

```
Copy to clipboard
![../../_images/src_inverse_rendering_volume_optimization_21_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_volume_optimization_21_0.png)
## Volume upsampling[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#Volume-upsampling "Link to this heading")
The results above don’t look great. One reason is the low resolution of the optimized volume grid. Let’s try to increase the resolution of the current grid and continue the optimization for a few more iterations. In practice it is almost always beneficial to leverage such a “multi-resolution” approach. At low resolution, the optimization will recover the overall shape, exploring a much simpler solution landscape. Moving on to a volume with a higher resolution allows recovering additional detail, while using the coarser solution as a starting point.
Luckily Dr.Jit provides [dr.upsample()](https://drjit.readthedocs.io/en/latest/reference.html#drjit.upsample), a functions for up-sampling tensor and texture data. We can easily create a higher resolution volume by passing the current optimzed tensor and specifying the desired shape (must be powers of two when upsampling `TensorXf`).
Copy to clipboard
```
opt[key] = dr.upsample(opt[key], shape=(64, 64, 64))
params.update(opt);

```
Copy to clipboard
Rendering the new, upsampled volume we can already notice a slight difference in the apparent sharpness. This is due to the _trilinear_ interpolation of density values that is used by the volumetric path tracer.
Copy to clipboard
```
upscale_images = [mi.render(scene, sensor=sensors[i], spp=ref_spp) for i in range(sensor_count)]

fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
for i in range(sensor_count):
    axs[i].imshow(mi.util.convert_to_bitmap(upscale_images[i]))
    axs[i].axis('off')

```
Copy to clipboard
![../../_images/src_inverse_rendering_volume_optimization_25_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_volume_optimization_25_0.png)
## Continuing the optimization[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#Continuing-the-optimization "Link to this heading")
Let’s now run our optimization loop for a few more iterations with the upscaled volume grid.
🗒 **Note**
The optimizer automatically resets the internal state (e.g., momentum) associated to the optimized variable when it detects a size change.
Copy to clipboard
```
for it in range(iteration_count):
    total_loss = 0.0
    for sensor_idx in range(sensor_count):
        img = mi.render(scene, params, sensor=sensors[sensor_idx], spp=8*spp, seed=it)
        loss = dr.mean(dr.square(img - ref_images[sensor_idx]))
        dr.backward(loss)
        opt.step()
        opt[key] = dr.clip(opt[key], 1e-6, 1.0)
        params.update(opt)
        total_loss += loss
    print(f"Iteration {it:02d}: error={total_loss}", end='\r')

```
Copy to clipboard
```
Iteration 19: error=0.01756774

```
Copy to clipboard
## Final results[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#Final-results "Link to this heading")
Finally we can render the final volume from the different view points and compare the images to the reference images.
Copy to clipboard
```
final_images = [mi.render(scene, sensor=sensors[i], spp=ref_spp) for i in range(sensor_count)]

fig, axs = plt.subplots(2, sensor_count, figsize=(14, 6))
for i in range(sensor_count):
    axs[0][i].imshow(mi.util.convert_to_bitmap(ref_images[i]))
    axs[0][i].axis('off')
    axs[1][i].imshow(mi.util.convert_to_bitmap(final_images[i]))
    axs[1][i].axis('off')

```
Copy to clipboard
![../../_images/src_inverse_rendering_volume_optimization_29_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_volume_optimization_29_0.png)
Of course the results could be further improved by taking more iterations or adopting other advanced optimization schemes, such as multi-resolution rendering where the rendering resolution is increased throughout the optimization process. Additionally, it can sometimes be beneficial to add a sparsity (e.g., an \\(L_1\\) loss on the density values) or smoothness prior (e.g., a total variation regularizer penalizing differences between neighboring voxels).
## See also[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#See-also "Link to this heading")
  * [prbvolpath plugin](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_integrators.html#path-replay-backpropagation-volumetric-integrator-prbvolpath)
  * [heterogeneous plugin](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_media.html#heterogeneous-medium-heterogeneous)
  * [gridvolume plugin](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_media.html#grid-based-volume-data-source-gridvolume)
  * [mitsuba.VolumeGrid](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.VolumeGrid)
  * [mitsuba.TensorXf](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.TensorXf)
  * [dr.upsample](https://drjit.readthedocs.io/en/latest/reference.html#drjit.upsample)


[Develop and launch modern apps with MongoDB Atlas, a resilient data platform.](https://server.ethicalads.io/proxy/click/9505/019a7c7a-89c8-7080-a710-29fad8abe9dd/)
[Ads by EthicalAds](https://www.ethicalads.io/advertisers/topics/data-science/?ref=ea-text)
* * *
