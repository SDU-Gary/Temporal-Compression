---
url: https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html
crawled_at: 2025-11-13T17:12:08.515939
title: https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Download notebook ](https://raw.githubusercontent.com/mitsuba-renderer/mitsuba-tutorials/master/inverse_rendering/caustics_optimization.ipynb) [ Download data ](https://d38rqfq1h7iukm.cloudfront.net/scenes/tutorials/scenes.zip)
# Caustics optimization[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html#Caustics-optimization "Link to this heading")
## Overview[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html#Overview "Link to this heading")
This tutorial contains an advanced inverse rendering example: recovering the surface displacement (heightmap) of a slab of glass such that light passing through it focuses into a specific desired image.
This reproduces the results showcased in Section 4.3 of [Mitsuba 2: A Retargetable Forward and Inverse Renderer](https://rgl.epfl.ch/publications/NimierDavidVicini2019Mitsuba2).
🚀 **You will learn how to:**
  * Create a simple mesh from Python
  * Use the particle tracer integrator (ptracer)
  * Load a scene defined procedurally from Python
  * Apply a heightmap to a mesh from Python
  * Optimize “latent” variables, i.e. variables which are not directly defined as part of the scene but that affect it


The scene will be setup as follows:
  1. A directional area light (white or colorful, depending on the target image)
  2. Light from the emitter passes through a glass slab. We will optimize the slab’s surface (via a heightmap)…
  3. …so that light is focused on a receiving plane in a way that reproduces a desired target image.


In order to efficiently render and optimize this scene, we will use the Particle Tracer integrator ([ptracer](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_integrators.html#particle-tracer-ptracer)), which traces rays from the emitter rather than the sensor.
![Caustic Optimization diagram](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html)
## 0. Setup[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html#0.-Setup "Link to this heading")
We start by importing Mitsuba and selecting an appropriate variant supporting automatic differentiation (AD), as it is required to compute gradients with respect to the slab’s surface.
Copy to clipboard
```
importos
fromos.pathimport realpath, join

importdrjitasdr
importmitsubaasmi

mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')

```
Copy to clipboard
## 1. Choosing a configuration[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html#1.-Choosing-a-configuration "Link to this heading")
In this tutorial, we can attempt to reproduce either a grayscale image using a uniform emitter, or a color image using an RGB emitter. Here, we define those two options and select one.
Feel free to define additional configurations, e.g. to target a different reference image of your choice.
Copy to clipboard
```
SCENE_DIR = realpath('../scenes')

CONFIGS = {
    'wave': {
        'emitter': 'gray',
        'reference': join(SCENE_DIR, 'references/wave-1024.jpg'),
    },
    'sunday': {
        'emitter': 'bayer',
        'reference': join(SCENE_DIR, 'references/sunday-512.jpg'),
    },
}

# Pick one of the available configs
config_name = 'sunday'
# config_name = 'wave'

config = CONFIGS[config_name]
print('[i] Reference image selected:', config['reference'])
mi.Bitmap(config['reference'])

```
Copy to clipboard
```
[i] Reference image selected: /home/rami/mitsuba3/tutorials/scenes/references/sunday-512.jpg

```
Copy to clipboard
Copy to clipboard
![](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html)
In the following cell we define the hyper parameters controlling the optimization, such as the number of iterations and number of samples per pixels for the differentiable rendering simulation.
Copy to clipboard
```
if 'PYTEST_CURRENT_TEST' not in os.environ:
    config.update({
        'render_resolution': (128, 128),
        'heightmap_resolution': (512, 512),
        'n_upsampling_steps': 4,
        'spp': 32,
        'max_iterations': 1000,
        'learning_rate': 3e-5,
    })
else:
    # IGNORE THIS: When running under pytest, adjust parameters to reduce computation time
    config.update({
        'render_resolution': (64, 64),
        'heightmap_resolution': (128, 128),
        'n_upsampling_steps': 0,
        'spp': 8,
        'max_iterations': 25,
        'learning_rate': 3e-5,
    })


output_dir = realpath(join('.', 'outputs', config_name))
os.makedirs(output_dir, exist_ok=True)
print('[i] Results will be saved to:', output_dir)

```
Copy to clipboard
```
[i] Results will be saved to: /home/rami/mitsuba3/tutorials/inverse_rendering/outputs/sunday

```
Copy to clipboard
## 2. Creating the scene[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html#2.-Creating-the-scene "Link to this heading")
Depending on the chosen configuration, a different type of emitter will need to be used. For this reason, we define the scene dynamically directly from Python as a dictionary and load it with `load_dict()`.
Copy to clipboard
```
# Make sure that resources from the scene directory can be found
mi.file_resolver().append(SCENE_DIR)

```
Copy to clipboard
### Creating the lens mesh[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html#Creating-the-lens-mesh "Link to this heading")
The goal of the optimization is to recover the heightfield that needs to be applied to a slab of glass so that it focuses light in just the right way to reproduce the desired target image.
The heightmap will be represented as a texture and applied to the slab’s vertices. For this technique to be effective, the slab must have enough geometric resolution (vertices) to match the heightmap texture.
![Lens mesh preview](https://mitsuba.readthedocs.io/en/stable/_images/lens-heightmap.png)
Here, we generate the appropriate mesh directly from Python: a simple tesselated plane with the desired resolution and save it to disk.
Copy to clipboard
```
defcreate_flat_lens_mesh(resolution):
    # Generate UV coordinates
    U, V = dr.meshgrid(
        dr.linspace(mi.Float, 0, 1, resolution[0]),
        dr.linspace(mi.Float, 0, 1, resolution[1]),
        indexing='ij'
    )
    texcoords = mi.Vector2f(U, V)

    # Generate vertex coordinates
    X = 2.0 * (U - 0.5)
    Y = 2.0 * (V - 0.5)
    vertices = mi.Vector3f(X, Y, 0.0)

    # Create two triangles per grid cell
    faces_x, faces_y, faces_z = [], [], []
    for i in range(resolution[0] - 1):
        for j in range(resolution[1] - 1):
            v00 = i * resolution[1] + j
            v01 = v00 + 1
            v10 = (i + 1) * resolution[1] + j
            v11 = v10 + 1
            faces_x.extend([v00, v01])
            faces_y.extend([v10, v10])
            faces_z.extend([v01, v11])

    # Assemble face buffer
    faces = mi.Vector3u(faces_x, faces_y, faces_z)

    # Instantiate the mesh object
    mesh = mi.Mesh("lens-mesh", resolution[0] * resolution[1], len(faces_x), has_vertex_texcoords=True)

    # Set its buffers
    mesh_params = mi.traverse(mesh)
    mesh_params['vertex_positions'] = dr.ravel(vertices)
    mesh_params['vertex_texcoords'] = dr.ravel(texcoords)
    mesh_params['faces'] = dr.ravel(faces)
    mesh_params.update()

    return mesh

```
Copy to clipboard
Copy to clipboard
```
lens_res = config.get('lens_res', config['heightmap_resolution'])
lens_fname = join(output_dir, 'lens_{}_{}.ply'.format(*lens_res))

if not os.path.isfile(lens_fname):
    m = create_flat_lens_mesh(lens_res)
    m.write_ply(lens_fname)
    print('[+] Wrote lens mesh ({}x{} tesselation) file to: {}'.format(*lens_res, lens_fname))

```
Copy to clipboard
### Creating the emitter[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html#Creating-the-emitter "Link to this heading")
As explained previously, depending on whether we are trying to reproduce a grayscale or colorful target image, we setup the emitter to either emit constant white light or an RGB Bayer pattern. In the latter case, the pattern is generated on-the-fly and passed to the emitter as an in-memory [Bitmap](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_textures.html#bitmap-texture-bitmap) texture.
Copy to clipboard
```
emitter = None
if config['emitter'] == 'gray':
    emitter = {
        'type':'directionalarea',
        'radiance': {
            'type': 'spectrum',
            'value': 0.8
        },
    }
elif config['emitter'] == 'bayer':
    bayer = dr.zeros(mi.TensorXf, (32, 32, 3))
    bayer[ ::2,  ::2, 2] = 2.2
    bayer[ ::2, 1::2, 1] = 2.2
    bayer[1::2, 1::2, 0] = 2.2

    emitter = {
        'type':'directionalarea',
        'radiance': {
            'type': 'bitmap',
            'bitmap': mi.Bitmap(bayer),
            'raw': True,
            'filter_type': 'nearest'
        },
    }

```
Copy to clipboard
### Creating the integrator[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html#Creating-the-integrator "Link to this heading")
The chosen light source emits light in a single direction, which would be very difficult (or impossible) to sample correctly with a standard path tracer. For this reason, we use a particle tracer (`ptracer`), which starts rays from the emitters rather than the sensor.
Copy to clipboard
```
integrator = {
    'type': 'ptracer',
    'samples_per_pass': 256,
    'max_depth': 4,
    'hide_emitters': False,
}

```
Copy to clipboard
### Assembling the scene[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html#Assembling-the-scene "Link to this heading")
The sensor looks directly at the receiving plane where the caustic will be formed. The light source and optimized lens will stand behind the camera. Note that since the camera is an idealized pinhole camera and does not occupy any space, it will not cast any shadow on the receiving plane.
Copy to clipboard
```
# Looking at the receiving plane, not looking through the lens
sensor_to_world = mi.ScalarTransform4f().look_at(
    target=[0, -20, 0],
    origin=[0, -4.65, 0],
    up=[0, 0, 1]
)
resx, resy = config['render_resolution']
sensor = {
    'type': 'perspective',
    'near_clip': 1,
    'far_clip': 1000,
    'fov': 45,
    'to_world': sensor_to_world,

    'sampler': {
        'type': 'independent',
        'sample_count': 512  # Not really used
    },
    'film': {
        'type': 'hdrfilm',
        'width': resx,
        'height': resy,
        'pixel_format': 'rgb',
        'rfilter': {
            # Important: smooth reconstruction filter with a footprint larger than 1 pixel.
            'type': 'gaussian'
        }
    },
}

```
Copy to clipboard
We can now put everything together into a single large dictionary, where we also define the remaining geometry (receiving plane, geometry, etc).
Copy to clipboard
```
scene = {
    'type': 'scene',
    'sensor': sensor,
    'integrator': integrator,
    # Glass BSDF
    'simple-glass': {
        'type': 'dielectric',
        'id': 'simple-glass-bsdf',
        'ext_ior': 'air',
        'int_ior': 1.5,
        'specular_reflectance': { 'type': 'spectrum', 'value': 0 },
    },
    'white-bsdf': {
        'type': 'diffuse',
        'id': 'white-bsdf',
        'reflectance': { 'type': 'rgb', 'value': (1, 1, 1) },
    },
    'black-bsdf': {
        'type': 'diffuse',
        'id': 'black-bsdf',
        'reflectance': { 'type': 'spectrum', 'value': 0 },
    },
    # Receiving plane
    'receiving-plane': {
        'type': 'obj',
        'id': 'receiving-plane',
        'filename': 'meshes/rectangle.obj',
        'to_world': \
            mi.ScalarTransform4f().look_at(
                target=[0, 1, 0],
                origin=[0, -7, 0],
                up=[0, 0, 1]
            ).scale((5, 5, 5)),
        'bsdf': {'type': 'ref', 'id': 'white-bsdf'},
    },
    # Glass slab, excluding the 'exit' face (added separately below)
    'slab': {
        'type': 'obj',
        'id': 'slab',
        'filename': 'meshes/slab.obj',
        'to_world': mi.ScalarTransform4f().rotate(axis=(1, 0, 0), angle=90),
        'bsdf': {'type': 'ref', 'id': 'simple-glass'},
    },
    # Glass rectangle, to be optimized
    'lens': {
        'type': 'ply',
        'id': 'lens',
        'filename': lens_fname,
        'to_world': mi.ScalarTransform4f().rotate(axis=(1, 0, 0), angle=90),
        'bsdf': {'type': 'ref', 'id': 'simple-glass'},
    },

    # Directional area emitter placed behind the glass slab
    'focused-emitter-shape': {
        'type': 'obj',
        'filename': 'meshes/rectangle.obj',
        'to_world': mi.ScalarTransform4f().look_at(
            target=[0, 0, 0],
            origin=[0, 5, 0],
            up=[0, 0, 1]
        ),
        'bsdf': {'type': 'ref', 'id': 'black-bsdf'},
        'focused-emitter': emitter,
    },
}

```
Copy to clipboard
Finally, the scene is loaded which instantiates all of the appropriate plugins, loads the geometry, etc.
Copy to clipboard
```
scene = mi.load_dict(scene)

```
Copy to clipboard
## 3. Loading the reference image[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html#3.-Loading-the-reference-image "Link to this heading")
Now that the sensor has been defined, we can load the reference image and ensure that its resolution matches the render resolution.
Copy to clipboard
```
defload_ref_image(config, resolution, output_dir):
    b = mi.Bitmap(config['reference'])
    b = b.convert(mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float32, False)
    if dr.any(b.size() != resolution):
        b = b.resample(resolution)

    mi.util.write_bitmap(join(output_dir, 'out_ref.exr'), b)

    print('[i] Loaded reference image from:', config['reference'])
    return mi.TensorXf(b)

# Make sure the reference image will have a resolution matching the sensor
sensor = scene.sensors()[0]
crop_size = sensor.film().crop_size()
image_ref = load_ref_image(config, crop_size, output_dir=output_dir)

```
Copy to clipboard
```
[i] Loaded reference image from: /home/rami/mitsuba3/tutorials/scenes/references/sunday-512.jpg

```
Copy to clipboard
## 4. Creating the displacement texture[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html#4.-Creating-the-displacement-texture "Link to this heading")
Rather than optimizing the unconstrained vertex positions of the lens directly, we optimize values of a high-resolution heightmap. Here, we create the heightmap texture and create an optimizer that will work on its values.
Notice how the `traverse()` method is used directly on our new texture object, rather than on the scene loaded earlier.
Copy to clipboard
```
initial_heightmap_resolution = [r // (2 ** config['n_upsampling_steps'])
                                for r in config['heightmap_resolution']]
upsampling_steps = dr.square(dr.linspace(mi.Float, 0, 1, config['n_upsampling_steps']+1, endpoint=False).numpy()[1:])
upsampling_steps = (config['max_iterations'] * upsampling_steps).astype(int)
print('The resolution of the heightfield will be doubled at iterations:', upsampling_steps)

heightmap_texture = mi.load_dict({
    'type': 'bitmap',
    'id': 'heightmap_texture',
    'bitmap': mi.Bitmap(dr.zeros(mi.TensorXf, initial_heightmap_resolution)),
    'raw': True,
})

# Actually optimized: the heightmap texture
params = mi.traverse(heightmap_texture)
params.keep(['data'])
opt = mi.ad.Adam(lr=config['learning_rate'], params=params)

```
Copy to clipboard
```
The resolution of the heightfield will be doubled at iterations: [ 40 160 360 640]

```
Copy to clipboard
## 5. Applying the displacement texture[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html#5.-Applying-the-displacement-texture "Link to this heading")
At each iteration, the lens’ vertices will displaced from their original position along their normal by the value of the heightmap. Don’t forget that the geometric resolution of the lens mesh (number of vertices) must also be high enough for this technique to work as expected.
Copy to clipboard
```
params_scene = mi.traverse(scene)

# We will always apply displacements along the original normals and
# starting from the original positions.
positions_initial = dr.unravel(mi.Vector3f, params_scene['lens.vertex_positions'])
normals_initial   = dr.unravel(mi.Vector3f, params_scene['lens.vertex_normals'])

lens_si = dr.zeros(mi.SurfaceInteraction3f, dr.width(positions_initial))
lens_si.uv = dr.unravel(type(lens_si.uv), params_scene['lens.vertex_texcoords'])

defapply_displacement(amplitude = 1.):
    # Enforce reasonable range. For reference, the receiving plane
    # is 7 scene units away from the lens.
    vmax = 1 / 100.
    params['data'] = dr.clip(params['data'], -vmax, vmax)
    dr.enable_grad(params['data'])
    params.update()

    height_values = heightmap_texture.eval_1(lens_si)
    new_positions = (height_values * normals_initial * amplitude + positions_initial)
    params_scene['lens.vertex_positions'] = dr.ravel(new_positions)
    params_scene.update()

```
Copy to clipboard
## 6. Running the optimization[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html#6.-Running-the-optimization "Link to this heading")
We’re finally ready to start the optimization itself!
At each iteration, we apply the current heightmap displacement to the lens surface and render the scene with automatic differentiation enabled.
We then compare the render to our target image with a scale-independent L2 loss. We divide out the average brightness in the loss so that the general brightness of the emitter (set arbitrarily) does not interfere with the optimization.
After backpropagating through the computation graph, we use the gradients of the loss w.r.t. the heightmap values to update the heightmap.
Copy to clipboard
```
defscale_independent_loss(image, ref):
"""Brightness-independent L2 loss function."""
    scaled_image = image / dr.mean(dr.detach(image))
    scaled_ref = ref / dr.mean(ref)
    return dr.mean(dr.square(scaled_image - scaled_ref))

```
Copy to clipboard
We add two common tricks to improve the quality of the optimization:
  * Increasing the rendering quality (sample count) and decreasing the learning rate towards the end of the optimization.
  * Progressively increasing the resolution of the heightmap being optimized.


Copy to clipboard
```
importtime
start_time = time.time()
mi.set_log_level(mi.LogLevel.Warn)
iterations = config['max_iterations']
loss_values = []
spp = config['spp']

for it in range(iterations):
    t0 = time.time()

    # Apply displacement and update the scene BHV accordingly
    apply_displacement()

    # Perform a differentiable rendering of the scene
    image = mi.render(scene, params, seed=it, spp=2 * spp, spp_grad=spp)

    # Scale-independent L2 function
    loss = scale_independent_loss(image, image_ref)

    # Back-propagate errors to input parameters and take an optimizer step
    dr.backward(loss)

    # Take a gradient step
    opt.step()

    # Increase resolution of the heightmap
    if it in upsampling_steps:
        opt['data'] = dr.upsample(opt['data'], scale_factor=(2, 2, 1))

    # Carry over the update to our "latent variable" (the heightmap values)
    params.update(opt)

    # Log progress
    elapsed_ms = 1000. * (time.time() - t0)
    current_loss = loss.array[0]
    loss_values.append(current_loss)
    mi.logger().log_progress(
        it / (iterations-1),
        f'Iteration {it:03d}: loss={current_loss:g} (took {elapsed_ms:.0f}ms)',
        'Caustic Optimization', '')


    # Increase rendering quality toward the end of the optimization
    if it in (int(0.7 * iterations), int(0.9 * iterations)):
        spp *= 2
        opt.set_learning_rate(0.5 * opt.learning_rate())


end_time = time.time()
print(((end_time - start_time) * 1000) / iterations, ' ms per iteration on average')
mi.set_log_level(mi.LogLevel.Info)

```
Copy to clipboard
```
37.61896562576294  ms per iteration on average

```
Copy to clipboard
Once the optimization has completed, we save the final heightmap and the corresponding lens with displacement applied.
Copy to clipboard
```
mi.set_log_level(mi.LogLevel.Error)
fname = join(output_dir, 'heightmap_final.exr')
mi.util.write_bitmap(fname, params['data'])
print('[+] Saved final heightmap state to:', os.path.basename(fname))

fname = join(output_dir, 'lens_displaced.ply')
apply_displacement()
lens_mesh = [m for m in scene.shapes() if m.id() == 'lens'][0]
lens_mesh.write_ply(fname)
print('[+] Saved displaced lens to:', os.path.basename(fname))

```
Copy to clipboard
```
[+] Saved final heightmap state to: heightmap_final.exr
[+] Saved displaced lens to: lens_displaced.ply

```
Copy to clipboard
## 7. Visualizing the results[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html#7.-Visualizing-the-results "Link to this heading")
Finally, we plot the evolution of the loss and show the final result next to the reference.
Copy to clipboard
```
importmatplotlib.pyplotasplt

defshow_image(ax, img, title):
    ax.imshow(mi.util.convert_to_bitmap(img))
    ax.axis('off')
    ax.set_title(title)

defshow_heightmap(fig, ax, values, title):
    im = ax.imshow(values.squeeze(), vmax=1e-4)
    fig.colorbar(im, ax=ax)
    ax.axis('off')
    ax.set_title(title)

fig, ax = plt.subplots(2, 2, figsize=(11, 10))
ax = ax.ravel()
ax[0].plot(loss_values)
ax[0].set_xlabel('Iteration'); ax[0].set_ylabel('Loss value'); ax[0].set_title('Convergence plot')

show_heightmap(fig, ax[1], params['data'].numpy(), 'Final heightmap')
show_image(ax[2], image_ref, 'Reference')
show_image(ax[3], image,     'Final state')
plt.show()

```
Copy to clipboard
![../../_images/src_inverse_rendering_caustics_optimization_47_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_caustics_optimization_47_0.png)
Congratulations! Feel free to define your own target images at the top of the notebook and run this tutorial again.
If you would like to improve the quality of the results, you could try the following: - Letting the optimization run for more iterations - Tweaking the learning rate and sample count - Progressively increasing the resolution of the heightmap through optimization, e.g. starting from a 16x16 heightmap and doubling the resolution every N iterations.
## See also[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html#See-also "Link to this heading")
  * [ptracer plugin](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_integrators.html#particle-tracer-ptracer)
  * [bitmap plugin](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_textures.html#bitmap-texture-bitmap)


* * *
