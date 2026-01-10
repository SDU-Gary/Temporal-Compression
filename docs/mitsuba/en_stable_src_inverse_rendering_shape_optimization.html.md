---
url: https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html
crawled_at: 2025-11-13T17:11:51.951374
title: https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Download notebook ](https://raw.githubusercontent.com/mitsuba-renderer/mitsuba-tutorials/master/inverse_rendering/shape_optimization.ipynb) [ Download data ](https://d38rqfq1h7iukm.cloudfront.net/scenes/tutorials/scenes.zip)
# Shape optimization[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html#Shape-optimization "Link to this heading")
## Overview[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html#Overview "Link to this heading")
In this tutorial, we will optimize a triangle mesh to match a target shape specfied using a set of reference renderings.
Gradients with regards to vertex positions are typically extremely sparse, since only vertices located on visibility discontinuities receive a contribution. As a consequence, naively optimizing a triangle mesh generally results in horrible, tangled meshes.
To avoid this, we will use the method from the paper “[Large Steps in Inverse Rendering of Geometry](http://rgl.epfl.ch/publications/Nicolet2021Large)”. This method optimizes a latent variable that enables smoother gradients.
🚀 **You will learn how to:**
  * Use the “large steps” algorithm to optimize a shape
  * Use remeshing to refine the optimized shape


## Setup[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html#Setup "Link to this heading")
As always, let’s import `drjit` and `mitsuba` and set a differentiation-aware variant.
Copy to clipboard
```
importdrjitasdr
importmitsubaasmi
importmatplotlib.pyplotasplt # We'll also want to plot some outputs
importos

mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')

```
Copy to clipboard
## Setting up sensors[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html#Setting-up-sensors "Link to this heading")
In order to accurately recover a shape, we need several reference renderings, taken from different viewpoints. Similar to the [volume optimisation tutorial](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#Optimization), we render each viewpoint separately during the optimisation.
Note that we also have to set the `sample_border` flag to `True` for the [hdrfilm](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_films.html#high-dynamic-range-film-hdrfilm) plugin. By enabling this option, Mitsuba will render regions slightly beyond the film’s boundaries, enabling us to track objects that enter or exit the frame.
Here we will generate 8 viewpoints evenly distributed on the sphere, using the [Fibonacci lattice](http://extremelearning.com.au/evenly-distributing-points-on-a-sphere/).
Copy to clipboard
```
frommitsubaimport ScalarTransform4f as T

sensor_count = 8
sensors = []

golden_ratio = (1 + 5**0.5)/2
for i in range(sensor_count):
    theta = 2 * dr.pi * i / golden_ratio
    phi = dr.acos(1 - 2*(i+0.5)/sensor_count)

    d = 5
    origin = [
        d * dr.cos(theta) * dr.sin(phi),
        d * dr.sin(theta) * dr.sin(phi),
        d * dr.cos(phi)
    ]

    sensors.append(mi.load_dict({
        'type': 'perspective',
        'fov': 45,
        'to_world': T().look_at(target=[0, 0, 0], origin=origin, up=[0, 1, 0]),
        'film': {
            'type': 'hdrfilm',
            'width': 256, 'height': 256,
            'filter': {'type': 'gaussian'},
            'sample_border': True,
        },
        'sampler': {
            'type': 'independent',
            'sample_count': 128
        },
    }))

```
Copy to clipboard
## Scene construction[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html#Scene-construction "Link to this heading")
Let’s now generate the reference renderings. We will load a scene with tthe target mesh and an environment map. Note the use of the `direct_reparam` integrator, that properly accounts for visibility discontinuities when differentiating, as in the [object pose estimation tutorial](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/object_pose_estimation.html). There won’t be many shadows in our scene intitially, so let’s turn off indirect visibility effects in the integrator to speed the optimization up.
Copy to clipboard
```
scene_dict = {
    'type': 'scene',
    'integrator': {
        'type': 'direct_projective',
        # Indirect visibility effects aren't that important here
        # let's turn them off and save some computation time
        'sppi': 0,
    },
    'emitter': {
        'type': 'envmap',
        'filename': "../scenes/textures/envmap2.exr",
    },
    'shape': {
        'type': 'ply',
        'filename': "../scenes/meshes/suzanne.ply",
        'bsdf': {'type': 'diffuse'}
    }
}

scene_target = mi.load_dict(scene_dict)

```
Copy to clipboard
We can now generate the reference image, our goal is to reconstruct [Blender’s Suzanne](https://en.wikipedia.org/wiki/Blender_\(software\)).
Copy to clipboard
```
defplot_images(images):
    fig, axs = plt.subplots(1, len(images), figsize=(20, 5))
    for i in range(len(images)):
        axs[i].imshow(mi.util.convert_to_bitmap(images[i]))
        axs[i].axis('off')

```
Copy to clipboard
Copy to clipboard
```
ref_images = [mi.render(scene_target, sensor=sensors[i], spp=256) for i in range(sensor_count)]
plot_images(ref_images)

```
Copy to clipboard
![../../_images/src_inverse_rendering_shape_optimization_12_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_shape_optimization_12_0.png)
The starting geometry which we’ll optimize is a relatively dense sphere. More challenging scenes and target shapes might require a better initialization.
Copy to clipboard
```
scene_dict['shape']['filename'] = '../scenes/meshes/ico_10k.ply'
scene_source = mi.load_dict(scene_dict)

init_imgs = [mi.render(scene_source, sensor=sensors[i], spp=128) for i in range(sensor_count)]
plot_images(init_imgs)

```
Copy to clipboard
![../../_images/src_inverse_rendering_shape_optimization_14_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_shape_optimization_14_0.png)
## Naive optimization[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html#Naive-optimization "Link to this heading")
It is tempting to apply the same steps as in previous tutorials, i.e. simply render the scene, compute a loss, and differentiate. Let’s try that first.
As per usual, all that is needed is an `Optimizer` and a simple optimization loop.
Copy to clipboard
```
params = mi.traverse(scene_source)
opt = mi.ad.Adam(lr=3e-2)
opt['shape.vertex_positions'] = params['shape.vertex_positions']

```
Copy to clipboard
Copy to clipboard
```
for it in range(5):
    total_loss = mi.Float(0.0)

    for sensor_idx in range(sensor_count):
        params.update(opt)

        img = mi.render(scene_source, params, sensor=sensors[sensor_idx], seed=it)

        # L2 Loss
        loss = dr.mean(dr.square(img - ref_images[sensor_idx]))
        dr.backward(loss)

        opt.step()

        total_loss += loss

    print(f"Iteration {1+it:03d}: Loss = {total_loss}", end='\r')

```
Copy to clipboard
```
Iteration 005: Loss = [0.336332]

```
Copy to clipboard
Copy to clipboard
```
final_imgs = [mi.render(scene_source, sensor=sensors[i], spp=128) for i in range(sensor_count)]
plot_images(final_imgs)

```
Copy to clipboard
![../../_images/src_inverse_rendering_shape_optimization_19_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_shape_optimization_19_0.png)
As you can see, even after only a few steps, this approach produces an unusable mess of triangles. This is a consequence of the sparsity of the visibility-related gradients. Indeed, these are only present on edges that are on the silhouette of the mesh for a given viewpoint.
Copy to clipboard
```
# Reset the scene
scene_source = mi.load_dict(scene_dict)
params = mi.traverse(scene_source)

```
Copy to clipboard
## Large Steps Optimization[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html#Large-Steps-Optimization "Link to this heading")
Rather than directly optimizing cartesian vertex coordinates, the work of [[Nicolet et al. 2021]](http://rgl.epfl.ch/publications/Nicolet2021Large) “ _Large Steps in Inverse Rendering of Geometry_ ” introduces a latent representation that improves the smoothnes of gradients. In short, a latent variable u is defined as u=(I+λL)v, where v denotes the vertex positions, L is the (combinatorial) Laplacian of the given mesh, and λ is a user-defined hyper-parameter. Intuitively, this parameter λ defines by how much the gradients should be smoothed out on the surface.
This approach is readily available in Mitsuba in the `mi.ad.LargeSteps` class. It requires [cholespy](http://rgl.epfl.ch/publications/Nicolet2021Large), a Python package to solve sparse linear systems with Cholesky factorisations.
Copy to clipboard
```
!pip
```
Copy to clipboard
```
Requirement already satisfied: cholespy in /home/nroussel/.pyenv/versions/3.10.4/lib/python3.10/site-packages (2.1.0)
WARNING: You are using pip version 22.0.4; however, version 24.3.1 is available.
You should consider upgrading via the '/home/nroussel/.pyenv/versions/3.10.4/bin/python3.10 -m pip install --upgrade pip' command.
```
Copy to clipboard
The `LargeSteps` helper is instanciated from the starting shapes’ vertex positions and faces. If the mesh has duplicate vertices (e.g. due to face normals or UV seams), it will internally convert the mesh to a “unique” representation.
Copy to clipboard
```
lambda_ = 25
ls = mi.ad.LargeSteps(params['shape.vertex_positions'], params['shape.faces'], lambda_)

```
Copy to clipboard
We also use a slightly modified version of the [Adam](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.ad.Adam) optimizer, that uses a uniform second moment for all parameters. This can be done by setting `uniform=True` when instantiating the optimizer.
Copy to clipboard
```
opt = mi.ad.Adam(lr=1e-1, uniform=True)

```
Copy to clipboard
The `LargeSteps` class has a couple utility methods that can be used to convert back and forth between cartesian and differential representation of the vertex coordinates. The `LargeSteps.to_differential` method is used here to initialize the latent variable.
Copy to clipboard
```
opt['u'] = ls.to_differential(params['shape.vertex_positions'])

```
Copy to clipboard
The optimisation loop must also be slighlty changed. We now need to update the shape using the latent variable, this additional step can be done by using `LargeSteps.from_differential`.
Copy to clipboard
```
iterations = 100 if 'PYTEST_CURRENT_TEST' not in os.environ else 5
for it in range(iterations):
    total_loss = mi.Float(0.0)

    for sensor_idx in range(sensor_count):
        # Retrieve the vertex positions from the latent variable
        params['shape.vertex_positions'] = ls.from_differential(opt['u'])
        params.update()

        img = mi.render(scene_source, params, sensor=sensors[sensor_idx], seed=it)

        # L1 Loss
        loss = dr.mean(dr.abs(img - ref_images[sensor_idx]))

        dr.backward(loss)
        opt.step()

        total_loss += loss

    print(f"Iteration {1+it:03d}: Loss = {total_loss}", end='\r')

```
Copy to clipboard
```
Iteration 100: Loss = [0.039961]]

```
Copy to clipboard
## Intermediate result[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html#Intermediate-result "Link to this heading")
Copy to clipboard
```
# Update the mesh after the last iteration's gradient step
params['shape.vertex_positions'] = ls.from_differential(opt['u'])
params.update();

```
Copy to clipboard
Let us again plot the result of our optimization.
Copy to clipboard
```
final_imgs = [mi.render(scene_source, sensor=sensors[i], spp=128) for i in range(sensor_count)]
plot_images(final_imgs)

```
Copy to clipboard
![../../_images/src_inverse_rendering_shape_optimization_36_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_shape_optimization_36_0.png)
## Remeshing[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html#Remeshing "Link to this heading")
The result above can be further improved with the help of remeshing. By increasing the tesselation of the mesh, we will be able to recover more details of the target shape. Intuitively, the intent of this step is similar to other “coarse-to-fine” optimization strategies. For example, in the [caustics](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html) or the [volume optimization](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html) tutorial we increase the resolution of texture that is being optimized over time.
We will use the Botsch-Kobbelt remeshing algorithm provided by the `gpytoolbox` package:
Copy to clipboard
```
!pip;

```
Copy to clipboard
```
Requirement already satisfied: gpytoolbox in /home/nroussel/.pyenv/versions/3.10.4/lib/python3.10/site-packages (0.3.2)
Requirement already satisfied: numpy<2 in /home/nroussel/.pyenv/versions/3.10.4/lib/python3.10/site-packages (from gpytoolbox) (1.24.0)
Requirement already satisfied: scikit-image in /home/nroussel/.pyenv/versions/3.10.4/lib/python3.10/site-packages (from gpytoolbox) (0.24.0)
Requirement already satisfied: scipy in /home/nroussel/.pyenv/versions/3.10.4/lib/python3.10/site-packages (from gpytoolbox) (1.14.0)
Requirement already satisfied: tifffile>=2022.8.12 in /home/nroussel/.pyenv/versions/3.10.4/lib/python3.10/site-packages (from scikit-image->gpytoolbox) (2024.8.24)
Requirement already satisfied: networkx>=2.8 in /home/nroussel/.pyenv/versions/3.10.4/lib/python3.10/site-packages (from scikit-image->gpytoolbox) (3.3)
Requirement already satisfied: imageio>=2.33 in /home/nroussel/.pyenv/versions/3.10.4/lib/python3.10/site-packages (from scikit-image->gpytoolbox) (2.35.1)
Requirement already satisfied: lazy-loader>=0.4 in /home/nroussel/.pyenv/versions/3.10.4/lib/python3.10/site-packages (from scikit-image->gpytoolbox) (0.4)
Requirement already satisfied: pillow>=9.1 in /home/nroussel/.pyenv/versions/3.10.4/lib/python3.10/site-packages (from scikit-image->gpytoolbox) (10.4.0)
Requirement already satisfied: packaging>=21 in /home/nroussel/.pyenv/versions/3.10.4/lib/python3.10/site-packages (from scikit-image->gpytoolbox) (24.1)
WARNING: You are using pip version 22.0.4; however, version 24.3.1 is available.
You should consider upgrading via the '/home/nroussel/.pyenv/versions/3.10.4/bin/python3.10 -m pip install --upgrade pip' command.
```
Copy to clipboard
Since the `gpytoolbox` package expects NumPy arrays, we will first convert the mesh data to the correct format.
Copy to clipboard
```
importnumpyasnp
v_np = params['shape.vertex_positions'].numpy().reshape((-1,3)).astype(np.float64)
f_np = params['shape.faces'].numpy().reshape((-1,3))

```
Copy to clipboard
The Botsch-Kobbelt remeshing algorithm takes a “target edge length” as input argument. This controls the desired tesselation of the mesh. Since we want to increase resolution, we will set this as half of the mean edge length of the current mesh.
Copy to clipboard
```
# Compute average edge length
l0 = np.linalg.norm(v_np[f_np[:,0]] - v_np[f_np[:,1]], axis=1)
l1 = np.linalg.norm(v_np[f_np[:,1]] - v_np[f_np[:,2]], axis=1)
l2 = np.linalg.norm(v_np[f_np[:,2]] - v_np[f_np[:,0]], axis=1)

target_l = np.mean([l0, l1, l2]) / 2

```
Copy to clipboard
We can now run the Botsch-Kobbelt remeshing algorithm. It runs for a user-specified number of iterations, which we set to 5 here. Further details about this algorithm can be found it the package’s [documentation](https://gpytoolbox.org/0.1.0/remesh_botsch/).
Copy to clipboard
```
fromgpytoolboximport remesh_botsch

v_new, f_new = remesh_botsch(v_np, f_np, i=5, h=target_l, project=True)

```
Copy to clipboard
The new vertices and faces must now be passed to our Mitsuba `Mesh`. If the mesh has other attributes (e.g. UV coordinates), they also need to be updated. By default, Mitsuba will reset these to 0 if the vertex or face count is altered.
Copy to clipboard
```
params['shape.vertex_positions'] = mi.Float(v_new.flatten().astype(np.float32))
params['shape.faces'] = mi.Int(f_new.flatten())
params.update();

```
Copy to clipboard
Since the mesh topology has changed, we also need to compute a new latent variable.
Copy to clipboard
```
ls = mi.ad.LargeSteps(params['shape.vertex_positions'], params['shape.faces'], lambda_)
opt = mi.ad.Adam(lr=1e-1, uniform=True)
opt['u'] = ls.to_differential(params['shape.vertex_positions'])

```
Copy to clipboard
We had disabled indirect visibility derivatives so far, as they weren’t really needed. However, the mesh now does shadow itself a fair amount, indirect visibility effects became more important and may help with optimizing the last few details of the shape.
Copy to clipboard
```
integrator = mi.load_dict({
    'type': 'direct_projective'
})

```
Copy to clipboard
Let’s continue the optimization.
Copy to clipboard
```
iterations = 100 if 'PYTEST_CURRENT_TEST' not in os.environ else 5
for it in range(iterations):
    total_loss = mi.Float(0.0)
    for sensor_idx in range(sensor_count):
        # Retrieve the vertex positions from the latent variable
        params['shape.vertex_positions'] = ls.from_differential(opt['u'])
        params.update()

        img = mi.render(scene_source, params, sensor=sensors[sensor_idx], seed=it, integrator=integrator)

        # L1 Loss
        loss = dr.mean(dr.abs(img - ref_images[sensor_idx]))
        dr.backward(loss)
        total_loss += loss
    opt.step()

    print(f"Iteration {1+it:03d}: Loss = {total_loss}", end='\r')

```
Copy to clipboard
```
Iteration 100: Loss = [0.0142433]

```
Copy to clipboard
## Final result[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html#Final-result "Link to this heading")
We recover the final state from the latent variable:
Copy to clipboard
```
params['shape.vertex_positions'] = ls.from_differential(opt['u'])
params.update();

```
Copy to clipboard
Finally, let’s compare our end result (bottom) to our reference views (top).
Copy to clipboard
```
final_imgs = [mi.render(scene_source, sensor=sensors[i], spp=128) for i in range(sensor_count)]

fig, ax = plt.subplots(nrows=2, ncols=sensor_count, figsize=(5*sensor_count, 10))
ax[0][0].set_title("Reference", y=0.3, x=-0.1, rotation=90, fontsize=20)
ax[1][0].set_title("Optimized shape", y=0.2, x=-0.1, rotation=90, fontsize=20)
for i in range(sensor_count):
    ax[0][i].imshow(mi.util.convert_to_bitmap(ref_images[i]))
    ax[0][i].axis('off')
    ax[1][i].imshow(mi.util.convert_to_bitmap(final_imgs[i]))
    ax[1][i].axis('off')

```
Copy to clipboard
![../../_images/src_inverse_rendering_shape_optimization_58_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_shape_optimization_58_0.png)
Note that the results could be further improved by e.g. using more input views, or using a less agressive step size and more iterations.
## See also[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html#See-also "Link to this heading")
  * [mitsuba.ad.LargeSteps](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.ad.LargeSteps)
  * [direct_projective plugin](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_integrators.html#direct-illumination-projective-sampling-direct-projective)


* * *
