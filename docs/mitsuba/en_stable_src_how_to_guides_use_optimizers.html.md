---
url: https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/use_optimizers.html
crawled_at: 2025-11-13T17:14:39.106526
title: https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/use_optimizers.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/use_optimizers.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/use_optimizers.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Download notebook ](https://raw.githubusercontent.com/mitsuba-renderer/mitsuba-tutorials/master/how_to_guides/use_optimizers.ipynb) [ Download data ](https://d38rqfq1h7iukm.cloudfront.net/scenes/tutorials/scenes.zip)
# Detailed look at `Optimizer`[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/use_optimizers.html#Detailed-look-at-Optimizer "Link to this heading")
In the [Gradient-based optimization](https://mitsuba.readthedocs.io/en/latest/src/inverse_rendering/gradient_based_opt.html) tutorial, Mitsuba’s `Optimizer` class was used to build an optimization loop. In this tutorial, we will study the API of optimizers in more detail. It is designed to be convenient to use when directly optimizing parameters in a Mitsuba scene, but also flexible enough for more flexible use-cases (e.g. chaining together a neural network and differentiable rendering step in the same computation graph).
Let’s start by setting an AD-compatible variant.
Copy to clipboard
```
importmitsubaasmi
importdrjitasdr

mi.set_variant('llvm_ad_rgb')

```
Copy to clipboard
## The basics[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/use_optimizers.html#The-basics "Link to this heading")
To perform [gradient-based optimization](https://en.wikipedia.org/wiki/Gradient_descent), Mitsuba ships with standard optimizers including _Stochastic Gradient Descent_ ([SGD](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.ad.SGD)) with and without momentum, as well as [Adam](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.ad.Adam). Those both inherit from the [Optimizer](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.ad.Optimizer) base class and can be found in the `mitsuba.ad` submodule.
The `Optimizer` class behaves like a Python `dict` with some extra logic and methods. This how-to guide will take you through its API and highlight the best-practices and pittfalls related to this class.
Let’s first construct a simple `SGD` optimizer with a learning rate of `0.25`.
Copy to clipboard
```
opt = mi.ad.SGD(lr=0.25)

```
Copy to clipboard
We can now specify a variable to be optimized. The `Optimizer` will automatically enable gradient computation on the stored variable as it is necessary for further computation to produce any gradients.
Copy to clipboard
```
opt['x'] = mi.Float(1.0)
opt

```
Copy to clipboard
Copy to clipboard
```
SGD[
  variables = ['x'],
  lr = {'default': 0.25},
  momentum = 0
]

```
Copy to clipboard
It is also possible to directly pass a set of variables to be optimized in the constructor of the `Optimizer`
Copy to clipboard
```
opt = mi.ad.SGD(lr=0.25, params={'x': mi.Float(1.0), 'y': mi.Float(2.0)})
opt

```
Copy to clipboard
Copy to clipboard
```
SGD[
  variables = ['x', 'y'],
  lr = {'default': 0.25},
  momentum = 0
]

```
Copy to clipboard
It also provides a similar API to perform basic dictionary manipulations.
Copy to clipboard
```
for k, v in opt.items():
    print(f"{k}: {v}")

```
Copy to clipboard
```
x: [1]
y: [2]

```
Copy to clipboard
⚠ It is important to note that a copy of the variable is made when assigned to the Optimizer via the `__setitem__` method. For instance in the following code, the original variable won’t be attached to the AD graph, and its value will remain unchanged.
Copy to clipboard
```
y = mi.Float(2.0)
opt['y'] = y  # A copy is made here
opt['y'] += 1.0

print(f"Original:  {y}, grad_enabled={dr.grad_enabled(y)}.")
print(f"Optimizer: {opt['y']}, grad_enabled={dr.grad_enabled(opt['y'])}.")

```
Copy to clipboard
```
Original:  [2], grad_enabled=False.
Optimizer: [3], grad_enabled=True.

```
Copy to clipboard
It is therefore crucial to use the variable held by the optimizer to perform the differentiable computation in order to produce the proper gradients. Here is a simple example where `x` and `y` are used in some calculation for which we then request the gradients to be backpropagated. We can validate that the gradients are adequately propagated to the optimizer variables.
Copy to clipboard
```
z = opt['x'] + 2.0 * opt['y']
dr.backward(z)

print(f"x grad={dr.grad(opt['x'])}")
print(f"y grad={dr.grad(opt['y'])}")

```
Copy to clipboard
```
x grad=[1]
y grad=[2]

```
Copy to clipboard
During the optimization, the role of the optimizer will be to take a gradient step according to its update rule. In the case of a simple SGD optimizer with no momentum, the update rule is:
\\[x_{i+1} = x_i - \texttt{grad}(x_i) \times \texttt{lr}\\]
The `Optimizer` method [step()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.ad.Optimizer.step) will apply its update rule to all the variables.
Copy to clipboard
```
print(f"Before the gradient step: x={opt['x']}, y={opt['y']}")
opt.step()
print(f"After the gradient step:  x={opt['x']}, y={opt['y']}")

```
Copy to clipboard
```
Before the gradient step: x=[1], y=[3]
After the gradient step:  x=[0.75], y=[2.5]

```
Copy to clipboard
After performing the update rule, the `Optimizer` also resets the gradient values of its variables to `0.0` and ensures gradient computations are still enabled on all its variables. This ensures that everything is ready for the next iteration of the optimization loop.
## Optimizing scene parameters[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/use_optimizers.html#Optimizing-scene-parameters "Link to this heading")
In the context of differentiable rendering, we are often interested in optimizing parameters of a Mitsuba scene, which are exposed via the ``traverse()` <<https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.traverse>>`__ mechanism. This function returns a ``SceneParameters` <<https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.SceneParameters>>`__ object which exposes dictionary-like interface.
Copy to clipboard
```
scene = mi.load_file('../scenes/cbox.xml')
params = mi.traverse(scene)
params

```
Copy to clipboard
Copy to clipboard
```
SceneParameters[
  ----------------------------------------------------------------------------------------------
  Name                                       Flags    Type           Parent
  ----------------------------------------------------------------------------------------------
  sensor.near_clip                                    float          PerspectiveCamera
  sensor.far_clip                                     float          PerspectiveCamera
  sensor.shutter_open                                 float          PerspectiveCamera
  sensor.shutter_open_time                            float          PerspectiveCamera
  sensor.film.size                                    ScalarVector2u HDRFilm
  sensor.film.crop_size                               ScalarVector2u HDRFilm
  sensor.film.crop_offset                             ScalarPoint2u  HDRFilm
  sensor.x_fov                               ∂, D     Float          PerspectiveCamera
  sensor.principal_point_offset_x            ∂, D     Float          PerspectiveCamera
  sensor.principal_point_offset_y            ∂, D     Float          PerspectiveCamera
  sensor.to_world                            ∂, D     Transform4f    PerspectiveCamera
  gray.reflectance.value                     ∂        Color3f        SRGBReflectanceSpectrum
  white.reflectance.value                    ∂        Color3f        SRGBReflectanceSpectrum
  green.reflectance.value                    ∂        Color3f        SRGBReflectanceSpectrum
  red.reflectance.value                      ∂        Color3f        SRGBReflectanceSpectrum
  glass.eta                                           float          SmoothDielectric
  mirror.eta.value                           ∂, D     Float          UniformSpectrum
  mirror.k.value                             ∂, D     Float          UniformSpectrum
  mirror.specular_reflectance.value          ∂        Float          UniformSpectrum
  light.emitter.sampling_weight                       float          AreaLight
  light.emitter.radiance.value               ∂        Color3f        SRGBReflectanceSpectrum
  light.silhouette_sampling_weight                    float          OBJMesh
  light.faces                                         UInt           OBJMesh
  light.vertex_positions                     ∂, D     Float          OBJMesh
  light.vertex_normals                       ∂, D     Float          OBJMesh
  light.vertex_texcoords                     ∂        Float          OBJMesh
  floor.silhouette_sampling_weight                    float          OBJMesh
  floor.faces                                         UInt           OBJMesh
  floor.vertex_positions                     ∂, D     Float          OBJMesh
  floor.vertex_normals                       ∂, D     Float          OBJMesh
  floor.vertex_texcoords                     ∂        Float          OBJMesh
  ceiling.silhouette_sampling_weight                  float          OBJMesh
  ceiling.faces                                       UInt           OBJMesh
  ceiling.vertex_positions                   ∂, D     Float          OBJMesh
  ceiling.vertex_normals                     ∂, D     Float          OBJMesh
  ceiling.vertex_texcoords                   ∂        Float          OBJMesh
  back.silhouette_sampling_weight                     float          OBJMesh
  back.faces                                          UInt           OBJMesh
  back.vertex_positions                      ∂, D     Float          OBJMesh
  back.vertex_normals                        ∂, D     Float          OBJMesh
  back.vertex_texcoords                      ∂        Float          OBJMesh
  greenwall.silhouette_sampling_weight                float          OBJMesh
  greenwall.faces                                     UInt           OBJMesh
  greenwall.vertex_positions                 ∂, D     Float          OBJMesh
  greenwall.vertex_normals                   ∂, D     Float          OBJMesh
  greenwall.vertex_texcoords                 ∂        Float          OBJMesh
  redwall.silhouette_sampling_weight                  float          OBJMesh
  redwall.faces                                       UInt           OBJMesh
  redwall.vertex_positions                   ∂, D     Float          OBJMesh
  redwall.vertex_normals                     ∂, D     Float          OBJMesh
  redwall.vertex_texcoords                   ∂        Float          OBJMesh
  mirrorsphere.silhouette_sampling_weight             float          Sphere
  mirrorsphere.to_world                      ∂, D     Transform4f    Sphere
  glasssphere.silhouette_sampling_weight              float          Sphere
  glasssphere.to_world                       ∂, D     Transform4f    Sphere
]

```
Copy to clipboard
This is very convinient as it can be directly passed to the `Optimizer` constructor to set all the scene parameters to be optimized. In this case the `Optimizer` will actually ignore all the variables that are not marked as differentiable (see the ∂ flag in the print of `params` object above).
However, it is sometimes preferable to only optimize a subset of the params, which can easily be achieved by filtering out items in the `params` object using the `keep` method. This method accepts both a list of keys or a REGEX.
Copy to clipboard
```
params.keep(r'.*\.reflectance\.value')
params

```
Copy to clipboard
Copy to clipboard
```
SceneParameters[
  ------------------------------------------------------------------------------
  Name                       Flags    Type    Parent
  ------------------------------------------------------------------------------
  gray.reflectance.value     ∂        Color3f SRGBReflectanceSpectrum
  white.reflectance.value    ∂        Color3f SRGBReflectanceSpectrum
  green.reflectance.value    ∂        Color3f SRGBReflectanceSpectrum
  red.reflectance.value      ∂        Color3f SRGBReflectanceSpectrum
]

```
Copy to clipboard
We can now constructor an Optimizer that will optimize those reflectance values:
Copy to clipboard
```
opt = mi.ad.SGD(lr=0.25, params=params)
opt

```
Copy to clipboard
Copy to clipboard
```
SGD[
  variables = ['gray.reflectance.value', 'white.reflectance.value', 'green.reflectance.value', 'red.reflectance.value'],
  lr = {'default': 0.25},
  momentum = 0
]

```
Copy to clipboard
Of course, as done before, it is also possible to start from an empty optimizer and set variables one by one from the `params` object.
Copy to clipboard
```
opt = mi.ad.SGD(lr=0.25)
opt['red.reflectance.value'] = params['red.reflectance.value']
opt

```
Copy to clipboard
Copy to clipboard
```
SGD[
  variables = ['red.reflectance.value'],
  lr = {'default': 0.25},
  momentum = 0
]

```
Copy to clipboard
As explained above, the loaded parameters will be copied internally, so any attempt to change their value in the optimizer will not directly be reflected in `params`.
Copy to clipboard
```
opt['red.reflectance.value'] *= 0.5

print(f"params:   {params['red.reflectance.value']}")
print(f"optimize: {opt['red.reflectance.value']}")

```
Copy to clipboard
```
params:   [[0.570068, 0.0430135, 0.0443706]]
optimize: [[0.285034, 0.0215067, 0.0221853]]

```
Copy to clipboard
In order to propagate those changes to `params` (and to the `Scene` itself), we use the [update()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.SceneParameters.update) method of `SceneParameters`. Internally this method will look for the variables keys of the optimizer matching with the ones in `params` and overwrite the corresponding values. Then it will update the internal scene state (e.g. re-build the BVH when some geometry data was modified).
Copy to clipboard
```
params.update(opt);

print(f"params:   {params['red.reflectance.value']}")
print(f"optimize: {opt['red.reflectance.value']}")

```
Copy to clipboard
```
params:   [[0.285034, 0.0215067, 0.0221853]]
optimize: [[0.285034, 0.0215067, 0.0221853]]

```
Copy to clipboard
## Optimizing latent variables[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/use_optimizers.html#Optimizing-latent-variables "Link to this heading")
In more complex optimization scenarios, scene parameters might be described **as a function** of some other parameters. In such a scenario, we would be interested in optimizing those other parameters instead of the scene parameters directly.
For example, this would be needed when generating the vertex positions of a mesh using a neural network: we’d want to optimize the weights of the neural network, not the vertex positions themselves. Another example could be a procedurally generated texture, maybe from a physically-based model that can be tweaked with a few parameters.
We refer to this type of “external” parameters as latent variables. Many of the design decisions of the [Optimizer](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.ad.Optimizer) class were made to support latent variables.
For a simpler example, let’s consider the case where we are aiming at optimizing the translation vector of a 3D mesh object in our scene. Even from an convexity standpoint, optimizing those three translation values will be much easier that having to optimize all the vertex positions simultaneously and hope for the best.
Let’s fetch the scene parameters again and initialize our optimizer one more time.
Copy to clipboard
```
params = mi.traverse(scene)
opt = mi.ad.SGD(lr=0.25)

```
Copy to clipboard
We can then append a latent variable to the optimizer, similar do what we did in the first section of this How-to Guide.
Copy to clipboard
```
opt['translation'] = mi.Vector3f(0, 0, 0)

```
Copy to clipboard
In this scenario, it will be the user’s responsability to propagate the changes of the latent variable to the scene parameters _every time the latent variable is updated_. To this end, it’s helpful to define a dedicated update function.
Note that the vertex positions on a mesh are stored in a linearized fashion in Mitsuba (`xyzxyzxyzxyz...`). In order to apply the translation, we first reorder the initial coordinates into 3D points. Once the translation has been applied, we need to write the new values back into the linearized array before the assignment to the scene parameters. For those operations, we use the `dr.unravel` and `dr.ravel` helper functions of DrJIT.
Copy to clipboard
```
# Copy or our vertex positions (and convert them to 3D points)
initial_vertex_pos = dr.unravel(mi.Point3f, params['redwall.vertex_positions'])

# Now we define the update rule
defupdate_vertex_pos():
    # Create the translation transformation
    T = mi.Transform4f.translate(opt['translation'])

    # Apply the transformation to the vertex positions
    new_vertex_pos = T @ initial_vertex_pos

    # Flatten the vertex position array before assigning it to `params`
    params['redwall.vertex_positions'] = dr.ravel(new_vertex_pos)\

    # Propagate changes through the scene (e.g. rebuild BVH)
    params.update()

```
Copy to clipboard
With this function implemented, all we need to do is to make sure to call it after every optimizer step to update the vertex positions accordingly.
💭 On top of propagating the changes of values, calling `update_vertex_pos()` also builds the computational graph necessary to the automatic differentiation layer to later compute the gradients of the scene parameters with respect to the latent variable.
## Optimizer state[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/use_optimizers.html#Optimizer-state "Link to this heading")
On top of carrying variables to optimize, the [Optimizer](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.ad.Optimizer) also holds an internal state used in its update rule. For instance, the momentum-based [SGD](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.ad.SGD) optimizer tracks the velocity of the previous iteration to apply the momentum in its [step()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.ad.Optimizer.step) method. In most cases this state is stored on a per-parameter basis.
In some cases, it is useful to reset the state of an `Optimizer` (e.g. optimization scheduling that resizes the optimized volume grid). For this, we can use the [reset()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.ad.Optimizer.reset) method which will zero-initialize the internal state associated with a specific parameter.
Copy to clipboard
```
opt.reset('translation')

```
Copy to clipboard
Another useful feature of the Mitsuba `Optimizer` is its ability to mask the update of its internal state based on the presence or not of gradients w.r.t. each variables.
This can be useful in Monte Carlo simulations such as differentiable path tracing, where not all optimized parameters necessarily receive gradients at each iteration. We found that in this situation, updating the state of the optimizer for parameters that did not receive gradients may degrade convergence. The update should rather be discarded instead. When constructing the `Optimizer`, it is possible to specific whether to discard such updates when the gradients are zero using the `mask_updates` argument (the default is `False`).
Copy to clipboard
```
opt = mi.ad.SGD(lr=0.25, mask_updates=True)

```
Copy to clipboard
Finally, the Mitsuba implementation of the `SGD` optimizers also supports per-parameter learning rates. This is useful to control the magnitude of the gradient step taken on individual parameters. This can be achieved using the `set_learning_rate()` method, which takes a optional `key` argument to specify for which parameter to set the learning rate, or alternatively a `dict` with all keys for which to override the learning rate.
Copy to clipboard
```
opt['x'] = mi.Float(1.0)
opt['y'] = mi.Float(1.0)

opt.set_learning_rate({'x': 0.125, 'y': 0.25})

```
Copy to clipboard
[**MongoDB Atlas is the global, multi-cloud** database developers prefer for building modern apps.](https://server.ethicalads.io/proxy/click/9510/019a7c7e-a895-7da2-8e44-7077ad3966a6/)
[Ads by EthicalAds](https://www.ethicalads.io/advertisers/?ref=ea-text)
* * *
