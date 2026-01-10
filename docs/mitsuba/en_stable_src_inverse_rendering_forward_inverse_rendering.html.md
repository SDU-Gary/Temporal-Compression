---
url: https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html
crawled_at: 2025-11-13T17:10:59.708862
title: https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Download notebook ](https://raw.githubusercontent.com/mitsuba-renderer/mitsuba-tutorials/master/inverse_rendering/forward_inverse_rendering.ipynb) [ Download data ](https://d38rqfq1h7iukm.cloudfront.net/scenes/tutorials/scenes.zip)
# Forward inverse rendering[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html#Forward-inverse-rendering "Link to this heading")
## Overview[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html#Overview "Link to this heading")
The previous example demonstrated _reverse-mode differentiation_ (a.k.a. backpropagation) where a desired small change to the output image was converted into a small change to the scene parameters. Mitsuba and Dr.Jit can also propagate derivatives in the other direction, i.e., from input parameters to the output image. This technique, known as _forward mode differentiation_ , is typically less suitable for optimization, as the contribution from each parameter must be handled using a separate rendering pass. That said, this mode can be very educational since it enables visualizations of the effect of individual scene parameters on the rendered image.
🚀 **You will learn how to:**
  * Manually set scene parameters as differentiable
  * Perform forward-mode differentiation with Dr.Jit
  * Visualize gradient images with matplotlib


## Setup[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html#Setup "Link to this heading")
We start by setting an AD-compatible variant (here `llvm_ad_rgb`) and load the Cornell Box scene.
Copy to clipboard
```
importdrjitasdr
importmitsubaasmi

mi.set_variant('llvm_ad_rgb')

scene = mi.load_file('../scenes/cbox.xml')

```
Copy to clipboard
## Preparing the scene[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html#Preparing-the-scene "Link to this heading")
Forward mode differentiable rendering begins analogously to reverse mode, by marking the parameters of interest as differentiable (in this example, we do so manually instead of using an `Optimizer`).
Our goal here is to visualize how changes of the green wall’s color affect the final rendered image. Note that we are rendering this image using a physically-based _path tracer_ , which means that it accounts for globlal illumination, reflection, refraction, and so on. Gradients computated from this simulation will also expose such effects.
Copy to clipboard
```
params = mi.traverse(scene)

key = 'green.reflectance.value'

# Mark the green wall color parameter as differentiable
dr.enable_grad(params[key])

# Propagate this change to the scene internal state
params.update();

```
Copy to clipboard
## Rendering[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html#Rendering "Link to this heading")
We can then perform the simulation to be differentiated. In this case, we simply render an image using the `mi.render()` routine, which will in turn call the scene’s path tracer integrator.
As we have marked the wall color as _differentiable_ , its role in the rendering process is recorded in the autodiff graph.
Copy to clipboard
```
image = mi.render(scene, params, spp=128)

```
Copy to clipboard
The [dr.forward()](https://drjit.readthedocs.io/en/latest/reference.html#drjit.forward) function will assign a gradient value of `1.0` to the given variables and forward-propagate those gradients through the previously recorded computation graph. During this process, gradient will be accumulated in the output nodes of this graph (here, the rendered image). Finally, the gradients can be read using [dr.grad()](https://drjit.readthedocs.io/en/latest/reference.html#drjit.grad).
For more detailed information about differentiation with DrJit, please refer to the [documentation](https://drjit.readthedocs.io/en/latest).
Copy to clipboard
```
# Forward-propagate gradients through the computation graph
dr.forward(params[key])

# Fetch the image gradient values
grad_image = dr.grad(image)

```
Copy to clipboard
## Visualizing the gradient image[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html#Visualizing-the-gradient-image "Link to this heading")
The gradient value of the `image` variable will share the same type (here `TensorXf`) hence it can easily be visualized as any other image. When displayin the gradient image below, we boost the gain a bit to better see the _global effect_ of the wall color on the rest of the scene.
Copy to clipboard
```
importmatplotlib.pyplotasplt
plt.imshow(grad_image * 2.0)
plt.axis('off');

```
Copy to clipboard
```
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

```
Copy to clipboard
![../../_images/src_inverse_rendering_forward_inverse_rendering_11_1.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_forward_inverse_rendering_11_1.png)
Note however that gradient values are not necessarily within the `[0, 1]` range, and so it makes more sense to use a color map and visualize each color channel of the gradient image individually.
Copy to clipboard
```
frommatplotlibimport pyplot as plt
importmatplotlib.cmascm

cmap = cm.coolwarm
vlim = dr.max(dr.abs(grad_image).array)[0]
print(f'Remapping colors within range: [{-vlim:.2f}, {vlim:.2f}]')

fig, axx = plt.subplots(1, 3, figsize=(8, 3))
for i, ax in enumerate(axx):
    ax.imshow(grad_image[..., i], cmap=cm.coolwarm, vmin=-vlim, vmax=vlim)
    ax.set_title('RGB'[i] + ' gradients')
    ax.axis('off')
fig.tight_layout()
plt.show()

```
Copy to clipboard
```
Remapping colors within range: [-2.79, 2.79]

```
Copy to clipboard
![../../_images/src_inverse_rendering_forward_inverse_rendering_13_1.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_forward_inverse_rendering_13_1.png)
## Using latent variables (advanced)[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html#Using-latent-variables-\(advanced\) "Link to this heading")
In more complex scenarios, scene parameters might themselves be the result of differentiable computations, then depending on other _latent variables_. For instance, the color of the CBox wall could be the result of the evalutation of a neural network, or some other procedural process.
In this case, it is important for the gradients of those latent variables to be propagate to the scene parameters **before** calling `mi.render()`. This way, the computation happing inside the renderer will only be responsible for propagating the gradients further to the output image. Failing to do so will result in uncorrect gradient as the loop tracing will destroy the part of the computational graph that took place outside of the loop (e.g. here linking the latent variable to the scene parameters).
In this example, we are simply going to parameterize the wall color using simple arithmetics and forward propagate the gradients properly.
Copy to clipboard
```
# Our latent variable
theta = mi.Float(0.5)
dr.enable_grad(theta)

# The wall color now depends on `theta`
params[key] = mi.Color3f(
    0.2 * theta,
    0.5 * theta,
    0.8 * theta
)

# Propagate this change to the scene internal state
params.update();

```
Copy to clipboard
Dr.Jit exposes various [dr.ADFlag](https://drjit.readthedocs.io/en/latest/reference.html#drjit.traverse) flags to control how the AD graph should be affected by the AD traversal. [dr.ADFlag.ClearEdges](https://drjit.readthedocs.io/en/latest/reference.html#drjit.traverse) specifies that gradient values should be kept in all variables during the traversal. This is to handle the case where a scene parameter depends on another scene parameter. Without this flag, the first scene parameter wouldn’t be considered as a _leaf_ node during the traversal and its gradient would be set to zero instead.
The following line propagates gradients from `theta` to the 3 channels of the `green.reflectance.value` scene parameters.
Copy to clipboard
```
dr.forward(theta, dr.ADFlag.ClearEdges)

```
Copy to clipboard
As done before, we can now render the image and forward progate the gradients from the scene parameters to the output image.
Unfortunately `dr.forward()` will overwrite the gradients of the provided variable, so it is necessary to use a different function in this situation. [dr.forward_to()](https://drjit.readthedocs.io/en/latest/reference.html#drjit.forward) automatically propagates gradients to a specified variable after finding all possible sources by inspecting the AD graph. It does so without overwritting the gradient value in those which is what we need in this context.
Copy to clipboard
```
image = mi.render(scene, params, spp=128)

# Forward-propagate the gradients to the image
dr.forward_to(image)

# Visualize the gradient image
mi.Bitmap(dr.grad(image))

```
Copy to clipboard
Copy to clipboard
![](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html)
## See also[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html#See-also "Link to this heading")
  * [mitsuba.ad.Optimizer](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.ad.Optimizer)
  * [prb plugin](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_integrators.html#path-replay-backpropagation-prb)
  * [drjit.forward()](https://drjit.readthedocs.io/en/latest/reference.html#drjit.forward)
  * [drjit.forward_to()](https://drjit.readthedocs.io/en/latest/reference.html#drjit.forward_to)


* * *
