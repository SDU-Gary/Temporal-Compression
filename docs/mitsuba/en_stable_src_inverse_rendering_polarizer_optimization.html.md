---
url: https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html
crawled_at: 2025-11-13T17:16:31.306007
title: https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Download notebook ](https://raw.githubusercontent.com/mitsuba-renderer/mitsuba-tutorials/master/inverse_rendering/polarizer_optimization.ipynb) [ Download data ](https://d38rqfq1h7iukm.cloudfront.net/scenes/tutorials/scenes.zip)
# Polarizer optimization[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html#Polarizer-optimization "Link to this heading")
## Overview[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html#Overview "Link to this heading")
An interesting feature of Mitsuba is its ability to account for the polarization state of light. This becomes even more powerful when combined with differentiable rendering.
This tutorial demonstrates how those two concepts can be used together to perform a simple optimization. The setup is the following: we place two linear polarization filters in front of the camera. Initially, these are rotated in such a way that all the light passes through them. The optimization process will attempt to rotate one of the filter to minimize the overall brightness of the rendered image. Indeed, it is known that rotating this filter by 90 degrees will lead to complete cancelation of the polarization state, resulting in a darker image.
More information about polarization can be found [here](https://mitsuba.readthedocs.io/en/latest/src/key_topics/polarization.html).
🚀 **You will learn how to:**
  * Employ differentiable rendering in the context of polarized rendering
  * Optimize latent variables to control the rotation of an object


## Reference image[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html#Reference-image "Link to this heading")
As usual, let’s import the necessary libraries. For the sake of this tutorial, we already provide an XML file for the scene containing both linear polarization filter (e.g. using the [polarizer](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#linear-polarizer-material-polarizer) BSDF).
Copy to clipboard
```
importdrjitasdr
importmitsubaasmi

mi.set_variant('llvm_ad_rgb_polarized')

scene = mi.load_file('../scenes/polarizers.xml')

```
Copy to clipboard
We can then perform the rendering of our initial scene. As expected, the two filters are aligned and let linearly polarized light through.
Copy to clipboard
```
image_init = mi.render(scene, spp=8)

mi.util.convert_to_bitmap(image_init)

```
Copy to clipboard
Copy to clipboard
![](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html)
## Setup optimization[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html#Setup-optimization "Link to this heading")
As in the previous tutorial on [pose estimation](https://mitsuba.readthedocs.io/en/latest/src/inverse_rendering/reparam_optimization.html), we setup the optimization using a latent variable to control the rotation of the filter. This rotation angle will be used to construct a transformation matrix that will be applied to all vertices of the filter’s mesh. For convenience, we define a function that does all of this, which we will also call later during the optimization loop.
It is important to apply the rotation once before starting the optimization loop as this will _bind_ the optimizer variable to the scene parameter. Otherwise during backpropagation the gradients wouldn’t be propagate all the way to the optimizer’s variable.
Copy to clipboard
```
params = mi.traverse(scene)

# Key of the scene parameter to be optimized
key = 'filter2.vertex_positions'

# Get the initial vertex positions
v_positions_init = dr.unravel(mi.Vector3f, params[key])

# Instantiate an Adam optimizer and define a latent variable `rotation`
opt = mi.ad.Adam(lr=1.0)
opt['rotation'] = mi.Float(0.0)

# Apply optimized rotation value to mesh vertices
defapply_rotation():
    transform = mi.Transform4f().rotate([0, 0, 1], opt['rotation'])
    positions_new = transform @ v_positions_init
    params[key] = dr.ravel(positions_new)
    params.update()

# Perform the first rotation to enable derivative tracking on the scene parameters
apply_rotation()

```
Copy to clipboard
## Optimization[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html#Optimization "Link to this heading")
Everything is now ready to run the optimization loop.
In the following cell we define the hyper parameters controlling our optimization loop, such as the number of iterations:
Copy to clipboard
```
iteration_count = 100

```
Copy to clipboard
In this example the loss function doesn’t compare against a reference as the goal is simply to make the image darker. For this we simply use the sum of the pixel values in the rendered image.
Copy to clipboard
```
angles = []
losses = []

for it in range(iteration_count):
    # Perform the differentiable rendering simulation
    image = mi.render(scene, params=params, seed=it, spp=1)

    # Objective: no comparison against a reference, the goal is simply to make the image darker
    ob_val = dr.mean(image)

    # Backpropagate loss to input parameters
    dr.backward(ob_val)

    # Optimizer: take a gradient step
    opt.step()

    # Apply rotation and update the scene parameters
    apply_rotation()

    print(f"Iteration: {it:2}, rot: {opt['rotation'][0]:.4f}, loss: {ob_val}", end='\r')
    angles.append(opt['rotation'][0])
    losses.append(ob_val.array[0])

print()
print('Optimization complete!')

```
Copy to clipboard
```
Iteration: 99, rot: 90.0260, loss: 0.0816507
Optimization complete!

```
Copy to clipboard
## Results[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html#Results "Link to this heading")
We can now look at the optimized scene, which appears much darker as expected.
Copy to clipboard
```
image_final = mi.render(scene, seed=0, spp=8)

mi.util.convert_to_bitmap(image_final)

```
Copy to clipboard
Copy to clipboard
![](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html)
We plot the filter rotation and image loss accross the optimization loop.
Copy to clipboard
```
frommatplotlibimport pyplot as plt

fig, ax = plt.subplots(ncols=2, figsize=(12,4))

ax[0].plot(angles);
ax[0].set_ylabel('angle');
ax[0].set_title('Filter rotation');
ax[0].set_xlim([0, 99]);
ax[0].set_ylim([0, 100])

ax[1].plot(losses);
ax[1].set_title('Image loss')
ax[1].set_xlim([0, 99]);
ax[1].set_ylim([0.08, 0.11])

plt.show()

```
Copy to clipboard
![../../_images/src_inverse_rendering_polarizer_optimization_16_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_polarizer_optimization_16_0.png)
## See also[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html#See-also "Link to this heading")
  * [polarizer plugin](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#linear-polarizer-material-polarizer)


* * *
