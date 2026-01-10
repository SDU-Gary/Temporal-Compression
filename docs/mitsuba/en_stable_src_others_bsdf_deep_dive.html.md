---
url: https://mitsuba.readthedocs.io/en/stable/src/others/bsdf_deep_dive.html
crawled_at: 2025-11-13T17:16:56.958270
title: https://mitsuba.readthedocs.io/en/stable/src/others/bsdf_deep_dive.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/others/bsdf_deep_dive.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/others/bsdf_deep_dive.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Download notebook ](https://raw.githubusercontent.com/mitsuba-renderer/mitsuba-tutorials/master/others/bsdf_deep_dive.ipynb) [ Download data ](https://d38rqfq1h7iukm.cloudfront.net/scenes/tutorials/scenes.zip)
# Deep dive into a BSDF[¶](https://mitsuba.readthedocs.io/en/stable/src/others/bsdf_deep_dive.html#Deep-dive-into-a-BSDF "Link to this heading")
## Overview[¶](https://mitsuba.readthedocs.io/en/stable/src/others/bsdf_deep_dive.html#Overview "Link to this heading")
As you have probably already discovered, Mitsuba 3 can do much more than rendering. In this tutorial we will show how to instantiate a BSDF plugin using Python dictionaries and plot its distribution function using `matplotlib`.
🚀 **You will learn how to:**
  * Instanciate Mitsuba objects using Python dict and mitsuba.load_dict
  * Perform vectorized computations using a JIT variant of Mitsuba


## Setup[¶](https://mitsuba.readthedocs.io/en/stable/src/others/bsdf_deep_dive.html#Setup "Link to this heading")
Of course, let’s start with the usual Python imports! As emphasized in previous tutorials, Mitsuba requires a specific variant to be set before performing any other imports or computations. For this tutorial, we are going to use one of the JIT vectorized variant of the system. This will allow us to write code as if it was operating on normal scalar values, and have it run on arbitrary-sized arrays of values on the CPU or GPU.
Copy to clipboard
```
importdrjitasdr
importmitsubaasmi

mi.set_variant('llvm_ad_rgb')

```
Copy to clipboard
## Instantiating a BSDF[¶](https://mitsuba.readthedocs.io/en/stable/src/others/bsdf_deep_dive.html#Instantiating-a-BSDF "Link to this heading")
One easy way to instanciate Mitsuba objects (e.g., [Shape](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_shapes.html), [BSDF](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html), …) is using the [load_dict](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.load_dict) function. This function takes as input a Python `dict` following a similar structure to the XML scene description and instantiates the corresponding plugin. You can learn more about the specific format of this `dict` by reading the dedicated section in the [documentation](https://mitsuba.readthedocs.io/en/latest/src/key_topics/scene_format.html#scene-python-dict-format).
In this scenario, we want to construct a [roughconductor BSDF](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#rough-conductor-material-roughconductor) with a high roughness value and a GGX microfacet distribution.
Copy to clipboard
```
bsdf = mi.load_dict({
    'type': 'roughconductor',
    'alpha': 0.2,
    'distribution': 'ggx'
})

```
Copy to clipboard
## Vectorized evaluation of the BSDF[¶](https://mitsuba.readthedocs.io/en/stable/src/others/bsdf_deep_dive.html#Vectorized-evaluation-of-the-BSDF "Link to this heading")
We will now evaluate this BSDF for a whole array of directions at once, leveraging the enabled vectorize backend. Similarly to working on `numpy` arrays, we use DrJit routines to perform array-based arithmetics.
For instance, here we start by defining a function to map from spherical and Euclidean coordinates.
Copy to clipboard
```
defsph_to_dir(theta, phi):
"""Map spherical to Euclidean coordinates"""
    st, ct = dr.sincos(theta)
    sp, cp = dr.sincos(phi)
    return mi.Vector3f(cp * st, sp * st, ct)

```
Copy to clipboard
We can then use this function to generate a set of directions to evaluate the BSDF with.
Copy to clipboard
```
# Create a (dummy) surface interaction to use for the evaluation of the BSDF
si = dr.zeros(mi.SurfaceInteraction3f)

# Specify an incident direction with 45 degrees elevation
si.wi = sph_to_dir(dr.deg2rad(45.0), 0.0)

# Create grid in spherical coordinates and map it onto the sphere
res = 300
theta_o, phi_o = dr.meshgrid(
    dr.linspace(mi.Float, 0,     dr.pi,     res),
    dr.linspace(mi.Float, 0, 2 * dr.pi, 2 * res)
)
wo = sph_to_dir(theta_o, phi_o)

# Evaluate the whole array (18000 directions) at once
values = bsdf.eval(mi.BSDFContext(), si, wo)

```
Copy to clipboard
## Plotting the results[¶](https://mitsuba.readthedocs.io/en/stable/src/others/bsdf_deep_dive.html#Plotting-the-results "Link to this heading")
Dr.Jit arrays of any flavour can easily be converted to an array type of other mainstream libraries, such as `numpy`, `PyTorch`, `JAX` and `TensorFlow`. For more detailed information on this, take a look at the extensive [drjit documentation](https://drjit.readthedocs.io/en/master/). In our case we are going to convert our drjit array to a numpy array.
Copy to clipboard
```
importnumpyasnp
values_np = np.array(values)

```
Copy to clipboard
We can now use our favourite plotting library to visualize the BSDF distribution (here we use `matplotlib`).
Copy to clipboard
```
importmatplotlib.pyplotasplt

# Extract red channel of BRDF values and reshape into 2D grid
values_r = values_np[0, :]
values_r = values_r.reshape(2 * res, res).T

# Plot values for spherical coordinates
fig, ax = plt.subplots(figsize=(8, 4))

im = ax.imshow(values_r, extent=[0, 2 * np.pi, np.pi, 0], cmap='jet')

ax.set_xlabel(r'$\phi_o$', size=10)
ax.set_xticks([0, dr.pi, dr.two_pi])
ax.set_xticklabels(['0', '$\\pi$', '$2\\pi$'])
ax.set_ylabel(r'$\theta_o$', size=10)
ax.set_yticks([0, dr.pi / 2, dr.pi])
ax.set_yticklabels(['0', '$\\pi/2$', '$\\pi$']);

```
Copy to clipboard
![../../_images/src_others_bsdf_deep_dive_13_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_others_bsdf_deep_dive_13_0.png)
## See also[¶](https://mitsuba.readthedocs.io/en/stable/src/others/bsdf_deep_dive.html#See-also "Link to this heading")
  * [mitsuba.load_dict()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.load_dict)
  * [mitsuba.BSDF.eval()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.BSDF.eval)


* * *
