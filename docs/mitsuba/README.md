# Documentation Index: mitsuba-docs

**Total Documents:** 61 files

## Quick Navigation

- [Getting Started](#getting-started) (9 docs)
- [Lights & Emitters](#lights--emitters) (2 docs)
- [Materials & BSDFs](#materials--bsdfs) (1 docs)
- [Rendering](#rendering) (11 docs)
- [Scene Format](#scene-format) (10 docs)
- [Tutorials](#tutorials) (28 docs)

---

## Getting Started

### [Custom Python plugin](./en_stable_src_others_custom_plugin.html.md)

**Summary:** In Mitsuba it easy to add custom code for BSDFs, integrators, emitters, sensors, and more. This tutorial will show you how to create custom plugins in Python and register them for use. To illustrate this, we are going to implement a new tinted die...

**Sections:** Overview, Setup, Implementation, Plugin registration, Plugin instantiation

**Keywords:** `
        self.eta = 1.33
        if props.has_property('eta'):
            self.eta = props['eta']

        self.tint = mi.color3f(props['tint'])

        # set the bsdf flags
        reflection_flags   = mi.bsdfflags.deltareflection   | mi.bsdfflags.frontside | mi.bsdfflags.backside
        transmission_flags = mi.bsdfflags.deltatransmission | mi.bsdfflags.frontside | mi.bsdfflags.backside
        self.m_components  = [reflection_flags, transmission_flags]
        self.m_flags = reflection_flags | transmission_flags

    defsample(self, ctx, si, sample1, sample2, active):
        # compute fresnel terms
        cos_theta_i = mi.frame3f.cos_theta(si.wi)
        r_i, cos_theta_t, eta_it, eta_ti = mi.fresnel(cos_theta_i, self.eta)
        t_i = dr.maximum(1.0 - r_i, 0.0)

        # pick between reflection and transmission
        selected_r = (sample1 <= r_i) & active

        # fill up the bsdfsample struct
        bs = mi.bsdfsample3f()
        bs.pdf = dr.select(selected_r, r_i, t_i)
        bs.sampled_component = dr.select(selected_r, mi.uint32(0), mi.uint32(1))
        bs.sampled_type      = dr.select(selected_r, mi.uint32(+mi.bsdfflags.deltareflection),
                                                     mi.uint32(+mi.bsdfflags.deltatransmission))
        bs.wo = dr.select(selected_r,
                          mi.reflect(si.wi),
                          mi.refract(si.wi, cos_theta_t, eta_ti))
        bs.eta = dr.select(selected_r, 1.0, eta_it)

        # for reflection, tint based on the incident angle (more tint at grazing angle)
        value_r = dr.lerp(mi.color3f(self.tint), mi.color3f(1.0), dr.clip(cos_theta_i, 0.0, 1.0))

        # for transmission, radiance must be scaled to account for the solid angle compression
        value_t = mi.color3f(1.0) * dr.square(eta_ti)

        value = dr.select(selected_r, value_r, value_t)

        return (bs, value)

    defeval(self, ctx, si, wo, active):
        return 0.0

    defpdf(self, ctx, si, wo, active):
        return 0.0

    defeval_pdf(self, ctx, si, wo, active):
        return 0.0, 0.0

    deftraverse(self, cb):
        cb.put('tint', self.tint, mi.paramflags.differentiable)

    defparameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

    defto_string(self):
        return ('mybsdf[\n'
                '    eta=%s,\n'
                '    tint=%s,\n'
                ']' % (self.eta, self.tint))

`, `
classmybsdf(mi.bsdf):
    def__init__(self, props):
        mi.bsdf.__init__(self, props)

        # read 'eta' and 'tint' properties from `, `
copy to clipboard
`, `
copy to clipboard
## implementation[¶](https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html#implementation "link to this heading")
as mentioned in the tutorial overview, we are going to implement a tinted dielectric bsdf in this tutorial. this code is very similar to the actual [c++ implementation](https://github.com/mitsuba-renderer/mitsuba3/blob/master/src/bsdfs/dielectric.cpp) of the [dielectric bsdf](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#smooth-dielectric-material-dielectric), so we will not look at it in great detail.
first, our bsdf python class `, `
copy to clipboard
## plugin instantiation[¶](https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html#plugin-instantiation "link to this heading")
you can now use this plugin like you would with any other bsdf plugin and set the appropriate properties of the bsdf expected in its constructor in the xml or `, `
copy to clipboard
## plugin registration[¶](https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html#plugin-registration "link to this heading")
there’s only one more thing to do before we can use our custom bsdf in scenes. we need to register it in the system so it can be used. this can be done by calling the [register_bsdf()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.register_bsdf) function and specifying the name to be used to instantiate this plugin. the function takes a _constructor lambda_ function as the second parameter.
📑 **note**
similar functions exist for other types of plugins, e.g.
  * register_integrator
  * register_emitter
  * register_sensor
  * register_film
  * register_mesh
  * register_texture
  * register_volume
  * register_phasefunction
  * register_medium
  * register_sampler


copy to clipboard
`, `
copy to clipboard
## rendering[¶](https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html#rendering "link to this heading")
finally, let’s use our custom bsdf in an actual scene and render it to see how our tinted bsdf looks like.
copy to clipboard
`, `
copy to clipboard
copy to clipboard
`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html

---

### [Differences to previous versions](./en_stable_src_key_topics_differences.html.md)

**Summary:** Porting Mitsuba 2 Python scripts and C++ plugins to Mitsuba 3 is straightforward as Mitsuba 3 only underwent minimal API changes. Moreover, Mitsuba 3 maintains scene compatibility with Mitsuba 2. One of the biggest changes is the replacement of th...

**Sections:** Differences to Mitsuba 2, Differences to Mitsuba 0.6

**Keywords:** `


in python dr.jit constants switch from camelcase to snake_case:
  * `, `
  * `, `
  * …


to learn more about dr.jit and its dissimilarities with enoki, see the [dr.jit documentation](https://readthedocs.org/projects/drjit/badge/?version=latest).
## differences to mitsuba 0.6[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/differences.html#differences-to-mitsuba-0-6 "link to this heading")
mitsuba 3 strives to retain scene compatibility with its predecessor mitsuba 0.6. however, in most other respects, it is a completely new system following a different set of goals.
### missing features[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/differences.html#missing-features "link to this heading")
a number of mitsuba 0.6 features are missing in mitsuba 3. we plan to port some of these features in the future and discard others. the following list provides an overview of the main missing features:
  * **shapes** : the basic shapes (ply/obj/serialized triangle meshes, rectangles, spheres, cylinders) are all supported. however, instancing and assemblies of hair fibers are still missing.
  * **integrators** : only surface and volumetric path tracers are provided. the following are missing:
    * **bidirectional path tracing**
    * **progressive photon mapping**
    * **path space metropolis light transport / manifold exploration / energy redistribution path tracing:**
path-space mcmc techniques were an incredibly complex component of the previous generation of mitsuba. we do not currently plan to port these and recommend that you stick with mitsuba 0.6 if your application depends on them.
  * **animation** : specification and handling of temporally varying transformations, e.g., for motion blur is not yet implemented.
  * **diffusion-based subsurface scattering** : currently missing, status undecided.


### scene format[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/differences.html#scene-format "link to this heading")
mitsuba 3’s xml scene format is almost identical to that of mitsuba 0.6. most plugins have the same name and same parameters, and we made sure that parameters behave in the same way as in mitsuba 0.6.
one significant change is that mitsuba 3 uses “`, `
# before
importenokiasek
... = ek.sin(...)

# now
importdrjitasdr
... = dr.sin(...)

`, `
// before
namespaceek=enoki;
...=ek::sin(...);

// now
namespacedr=drjit;
...=dr::sin(...);

`, `
<!-- old notation -->
<pointname="position"x="0"y="0"z="-100"/>

<!-- new notation -->
<pointname="position"value="0, 0, -100"/>

`, `
<sensortype="perspective">
<stringname="fov_axis"value="smaller"/>
<floatname="near_clip"value="10"/>
<floatname="far_clip"value="2800"/>
<floatname="focus_distance"value="1000"/>
<transformname="to_world">
<translatevalue="0, 0, -100"/>
</transform>
</sensor>

`, `
<sensortype="perspective">
<stringname="fovaxis"value="smaller"/>
<floatname="nearclip"value="10"/>
<floatname="farclip"value="2800"/>
<floatname="focusdistance"value="1000"/>
<transformname="toworld">
<translatex="0"y="0"z="-100"/>
</transform>
</sensor>

`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/key_topics/differences.html

---

### [Dr.Jit quickstart](./en_stable_src_quickstart_drjit_quickstart.html.md)

**Summary:** This short tutorial recaps the basic functionalities and routines of the Dr.Jit library. You can also find more information on the Dr.Jit documentation. On the Python side, the Dr.Jit syntax is very similar to NumPy. Moreover, as we will see later...

**Sections:** Overview, Similarity with NumPy, Array construction routines, Masking, Basic math arithmetic

**Keywords:** `
# initialize floating-point array of size 5 with zeros
a = dr.zeros(float, 5) # np.zeros(5)
print(f'dr.zeros: {a}')

# initialize floating-point array of size 5 with a constant value
a = dr.full(float, 0.1, 5) # np.ones(5, 0.4)
print(f'dr.full: {a}')

a = dr.arange(uint32, 5) # np.arange(5)
print(f'dr.arange: {a}')

# return evenly spaced numbers over a specified interval
a = dr.linspace(float, 0.0, 2.0, 5) # np.linspace(0.0, 2.0, 5)
print(f'dr.linespace: {a}')

`, `
a = dr.arange(float, 5) + 1
print(f'a: {a}')

# horizontal sum
b = dr.sum(a) # np.sum(a)
print(f'dr.sum(a): {b}')

# horizontal product
b = dr.prod(a) # np.prod(a)
print(f'dr.prod(a): {b}')

# mean value over the entire array
b = dr.mean(a) # np.mean(a)
print(f'dr.mean(a): {b}')

m = a > 2
print(f'm: {m}')

# true if all value of the mask array are true
b = dr.all(m) # np.all(m)
print(f'dr.all(m): {b}')

# true if any value of the mask array are true
b = dr.any(m) # np.any(m)
print(f'dr.any(m): {b}')

# true if no value of the mask array are true
b = dr.none(m) # ~np.any(m)
print(f'dr.none(m): {b}')

`, `
a: [1.0, 2.0, 3.0, 4.0, 5.0]
dr.sum(a): [15.0]
dr.prod(a): [120.0]
dr.mean(a): [3.0]
m: [false, false, true, true, true]
dr.all(m): false
dr.any(m): true
dr.none(m): false

`, `
c -> (<class 'drjit.llvm.float'>) = [9.0, 8.0, 7.0, 6.0]
d -> (<class 'numpy.ndarray'>) = [9. 8. 7. 6.]

`, `
copy to clipboard
`, `
copy to clipboard
##  `, `
copy to clipboard
## array construction routines[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html#array-construction-routines "link to this heading")
this section provides an overview of various dr.jit routines (and their numpy correspondence) to construct arrays.
copy to clipboard
`, `
copy to clipboard
## basic math arithmetic[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html#basic-math-arithmetic "link to this heading")
all common math operators like `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html

---

### [Getting started](./en_stable.md)

**Summary:** Mitsuba 3 is a research-oriented rendering system for forward and inverse light-transport simulation. It consists of a small set of core libraries and a wide variety of plugins that implement functionality ranging from materials and light sources ...

**Sections:** Installation, Hello World!, Quickstart, Video tutorials, Citation

**Keywords:** `
@software{jakob2022mitsuba3,
title={mitsuba 3 renderer},
author={wenzel jakob and sébastien speierer and nicolas roussel and merlin nimier-david and delio vicini and tizian zeltner and baptiste nicolet and miguel crespo and vincent leroy and ziyi zhang},
note={https://mitsuba-renderer.org},
version={3.0.1},
year=2022,
}

`, `
copy to clipboard
## quickstart[¶](https://mitsuba.readthedocs.io/en/stable/#quickstart "link to this heading")
for the new users, we put together absolute beginner’s tutorials for both dr.jit and mitsuba.
dr.jit quickstart
[![_images/drjit-logo.png](https://mitsuba.readthedocs.io/en/stable/_images/drjit-logo.png) ](https://mitsuba.readthedocs.io/en/stable/_images/drjit-logo.png)
[src/quickstart/drjit_quickstart.html](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html)
mitsuba quickstart
[![_images/mitsuba-logo.png](https://mitsuba.readthedocs.io/en/stable/_images/mitsuba-logo.png) ](https://mitsuba.readthedocs.io/en/stable/_images/mitsuba-logo.png)
[src/quickstart/mitsuba_quickstart.html](https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html)
## video tutorials[¶](https://mitsuba.readthedocs.io/en/stable/#video-tutorials "link to this heading")
the following [youtube playlist](https://www.youtube.com/playlist?list=pli9y-85z_po6da-pytngtns2n4fhpble5) contains various video tutorials related to mitsuba 3 and dr.jit, perfect to get you started with those two libraries.
## citation[¶](https://mitsuba.readthedocs.io/en/stable/#citation "link to this heading")
when using mitsuba 3 in academic projects, please cite:
`, `
importmitsubaasmi

mi.set_variant('scalar_rgb')

img = mi.render(mi.load_dict(mi.cornell_box()))

mi.bitmap(img).write('cbox.exr')

`, `getting`, `importmitsubaasmi`, `jakob2022mitsuba3`, `light`, `llvm >= 11.1`

**Source:** https://mitsuba.readthedocs.io/en/stable

---

### [Getting started](./index.md)

**Summary:** Mitsuba 3 is a research-oriented rendering system for forward and inverse light-transport simulation. It consists of a small set of core libraries and a wide variety of plugins that implement functionality ranging from materials and light sources ...

**Sections:** Installation, Hello World!, Quickstart, Video tutorials, Citation

**Keywords:** `
@software{jakob2022mitsuba3,
title={mitsuba 3 renderer},
author={wenzel jakob and sébastien speierer and nicolas roussel and merlin nimier-david and delio vicini and tizian zeltner and baptiste nicolet and miguel crespo and vincent leroy and ziyi zhang},
note={https://mitsuba-renderer.org},
version={3.0.1},
year=2022,
}

`, `
copy to clipboard
## quickstart[¶](https://mitsuba.readthedocs.io/en/stable/#quickstart "link to this heading")
for the new users, we put together absolute beginner’s tutorials for both dr.jit and mitsuba.
dr.jit quickstart
[![_images/drjit-logo.png](https://mitsuba.readthedocs.io/en/stable/_images/drjit-logo.png) ](https://mitsuba.readthedocs.io/en/stable/_images/drjit-logo.png)
[src/quickstart/drjit_quickstart.html](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html)
mitsuba quickstart
[![_images/mitsuba-logo.png](https://mitsuba.readthedocs.io/en/stable/_images/mitsuba-logo.png) ](https://mitsuba.readthedocs.io/en/stable/_images/mitsuba-logo.png)
[src/quickstart/mitsuba_quickstart.html](https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html)
## video tutorials[¶](https://mitsuba.readthedocs.io/en/stable/#video-tutorials "link to this heading")
the following [youtube playlist](https://www.youtube.com/playlist?list=pli9y-85z_po6da-pytngtns2n4fhpble5) contains various video tutorials related to mitsuba 3 and dr.jit, perfect to get you started with those two libraries.
## citation[¶](https://mitsuba.readthedocs.io/en/stable/#citation "link to this heading")
when using mitsuba 3 in academic projects, please cite:
`, `
importmitsubaasmi

mi.set_variant('scalar_rgb')

img = mi.render(mi.load_dict(mi.cornell_box()))

mi.bitmap(img).write('cbox.exr')

`, `getting`, `importmitsubaasmi`, `jakob2022mitsuba3`, `light`, `llvm >= 11.1`

**Source:** https://mitsuba.readthedocs.io/

---

### [Getting started](./en_stable_index.html.md)

**Summary:** Mitsuba 3 is a research-oriented rendering system for forward and inverse light-transport simulation. It consists of a small set of core libraries and a wide variety of plugins that implement functionality ranging from materials and light sources ...

**Sections:** Installation, Hello World!, Quickstart, Video tutorials, Citation

**Keywords:** `
@software{jakob2022mitsuba3,
title={mitsuba 3 renderer},
author={wenzel jakob and sébastien speierer and nicolas roussel and merlin nimier-david and delio vicini and tizian zeltner and baptiste nicolet and miguel crespo and vincent leroy and ziyi zhang},
note={https://mitsuba-renderer.org},
version={3.0.1},
year=2022,
}

`, `
copy to clipboard
## quickstart[¶](https://mitsuba.readthedocs.io/en/stable/index.html#quickstart "link to this heading")
for the new users, we put together absolute beginner’s tutorials for both dr.jit and mitsuba.
dr.jit quickstart
[![_images/drjit-logo.png](https://mitsuba.readthedocs.io/en/stable/_images/drjit-logo.png) ](https://mitsuba.readthedocs.io/en/stable/_images/drjit-logo.png)
[src/quickstart/drjit_quickstart.html](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html)
mitsuba quickstart
[![_images/mitsuba-logo.png](https://mitsuba.readthedocs.io/en/stable/_images/mitsuba-logo.png) ](https://mitsuba.readthedocs.io/en/stable/_images/mitsuba-logo.png)
[src/quickstart/mitsuba_quickstart.html](https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html)
## video tutorials[¶](https://mitsuba.readthedocs.io/en/stable/index.html#video-tutorials "link to this heading")
the following [youtube playlist](https://www.youtube.com/playlist?list=pli9y-85z_po6da-pytngtns2n4fhpble5) contains various video tutorials related to mitsuba 3 and dr.jit, perfect to get you started with those two libraries.
## citation[¶](https://mitsuba.readthedocs.io/en/stable/index.html#citation "link to this heading")
when using mitsuba 3 in academic projects, please cite:
`, `
importmitsubaasmi

mi.set_variant('scalar_rgb')

img = mi.render(mi.load_dict(mi.cornell_box()))

mi.bitmap(img).write('cbox.exr')

`, `getting`, `importmitsubaasmi`, `jakob2022mitsuba3`, `light`, `llvm >= 11.1`

**Source:** https://mitsuba.readthedocs.io/en/stable/index.html

---

### [Mitsuba quickstart](./en_stable_src_quickstart_mitsuba_quickstart.html.md)

**Summary:** In this tutorial, you will render your very first image using Mitsuba 3! 🚀 You will learn how to: Import Mitsuba in Python and set the “variant” Load a scene from disk Render a scene Write a rendered image to disk

**Sections:** Overview, Importing Mitsuba, Loading a scene, Rendering a scene, Writing an image to file

**Keywords:** `
['scalar_rgb', 'llvm_ad_rgb']

`, `
clipping input data to the valid range for imshow with rgb data ([0..1] for floats or [0..255] for integers).

`, `
copy to clipboard
`, `
copy to clipboard
![../../_images/src_quickstart_mitsuba_quickstart_14_1.png](https://mitsuba.readthedocs.io/en/stable/_images/src_quickstart_mitsuba_quickstart_14_1.png)
## writing an image to file[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html#writing-an-image-to-file "link to this heading")
the [mi.util.write_bitmap()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.util.write_bitmap) function allows to save an image to disk and supports multiple file formats. if a low dynamic range (ldr) format is selected (e.g., png), this function tonemaps the image to the srgb color space before saving.
copy to clipboard
`, `
copy to clipboard
## rendering a scene[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html#rendering-a-scene "link to this heading")
once loaded into memory, a scene can be rendered using the [render()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.render) function. the `, `
copy to clipboard
copy to clipboard
`, `
copy to clipboard
for this tutorial, we will use the simplest variant: `, `
copy to clipboard
the render function returns the generated image as a tensor ([mi.tensorxf](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.tensorxf), similar to a numpy array) in linear rgb color space. the tensor class interfaces seamlessly with functions expecting numpy arrays. for example, we can display the image using `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html

---

### [Projective sampling integrators](./en_stable_src_inverse_rendering_projective_sampling_integrators.html.md)

**Summary:** In this tutorial, we will optimize a curve such that its shadows match some target reference. We will do this by only observing the shadows and making use of the the projective sampling integrators. Gradients introduced by indirect visibilty effec...

**Sections:** Overview, Setup, Scene, Reference image, Projective sampling integrator

**Keywords:** `
# reset the circle to its initial position before starting the optimization.
params[key] = initial_control_points
params.update();

`, `
32/32

`, `
450/450 loss: 0.00164581

`, `
bitmap_ref = mi.bitmap('../scenes/references/starmoon.exr')
image_ref = mi.tensorxf(bitmap_ref)

# reshape into [height, widht, 1]
resolution = bitmap_ref.size()
image_ref = mi.tensorxf(image_ref.array, shape=(resolution[1], resolution[0], 1))

bitmap_ref

`, `
copy to clipboard
`, `
copy to clipboard
![../../_images/src_inverse_rendering_projective_sampling_integrators_19_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_projective_sampling_integrators_19_0.png)
### the `, `
copy to clipboard
![../../_images/src_inverse_rendering_projective_sampling_integrators_26_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_projective_sampling_integrators_26_0.png)
## optimization[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/projective_sampling_integrators.html#optimization "link to this heading")
the previous section is not only meant to be a small educational introduction to the projective sampling integrators, it also allows us to establish good hyperparameters for our integrator. this can be critical in certain optimization problems as the quality of the gradients has a direct influence on the outcome of the task.
copy to clipboard
`, `
copy to clipboard
## scene[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/projective_sampling_integrators.html#scene "link to this heading")
we want to cast two distinct shadows from a single object and have the shadows form specific shapes. more specifically, we’ll be optimizing a 3d curve using the [bsplinecurve](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_shapes.html#b-spline-curve-bsplinecurve) plugin such that its two shadows “draw” a moon and a star.
we provide you with a pre-built scene in a xml file, we just need to load it. let’s also render it.
copy to clipboard
`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/projective_sampling_integrators.html

---

### [Rendering from multiple view points](./en_stable_src_rendering_multi_view_rendering.html.md)

**Summary:** In this tutorial, you will learn how to render a given scene from multiple points of view. This can be very convenient if you wish to generate a large synthetic dataset, or are doing some multi-view optimization.

**Sections:** Overview, Setup, Loading a scene, Creating sensors, Rendering using a specific sensor

**Keywords:** `
# create an alias for convenience
frommitsubaimport scalartransform4f as t

scene = mi.load_dict({
    'type': 'scene',
    # the keys below correspond to object ids and can be chosen arbitrarily
    'integrator': {'type': 'path'},
    'light': {'type': 'constant'},
    'teapot': {
        'type': 'ply',
        'filename': '../scenes/meshes/teapot.ply',
        'to_world': t().translate([0, 0, -1.5]),
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {'type': 'rgb', 'value': [0.1, 0.2, 0.3]},
        },
    },
})

`, `
copy to clipboard
## loading a scene[¶](https://mitsuba.readthedocs.io/en/stable/src/rendering/multi_view_rendering.html#loading-a-scene "link to this heading")
in [previous tutorials](https://mitsuba.readthedocs.io/en/latest/src/quickstart/mitsuba_quickstart.html#loading-a-scene), we have seen how to load a mitsuba scene from an xml file. in mitsuba 3, it is also possbile to load a scene defined by python dictionary using [load_dict()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.load_dict).
in fact, the `, `
copy to clipboard
## rendering using a specific sensor[¶](https://mitsuba.readthedocs.io/en/stable/src/rendering/multi_view_rendering.html#rendering-using-a-specific-sensor "link to this heading")
the [render()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.render) function can take quite a few aditional arguments. in a previous tutorial, we have seen that we can specify the number of samples per pixel with the keyword argument `, `
copy to clipboard
finally, we can use the convenient [bitmap](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.bitmap) class to quickly visualize our six rendered images directly in the notebook.
copy to clipboard
`, `
copy to clipboard
next, let’s create our sensors!
## creating sensors[¶](https://mitsuba.readthedocs.io/en/stable/src/rendering/multi_view_rendering.html#creating-sensors "link to this heading")
mitsuba provides a high level `, `
copy to clipboard
we are now ready to render from each of our sensors.
copy to clipboard
`, `
defload_sensor(r, phi, theta):
    # apply two rotations to convert from spherical coordinates to world 3d coordinates.
    origin = t().rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.scalarpoint3f([0, 0, r])

    return mi.load_dict({
        'type': 'perspective',
        'fov': 39.3077,
        'to_world': t().look_at(
            origin=origin,
            target=[0, 0, 0],
            up=[0, 0, 1]
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 16
        },
        'film': {
            'type': 'hdrfilm',
            'width': 256,
            'height': 256,
            'rfilter': {
                'type': 'tent',
            },
            'pixel_format': 'rgb',
        },
    })

`, `
frommatplotlibimport pyplot as plt
fig = plt.figure(figsize=(10, 7))
fig.subplots_adjust(wspace=0, hspace=0)
for i in range(sensor_count):
    ax = fig.add_subplot(2, 3, i + 1).imshow(images[i] ** (1.0 / 2.2))
    plt.axis("off")

`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/rendering/multi_view_rendering.html

---

## Lights & Emitters

### [Choosing variants](./en_stable_src_key_topics_variants.html.md)

**Summary:** Mitsuba 3 is a retargetable rendering system that provides a set of different system “ variants ” that change elementary aspects of simulation—they can for instance replace the representation of color to support monochromatic, RGB, spectral, or ev...

**Sections:** Part 1: Computational backend, Part 2: Automatic differentiation, Part 3: Color representation, Part 4: Polarization, Part 5: Precision

**Keywords:** `_double`, `choosing`, `cuda`, `dr.jit`, `light`, `llvm`, `mono`, `path`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html

---

### [Mitsuba on WSL 2](./en_stable_src_optix_setup.html.md)

**Summary:** Mitsuba uses the NVIDIA OptiX framework for hardware-accelerated ray tracing. While OptiX is not yet officially supported on the Windows Subsystem for Linux 2 (WSL 2), it is possible to get it to run in practice. The following instructions are bas...

**Keywords:** `
$ bash
`, `
$ ln
`, `
$ mkdir&&&&&&&&&&&&&&"c:\windows\system32\lxss\lib"

`, `
c:\users\...> wsl --shutdown

`, `
copy to clipboard
create a symbolic link that exposes the already installed cuda driver to runtime loading:
`, `
copy to clipboard
next, copy-paste and run the following command:
`, `
copy to clipboard
this will open two explorer windows: one to a system path containing internal wsl driver files (`, ` directory containing files that need to be copied to `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/optix_setup.html

---

## Materials & BSDFs

### [BSDFs](./en_stable_src_generated_plugins_bsdfs.html.md)

**Summary:** > ![../../images/bsdfoverview.jpg ](https://mitsuba.readthedocs.io/en/stable/images/bsdfoverview.jpg) > Schematic overview of the most important surface scattering models in Mitsuba 3. The arrows indicate possible outcomes of an interaction with a...

**Sections:** Correctness considerations, Smooth diffuse material (diffuse), Smooth dielectric material (dielectric), Thin dielectric material (thindielectric), Rough dielectric material (roughdielectric)

**Keywords:** `
'type': '...',
'glass':  {
    'type': 'dielectric',
    'int_ior': 1.504,
    'ext_ior': 1.0
},
'interior': {
    'type': 'homogeneous',
    'scale': 4,
    'sigma_t': {
        'type': 'rgb',
        'value': [1, 1, 0.5]
    },
    'albedo': {
        'type': 'rgb',
        'value': [0.0, 0.0, 0.0]
    }
}

`, `
'type': 'blendbsdf',
'weight': {
    'type': 'bitmap',
    'filename': 'pattern.png'
},
'bsdf_0': {
    'type': 'conductor'
},
'bsdf_1': {
    'type': 'roughplastic',
    'diffuse_reflectance': 0.1
}

`, `
'type': 'bumpmap',
'arbitrary': {
    'raw': true,
    'filename': 'textures/bumpmap.jpg'
},
'bsdf': {
    'type': 'roughplastic'
}

`, `
'type': 'circular',
'left_handed': true

`, `
'type': 'conductor',
'eta':  {
    'type': 'spectrum',
    'filename': 'conductorior.eta.spd'
},
'k':  {
    'type': 'spectrum',
    'filename': 'conductorior.k.spd'
}

`, `
'type': 'conductor',
'material': 'au'

`, `
'type': 'dielectric',
'int_ior': 'water',
'ext_ior': 'air'

`, `
'type': 'diffuse',
'reflectance': {
    'type': 'bitmap',
    'filename': 'wood.jpg'
}

`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html

---

## Rendering

### [Bibliography](./en_stable_zz_bibliography.html.md)

**Summary:** [Azz78] R. M. A. Azzam. Photopolarimetric measurement of the mueller matrix by fourier analysis of a single detected signal. Opt. Lett. , 2(6):148–150, Jun 1978. URL: <https://ol.osa.org/abstract.cfm?URI=ol-2-6-148>, doi:10.1364/OL.2.000148.

**Keywords:** `bibliography`, `bsdf`, `light`, `path`, `tracing`

**Source:** https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html

---

### [Films](./en_stable_src_generated_plugins_films.html.md)

**Summary:** A film defines how conducted measurements are stored and converted into the final output file that is written to disk at the end of the rendering process. In the XML scene description language, a normal film configuration might look as follows:

**Sections:** High dynamic range film (hdrfilm), Spectral film (specfilm)

**Keywords:** `
'type': 'hdrfilm',
'pixel_format': 'rgba',
'width': 1920,
'height': 1080

`, `
'type': 'scene',

# .. scene contents ..

'sensor_id': {
    'type': '<sensor_type>',

    # write to a high dynamic range exr image
    'film_id': {
        'type': 'hdrfilm',
        # specify the desired resolution (e.g. full hd)
        'width': 1920,
        'height': 1080,
        # use a gaussian reconstruction filter
        'filter': { 'type': 'gaussian' }
    }
}

`, `
'type': 'specfilm',
'width': 1920,
'height': 1080,
'band1_red': {
    'type': 'spectrum',
    'filename': 'data_red.spd'
},
'band1_green': {
    'type': 'spectrum',
    'filename': 'data_green.spd'
},
'band1_blue': {
    'type': 'spectrum',
    'filename': 'data_blue.spd'
}

`, `
<filmtype="hdrfilm">
<stringname="pixel_format"value="rgba"/>
<integername="width"value="1920"/>
<integername="height"value="1080"/>
</film>

`, `
<filmtype="specfilm">
<integername="width"value="1920"/>
<integername="height"value="1080"/>
<spectrumname="band1_red"filename="data_red.spd"/>
<spectrumname="band2_green"filename="data_green.spd"/>
<spectrumname="band3_blue"filename="data_blue.spd"/>
</film>

`, `
<sceneversion="3.0.0">
<!-- .. scene contents -->

<sensortype=".. sensor type ..">
<!-- .. sensor parameters .. -->

<!-- write to a high dynamic range exr image -->
<filmtype="hdrfilm">
<!-- specify the desired resolution (e.g. full hd) -->
<integername="width"value="1920"/>
<integername="height"value="1080"/>

<!-- use a gaussian reconstruction filter. -->
<rfiltertype="gaussian"/>
</film>
</sensor>
</scene>

`, `
copy to clipboard
`, `
copy to clipboard
## spectral film (specfilm)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_films.html#spectral-film-specfilm "link to this heading")
parameter | type | description | flags  
---|---|---|---  
width, height | integer | width and height of the camera sensor in pixels. (default: 768, 576) |   
component_format | string | specifies the desired floating point component format of output images. the options are float16, float32, or uint32. (default: float16) |   
crop_offset_x, crop_offset_y, crop_width, crop_height | integer | these parameters can optionally be provided to select a sub-rectangle of the output. in this case, only the requested regions will be rendered. (default: unused) |   
sample_border | boolean | if set to true, regions slightly outside of the film plane will also be sampled. this may improve the image quality at the edges, especially when using very large reconstruction filters. in general, this is not needed though. (default: false, i.e. disabled) |   
compensate | boolean | if set to true, sample accumulation will be performed using kahan-style error-compensated accumulation. this can be useful to avoid roundoff error when accumulating very many samples to compute reference solutions using single precision variants of mitsuba. this feature is currently only supported in jit variants and can make sample accumulation quite a bit more expensive. (default: false, i.e. disabled) |   
(nested plugin) | rfilter | reconstruction filter that should be used by the film. (default: gaussian, a windowed gaussian filter) |   
state parameters |  |  |   
(nested plugins) | spectrum | one or several sensor response functions (srf) used to compute different spectral bands | p  
size | `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_films.html

---

### [Gradient-based optimization](./en_stable_src_inverse_rendering_gradient_based_opt.html.md)

**Summary:** Mitsuba 3 can be used to solve inverse problems involving light using a technique known as differentiable rendering. It interprets the rendering algorithm as a function \\(f(\mathbf{x})\\) that converts an input \\(\mathbf{x}\\) (the scene descrip...

**Sections:** Overview, Setup, Scene loading, Reference image, Initial state

**Keywords:** `
<defaultname="spp"value="128"/>
<defaultname="res"value="256"/>
<defaultname="max_depth"value="6"/>
<defaultname="integrator"value="path"/>

`, `
copy to clipboard
`, `
copy to clipboard
## reference image[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/gradient_based_opt.html#reference-image "link to this heading")
we render a reference image of the original scene that will later be used in the objective function for the optimization. ideally, this reference image should expose very little noise as it will pertube optimization process otherwise. for best results, we should render it with an even larger sample count.
copy to clipboard
`, `
copy to clipboard
## results[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/gradient_based_opt.html#results "link to this heading")
we can now render the scene again to check whether the optimization process successfully recovered the color of the red wall.
copy to clipboard
`, `
copy to clipboard
## scene loading[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/gradient_based_opt.html#scene-loading "link to this heading")
before loading the scene, let’s note that in `, `
copy to clipboard
as expected, when rendering the scene again, the wall has changed color.
copy to clipboard
`, `
copy to clipboard
at every iteration of the gradient descent, we will compute the derivatives of the scene parameters with respect to the objective function. in this simple experiment, we use the [mean square error](https://en.wikipedia.org/wiki/mean_squared_error), or \\(l_2\\) error, between the current image and the reference created above.
copy to clipboard
`, `
copy to clipboard
copy to clipboard
![](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/gradient_based_opt.html)
## initial state[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/gradient_based_opt.html#initial-state "link to this heading")
using the `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/gradient_based_opt.html

---

### [Integrators](./en_stable_src_generated_plugins_integrators.html.md)

**Summary:** In Mitsuba 3, the different rendering techniques are collectively referred to as integrators , since they perform integration over a high-dimensional space. Each integrator represents a specific approach for solving the light transport equation—us...

**Sections:** Direct illumination integrator (direct), Path tracer (path), Arbitrary Output Variables integrator (aov), Volumetric path tracer (volpath), Volumetric path tracer with spectral MIS (volpathmis)

**Keywords:** `
'type': 'aov',
'aovs': 'dd.y:depth,nn:sh_normal',
'my_image': {
    'type': 'path',
}

`, `
'type': 'depth'

`, `
'type': 'direct'

`, `
'type': 'direct_projective',
'sppc': 32,
'sppp': 32,
'sppi': 128,
'guiding': 'octree',
'guiding_proj': true,
'guiding_rounds': 1

`, `
'type': 'moment',
'nested': {
    'type': 'path',
}

`, `
'type': 'path',
'max_depth': 8

`, `
'type': 'prb',
'max_depth': 8

`, `
'type': 'prb_basic',
'max_depth': 8

`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html

---

### [Participating media](./en_stable_src_generated_plugins_media.html.md)

**Summary:** In Mitsuba, participating media are used to simulate materials ranging from fog, smoke, and clouds, over translucent materials such as skin or milk, to “fuzzy” structured substances such as woven or knitted cloth. This section describes the two av...

**Sections:** Homogeneous medium (homogeneous), Heterogeneous medium (heterogeneous)

**Keywords:** `
# declare a heterogeneous participating medium named 'smoke'
'smoke': {
    'type': 'heterogeneous',

    # acquire extinction values from an external data file
    'sigma_t': {
        'type': 'gridvolume',
        'filename': 'frame_0150.vol'
    },

    # the albedo is constant and set to 0.9
    'albedo': 0.9,

    # use an isotropic phase function
    'phase': {
        'type': 'isotropic'
    },

    # scale the density values as desired
    'scale': 200
},

# attach the index-matched medium to a shape in the scene
'shape': {
    'type': 'obj',
    # load an obj file, which contains a mesh version
    # of the axis-aligned box of the volume data file
    'filename': 'bounds.obj',

    # reference the medium by id
    'interior': 'smoke',
    # if desired, this shape could also declare
    # a bsdf to create an index-mismatched
    # transition, e.g.
    # 'bsdf': {
    #     'type': 'isotropic'
    # },
}

`, `
'type': 'homogeneous',
'albedo': {
    'type': 'rgb',
    'value': [0.99, 0.9, 0.96]
},
'sigma_t': 5,
# the extinction is also allowed to be spectrally varying
# since rgb values have to be in the [0, 1]
# 'sigma_t': {
#     'value': [0.5, 0.25, 0.8]
# }

# a homogeneous medium needs to have a constant extinction,
# but can have a spatially varying albedo:
# 'albedo': {
#     'type': 'gridvolume',
#     'filename': 'albedo.vol'
# }

'phase': {
    'type': 'hg',
    'g': 0.7
}

`, `
'type': 'scene',

# .. scene contents ..

'fog': {
    'type': 'homogeneous',
    # .. homogeneous medium parameters ..
},

'sensor_id': {
    'type': 'perspective',
    # .. perspective camera parameters ..

    # reference the fog medium
    'medium' : {
        'type' : 'ref',
        'id' : 'fog'
    }
}

`, `
'type': 'scene',
'shape_id': {
    'type': '<shape_type>',
    # .. shape parameters ..

    'interior': {
        'type': '<medium_type>',
        # .. medium parameters ..
    },
    'exterior': {
        'type': '<medium_type>',
        # .. medium parameters ..
    }
}

`, `
<!-- declare a heterogeneous participating medium named 'smoke' -->
<mediumtype="heterogeneous"id="smoke">
<!-- acquire extinction values from an external data file -->
<volumename="sigma_t"type="gridvolume">
<stringname="filename"value="frame_0150.vol"/>
</volume>

<!-- the albedo is constant and set to 0.9 -->
<floatname="albedo"value="0.9"/>

<!-- use an isotropic phase function -->
<phasetype="isotropic"/>

<!-- scale the density values as desired -->
<floatname="scale"value="200"/>
</medium>

<!-- attach the index-matched medium to a shape in the scene -->
<shapetype="obj">
<!-- load an obj file, which contains a mesh version
         of the axis-aligned box of the volume data file -->
<stringname="filename"value="bounds.obj"/>

<!-- reference the medium by id -->
<refname="interior"id="smoke"/>
<!-- if desired, this shape could also declare
        a bsdf to create an index-mismatched
        transition, e.g.
        <bsdf type="dielectric"/>
    -->
</shape>

`, `
<mediumid="mymedium"type="homogeneous">
<rgbname="albedo"value="0.99, 0.9, 0.96"/>
<floatname="sigma_t"value="5"/>

<!-- the extinction is also allowed to be spectrally varying
         since rgb values have to be in the [0, 1]
        <rgb name="sigma_t" value="0.5, 0.25, 0.8"/>
    -->

<!-- a homogeneous medium needs to have a constant extinction,
        but can have a spatially varying albedo:

        <volume name="albedo" type="gridvolume">
            <string name="filename" value="albedo.vol"/>
        </volume>
    -->

<phasetype="hg">
<floatname="g"value="0.7"/>
</phase>
</medium>

`, `
<sceneversion="3.0.0">
<!-- .. scene contents .. -->

<mediumtype="homogeneous"id="fog">
<!-- .. homogeneous medium parameters .. -->
</medium>
<sensortype="perspective">
<!-- .. perspective camera parameters .. -->
<!-- reference the fog medium from within the sensor declaration
            to make it aware that it is embedded inside this medium -->
<refid="fog"/>
</sensor>
</scene>

`, `
<sceneversion="3.0.0">
<shapetype=".. shape type ..">
<mediumname="interior"type="... medium type ...">
</medium>
<mediumname="exterior"type="... medium type ...">
</medium>
<!-- alternatively: reference named media that
            have been declared previously
            <ref name="interior" id="mymedium1"/>
            <ref name="exterior" id="mymedium2"/>
        -->
</shape>
</scene>

`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_media.html

---

### [Phase functions](./en_stable_src_generated_plugins_phase.html.md)

**Summary:** This section contains a description of all implemented medium scattering models, which are also known as phase functions. These are very similar in principle to surface scattering models (or BSDFs), and essentially describe where light travels aft...

**Sections:** Isotropic phase function (isotropic), Henyey-Greenstein phase function (hg), SGGX phase function (sggx), Blended phase function (blendphase), Lookup table phase function (tabphase)

**Keywords:** `
'type': 'blendphase',
'weight': 0.5,
'phase_0': {
    'type': 'isotropic'
},
'phase_1': {
    'type': 'hg',
    'g': 0.2
}

`, `
'type': 'hg',
'g': 0.1

`, `
'type': 'isotropic'

`, `
'type': 'rayleigh'

`, `
'type': 'sggx',
's': {
    'type': 'gridvolume',
    'filename': 'volume.vol'
}

`, `
<phasetype="blendphase">
<floatname="weight"value="0.5"/>
<phasename="phase_0"type="isotropic"/>
<phasename="phase_1"type="hg">
<floatname="g"value="0.2"/>
</phase>
</phase>

`, `
<phasetype="hg">
<floatname="g"value="0.1"/>
</phase>

`, `
<phasetype="isotropic"/>

`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_phase.html

---

### [Plugin reference](./en_stable_src_plugin_reference.html.md)

**Summary:** BSDFs Emitters Films Integrators Participating media Phase functions Reconstruction filters Samplers Sensors Shapes Spectra Textures Volumes The subsections above describe the available Mitsuba 3 plugins, usually along with example renderings and ...

**Sections:** Overview, Scene-wide attributes

**Keywords:** `
<integratortype="amazing">
<booleanname="softer_rays"value="true"/>
<floatname="dark_matter"value="0.44"/>
<!-- nested unnamed integrator -->
<integratortype="path"/>
<!-- nested texture named puppies -->
<texturename="puppies"type="bitmap">
<stringname="filename"value="cute.jpg"/>
</texture>
</integrator>

`, `
<integratortype="amazing">
<booleanname="softer_rays"value="true"/>
<floatname="dark_matter"value="0.44"/>
</integrator>

`, `
<sceneversion="3.0.0">
<booleanname="embree_use_robust_intersection"value="true"/>
</scene>

`, `
copy to clipboard
`, `
copy to clipboard
### flags[¶](https://mitsuba.readthedocs.io/en/stable/src/plugin_reference.html#flags "link to this heading")
each parameter optionally has flags that are listed in the last column. these flags indicate whether the parameter is differentiable or not, or whether it introduces discontinuities and thus needs special treatment.
see [`, `
copy to clipboard
in some cases, plugins also indicate that they accept nested plugins as input arguments. these can either be _named_ or _unnamed_. if the `, `
{
    'type': 'amazing',
    'softer_rays': true,
    'dark_matter': 0.44
}

`, `
{
    'type': 'amazing',
    'softer_rays': true,
    'dark_matter': 0.44,
    # nested unnamed integrator
    'foo': {
        'type': 'path'
    },
    # nested texture named puppies
    'puppies': {
        'type': 'bitmap',
        'filename': 'cute.jpg'
    }
}

`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/plugin_reference.html

---

### [Release notes](./en_stable_release_notes.html.md)

**Summary:** Being an experimental research framework, Mitsuba 3 does not strictly follow the Semantic Versioning convention. That said, we will strive to document breaking API changes in the release notes below. October 17, 2025

**Sections:** Mitsuba 3.7.1, Mitsuba 3.7.0, Mitsuba 3.6.4, Mitsuba 3.6.3, Mitsuba 3.6.2

**Keywords:** `#345`, `*_rgb`, `*_spectral`, `<rgb ..>`, `<spectrum ..>`, `@dr.freeze`, `@drjit.freeze`, `@drjit.wrap`

**Source:** https://mitsuba.readthedocs.io/en/stable/release_notes.html

---

### [Samplers](./en_stable_src_generated_plugins_samplers.html.md)

**Summary:** When rendering an image, Mitsuba 3 has to solve a high-dimensional integration problem that involves the geometry, materials, lights, and sensors that make up the scene. Because of the mathematical complexity of these integrals, it is generally im...

**Sections:** Independent sampler (independent), Stratified sampler (stratified), Correlated Multi-Jittered sampler (multijitter), Orthogonal Array sampler (orthogonal), Low discrepancy sampler (ldsampler)

**Keywords:** `
'type': 'independent',
'sample_count': '64'

`, `
'type': 'ldsampler',
'sample_count': '64'

`, `
'type': 'multijitter',
'sample_count': '64'

`, `
'type': 'orthogonal',
'sample_count': '4'

`, `
'type': 'stratified',
'sample_count': '4'

`, `
<samplertype="independent">
<integername="sample_count"value="64"/>
</sampler>

`, `
<samplertype="ldsampler">
<integername="sample_count"value="64"/>
</sampler>

`, `
<samplertype="multijitter">
<integername="sample_count"value="64"/>
</sampler>

`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html

---

### [Scripting a renderer](./en_stable_src_rendering_scripting_renderer.html.md)

**Summary:** Mitsuba provides a flexible API that allows developing custom rendering pipelines that largely bypass the high-level interfaces and machinery used in the previous tutorials. This enables the use of the Mitsuba to rapidly prototype new and unconven...

**Sections:** Overview, Setup, Spawning rays, Ambient occlusion, Displaying the result

**Keywords:** `
# accumulated result
result = mi.float(0)

@dr.syntax
defmy_loop(result, si, rng, ambient_ray_count):
    # loop iteration counter
    i = mi.uint32(0)

    while (si.is_valid() & (i < ambient_ray_count)):
        # 1. draw some random numbers
        sample_1, sample_2 = rng.next_float32(), rng.next_float32()

        # 2. compute directions on the hemisphere using the random numbers
        wo_local = mi.warp.square_to_uniform_hemisphere([sample_1, sample_2])

        # alternatively, we could also sample a cosine-weighted hemisphere
        # wo_local = mi.warp.square_to_cosine_hemisphere([sample_1, sample_2])

        # 3. transform the sampled directions to world space
        wo_world = si.sh_frame.to_world(wo_local)

        # 4. spawn a new ray starting at the surface interactions
        ray_2 = si.spawn_ray(wo_world)

        # 5. set a maximum intersection distance to only account for the close-by geometry
        ray_2.maxt = ambient_range

        # 6. accumulate a value of 1 if not occluded (0 otherwise)
        result[~scene.ray_test(ray_2)] += 1.0

        # 7. increase loop iteration counter
        i += 1
    # divide the result by the number of samples
    return result / ambient_ray_count


result = my_loop(result, si, rng, ambient_ray_count)

`, `
# camera origin in world space
cam_origin = mi.point3f(0, 1, 3)

# camera view direction in world space
cam_dir = dr.normalize(mi.vector3f(0, -0.5, -1))

# camera width and height in world space
cam_width  = 2.0
cam_height = 2.0

# image pixel resolution
image_res = (256, 256)

`, `
# construct a grid of 2d coordinates
x, y = dr.meshgrid(
    dr.linspace(mi.float, -cam_width  / 2,   cam_width / 2, image_res[0]),
    dr.linspace(mi.float, -cam_height / 2,  cam_height / 2, image_res[1])
)

# ray origin in local coordinates
ray_origin_local = mi.vector3f(x, y, 0)

# ray origin in world coordinates
ray_origin = mi.frame3f(cam_dir).to_world(ray_origin_local) + cam_origin

`, `
# initialize the random number generator
rng = mi.pcg32(size=dr.prod(image_res))

`, `
ambient_range = 0.75
ambient_ray_count = 256

`, `
copy to clipboard
## ambient occlusion[¶](https://mitsuba.readthedocs.io/en/stable/src/rendering/scripting_renderer.html#ambient-occlusion "link to this heading")
ambient occlusion is a rendering technique that calculates the average local occlusion of surfaces. for a point on the surface, we trace a set of rays (`, `
copy to clipboard
## displaying the result[¶](https://mitsuba.readthedocs.io/en/stable/src/rendering/scripting_renderer.html#displaying-the-result "link to this heading")
the algorithm above accumulated ambient occlusion samples in a 1-dimensional array `, `
copy to clipboard
in the following code, we use the `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/rendering/scripting_renderer.html

---

### [Sensors](./en_stable_src_generated_plugins_sensors.html.md)

**Summary:** In Mitsuba 3, sensors , along with a film , are responsible for recording radiance measurements in some usable format. In the XML scene description language, a sensor declaration looks as follows: XMLPython

**Sections:** Orthographic camera (orthographic), Perspective pinhole camera (perspective), Perspective camera with a thin lens (thinlens), Distant radiancemeter sensor (distant), Batch sensor (batch)

**Keywords:** `
'type': 'batch',
# two perpendicular viewing directions
'sensor1': {
    'type': 'perspective',
    'fov': 45,
    'to_world': mi.scalaraffinetransform4f().look_at(
        origin=[0, 0, 1],
        target=[0, 0, 0],
        up=[0, 1, 0]
    )
},
'sensor2': {
    'type': 'perspective',
    'fov': 45,
    'to_world': mi.scalaraffinetransform4f().look_at(
        origin=[1, 0, 0],
        target=[0, 0, 0],
        up=[0, 1, 0]
    ),
}
'film_id': {
    'type': '<film_type>',
    # ...
},
'sampler_id': {
    'type': '<sampler_type>',
    # ...
}

`, `
'type': 'orthographic',
'to_world': mi.scalaraffinetransform4f().look_at(
    origin=[1, 1, 1],
    target=[1, 2, 1],
    up=[0, 0, 1]
) @ mi.scalaraffinetransform4f().scale([10, 10, 1])

`, `
'type': 'perspective',
'fov': 45,
'to_world': mi.scalaraffinetransform4f().look_at(
    origin=[1, 1, 1],
    target=[1, 2, 1],
    up=[0, 0, 1]
),
'film_id': {
    'type': '<film_type>',
    # ...
},
'sampler_id': {
    'type': '<sampler_type>',
    # ...
}

`, `
'type': 'scene',

# .. scene contents ..

'sensor_id': {
    'type': '<sensor_type>',

    'film_id': {
        'type': '<film_type>',
        # ...
    },
    'sampler_id': {
        'type': '<sampler_type>',
        # ...
    }
}

`, `
'type': 'sphere',
'sensor': {
    'type': 'irradiancemeter'
    'film': {
        # ...
    }
}

`, `
'type': 'thinlens',
'fov': 45,
'to_world': mi.scalaraffinetransform4f().look_at(
    origin=[1, 1, 1],
    target=[1, 2, 1],
    up=[0, 0, 1]
),
'focus_distance': 1.0,
'aperture_radius': 0.1,
'film_id': {
    'type': '<film_type>',
    # ...
},
'sampler_id': {
    'type': '<sampler_type>',
    # ...
}

`, `
<sceneversion=3.0.0>
<!-- .. scene contents .. -->

<sensortype=".. sensor type ..">
<!-- .. sensor parameters .. -->

<samplertype=".. sampler type ..">
<!-- .. sampler parameters .. -->
</samplers>

<filmtype=".. film type ..">
<!-- .. film parameters .. -->
</film>
</sensor>
</scene>

`, `
<sensortype="batch">
<!-- two perpendicular viewing directions -->
<sensorname="sensor_1"type="perspective">
<floatname="fov"value="45"/>
<transformname="to_world">
<lookatorigin="0, 0, 1"target="0, 0, 0"up="0, 1, 0"/>
</transform>
</sensor>
<sensorname="sensor_2"type="perspective">
<floatname="fov"value="45"/>
<transformname="to_world">
<look_atorigin="1, 0, 0"target="1, 2, 1"up="0, 1, 0"/>
</transform>
</sensor>
<!-- film -->
<!-- sampler -->
</sensor>

`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_sensors.html

---

## Scene Format

### [Deep dive into a BSDF](./en_stable_src_others_bsdf_deep_dive.html.md)

**Summary:** As you have probably already discovered, Mitsuba 3 can do much more than rendering. In this tutorial we will show how to instantiate a BSDF plugin using Python dictionaries and plot its distribution function using matplotlib.

**Sections:** Overview, Setup, Instantiating a BSDF, Vectorized evaluation of the BSDF, Plotting the results

**Keywords:** `
# create a (dummy) surface interaction to use for the evaluation of the bsdf
si = dr.zeros(mi.surfaceinteraction3f)

# specify an incident direction with 45 degrees elevation
si.wi = sph_to_dir(dr.deg2rad(45.0), 0.0)

# create grid in spherical coordinates and map it onto the sphere
res = 300
theta_o, phi_o = dr.meshgrid(
    dr.linspace(mi.float, 0,     dr.pi,     res),
    dr.linspace(mi.float, 0, 2 * dr.pi, 2 * res)
)
wo = sph_to_dir(theta_o, phi_o)

# evaluate the whole array (18000 directions) at once
values = bsdf.eval(mi.bsdfcontext(), si, wo)

`, `
bsdf = mi.load_dict({
    'type': 'roughconductor',
    'alpha': 0.2,
    'distribution': 'ggx'
})

`, `
copy to clipboard
## instantiating a bsdf[¶](https://mitsuba.readthedocs.io/en/stable/src/others/bsdf_deep_dive.html#instantiating-a-bsdf "link to this heading")
one easy way to instanciate mitsuba objects (e.g., [shape](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_shapes.html), [bsdf](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html), …) is using the [load_dict](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.load_dict) function. this function takes as input a python `, `
copy to clipboard
## plotting the results[¶](https://mitsuba.readthedocs.io/en/stable/src/others/bsdf_deep_dive.html#plotting-the-results "link to this heading")
dr.jit arrays of any flavour can easily be converted to an array type of other mainstream libraries, such as `, `
copy to clipboard
## vectorized evaluation of the bsdf[¶](https://mitsuba.readthedocs.io/en/stable/src/others/bsdf_deep_dive.html#vectorized-evaluation-of-the-bsdf "link to this heading")
we will now evaluate this bsdf for a whole array of directions at once, leveraging the enabled vectorize backend. similarly to working on `, `
copy to clipboard
we can now use our favourite plotting library to visualize the bsdf distribution (here we use `, `
copy to clipboard
we can then use this function to generate a set of directions to evaluate the bsdf with.
copy to clipboard
`, `
defsph_to_dir(theta, phi):
"""map spherical to euclidean coordinates"""
    st, ct = dr.sincos(theta)
    sp, cp = dr.sincos(phi)
    return mi.vector3f(cp * st, sp * st, ct)

`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/others/bsdf_deep_dive.html

---

### [Emitters](./en_stable_src_generated_plugins_emitters.html.md)

**Summary:** > ![../../images/emitteroverview.jpg ](https://mitsuba.readthedocs.io/en/stable/images/emitteroverview.jpg) > Schematic overview of the emitters in Mitsuba 3. The arrows indicate the directional distribution of light.

**Sections:** Area light (area), Point light source (point), Constant environment emitter (constant), Environment emitter (envmap), Sun and sky emitter (sunsky)

**Keywords:** `
'type': 'constant',
'radiance': {
    'type': 'rgb',
    'value': 1.0,
}

`, `
'type': 'directional',
'direction': [1.0, 0.0, 0.0],
'irradiance': {
    'type': 'rgb',
    'value': 1.0,
}

`, `
'type': 'envmap',
'filename': 'textures/museum.exr'

`, `
'type': 'point',
'position': [0.0, 5.0, 0.0],
'intensity': {
    'type': 'spectrum',
    'value': 1.0,
}

`, `
'type': 'projector',
'irradiance': {
    'type': 'rgb',
    'value': 1.0,
},
'fov': 45,
'to_world': mi.scalaraffinetransform4f().look_at(
    origin=[1, 1, 1],
    target=[1, 2, 1],
    up=[0, 0, 1]
)

`, `
'type': 'scene',

# .. scene contents ..

'emitter_id': {
    'type': 'point',
    'position': [0, 0, -2],
    'intensity': {
        'type': 'spectrum',
        'value': 1.0,
    }
},

'shape_id': {
    'type': 'sphere'
}

`, `
'type': 'scene',

# .. scene contents ..

'shape_id': {
    'type': 'sphere',
    'emitter': {
        'type': 'area',
        'radiance': {
            'type': 'rgb',
            'value': 1.0,
        }
    }
}

`, `
'type': 'sphere',
'emitter': {
    'type': 'area',
    'radiance': {
        'type': 'rgb',
        'value': 1.0,
    }
}

`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html

---

### [Gallery](./en_stable_src_gallery.html.md)

**Summary:** On this page you will find different 3D scenes ready to be used with Mitsuba 3. All scenes in this gallery marked with 🅱️ originate from Benedikt Bitterli’s rendering resources. On the official webpage you can also download the same scenes for oth...

**Sections:** Simple scenes, Single object, Architecture, Documentation banners

**Keywords:** `gallery`, `light`, `material`, `scene`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/gallery.html

---

### [Image I/O and manipulation](./en_stable_src_how_to_guides_image_io_and_manipulation.html.md)

**Summary:** This how-to guide covers the visualization and manipulation of images with Bitmap. To get started, we import the mitsuba library and set a variant. Copy to clipboard  importmitsubaasmi mi.setvariant('scalarrgb')

**Sections:** Reading an image from disk, Bitmap conversion, Metadata, Array interface, Multichannel images

**Keywords:** `
# convert image to uint8
bmp_small = bmp_small.convert(mi.bitmap.pixelformat.rgb, mi.struct.type.uint8, true)
# write image to jpeg file
bmp_small.write('../scenes/textures/flower_photo_downscale.jpeg')

# or equivalently ...

# use the helper function to achieve the same result
mi.util.write_bitmap('../scenes/textures/flower_photo_downscale.jpeg', bmp_small, write_async=true)

`, `
# here we convert the list of pairs into a dict for easier use
res = dict(bmp_exr.split())

# plot the image, shading normal and depth buffer
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(res['image']);     axs[0].axis('off'); axs[0].set_title('image');
axs[1].imshow(res['sh_normal']); axs[1].axis('off'); axs[1].set_title('sh_normal');
axs[2].imshow(res['depth']);     axs[2].axis('off'); axs[2].set_title('depth');

`, `
# it is also possible to create a bitmap object from a numpy array
bmp_np = mi.bitmap(np.tile(np.linspace(0, 1, 100), (100, 1)).reshape((100,100,1)))

plt.imshow(bmp_np); plt.axis('off');

print(bmp_np)

`, `
# set a variant (in order of preference)
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')

# use the `, `
# some tensorial manipulation (here we blackout the upper left corner of the image)
img[:750, :750, :] = 0.0

# create a bitmap from the tensorxf object
bmp_new = mi.bitmap(img)

# specify that the underlying data is already gamma corrected
bmp_new.set_srgb_gamma(true)

plt.imshow(bmp_new); plt.axis('off');

print(bmp_new)

`, `
# thanks to the array interface protocol, it is easy to create a tensorxf from a bitmap object
img = mi.tensorxf(bmp)

img.shape

`, `
(1500, 1500, 3)

`, `
(numpy.ndarray, dtype('float32'), (1500, 1500, 3))

`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/image_io_and_manipulation.html

---

### [Key Topics](./en_stable_src_key_topics.html.md)

**Summary:** The following document aim at clarifying a particular part of the system or the background theory. They will often be referred by the tutorials and guides for in depth knowledge about a specific subject.

**Sections:** Overview, Topics

**Keywords:** `light`, `scene`, `topics`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/key_topics.html

---

### [Reconstruction filters](./en_stable_src_generated_plugins_rfilters.html.md)

**Summary:** Image reconstruction filters are responsible for converting a series of radiance samples generated jointly by the sampler and integrator into the final output image that will be written to disk at the end of a rendering process. This section gives...

**Sections:** Box filter (box), Tent filter (tent), Gaussian filter (gaussian), Mitchell filter (mitchell), Catmull-Rom filter (catmullrom)

**Keywords:** `
'type': 'box',

`, `
'type': 'catmullrom',

`, `
'type': 'gaussian',
'stddev': 0.25

`, `
'type': 'lanczos',
'lobes': 4

`, `
'type': 'mitchell',
'a': 0.25,
'b': 0.55

`, `
'type': 'tent',
'radius': 1.25,

`, `
<rfiltertype="box"/>

`, `
<rfiltertype="catmullrom"/>

`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_rfilters.html

---

### [Scene XML file format](./en_stable_src_key_topics_scene_format.html.md)

**Summary:** Mitsuba uses a simple and general XML-based format to represent scenes. Since the framework’s philosophy is to represent discrete blocks of functionality as plugins, a scene file can be interpreted as “recipe” specifying which plugins should be in...

**Sections:** Properties, References, Default parameters, Including external files, Aliases

**Keywords:** `
# first create a bsdf (could use xml.load_string(..) as well)
my_bsdf = mi.load_dict({
    "type": "roughconductor",
    "alpha": 0.14,
})

# pass the bsdf object in the dictionary
sphere = load_dict({
    "type": "sphere",
    "something": my_bsdf
})

`, `
# passing gray-scale value
"color_property": {
    "type": "rgb",
    "value": 0.44
}

# passing tristimulus values
"color_property": {
    "type": "rgb",
    "value": [0.7, 0.1, 0.5]
}

# providing a spectral file
"color_property": {
    "type": "spectrum",
    "filename": "filename.spd"
}

# providing a list of (wavelength, value) pairs
"color_property": {
    "type": "spectrum",
    "value": [(400.0, 0.5), (500.0, 0.8), (600.0, 0.2)]
}

`, `
<booleanname="bool_property"value="true"/>

`, `
<bsdftype="diffuse">
<rgbname="reflectance"value="$reflectance"/>
</bsdf>

`, `
<bsdftype="diffuse"id="my_material_1"/>
<aliasid="my_material_1"as="my_material_2"/>

`, `
<defaultname="reflectance"value="something"/>

`, `
<includefilename="nested-scene-$version.xml"/>

`, `
<includefilename="nested-scene.xml"/>

`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html

---

### [Shapes](./en_stable_src_generated_plugins_shapes.html.md)

**Summary:** This section presents an overview of the shape plugins that are released along with the renderer. In Mitsuba 3, shapes define surfaces that mark transitions between different types of materials. For instance, a shape could describe a boundary betw...

**Sections:** Wavefront OBJ mesh loader (obj), PLY (Stanford Triangle Format) mesh loader (ply), Serialized mesh loader (serialized), Cube (cube), Sphere (sphere)

**Keywords:** `
# declare a named shape group containing two objects
'my_shape_group': {
    'type': 'shapegroup',
    'first_object': {
        'type': 'ply',
        'bsdf': {
            'type': 'roughconductor',
        }
    },
    'second_object': {
        'type': 'sphere',
        'to_world': mi.scalaraffinetransform4f().scale([5, 5, 5]).translate([0, 20, 0])
        'bsdf': {
            'type': 'diffuse',
        }
    }
},

# instantiate the shape group without any kind of transformation
'first_instance': {
    'type': 'instance',
    'shapegroup': {
        'type': 'ref',
        'id': 'my_shape_group'
    }
},

# create instance of the shape group, but rotated, scaled, and translated
'second_instance': {
    'type': 'instance',
    'to_world': mi.scalaraffinetransform4f().rotate([1, 0, 0], 45).scale([1.5, 1.5, 1.5]).translate([0, 10, 0]),
    'shapegroup': {
        'type': 'ref',
        'id': 'my_shape_group'
    }
}

`, `
'curves': {
    'type': 'bsplinecurve',
    'to_world': mi.scalaraffinetransform4f().translate([1, 0, 0]).scale([2, 2, 2]),
    'filename': 'curves.txt'
}

`, `
'curves': {
    'type': 'linearcurve',
    'to_world': mi.scalaraffinetransform4f().scale([2, 2, 2]).translate([1, 0, 0]),
    'filename': 'curves.txt'
},

`, `
'primitives': {
    'type': 'ellipsoids',
    'filename': 'my_primitives.ply'
}

`, `
'primitives': {
    'type': 'ellipsoidsmesh',
    'filename': 'my_primitives.ply'
}

`, `
'sphere_1': {
    'type': 'sphere',
    'to_world': mi.scalaraffinetransform4f().scale([2, 2, 2]).translate([1, 0, 0]),
    'bsdf': {
        'type': 'diffuse'
    }
},

'sphere_2': {
    'type': 'sphere',
    'center': [1, 0, 0],
    'radius': 2,
    'bsdf': {
        'type': 'diffuse'
    }
}

`, `
'type': 'cube',
'to_world': mi.scalaraffinetransform4f([2, 10, 1])

`, `
'type': 'cylinder',
'radius': 0.3,
'material': {
    'type': 'diffuse'
}

`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html

---

### [Spectra](./en_stable_src_generated_plugins_spectra.html.md)

**Summary:** This section describes the plugins behind spectral reflectance or emission used in Mitsuba 3. On an implementation level, these behave very similarly to the texture plugins described earlier (but lacking their spatially varying property) and can t...

**Sections:** Uniform spectrum (uniform), Regular spectrum (regular), Irregular spectrum (irregular), sRGB spectrum (srgb), D65 spectrum (d65)

**Keywords:** `
'type': '.. shape type ..',
'emitter': {
    'type': 'area',
    'radiance': {
        'type': 'blackbody',
        'temperature': 5000
    }
}

`, `
'type': '.. shape type ..',
'emitter': {
    'type': 'area',
    'radiance': { 'type': 'd65', }
}

`, `
'type': 'irregular',
'wavelengths': '400, 700',
'values': '0.1, 0.2'

`, `
'type': 'rawconstant',
'value': 0.5  # or [0.5, -2.0, 0.3]

`, `
'type': 'regular',
'wavelength_min': 400,
'wavelength_max': 700,
'values': '0.1, 0.2'

`, `
'type': 'scene',
'bsdf_id': {
    'type': '<bsdf_type>',

    '<parameter name>': {
        'type': '<spectrum type>',
        # .. spectrum parameters ..
    }
}

`, `
'type': 'srgb',
'color': [10, 20, 250]

`, `
'type': 'uniform',
'value': 0.1

`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html

---

### [Volumes](./en_stable_src_generated_plugins_volumes.html.md)

**Summary:** This section covers the different types of volume data sources included with Mitsuba. These plug-ins are intended to be used together with the medium plugins and provide three-dimensional spatially varying density and albedo fields.

**Sections:** Grid-based volume data source (gridvolume), Constant-valued volume data source (constvolume)

**Keywords:** `
'type': 'heterogeneous',
'albedo': {
    'type': 'constvolume',
    'value': {
        'type': 'rgb',
        'value': [0.99, 0.8, 0.8]
    }
}

# shorthand: this will create a 'constvolume' internally
'albedo': {
    'type': 'rgb',
    'value': [0.99, 0.8, 0.8]
}

`, `
'type': 'heterogeneous',
'albedo': {
    'type': 'grid',
    'filename': 'my_volume.vol'
}

`, `
<mediumtype="heterogeneous">
<volumetype="constvolume"name="albedo">
<rgbname="value"value="0.99, 0.8, 0.8"/>
</volume>

<!-- shorthand: this will create a 'constvolume' internally -->
<rgbname="albedo"value="0.99, 0.99, 0.99"/>
</medium>

`, `
<mediumtype="heterogeneous">
<volumetype="grid"name="albedo">
<stringname="filename"value="my_volume.vol"/>
</volume>
</medium>

`, `
copy to clipboard
`, `
copy to clipboard
## constant-valued volume data source (constvolume)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_volumes.html#constant-valued-volume-data-source-constvolume "link to this heading")
parameter | type | description | flags  
---|---|---|---  
value | float or spectrum | specifies the value of the constant volume. | p, ∂  
this plugin provides a volume data source that is constant throughout its domain. depending on how it is used, its value can either be a scalar or a color spectrum.
xmlpython
`, `albedo`, `clamp`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_volumes.html

---

## Tutorials

### [API reference](./en_stable_src_api_reference.html.md)

**Summary:** This API reference documentation was automatically generated using the Autodoc Sphinx extension. Autodoc automatically processes the documentation of Mitsuba’s Python bindings, hence all C++ function and class signatures are documented through the...

**Sections:** Overview, Core, Parsing, Object, Properties

**Keywords:** `



_property_ center[¶](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.boundingsphere3f.center "link to this definition") 
    
(self) -> [`, `



_property_ center[¶](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.scalarboundingsphere3f.center "link to this definition") 
    
(self) -> [`, `



assign(_self_ , _arg_)[¶](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.frame3f.assign "link to this definition") 
     

parameter `, `



assign(_self_ , _arg_)[¶](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.scalaraffinetransform3d.assign "link to this definition") 
     

parameter `, `



assign(_self_ , _arg_)[¶](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.scalaraffinetransform3f.assign "link to this definition") 
     

parameter `, `



assign(_self_ , _arg_)[¶](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.scalaraffinetransform4d.assign "link to this definition") 
     

parameter `, `



assign(_self_ , _arg_)[¶](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.scalaraffinetransform4f.assign "link to this definition") 
     

parameter `, `



assign(_self_ , _arg_)[¶](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.scalarprojectivetransform3d.assign "link to this definition") 
     

parameter `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/api_reference.html

---

### [C++ Plugins & Macros](./en_stable_src_developer_guide_writing_plugin.html.md)

**Summary:** This section provides an overview of the “skeleton” of a basic plugin definition, including an explanation of the roles of the various MI macros. These macros are responsible for importing types, instantiating variants, and providing run-time type...

**Sections:** Example code, Macros

**Keywords:** `
copy to clipboard
## macros[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html#macros "link to this heading")
###  mi_mask_argument(mask)[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html#mi-mask-argument-mask "link to this heading")
this macro typically occurs at the beginning of a function that takes a mask as an argument.
`, `
copy to clipboard
###  mi_declare_class(name)[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html#mi-declare-class-name "link to this heading")
this macro should be invoked within the class declaration of the plugin. the provided `, `
copy to clipboard
###  mi_import_core_types()[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html#mi-import-core-types "link to this heading")
this macro will generate a sequence of `, `
copy to clipboard
###  mi_import_types(…)[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html#mi-import-types "link to this heading")
this macro invokes `, `
copy to clipboard
expands to
`, `
copy to clipboard
masking is not really needed on scalar variants: if a function is called, we assume that `, `
copy to clipboard
which turns the mask argument into a compile-time constant on scalar targets, allowing the compiler to optimize away the undesired branches.
mitsuba ships with a powerful sampling profiler that facilitates tracking down hot-spots during rendering. the last line of this macro (`, `
copy to clipboard
which turns the mask argument into a compile-time constant on scalar targets, allowing the compiler to optimize away the undesired branches. `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html

---

### [Caustics optimization](./en_stable_src_inverse_rendering_caustics_optimization.html.md)

**Summary:** This tutorial contains an advanced inverse rendering example: recovering the surface displacement (heightmap) of a slab of glass such that light passing through it focuses into a specific desired image.

**Sections:** Overview, 0. Setup, 1. Choosing a configuration, 2. Creating the scene, 3. Loading the reference image

**Keywords:** `
# looking at the receiving plane, not looking through the lens
sensor_to_world = mi.scalartransform4f().look_at(
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
        'sample_count': 512  # not really used
    },
    'film': {
        'type': 'hdrfilm',
        'width': resx,
        'height': resy,
        'pixel_format': 'rgb',
        'rfilter': {
            # important: smooth reconstruction filter with a footprint larger than 1 pixel.
            'type': 'gaussian'
        }
    },
}

`, `
# make sure that resources from the scene directory can be found
mi.file_resolver().append(scene_dir)

`, `
37.61896562576294  ms per iteration on average

`, `
[+] saved final heightmap state to: heightmap_final.exr
[+] saved displaced lens to: lens_displaced.ply

`, `
[i] loaded reference image from: /home/rami/mitsuba3/tutorials/scenes/references/sunday-512.jpg

`, `
[i] reference image selected: /home/rami/mitsuba3/tutorials/scenes/references/sunday-512.jpg

`, `
[i] results will be saved to: /home/rami/mitsuba3/tutorials/inverse_rendering/outputs/sunday

`, `
copy to clipboard
`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html

---

### [Compiling the system](./en_stable_src_developer_guide_compiling.html.md)

**Summary:** Compiling Mitsuba 3 from scratch requires recent versions of CMake (at least 3.9.0) and Python (at least 3.8). Further platform-specific dependencies and compilation instructions are provided below for each operating system. Some additional steps ...

**Sections:** Cloning the repository, Configuring mitsuba.conf, Linux, Windows, macOS

**Keywords:** `
# create a directory where build products are stored
mkdircd
`, `
# for running tests
sudo
`, `
# install recent versions build tools, including clang
sudo# install libraries for image i/o
sudo# install required python packages
sudo
`, `
# on linux / mac os
source# on windows (cmd)
c:/.../mitsuba3/build/release># on windows (powershell)
c:/.../mitsuba3/build/release>\setpath.ps1

`, `
# to be safe, explicitly ask for the 64 bit version of visual studio
cmake"visual studio 17 2022"
`, `
'!f(){ git pull "$@" && git submodule update --init --recursive; }; f'

`, `
cd
`, `
copy to clipboard
**tested version**
  * macos big sur 11.5.2
  * appleclang 13.2.0.0.1.1638488800
  * xcode 12.0.5
  * cmake 3.24.2
  * python 3.9.5


## running mitsuba[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html#running-mitsuba "link to this heading")
once mitsuba is compiled, run the `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html

---

### [Detailed look at `Optimizer`](./en_stable_src_how_to_guides_use_optimizers.html.md)

**Summary:** In the Gradient-based optimization tutorial, Mitsuba’s Optimizer class was used to build an optimization loop. In this tutorial, we will study the API of optimizers in more detail. It is designed to be convenient to use when directly optimizing pa...

**Sections:** The basics, Optimizing scene parameters, Optimizing latent variables, Optimizer state

**Keywords:** `
    params['redwall.vertex_positions'] = dr.ravel(new_vertex_pos)\

    # propagate changes through the scene (e.g. rebuild bvh)
    params.update()

`, `
# copy or our vertex positions (and convert them to 3d points)
initial_vertex_pos = dr.unravel(mi.point3f, params['redwall.vertex_positions'])

# now we define the update rule
defupdate_vertex_pos():
    # create the translation transformation
    t = mi.transform4f.translate(opt['translation'])

    # apply the transformation to the vertex positions
    new_vertex_pos = t @ initial_vertex_pos

    # flatten the vertex position array before assigning it to `, `
before the gradient step: x=[1], y=[3]
after the gradient step:  x=[0.75], y=[2.5]

`, `
copy to clipboard
`, `
copy to clipboard
## optimizing latent variables[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/use_optimizers.html#optimizing-latent-variables "link to this heading")
in more complex optimization scenarios, scene parameters might be described **as a function** of some other parameters. in such a scenario, we would be interested in optimizing those other parameters instead of the scene parameters directly.
for example, this would be needed when generating the vertex positions of a mesh using a neural network: we’d want to optimize the weights of the neural network, not the vertex positions themselves. another example could be a procedurally generated texture, maybe from a physically-based model that can be tweaked with a few parameters.
we refer to this type of “external” parameters as latent variables. many of the design decisions of the [optimizer](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.ad.optimizer) class were made to support latent variables.
for a simpler example, let’s consider the case where we are aiming at optimizing the translation vector of a 3d mesh object in our scene. even from an convexity standpoint, optimizing those three translation values will be much easier that having to optimize all the vertex positions simultaneously and hope for the best.
let’s fetch the scene parameters again and initialize our optimizer one more time.
copy to clipboard
`, `
copy to clipboard
## the basics[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/use_optimizers.html#the-basics "link to this heading")
to perform [gradient-based optimization](https://en.wikipedia.org/wiki/gradient_descent), mitsuba ships with standard optimizers including _stochastic gradient descent_ ([sgd](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.ad.sgd)) with and without momentum, as well as [adam](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.ad.adam). those both inherit from the [optimizer](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.ad.optimizer) base class and can be found in the `, `
copy to clipboard
after performing the update rule, the `, `
copy to clipboard
another useful feature of the mitsuba `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/use_optimizers.html

---

### [Developer’s Guide](./en_stable_src_developer_guide.html.md)

**Summary:** This section is addressed to the users interested in modifying the core of the system or even contributing to the codebase. New developers will want to begin by thoroughly reading the documentation of Dr.Jit before looking at any Mitsuba code. Dr....

**Sections:** Overview, Code structure, Coding style, Contributing, Going further

**Keywords:** `bsdf`, `developer`, `guide`, `light`, `plugin`, `render`, `shapes`, `src/core`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/developer_guide.html

---

### [Editing a scene](./en_stable_src_rendering_editing_a_scene.html.md)

**Summary:** In this tutorial, you will learn how to modify a Mitsuba scene after it has been loaded from a file. You might want to edit a scene before (re-)rendering it for many reasons. Maybe a corner is dim, or an object should be moved a bit to the left. T...

**Sections:** Overview, Loading a scene, Accessing scene parameters, Edit the scene, See also

**Keywords:** `
# give a red tint to light1 and a green tint to light2
params['light1.intensity.value'] *= [1.5, 0.2, 0.2]
params['light2.intensity.value'] *= [0.2, 1.5, 0.2]

# apply updates
params.update();

`, `
# translate the teapot a little bit
v = dr.unravel(mi.point3f, params['teapot.vertex_positions'])
v.z += 0.5
params['teapot.vertex_positions'] = dr.ravel(v)

# apply changes
params.update();

`, `
copy to clipboard
`, `
copy to clipboard
![../../_images/src_rendering_editing_a_scene_5_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_rendering_editing_a_scene_5_0.png)
## accessing scene parameters[¶](https://mitsuba.readthedocs.io/en/stable/src/rendering/editing_a_scene.html#accessing-scene-parameters "link to this heading")
any mitsuba object can be inspected using the [traverse()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.traverse) function, which returns a instance of [sceneparameters](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.sceneparameters). it has a similar api to python `, `
copy to clipboard
## edit the scene[¶](https://mitsuba.readthedocs.io/en/stable/src/rendering/editing_a_scene.html#edit-the-scene "link to this heading")
similarly to a python `, `
copy to clipboard
after rendering the scene again, we can easily compare the rendered images using `, `
copy to clipboard
as you can see, the first level of our scene graph has 4 objects:
  * the camera (`, `
copy to clipboard
let’s quickly render this scene.
copy to clipboard
`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/rendering/editing_a_scene.html

---

### [Forward inverse rendering](./en_stable_src_inverse_rendering_forward_inverse_rendering.html.md)

**Summary:** The previous example demonstrated reverse-mode differentiation (a.k.a. backpropagation) where a desired small change to the output image was converted into a small change to the scene parameters. Mitsuba and Dr.Jit can also propagate derivatives i...

**Sections:** Overview, Setup, Preparing the scene, Rendering, Visualizing the gradient image

**Keywords:** `
# forward-propagate gradients through the computation graph
dr.forward(params[key])

# fetch the image gradient values
grad_image = dr.grad(image)

`, `
# our latent variable
theta = mi.float(0.5)
dr.enable_grad(theta)

# the wall color now depends on `, `
clipping input data to the valid range for imshow with rgb data ([0..1] for floats or [0..255] for integers).

`, `
copy to clipboard
`, `
copy to clipboard
![../../_images/src_inverse_rendering_forward_inverse_rendering_11_1.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_forward_inverse_rendering_11_1.png)
note however that gradient values are not necessarily within the `, `
copy to clipboard
![../../_images/src_inverse_rendering_forward_inverse_rendering_13_1.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_forward_inverse_rendering_13_1.png)
## using latent variables (advanced)[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html#using-latent-variables-\(advanced\) "link to this heading")
in more complex scenarios, scene parameters might themselves be the result of differentiable computations, then depending on other _latent variables_. for instance, the color of the cbox wall could be the result of the evalutation of a neural network, or some other procedural process.
in this case, it is important for the gradients of those latent variables to be propagate to the scene parameters **before** calling `, `
copy to clipboard
## preparing the scene[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html#preparing-the-scene "link to this heading")
forward mode differentiable rendering begins analogously to reverse mode, by marking the parameters of interest as differentiable (in this example, we do so manually instead of using an `, `
copy to clipboard
## rendering[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html#rendering "link to this heading")
we can then perform the simulation to be differentiated. in this case, we simply render an image using the `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html

---

### [Granular phase function](./en_stable_src_others_granular_phase_function.html.md)

**Summary:** This tutorial showcases a more practical application and is a great introduction to write efficient integrator-like rendering code with the Mitsuba library. In volumetric rendering, the phase function describes the angular distribution of light sc...

**Sections:** Overview, Setup, Initializating the scene, Generating primary rays, Intra-grain transport

**Keywords:** `
# resolution of the histogram
histogram_size = 512

# convert escaping directions into histogram bin indices
cos_theta = mi.frame3f.cos_theta(frame.to_local(rays.d))
theta = dr.acos(cos_theta)
theta_idx = mi.uint32(theta / dr.pi * histogram_size)

# account for projection jacobian
throughput *= 1.0 / dr.sqrt(1 - cos_theta**2)

# accumulate values into the histogram
histogram = dr.zeros(mi.float, histogram_size)
dr.scatter_reduce(dr.reduceop.add, histogram, throughput, theta_idx, valid)

# execute the kernel by evaluating the histogram
dr.eval(histogram);

`, `
# sample ray directions
d = mi.warp.square_to_uniform_sphere(sampler.next_2d())

# construct coordinate frame object
frame = mi.frame3f(d)

# sample ray origins
xy_local = 2.0 * sampler.next_2d() - 1.0
local_o = mi.vector3f(xy_local.x, xy_local.y, -1.0)
world_o = frame.to_world(local_o)

# move ray origin according to scene bounding sphere
bsphere = scene.bbox().bounding_sphere()
o = world_o * bsphere.radius + bsphere.center

# construct rays
rays = mi.ray3f(o, d)

`, `
@dr.syntax
defsample(rays: mi.ray3f, sampler: mi.sampler):

    # find first ray intersection with the object
    si = scene.ray_intersect(rays)
    valid = si.is_valid()

    # maximum number of bounces
    max_bounces = 10

    # loop state variables
    throughput = mi.spectrum(1.0)
    active = mi.bool(valid)
    i = mi.uint32(0)

    while active & (i < max_bounces):
        # sample new direction
        ctx = mi.bsdfcontext()
        bs, bsdf_val = bsdf.sample(ctx, si, sampler.next_1d(), sampler.next_2d(), active)

        # update throughput and rays for next bounce
        throughput[active] *= bsdf_val
        rays[active] = si.spawn_ray(si.to_world(bs.wo))

        # find next intersection
        si = scene.ray_intersect(rays, active)
        active &= si.is_valid()

        # increase loop iteration counter
        i += 1

    # only account for rays that have escaped
    valid &= ~active

    # we don't care about a specific color for this tutorial
    return rays, mi.luminance(throughput), valid

rays, throughput, valid = sample(rays, sampler)

`, `
copy to clipboard
## calculating the histogram[¶](https://mitsuba.readthedocs.io/en/stable/src/others/granular_phase_function.html#calculating-the-histogram "link to this heading")
we are only interested in the light paths that have escaped the object, hence we first need to update the `, `
copy to clipboard
## generating primary rays[¶](https://mitsuba.readthedocs.io/en/stable/src/others/granular_phase_function.html#generating-primary-rays "link to this heading")
in the following cell, we are going to generate our primary rays, coming from all directions towards the center of the scene. for this, we use the sampler instance to generate random ray directions and offsets for the ray origins. the `, `
copy to clipboard
## initializating the scene[¶](https://mitsuba.readthedocs.io/en/stable/src/others/granular_phase_function.html#initializating-the-scene "link to this heading")
to keep this tutorial simple, we are only going to compute the phase function of a dielectric sphere. although only minor changes would be necessary to allow the use of other shapes and materials in this script (e.g., a frosty glass bunny ❄️🐰).
on top of various mesh loaders (e.g., `, `
copy to clipboard
## intra-grain transport[¶](https://mitsuba.readthedocs.io/en/stable/src/others/granular_phase_function.html#intra-grain-transport "link to this heading")
let’s now use those rays to construct light paths bouncing many times inside of the particle object.
for this, we perform a first ray intersection query with the scene using [scene.ray_intersect()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.scene.ray_intersect) which will return a [surfaceinteraction3f](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.surfaceinteraction3f) object, containing the surface interaction information. we use [is_valid()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.surfaceinteraction3f) to find the rays that actually interact with the object.
after initializing the `, `
copy to clipboard
## plotting the histogram[¶](https://mitsuba.readthedocs.io/en/stable/src/others/granular_phase_function.html#plotting-the-histogram "link to this heading")
let’s now take a look at the resulting angular distribution! we plot it in log scale on a regular plot as well as a polar plot.
copy to clipboard
`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/others/granular_phase_function.html

---

### [How-to Guides](./en_stable_src_how_to_guides.html.md)

**Summary:** The following guides explain important features of the library in detail and are dedicated to users with some experience. Unlike the API reference, they are step-by-step guides that explore what it is possible to do with specific abstractions of M...

**Sections:** Overview, Guides

**Keywords:** `guides`, `light`, `optimizer`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/how_to_guides.html

---

### [Inverse rendering tutorials](./en_stable_src_inverse_rendering_tutorials.html.md)

**Summary:** Mitsuba 3 can be used to solve inverse problems involving light using a technique known as differentiable rendering. This enables us to estimate physical attributes of a scene, e.g., reflectance, geometry, and lighting, from images. The following ...

**Keywords:** `inverse`, `light`, `rendering`, `scene`, `tutorials`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering_tutorials.html

---

### [Mesh I/O and manipulation](./en_stable_src_how_to_guides_mesh_io_and_manipulation.html.md)

**Summary:** Mitsuba provides an abstract Shape class to handle all geometric shapes. For triangle meshes, it has a concrete class Mesh that is further extended by 3 plugins which can load meshes directly from a file:

**Sections:** Reading a mesh from disk, Procedural mesh, Writing a mesh to disk, Adding and editing attributes

**Keywords:** `
# create an empty mesh (allocates buffers of the correct size)
mesh = mi.mesh(
    "wavydisk",
    vertex_count=n,
    face_count=n - 1,
    has_vertex_normals=false,
    has_vertex_texcoords=false,
)

`, `
# wavy disk construction
#
# let n define the total number of vertices, the first n-1 vertices will compose
# the fringe of the disk, while the last vertex should be placed at the center.
# the first n-1 vertices must have their height modified such that they oscillate
# with some given frequency and amplitude. to compute the face indices, we define
# the first vertex of every face to be the vertex at the center (idx=n-1) and the
# other two can be assigned sequentially (modulo n-2).

# disk with a wavy fringe parameters
n = 100
frequency = 12.0
amplitude = 0.4

# generate the vertex positions
theta = dr.linspace(mi.float, 0.0, dr.two_pi, n)
x, y = dr.sincos(theta)
z = amplitude * dr.sin(theta * frequency)
vertex_pos = mi.point3f(x, y, z)

# move the last vertex to the center
vertex_pos[dr.arange(mi.uint32, n) == n - 1] = 0.0

# generate the face indices
idx = dr.arange(mi.uint32, n - 1)
face_indices = mi.vector3u(n - 1, (idx + 1) % (n - 2), idx % (n - 2))

`, `
[(mesh[
  name = "wavydisk",
  bbox = boundingbox3f[
    min = [-0.999874, -0.999497, -0.399547],
    max = [0.999874, 1, 0.399547]
  ],
  vertex_count = 100,
  vertices = [1.17 kib of vertex data],
  face_count = 99,
  faces = [1.16 kib of face data],
  face_normals = 0
], {'vertex_positions', 'faces'})]

`, `
[(plymesh[
    name = "wavydisk.ply",
    bbox = boundingbox3f[
      min = [-0.999874, -0.999497, -0.399547],
      max = [0.999874, 1, 0.399547]
    ],
    vertex_count = 100,
    vertices = [3.52 kib of vertex data],
    face_count = 99,
    faces = [1.16 kib of face data],
    face_normals = 0,
    mesh attributes = [
      vertex_color: 3 floats
    ]
  ],
  {'vertex_color'})]

`, `
bunny = mi.load_dict({
    "type": "ply",
    "filename": "../scenes/meshes/bunny.ply",
    "face_normals": false,
    "to_world": mi.scalartransform4f().rotate([0, 0, 1], angle=10),
})

print(bunny)

`, `
copy to clipboard
`, `
copy to clipboard
![../../_images/src_how_to_guides_mesh_io_and_manipulation_12_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_how_to_guides_mesh_io_and_manipulation_12_0.png)
## writing a mesh to disk[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/mesh_io_and_manipulation.html#writing-a-mesh-to-disk "link to this heading")
no matter how a `, `
copy to clipboard
## adding and editing attributes[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/mesh_io_and_manipulation.html#adding-and-editing-attributes "link to this heading")
meshes in mitsuba can have additional attributes per face or per vertex. each attribute is either one or several floating point numbers, no other types are supported.
the `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/mesh_io_and_manipulation.html

---

### [Object pose estimation](./en_stable_src_inverse_rendering_object_pose_estimation.html.md)

**Summary:** In this tutorial, we will show how to optimize the pose of an object while correctly accounting for the visibility discontinuities. We are going to optimize several latent variables that control the translation and rotation of the object.

**Sections:** Overview, Setup, `direct_projective` and scene construction, Reference image, Optimizer and latent variables

**Keywords:** `
apply_transformation(params, opt)

img_init = mi.render(scene, seed=0, spp=1024)

mi.util.convert_to_bitmap(img_init)

`, `
copy to clipboard
`, `
copy to clipboard
##  `, `
copy to clipboard
## reference image[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/object_pose_estimation.html#reference-image "link to this heading")
next we generate the target rendering. we will later modify the bunny’s position and rotation to set the initial optimization state.
copy to clipboard
`, `
copy to clipboard
## visualizing the results[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/object_pose_estimation.html#visualizing-the-results "link to this heading")
finally, let’s visualize the results and plot the loss over iterations
copy to clipboard
`, `
copy to clipboard
copy to clipboard
![](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/object_pose_estimation.html)
## optimizer and latent variables[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/object_pose_estimation.html#optimizer-and-latent-variables "link to this heading")
as done in previous tutorial, we access the scene parameters using the `, `
copy to clipboard
copy to clipboard
![](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/object_pose_estimation.html)
in the following cell we define the hyper parameters controlling the optimization, such as the number of iterations and number of samples per pixels for the differentiable rendering simulation:
copy to clipboard
`, `
copy to clipboard
from the optimizer’s point of view, those variables are the same as any other variables optimized in the previous tutorials, to the exception that when calling `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/object_pose_estimation.html

---

### [Other tutorials](./en_stable_src_others_tutorials.html.md)

**Summary:** Mitsuba 3 is more than a retargetable & differentiable renderer. When used from Python, it can be viewed as a general rendering toolkit. In this section you will find tutorials that will teach you how to use Mitsuba 3 in contexts other than render...

**Keywords:** `bsdf`, `light`, `other`, `plugin`, `tutorials`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/others_tutorials.html

---

### [Polarization](./en_stable_src_key_topics_polarization.html.md)

**Summary:** ![../../images/teaser.jpg ](https://mitsuba.readthedocs.io/en/stable/images/teaser.jpg) The retargetable design of the Mitsuba 3 rendering system can be leveraged to optionally keep track of the full polarization state of light, meaning that it si...

**Sections:** Introduction, Mathematics of polarized light, Fresnel equations, Implementation

**Keywords:** `
 1// from `, `
 1// incident and outgoing directions, given in local space.
 2vector3fwo_local=...;
 3vector3fwi_local=...;
 4
 5// due to the coordinate system rotations for polarization-aware
 6// pbsdfs below we need to know the propagation direction of light.
 7// in the following, light arrives along `, `
 1bsdfcontextctx;
 2surfaceinteraction3fsi=...;
 3
 4// incident direction in world space ...
 5vector3fwi=...;
 6// ... and transformed into local space
 7si.wi=si.to_local(wi);
 8
 9// sample based on random numbers (in local space)
10auto[bs,bsdf_weight]=bsdf->sample(ctx,sampler.next_1d(),sampler.next_2d());
11
12// at this point, the mueller matrix `, `
 1bsdfcontextctx;// used to pass optional flags to bsdfs
 2surfaceinteraction3fsi=...;// returned from ray-scene intersections.
 3
 4// incident and outgoing directions in world space
 5vector3fwo=...;
 6vector3fwi=...;
 7
 8// transform into local space ...
 9vector3fwo_local=si.to_local(wo);
10si.wi=si.to_local(wi);
11
12// ... and evaluate the bsdf
13spectrumbsdf_val=bsdf->eval(ctx,si,wo_local);
14
15// at this point, the mueller matrix `, `
 1bsdfcontextctx;// used to pass optional flags to bsdfs
 2surfaceinteraction3fsi=...;// returned from ray-scene intersections.
 3
 4// incident and outgoing directions in world space
 5vector3fwo=...;
 6vector3fwi=...;
 7
 8// transform into local space ...
 9vector3fwo_local=si.to_local(wo);
10si.wi=si.to_local(wi);
11
12// ... and evaluate the pbsdf.
13spectrumbsdf_val=bsdf->eval(ctx,si,wo_local);
14
15// the returned mueller matrix `, `
 2
 3// "rotate" a mueller matrix in the general case (i.e. when both
 4// sides use a different basis).
 5//
 6// before:
 7//     matrix m operates from `, `
 2
 3// call a nested integrator (e.g. the path tracer)
 4auto[result,mask]=integrator->sample(scene,sampler,ray,...);
 5
 6// compute the implicit stokes reference for the incoming light path
 7vector3fbasis_out=mueller::stokes_basis(-ray.d);
 8
 9// get the camera transformation and evaluate for the current sampled `, `
 2
 3// construct the mueller matrix that rotates the reference frame of a stokes
 4// vector by an angle `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html

---

### [Polarized rendering](./en_stable_src_rendering_polarized_rendering.html.md)

**Summary:** In spirit, this example is similar to Mitsuba quickstart but we’ll highlight a few things to keep in mind regarding polarization-aware rendering in Mitsuba 3. 🕶 🚀 You will learn how to: Build a scene with polarization support in mind.

**Sections:** Overview, Polarized variants, Loading a scene, Rendering, Visualization

**Keywords:** `
bitmap = mi.bitmap(image, channel_names=['r', 'g', 'b'] + scene.integrator().aov_names())
bitmap.write('cbox_pol_output.exr')
bitmap

`, `
bitmap[
  pixel_format = multichannel,
  component_format = float32,
  size = [256, 256],
  srgb_gamma = 0,
  struct = struct<60>[
    float32 r; // @0, premultiplied alpha
    float32 g; // @4, premultiplied alpha
    float32 b; // @8, premultiplied alpha
    float32 s0.r; // @12, premultiplied alpha
    float32 s0.g; // @16, premultiplied alpha
    float32 s0.b; // @20, premultiplied alpha
    float32 s1.r; // @24, premultiplied alpha
    float32 s1.g; // @28, premultiplied alpha
    float32 s1.b; // @32, premultiplied alpha
    float32 s2.r; // @36, premultiplied alpha
    float32 s2.g; // @40, premultiplied alpha
    float32 s2.b; // @44, premultiplied alpha
    float32 s3.r; // @48, premultiplied alpha
    float32 s3.g; // @52, premultiplied alpha
    float32 s3.b; // @56, premultiplied alpha
  ],
  data = [ 3.75 mib of image data ]
]

`, `
channels = dict(bitmap.split())
print(channels.keys())

`, `
clipping input data to the valid range for imshow with rgb data ([0..1] for floats or [0..255] for integers).

`, `
copy to clipboard
`, `
copy to clipboard
![../../_images/src_rendering_polarized_rendering_17_1.png](https://mitsuba.readthedocs.io/en/stable/_images/src_rendering_polarized_rendering_17_1.png)
for the three stokes vector components that encode the polarization state, we use a divergent colormap (`, `
copy to clipboard
## loading a scene[¶](https://mitsuba.readthedocs.io/en/stable/src/rendering/polarized_rendering.html#loading-a-scene "link to this heading")
here we load a variation of the famous cornell box scene that includes a few glass and metal spheres. these types of materials polarize light that interacts with them and thus makes for a more interesting test scene.
copy to clipboard
`, `
copy to clipboard
as stated above, the [stokes](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_integrators.html#stokes-vector-integrator-stokes) integrator outputs many extra channels including the complete polarization state of the light arriving at the camera. this is the reason why the rendered tensorial image contains so many channels. in the following, we construct a [bitmap](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#bitmap) object with the proper channel names in order to write it to a exr file on disk.
copy to clipboard
`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/rendering/polarized_rendering.html

---

### [Polarizer optimization](./en_stable_src_inverse_rendering_polarizer_optimization.html.md)

**Summary:** An interesting feature of Mitsuba is its ability to account for the polarization state of light. This becomes even more powerful when combined with differentiable rendering. This tutorial demonstrates how those two concepts can be used together to...

**Sections:** Overview, Reference image, Setup optimization, Optimization, Results

**Keywords:** `
angles = []
losses = []

for it in range(iteration_count):
    # perform the differentiable rendering simulation
    image = mi.render(scene, params=params, seed=it, spp=1)

    # objective: no comparison against a reference, the goal is simply to make the image darker
    ob_val = dr.mean(image)

    # backpropagate loss to input parameters
    dr.backward(ob_val)

    # optimizer: take a gradient step
    opt.step()

    # apply rotation and update the scene parameters
    apply_rotation()

    print(f"iteration: {it:2}, rot: {opt['rotation'][0]:.4f}, loss: {ob_val}", end='\r')
    angles.append(opt['rotation'][0])
    losses.append(ob_val.array[0])

print()
print('optimization complete!')

`, `
copy to clipboard
`, `
copy to clipboard
## optimization[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html#optimization "link to this heading")
everything is now ready to run the optimization loop.
in the following cell we define the hyper parameters controlling our optimization loop, such as the number of iterations:
copy to clipboard
`, `
copy to clipboard
## results[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html#results "link to this heading")
we can now look at the optimized scene, which appears much darker as expected.
copy to clipboard
`, `
copy to clipboard
copy to clipboard
![](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html)
## setup optimization[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html#setup-optimization "link to this heading")
as in the previous tutorial on [pose estimation](https://mitsuba.readthedocs.io/en/latest/src/inverse_rendering/reparam_optimization.html), we setup the optimization using a latent variable to control the rotation of the filter. this rotation angle will be used to construct a transformation matrix that will be applied to all vertices of the filter’s mesh. for convenience, we define a function that does all of this, which we will also call later during the optimization loop.
it is important to apply the rotation once before starting the optimization loop as this will _bind_ the optimizer variable to the scene parameter. otherwise during backpropagation the gradients wouldn’t be propagate all the way to the optimizer’s variable.
copy to clipboard
`, `
copy to clipboard
copy to clipboard
![](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html)
we plot the filter rotation and image loss accross the optimization loop.
copy to clipboard
`, `
copy to clipboard
in this example the loss function doesn’t compare against a reference as the goal is simply to make the image darker. for this we simply use the sum of the pixel values in the rendered image.
copy to clipboard
`, `
copy to clipboard
we can then perform the rendering of our initial scene. as expected, the two filters are aligned and let linearly polarized light through.
copy to clipboard
`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/polarizer_optimization.html

---

### [Porting to Mitsuba 3.6.0](./en_stable_porting_3_6.html.md)

**Summary:** Mitsuba 3.6.0 contains a number of significant changes relative to the prior release, some breaking, that are predominantly driven by the dependence to the new version of Dr.Jit 1.0.0. This guide is intended to assist users of prior releases of Mi...

**Sections:** Symbolic control flow, Bitmap textures: Half-precision storage by default where possible, C++ interface changes, Miscellaneous

**Keywords:** `
  * rename of function decorator `, `
# don't do this!
@dr.syntax
defbad_code(x : mi.float):
  out : mi.float  = mi.float(0)
  if x < 2:
    out = mi.float(1)
  else
    out = mi.float(2)

  return out

x = dr.arange(mi.float, 5)
y = bad_code()

`, `
// registered getter as drjit_call_getter
uint32_tbase::getter()const{returnm_getter;}

myplugin(constproperties&props):base(props){
...
m_getter=m_components[0];
}

`, `
boolres=a==b;

`, `
boolres=dr::all(a==b);

`, `
copy to clipboard
## c++ interface changes[¶](https://mitsuba.readthedocs.io/en/stable/porting_3_6.html#c-interface-changes "link to this heading")
mitsuba 3.6.0 has also introduced changes that affect c++ developers who have extended mitsuba 3. as with the python interface, most of these changes are driven by dr.jit 1.0.0 and we again recommend users first begin by reading the [dr.jit documentation](https://drjit.readthedocs.io/en/latest/) and in particular the dedicated section on the [dr.jit c++ interface](https://drjit.readthedocs.io/en/latest/cpp.html).
### control flow[¶](https://mitsuba.readthedocs.io/en/stable/porting_3_6.html#control-flow "link to this heading")
analogous to dr.jit’s vectorized control flow changes in python, in c++ `, `
copy to clipboard
## miscellaneous[¶](https://mitsuba.readthedocs.io/en/stable/porting_3_6.html#miscellaneous "link to this heading")
  * dr.jit v1.0.0 raises the minimum supported llvm version to 11
  * rename of function `, `
copy to clipboard
###  `

**Source:** https://mitsuba.readthedocs.io/en/stable/porting_3_6.html

---

### [PyTorch and Mitsuba interoperability](./en_stable_src_inverse_rendering_pytorch_mitsuba_interoperability.html.md)

**Summary:** This tutorial shows how to mix differentiable computations between Mitsuba and PyTorch. The ability to combine these frameworks allows us to squeeze an entire rendering pipeline between neural layers whilst still preserving the differentiability (...

**Sections:** Overview, Setup, Load texture dataset, Scene construction, Wrap the rendering code

**Keywords:** `
@dr.wrap(source='torch', target='drjit')
defrender_texture(texture, spp=256, seed=1):
    params[key] = texture
    params.update()
    return mi.render(scene, params, spp=spp, seed=seed, seed_grad=seed+1)

`, `
classmodel1(nn.module):
    def__init__(self):
        super(model1, self).__init__()
        self.layers = nn.sequential(
            nn.linear(res**2, res**2),
            nn.sigmoid(),
        )

    defforward(self, texture):
        texture = texture.torch()
        # evaluate the model one channel as a time
        rgb = [self.layers(texture[:, :, i].view(-1)) for i in range(3)]
        # reconstruct and return the 3d tensor
        return torch.stack([c.view(res, res) for c in rgb], dim=2)

model = model1()
if 'cuda' in mi.variant():
    model = model.cuda()

`, `
copy to clipboard
`, `
copy to clipboard
![../../_images/src_inverse_rendering_pytorch_mitsuba_interoperability_16_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_pytorch_mitsuba_interoperability_16_0.png)
## instantiate a fully connected layer[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/pytorch_mitsuba_interoperability.html#instantiate-a-fully-connected-layer "link to this heading")
in this synthetic example, we don’t really need a neural network as we are only trying to optimize a mapping of pixels from the input texture image to a “pre-distorted” texture image that counteracts the effect of the refractive objects placed in front of the camera.
we are well aware that better techniques could be used to perform such task. moreover, further processing on the weights of the fully connected layer could be done to improve the convergence of the optimization (e.g. normalization). but for the sake of simplicity, in this tutorial we stick to the basics: a single fully connected layer followed by a sigmoid function to ensure the texture values lie in between `, `
copy to clipboard
![../../_images/src_inverse_rendering_pytorch_mitsuba_interoperability_20_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_pytorch_mitsuba_interoperability_20_0.png)
## optimization loop[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/pytorch_mitsuba_interoperability.html#optimization-loop "link to this heading")
this optimization loop is similar to the one you will find in any other beginner pytorch tutorial.
we first initialize an `, `
copy to clipboard
![../../_images/src_inverse_rendering_pytorch_mitsuba_interoperability_25_1.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_pytorch_mitsuba_interoperability_25_1.png)
## results[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/pytorch_mitsuba_interoperability.html#results "link to this heading")
as you can see in the results below, the fully connected layer is able to properly pre-distort the texture images so that the renderings approximatively match the input images.
these results are far from perfect, which is partially due to the simplicity of the pipeline implemented in this tutorial. for instance, it would make sense to use a smoothing regularization term on the distorted image to reduce the noise observed in the results below. other neural network architecture might also be more suited for this task.
copy to clipboard
`, `
copy to clipboard
![../../_images/src_inverse_rendering_pytorch_mitsuba_interoperability_8_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_pytorch_mitsuba_interoperability_8_0.png)
for the sake of simplicity in this tutorial, we will assume that all texture images have the same resolution. moreover, we will make sure that the pipeline renders images at that resolution to simplify the computation of the objective function.
copy to clipboard
`, `
copy to clipboard
## scene construction[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/pytorch_mitsuba_interoperability.html#scene-construction "link to this heading")
the scene/setup for this experiment is straighforward. first we instanciate a `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/pytorch_mitsuba_interoperability.html

---

### [Radiance Field Reconstruction (NeRF-like)](./en_stable_src_inverse_rendering_radiance_field_reconstruction.html.md)

**Summary:** In this tutorial, we will learn how to implement a 3D scene reconstruction pipeline similar to the one presented in the following work: ReLU Fields: The Little Non-linearity That Could. This simple NeRF-like method reconstructs the radiance field ...

**Sections:** Setup, Parameters, Creating multiple sensors, Rendering synthetic reference images, NeRF-like PRB integrator

**Keywords:** `
# rendering resolution
render_res = 256

# number of stages
num_stages = 4

# number of training iteration per stage
num_iterations_per_stage = 15

# learning rate
learning_rate = 0.2

# initial grid resolution
grid_init_res = 16

# spherical harmonic degree to be use for view-dependent appearance modeling
sh_degree = 2

# enable relu in integrator
use_relu = true

# number of sensors
sensor_count = 7

`, `
classradiancefieldprb(mi.python.ad.integrators.common.rbintegrator):
    def__init__(self, props=mi.properties()):
        super().__init__(props)
        self.bbox = mi.scalarboundingbox3f([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        self.use_relu = use_relu
        self.grid_res = grid_init_res
        # initialize the 3d texture for the density and sh coefficients
        res = self.grid_res
        self.sigmat = mi.texture3f(dr.full(mi.tensorxf, 0.01, shape=(res, res, res, 1)))
        self.sh_coeffs = mi.texture3f(dr.full(mi.tensorxf, 0.1, shape=(res, res, res, 3 * (sh_degree + 1) ** 2)))

    defeval_emission(self, pos, direction):
        spec = mi.spectrum(0)
        sh_dir_coef = dr.sh_eval(direction, sh_degree)
        sh_coeffs = self.sh_coeffs.eval(pos)
        for i, sh in enumerate(sh_dir_coef):
            spec += sh * mi.spectrum(sh_coeffs[3 * i:3 * (i + 1)])
        return dr.clip(spec, 0.0, 1.0)

    @dr.syntax
    defsample(self, mode, scene, sampler,
               ray, δl, state_in, active, **kwargs):
        primal = mode == dr.admode.primal

        ray = mi.ray3f(ray)
        hit, mint, maxt = self.bbox.ray_intersect(ray)

        active = mi.bool(active)
        active &= hit  # ignore rays that miss the bbox
        if not primal:  # if the gradient is zero, stop early
            active &= dr.any(δl != 0)

        step_size = mi.float(1.0 / self.grid_res)
        t = mi.float(mint) + sampler.next_1d(active) * step_size
        l = mi.spectrum(0.0 if primal else state_in)
        δl = mi.spectrum(δl if δl is not none else 0)
        β = mi.spectrum(1.0) # throughput

        while active:
            p = ray(t)
            with dr.resume_grad(when=not primal):
                sigmat = self.sigmat.eval(p)[0]
                if self.use_relu:
                    sigmat = dr.maximum(sigmat, 0.0)
                tr = dr.exp(-sigmat * step_size)
                # evaluate the directionally varying emission (weighted by transmittance)
                le = β * (1.0 - tr) * self.eval_emission(p, ray.d)

            β *= tr
            l = l + le if primal else l - le

            with dr.resume_grad(when=not primal):
                if not primal:
                    dr.backward_from(δl * (l * tr / dr.detach(tr) + le))

            t += step_size
            active &= (t < maxt) & dr.any(β != 0.0)


        return l if primal else δl, mi.bool(true), [], l

    deftraverse(self, cb):
        cb.put("sigmat", self.sigmat.tensor(), mi.paramflags.differentiable)
        cb.put('sh_coeffs', self.sh_coeffs.tensor(), mi.paramflags.differentiable)

    defparameters_changed(self, keys):
        self.sigmat.update_inplace()
        self.sh_coeffs.update_inplace()
        self.grid_res = self.sigmat.shape[0]

mi.register_integrator("rf_prb", lambda props: radiancefieldprb(props))

`, `
copy to clipboard
`, `
copy to clipboard
![../../_images/src_inverse_rendering_radiance_field_reconstruction_11_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_radiance_field_reconstruction_11_0.png)
## nerf-like prb integrator[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/radiance_field_reconstruction.html#nerf-like-prb-integrator "link to this heading")
unlike the other tutorials, this pipeline doesn’t use any of the conventional physically-based rendering algorithms such as path-tracing. instead, it implements a differentiable integrator for emissive volumes that directly uses a density and sh coefficient grid.
we define an integrator `, `
copy to clipboard
![../../_images/src_inverse_rendering_radiance_field_reconstruction_15_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_radiance_field_reconstruction_15_0.png)
## optimization[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/radiance_field_reconstruction.html#optimization "link to this heading")
we use an `, `
copy to clipboard
![../../_images/src_inverse_rendering_radiance_field_reconstruction_21_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_radiance_field_reconstruction_21_0.png)
![../../_images/src_inverse_rendering_radiance_field_reconstruction_21_1.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_radiance_field_reconstruction_21_1.png)
![../../_images/src_inverse_rendering_radiance_field_reconstruction_21_2.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_radiance_field_reconstruction_21_2.png)
![../../_images/src_inverse_rendering_radiance_field_reconstruction_21_3.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_radiance_field_reconstruction_21_3.png)
![../../_images/src_inverse_rendering_radiance_field_reconstruction_21_4.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_radiance_field_reconstruction_21_4.png)
![../../_images/src_inverse_rendering_radiance_field_reconstruction_21_5.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_radiance_field_reconstruction_21_5.png)
we can also take a closer look at one of the view point:
copy to clipboard
`, `
copy to clipboard
![../../_images/src_inverse_rendering_radiance_field_reconstruction_23_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_radiance_field_reconstruction_23_0.png)
when working with optimization pipelines, it is always very informative to take a look at the graph of objective function values.
copy to clipboard
`, `
copy to clipboard
## creating multiple sensors[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/radiance_field_reconstruction.html#creating-multiple-sensors "link to this heading")
as done in many of the other tutorials, we instantiate a couple of sensors to render our synthetic scene from different viewpoints. here the cameras are placed in a circle around the `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/radiance_field_reconstruction.html

---

### [Rendering tutorials](./en_stable_src_rendering_tutorials.html.md)

**Summary:** The best place to start is with user-friendly tutorials that will show you how to use Mitsuba 3 with complete, end-to-end examples. The following tutorials cover the basic operations ones might need to know when it comes to rendering with Mitsuba 3:

**Keywords:** `light`, `rendering`, `scene`, `tutorials`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/rendering_tutorials.html

---

### [Shape optimization](./en_stable_src_inverse_rendering_shape_optimization.html.md)

**Summary:** In this tutorial, we will optimize a triangle mesh to match a target shape specfied using a set of reference renderings. Gradients with regards to vertex positions are typically extremely sparse, since only vertices located on visibility discontin...

**Sections:** Overview, Setup, Setting up sensors, Scene construction, Naive optimization

**Keywords:** `
!pip
`, `
!pip;

`, `
# compute average edge length
l0 = np.linalg.norm(v_np[f_np[:,0]] - v_np[f_np[:,1]], axis=1)
l1 = np.linalg.norm(v_np[f_np[:,1]] - v_np[f_np[:,2]], axis=1)
l2 = np.linalg.norm(v_np[f_np[:,2]] - v_np[f_np[:,0]], axis=1)

target_l = np.mean([l0, l1, l2]) / 2

`, `
# reset the scene
scene_source = mi.load_dict(scene_dict)
params = mi.traverse(scene_source)

`, `
# update the mesh after the last iteration's gradient step
params['shape.vertex_positions'] = ls.from_differential(opt['u'])
params.update();

`, `
copy to clipboard
`, `
copy to clipboard
![../../_images/src_inverse_rendering_shape_optimization_12_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_shape_optimization_12_0.png)
the starting geometry which we’ll optimize is a relatively dense sphere. more challenging scenes and target shapes might require a better initialization.
copy to clipboard
`, `
copy to clipboard
![../../_images/src_inverse_rendering_shape_optimization_14_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_shape_optimization_14_0.png)
## naive optimization[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html#naive-optimization "link to this heading")
it is tempting to apply the same steps as in previous tutorials, i.e. simply render the scene, compute a loss, and differentiate. let’s try that first.
as per usual, all that is needed is an `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html

---

### [Testing](./en_stable_src_developer_guide_testing.html.md)

**Summary:** To run the test suite, simply invoke pytest:  pytest  Copy to clipboard If you want to skip the execution of the slower tests, you can do so by running the test suite with the following flag.  'not slow'

**Sections:** Testing multiple variants, Chi^2 tests, Rendering test suite and Z-test

**Keywords:** `
# or to run a single test file
pytest
`, `
# this test will run on all available variants
deftest_hello_world(variants_all):
    print(f'hello {mi.variant()}')
    assert true

`, `
'not slow'

`, `
copy to clipboard
here is the figure generated by the `, `
copy to clipboard
here is the list of available fixtures:
  * variants_all: execute on all variants
  * variants_all_scalar: execute on all `, `
copy to clipboard
if you want to skip the execution of the slower tests, you can do so by running the test suite with the following flag.
`, `
copy to clipboard
in case of failure, the target density and histogram were written to `, `
copy to clipboard
the build system also exposes a `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/developer_guide/testing.html

---

### [Textures](./en_stable_src_generated_plugins_textures.html.md)

**Summary:** The following section describes the available texture data sources. In Mitsuba 3, textures are objects that can be attached to certain surface scattering model parameters to introduce spatial variation. In the documentation, these are listed as su...

**Sections:** Bitmap texture (bitmap), Checkerboard texture (checkerboard), Mesh attribute texture (mesh_attribute), Volumetric texture (volume)

**Keywords:** `
'type': 'bitmap',
'filename': 'texture.png',
'wrap_mode': 'mirror'

`, `
'type': 'checkerboard',
'color0': [0.1, 0.1, 0.1],
'color1': [0.5, 0.5, 0.5]

`, `
'type': 'ply',
'filename': 'my_mesh_with_vertex_color_attr.ply',
'bsdf': {
    'type': 'diffuse',
    'reflectance': {
        'type': 'mesh_attribute',
        'name': 'vertex_color'
    }
}

`, `
'type': 'scene',

# .. scene contents ..

# create a bsdf that supports textured parameters
'my_textured_material': {
    'type': '<bsdf_type>',
    '<parameter_name>' : {
        'type': '<texture_type>',
        # .. texture parameters ..
        'to_uv': mi.scalar_rgb.scalartransform4f.scale([2, 2, 0]).translate([0.5, 1.0, 0]) # third dimension is ignored
    }

    # .. non-spatially varying bsdf parameters ..
}

`, `
'type': 'scene',

# .. scene contents ..

'texture_id': {
    'type': '<texture_type>',
    # .. texture parameters ..
},

# create a bsdf that supports textured parameters
'my_textured_material': {
    'type': '<bsdf_type>',
    '<parameter_name>' : {
        'type' : 'ref',
        'id' : 'texture_id'
    }

    # .. non-spatially varying bsdf parameters ..
}

`, `
'type': 'volume',
'volume': {
    'type': 'gridvolume',
    'filename': 'my_volume.vol'
}

`, `
<sceneversion="3.0.0">
<!-- create a bsdf that supports textured parameters -->
<bsdftype=".. bsdf type .."id="my_textured_material">
<texturetype=".. texture type .."name=".. parameter name ..">
<!-- .. texture parameters go here .. -->

<transformname="to_uv">
<!-- scale texture by factor of 2 -->
<scalex="2"y="2"/>
<!-- offset texture by [0.5, 1.0] -->
<translatex="0.5"y="1.0"/>
</transform>
</texture>

<!-- .. non-spatially varying bsdf parameters ..-->
</bsdf>
</scene>

`, `
<sceneversion="3.0.0">
<!-- create a named texture at the top level -->
<texturetype=".. texture type .."id="my_named_texture">
<!-- .. texture parameters go here .. -->
</texture>

<!-- create a bsdf that supports textured parameters -->
<bsdftype=".. bsdf type ..">
<!-- example of referencing a named texture -->
<refid="my_named_texture"name=".. parameter name .."/>

<!-- .. non-spatially varying bsdf parameters ..-->
</bsdf>
</scene>

`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html

---

### [Transformation toolbox](./en_stable_src_how_to_guides_transformation_toolbox.html.md)

**Summary:** This how-to guide explores the different tools available in Mitsuba 3 to manipulate cartesian coordinate systems. When generating datasets, researching advanced light transport algorithms, or developing new appearance models, you will quickly real...

**Sections:** Frame, Transform

**Keywords:** `
[1, -1, 0.959596]

`, `
[1, 2, 0.292929]

`, `
[1, 2, 3]

`, `
[6, 2, 2]

`, `
copy to clipboard
`, `
copy to clipboard
## frame[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#frame "link to this heading")
the [frame3f](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.frame3f) class stores a three-dimensional orthonormal coordinate frame. this class is very handy when you wish to convert vectors between different cartesian coordinates systems.
### frame initialization[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#frame-initialization "link to this heading")
a `, `
copy to clipboard
### applying transforms[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#applying-transforms "link to this heading")
the python `, `
copy to clipboard
### converting to/from local frames[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#converting-to/from-local-frames "link to this heading")
the two methods below are the main operations you will be using to convert between different coordinate frames.
  * [frame3f.to_local()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.frame3f.to_local)
  * [frame3f.to_world()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.frame3f.to_world)


copy to clipboard
`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html

---

### [Variants in C++](./en_stable_src_developer_guide_variants_cpp.html.md)

**Summary:** As described in the section on choosing variants, Mitsuba 3 code can be compiled into different variants, which are parameterized by their computational backend and representation of color. To enable such retargeting from a single implementation, ...

**Sections:** Type aliases, Branching and masking, JIT backend synchronization point, Pointer types, Variant-specific code

**Keywords:** `
// --------------------
// scalar code (discouraged)

scenescene=...;
ray3fray=...;
surfaceinteraction3fsi=scene->ray_intersect(ray);

if(si.is_valid())
return1.f;
else
return0.f;

// --------------------
// generic code

scenescene=...;
ray3fray=...;
surfaceinteraction3fsi=scene->ray_intersect(ray);

returndr::select(si.is_valid(),1.0f,0.f);

`, `
// imports bsdfptr, emitterptr, etc..
mi_import_types()

scenescene=...;
maskactive=...;
ray3fray=...;
surfaceinteraction3fsi=scene->ray_intersect(ray,active);

// array of pointers if float is an array
bsdfptrbsdf=si.bsdf();

// dr.jit is able to dispatch method calls involving arrays of pointers
bsdf->eval(...,active);

`, `
// mask specifying the active lanes
maskactive=...;

scenescene=...;
ray3fray=...;
surfaceinteraction3fsi=scene->ray_intersect(ray,active);

returndr::select(active&si.is_valid(),1.0f,0.f);

`, `
copy to clipboard
## jit backend synchronization point[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/variants_cpp.html#jit-backend-synchronization-point "link to this heading")
as described in [dr.jit’s documentation](https://enoki.readthedocs.io/en/master/gpu.html#suggestions-regarding-horizontal-operations), the `, `
copy to clipboard
in `, `
copy to clipboard
more information on vectorized method calls is provided in the [dr.jit documentation](https://enoki.readthedocs.io/en/master/calls.html).
## variant-specific code[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/variants_cpp.html#variant-specific-code "link to this heading")
the c++17 `, `
copy to clipboard
moreover, most of the functions/methods take an _optional_ `, `
copy to clipboard
note
the `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/developer_guide/variants_cpp.html

---

### [Volumetric inverse rendering](./en_stable_src_inverse_rendering_volume_optimization.html.md)

**Summary:** In this tutorial, we use Mitsuba’s differentiable volumetric path tracer to optimize a scattering volume to match a set of (synthetic) reference images. We will optimize a 3D volume density that’s stored on a regular grid. The optimization will ac...

**Sections:** Overview, Setup, Creating multiple sensors, Rendering synthetic reference images, Setting up the optimization scene

**Keywords:** `
copy to clipboard
`, `
copy to clipboard
![../../_images/src_inverse_rendering_volume_optimization_10_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_volume_optimization_10_0.png)
## setting up the optimization scene[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#setting-up-the-optimization-scene "link to this heading")
our goal is now to optimize a 3d volume density (also called extinction) to match the previously generated reference images. for this we create a second scene, where we replace the reference volume by a simple uniform initialization.
to initialize a volume grid from python, we use the [volumegrid](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.volumegrid) object in conjunction with [tensorxf](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.tensorxf). the `, `
copy to clipboard
![../../_images/src_inverse_rendering_volume_optimization_14_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_volume_optimization_14_0.png)
## optimization[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#optimization "link to this heading")
we instantiate an `, `
copy to clipboard
![../../_images/src_inverse_rendering_volume_optimization_21_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_volume_optimization_21_0.png)
## volume upsampling[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#volume-upsampling "link to this heading")
the results above don’t look great. one reason is the low resolution of the optimized volume grid. let’s try to increase the resolution of the current grid and continue the optimization for a few more iterations. in practice it is almost always beneficial to leverage such a “multi-resolution” approach. at low resolution, the optimization will recover the overall shape, exploring a much simpler solution landscape. moving on to a volume with a higher resolution allows recovering additional detail, while using the coarser solution as a starting point.
luckily dr.jit provides [dr.upsample()](https://drjit.readthedocs.io/en/latest/reference.html#drjit.upsample), a functions for up-sampling tensor and texture data. we can easily create a higher resolution volume by passing the current optimzed tensor and specifying the desired shape (must be powers of two when upsampling `, `
copy to clipboard
![../../_images/src_inverse_rendering_volume_optimization_25_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_inverse_rendering_volume_optimization_25_0.png)
## continuing the optimization[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#continuing-the-optimization "link to this heading")
let’s now run our optimization loop for a few more iterations with the upscaled volume grid.
🗒 **note**
the optimizer automatically resets the internal state (e.g., momentum) associated to the optimized variable when it detects a size change.
copy to clipboard
`, `
copy to clipboard
## creating multiple sensors[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#creating-multiple-sensors "link to this heading")
we cannot hope to obtain a robust volumetric reconstruction using only a single reference image. multiple viewpoints are needed to sufficiently constrain the reconstructed volume density. using a multi-view optimization we can recover volume parameters that generalize to novel views (and illumination conditions).
in this tutorial, we use 5 sensors placed on a half circle around the origin. for the simple optimization in this tutorial this is sufficient, but more complex scenes may require using significantly more views (e.g., using 50-100 sensors is not unreasonable).
copy to clipboard
`, `
copy to clipboard
## final results[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#final-results "link to this heading")
finally we can render the final volume from the different view points and compare the images to the reference images.
copy to clipboard
`, `
copy to clipboard
## intermediate results[¶](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html#intermediate-results "link to this heading")
we have only performed a few iterations so far and can take a look at the current results.
copy to clipboard
`

**Source:** https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/volume_optimization.html

---

### [Writing documentation](./en_stable_src_developer_guide_documentation.html.md)

**Summary:** Mitsuba uses a multi-stage documentation generation process that combines C++ docstring extraction, plugin documentation generation, and Sphinx-based HTML generation. This guide explains how the system works and how to build documentation.

**Sections:** Prerequisites, Documentation sources, Build process overview, Detailed build steps, Notebook tutorials

**Keywords:** `
# extract c++ docstrings → include/mitsuba/python/docstr.h
ninja# build main library and python bindings
ninja# generate api reference documentation
ninja# build final html documentation

`, `
copy to clipboard
## detailed build steps[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/documentation.html#detailed-build-steps "link to this heading")
  1. **docstring extraction** (`, `
copy to clipboard
in order to hide a cell of a notebook in the documentation, add the following to the metadata of that cell:
`, `
{
"nbsphinx":"hidden"
}

`, `
{
"nbsphinx-thumbnail":{}
}

`, ` by running plugin extraction, processing notebooks, and combining all sources.


## notebook tutorials[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/documentation.html#notebook-tutorials "link to this heading")
we are using the [nbsphinx](https://nbsphinx.readthedocs.io/) sphinx extension to render our tutorials in the online documentation.
the thumbnail of a notebook in the gallery can be the output image of a cell in the notebook. for this, simply add the following to the metadata of that cell:
`, ` for python bindings.
  2. **main build** (`, ` to generate `

**Source:** https://mitsuba.readthedocs.io/en/stable/src/developer_guide/documentation.html

---

## How to Use This Index

### Method 1: Search by Category
Use the Quick Navigation links above to jump to specific topics.

### Method 2: Search by Keyword
Check `index.json` for the complete keyword-to-file mapping. For example:
```bash
# Search for files about 'rendering'
cat index.json | grep -A 5 '"rendering"'
```

### Method 3: Use Grep
Search across all documentation files:
```bash
grep -r 'path tracing' .
```

### Method 4: Use index.json Programmatically
```python
import json
with open('index.json') as f:
    index = json.load(f)
    # Find files by category
    rendering_docs = index['by_category']['Rendering']
    # Find files by topic
    integrator_docs = index['topics']['integrator']
```
