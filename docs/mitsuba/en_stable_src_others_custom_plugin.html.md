---
url: https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html
crawled_at: 2025-11-13T17:15:21.109591
title: https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Download notebook ](https://raw.githubusercontent.com/mitsuba-renderer/mitsuba-tutorials/master/others/custom_plugin.ipynb) [ Download data ](https://d38rqfq1h7iukm.cloudfront.net/scenes/tutorials/scenes.zip)
# Custom Python plugin[¶](https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html#Custom-Python-plugin "Link to this heading")
## Overview[¶](https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html#Overview "Link to this heading")
In Mitsuba it easy to add custom code for BSDFs, integrators, emitters, sensors, and more. This tutorial will show you how to create custom plugins in Python and register them for use.
To illustrate this, we are going to implement a new _tinted dielectric_ BSDF, that behaves much like a regular [dielectric BSDF](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#smooth-dielectric-material-dielectric) but adds a colorful tint to the reflections at grazing angle. We will then register this new BSDF and use it to render a simple scene.
🚀 **You will learn how to:**
  * Create a custom BSDF plugin in Python
  * Register a custom plugin to the system
  * Use a registered custom plugin


## Setup[¶](https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html#Setup "Link to this heading")
Custom Mitsuba plugins written in Python work best with Just-In-Time (JIT) variants. This is because it would be pretty inefficient to execute Python BSDF code for millions of _scalar_ light paths. JIT variants, on the other hand, only execute a few calls to those methods on arrays containing millions of entries at once, mitigating the overhead coming from the the Python layer. In this example we will therefore stick with the `llvm_ad_rgb` variant.
Copy to clipboard
```
importdrjitasdr
importmitsubaasmi

mi.set_variant('llvm_ad_rgb')

```
Copy to clipboard
## Implementation[¶](https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html#Implementation "Link to this heading")
As mentioned in the tutorial overview, we are going to implement a tinted dielectric BSDF in this tutorial. This code is very similar to the actual [C++ implementation](https://github.com/mitsuba-renderer/mitsuba3/blob/master/src/bsdfs/dielectric.cpp) of the [dielectric BSDF](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html#smooth-dielectric-material-dielectric), so we will not look at it in great detail.
First, our BSDF Python class `MyBSDF` needs to inherit from [BSDF](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.BSDF). This allows us to override the constructor method as well as [sample()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.BSDF), [eval()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.BSDF) and [pdf()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.BSDF).
The constructor takes a [Properties](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.Properties) object as an argument, which can be used to read parameters defined in the XML scene description or passed in the [load_dict()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.load_dict) dictionary. Here we read the index of refraction ratio `eta` as well as the tint color `tint` from `props`. In the constructor, we also properly set the BSDF members used in other methods like `m_flags` and `m_components`.
Similarly to the regular dielectric BSDF, the `eval()` and `pdf()` methods of our custom BSDF should always return `0.0` and never be called as it is a degenerate BSDF described by a Dirac delta distribution.
Regarding the `sample()` method, apart from the computation of the tinted reflection value `value_r`, the rest of code should be identical to the [C++ implementation](https://github.com/mitsuba-renderer/mitsuba3/blob/master/src/bsdfs/dielectric.cpp) of `dielectric`.
Note that it is also possible to override the [to_string()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.Object.to_string) method which is called in any printing/logging routine.
Finally, we override the implementation of the [traverse()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.Object.traverse) and [parameters_changed()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.Object.parameters_changed) methods to expose the `tint` parameter via the _traverse_ mechanism. This will allow us to edit this parameter after the BSDF is instanciated.
Copy to clipboard
```
classMyBSDF(mi.BSDF):
    def__init__(self, props):
        mi.BSDF.__init__(self, props)

        # Read 'eta' and 'tint' properties from `props`
        self.eta = 1.33
        if props.has_property('eta'):
            self.eta = props['eta']

        self.tint = mi.Color3f(props['tint'])

        # Set the BSDF flags
        reflection_flags   = mi.BSDFFlags.DeltaReflection   | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        transmission_flags = mi.BSDFFlags.DeltaTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components  = [reflection_flags, transmission_flags]
        self.m_flags = reflection_flags | transmission_flags

    defsample(self, ctx, si, sample1, sample2, active):
        # Compute Fresnel terms
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        r_i, cos_theta_t, eta_it, eta_ti = mi.fresnel(cos_theta_i, self.eta)
        t_i = dr.maximum(1.0 - r_i, 0.0)

        # Pick between reflection and transmission
        selected_r = (sample1 <= r_i) & active

        # Fill up the BSDFSample struct
        bs = mi.BSDFSample3f()
        bs.pdf = dr.select(selected_r, r_i, t_i)
        bs.sampled_component = dr.select(selected_r, mi.UInt32(0), mi.UInt32(1))
        bs.sampled_type      = dr.select(selected_r, mi.UInt32(+mi.BSDFFlags.DeltaReflection),
                                                     mi.UInt32(+mi.BSDFFlags.DeltaTransmission))
        bs.wo = dr.select(selected_r,
                          mi.reflect(si.wi),
                          mi.refract(si.wi, cos_theta_t, eta_ti))
        bs.eta = dr.select(selected_r, 1.0, eta_it)

        # For reflection, tint based on the incident angle (more tint at grazing angle)
        value_r = dr.lerp(mi.Color3f(self.tint), mi.Color3f(1.0), dr.clip(cos_theta_i, 0.0, 1.0))

        # For transmission, radiance must be scaled to account for the solid angle compression
        value_t = mi.Color3f(1.0) * dr.square(eta_ti)

        value = dr.select(selected_r, value_r, value_t)

        return (bs, value)

    defeval(self, ctx, si, wo, active):
        return 0.0

    defpdf(self, ctx, si, wo, active):
        return 0.0

    defeval_pdf(self, ctx, si, wo, active):
        return 0.0, 0.0

    deftraverse(self, cb):
        cb.put('tint', self.tint, mi.ParamFlags.Differentiable)

    defparameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

    defto_string(self):
        return ('MyBSDF[\n'
                '    eta=%s,\n'
                '    tint=%s,\n'
                ']' % (self.eta, self.tint))

```
Copy to clipboard
## Plugin registration[¶](https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html#Plugin-registration "Link to this heading")
There’s only one more thing to do before we can use our custom BSDF in scenes. We need to register it in the system so it can be used. This can be done by calling the [register_bsdf()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.register_bsdf) function and specifying the name to be used to instantiate this plugin. The function takes a _constructor lambda_ function as the second parameter.
📑 **Note**
Similar functions exist for other types of plugins, e.g.
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


Copy to clipboard
```
mi.register_bsdf("mybsdf", lambda props: MyBSDF(props))

```
Copy to clipboard
## Plugin instantiation[¶](https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html#Plugin-instantiation "Link to this heading")
You can now use this plugin like you would with any other BSDF plugin and set the appropriate properties of the BSDF expected in its constructor in the XML or `dict` representation.
Copy to clipboard
```
my_bsdf = mi.load_dict({
    'type' : 'mybsdf',
    'tint' : [0.2, 0.9, 0.2],
    'eta' : 1.33
})

my_bsdf

```
Copy to clipboard
Copy to clipboard
```
MyBSDF[
    eta=1.33,
    tint=[[0.2, 0.9, 0.2]],
]

```
Copy to clipboard
## Rendering[¶](https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html#Rendering "Link to this heading")
Finally, let’s use our custom BSDF in an actual scene and render it to see how our tinted BSDF looks like.
Copy to clipboard
```
scene = mi.load_dict({
    'type': 'scene',
    'integrator': {
        'type': 'path'
    },
    'light': {
        'type': 'constant',
        'radiance': 0.99,
    },
    'sphere' : {
        'type': 'sphere',
        'bsdf': my_bsdf
    },
    'sensor': {
        'type': 'perspective',
        'to_world': mi.ScalarTransform4f().look_at(origin=[0, -5, 5],
                                                 target=[0, 0, 0],
                                                 up=[0, 0, 1]),
    }
})

image = mi.render(scene)

mi.Bitmap(image).convert(srgb_gamma=True)

```
Copy to clipboard
Copy to clipboard
![](https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html)
## Edit parameters[¶](https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html#Edit-parameters "Link to this heading")
As expected, it is possible to access our custom BSDF’s parameters using the [traverse](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.traverse) mechanism.
Copy to clipboard
```
params = mi.traverse(scene)
params

```
Copy to clipboard
Copy to clipboard
```
SceneParameters[
  ----------------------------------------------------------------------------------------
  Name                                 Flags    Type           Parent
  ----------------------------------------------------------------------------------------
  light.sampling_weight                         float          ConstantBackgroundEmitter
  light.radiance.value                 ∂        Float          UniformSpectrum
  sensor.near_clip                              float          PerspectiveCamera
  sensor.far_clip                               float          PerspectiveCamera
  sensor.shutter_open                           float          PerspectiveCamera
  sensor.shutter_open_time                      float          PerspectiveCamera
  sensor.film.size                              ScalarVector2u HDRFilm
  sensor.film.crop_size                         ScalarVector2u HDRFilm
  sensor.film.crop_offset                       ScalarPoint2u  HDRFilm
  sensor.x_fov                         ∂, D     Float          PerspectiveCamera
  sensor.principal_point_offset_x      ∂, D     Float          PerspectiveCamera
  sensor.principal_point_offset_y      ∂, D     Float          PerspectiveCamera
  sensor.to_world                      ∂, D     Transform4f    PerspectiveCamera
  sphere.bsdf.tint                     ∂        Color3f        BSDF
  sphere.silhouette_sampling_weight             float          Sphere
  sphere.to_world                      ∂, D     Transform4f    Sphere
]

```
Copy to clipboard
We can then update the `tint` value:
Copy to clipboard
```
key = 'sphere.bsdf.tint'
params[key] = mi.Color3f(0.9, 0.2, 0.2)
params.update();

```
Copy to clipboard
```
🏝️ there is nothing to do here 🏝️

```
Copy to clipboard
When re-rendering this scene, we now see that the new tint is indeed used!
Copy to clipboard
```
image = mi.render(scene)

mi.Bitmap(image).convert(srgb_gamma=True)

```
Copy to clipboard
Copy to clipboard
![](https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html)
## See also[¶](https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html#See-also "Link to this heading")
  * [mitsuba.BSDF](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.BSDF)
  * [mitsuba.register_bsdf()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.register_bsdf)


[**Simplify infrastructure** with MongoDB Atlas, the leading developer data platform](https://server.ethicalads.io/proxy/click/9504/019a7c7f-6bb8-7533-b095-45109ba5ce7d/)
[Ads by EthicalAds](https://www.ethicalads.io/advertisers/?ref=ea-text)
* * *
