---
url: https://mitsuba.readthedocs.io/en/stable/src/plugin_reference.html
crawled_at: 2025-11-13T17:10:34.673817
title: https://mitsuba.readthedocs.io/en/stable/src/plugin_reference.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/plugin_reference.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/plugin_reference.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[![../_images/banner_06.jpg](https://mitsuba.readthedocs.io/en/stable/_images/banner_06.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/banner_06.jpg)
# Plugin reference[¶](https://mitsuba.readthedocs.io/en/stable/src/plugin_reference.html#plugin-reference "Link to this heading")
  * [BSDFs](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html)
  * [Emitters](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html)
  * [Films](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_films.html)
  * [Integrators](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html)
  * [Participating media](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_media.html)
  * [Phase functions](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_phase.html)
  * [Reconstruction filters](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_rfilters.html)
  * [Samplers](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html)
  * [Sensors](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_sensors.html)
  * [Shapes](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html)
  * [Spectra](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html)
  * [Textures](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html)
  * [Volumes](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_volumes.html)


## Overview[¶](https://mitsuba.readthedocs.io/en/stable/src/plugin_reference.html#overview "Link to this heading")
The subsections above describe the available Mitsuba 3 plugins, usually along with example renderings and a description of what each parameter does. They are separated into subsections covering textures, surface scattering models, etc.
The documentation of a plugin always starts with a table similar to the one below:
Parameter | Type | Description | Flags  
---|---|---|---  
soft_rays | boolean | Try not to damage objects in the scene by shooting softer rays (Default: false) |   
dark_matter | float | Controls the proportionate amount of dark matter present in the scene. (Default: 0.42) |   
(Nested plugin) | integrator | A nested integrator which does the actual hard work |   
Suppose this hypothetical plugin is an integrator named `amazing`. Then, based on this description, it can be instantiated from an XML scene file using a custom configuration such as:
XMLPython
```
<integratortype="amazing">
<booleanname="softer_rays"value="true"/>
<floatname="dark_matter"value="0.44"/>
</integrator>

```
Copy to clipboard
```
{
    'type': 'amazing',
    'softer_rays': True,
    'dark_matter': 0.44
}

```
Copy to clipboard
In some cases, plugins also indicate that they accept nested plugins as input arguments. These can either be _named_ or _unnamed_. If the `amazing` integrator also accepted the following two parameters:
Parameter | Type | Description | Flags  
---|---|---|---  
(Nested plugin) | integrator | A nested integrator which does the actual hard work |   
puppies | texture | This must be used to supply a cute picture of puppies |   
then it can be instantiated e.g. as follows:
XMLPython
```
<integratortype="amazing">
<booleanname="softer_rays"value="true"/>
<floatname="dark_matter"value="0.44"/>
<!-- Nested unnamed integrator -->
<integratortype="path"/>
<!-- Nested texture named puppies -->
<texturename="puppies"type="bitmap">
<stringname="filename"value="cute.jpg"/>
</texture>
</integrator>

```
Copy to clipboard
```
{
    'type': 'amazing',
    'softer_rays': True,
    'dark_matter': 0.44,
    # Nested unnamed integrator
    'foo': {
        'type': 'path'
    },
    # Nested texture named puppies
    'puppies': {
        'type': 'bitmap',
        'filename': 'cute.jpg'
    }
}

```
Copy to clipboard
### Flags[¶](https://mitsuba.readthedocs.io/en/stable/src/plugin_reference.html#flags "Link to this heading")
Each parameter optionally has flags that are listed in the last column. These flags indicate whether the parameter is differentiable or not, or whether it introduces discontinuities and thus needs special treatment.
See [`mitsuba.ParamFlags`](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.ParamFlags "mitsuba.ParamFlags") for their documentation.
## Scene-wide attributes[¶](https://mitsuba.readthedocs.io/en/stable/src/plugin_reference.html#scene-wide-attributes "Link to this heading")
The Scene object exposes scene-wide attributes.
**Embree BVH mode:** We expose a scene-level flag to enable Embree’s “robust” mode. Enabling this flag makes Embree use slightly slower but more robust ray intersection computations. Embree’s default “fast” mode can miss ray intersections when a ray perfectly hits the edge between two adjacent triangles. Enabling the robust intersection mode fixes that in most cases. Note that Embree cannot guarantee that all intersections are reported for rays that exactly hit a vertex, which is a known limitation.
**Thread reordering:** Ray intersection methods on the scene can take a `reorder` argument to specifiy whether or not threads should be shuffled into coherent groups after the intersection (shader execution reordering). Both this flag and the one passed to the method must be set in order for the reordering to take place. By splitting this responsability, we allow users to indiciate if rendering the scene benefits from SER without having to modify the ray intersection call site. This feature is only relevant for the CUDA backend and has no effect in scalar or LLVM variants.
Parameter | Type | Description | Flags  
---|---|---|---  
embree_use_robust_intersection | bool | Whether Embree uses the robust mode flag `RTC_SCENE_FLAG_ROBUST` (Default: false). |   
allow_thread_reordering | bool | Whether or not to reorder threads into coherent groups after a ray intersection if requested (Default: true). | P  
When creating a scene, the scene-wide attributes can be specified as follows:
XMLPython
```
<sceneversion="3.0.0">
<booleanname="embree_use_robust_intersection"value="true"/>
</scene>

```
Copy to clipboard
```
{
    'type': 'scene',
    'embree_use_robust_intersection': True,
    'allow_thread_reordering': True,
}

```
Copy to clipboard
* * *
  *[P]: This parameter will be exposed as a scene parameter
