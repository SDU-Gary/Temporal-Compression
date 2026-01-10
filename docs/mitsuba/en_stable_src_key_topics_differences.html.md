---
url: https://mitsuba.readthedocs.io/en/stable/src/key_topics/differences.html
crawled_at: 2025-11-13T17:10:50.689427
title: https://mitsuba.readthedocs.io/en/stable/src/key_topics/differences.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/key_topics/differences.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/key_topics/differences.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Differences to previous versions[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/differences.html#differences-to-previous-versions "Link to this heading")
## Differences to Mitsuba 2[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/differences.html#differences-to-mitsuba-2 "Link to this heading")
Porting Mitsuba 2 Python scripts and C++ plugins to Mitsuba 3 is straightforward as Mitsuba 3 only underwent minimal API changes. Moreover, Mitsuba 3 maintains scene compatibility with Mitsuba 2.
### Enoki → Dr.Jit[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/differences.html#enoki-dr-jit "Link to this heading")
One of the biggest changes is the replacement of the Enoki backend with Dr.Jit. To account for this in your code, simply replace the namespace definition and aliases as follows:
C++Python
```
// Before
namespaceek=enoki;
...=ek::sin(...);

// Now
namespacedr=drjit;
...=dr::sin(...);

```
Copy to clipboard
```
# Before
importenokiasek
... = ek.sin(...)

# Now
importdrjitasdr
... = dr.sin(...)

```
Copy to clipboard
Moreover, note the following non-exhaustive list API changes in Dr.Jit (also valid for C++):
  * `ek.zero(..)` → `dr.zeros(..)`
  * `ek.min(..)` → `dr.minimum(..)`
  * `ek.max(..)` → `dr.maximum(..)`
  * `ek.hmin(..)` → `dr.min(..)`
  * `ek.hmax(..)` → `dr.max(..)`
  * `ek.hsum(..)` → `dr.sum(..)`
  * `ek.hprod(..)` → `dr.prod(..)`
  * `ek.hmean(..)` → `dr.mean(..)`
  * `ek.hsum_async(..)` → `dr.sum(..)`
  * `ek.hprod_async(..)` → `dr.prod(..)`
  * `ek.hmean_async(..)` → `dr.mean(..)`
  * `dr.is_diff_array_v` → `dr.is_diff_v`
  * `dr.is_jit_array_v` → `dr.is_jit_v`


In Python Dr.Jit constants switch from CamelCase to snake_case:
  * `ek.Pi` → `dr.pi`
  * `ek.TwoPi` → `dr.two_pi`
  * …


To learn more about Dr.Jit and its dissimilarities with Enoki, see the [Dr.Jit documentation](https://readthedocs.org/projects/drjit/badge/?version=latest).
## Differences to Mitsuba 0.6[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/differences.html#differences-to-mitsuba-0-6 "Link to this heading")
Mitsuba 3 strives to retain scene compatibility with its predecessor Mitsuba 0.6. However, in most other respects, it is a completely new system following a different set of goals.
### Missing features[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/differences.html#missing-features "Link to this heading")
A number of Mitsuba 0.6 features are missing in Mitsuba 3. We plan to port some of these features in the future and discard others. The following list provides an overview of the main missing features:
  * **Shapes** : the basic shapes (PLY/OBJ/Serialized triangle meshes, rectangles, spheres, cylinders) are all supported. However, instancing and assemblies of hair fibers are still missing.
  * **Integrators** : Only surface and volumetric path tracers are provided. The following are missing:
    * **Bidirectional path tracing**
    * **Progressive photon mapping**
    * **Path space Metropolis light transport / Manifold exploration / Energy redistribution path tracing:**
Path-space MCMC techniques were an incredibly complex component of the previous generation of Mitsuba. We do not currently plan to port these and recommend that you stick with Mitsuba 0.6 if your application depends on them.
  * **Animation** : specification and handling of temporally varying transformations, e.g., for motion blur is not yet implemented.
  * **Diffusion-based subsurface scattering** : currently missing, status undecided.


### Scene format[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/differences.html#scene-format "Link to this heading")
Mitsuba 3’s XML scene format is almost identical to that of Mitsuba 0.6. Most plugins have the same name and same parameters, and we made sure that parameters behave in the same way as in Mitsuba 0.6.
One significant change is that Mitsuba 3 uses “`underscore_case`”-style instead of “`camelCase`”-style capitalization. This is part of a concerted change to the entire renderer that also touches all C++ and Python interfaces (see the developer guide for reasons behind this). To given an example: a perspective camera definition, which might have looked like the following in Mitsuba 0.6
```
<sensortype="perspective">
<stringname="fovAxis"value="smaller"/>
<floatname="nearClip"value="10"/>
<floatname="farClip"value="2800"/>
<floatname="focusDistance"value="1000"/>
<transformname="toWorld">
<translatex="0"y="0"z="-100"/>
</transform>
</sensor>

```
Copy to clipboard
now reads
```
<sensortype="perspective">
<stringname="fov_axis"value="smaller"/>
<floatname="near_clip"value="10"/>
<floatname="far_clip"value="2800"/>
<floatname="focus_distance"value="1000"/>
<transformname="to_world">
<translatevalue="0, 0, -100"/>
</transform>
</sensor>

```
Copy to clipboard
The above snippet also shows an unrelated change: the preferred syntax for specifying positions, translations, etc., was shortened:
```
<!-- old notation -->
<pointname="position"x="0"y="0"z="-100"/>

<!-- new notation -->
<pointname="position"value="0, 0, -100"/>

```
Copy to clipboard
All of these changes can be automated, and Mitsuba performs them internally when it detects a scene with a version number lower than 2.0.0. Invoke the `mitsuba` binary with the `-u` parameter if you would like it to write the updated scene description back to disk.
* * *
