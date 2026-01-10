---
url: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html
crawled_at: 2025-11-13T17:09:55.478968
title: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Spectra[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectra "Link to this heading")
This section describes the plugins behind spectral reflectance or emission used in Mitsuba 3. On an implementation level, these behave very similarly to the [texture plugins](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html#sec-textures) described earlier (but lacking their spatially varying property) and can thus be used similarly as either BSDF or emitter parameters:
XMLPython
```
<sceneversion="3.0.0">
<bsdftype=".. BSDF type ..">
<!-- Explicitly add a uniform spectrum plugin -->
<spectrumtype=".. spectrum type .."name=".. parameter name ..">
<!-- Spectrum parameters go here -->
</spectrum>
</bsdf>
</scene>

```
Copy to clipboard
```
'type': 'scene',
'bsdf_id': {
    'type': '<bsdf_type>',

    '<parameter name>': {
        'type': '<spectrum type>',
        # .. spectrum parameters ..
    }
}

```
Copy to clipboard
In practice, it is however discouraged to instantiate plugins in this explicit way and the XML scene description parser directly parses a number of common (shorter) `<spectrum>` and `<rgb>` tags. See the corresponding section about the [scene file format](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#sec-file-format) for details.
The following two tables summarize which underlying plugins get instantiated in each case, accounting for differences between reflectance and emission properties and all different color modes. Each plugin is briefly summarized below.
XML description | Monochrome mode | RGB mode | Spectral mode  
---|---|---|---  
`<spectrum name=".." value="0.5"/>` | [uniform](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-uniform) | [uniform](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-uniform) | [uniform](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-uniform)  
`<spectrum name=".." value="400:0.1, 700:0.2"/>` | [uniform](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-uniform) | [srgb](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-srgb) | [regular](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-regular)/[irregular](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-irregular)  
`<spectrum name=".." filename=".."/>` | [uniform](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-uniform) | [srgb](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-srgb) | [regular](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-regular)/[irregular](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-irregular)  
`<rgb name=".." value="0.5, 0.2, 0.5"/>` | [srgb](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-srgb) | [srgb](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-srgb) | [srgb](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-srgb)  
Spectra used for reflectance (within BSDFs)
XML description | Monochrome mode | RGB mode | Spectral mode  
---|---|---|---  
`<spectrum name=".." value="0.5"/>` | [uniform](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-uniform) | [srgb](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-srgb) | [uniform](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-uniform)  
`<spectrum name=".." value="400:0.1, 700:0.2"/>` | [uniform](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-uniform) | [srgb](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-srgb) | [regular](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-regular)/[irregular](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-irregular)  
`<spectrum name=".." filename=".."/>` | [uniform](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-uniform) | [srgb](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-srgb) | [regular](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-regular)/[irregular](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-irregular)  
`<rgb name=".." value="0.5, 0.2, 0.5"/>` | [d65](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-d65) | [srgb](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-srgb) | [d65](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-d65)  
Spectra used for emission (within emitters)
A uniform spectrum does not produce a uniform RGB response in sRGB (which has a D65 white point). Hence giving `<spectrum name=".." value="1.0"/>` as the radiance value of an emitter will result in a purple-ish color. On the other hand, using such spectrum for a BSDF reflectance value will result in an object appearing white. Both RGB and spectral modes of Mitsuba 3 will exhibit this behavior consistently. The figure below illustrates this for combinations of inputs for the emitter radiance (here using a [constant](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#emitter-constant) emitter) and the BSDF reflectance (here using a [diffuse](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-diffuse) BSDF).
[![../../_images/spectrum_rgb_table.png](https://mitsuba.readthedocs.io/en/stable/_images/spectrum_rgb_table.png) ](https://mitsuba.readthedocs.io/en/stable/_images/spectrum_rgb_table.png)
Warning
While it is possible to define unbounded RGB properties (such as the `eta` value for a [conductor BSDF](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-conductor)) using `<rgb name=".." value=".."/>` tag, it is highly recommended to directly define a spectrum curve (or use a material from `conductor-ior-list`) as the spectral uplifting algorithm implemented in Mitsuba won’t be able to guarantee that the produced spectrum will behave consistently in both RGB and spectral modes.
## Uniform spectrum (uniform)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#uniform-spectrum-uniform "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
wavelength_min | float | Lower bound of the wavelength sampling range in nanometers. Default: 360 nm |   
wavelength_max | float | Upper bound of the wavelength sampling range in nanometers. Default: 830 nm |   
value | float | Value of the spectral function across the specified spectral range. | P, ∂  
This spectrum returns a constant reflectance or emission value over the spectral dimension. It implements a uniform sampling method on a finite spectral range controlled by the `wavelength_min` and `wavelength_max` parameters.
XMLPython
```
<spectrumtype="uniform">
<floatname="value"value="0.1"/>
</spectrum>

```
Copy to clipboard
```
'type': 'uniform',
'value': 0.1

```
Copy to clipboard
## Regular spectrum (regular)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#regular-spectrum-regular "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
wavelength_min | float | Minimum wavelength of the spectral range in nanometers. |   
wavelength_max | float | Maximum wavelength of the spectral range in nanometers. |   
values | string | Values of the spectral function at spectral range extremities. | P, ∂  
State parameters |  |  |   
range | string | Spectral emission range. | P, ∂  
This spectrum returns linearly interpolated reflectance or emission values from _regularly_ placed samples.
XMLPython
```
<spectrumtype="regular">
<stringname="range"value="400, 700">
<stringname="values"value="0.1, 0.2">
</spectrum>

```
Copy to clipboard
```
'type': 'regular',
'wavelength_min': 400,
'wavelength_max': 700,
'values': '0.1, 0.2'

```
Copy to clipboard
## Irregular spectrum (irregular)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#irregular-spectrum-irregular "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
wavelengths | string | Wavelength values where the function is defined. | P, ∂  
values | string | Values of the spectral function at the specified wavelengths. | P, ∂  
This spectrum returns linearly interpolated reflectance or emission values from _irregularly_ placed samples.
XMLPython
```
<spectrumtype="irregular">
<stringname="wavelengths"value="400, 700">
<stringname="values"value="0.1, 0.2">
</spectrum>

```
Copy to clipboard
```
'type': 'irregular',
'wavelengths': '400, 700',
'values': '0.1, 0.2'

```
Copy to clipboard
## sRGB spectrum (srgb)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#srgb-spectrum-srgb "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
color | color | The corresponding sRGB color value. |   
State parameters |  |  |   
value | color | Spectral upsampling model coefficients of the srgb color value. | P, ∂  
In spectral render modes, this smooth spectrum is the result of the _spectral upsampling_ process [[JH19](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id9 "Wenzel Jakob and Johannes Hanika. A low-dimensional function space for efficient spectral upsampling. Computer Graphics Forum \(Proceedings of Eurographics\), March 2019. URL: https://rgl.epfl.ch/publications/Jakob2019Spectral.")] used by the system. In RGB render modes, this spectrum represents a constant RGB value. In monochrome modes, this spectrum represents a constant luminance value.
XMLPython
```
<spectrumtype="srgb">
<rgbname="color"value="10, 20, 250"/>
</spectrum>

```
Copy to clipboard
```
'type': 'srgb',
'color': [10, 20, 250]

```
Copy to clipboard
## D65 spectrum (d65)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#d65-spectrum-d65 "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
color | color | The corresponding sRGB color value. |   
scale | float | Optional scaling factor applied to the emitted spectrum. (Default: 1.0) |   
State parameters |  |  |   
(Nested plugin) | texture | Underlying texture/spectra to be multiplied by D65. | P, ∂  
color | color | Spectral upsampling model coefficients of the srgb color value. | P, ∂  
The CIE Standard Illuminant D65 corresponds roughly to the average midday light in Europe, also called a daylight illuminant. It is the default emission spectrum used for light sources in all spectral rendering modes.
The D65 spectrum can be multiplied by a color value specified using the `color` parameters.
Alternatively, it is possible to modulate the D65 illuminant with a spectrally orand spatially varying signal defined by a nested texture plugin. This is used in many emitter plugins when the radiance quantity might be driven by a 2D texture but also needs to be multiplied with the D65 spectrum.
In RGB rendering modes, the D65 illuminant isn’t relevant therefore this plugin expands into another plugin type (e.g. `uniform`, `srgb`, …) as the product isn’t required in this case.
XMLPython
```
<shapetype=".. shape type ..">
<emittertype="area">
<spectrumtype="d65"/>
</emitter>
</shape>

```
Copy to clipboard
```
'type': '.. shape type ..',
'emitter': {
    'type': 'area',
    'radiance': { 'type': 'd65', }
}

```
Copy to clipboard
## Raw constant-valued texture (rawconstant)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#raw-constant-valued-texture-rawconstant "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
value | float or vector | The constant value(s) to be returned. Can be a single float or a 3D vector. | P, ∂  
A constant-valued texture that returns the same value regardless of color mode, UV coordinates or wavelength. No color conversion or range validation takes place. The value can be 1D or 3D. For 1D inputs, the same value is replicated across components when a 3D value is queried.
If color-handling is desired, see the [spectrum-srgb](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#spectrum-srgb) plugin instead.
XMLXMLPython
```
<texturetype="rawconstant">
<floatname="value"value="0.5"/>
</texture>

```
Copy to clipboard
```
<texturetype="rawconstant">
<vectorname="value"value="0.5, 1.0, 0.3"/>
</texture>

```
Copy to clipboard
```
'type': 'rawconstant',
'value': 0.5  # or [0.5, -2.0, 0.3]

```
Copy to clipboard
## Blackbody spectrum (blackbody)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#blackbody-spectrum-blackbody "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
wavelength_min | float | Minimum wavelength of the spectral range in nanometers. (Default: 360nm) |   
wavelength_max | float | Maximum wavelength of the spectral range in nanometers. (Default: 830nm) |   
temperature | float | Black body temperature in Kelvins. | P  
This is a black body radiation spectrum for a specified temperature And therefore takes a single float-valued parameter temperature (in Kelvins).
This is the only spectrum type that needs to be explicitly instantiated in its full XML description:
XMLPython
```
<shapetype=".. shape type ..">
<emittertype="area">
<spectrumtype="blackbody"name="radiance">
<floatname="temperature"value="5000"/>
</spectrum>
</emitter>
</shape>

```
Copy to clipboard
```
'type': '.. shape type ..',
'emitter': {
    'type': 'area',
    'radiance': {
        'type': 'blackbody',
        'temperature': 5000
    }
}

```
Copy to clipboard
This spectrum type only makes sense for specifying emission and is unavailable in non-spectral rendering modes.
Note that attaching a black body spectrum to the intensity property of a emitter introduces physical units into the rendering process of Mitsuba 3, which is ordinarily a unitless system. Specifically, the black body spectrum has units of power (\\(W\\)) per unit area (\\(m^{-2}\\)) per steradian (\\(sr^{-1}\\)) per unit wavelength (\\(nm^{-1}\\)). As a consequence, your scene should be modeled in meters for this plugin to work properly.
[Develop and launch modern apps with MongoDB Atlas, a resilient data platform.](https://server.ethicalads.io/proxy/click/9505/019a7c7a-46fd-7343-8a2b-dc42fe160149/)
[Ads by EthicalAds](https://www.ethicalads.io/advertisers/?ref=ea-text)
* * *
  *[P]: This parameter will be exposed as a scene parameter
  *[∂]: This parameter is differentiable
