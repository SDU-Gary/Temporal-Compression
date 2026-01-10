---
url: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_films.html
crawled_at: 2025-11-13T17:15:11.851677
title: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_films.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_films.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_films.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Films[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_films.html#films "Link to this heading")
A film defines how conducted measurements are stored and converted into the final output file that is written to disk at the end of the rendering process.
In the XML scene description language, a normal film configuration might look as follows:
XMLPython
```
<sceneversion="3.0.0">
<!-- .. scene contents -->

<sensortype=".. sensor type ..">
<!-- .. sensor parameters .. -->

<!-- Write to a high dynamic range EXR image -->
<filmtype="hdrfilm">
<!-- Specify the desired resolution (e.g. full HD) -->
<integername="width"value="1920"/>
<integername="height"value="1080"/>

<!-- Use a Gaussian reconstruction filter. -->
<rfiltertype="gaussian"/>
</film>
</sensor>
</scene>

```
Copy to clipboard
```
'type': 'scene',

# .. scene contents ..

'sensor_id': {
    'type': '<sensor_type>',

    # Write to a high dynamic range EXR image
    'film_id': {
        'type': 'hdrfilm',
        # Specify the desired resolution (e.g. full HD)
        'width': 1920,
        'height': 1080,
        # Use a Gaussian reconstruction filter
        'filter': { 'type': 'gaussian' }
    }
}

```
Copy to clipboard
The `<film>` plugin should be instantiated inside a `<sensor>` declaration. Note how the output filename is never specified—it is automatically inferred from the scene filename and can be manually overridden by passing the configuration parameter `-o` to the `mitsuba` executable when rendering from the command line.
## High dynamic range film (hdrfilm)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_films.html#high-dynamic-range-film-hdrfilm "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
width, height | integer | Width and height of the camera sensor in pixels. (Default: 768, 576) |   
file_format | string | Denotes the desired output file format. The options are openexr (for ILM’s OpenEXR format), rgbe (for Greg Ward’s RGBE format), or pfm (for the Portable Float Map format). (Default: openexr) |   
pixel_format | string | Specifies the desired pixel format of output images. The options are luminance, luminance_alpha, rgb, rgba, xyz and xyza. (Default: rgb) |   
component_format | string | Specifies the desired floating point component format of output images (when saving to disk). The options are float16, float32, or uint32. (Default: float16) |   
crop_offset_x, crop_offset_y, crop_width, crop_height | integer | These parameters can optionally be provided to select a sub-rectangle of the output. In this case, only the requested regions will be rendered. (Default: Unused) |   
sample_border | boolean | If set to true, regions slightly outside of the film plane will also be sampled. This may improve the image quality at the edges, especially when using very large reconstruction filters. In general, this is not needed though. (Default: false, i.e. disabled) |   
compensate | boolean | If set to true, sample accumulation will be performed using Kahan-style error-compensated accumulation. This can be useful to avoid roundoff error when accumulating very many samples to compute reference solutions using single precision variants of Mitsuba. This feature is currently only supported in JIT variants and can make sample accumulation quite a bit more expensive. (Default: false, i.e. disabled) |   
(Nested plugin) | rfilter | Reconstruction filter that should be used by the film. (Default: gaussian, a windowed Gaussian filter) |   
State parameters |  |  |   
size | `Vector2u` | Width and height of the camera sensor in pixels | P  
crop_size | `Vector2u` | Size of the sub-rectangle of the output in pixels | P  
crop_offset | `Point2u` | Offset of the sub-rectangle of the output in pixels | P  
This is the default film plugin that is used when none is explicitly specified. It stores the captured image as a high dynamic range OpenEXR file and tries to preserve the rendering as much as possible by not performing any kind of post processing, such as gamma correction—the output file will record linear radiance values.
When writing OpenEXR files, the film will either produce a luminance, luminance/alpha, RGB(A), or XYZ(A) tristimulus bitmap having a float16, float32, or uint32-based internal representation based on the chosen parameters. The default configuration is RGB with a float16 component format, which is appropriate for most purposes.
For OpenEXR files, Mitsuba 3 also supports fully general multi-channel output; refer to the [aov](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#integrator-aov) or [stokes](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#integrator-stokes) plugins for details on how this works.
The plugin can also write RLE-compressed files in the Radiance RGBE format pioneered by Greg Ward (set file_format=rgbe), as well as the Portable Float Map format (set file_format=pfm). In the former case, the component_format and pixel_format parameters are ignored, and the output is float8-compressed RGB data. PFM output is restricted to float32-valued images using the rgb or luminance pixel formats. Due to the superior accuracy and adoption of OpenEXR, the use of these two alternative formats is discouraged however.
When RGB(A) output is selected, the measured spectral power distributions are converted to linear RGB based on the CIE 1931 XYZ color matching curves and the ITU-R Rec. BT.709-3 primaries with a D65 white point.
The following XML snippet describes a film that writes a full-HD RGBA OpenEXR file:
XMLPython
```
<filmtype="hdrfilm">
<stringname="pixel_format"value="rgba"/>
<integername="width"value="1920"/>
<integername="height"value="1080"/>
</film>

```
Copy to clipboard
```
'type': 'hdrfilm',
'pixel_format': 'rgba',
'width': 1920,
'height': 1080

```
Copy to clipboard
## Spectral film (specfilm)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_films.html#spectral-film-specfilm "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
width, height | integer | Width and height of the camera sensor in pixels. (Default: 768, 576) |   
component_format | string | Specifies the desired floating point component format of output images. The options are float16, float32, or uint32. (Default: float16) |   
crop_offset_x, crop_offset_y, crop_width, crop_height | integer | These parameters can optionally be provided to select a sub-rectangle of the output. In this case, only the requested regions will be rendered. (Default: Unused) |   
sample_border | boolean | If set to true, regions slightly outside of the film plane will also be sampled. This may improve the image quality at the edges, especially when using very large reconstruction filters. In general, this is not needed though. (Default: false, i.e. disabled) |   
compensate | boolean | If set to true, sample accumulation will be performed using Kahan-style error-compensated accumulation. This can be useful to avoid roundoff error when accumulating very many samples to compute reference solutions using single precision variants of Mitsuba. This feature is currently only supported in JIT variants and can make sample accumulation quite a bit more expensive. (Default: false, i.e. disabled) |   
(Nested plugin) | rfilter | Reconstruction filter that should be used by the film. (Default: gaussian, a windowed Gaussian filter) |   
State parameters |  |  |   
(Nested plugins) | spectrum | One or several Sensor Response Functions (SRF) used to compute different spectral bands | P  
size | `Vector2u` | Width and height of the camera sensor in pixels | P  
crop_size | `Vector2u` | Size of the sub-rectangle of the output in pixels | P  
crop_offset | `Point2u` | Offset of the sub-rectangle of the output in pixels | P  
This plugin stores one or several spectral bands as a multichannel spectral image in a high dynamic range OpenEXR file and tries to preserve the rendering as much as possible by not performing any kind of post-processing, such as gamma correction—the output file will record linear radiance values.
Given one or several spectral sensor response functions (SRFs), the film will store in each channel the captured radiance weighted by one of the SRFs (which do not have to be limited to the range of the visible spectrum). The name of the channels in the final image appears given their alphabetical order (not the location in the definition).
To reduce noise, this plugin implements two strategies: first, it creates a combined continuous distribution with all the different SRFs using inverse transform sampling. Then it distributes samples across all the spectral ranges of wavelengths covered by the SRFs. These strategies greatly reduce the spectral noise that would appear if each channel were calculated independently.
[![../../_images/cbox_complete.png](https://mitsuba.readthedocs.io/en/stable/_images/cbox_complete.png) ](https://mitsuba.readthedocs.io/en/stable/_images/cbox_complete.png)
`RGB` spectral rendering[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_films.html#id1 "Link to this image")
[![../../_images/band1_red.png](https://mitsuba.readthedocs.io/en/stable/_images/band1_red.png) ](https://mitsuba.readthedocs.io/en/stable/_images/band1_red.png)
`CH-1`: band1_red[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_films.html#id2 "Link to this image")
[![../../_images/band2_green.png](https://mitsuba.readthedocs.io/en/stable/_images/band2_green.png) ](https://mitsuba.readthedocs.io/en/stable/_images/band2_green.png)
`CH-2`: band2_green[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_films.html#id3 "Link to this image")
[![../../_images/band3_blue.png](https://mitsuba.readthedocs.io/en/stable/_images/band3_blue.png) ](https://mitsuba.readthedocs.io/en/stable/_images/band3_blue.png)
`CH-3`: band3_blue[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_films.html#id4 "Link to this image")
The following snippet describes a film that writes the previously shown 3-channel OpenEXR file with each channel containing the sensor response to each defined spectral band (it is possible to load a spectrum from a file, see the [Spectrum definition](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#color-spectra) section). Notice that in this example, each band contains the spectral sensitivity of one of the `rgb` channels.
XMLPython
```
<filmtype="specfilm">
<integername="width"value="1920"/>
<integername="height"value="1080"/>
<spectrumname="band1_red"filename="data_red.spd"/>
<spectrumname="band2_green"filename="data_green.spd"/>
<spectrumname="band3_blue"filename="data_blue.spd"/>
</film>

```
Copy to clipboard
```
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

```
Copy to clipboard
[**Simplify infrastructure** with MongoDB Atlas, the leading developer data platform](https://server.ethicalads.io/proxy/click/9504/019a7c7f-25d6-7110-a928-4a1b78392ccd/)
[Ads by EthicalAds](https://www.ethicalads.io/advertisers/topics/data-science/?ref=ea-text)
* * *
  *[P]: This parameter will be exposed as a scene parameter
