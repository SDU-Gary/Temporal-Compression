---
url: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_volumes.html
crawled_at: 2025-11-13T17:11:17.035929
title: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_volumes.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_volumes.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_volumes.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Volumes[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_volumes.html#volumes "Link to this heading")
This section covers the different types of volume data sources included with Mitsuba. These plug-ins are intended to be used together with the medium plugins and provide three-dimensional spatially varying density and albedo fields.
## Grid-based volume data source (gridvolume)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_volumes.html#grid-based-volume-data-source-gridvolume "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
filename | string | Filename of the volume to be loaded |   
grid | VolumeGrid object | When creating a grid volume at runtime, e.g. from Python or C++, an existing `VolumeGrid` instance can be passed directly rather than loading it from the filesystem with filename. |   
use_grid_bbox | boolean | When set to `true`, the bounding box information contained in the `VolumeGrid` object (or the file it was loaded from) will be used. By default, it is assumed that the grid is defined in the unit cube spanning (0, 0, 0) x (1, 1, 1). Any evaluation of the volume outside of the bounding box is handled by the `wrap_mode`. (Default: false) |   
data | tensor | Tensor array containing the grid data. This parameter can only be specified when building this plugin at runtime from Python or C++ and cannot be specified in the XML scene description. The raw parameter must also be set to true when using a tensor. | P, ∂  
filter_type | string |  Specifies how voxel values are interpolated. The following options are currently available:
  * `trilinear` (default): perform trilinear interpolation.
  * `nearest`: disable interpolation. In this mode, the plugin performs nearest neighbor lookups of volume values.

|   
wrap_mode | string |  Controls the behavior of volume evaluations that fall outside of the [0,1] range. The following options are currently available:
  * `clamp` (default): clamp coordinates to the edge of the volume.
  * `repeat`: tile the volume infinitely.
  * `mirror`: mirror the volume along its boundaries.

|   
raw | boolean | Should the transformation to the stored color data (e.g. sRGB to linear, spectral upsampling) be disabled? You will want to enable this when working with non-color, 3-channel volume data. Currently, no plugin needs this option to be set to true (Default: false) |   
to_world | transform | Specifies an optional 4x4 transformation matrix that will be applied to volume coordinates. |   
accel | boolean | Hardware acceleration features can be used in CUDA mode. These features can cause small differences as hardware interpolation methods typically have a loss of precision (not exactly 32-bit arithmetic). (Default: true) |   
This class implements access to volume data stored on a 3D grid using a simple binary exchange format (compatible with Mitsuba 0.6). When appropriate, spectral upsampling is applied at loading time to convert RGB values to spectra that can be used in the renderer. We provide a small [helper utility](https://github.com/mitsuba-renderer/mitsuba2-vdb-converter) to convert OpenVDB files to this format. The format uses a little endian encoding and is specified as follows:
Volume file format[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_volumes.html#id1 "Link to this table") Position | Content  
---|---  
Bytes 1-3 | ASCII Bytes ’V’, ’O’, and ’L’  
Byte 4 | File format version number (currently 3)  
Bytes 5-8 | Encoding identified (32-bit integer). Currently, only a value of 1 is supported (float32-based representation)  
Bytes 9-12 | Number of cells along the X axis (32 bit integer)  
Bytes 13-16 | Number of cells along the Y axis (32 bit integer)  
Bytes 17-20 | Number of cells along the Z axis (32 bit integer)  
Bytes 21-24 | Number of channels (32 bit integer, supported values: 1, 3 or 6)  
Bytes 25-48 | Axis-aligned bounding box of the data stored in single precision (order: xmin, ymin, zmin, xmax, ymax, zmax)  
Bytes 49-* | Binary data of the volume stored in the specified encoding. The data are ordered so that the following C-style indexing operation makes sense after the file has been loaded into memory: `data[((zpos*yres + ypos)*xres + xpos)*channels + chan]` where (xpos, ypos, zpos, chan) denotes the lookup location.  
XMLPython
```
<mediumtype="heterogeneous">
<volumetype="grid"name="albedo">
<stringname="filename"value="my_volume.vol"/>
</volume>
</medium>

```
Copy to clipboard
```
'type': 'heterogeneous',
'albedo': {
    'type': 'grid',
    'filename': 'my_volume.vol'
}

```
Copy to clipboard
## Constant-valued volume data source (constvolume)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_volumes.html#constant-valued-volume-data-source-constvolume "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
value | float or spectrum | Specifies the value of the constant volume. | P, ∂  
This plugin provides a volume data source that is constant throughout its domain. Depending on how it is used, its value can either be a scalar or a color spectrum.
XMLPython
```
<mediumtype="heterogeneous">
<volumetype="constvolume"name="albedo">
<rgbname="value"value="0.99, 0.8, 0.8"/>
</volume>

<!-- shorthand: this will create a 'constvolume' internally -->
<rgbname="albedo"value="0.99, 0.99, 0.99"/>
</medium>

```
Copy to clipboard
```
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

```
Copy to clipboard
* * *
  *[P]: This parameter will be exposed as a scene parameter
  *[∂]: This parameter is differentiable
