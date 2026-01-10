---
url: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html
crawled_at: 2025-11-13T17:12:34.572037
title: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Textures[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html#textures "Link to this heading")
The following section describes the available texture data sources. In Mitsuba 3, textures are objects that can be attached to certain surface scattering model parameters to introduce spatial variation. In the documentation, these are listed as supporting the texture type. See the last sections about [BSDFs](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#sec-bsdfs) for many examples.
Textures take an (optional) `<transform>` called to_uv which can be used to translate, scale, or rotate the lookup into the texture accordingly.
An example in XML looks as follows:
XMLPython
```
<sceneversion="3.0.0">
<!-- Create a BSDF that supports textured parameters -->
<bsdftype=".. BSDF type .."id="my_textured_material">
<texturetype=".. texture type .."name=".. parameter name ..">
<!-- .. Texture parameters go here .. -->

<transformname="to_uv">
<!-- Scale texture by factor of 2 -->
<scalex="2"y="2"/>
<!-- Offset texture by [0.5, 1.0] -->
<translatex="0.5"y="1.0"/>
</transform>
</texture>

<!-- .. Non-spatially varying BSDF parameters ..-->
</bsdf>
</scene>

```
Copy to clipboard
```
'type': 'scene',

# .. scene contents ..

# Create a BSDF that supports textured parameters
'my_textured_material': {
    'type': '<bsdf_type>',
    '<parameter_name>' : {
        'type': '<texture_type>',
        # .. texture parameters ..
        'to_uv': mi.scalar_rgb.ScalarTransform4f.scale([2, 2, 0]).translate([0.5, 1.0, 0]) # Third dimension is ignored
    }

    # .. non-spatially varying BSDF parameters ..
}

```
Copy to clipboard
Similar to BSDFs, named textures can alternatively be defined at the top level of the scene and later referenced. This is particularly useful if the same texture would be loaded many times otherwise.
XMLPython
```
<sceneversion="3.0.0">
<!-- Create a named texture at the top level -->
<texturetype=".. texture type .."id="my_named_texture">
<!-- .. Texture parameters go here .. -->
</texture>

<!-- Create a BSDF that supports textured parameters -->
<bsdftype=".. BSDF type ..">
<!-- Example of referencing a named texture -->
<refid="my_named_texture"name=".. parameter name .."/>

<!-- .. Non-spatially varying BSDF parameters ..-->
</bsdf>
</scene>

```
Copy to clipboard
```
'type': 'scene',

# .. scene contents ..

'texture_id': {
    'type': '<texture_type>',
    # .. texture parameters ..
},

# Create a BSDF that supports textured parameters
'my_textured_material': {
    'type': '<bsdf_type>',
    '<parameter_name>' : {
        'type' : 'ref',
        'id' : 'texture_id'
    }

    # .. non-spatially varying BSDF parameters ..
}

```
Copy to clipboard
## Bitmap texture (bitmap)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html#bitmap-texture-bitmap "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
filename | string | Filename of the bitmap to be loaded |   
bitmap | Bitmap object | When creating a Bitmap texture at runtime, e.g. from Python or C++, an existing Bitmap image instance can be passed directly rather than loading it from the filesystem with filename. |   
data | tensor | Tensor array containing the texture data. Similarly to the bitmap parameter, this field can only be used at runtime. The raw parameter must also be set to true. | P, ∂  
filter_type | string |  Specifies how pixel values are interpolated and filtered when queried over larger UV regions. The following options are currently available:
  * `bilinear` (default): perform bilinear interpolation, but no filtering.
  * `nearest`: disable filtering and interpolation. In this mode, the plugin performs nearest neighbor lookups of texture values.

|   
wrap_mode | string |  Controls the behavior of texture evaluations that fall outside of the [0,1] range. The following options are currently available:
  * `repeat` (default): tile the texture infinitely.
  * `mirror`: mirror the texture along its boundaries.
  * `clamp`: clamp coordinates to the edge of the texture.

|   
format | string |  Specifies the underlying texture storage format. The following options are currently available:
  * 

`auto` (default): If loading a texture from a bitmap, use half
     precision for bitmap data with 16 or lower bit depth, otherwise use the native floating point representation of the Mitsuba variant. For variants using a spectral color representation this option is the same as `variant`.
  * 

`variant`: Use the corresponding native floating point representation
     of the Mitsuba variant
  * `fp16`: Forcibly store the texture in half precision

|   
raw | boolean | Should the transformation to the stored color data (e.g. sRGB to linear, spectral upsampling) be disabled? You will want to enable this when working with bitmaps storing normal maps that use a linear encoding. (Default: false) |   
to_uv | transform | Specifies an optional 3x3 transformation matrix that will be applied to UV values. A 4x4 matrix can also be provided, in which case the extra row and column are ignored. | P  
accel | boolean | Hardware acceleration features can be used in CUDA mode. These features can cause small differences as hardware interpolation methods typically have a loss of precision (not exactly 32-bit arithmetic). (Default: true) |   
This plugin provides a bitmap texture that performs interpolated lookups given a JPEG, PNG, OpenEXR, RGBE, TGA, or BMP input file.
When loading the plugin, the data is first converted into a usable color representation for the renderer:
  * In rgb modes, sRGB textures are converted into a linear color space.
  * In spectral modes, sRGB textures are _spectrally upsampled_ to plausible smooth spectra [[JH19](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id9 "Wenzel Jakob and Johannes Hanika. A low-dimensional function space for efficient spectral upsampling. Computer Graphics Forum \(Proceedings of Eurographics\), March 2019. URL: https://rgl.epfl.ch/publications/Jakob2019Spectral.")] and stored an intermediate representation that enables efficient queries at render time.
  * In monochrome modes, sRGB textures are converted to grayscale.


These conversions can alternatively be disabled with the raw flag, e.g. when textured data is already in linear space or does not represent colors at all.
XMLPython
```
<texturetype="bitmap">
<stringname="filename"value="texture.png"/>
<stringname="wrap_mode"value="mirror"/>
</texture>

```
Copy to clipboard
```
'type': 'bitmap',
'filename': 'texture.png',
'wrap_mode': 'mirror'

```
Copy to clipboard
## Checkerboard texture (checkerboard)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html#checkerboard-texture-checkerboard "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
color0, color1 | spectrum or texture | Color values for the two differently-colored patches (Default: 0.4 and 0.2) | P, ∂  
to_uv | transform | Specifies an optional 3x3 UV transformation matrix. A 4x4 matrix can also be provided. In that case, the last row and columns will be ignored. (Default: none) | P  
This plugin provides a simple procedural checkerboard texture with customizable colors.
[![../../_images/texture_checkerboard.jpg](https://mitsuba.readthedocs.io/en/stable/_images/texture_checkerboard.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/texture_checkerboard.jpg)
Checkerboard applied to the material test object as well as the ground plane.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html#id2 "Link to this image")
XMLPython
```
<texturetype="checkerboard">
<rgbname="color0"value="0.1, 0.1, 0.1"/>
<rgbname="color1"value="0.5, 0.5, 0.5"/>
</texture>

```
Copy to clipboard
```
'type': 'checkerboard',
'color0': [0.1, 0.1, 0.1],
'color1': [0.5, 0.5, 0.5]

```
Copy to clipboard
## Mesh attribute texture (mesh_attribute)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html#mesh-attribute-texture-mesh-attribute "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
name | string | Name of the attribute to evaluate. It should always start with `"vertex_"` or `"face_"`. |   
scale | float | Scaling factor applied to the interpolated attribute value during evaluation. (Default: 1.0) | P  
This plugin provides a simple mechanism to expose Mesh attributes (e.g. vertex color) as a texture.
[![../../_images/texture_mesh_attribute_vertex.jpg](https://mitsuba.readthedocs.io/en/stable/_images/texture_mesh_attribute_vertex.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/texture_mesh_attribute_vertex.jpg)
Bunny with random vertex color (using barycentric interpolation).[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html#id3 "Link to this image")
[![../../_images/texture_mesh_attribute_face.jpg](https://mitsuba.readthedocs.io/en/stable/_images/texture_mesh_attribute_face.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/texture_mesh_attribute_face.jpg)
Bunny with random face color.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html#id4 "Link to this image")
The following XML snippet describes a mesh with diffuse material, whose reflectance is specified using the `vertex_color` attribute of that mesh:
XMLPython
```
<shapetype="ply">
<stringname="filename"value="my_mesh_with_vertex_color_attr.ply"/>

<bsdftype="diffuse">
<texturetype="mesh_attribute"name="reflectance">
<stringname="name"value="vertex_color"/>
</texture>
</bsdf>
</shape>

```
Copy to clipboard
```
'type': 'ply',
'filename': 'my_mesh_with_vertex_color_attr.ply',
'bsdf': {
    'type': 'diffuse',
    'reflectance': {
        'type': 'mesh_attribute',
        'name': 'vertex_color'
    }
}

```
Copy to clipboard
Note
For spectral variants of the renderer (e.g. `scalar_spectral`), when a mesh attribute name contains the string `"color"`, the tri-stimulus RGB values will be converted to `rgb2spec` model coefficients automatically.
## Volumetric texture (volume)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html#volumetric-texture-volume "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
volume | float, spectrum or volume | Volumetric texture (Default: 0.75). | P, ∂  
This plugin allows using a 3D texture (i.e. a [volume](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_volumes.html#sec-volume) plugin) to texture a 2D surface. This is intended to be used to texture surfaces without a meaningful UV parameterization (e.g., an implicit surface) or to apply procedural 3D textures. At a given point on a surface, the texture value will be determined by looking up the corresponding value in the referenced `volume`. This is done in world space and potentially requires using the volume’s `to_world` transformation to align the volume with the object using the texture.
XMLPython
```
<texturetype="volume">
<volumename="volume"type="gridvolume">
<stringname="filename"value="my_volume.vol"/>
</volume>
</texture>

```
Copy to clipboard
```
'type': 'volume',
'volume': {
    'type': 'gridvolume',
    'filename': 'my_volume.vol'
}

```
Copy to clipboard
* * *
  *[P]: This parameter will be exposed as a scene parameter
  *[∂]: This parameter is differentiable
