---
url: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html
crawled_at: 2025-11-13T17:12:26.134171
title: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Shapes[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#shapes "Link to this heading")
This section presents an overview of the shape plugins that are released along with the renderer.
In Mitsuba 3, shapes define surfaces that mark transitions between different types of materials. For instance, a shape could describe a boundary between air and a solid object, such as a piece of rock. Alternatively, a shape can mark the beginning of a region of space that isn’t solid at all, but rather contains a participating medium, such as smoke or steam. Finally, a shape can be used to create an object that emits light on its own.
Shapes are usually declared along with a surface scattering model named _BSDF_ (see the [respective section](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#sec-bsdfs)). This BSDF characterizes what happens at the surface. In the XML scene description language, this might look like the following:
XMLPython
```
<sceneversion="3.0.0">
<!-- .. scene contents .. -->

<shapetype=".. shape type ..">
<bsdftype=".. BSDF type ..">
</bsdf>

<!-- Alternatively: reference a named BSDF that
            has been declared previously

            <ref id="my_bsdf"/>
        -->
</shape>
</scene>

```
Copy to clipboard
```
'type': 'scene',

# .. scene contents ..

'shape_id': {
    'type': '<shape_type>',
    'bsdf_id': {
        'type': '<bsdf_type>',
        # .. bsdf parameters ..
    }

    # Alternatively, reference a named BSDF that had been declared previously
    # 'bsdf_id' : {
    #     'type' : 'ref',
    #     'id' : 'some_bsdf_id'
    # }
}

```
Copy to clipboard
The following subsections discuss the available shape types in greater detail.
## Wavefront OBJ mesh loader (obj)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#wavefront-obj-mesh-loader-obj "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
filename | string | Filename of the OBJ file that should be loaded |   
face_normals | boolean | When set to true, any existing or computed vertex normals are discarded and _face normals_ will instead be used during rendering. This gives the rendered object a faceted appearance. (Default: false) |   
flip_tex_coords | boolean | Treat the vertical component of the texture as inverted? Most OBJ files use this convention. (Default: true) |   
flip_normals | boolean | Is the mesh inverted, i.e. should the normal vectors be flipped? (Default:false, i.e. the normals point outside) |   
to_world | transform | Specifies an optional linear object-to-world transformation. (Default: none, i.e. object space = world space) |   
State parameters |  |  |   
vertex_count | integer | Total number of vertices | P  
face_count | integer | Total number of faces | P  
faces | uint32[] | Face indices buffer (flatten) | P  
vertex_positions | float[] | Vertex positions buffer (flatten) pre-multiplied by the object-to-world transformation. | P, ∂, D  
vertex_normals | float[] | Vertex normals buffer (flatten) pre-multiplied by the object-to-world transformation. | P, ∂, D  
vertex_texcoords | float[] | Vertex texcoords buffer (flatten) | P, ∂  
(Mesh attribute) | float[] | Mesh attribute buffer (flatten) | P, ∂  
This plugin implements a simple loader for Wavefront OBJ files. It handles meshes containing triangles and quadrilaterals, and it also imports vertex normals and texture coordinates.
Loading an ordinary OBJ file is as simple as writing:
XMLPython
```
<shapetype="obj">
<stringname="filename"value="my_shape.obj"/>
</shape>

```
Copy to clipboard
```
'type': 'obj',
'filename': 'my_shape.obj'

```
Copy to clipboard
Note
Importing geometry via OBJ files should only be used as an absolutely last resort. Due to inherent limitations of this format, the files tend to be unreasonably large, and parsing them requires significant amounts of memory and processing power. What’s worse is that the internally stored data is often truncated, causing a loss of precision. If possible, use the [ply](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#shape-ply) or [serialized](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#shape-serialized) plugins instead.
## PLY (Stanford Triangle Format) mesh loader (ply)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#ply-stanford-triangle-format-mesh-loader-ply "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
filename | string | Filename of the PLY file that should be loaded |   
face_normals | boolean | When set to true, any existing or computed vertex normals are discarded and _face normals_ will instead be used during rendering. This gives the rendered object a faceted appearance. (Default: false) |   
flip_tex_coords | boolean | Treat the vertical component of the texture as inverted? (Default: false) |   
flip_normals | boolean | Is the mesh inverted, i.e. should the normal vectors be flipped? (Default:false, i.e. the normals point outside) |   
State parameters |  |  |   
to_world | transform | Specifies an optional linear object-to-world transformation. (Default: none, i.e. object space = world space) |   
vertex_count | integer | Total number of vertices | P  
face_count | integer | Total number of faces | P  
faces | uint32[] | Face indices buffer (flatten) | P  
vertex_positions | float[] | Vertex positions buffer (flatten) pre-multiplied by the object-to-world transformation. | P, ∂, D  
vertex_normals | float[] | Vertex normals buffer (flatten) pre-multiplied by the object-to-world transformation. | P, ∂, D  
vertex_texcoords | float[] | Vertex texcoords buffer (flatten) | P, ∂  
(Mesh attribute) | float[] | Mesh attribute buffer (flatten) | P, ∂  
[![../../_images/shape_ply_bunny.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_ply_bunny.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_ply_bunny.jpg)
The Stanford bunny loaded with face_normals=false.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id7 "Link to this image")
[![../../_images/shape_ply_bunny_facet.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_ply_bunny_facet.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_ply_bunny_facet.jpg)
The Stanford bunny loaded with face_normals=true. Note the faceted appearance.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id8 "Link to this image")
This plugin implements a fast loader for the Stanford PLY format (both the ASCII and binary format, which is preferred for performance reasons). The current plugin implementation supports triangle meshes with optional UV coordinates, vertex normals and other custom vertex or face attributes.
Consecutive attributes with names sharing a common prefix and using one of the following schemes:
`{prefix}_{x|y|z|w}`, `{prefix}_{r|g|b|a}`, `{prefix}_{0|1|2|3}`, `{prefix}_{1|2|3|4}`
will be group together under a single multidimensional attribute named `{vertex|face}_{prefix}`.
RGB color attributes can also be defined without a prefix, following the naming scheme `{r|g|b|a}` or `{red|green|blue|alpha}`. Those attributes will be group together under a single multidimensional attribute named `{vertex|face}_color`.
XMLPython
```
<shapetype="ply">
<stringname="filename"value="my_shape.ply"/>
<booleanname="flip_normals"value="true"/>
</shape>

```
Copy to clipboard
```
'type': 'ply',
'filename': 'my_shape.ply',
'flip_normals': True

```
Copy to clipboard
Note
Values stored in a RBG color attribute will automatically be converted into spectral model coefficients when using a spectral variant of the renderer.
## Serialized mesh loader (serialized)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#serialized-mesh-loader-serialized "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
filename | string | Filename of the serialized file that should be loaded |   
shape_index | integer | A .serialized file may contain several separate meshes. This parameter specifies which one should be loaded. (Default: 0, i.e. the first one) |   
face_normals | boolean | When set to true, any existing or computed vertex normals are discarded and emph{face normals} will instead be used during rendering. This gives the rendered object a faceted appearance. (Default: false) |   
flip_normals | boolean | Is the mesh inverted, i.e. should the normal vectors be flipped? (Default:false, i.e. the normals point outside) |   
to_world | transform | Specifies an optional linear object-to-world transformation. (Default: none, i.e. object space = world space) |   
State parameters |  |  |   
vertex_count | integer | Total number of vertices | P  
face_count | integer | Total number of faces | P  
faces | uint32[] | Face indices buffer (flatten) | P  
vertex_positions | float[] | Vertex positions buffer (flatten) pre-multiplied by the object-to-world transformation. | P, ∂, D  
vertex_normals | float[] | Vertex normals buffer (flatten) pre-multiplied by the object-to-world transformation. | P, ∂, D  
vertex_texcoords | float[] | Vertex texcoords buffer (flatten) | P, ∂  
(Mesh attribute) | float[] | Mesh attribute buffer (flatten) | P, ∂  
The serialized mesh format represents the most space and time-efficient way of getting geometry information into Mitsuba 3. It stores indexed triangle meshes in a lossless gzip-based encoding that (after decompression) nicely matches up with the internally used data structures. Loading such files is considerably faster than the [ply](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#shape-ply) plugin and orders of magnitude faster than the [obj](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#shape-obj) plugin.
### Format description[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#format-description "Link to this heading")
The serialized file format uses the little endian encoding, hence all fields below should be interpreted accordingly. The contents are structured as follows:
Type | Content  
---|---  
uint16 | File format identifier: `0x041C`  
uint16 | File version identifier. Currently set to `0x0004`  
→ | From this point on, the stream is compressed by the DEFLATE algorithm.  
→ | The used encoding is that of the zlib library.  
uint32 |  An 32-bit integer whose bits can be used to specify the following flags:
  * `0x0001`: The mesh data includes per-vertex normals
  * `0x0002`: The mesh data includes texture coordinates
  * `0x0008`: The mesh data includes vertex colors
  * `0x0010`: Use face normals instead of smoothly interpolated vertex normals. Equivalent to specifying face_normals=true to the plugin.
  * `0x1000`: The subsequent content is represented in single precision
  * `0x2000`: The subsequent content is represented in double precision

  
string | A null-terminated string (utf-8), which denotes the name of the shape.  
uint64 | Number of vertices in the mesh  
uint64 | Number of triangles in the mesh  
array | Array of all vertex positions (X, Y, Z, X, Y, Z, …) specified in binary single or double precision format (as denoted by the flags)  
array | Array of all vertex normal directions (X, Y, Z, X, Y, Z, …) specified in binary single or double precision format. When the mesh has no vertex normals, this field is omitted.  
array | Array of all vertex texture coordinates (U, V, U, V, …) specified in binary single or double precision format. When the mesh has no texture coordinates, this field is omitted.  
array | Array of all vertex colors (R, G, B, R, G, B, …) specified in binary single or double precision format. When the mesh has no vertex colors, this field is omitted.  
array | Indexed triangle data (`[i1, i2, i3]`, `[i1, i2, i3]`, ..) specified in uint32 or in uint64 format (the latter is used when the number of vertices exceeds `0xFFFFFFFF`).  
### Multiple shapes[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#multiple-shapes "Link to this heading")
It is possible to store multiple meshes in a single .serialized file. This is done by simply concatenating their data streams, where every one is structured according to the above description. Hence, after each mesh, the stream briefly reverts back to an uncompressed format, followed by an uncompressed header, and so on. This is necessary for efficient read access to arbitrary sub-meshes.
### End-of-file dictionary[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#end-of-file-dictionary "Link to this heading")
In addition to the previous table, a .serialized file also concludes with a brief summary at the end of the file, which specifies the starting position of each sub-mesh:
Type | Content  
---|---  
uint64 | File offset of the first mesh (in bytes)—this is always zero.  
uint64 | File offset of the second mesh  
⋯ | ⋯  
uint64 | File offset of the last sub-shape  
uint32 | Total number of meshes in the .serialized file  
XMLPython
```
<shapetype="serialized">
<stringname="filename"value="shape.serialized"/>
<bsdftype='diffuse'/>
</shape>

```
Copy to clipboard
```
'type': 'serialized',
'filename': 'shape.serialized',
'material': {
    'type': 'diffuse',
}

```
Copy to clipboard
## Cube (cube)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#cube-cube "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
flip_normals | boolean | Is the cube inverted, i.e. should the normal vectors be flipped? (Default:false, i.e. the normals point outside) |   
to_world | transform | Specifies an optional linear object-to-world transformation. (Default: none (i.e. object space = world space)) |   
State parameters |  |  |   
vertex_count | integer | Total number of vertices | P  
face_count | integer | Total number of faces | P  
faces | uint32[] | Face indices buffer (flatten) | P  
vertex_positions | float[] | Vertex positions buffer (flatten) pre-multiplied by the object-to-world transformation. | P, ∂, D  
vertex_normals | float[] | Vertex normals buffer (flatten) pre-multiplied by the object-to-world transformation. | P, ∂, D  
vertex_texcoords | float[] | Vertex texcoords buffer (flatten) | P, ∂  
(Mesh attribute) | float[] | Mesh attribute buffer (flatten) | P, ∂  
This shape plugin describes a cube intersection primitive, based on the triangle mesh class. By default, it creates a cube between the world-space positions (−1, −1, −1) and (1, 1, 1). However, an arbitrary linear transformation may be specified to translate, rotate, scale or skew it as desired. The parameterization of this shape maps every face onto the rectangle [0,1]2 in uv space.
XMLPython
```
<shapetype="cube">
<transformname="to_world">
<scalex="2"y="10"z="1"/>
</transform>
</shape>

```
Copy to clipboard
```
'type': 'cube',
'to_world': mi.ScalarAffineTransform4f([2, 10, 1])

```
Copy to clipboard
## Sphere (sphere)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#sphere-sphere "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
center | point | Center of the sphere (Default: (0, 0, 0)) |   
radius | float | Radius of the sphere (Default: 1) |   
flip_normals | boolean | Is the sphere inverted, i.e. should the normal vectors be flipped? (Default:false, i.e. the normals point outside) |   
to_world | transform | Specifies an optional linear object-to-world transformation. Note that non-uniform scales and shears are not permitted! (Default: none, i.e. object space = world space) | P, ∂, D  
silhouette_sampling_weight | float | Weight associated with this shape when sampling silhoeuttes in the scene. (Default: 1) | P  
[![../../_images/shape_sphere_basic.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_sphere_basic.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_sphere_basic.jpg)
Basic example[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id9 "Link to this image")
[![../../_images/shape_sphere_parameterization.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_sphere_parameterization.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_sphere_parameterization.jpg)
A textured sphere with the default parameterization[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id10 "Link to this image")
This shape plugin describes a simple sphere intersection primitive. It should always be preferred over sphere approximations modeled using triangles.
A sphere can either be configured using a linear to_world transformation or the center and radius parameters (or both). The two declarations below are equivalent.
XMLPython
```
<shapetype="sphere">
<transformname="to_world">
<translatex="1"y="0"z="0"/>
<scalevalue="2"/>
</transform>
<bsdftype="diffuse"/>
</shape>

<shapetype="sphere">
<pointname="center"x="1"y="0"z="0"/>
<floatname="radius"value="2"/>
<bsdftype="diffuse"/>
</shape>

```
Copy to clipboard
```
'sphere_1': {
    'type': 'sphere',
    'to_world': mi.ScalarAffineTransform4f().scale([2, 2, 2]).translate([1, 0, 0]),
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

```
Copy to clipboard
When a [sphere](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#shape-sphere) shape is turned into an [area](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#emitter-area) light source, Mitsuba 3 switches to an efficient [sampling strategy](https://www.akalin.com/sampling-visible-sphere) by Fred Akalin that has particularly low variance. This makes it a good default choice for lighting new scenes.
[![../../_images/shape_sphere_light_mesh.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_sphere_light_mesh.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_sphere_light_mesh.jpg)
Spherical area light modeled using triangles[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id11 "Link to this image")
[![../../_images/shape_sphere_light_analytic.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_sphere_light_analytic.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_sphere_light_analytic.jpg)
Spherical area light modeled using the [sphere](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#shape-sphere) plugin[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id12 "Link to this image")
## Rectangle (rectangle)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#rectangle-rectangle "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
flip_normals | boolean | Is the rectangle inverted, i.e. should the normal vectors be flipped? (Default: false) |   
to_world | transform | Specifies a linear object-to-world transformation. (Default: none (i.e. object space = world space)) | P, ∂, D  
silhouette_sampling_weight | float | Weight associated with this shape when sampling silhoeuttes in the scene. (Default: 1) | P  
[![../../_images/shape_rectangle.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_rectangle.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_rectangle.jpg)
Basic example[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id13 "Link to this image")
[![../../_images/shape_rectangle_parameterization.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_rectangle_parameterization.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_rectangle_parameterization.jpg)
A textured rectangle with the default parameterization[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id14 "Link to this image")
This shape plugin describes a simple rectangular shape primitive. It is mainly provided as a convenience for those cases when creating and loading an external mesh with two triangles is simply too tedious, e.g. when an area light source or a simple ground plane are needed. By default, the rectangle covers the XY-range [−1,1]×[−1,1] and has a surface normal that points into the positive Z-direction. To change the rectangle scale, rotation, or translation, use the to_world parameter.
The following XML snippet showcases a simple example of a textured rectangle:
XMLPython
```
<shapetype="rectangle">
<bsdftype="diffuse">
<texturename="reflectance"type="checkerboard">
<transformname="to_uv">
<scalex="5"y="5"/>
</transform>
</texture>
</bsdf>
</shape>

```
Copy to clipboard
```
'type': 'rectangle',
'material': {
    'type': 'diffuse',
    'reflectance': {
        'type': 'checkerboard',
        'to_uv': mi.ScalarAffineTransform4f().scale([5, 5, 1])
    }
}

```
Copy to clipboard
## Disk (disk)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#disk-disk "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
flip_normals | boolean | Is the disk inverted, i.e. should the normal vectors be flipped? (Default: false) |   
to_world | transform | Specifies a linear object-to-world transformation. Note that non-uniform scales are not permitted! (Default: none, i.e. object space = world space) | P, ∂, D  
silhouette_sampling_weight | float | Weight associated with this shape when sampling silhoeuttes in the scene. (Default: 1) | P  
[![../../_images/shape_disk.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_disk.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_disk.jpg)
Basic example[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id15 "Link to this image")
[![../../_images/shape_disk_parameterization.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_disk_parameterization.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_disk_parameterization.jpg)
A textured disk with the default parameterization[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id16 "Link to this image")
This shape plugin describes a simple disk intersection primitive. It is usually preferable over discrete approximations made from triangles.
By default, the disk has unit radius and is located at the origin. Its surface normal points into the positive Z-direction. To change the disk scale, rotation, or translation, use the to_world parameter.
The following XML snippet instantiates an example of a textured disk shape:
XMLPython
```
<shapetype="disk">
<bsdftype="diffuse">
<texturename="reflectance"type="checkerboard">
<transformname="to_uv">
<scalex="2"y="10"/>
</transform>
</texture>
</bsdf>
</shape>

```
Copy to clipboard
```
'type': 'disk',
'material': {
    'type': 'diffuse',
    'reflectance': {
        'type': 'checkerboard',
        'to_uv': mi.ScalarAffineTransform4f([2, 10, 0])
    }
}

```
Copy to clipboard
## Cylinder (cylinder)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#cylinder-cylinder "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
p0 | point | Object-space starting point of the cylinder’s centerline. (Default: (0, 0, 0)) |   
p1 | point | Object-space endpoint of the cylinder’s centerline (Default: (0, 0, 1)) |   
radius | float | Radius of the cylinder in object-space units (Default: 1) |   
flip_normals | boolean | Is the cylinder inverted, i.e. should the normal vectors be flipped? (Default: false, i.e. the normals point outside) |   
to_world | transform | Specifies an optional linear object-to-world transformation. Note that non-uniform scales are not permitted! (Default: none, i.e. object space = world space) | P, ∂, D  
silhouette_sampling_weight | float | Weight associated with this shape when sampling silhoeuttes in the scene. (Default: 1) | P  
[![../../_images/shape_cylinder_onesided.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_cylinder_onesided.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_cylinder_onesided.jpg)
Cylinder with the default one-sided shading[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id17 "Link to this image")
[![../../_images/shape_cylinder_twosided.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_cylinder_twosided.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_cylinder_twosided.jpg)
Cylinder with two-sided shading[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id18 "Link to this image")
This shape plugin describes a simple cylinder intersection primitive. It should always be preferred over approximations modeled using triangles. Note that the cylinder does not have endcaps – also, its normals point outward, which means that the inside will be treated as fully absorbing by most material models. If this is not desirable, consider using the [twosided](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-twosided) plugin.
A simple example for instantiating a cylinder, whose interior is visible:
XMLPython
```
<shapetype="cylinder">
<floatname="radius"value="0.3"/>
<bsdftype="twosided">
<bsdftype="diffuse"/>
</bsdf>
</shape>

```
Copy to clipboard
```
'type': 'cylinder',
'radius': 0.3,
'material': {
    'type': 'diffuse'
}

```
Copy to clipboard
## B-spline curve (bsplinecurve)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#b-spline-curve-bsplinecurve "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
filename | string | Filename of the curves to be loaded |   
to_world | transform | Specifies a linear object-to-world transformation. Note that the control points’ raddii are invariant to this transformation! |   
silhouette_sampling_weight | float | Weight associated with this shape when sampling silhoeuttes in the scene. (Default: 1) | P  
State parameters |  |  |   
control_point_count | integer | Total number of control points | P  
segment_indices | uint32[] | Starting indices of a B-Spline segment | P  
control_points | float[] | Flattened control points buffer pre-multiplied by the object-to-world transformation. Each control point in the buffer is structured as follows: position_x, position_y, position_z, radius | P, ∂, D  
[![../../_images/shape_bsplinecurve_basic.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_bsplinecurve_basic.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_bsplinecurve_basic.jpg)
Basic example[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id19 "Link to this image")
[![../../_images/shape_bsplinecurve_parameterization.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_bsplinecurve_parameterization.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_bsplinecurve_parameterization.jpg)
A textured B-spline curve with the default parameterization[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id20 "Link to this image")
This shape plugin describes multiple cubic B-spline curves. They are hollow cylindrical tubes which can have varying radii along their length and are open-ended: they do not have endcaps. They can be made watertight by setting the radii of the extremities to 0. This shape should always be preferred over curve approximations modeled using triangles.
Although it is possible to define multiple curves as multiple separate objects, this plugin was intended to be used as an aggregate of curves. Of course, if the individual curves need different materials or other individual characteristics they need to be defined in separate objects.
The file from which curves are loaded defines a single control point per line using four real numbers. The first three encode the position and the last one is the radius of the control point. At least four control points need to be specified for a single curve. Empty lines between control points are used to indicate the beginning of a new curve. Here is an example of two curves, the first with 4 control points and static radii and the second with 6 control points and increasing radii:
```
-1.00.10.10.5
-0.31.21.00.5
0.30.31.10.5
1.01.41.20.5

-1.05.02.21
-2.34.02.32
3.33.02.23
4.02.02.34
4.01.02.25
4.00.02.36

```
Copy to clipboard
XMLPython
```
<shapetype="bsplinecurve">
<transformname="to_world">
<scalevalue="2"/>
<translatex="1"y="0"z="0"/>
</transform>
<stringname="filename"type="curves.txt"/>
</shape>

```
Copy to clipboard
```
'curves': {
    'type': 'bsplinecurve',
    'to_world': mi.ScalarAffineTransform4f().translate([1, 0, 0]).scale([2, 2, 2]),
    'filename': 'curves.txt'
}

```
Copy to clipboard
## Linear curve (linearcurve)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#linear-curve-linearcurve "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
filename | string | Filename of the curves to be loaded |   
to_world | transform | Specifies a linear object-to-world transformation. Note that the control points’ raddii are invariant to this transformation! |   
State parameters |  |  |   
control_point_count | integer | Total number of control points | P  
segment_indices | uint32[] | Starting indices of a linear segment | P  
control_points | float[] | Flattened control points buffer pre-multiplied by the object-to-world transformation. Each control point in the buffer is structured as follows: position_x, position_y, position_z, radius | P  
[![../../_images/shape_linearcurve_basic.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_linearcurve_basic.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_linearcurve_basic.jpg)
Basic example[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id21 "Link to this image")
[![../../_images/shape_linearcurve_parameterization.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_linearcurve_parameterization.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_linearcurve_parameterization.jpg)
A textured linear curve with the default parameterization[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id22 "Link to this image")
This shape plugin describes multiple linear curves. They are hollow cylindrical tubes which can have varying radii along their length. The linear segments are connected by a smooth spherical joint, and they are also terminated by a spherical endcap. This shape should always be preferred over curve approximations modeled using triangles.
Although it is possible to define multiple curves as multiple separate objects, this plugin was intended to be used as an aggregate of curves. Of course, if the individual curves need different materials or other individual characteristics they need to be defined in separate objects.
The file from which curves are loaded defines a single control point per line using four real numbers. The first three encode the position and the last one is the radius of the control point. At least two control points need to be specified for a single curve. Empty lines between control points are used to indicate the beginning of a new curve. Here is an example of two curves, the first with 2 control points and static radii and the second with 4 control points and increasing radii:
```
-1.00.10.10.5
1.01.41.20.5

-1.05.02.21
-2.34.02.32
4.01.02.25
4.00.02.36

```
Copy to clipboard
XMLPython
```
<shapetype="linearcurve">
<transformname="to_world">
<translatex="1"y="0"z="0"/>
<scalevalue="2"/>
</transform>
<stringname="filename"type="curves.txt"/>
</shape>

```
Copy to clipboard
```
'curves': {
    'type': 'linearcurve',
    'to_world': mi.ScalarAffineTransform4f().scale([2, 2, 2]).translate([1, 0, 0]),
    'filename': 'curves.txt'
},

```
Copy to clipboard
Note
The backfaces of the curves are culled. It is therefore impossible to intersect the curve with a ray that’s origin is inside of the curve.
## SDF Grid (sdfgrid)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#sdf-grid-sdfgrid "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
filename | string | Filename of the SDF grid data to be loaded. The expected file format aligns with a single-channel [grid-based volume data source](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_volumes.html#volume-gridvolume). If no filename is provided, the shape is initialised as an empty 2x2x2 grid. |   
grid | tensor | Tensor array containing the grid data. This parameter can only be specified when building this plugin at runtime from Python or C++ and cannot be specified in the XML scene description. | P, ∂, D  
normals | string | Specifies the method for computing shading normals. The options are analytic or smooth. (Default: smooth) |   
to_world | transform | Specifies a linear object-to-world transformation. (Default: none (i.e. object space = world space)) | P, ∂, D  
[![../../_images/shape_sdfgrid.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_sdfgrid.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_sdfgrid.jpg)
SDF grid with `smooth` shading normals[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id23 "Link to this image")
[![../../_images/shape_sdfgrid_analytic.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_sdfgrid_analytic.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_sdfgrid_analytic.jpg)
SDF grid with `analytic` shading normals[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id24 "Link to this image")
This shape plugin describes a signed distance function (SDF) grid shape primitive — that is, an SDF sampled onto a three-dimensional grid. The grid object-space is mapped over the range [0,1]3.
A smooth method for computing normals [[SEAM22](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id49 "Herman Hansson Söderlund, Alex Evans, and Tomas Akenine-Möller. Ray tracing of signed distance function grids. Journal of Computer Graphics Techniques \(JCGT\), 11\(3\):94–113, September 2022. URL: http://jcgt.org/published/0011/03/06/.")] is selected as the default approach to ensure continuity across grid cells.
Warning
Compared with the other available shape plugins, the SDF grid has a few important limitations. Namely:
  * It does not emit UV coordinates for texturing.
  * It cannot be used as an area emitter.


Note
When differentiating this shape, it does not leverage the work presented in [[VSJ22](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id50 "Delio Vicini, Sébastien Speierer, and Wenzel Jakob. Differentiable signed distance function rendering. Transactions on Graphics \(Proceedings of SIGGRAPH\), 41\(4\):125:1–125:18, July 2022. doi:10.1145/3528223.3530139.")]. However, a Mitsuba 3-based implementation of that technique is available on [its project’s page](https://github.com/rgl-epfl/differentiable-sdf-rendering).
XMLPython
```
<shapetype="sdfgrid">
<stringname="filename"value="data.sdf"/>
<bsdftype="diffuse"/>
</shape>

```
Copy to clipboard
```
'type': 'sdfgrid',
'filename': 'data.sdf'
'bsdf': {
    'type': 'diffuse'
}

```
Copy to clipboard
## Shape group (shapegroup)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#shape-group-shapegroup "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
(Nested plugin) | shape | One or more shapes that should be made available for geometry instancing |   
This plugin implements a container for shapes that should be made available for geometry instancing. Any shapes placed in a shapegroup will not be visible on their own—instead, the renderer will precompute ray intersection acceleration data structures so that they can efficiently be referenced many times using the [Instance (instance)](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#shape-instance) plugin. This is useful for rendering things like forests, where only a few distinct types of trees have to be kept in memory. An example is given below:
XMLPython
```
<!-- Declare a named shape group containing two objects -->
<shapetype="shapegroup"id="my_shape_group">
<shapetype="ply">
<stringname="filename"value="data.ply"/>
<bsdftype="roughconductor"/>
</shape>
<shapetype="sphere">
<transformname="to_world">
<translatey="20"/>
<scalevalue="5"/>
</transform>
<bsdftype="diffuse"/>
</shape>
</shape>

<!-- Instantiate the shape group without any kind of transformation -->
<shapetype="instance">
<refid="my_shape_group"/>
</shape>

<!-- Create instance of the shape group, but rotated, scaled, and translated -->
<shapetype="instance">
<refid="my_shape_group"/>
<transformname="to_world">
<translatez="10"/>
<scalevalue="1.5"/>
<rotatex="1"angle="45"/>
</transform>
</shape>

```
Copy to clipboard
```
# Declare a named shape group containing two objects
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
        'to_world': mi.ScalarAffineTransform4f().scale([5, 5, 5]).translate([0, 20, 0])
        'bsdf': {
            'type': 'diffuse',
        }
    }
},

# Instantiate the shape group without any kind of transformation
'first_instance': {
    'type': 'instance',
    'shapegroup': {
        'type': 'ref',
        'id': 'my_shape_group'
    }
},

# Create instance of the shape group, but rotated, scaled, and translated
'second_instance': {
    'type': 'instance',
    'to_world': mi.ScalarAffineTransform4f().rotate([1, 0, 0], 45).scale([1.5, 1.5, 1.5]).translate([0, 10, 0]),
    'shapegroup': {
        'type': 'ref',
        'id': 'my_shape_group'
    }
}

```
Copy to clipboard
## Instance (instance)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#instance-instance "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
(Nested plugin) | shapegroup | A reference to a shape group that should be instantiated. |   
to_world | transform | Specifies a linear object-to-world transformation. (Default: none (i.e. object space = world space)) | P, ∂, D  
This plugin implements a geometry instance used to efficiently replicate geometry many times. For details on how to create instances, refer to the [Shape group (shapegroup)](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#shape-shapegroup) plugin.
> [![../../_images/shape_instance_fractal.jpg](https://mitsuba.readthedocs.io/en/stable/_images/shape_instance_fractal.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/shape_instance_fractal.jpg)
> The Stanford bunny loaded a single time and instantiated 1365 times (equivalent to 100 million triangles)
Warning
  * Note that it is not possible to assign a different material to each instance — the material assignment specified within the shape group is the one that matters.
  * Shape groups cannot be used to replicate shapes with attached emitters, sensors, or subsurface scattering models.


## Ellipsoids (ellipsoids)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#ellipsoids-ellipsoids "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
filename | string | Specifies the PLY file containing the ellipsoid centers, scales, and quaternions. This parameter cannot be used if `data` or `centers` are provided. |   
data | tensor | A tensor of shape (N, 10) or (N * 10) that defines the ellipsoid centers, scales, and quaternions. This parameter cannot be used if `filename` or `centers` are provided. |   
centers | tensor | A tensor of shape (N, 3) specifying the ellipsoid centers. This parameter cannot be used if `filename` or `data` are provided. |   
scales | tensor | A tensor of shape (N, 3) specifying the ellipsoid scales. This parameter cannot be used if `filename` or `data` are provided. |   
quaternions | tensor | A tensor of shape (N, 3) specifying the ellipsoid quaternions. This parameter cannot be used if `filename` or `data` are provided. |   
scale_factor | float | A scaling factor applied to all ellipsoids when loading from a PLY file. (Default: 1.0) |   
extent | float | Specifies the extent of the ellipsoid. This effectively acts as an extra scaling factor on the ellipsoid, without having to alter the scale parameters. (Default: 3.0) | R  
extent_adaptive_clamping | float | If True, use adaptive extent values based on the `opacities` attribute of the volumetric primitives. (Default: False) | R  
to_world | transform | Specifies an optional linear object-to-world transformation to apply to all ellipsoids. | P, ∂, D  
(Nested plugin) | tensor | Specifies arbitrary ellipsoids attribute as a tensor of shape (N, D) with D the dimensionality of the attribute. For instance this can be used to define an opacity value for each ellipsoids, or a set of spherical harmonic coefficients as used in the volprim_rf_basic integrator. |   
This shape plugin defines a point cloud of anisotropic ellipsoid primitives using specified centers, scales, and quaternions. It employs a closed-form ray-intersection formula with backface culling. Although it is slower than the ellipsoidsmesh shape plugin, which uses tessellated ellipsoids for ray-triangle intersection hardware acceleration, it offers greater flexibility in computing intersections.
This shape also exposes an `extent` parameter, it acts as an extra scaling factor for the ellipsoids’ scales. Typically, this is used to define the support of a kernel function defined within the ellipsoid. For example, the `scale` parmaters of the ellipsoid will define the variances of a gaussian and the `extent` will multiple those value to define the effictive radius of the ellipsoid. When `extent_adaptive_clamping` is enabled, the extent is additionally multiplied by an opacity-dependent factor:2∗log⁡(opacity/0.01)/3
It is designed for use with volumetric primitive integrators, as detailed in [[CSB+25](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id58 "Jorge Condor, Sebastien Speierer, Lukas Bode, Aljaz Bozic, Simon Green, Piotr Didyk, and Adrian Jarabo. Don't Splat your Gaussians: Volumetric Ray-Traced Primitives for Modeling and Rendering Scattering and Emissive Media. ACM Trans. Graph., 2025. URL: https://doi.org/10.1145/3711853, doi:10.1145/3711853.")].
XMLPython
```
<shapetype="ellipsoids">
<stringname="filename"value="my_primitives.ply"/>
</shape>

```
Copy to clipboard
```
'primitives': {
    'type': 'ellipsoids',
    'filename': 'my_primitives.ply'
}

```
Copy to clipboard
## Mesh ellipsoids (ellipsoidsmesh)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#mesh-ellipsoids-ellipsoidsmesh "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
filename | string | Specifies the PLY file containing the ellipsoid centers, scales, and quaternions. This parameter cannot be used if `data` or `centers` are provided. |   
data | tensor | A tensor of shape (N, 10) or (N * 10) that defines the ellipsoid centers, scales, and quaternions. This parameter cannot be used if `filename` or `centers` are provided. |   
centers | tensor | A tensor of shape (N, 3) specifying the ellipsoid centers. This parameter cannot be used if `filename` or `data` are provided. |   
scales | tensor | A tensor of shape (N, 3) specifying the ellipsoid scales. This parameter cannot be used if `filename` or `data` are provided. |   
quaternions | tensor | A tensor of shape (N, 3) specifying the ellipsoid quaternions. This parameter cannot be used if `filename` or `data` are provided. |   
scale_factor | float | A scaling factor applied to all ellipsoids when loading from a PLY file. (Default: 1.0) |   
extent | float | Specifies the extent of the ellipsoid. This effectively acts as an extra scaling factor on the ellipsoid, without having to alter the scale parameters. (Default: 3.0) | R  
extent_adaptive_clamping | float | If True, use adaptive extent values based on the `opacities` attribute of the volumetric primitives. (Default: False) | R  
shell | string or [|mesh|](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#id25) | Specifies the shell type. Could be one of box, ico_sphere, or uv_sphere, as well as a custom child mesh object. (Default: `ico_sphere`) |   
to_world | transform | Specifies an optional linear object-to-world transformation to apply to all ellipsoids. | P, ∂, D  
(Nested plugin) | tensor | Specifies arbitrary ellipsoids attribute as a tensor of shape (N, D) with D the dimensionality of the attribute. For instance this can be used to define an opacity value for each ellipsoids, or a set of spherical harmonic coefficients as used in the volprim_rf_basic integrator. |   
This shape plugin defines a point cloud of anisotropic ellipsoid primitives given centers, scales, and quaternions, using a mesh-based representation with backface culling. This plugin is designed to leverage hardware acceleration for ray-triangle intersections, providing a performance advantage over analytical ellipsoid representations.
This shape also exposes an `extent` parameter, it acts as an extra scaling factor for the ellipsoids’ scales. Typically, this is used to define the support of a kernel function defined within the ellipsoid. For example, the `scale` parmaters of the ellipsoid will define the variances of a gaussian and the `extent` will multiple those value to define the effictive radius of the ellipsoid. When `extent_adaptive_clamping` is enabled, the extent is additionally multiplied by an opacity-dependent factor:2∗log⁡(opacity/0.01)/3
This shape is designed for use with volumetric primitive integrators, as detailed in [[CSB+25](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id58 "Jorge Condor, Sebastien Speierer, Lukas Bode, Aljaz Bozic, Simon Green, Piotr Didyk, and Adrian Jarabo. Don't Splat your Gaussians: Volumetric Ray-Traced Primitives for Modeling and Rendering Scattering and Emissive Media. ACM Trans. Graph., 2025. URL: https://doi.org/10.1145/3711853, doi:10.1145/3711853.")].
XMLPython
```
<shapetype="ellipsoidsmesh">
<stringname="filename"value="my_primitives.ply"/>
</shape>

```
Copy to clipboard
```
'primitives': {
    'type': 'ellipsoidsmesh',
    'filename': 'my_primitives.ply'
}

```
Copy to clipboard
* * *
  *[P]: This parameter will be exposed as a scene parameter
  *[∂]: This parameter is differentiable
  *[D]: This parameter might introduce discontinuities. Therefore it requires special handling during differentiation to prevent bias)
  *[R]: This parameter will be exposed as a scene parameter, but cannot be modified.
