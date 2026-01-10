---
url: https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html
crawled_at: 2025-11-13T17:17:22.914476
title: https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Scene XML file format[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#scene-xml-file-format "Link to this heading")
Mitsuba uses a simple and general XML-based format to represent scenes. Since the framework’s philosophy is to represent discrete blocks of functionality as plugins, a scene file can be interpreted as “recipe” specifying which plugins should be instantiated and how they should be put together. In the following, we will look at a few examples to get a feeling for the scope of the format.
A simple scene with a single mesh with no lighting and the default camera setup might look something like this:
```
<sceneversion="3.0.0">
<shapetype="obj">
<stringname="filename"value="dragon.obj"/>
</shape>
</scene>

```
Copy to clipboard
The `version` attribute in the first line denotes the release of Mitsuba that was used to create the scene. This information allows Mitsuba to correctly process the file regardless of any potential future changes in the scene description language.
This example already contains the most important things to know about format: it consists of _objects_ (such as the objects instantiated by the `<scene>` or `<shape>` tags), which can furthermore be nested within each other. Each object optionally accepts _properties_ (such as the `<string>` tag) that characterize its behavior. All objects except for the root object (the `<scene>`) cause the renderer to search and load a plugin from disk, hence you must provide the plugin name using `type=".."` parameter.
The object tags also let the renderer know _what kind_ of object is to be instantiated: for instance, any plugin loaded using the `<shape>` tag must conform to the _Shape_ interface, which is certainly the case for the plugin named `obj` (it contains a [Wavefront OBJ loader](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#shape-obj)). Similarly, you could write
```
<sceneversion="3.0.0">
<shapetype="sphere">
<floatname="radius"value="10"/>
</shape>
</scene>

```
Copy to clipboard
This loads a different plugin (`sphere`) which is still a _Shape_ but instead represents a [sphere](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#shape-sphere) configured with a radius of 10 world-space units. Mitsuba ships with a large number of plugins; please refer to the [Plugin reference](https://mitsuba.readthedocs.io/en/stable/src/plugin_reference.html#sec-plugins) section for a detailed overview of them.
The most common scene setup is to declare an integrator, some geometry, a sensor (e.g. a camera), a film, a sampler and one or more emitters. Here is a more complex example:
```
<sceneversion="3.0.0">
<integratortype="path">
<!-- Instantiate a path tracer with a max. path length of 8 -->
<integername="max_depth"value="8"/>
</integrator>

<!-- Instantiate a perspective camera with 45 degrees field of view -->
<sensortype="perspective">
<!-- Rotate the camera around the Y axis by 180 degrees -->
<transformname="to_world">
<rotatey="1"angle="180"/>
</transform>
<floatname="fov"value="45"/>

<!-- Render with 32 samples per pixel using a basic
             independent sampling strategy -->
<samplertype="independent">
<integername="sample_count"value="32"/>
</sampler>

<!-- Generate an EXR image at HD resolution -->
<filmtype="hdrfilm">
<integername="width"value="1920"/>
<integername="height"value="1080"/>
</film>
</sensor>

<!-- Add a dragon mesh made of rough glass (stored as OBJ file) -->
<shapetype="obj">
<stringname="filename"value="dragon.obj"/>

<bsdftype="roughdielectric">
<!-- Tweak the roughness parameter of the material -->
<floatname="alpha"value="0.01"/>
</bsdf>
</shape>

<!-- Add another mesh, this time, stored using Mitsuba's own
         (compact) binary representation -->
<shapetype="serialized">
<stringname="filename"value="lightsource.serialized"/>
<transformname="to_world">
<translatex="5"y="-3"z="1"/>
</transform>

<!-- This mesh is an area emitter -->
<emittertype="area">
<rgbname="radiance"value="100,400,100"/>
</emitter>
</shape>
</scene>

```
Copy to clipboard
This example introduces several new object types (`integrator`, `sensor`, `bsdf`, `sampler`, `film`, and `emitter`) and property types (`integer`, `transform`, and `rgb`). As you can see in the example, objects are usually declared at the top level except if there is some inherent relation that links them to another object. For instance, BSDFs are usually specific to a certain geometric object, so they appear as a child object of a shape. Similarly, the sampler and film affect the way in which rays are generated from the sensor and how it records the resulting radiance samples, hence they are nested inside it. The following table provides an overview of the available object types:
XML tag | Description | [`type`](https://docs.python.org/3/library/functions.html#type "\(in Python v3.13\)") examples  
---|---|---  
`bsdf` | BSDF describe the manner in which light interacts with surfaces in the scene (i.e., the _material_) | `diffuse`, `conductor`  
`emitter` | Emitter plugins specify light sources and their characteristic emission profiles. | `constant`, `envmap`, `point`  
`film` | Film plugins convert measurements into the final output file that is written to disk | `hdrfilm`, `specfilm`  
`integrator` | Integrators implement rendering techniques for solving the light transport equation | `path`, `direct`, `depth`  
`rfilter` | Reconstruction filters control how the `film` converts a set of samples into the output image | `box`, `gaussian`  
`sampler` | Sample generator plugins used by the `integrator` | `independent`, `multijitter`  
`sensor` | Sensor plugins like cameras are responsible for measuring radiance | `perspective`, `orthogonal`  
`shape` | Shape puglins define surfaces that mark transitions between different types of materials | `obj`, `ply`, `serialized`  
`texture` | Texture plugins represent spatially varying signals on surfaces | `bitmap`, `checkerboard`  
This table lists the different kind of _objects_ and their respective tags. It also provides an exemplary list of plugins for each category.
## Properties[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#properties "Link to this heading")
This subsection documents all of the ways in which properties can be supplied to objects. If you are more interested in knowing which properties a certain plugin accepts, you should look at the [plugin documentation](https://mitsuba.readthedocs.io/en/stable/src/plugin_reference.html#sec-plugins) instead.
### Numbers[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#numbers "Link to this heading")
Integer and floating point values can be passed as follows:
```
<integername="int_property"value="1234"/>
<floatname="float_property"value="-1.5e3"/>

```
Copy to clipboard
Note that you must adhere to the format expected by the object, i.e. you can’t pass an integer property to an object that expects a floating-point property associated with that name.
### Booleans[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#booleans "Link to this heading")
Boolean values can be passed as follows:
```
<booleanname="bool_property"value="true"/>

```
Copy to clipboard
### Strings[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#strings "Link to this heading")
Passing strings is similarly straightforward:
```
<stringname="string_property"value="This is a string"/>

```
Copy to clipboard
### Vectors, Positions[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#vectors-positions "Link to this heading")
Points and vectors can be specified as follows:
```
<pointname="point_property"value="3, 4, 5"/>
<vectorname="vector_property"value="3, 4, 5"/>

```
Copy to clipboard
Note
Mitsuba does not dictate a specific unit for position values (meters, centimeters, inches, etc.). The only requirement is that you consistently use one convention throughout the scene specification.
### RGB Colors[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#rgb-colors "Link to this heading")
In Mitsuba, colors are either specified using the `<rgb>` or `<spectrum>` tags. The interpretation of a RGB color value like
```
<rgbname="color_property"value="0.2, 0.8, 0.4"/>

```
Copy to clipboard
depends on the variant of the renderer that is currently active. For instance, `scalar_rgb` will simply forward the color value to the underlying plugin without changes. In contrast, `scalar_spectral` operates in the spectral domain where a RGB value is not meaningful—worse, there is an infinite set of spectra corresponding to each RGB color. Mitsuba uses the method of Jakob and Hanika [[JH19](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id9 "Wenzel Jakob and Johannes Hanika. A low-dimensional function space for efficient spectral upsampling. Computer Graphics Forum \(Proceedings of Eurographics\), March 2019. URL: https://rgl.epfl.ch/publications/Jakob2019Spectral.")] to choose a plausible smooth spectrum amongst all of these possibilities. An example is shown below:
[![../../_images/upsampling.jpg](https://mitsuba.readthedocs.io/en/stable/_images/upsampling.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/upsampling.jpg)
### Color spectra[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#color-spectra "Link to this heading")
A more accurate way or specifying color information involves the `<spectrum>` tag, which records a reflectance/intensity value for multiple discrete wavelengths specified in _nanometers_.
```
<spectrumname="color_property"value="400:0.56, 500:0.18, 600:0.58, 700:0.24"/>

```
Copy to clipboard
The resulting spectrum uses linearly interpolation for in-between wavelengths and equals zero outside of the specified wavelength range. The following short-hand notation creates a spectrum that is uniform across wavelengths:
```
<spectrumname="color_property"value="0.5"/>

```
Copy to clipboard
When spectral power or reflectance distributions are obtained from measurements (e.g. at 10nm intervals), they are usually quite unwieldy and can clutter the scene description. For this reason, there is yet another way to pass a spectrum by loading it from an external file:
```
<spectrumname="color_property"filename="measured_spectrum.spd"/><spectrumname="color_property"filename="measured_binary_spectrum.spb"/>
```
Copy to clipboard
The file should contain a single measurement per line, with the corresponding wavelength in nanometers and the measured value separated by a space. Comments are allowed. Here is an example:
Copy to clipboard
Mitsuba provides a function ([`mitsuba.spectrum_to_file()`](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.spectrum_to_file "mitsuba.spectrum_to_file")) to create such file, given the wavelengths and its values.
For more details regarding spectral information in Mitsuba 3, please have a look at the [corresponding section](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#sec-spectra) in the plugin documentation.
### Transformations[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#transformations "Link to this heading")
Transformations are the only kind of property that require more than a single tag. The idea is that, starting with the identity, one can build up a transformation using a sequence of commands. For instance, a transformation that does a translation followed by a rotation might be written like this:
```
<transformname="trafo_property">
<translatevalue="-1, 3, 4"/>
<rotatey="1"angle="45"/>
</transform>

```
Copy to clipboard
Mathematically, each incremental transformation in the sequence is left-multiplied onto the current one. The following choices are available:
  * Translations:
```
<translatevalue="-1, 3, 4"/>

```
Copy to clipboard
  * Counter-clockwise rotations around a specified axis. The angle is given in degrees:
```
<rotatevalue="0.701, 0.701, 0"angle="180"/>

```
Copy to clipboard
  * Scaling operation. The coefficients may also be negative to obtain a flip:
```
<scalevalue="5"/><!-- uniform scale -->
<scalevalue="2, 1, -1"/><!-- non-uniform scale -->

```
Copy to clipboard
  * Explicit 4x4 matrices in row-major order:
```
<matrixvalue="0 -0.53 0 -1.79 0.92 0 0 8.03 0 0 0.53 0 0 0 0 1"/>

```
Copy to clipboard
  * Explicit 3x3 matrices in row-major order. Internally, this will be converted to a 4x4 matrix with the same last row and column as the identity matrix.
```
<matrixvalue="0.57 0.2 0 0.1 -1 0 0 0 1"/>

```
Copy to clipboard
  * `lookat` transformations – this is primarily useful for setting up cameras. The `origin` coordinates specify the camera origin, `target` is the point that the camera will look at, and the (optional) `up` parameter determines the _upward_ direction in the final rendered image.
```
<lookatorigin="10, 50, -800"target="0, 0, 0"up="0, 1, 0"/>

```
Copy to clipboard


## References[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#references "Link to this heading")
Quite often, you will find yourself using an object (such as a material) in many places. To avoid having to declare it over and over again, which wastes memory, you can make use of references. Here is an example of how this works:
```
<sceneversion="3.0.0">
<texturetype="bitmap"id="my_image">
<stringname="filename"value="textures/my_image.jpg"/>
</texture>

<bsdftype="diffuse"id="my_material">
<!-- Reference the texture named my_image and pass it
             to the BSDF as the reflectance parameter -->
<refname="reflectance"id="my_image"/>
</bsdf>

<shapetype="obj">
<stringname="filename"value="meshes/my_shape.obj"/>

<!-- Reference the material named my_material -->
<refid="my_material"/>
</shape>
</scene>

```
Copy to clipboard
By providing a unique [`id`](https://docs.python.org/3/library/functions.html#id "\(in Python v3.13\)") attribute in the object declaration, the object is bound to that identifier upon instantiation. Referencing this identifier at a later point (using the `<ref id=".."/>` tag) will add the instance to the parent object.
Note
Note that while this feature is meant to efficiently handle materials, textures, and participating media that are referenced from multiple places, it cannot be used to instantiate geometry. The [instance](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#shape-instance) plugin should be used for that purpose.
## Default parameters[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#default-parameters "Link to this heading")
Scene may contain named parameters that are supplied via the command line:
```
<bsdftype="diffuse">
<rgbname="reflectance"value="$reflectance"/>
</bsdf>

```
Copy to clipboard
In this case, an error will be raised when the scene is loaded without an explicit command line argument of the form `-Dreflectance=...`. For convenience, it is possible to specify a default parameter value that take precedence when no command line arguments are given. The syntax for this is:
```
<defaultname="reflectance"value="something"/>

```
Copy to clipboard
and must precede the occurrences of the parameter in the XML file.
## Including external files[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#including-external-files "Link to this heading")
A scene can be split into multiple pieces for better readability. To include an external file, please use the following command:
```
<includefilename="nested-scene.xml"/>

```
Copy to clipboard
In this case, the file `nested-scene.xml` must be a proper scene file with a `<scene>` tag at the root.
This feature is often very convenient in conjunction with the `-D key=value` flag of the `mitsuba` command line renderer. This enables including different variants of a scene configuration by changing the command line parameters, without without having to touch the XML file:
```
<includefilename="nested-scene-$version.xml"/>

```
Copy to clipboard
## Aliases[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#aliases "Link to this heading")
It is sometimes useful to associate an object with multiple identifiers. This can be accomplished using the `alias as=".."` tag:
```
<bsdftype="diffuse"id="my_material_1"/>
<aliasid="my_material_1"as="my_material_2"/>

```
Copy to clipboard
After this statement, the diffuse scattering model will be bound to _both_ identifiers `my_material_1` and `my_material_2`.
## External resource folders[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#external-resource-folders "Link to this heading")
Using the `path` tag, it is possible to add a path to the list of search paths. This can be useful for instance when some meshes and textures are stored in a different directory, (e.g. when shared with other scenes). If the path is a relative path, Mitsuba 3 will first try to interpret it relative to the scene directory, then to other paths that are already on the search path (e.g. added using the `-a <path1>;<path2>;..` command line argument).
```
<pathvalue="../../my_resources"/>

```
Copy to clipboard
# Dictionary-based scene format[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#dictionary-based-scene-format "Link to this heading")
The function [`mitsuba.load_dict()`](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.load_dict "mitsuba.load_dict") provides a convenient alternative way of constructing Mitsuba objects using Python dictionaries. They express the same high-level structure while using native Python data types.
A dictionary must always contain an entry `"type"` to specify the name of the plugin to be instantiated. Dictionary keys must be strings and represent the name of the properties passed to the plugin. The property type is automatically deduced from the underlying Python type (e.g. `bool`, `float`, `int`, `str`, …). Using a dictionary as value creates a nested object.
The following snippets illustrate the similarity between the XML code and the Python dictionary structure. Also note that similarly, the [plugin documentation](https://mitsuba.readthedocs.io/en/stable/src/plugin_reference.html#sec-plugins) section provides both XML snippets and the corresponding python `dict` code examples for all referenced plugins.
XMLPython
```
<shapetype="obj">
<stringname="filename"value="dragon.obj"/>
<bsdftype="roughconductor">
<floatname="alpha"value="0.01"/>
</bsdf>
</shape>

```
Copy to clipboard
```
{
    "type": "obj",
    "filename": "dragon.obj",
    "bsdf_id": {
        "type": "roughconductor",
        "alpha": 0.01
    }
}

```
Copy to clipboard
Here is a concrete example on how to use [`mitsuba.load_dict()`](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.load_dict "mitsuba.load_dict"):
```
sphere = mi.load_dict({
    "type": "sphere",
    "center": [0, 0, -10],
    "radius": 10.0,
    "flip_normals": False,
    "bsdf": {
        "type": "dielectric"
    }
})

```
Copy to clipboard
It is also possible to call this function several times and pass previously constructed objects instead of nesting dictionaries:
```
# First create a BSDF (could use xml.load_string(..) as well)
my_bsdf = mi.load_dict({
    "type": "roughconductor",
    "alpha": 0.14,
})

# Pass the BSDF object in the dictionary
sphere = load_dict({
    "type": "sphere",
    "something": my_bsdf
})

```
Copy to clipboard
For convenience, a nested dictionary can be provided with a `"type"` entry equal to `"rgb"` or `"spectrum"`. Similarly to the XML parser, the `"value"` entry in that dictionary will be used to instantiate the right `Spectrum` plugin. (See the [corresponding section](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_spectra.html#sec-spectra))
Here as some examples of the possible use of the `"value"` entry in the nested dictionary:
```
# Passing gray-scale value
"color_property": {
    "type": "rgb",
    "value": 0.44
}

# Passing tristimulus values
"color_property": {
    "type": "rgb",
    "value": [0.7, 0.1, 0.5]
}

# Providing a spectral file
"color_property": {
    "type": "spectrum",
    "filename": "filename.spd"
}

# Providing a list of (wavelength, value) pairs
"color_property": {
    "type": "spectrum",
    "value": [(400.0, 0.5), (500.0, 0.8), (600.0, 0.2)]
}

```
Copy to clipboard
The following example constructs a Mitsuba complete scene using [`mitsuba.load_dict()`](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.load_dict "mitsuba.load_dict"):
```
scene = mi.load_dict({
    "type": "scene",
    "myintegrator": {
        "type": "path",
    },
    "mysensor": {
        "type": "perspective",
        "near_clip": 1.0,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.look_at(origin=[1, 1, 1],
                                                 target=[0, 0, 0],
                                                 up=[0, 0, 1]),
        "myfilm": {
            "type": "hdrfilm",
            "rfilter": {
                "type": "box"
            },
            "width": 1024,
            "height": 768,
        }, "mysampler": {
            "type": "independent",
            "sample_count": 4,
        },
    },
    "myemitter": {
        "type": "constant"
    },
    "myshape": {
        "type": "sphere",
        "mybsdf": {
            "type": "diffuse",
            "reflectance": {
                "type": "rgb",
                "value": [0.8, 0.1, 0.1],
            }
        }
    }
})

```
Copy to clipboard
Like in the XML parser, it is possible to declare scene objects once and then reference them elsewhere.
```
{
    "type": "scene",

    "shape_1": {
        "type": "obj",
        "filename": "shape_1.obj",
        "bsdf": {
            "type": "diffuse",
            "id": "my_material"
        }
    },

    "shape_2": {
        "type": "sphere",
        "filename": "shape_2.obj",

        # Reuse the material from shape 1
        "bsdf": {
            "type": "ref",
            "id": "my_material"  # Explicit ID reference
        }
    }
}

```
Copy to clipboard
A feature that is specific to the dictionary parser is that every object in the dictionary automatically receives an implicit ID based on the path within the dictionary (i.e., the sequence of dictionary keys needed to reach the object).
For example, in the snippet below, the objects receive the following implicit IDs: - The shape: `my_shape` - The material: `my_shape.my_material` - The texture: [``](https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#id3)my_shape.my_material.reflectance`
```
{
    "type": "scene",
    "my_shape": {
        "type": "sphere",
        "my_material": {
            "type": "diffuse",
            "reflectance": {
                "type": "bitmap",
                "filename": "texture.jpg"
            }
        }
    }
}

```
Copy to clipboard
**Important Notes** :
  * Dictionary keys cannot contain dots (`.`) as they are reserved for path delimiters. Use underscores instead.
  * Objects can be referenced before or after their declaration—forward references are fully supported.


As in the XML scene description, it is possible to add a path to the list of search paths. In the following example, the texture file can be found in the `/home/username/data/textures/` folder. Note that this added path can be relative or absolute.
```
{
    "type": "scene",
    'foo': { 'type': 'resources', 'path': '/home/username/data/textures'},
    "bsdf": {
        "type": "diffuse",
        "reflectance": {
            "type": "bitmap",
            "filename": "my_texture.exr", # relative to the folder defined above
    }
}

```
Copy to clipboard
[**MongoDB Atlas empowers you** to build modern apps where you want, how you want, at the speed you want.](https://server.ethicalads.io/proxy/click/9503/019a7c81-46d2-7bb3-b229-23a0d1d6762d/)
[Ads by EthicalAds](https://www.ethicalads.io/advertisers/?ref=ea-text)
* * *
