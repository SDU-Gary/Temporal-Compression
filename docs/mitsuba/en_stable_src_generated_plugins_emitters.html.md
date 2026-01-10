---
url: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html
crawled_at: 2025-11-13T17:13:56.558799
title: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Emitters[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#emitters "Link to this heading")
> [![../../_images/emitter_overview.jpg](https://mitsuba.readthedocs.io/en/stable/_images/emitter_overview.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/emitter_overview.jpg)
> Schematic overview of the emitters in Mitsuba 3. The arrows indicate the directional distribution of light.
Mitsuba 3 supports a number of different emitters/light sources, which can be classified into two main categories: emitters which are located somewhere within the scene, and emitters that surround the scene to simulate a distant environment.
Generally, light sources are specified as children of the `<scene>` element; for instance, the following snippet instantiates a point light emitter that illuminates a sphere:
XMLPython
```
<sceneversion="3.0.0">
<!-- .. scene contents .. -->

<emittertype="point">
<rgbname="intensity"value="1"/>
<pointname="position"x="0"y="0"z="-2"/>
</emitter>

<shapetype="sphere"/>
</scene>

```
Copy to clipboard
```
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

```
Copy to clipboard
An exception to this are area lights, which turn a geometric object into a light source. These are specified as children of the corresponding `<shape>` element:
XMLPython
```
<sceneversion="3.0.0">
<!-- .. scene contents .. -->

<shapetype="sphere">
<emittertype="area">
<rgbname="radiance"value="1"/>
</emitter>
</shape>
</scene>

```
Copy to clipboard
```
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

```
Copy to clipboard
## Area light (area)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#area-light-area "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
radiance | spectrum or texture | Specifies the emitted radiance in units of power per unit area per unit steradian. | P, ∂  
This plugin implements an area light, i.e. a light source that emits diffuse illumination from the exterior of an arbitrary shape. Since the emission profile of an area light is completely diffuse, it has the same apparent brightness regardless of the observer’s viewing direction. Furthermore, since it occupies a nonzero amount of space, an area light generally causes scene objects to cast soft shadows.
To create an area light source, simply instantiate the desired emitter shape and specify an area instance as its child:
XMLPython
```
<shapetype="sphere">
<emittertype="area">
<rgbname="radiance"value="1.0"/>
</emitter>
</shape>

```
Copy to clipboard
```
'type': 'sphere',
'emitter': {
    'type': 'area',
    'radiance': {
        'type': 'rgb',
        'value': 1.0,
    }
}

```
Copy to clipboard
## Point light source (point)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#point-light-source-point "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
intensity | spectrum | Specifies the radiant intensity in units of power per unit steradian. | P, ∂  
position | point | Alternative parameter for specifying the light source position. Note that only one of the parameters to_world and position can be used at a time. | P  
to_world | transform | Specifies an optional emitter-to-world transformation. (Default: none, i.e. emitter space = world space) |   
This emitter plugin implements a simple point light source, which uniformly radiates illumination into all directions.
XMLPython
```
<emittertype="point">
<pointname="position"value="0.0, 5.0, 0.0"/>
<rgbname="intensity"value="1.0"/>
</emitter>

```
Copy to clipboard
```
'type': 'point',
'position': [0.0, 5.0, 0.0],
'intensity': {
    'type': 'spectrum',
    'value': 1.0,
}

```
Copy to clipboard
## Constant environment emitter (constant)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#constant-environment-emitter-constant "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
radiance | spectrum | Specifies the emitted radiance in units of power per unit area per unit steradian. | P, ∂  
This plugin implements a constant environment emitter, which surrounds the scene and radiates diffuse illumination towards it. This is often a good default light source when the goal is to visualize some loaded geometry that uses basic (e.g. diffuse) materials.
XMLPython
```
<emittertype="constant">
<rgbname="radiance"value="1.0"/>
</emitter>

```
Copy to clipboard
```
'type': 'constant',
'radiance': {
    'type': 'rgb',
    'value': 1.0,
}

```
Copy to clipboard
## Environment emitter (envmap)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#environment-emitter-envmap "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
filename | string | Filename of the radiance-valued input image to be loaded; must be in latitude-longitude format. |   
bitmap | Bitmap object | When creating a Environment emitter at runtime, e.g. from Python or C++, an existing Bitmap image instance can be passed directly rather than loading it from the filesystem with filename. |   
scale | float | A scale factor that is applied to the radiance values stored in the input image. (Default: 1.0) | P, ∂  
to_world | transform | Specifies an optional emitter-to-world transformation. (Default: none, i.e. emitter space = world space) | P  
State parameters |  |  |   
mis_compensation | boolean | Compensate sampling for the presence of other Monte Carlo techniques that will be combined using multiple importance sampling (MIS)? This is extremely cheap to do and can slightly reduce variance. (Default: false) |   
data | tensor | Tensor array containing the radiance-valued data. | P, ∂, D  
This plugin provides a HDRI (high dynamic range imaging) environment map, which is a type of light source that is well-suited for representing “natural” illumination.
The implementation loads a captured illumination environment from a image in latitude-longitude format and turns it into an infinitely distant emitter. The conventions of this mapping are shown in this image:
[![../../_images/emitter_envmap_example.jpg](https://mitsuba.readthedocs.io/en/stable/_images/emitter_envmap_example.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/emitter_envmap_example.jpg)
The museum environment map by Bernhard Vogl that is used in many example renderings in this documentation.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#id6 "Link to this image")
[![../../_images/emitter_envmap_axes.jpg](https://mitsuba.readthedocs.io/en/stable/_images/emitter_envmap_axes.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/emitter_envmap_axes.jpg)
Coordinate conventions used when mapping the input image onto the sphere.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#id7 "Link to this image")
The plugin can work with all types of images that are natively supported by Mitsuba (i.e. JPEG, PNG, OpenEXR, RGBE, TGA, and BMP). In practice, a good environment map will contain high-dynamic range data that can only be represented using the OpenEXR or RGBE file formats. High quality free light probes are available on [Bernhard Vogl’s](http://dativ.at/lightprobes/) website or [Polyhaven](https://polyhaven.com/hdris).
XMLPython
```
<emittertype="envmap">
<stringname="filename"value="textures/museum.exr"/>
</emitter>

```
Copy to clipboard
```
'type': 'envmap',
'filename': 'textures/museum.exr'

```
Copy to clipboard
## Sun and sky emitter (sunsky)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#sun-and-sky-emitter-sunsky "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
turbidity | float | Atmosphere turbidity, must be within [1, 10] (Default: 3, clear sky in a temperate climate). Smaller turbidity values (∼ 1 − 2) produce an arctic-like clear blue sky, whereas larger values (∼ 8 − 10) create an atmosphere that is more typical of a warm, humid day. | P  
albedo | spectrum | Ground albedo, must be within [0, 1] for each wavelength/channel, (Default: 0.3). This cannot be spatially varying (e.g. have bitmap as type). | P  
latitude | float | Latitude of the location in degrees (Default: 35.689, Tokyo’s latitude). | P  
longitude | float | Longitude of the location in degrees (Default: 139.6917, Tokyo’s longitude). | P  
timezone | float | Timezone of the location in hours (Default: 9). | P  
year | integer | Year (Default: 2010). | P  
month | integer | Month (Default: 7). | P  
day | integer | Day (Default: 10). | P  
hour | float | Hour (Default: 15). | P  
minute | float | Minute (Default: 0). | P  
second | float | Second (Default: 0). | P  
sun_direction | vector | Direction of the sun in the sky (No defaults), cannot be specified along with one of the location/time parameters. | P, ∂  
sun_scale | float | Scale factor for the sun radiance (Default: 1). Can be used to turn the sun off (by setting it to 0). | P  
sky_scale | float | Scale factor for the sky radiance (Default: 1). Can be used to turn the sky off (by setting it to 0). | P  
sun_aperture | float | Aperture angle of the sun in degrees (Default: 0.5338, normal sun aperture). |   
to_world | transform | Specifies an optional emitter-to-world transformation. (Default: none, i.e. emitter space = world space) | P  
This plugin implements an environment emitter for the sun and sky dome. It uses the Hosek-Wilkie sun [[HW13](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id55 "Lukáš Hošek and Alexander Wilkie. Adding a solar-radiance function to the hošek-wilkie skylight model. IEEE Computer Graphics and Applications, 33\(3\):44-52, 2013. doi:10.1109/MCG.2013.18.")] and sky model [[HW12](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id56 "Lukas Hosek and Alexander Wilkie. An analytic model for full spectral sky-dome radiance. ACM Trans. Graph., 2012. doi:10.1145/2185520.2185591.")] to generate strong approximations of the sky-dome without the cost of path tracing the atmosphere.
The local reference frame of this emitter is Z-up and X being towards the north direction. This behaviour can be changed with the `to_world` parameter.
Internally, this emitter does not compute a bitmap of the sky-dome like an environment map, but evaluates the spectral radiance whenever it is needed. Consequently, sampling is done through a Truncated Gaussian Mixture Model pre-fitted to the given parameters [[VVP21](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id57 "Nick Vitsas, Konstantinos Vardis, and Georgios Papaioannou. Sampling Clear Sky Models using Truncated Gaussian Mixtures. In Adrien Bousseau and Morgan McGuire, editors, Eurographics Symposium on Rendering - DL-only Track. The Eurographics Association, 2021. doi:10.2312/sr.20211288.")].
### Parameter influence[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#parameter-influence "Link to this heading")
**Albedo (sky only)**
[![https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/sunsky_03_0_10.png](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/sunsky_03_0_10.png) ](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/sunsky_03_0_10.png)
albedo=0[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#id8 "Link to this image")
[![https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/sunsky_03_1_10.png](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/sunsky_03_1_10.png) ](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/sunsky_03_1_10.png)
albedo=20% green[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#id9 "Link to this image")
[![https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/sunsky_03_10_10.png](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/sunsky_03_10_10.png) ](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/sunsky_03_10_10.png)
albedo=1[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#id10 "Link to this image")
**Time and Location (sky only)**
[![https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_sunsky_time_docs.svg](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_sunsky_time_docs.svg) ](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_sunsky_time_docs.svg)
**Turbidity (sky only)**
[![https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_sunsky_turb_docs.svg](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_sunsky_turb_docs.svg) ](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_sunsky_turb_docs.svg)
**Sun and sky scale**
[![https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_sky.jpg](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_sky.jpg) ](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_sky.jpg)
Sky only sun_scale=0[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#id11 "Link to this image")
[![https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_sun.jpg](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_sun.jpg) ](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_sun.jpg)
Sun only sky_scale=0[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#id12 "Link to this image")
[![https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_sunsky.jpg](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_sunsky.jpg) ](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_sunsky.jpg)
Sun and sky combined (default parameters)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#id13 "Link to this image")
Warning
  * Note that attaching a `sunsky` emitter to the scene introduces physical units into the rendering process of Mitsuba 3, which is ordinarily a unitless system. Specifically, the evaluated spectral radiance has units of power (\\(W\\)) per unit area (\\(m^{-2}\\)) per steradian (\\(sr^{-1}\\)) per unit wavelength (\\(nm^{-1}\\)). As a consequence, your scene should be modeled in meters for this plugin to work properly.
  * The sun is an intense light source that subtends a tiny solid angle. This can be a problem for certain rendering techniques (e.g. path tracing), which produce high variance output (i.e. noise in renderings) when the scene also contains specular or glossy or materials.
  * Please be aware that given certain parameters, the sun’s radiance is ill-represented by the linear sRGB color space. Whether Mitsuba is rendering in spectral or RGB mode, if the final output is an sRGB image, it can happen that it contains negative pixel values or be over-saturated. These results are left un-clamped to let the user post-process the image to their liking, without losing information.


XMLPython
```
<emittertype="sunsky">
<floatname="hour"value="20.0"/>
</emitter>

```
Copy to clipboard
```
'type': 'sunsky',
'hour': 20.0

```
Copy to clipboard
## Spot light source (spot)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#spot-light-source-spot "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
intensity | spectrum | Specifies the maximum radiant intensity at the center in units of power per unit steradian. (Default: 1). This cannot be spatially varying (e.g. have bitmap as type). | P, ∂  
cutoff_angle | float | Cutoff angle, beyond which the spot light is completely black (Default: 20 degrees) |   
beam_width | float | Subtended angle of the central beam portion (Default: \\(cutoff_angle \times 3/4\\)) |   
texture | texture | An optional texture to be projected along the spot light. This must be spatially varying (e.g. have bitmap as type). | P, ∂  
to_world | transform | Specifies an optional emitter-to-world transformation. (Default: none, i.e. emitter space = world space) | P  
This plugin provides a spot light with a linear falloff. In its local coordinate system, the spot light is positioned at the origin and points along the positive Z direction. It can be conveniently reoriented using the lookat tag, e.g.:
XMLPython
```
<emittertype="spot">
<transformname="to_world">
<!-- Orient the light so that points from (1, 1, 1) towards (1, 2, 1) -->
<lookatorigin="1, 1, 1"target="1, 2, 1"up="0, 0, 1"/>
</transform>
<rgbname="intensity"value="1.0"/>
</emitter>

```
Copy to clipboard
```
'type': 'spot',
'to_world': mi.ScalarTransform4f().look_at(
    origin=[1, 1, 1],
    target=[1, 2, 1],
    up=[0, 0, 1]
),
'intensity': {
    'type': 'spectrum',
    'value': 1.0,
}

```
Copy to clipboard
The intensity linearly ramps up from cutoff_angle to beam_width (both specified in degrees), after which it remains at the maximum value. A projection texture may optionally be supplied.
[![../../_images/emitter_spot_no_texture.jpg](https://mitsuba.readthedocs.io/en/stable/_images/emitter_spot_no_texture.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/emitter_spot_no_texture.jpg)
Two spot lights with different colors and no texture specified.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#id14 "Link to this image")
[![../../_images/emitter_spot_texture.jpg](https://mitsuba.readthedocs.io/en/stable/_images/emitter_spot_texture.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/emitter_spot_texture.jpg)
A spot light with a texture specified.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#id15 "Link to this image")
## Directional area light (directionalarea)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#directional-area-light-directionalarea "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
radiance | spectrum | Specifies the emitted radiance in units of power per unit area per unit steradian. | P, ∂  
Similar to an area light, but emitting only in the normal direction.
Note
This can only be rendered correctly with a particle tracer, since rays traced from the camera and surfaces have zero probability of connecting with this emitter at exactly the correct angle.
XMLPython
```
<shapetype="sphere">
<emittertype="directionalarea">
<rgbname="radiance"value="1.0"/>
</emitter>
</shape>

```
Copy to clipboard
```
'type': 'sphere',
'emitter': {
    'type': 'directionalarea',
    'radiance': {
        'type': 'rgb',
        'value': 1.0,
    }
}

```
Copy to clipboard
## Distant directional emitter (directional)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#distant-directional-emitter-directional "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
irradiance | spectrum | Spectral irradiance, which corresponds to the amount of spectral power per unit area received by a hypothetical surface normal to the specified direction. | P, ∂  
to_world | transform | Emitter-to-world transformation matrix. | P  
direction | vector | Alternative (and exclusive) to `to_world`. Direction towards which the emitter is radiating in world coordinates. |   
This emitter plugin implements a distant directional source which radiates a specified power per unit area along a fixed direction. By default, the emitter radiates in the direction of the positive Z axis, i.e. \\((0, 0, 1)\\).
XMLPython
```
<emittertype="directional">
<vectorname="direction"value="1.0, 0.0, 0.0"/>
<rgbname="irradiance"value="1.0"/>
</emitter>

```
Copy to clipboard
```
'type': 'directional',
'direction': [1.0, 0.0, 0.0],
'irradiance': {
    'type': 'rgb',
    'value': 1.0,
}

```
Copy to clipboard
## Projection light source (projector)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#projection-light-source-projector "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
irradiance | texture | 2D texture specifying irradiance on the emitter’s virtual image plane, which lies at a distance of \\(z=1\\) from the pinhole. Note that this does not directly correspond to emitted radiance due to the presence of an additional directionally varying scale factor equal to the inverse sensitivity profile (a.k.a. importance) of a perspective camera. This ensures that a projection of a constant texture onto a plane is truly constant. | P, ∂  
scale | float | A scale factor that is applied to the radiance values stored in the above parameter. (Default: 1.0) | P, ∂  
to_world | transform | Specifies an optional camera-to-world transformation. (Default: none (i.e. camera space = world space)) | P  
fov | float | Denotes the camera’s field of view in degrees—must be between 0 and 180, excluding the extremes. Alternatively, it is also possible to specify a field of view using the focal_length parameter. |   
focal_length | string | Denotes the camera’s focal length specified using _35mm_ film equivalent units. Alternatively, it is also possible to specify a field of view using the fov parameter. See the main description for further details. (Default: 50mm) |   
fov_axis | string |  When the parameter fov is given (and only then), this parameter further specifies the image axis, to which it applies.
  1. x: fov maps to the x-axis in screen space.
  2. y: fov maps to the y-axis in screen space.
  3. diagonal: fov maps to the screen diagonal.
  4. smaller: fov maps to the smaller dimension (e.g. x when width < height)
  5. larger: fov maps to the larger dimension (e.g. y when width < height)

The default is x. |   
This emitter is the reciprocal counterpart of the perspective camera implemented by the [perspective](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_sensors.html#sensor-perspective) plugin. It accepts exactly the same parameters and employs the same pixel-to-direction mapping. In contrast to the perspective camera, it takes an extra texture (typically of type [bitmap](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html#texture-bitmap)) as input that it then projects into the scene, with an optional scaling factor.
Pixels are importance sampled according to their density, hence this operation remains efficient even if only a single pixel is turned on.
[![../../_images/emitter_projector_constant.jpg](https://mitsuba.readthedocs.io/en/stable/_images/emitter_projector_constant.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/emitter_projector_constant.jpg)
A projector lights with constant irradiance (no texture specified).[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#id16 "Link to this image")
[![../../_images/emitter_projector_textured.jpg](https://mitsuba.readthedocs.io/en/stable/_images/emitter_projector_textured.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/emitter_projector_textured.jpg)
A projector light with a texture specified.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#id17 "Link to this image")
XMLPython
```
<emittertype="projector">
<rgbname="irradiance"value="1.0"/>
<floatname="fov"value="45"/>
<transformname="to_world">
<lookatorigin="1, 1, 1"
target="1, 2, 1"
up="0, 0, 1"/>
</transform>
</emitter>

```
Copy to clipboard
```
'type': 'projector',
'irradiance': {
    'type': 'rgb',
    'value': 1.0,
},
'fov': 45,
'to_world': mi.ScalarAffineTransform4f().look_at(
    origin=[1, 1, 1],
    target=[1, 2, 1],
    up=[0, 0, 1]
)

```
Copy to clipboard
## Timed sun and sky emitter (timed_sunsky)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html#timed-sun-and-sky-emitter-timed-sunsky "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
turbidity | float | Atmosphere turbidity, must be within [1, 10] (Default: 3, clear sky in a temperate climate). Smaller turbidity values (∼ 1 − 2) produce an arctic-like clear blue sky, whereas larger values (∼ 8 − 10) create an atmosphere that is more typical of a warm, humid day. | P  
albedo | spectrum | Ground albedo, must be within [0, 1] for each wavelength/channel, (Default: 0.3). This cannot be spatially varying (e.g. have bitmap as type). | P  
latitude | float | Latitude of the location in degrees (Default: 35.689, Tokyo’s latitude). | P  
longitude | float | Longitude of the location in degrees (Default: 139.6917, Tokyo’s longitude). | P  
timezone | float | Timezone of the location in hours (Default: 9). | P  
window_start_time | float | Start hour for the daily average (Default: 7). | P  
window_end_time | float | Final hour for the daily average (Default: 19). | P  
start_year | integer | Year of the start of the average (Default: 2025). | P  
start_month | integer | Month of the start of the average (Default: 01). | P  
start_day | integer | Day of the start of the average (Default: 01). | P  
end_year | integer | Year of the end of the average (Default: start_year + 1). | P  
end_month | integer | Month of the end of the average (Default: start_month). | P  
end_day | integer | Day of the end of the average (Default: start_day). | P  
sun_scale | float | Scale factor for the sun radiance (Default: 1). Can be used to turn the sun off (by setting it to 0). | P  
sky_scale | float | Scale factor for the sky radiance (Default: 1). Can be used to turn the sky off (by setting it to 0). | P  
sun_aperture | float | Aperture angle of the sun in degrees (Default: 0.5338, normal sun aperture). |   
shutter_open | float | Shutter opening time (Default: 0). Used to vary sunsky appearance |   
shutter_close | float | Shutter closing time (Default: 1). Used to vary sunsky appearance |   
to_world | transform | Specifies an optional emitter-to-world transformation. (Default: none, i.e. emitter space = world space) | P  
This emitter represents a sun and sky environment emitter for a dynamic time interval (where time is passed as attribute of the various query records). It is particularly useful for applications like architectural visualization or horticultural studies, where the goal is to simulate the lighting conditions over multiple days, months, years, or even longer, rather than the lighting at a specific instant. If the goal is to render using the sunsky background emitter at a fixed point in time, please take a look at the sunsky that is optimised and more efficient for that.
The local reference frame of this emitter is Z-up and X being towards the north direction. This behaviour can be changed with the `to_world` parameter.
The plugin works by dynamically computing the Hosek-Wilkie sun [[HW13](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id55 "Lukáš Hošek and Alexander Wilkie. Adding a solar-radiance function to the hošek-wilkie skylight model. IEEE Computer Graphics and Applications, 33\(3\):44-52, 2013. doi:10.1109/MCG.2013.18.")] and sky model [[HW12](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id56 "Lukas Hosek and Alexander Wilkie. An analytic model for full spectral sky-dome radiance. ACM Trans. Graph., 2012. doi:10.1145/2185520.2185591.")] for the given time and direction of the ray/sample. The time parameter is controlled by the `shutter_open` and `shutter_close` parameters that should thus be the same as the sensor’s.
**Render with default settings and HDR film yielding an average over a year:**
[![https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_timed_sunsky.jpg](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_timed_sunsky.jpg) ](https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2025/09/15/sunsky/emitter_timed_sunsky.jpg)
XMLPython
```
<emittertype="timed_sunsky">
<integername="start_year"value="2026"/>
</emitter>

```
Copy to clipboard
```
'type': 'timed_sunsky',
'start_year': 2026

```
Copy to clipboard
Warning
  * Note that attaching a `timed_sunsky` emitter to the scene introduces physical units into the rendering process of Mitsuba 3, which is ordinarily a unitless system. Specifically, the evaluated spectral radiance has units of power (\\(W\\)) per unit area (\\(m^{-2}\\)) per steradian (\\(sr^{-1}\\)) per unit wavelength (\\(nm^{-1}\\)). As a consequence, your scene should be modeled in meters for this plugin to work properly.
  * The sun is an intense light source that subtends a tiny solid angle. This can be a problem for certain rendering techniques (e.g. path tracing), which produce high variance output (i.e. noise in renderings) when the scene also contains specular or glossy or materials.
  * Please be aware that given certain parameters, the sun’s radiance is ill-represented by the linear sRGB color space. Whether Mitsuba is rendering in spectral or RGB mode, if the final output is an sRGB image, it can happen that it contains negative pixel values or be over-saturated. These results are left un-clamped to let the user post-process the image to their liking, without losing information.
  * Note that this emitter is dependent on a valid sensor shutter open and close time. The sensor’s defaults being 0 and 0 respectively, this emitter will not see the time vary. Please set a valid shutter open and close time and pass the same time parameters to this plugin.


* * *
  *[P]: This parameter will be exposed as a scene parameter
  *[∂]: This parameter is differentiable
  *[D]: This parameter might introduce discontinuities. Therefore it requires special handling during differentiation to prevent bias)
