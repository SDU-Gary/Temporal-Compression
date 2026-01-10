---
url: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_media.html
crawled_at: 2025-11-13T17:13:47.685142
title: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_media.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_media.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_media.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Participating media[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_media.html#participating-media "Link to this heading")
In Mitsuba, participating media are used to simulate materials ranging from fog, smoke, and clouds, over translucent materials such as skin or milk, to “fuzzy” structured substances such as woven or knitted cloth. This section describes the two available types of media (homogeneous and heterogeneous). In practice, these will be combined with a [phase function](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_phase.html#sec-phasefunctions).
Participating media are usually attached to shapes in the scene. When a shape marks the transition to a participating medium, it is necessary to provide information about the two media that lie at the interior and exterior of the shape. This informs the renderer about what happens in the region of space surrounding the surface. In many practical use cases it is sufficient to only specify an interior medium and to assume the exterior medium (e.g., air) to not influence the light transport.
XMLPython
```
<sceneversion="3.0.0">
<shapetype=".. shape type ..">
<mediumname="interior"type="... medium type ...">
</medium>
<mediumname="exterior"type="... medium type ...">
</medium>
<!-- Alternatively: reference named media that
            have been declared previously
            <ref name="interior" id="myMedium1"/>
            <ref name="exterior" id="myMedium2"/>
        -->
</shape>
</scene>

```
Copy to clipboard
```
'type': 'scene',
'shape_id': {
    'type': '<shape_type>',
    # .. shape parameters ..

    'interior': {
        'type': '<medium_type>',
        # .. medium parameters ..
    },
    'exterior': {
        'type': '<medium_type>',
        # .. medium parameters ..
    }
}

```
Copy to clipboard
When a medium permeates a volume of space (e.g. fog) that includes sensors, it is important to assign the medium to them. This can be done using the referencing mechanism:
XMLPython
```
<sceneversion="3.0.0">
<!-- .. scene contents .. -->

<mediumtype="homogeneous"id="fog">
<!-- .. homogeneous medium parameters .. -->
</medium>
<sensortype="perspective">
<!-- .. perspective camera parameters .. -->
<!-- Reference the fog medium from within the sensor declaration
            to make it aware that it is embedded inside this medium -->
<refid="fog"/>
</sensor>
</scene>

```
Copy to clipboard
```
'type': 'scene',

# .. scene contents ..

'fog': {
    'type': 'homogeneous',
    # .. homogeneous medium parameters ..
},

'sensor_id': {
    'type': 'perspective',
    # .. perspective camera parameters ..

    # Reference the fog medium
    'medium' : {
        'type' : 'ref',
        'id' : 'fog'
    }
}

```
Copy to clipboard
## Homogeneous medium (homogeneous)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_media.html#homogeneous-medium-homogeneous "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
albedo | float, spectrum or volume | Single-scattering albedo of the medium (Default: 0.75). | P, ∂  
sigma_t | float or spectrum | Extinction coefficient in inverse scene units (Default: 1). | P, ∂  
scale | float | Optional scale factor that will be applied to the extinction parameter. It is provided for convenience when accommodating data based on different units, or to simply tweak the density of the medium. (Default: 1) | P  
sample_emitters | boolean | Flag to specify whether shadow rays should be cast from inside the volume (Default: true) If the medium is enclosed in a [dielectric](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-dielectric) boundary, shadow rays are ineffective and turning them off will significantly reduce render time. This can reduce render time up to 50% when rendering objects with subsurface scattering. |   
(Nested plugin) | phase | A nested phase function that describes the directional scattering properties of the medium. When none is specified, the renderer will automatically use an instance of isotropic. | P, ∂  
This class implements a homogeneous participating medium with support for arbitrary phase functions. This medium can be used to model effects such as fog or subsurface scattering.
The medium is parametrized by the single scattering albedo and the extinction coefficient \\(\sigma_t\\). The extinction coefficient should be provided in inverse scene units. For instance, when a world-space distance of 1 unit corresponds to a meter, the extinction coefficient should have units of inverse meters. For convenience, the scale parameter can be used to correct the units. For instance, when the scene is in meters and the coefficients are in inverse millimeters, set scale to 1000.
[![../../_images/medium_homogeneous_sss.jpg](https://mitsuba.readthedocs.io/en/stable/_images/medium_homogeneous_sss.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/medium_homogeneous_sss.jpg)
Homogeneous medium with constant albedo[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_media.html#id1 "Link to this image")
[![../../_images/medium_homogeneous_sss_textured.jpg](https://mitsuba.readthedocs.io/en/stable/_images/medium_homogeneous_sss_textured.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/medium_homogeneous_sss_textured.jpg)
Homogeneous medium with spatially varying albedo[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_media.html#id2 "Link to this image")
The homogeneous medium assumes the extinction coefficient to be constant throughout the medium. However, it supports the use of a spatially varying albedo.
XMLPython
```
<mediumid="myMedium"type="homogeneous">
<rgbname="albedo"value="0.99, 0.9, 0.96"/>
<floatname="sigma_t"value="5"/>

<!-- The extinction is also allowed to be spectrally varying
         Since RGB values have to be in the [0, 1]
        <rgb name="sigma_t" value="0.5, 0.25, 0.8"/>
    -->

<!-- A homogeneous medium needs to have a constant extinction,
        but can have a spatially varying albedo:

        <volume name="albedo" type="gridvolume">
            <string name="filename" value="albedo.vol"/>
        </volume>
    -->

<phasetype="hg">
<floatname="g"value="0.7"/>
</phase>
</medium>

```
Copy to clipboard
```
'type': 'homogeneous',
'albedo': {
    'type': 'rgb',
    'value': [0.99, 0.9, 0.96]
},
'sigma_t': 5,
# The extinction is also allowed to be spectrally varying
# since RGB values have to be in the [0, 1]
# 'sigma_t': {
#     'value': [0.5, 0.25, 0.8]
# }

# A homogeneous medium needs to have a constant extinction,
# but can have a spatially varying albedo:
# 'albedo': {
#     'type': 'gridvolume',
#     'filename': 'albedo.vol'
# }

'phase': {
    'type': 'hg',
    'g': 0.7
}

```
Copy to clipboard
## Heterogeneous medium (heterogeneous)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_media.html#heterogeneous-medium-heterogeneous "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
albedo | float, spectrum or volume | Single-scattering albedo of the medium (Default: 0.75). | P, ∂  
sigma_t | float, spectrum or volume | Extinction coefficient in inverse scene units (Default: 1). | P, ∂  
scale | float | Optional scale factor that will be applied to the extinction parameter. It is provided for convenience when accommodating data based on different units, or to simply tweak the density of the medium. (Default: 1) | P  
sample_emitters | boolean | Flag to specify whether shadow rays should be cast from inside the volume (Default: true) If the medium is enclosed in a [dielectric](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-dielectric) boundary, shadow rays are ineffective and turning them off will significantly reduce render time. This can reduce render time up to 50% when rendering objects with subsurface scattering. |   
(Nested plugin) | phase | A nested phase function that describes the directional scattering properties of the medium. When none is specified, the renderer will automatically use an instance of isotropic. | P, ∂  
This plugin provides a flexible heterogeneous medium implementation, which acquires its data from nested volume instances. These can be constant, use a procedural function, or fetch data from disk, e.g. using a 3D grid.
The medium is parametrized by the single scattering albedo and the extinction coefficient \\(\sigma_t\\). The extinction coefficient should be provided in inverse scene units. For instance, when a world-space distance of 1 unit corresponds to a meter, the extinction coefficient should have units of inverse meters. For convenience, the scale parameter can be used to correct the units. For instance, when the scene is in meters and the coefficients are in inverse millimeters, set scale to 1000.
Both the albedo and the extinction coefficient can either be constant or textured, and both parameters are allowed to be spectrally varying.
XMLPython
```
<!-- Declare a heterogeneous participating medium named 'smoke' -->
<mediumtype="heterogeneous"id="smoke">
<!-- Acquire extinction values from an external data file -->
<volumename="sigma_t"type="gridvolume">
<stringname="filename"value="frame_0150.vol"/>
</volume>

<!-- The albedo is constant and set to 0.9 -->
<floatname="albedo"value="0.9"/>

<!-- Use an isotropic phase function -->
<phasetype="isotropic"/>

<!-- Scale the density values as desired -->
<floatname="scale"value="200"/>
</medium>

<!-- Attach the index-matched medium to a shape in the scene -->
<shapetype="obj">
<!-- Load an OBJ file, which contains a mesh version
         of the axis-aligned box of the volume data file -->
<stringname="filename"value="bounds.obj"/>

<!-- Reference the medium by ID -->
<refname="interior"id="smoke"/>
<!-- If desired, this shape could also declare
        a BSDF to create an index-mismatched
        transition, e.g.
        <bsdf type="dielectric"/>
    -->
</shape>

```
Copy to clipboard
```
# Declare a heterogeneous participating medium named 'smoke'
'smoke': {
    'type': 'heterogeneous',

    # Acquire extinction values from an external data file
    'sigma_t': {
        'type': 'gridvolume',
        'filename': 'frame_0150.vol'
    },

    # The albedo is constant and set to 0.9
    'albedo': 0.9,

    # Use an isotropic phase function
    'phase': {
        'type': 'isotropic'
    },

    # Scale the density values as desired
    'scale': 200
},

# Attach the index-matched medium to a shape in the scene
'shape': {
    'type': 'obj',
    # Load an OBJ file, which contains a mesh version
    # of the axis-aligned box of the volume data file
    'filename': 'bounds.obj',

    # Reference the medium by ID
    'interior': 'smoke',
    # If desired, this shape could also declare
    # a BSDF to create an index-mismatched
    # transition, e.g.
    # 'bsdf': {
    #     'type': 'isotropic'
    # },
}

```
Copy to clipboard
* * *
  *[P]: This parameter will be exposed as a scene parameter
  *[∂]: This parameter is differentiable
