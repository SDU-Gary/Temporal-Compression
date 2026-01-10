---
url: https://mitsuba.readthedocs.io/en/stable/src/rendering/editing_a_scene.html
crawled_at: 2025-11-13T17:11:25.599743
title: https://mitsuba.readthedocs.io/en/stable/src/rendering/editing_a_scene.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/rendering/editing_a_scene.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/rendering/editing_a_scene.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Download notebook ](https://raw.githubusercontent.com/mitsuba-renderer/mitsuba-tutorials/master/rendering/editing_a_scene.ipynb) [ Download data ](https://d38rqfq1h7iukm.cloudfront.net/scenes/tutorials/scenes.zip)
# Editing a scene[¶](https://mitsuba.readthedocs.io/en/stable/src/rendering/editing_a_scene.html#Editing-a-scene "Link to this heading")
## Overview[¶](https://mitsuba.readthedocs.io/en/stable/src/rendering/editing_a_scene.html#Overview "Link to this heading")
In this tutorial, you will learn how to modify a Mitsuba scene after it has been loaded from a file. You might want to edit a scene before (re-)rendering it for many reasons. Maybe a corner is dim, or an object should be moved a bit to the left. Thankfully we can use the _traverse_ mechanism to perform such modifications in Python with Mitsuba 3. As we will see in later tutorials, this mechanism is also essential for inverse rendering applications and more.
🚀 **You will learn how to:**
  * List exposed parameters of Mitsuba objects
  * Edit a scene and update its internal state accordingly


## Loading a scene[¶](https://mitsuba.readthedocs.io/en/stable/src/rendering/editing_a_scene.html#Loading-a-scene "Link to this heading")
Following the same steps as in [Mitsuba quickstart tutorial](https://mitsuba.readthedocs.io/en/latest/src/quickstart/mitsuba_quickstart.html), let’s import `mitsuba`, set the desired variant and load a scene from an XML file on disk.
Copy to clipboard
```
importdrjitasdr
importmitsubaasmi
mi.set_variant('llvm_ad_rgb')

scene = mi.load_file("../scenes/simple.xml")

```
Copy to clipboard
Let’s quickly render this scene.
Copy to clipboard
```
original_image = mi.render(scene, spp=128)

importmatplotlib.pyplotasplt
plt.axis('off')
plt.imshow(original_image ** (1.0 / 2.2));

```
Copy to clipboard
![../../_images/src_rendering_editing_a_scene_5_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_rendering_editing_a_scene_5_0.png)
## Accessing scene parameters[¶](https://mitsuba.readthedocs.io/en/stable/src/rendering/editing_a_scene.html#Accessing-scene-parameters "Link to this heading")
Any Mitsuba object can be inspected using the [traverse()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.traverse) function, which returns a instance of [SceneParameters](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.SceneParameters). It has a similar API to Python `dict` and holds all parameters that are exposed by the input object and its children. Therefore, when given a scene as input, this function will return the parameters of all the objects present in the scene.
Let’s print the paramters available in our teapot scene.
Copy to clipboard
```
params = mi.traverse(scene)
print(params)

```
Copy to clipboard
```
SceneParameters[
  ----------------------------------------------------------------------------------------
  Name                                 Flags    Type           Parent
  ----------------------------------------------------------------------------------------
  sensor.near_clip                              float          PerspectiveCamera
  sensor.far_clip                               float          PerspectiveCamera
  sensor.shutter_open                           float          PerspectiveCamera
  sensor.shutter_open_time                      float          PerspectiveCamera
  sensor.film.size                              ScalarVector2u HDRFilm
  sensor.film.crop_size                         ScalarVector2u HDRFilm
  sensor.film.crop_offset                       ScalarPoint2u  HDRFilm
  sensor.x_fov                         ∂, D     Float          PerspectiveCamera
  sensor.principal_point_offset_x      ∂, D     Float          PerspectiveCamera
  sensor.principal_point_offset_y      ∂, D     Float          PerspectiveCamera
  sensor.to_world                      ∂, D     Transform4f    PerspectiveCamera
  teapot.bsdf.reflectance.value        ∂        Color3f        SRGBReflectanceSpectrum
  teapot.silhouette_sampling_weight             float          PLYMesh
  teapot.faces                                  UInt           PLYMesh
  teapot.vertex_positions              ∂, D     Float          PLYMesh
  teapot.vertex_normals                ∂, D     Float          PLYMesh
  teapot.vertex_texcoords              ∂        Float          PLYMesh
  light1.sampling_weight                        float          PointLight
  light1.position                               Point3f        PointLight
  light1.intensity.value               ∂        Color3f        SRGBReflectanceSpectrum
  light2.sampling_weight                        float          PointLight
  light2.position                               Point3f        PointLight
  light2.intensity.value               ∂        Color3f        SRGBReflectanceSpectrum
]

```
Copy to clipboard
As you can see, the first level of our scene graph has 4 objects:
  * the camera (`sensor`)
  * the teapot mesh (`teapot`)
  * two light sources (`light1` and `light2`).


Some of those objects have nested child objects, like `teapot.bsdf`.
Names like _teapot_ are defined in the `id` field in the XML file. Parameters such as `teapot.vertex_positions` or `sensor.far_clip` are documented in their respective parent’s plugin documentation (see [PLYMesh](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_shapes.html#ply-stanford-triangle-format-mesh-loader-ply) and [PerspectiveCamera](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_sensors.html#perspective-pinhole-camera-perspective)). The plugin documentation lists which parameters are exposed, as well as all input parameters it takes in the XML file.
If you wish to modifiy a plugin’s parameter that is not exposed with `traverse`, you still have the option to modify the XML file directly. `traverse` is merely a convenience function to edit scene objects in-place.
Individual scene parameters can be accessed with the `__getitem__` operator, providing the `key` corresponding to the parameter. Let’s print some scene parameter values.
Copy to clipboard
```
print('sensor.near_clip:             ',  params['sensor.near_clip'])
print('teapot.bsdf.reflectance.value:',  params['teapot.bsdf.reflectance.value'])
print('light1.intensity.value:       ',  params['light1.intensity.value'])

```
Copy to clipboard
```
sensor.near_clip:              0.009999999776482582
teapot.bsdf.reflectance.value: [[0.9, 0.9, 0]]
light1.intensity.value:        [[100, 100, 100]]

```
Copy to clipboard
## Edit the scene[¶](https://mitsuba.readthedocs.io/en/stable/src/rendering/editing_a_scene.html#Edit-the-scene "Link to this heading")
Similarly to a Python `dict`, parameters can be modified in-place using the `__setitem__` operator. However, it is necessary to call the `SceneParameters.update` method to properly apply the desired changes.
Some objects need to be notified if the children have been updated. For instance, a change to the vertex position buffer of a mesh will trigger the recomputation of the Embree/Optix BHV.
Internally, the `SceneParameters` object will record every update written to it. Using `SceneParameters.update` will propagate all updates through the dependency graph, and perform all necessary updates to the parent objects.
Copy to clipboard
```
# Give a red tint to light1 and a green tint to light2
params['light1.intensity.value'] *= [1.5, 0.2, 0.2]
params['light2.intensity.value'] *= [0.2, 1.5, 0.2]

# Apply updates
params.update();

```
Copy to clipboard
Mesh editing is also possible but requires specifying the layout of the stored data. See [transformation toolbox](https://mitsuba.readthedocs.io/en/latest/src/how_to_guides/mesh_io_and_manipulation.html) and [mesh manipulation](https://mitsuba.readthedocs.io/en/latest/src/how_to_guides/image_io_and_manipulation.html) for more geometry and mesh operations.
Copy to clipboard
```
# Translate the teapot a little bit
V = dr.unravel(mi.Point3f, params['teapot.vertex_positions'])
V.z += 0.5
params['teapot.vertex_positions'] = dr.ravel(V)

# Apply changes
params.update();

```
Copy to clipboard
After rendering the scene again, we can easily compare the rendered images using `matplotlib`.
Copy to clipboard
```
modified_image = mi.render(scene, spp=128)
fig = plt.figure(figsize=(10, 10))
fig.add_subplot(1,2,1).imshow(original_image); plt.axis('off'); plt.title('original')
fig.add_subplot(1,2,2).imshow(modified_image); plt.axis('off'); plt.title('modified');

```
Copy to clipboard
![../../_images/src_rendering_editing_a_scene_16_0.png](https://mitsuba.readthedocs.io/en/stable/_images/src_rendering_editing_a_scene_16_0.png)
## See also[¶](https://mitsuba.readthedocs.io/en/stable/src/rendering/editing_a_scene.html#See-also "Link to this heading")
  * [mitsuba.traverse()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.traverse)
  * [mitsuba.SceneParameters](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html##mitsuba.SceneParameters)


* * *
