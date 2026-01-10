---
url: https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html
crawled_at: 2025-11-13T17:16:40.113048
title: https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Download notebook ](https://raw.githubusercontent.com/mitsuba-renderer/mitsuba-tutorials/master/how_to_guides/transformation_toolbox.ipynb) [ Download data ](https://d38rqfq1h7iukm.cloudfront.net/scenes/tutorials/scenes.zip)
# Transformation toolbox[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#Transformation-toolbox "Link to this heading")
This how-to guide explores the different tools available in Mitsuba 3 to manipulate cartesian coordinate systems. When generating datasets, researching advanced light transport algorithms, or developing new appearance models, you will quickly realize how essential those tools are, so we strongly recommend all users to go through this guide.
Copy to clipboard
```
importmitsubaasmi

mi.set_variant("scalar_rgb")

```
Copy to clipboard
## Frame[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#Frame "Link to this heading")
The [Frame3f](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.Frame3f) class stores a three-dimensional orthonormal coordinate frame. This class is very handy when you wish to convert vectors between different cartesian coordinates systems.
### Frame initialization[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#Frame-initialization "Link to this heading")
A `Frame3f` can be initialized in different ways as shown below. When given a single vector, it will make use of [coordinate_system()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.coordinate_system) to compute the other two basis vectors.
Copy to clipboard
```
mi.Frame3f()  # Empty frame

mi.Frame3f(
    [1, 0, 0],  # s
    [0, 1, 0],  # t
    [0, 0, 1],  # n
)

mi.Frame3f([0, 1, 0])  # n

```
Copy to clipboard
Copy to clipboard
```
Frame[
  s = [1, -0, -0],
  t = [-0, 0, -1],
  n = [0, 1, 0]
]

```
Copy to clipboard
### Converting to/from local frames[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#Converting-to/from-local-frames "Link to this heading")
The two methods below are the main operations you will be using to convert between different coordinate frames.
  * [Frame3f.to_local()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.Frame3f.to_local)
  * [Frame3f.to_world()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.Frame3f.to_world)


Copy to clipboard
```
frame = mi.Frame3f(
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
)

world_vector = mi.Vector3f([3, 2, 1])  # In world frame
local_vector = frame.to_local(world_vector)
local_vector

```
Copy to clipboard
Copy to clipboard
```
[1, 2, 3]

```
Copy to clipboard
### Spherical coordinates[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#Spherical-coordinates "Link to this heading")
Mitsuba 3 provides convenience methods to efficiently compute certain trigonometric evaluations of spherical coordinates with respect to a `Frame3f`. We use the naming convention that _theta_ is the elevation and _phi_ is the azimuth. For example, you can call `Frame3f.sin_theta_2()` or `Frame3f.cos_phi()`. As always, the full list of methods is availble in the [reference API](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.Frame3f).
## Transform[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#Transform "Link to this heading")
The `Transform4f` and `Transform3f` classes provides several static functions to create common transformations, such as `translate`, `scale`, `rotate` and `look_at`. These are often used for setting `"to_world"` object parameters in Python using `load_dict()`. As we will see later, those transformations can also be applied to a `Vector`, `Point`, `Normal` and even a `Ray3f`.
Note that all transforms are in homogenous coordiantes. `Transform4f` can therefore be applied to 3-dimensional objects and `Transform3f` to 2-dimensional objects.
The `Transform4f` and `Transform3f` objects hold both the transformation matrix and its transpose of inverse. For convenience, there is also a `Transform4f.inverse()` method. All put together, this makes transforming back and forth straightforward.
🗒 **Note**
Often when working with a vectorized variant of Mitsuba (e.g. `llvm_ad_rgb`), we still want to work with scalar transformation. For instance in the context of scene loading when setting `to_world` transformations. Mitsuba data-structure types such `Transform4f` can be prefixed with `Scalar` to indicate that no matter which variant of Mitsuba is enabled, this type should always refer to the CPU scalar version (which can also be accessed with `mitsuba.scalar_rgb.Transform4f`. The same applies to all basic types (e.g. `Float`, `UInt32`) and other data-structure types (e.g. `Ray3f`, `SurfaceInteraction3f`) which can all be prefixed with `Scalar`.
### Transform initialization[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#Transform-initialization "Link to this heading")
There are several ways to instanciate a transformation object. For example one can create a `Transform4f` from a `numpy` array directly, or a simple Python `list`.
Copy to clipboard
```
importnumpyasnp

# Default constructor is identity matrix
identity = mi.Transform4f()

np_mat = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
)
mi_mat = mi.Matrix3f(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
)
list_mat = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

# Build from different types
t_from_np = mi.Transform3f(np_mat)
t_from_mi = mi.Transform3f(mi_mat)
t_from_list = mi.Transform3f(list_mat)

# Broadcasting
t_from_value = mi.Transform3f(3)  # Scaled identity matrix
t_from_row = mi.Transform3f([3, 2, 3])  # Broadcast over matrix columns
t_from_row

```
Copy to clipboard
Copy to clipboard
```
Transform[
  matrix=[[3, 3, 3],
          [2, 2, 2],
          [3, 3, 3]],
  inverse_transpose=[[-nan, -nan, -nan],
                     [-nan, -nan, -nan],
                     [-nan, -nan, -nan]]
]

```
Copy to clipboard
We then have a few static function helpful to construct common transformations:
#### Translate[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#Translate "Link to this heading")
Copy to clipboard
```
mi.Transform4f().translate([10, 20, 30])

```
Copy to clipboard
Copy to clipboard
```
Transform[
  matrix=[[1, 0, 0, 10],
          [0, 1, 0, 20],
          [0, 0, 1, 30],
          [0, 0, 0, 1]],
  inverse_transpose=[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [-10, -20, -30, 1]]
]

```
Copy to clipboard
#### Scale[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#Scale "Link to this heading")
Copy to clipboard
```
mi.Transform4f().scale([10, 20, 30])

```
Copy to clipboard
Copy to clipboard
```
Transform[
  matrix=[[10, 0, 0, 0],
          [0, 20, 0, 0],
          [0, 0, 30, 0],
          [0, 0, 0, 1]],
  inverse_transpose=[[0.1, 0, 0, 0],
                     [0, 0.05, 0, 0],
                     [0, 0, 0.0333333, 0],
                     [0, 0, 0, 1]]
]

```
Copy to clipboard
#### Rotate[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#Rotate "Link to this heading")
Copy to clipboard
```
mi.Transform4f().rotate(axis=[0, 1, 0], angle=90)

```
Copy to clipboard
Copy to clipboard
```
Transform[
  matrix=[[-4.37114e-08, 0, 1, 0],
          [0, 1, 0, 0],
          [-1, 0, -4.37114e-08, 0],
          [0, 0, 0, 1]],
  inverse_transpose=[[-4.37114e-08, 0, 1, 0],
                     [0, 1, 0, 0],
                     [-1, 0, -4.37114e-08, 0],
                     [0, 0, 0, 1]]
]

```
Copy to clipboard
#### Look at[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#Look-at "Link to this heading")
Copy to clipboard
```
mi.Transform4f().look_at(origin=[0, 0, 2], target=[0, 0, 0], up=[0, 1, 0])

```
Copy to clipboard
Copy to clipboard
```
Transform[
  matrix=[[-1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, -1, 2],
          [0, 0, 0, 1]],
  inverse_transpose=[[-1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 2, 1]]
]

```
Copy to clipboard
#### Perspective[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#Perspective "Link to this heading")
The perspective projection does the following: - (1) Project camera space points onto z=1 plane, and non-linearly map z-coordinates from [near,far] to [0,1], - (2) Scale (x,y) such that the visible region specified by `fov` lies in [−1,1]×[−1,1].
_Note_ : Starting with Mitsuba 3.7, perspective transformations are handled using a separate variant of the transform class named `ProjectiveTransform4f`. This subsumes all behavior of the simplified `Transform4f` (which is now also available under the name `AffineTransform4f`) and additionally handles
Copy to clipboard
```
trafo = mi.ProjectiveTransform4f().perspective(fov=90, near=0.1, far=10)
print(trafo)
trafo @ mi.Point3f(2, -2, 2)

```
Copy to clipboard
```
Transform[
  matrix=[[1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1.0101, -0.10101],
          [0, 0, 1, 0]],
  inverse_transpose=[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, -9.9],
                     [0, 0, 1, 10]]
]

```
Copy to clipboard
Copy to clipboard
```
[1, -1, 0.959596]

```
Copy to clipboard
#### Orthographic[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#Orthographic "Link to this heading")
The orthographic projection maps the z-coordinate to [0,1].
Copy to clipboard
```
trafo = mi.Transform4f().orthographic(near=0.1, far=10)
print(trafo)
trafo @ mi.Point3f(1, 2, 3)

```
Copy to clipboard
```
Transform[
  matrix=[[1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 0.10101, -0.010101],
          [0, 0, 0, 1]],
  inverse_transpose=[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 9.9, 0],
                     [0, 0, 0.1, 1]]
]

```
Copy to clipboard
Copy to clipboard
```
[1, 2, 0.292929]

```
Copy to clipboard
### From/to frame[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#From/to-frame "Link to this heading")
⚠️ Only available for Transform4f
`mi.Transform4f().to_frame(frame)` is the matrix representation of the function `frame.to_local()`
Copy to clipboard
```
frame = mi.Frame3f(
    [0, 0, 1],
    [0, 2, 0],
    [3, 0, 0],
)
mi.Transform4f().to_frame(frame)

```
Copy to clipboard
Copy to clipboard
```
Transform[
  matrix=[[0, 0, 3, 0],
          [0, 2, 0, 0],
          [1, 0, 0, 0],
          [0, 0, 0, 1]],
  inverse_transpose=[[0, 0, 3, 0],
                     [0, 2, 0, 0],
                     [1, 0, 0, 0],
                     [0, 0, 0, 1]]
]

```
Copy to clipboard
### Applying transforms[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#Applying-transforms "Link to this heading")
The Python `@` (`__matmul__`) operator can be used to apply `Transform` objects to points, vectors, normals and rays or multiply transforms with other transforms. Depending on the operand’s type, the operation has a different effect.
  * `Vector3f`: A typical matrix multiplication ignoring the homogenous coordinates (e.g. translation)
  * `Point3f`: Adjusted matrix multiplication taking into account homogenous coordinates
  * `Normal3f`: Matrix multiplication using the inverse transpose to handle non-uniform scaling of surface normals
  * `Ray3f`: Both the ray origin (`mi.Point`) and the ray direction (`mi.Vector`) are transformed with the `@` operator
  * `Transform4f`: Combine both transformation.


Copy to clipboard
```
t = mi.Transform4f().translate([0, 1, 2])
t = t @ mi.Transform4f().scale([1, 2, 3])
v = mi.Vector3f([3, 4, 5])
p = mi.Point3f([3, 4, 5])
n = mi.Normal3f([1, 0, 0])

print(f"{t@v=}")
print(f"{t@p=}")
print(f"{t@n=}")

```
Copy to clipboard
```
t @ v=[3, 8, 15]
t @ p=[3, 9, 17]
t @ n=[1, 0, 0]

```
Copy to clipboard
#### Transformation order[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#Transformation-order "Link to this heading")
Transformations in Mitsuba are applied from right to left, similar to how such operations would be written in mathematical form. This means that when multiple transformations are chained together, the net transformation is equivalent to first performing the rightmost transformation, followed by the second rightmost transformation, and so on.
In the following example, the point will first be scaled and then transposed.
Copy to clipboard
```
S = mi.Transform4f().scale(2.0)
T = mi.Transform4f().translate([4, 0, 0])
v = mi.Point3f([1, 1, 1])

trasfo = T @ S

print(trasfo @ v)

```
Copy to clipboard
```
[6, 2, 2]

```
Copy to clipboard
#### Chaining transforms[¶](https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/transformation_toolbox.html#Chaining-transforms "Link to this heading")
For convinience, it is also possible to chain transformation intialization as follows:
Copy to clipboard
```
mi.Transform4f().scale(2.0).translate([1, 0, 0])

```
Copy to clipboard
Copy to clipboard
```
Transform[
  matrix=[[2, 0, 0, 2],
          [0, 2, 0, 0],
          [0, 0, 2, 0],
          [0, 0, 0, 1]],
  inverse_transpose=[[0.5, 0, 0, 0],
                     [0, 0.5, 0, 0],
                     [0, 0, 0.5, 0],
                     [-1, 0, 0, 1]]
]

```
Copy to clipboard
The code above is equivalent to:
Copy to clipboard
```
mi.Transform4f().scale(2.0) @ mi.Transform4f().translate([1, 0, 0])

```
Copy to clipboard
Copy to clipboard
```
Transform[
  matrix=[[2, 0, 0, 2],
          [0, 2, 0, 0],
          [0, 0, 2, 0],
          [0, 0, 0, 1]],
  inverse_transpose=[[0.5, 0, 0, 0],
                     [0, 0.5, 0, 0],
                     [0, 0, 0.5, 0],
                     [-1, 0, 0, 1]]
]

```
Copy to clipboard
* * *
