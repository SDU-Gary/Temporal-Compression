---
url: https://mitsuba.readthedocs.io/en/stable/src/developer_guide/variants_cpp.html
crawled_at: 2025-11-13T17:11:08.343510
title: https://mitsuba.readthedocs.io/en/stable/src/developer_guide/variants_cpp.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/variants_cpp.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/variants_cpp.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Variants in C++[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/variants_cpp.html#variants-in-c "Link to this heading")
As described in the section on [choosing variants](https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html#sec-variants), Mitsuba 3 code can be compiled into different variants, which are parameterized by their computational backend and representation of color. To enable such retargeting from a single implementation, the system relies on C++ templates and metaprogramming. Indeed, most C++ classes and functions in Mitsuba 3 are templates with the following two type parameters:
  * `Float` and
  * `Spectrum`.


These two parameters exactly correspond to the previously mentioned computational backend and color representation. During compilation, Mitsuba’s build system reads the `mitsuba.conf` file and substitutes the types of selected variants into these template parameters. For example,
Copy to clipboard
causes an [explicit template instantiation](https://en.cppreference.com/w/cpp/language/class_template#Explicit_instantiation) with
```
Float=float;
Spectrum=Color<Float,3>;

```
Copy to clipboard
The resulting C++ symbols will be added to the shared libraries (`dist/libmitsuba.so`, `dist/plugin/*.so`, …). At runtime, the user-specified variant will determine the set of symbols to be used by the renderer.
## Type aliases[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/variants_cpp.html#type-aliases "Link to this heading")
Of course, `Float` and `Spectrum` are not the only types that are used in a renderer. It also access to integer arithmetic types, vectors, normals, points, rays, matrices, and so on. More complex data structures like intersection and sampling records are also commonly used.
It would be tedious to have to define these types in the context of a given variant. To facilitate this process, Mitsuba provides a helper macro to import suitable types that are inferred from the definition of `Float` and `Spectrum`.
For example, after evaluating this macro at the beginning of a templated function, we are then able to use other templated Mitsuba types (e.g. `Vector2f`, `Ray3f`, `SurfaceInteraction`, …) as if they were not templated.
```
template<typenameFloat,typenameSpectrum>
voidmy_function(){
/// Import type aliases (e.g. using Vector3f = Vector<Float, 3>;)
MI_IMPORT_TYPES()

// Can now use those types as if they were not templated
Point3fp(4.f,3.f,0.f);
Vector3fv(1.f,0.f,0.f);
Ray3fray(p,v);
std::cout<<ray<<std::endl;
}

```
Copy to clipboard
Note
The `MI_VARIANT` macro is often used as a shorthand notation instead of the somewhat verbose `template <typename Float, typename Spectrum>`.
Those macros are described in more detail in [this section](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html#sec-plugin-macros-cpp).
## Branching and masking[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/variants_cpp.html#branching-and-masking "Link to this heading")
When dealing with vectorized computational backends (e.g. `llvm_*`, `cuda_*`), additional scrutiny is needed to adapt C++ branching logic (in particular, `if` statements).
Consider the result of a ray intersection in scalar mode. The resulting [`SurfaceInteraction3f`](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.SurfaceInteraction3f "mitsuba.SurfaceInteraction3f") record holds information concerning a single surface intersection. In this case, conditional logic works fine using normal `if` statements.
On the other hand, the same data structure in a vectorized backend (e.g. `cuda_rgb`) holds information concerning _many_ surface intersections. Since any condition may only be true for a subset of the elements, conditionals logic can no longer be carried out using ordinary `if` statements.
The alternative operation `dr::select(mask, arg1, arg2)` takes a _mask_ argument (typically the result of a comparison) and evaluates `(mask ? arg1 : arg2)` in parallel for each lane. We refer to [Dr.Jit’s documentation](https://enoki.readthedocs.io/en/master/basics.html#working-with-masks) for further information on working with masks. The following shows an example contrasting these two cases:
```
// --------------------
// Scalar code (discouraged)

Scenescene=...;
Ray3fray=...;
SurfaceInteraction3fsi=scene->ray_intersect(ray);

if(si.is_valid())
return1.f;
else
return0.f;

// --------------------
// Generic code

Scenescene=...;
Ray3fray=...;
SurfaceInteraction3fsi=scene->ray_intersect(ray);

returndr::select(si.is_valid(),1.0f,0.f);

```
Copy to clipboard
Moreover, most of the functions/methods take an _optional_ `active` parameter that encodes which _lanes_ remain active. In the example above, we can e.g. provide this information to the `ray_intersect` routine to avoid computation (particularly, memory reads) associated with invalid entries. The updated code then reads:
```
// Mask specifying the active lanes
Maskactive=...;

Scenescene=...;
Ray3fray=...;
SurfaceInteraction3fsi=scene->ray_intersect(ray,active);

returndr::select(active&si.is_valid(),1.0f,0.f);

```
Copy to clipboard
## JIT backend synchronization point[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/variants_cpp.html#jit-backend-synchronization-point "Link to this heading")
As described in [Dr.Jit’s documentation](https://enoki.readthedocs.io/en/master/gpu.html#suggestions-regarding-horizontal-operations), the `cuda` and `llvm` computational backends rely on a JIT compiler that dynamically generates kernels using NVIDIA’s PTX intermediate language. This JIT compiler is highly efficient for _vertical_ operations (additions, multiplications, gathers, scatters, etc.). However, applying a _horizontal_ operations (e.g. `dr::any()`, `dr::all()`, `dr::hsum()`, etc.) to a `JITArray<T>` will flush all currently queued computations, which limits the amount of parallelism.
In many cases, horizontal mask-related operations can safely be skipped if this yields a performance benefit. For this reason, the Mitsuba 3 codebase makes frequent use of alternative reduction operations (`any_or<>()`, `all_or<>()`, …) that skip evaluation on GPU targets.
For example, the code `...` in the example below will only be executed if `condition` is `true` in `scalar_*` variants.
```
Maskcondition=...;
if(any_or<true>(condition)){
...
}

```
Copy to clipboard
In `cuda` and `llvm` variants, we are typically working with arrays containing millions of elements, and it is quite likely that at least of one of the array entries will in any case trigger execution of the `...`. The `any_or<true>(condition)` then skips the costly horizontal reduction and always assumes the condition to be true.
## Pointer types[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/variants_cpp.html#pointer-types "Link to this heading")
The `MI_IMPORT_TYPES` macro also imports variant-specific type aliases for pointer types. This is important: for example, consider the `BSDF` associated with a surface intersection. In a scalar variant , this is nicely represented using a `const BSDF *` pointer. However, on a vectorized variants (`cuda_*`, `llvm_*`), the intersection is in fact an array of many intersections, and the simple pointer is therefore replaced by an _array of pointers*_. These pointer aliases are used as follows:
```
// Imports BSDFPtr, EmitterPtr, etc..
MI_IMPORT_TYPES()

Scenescene=...;
Maskactive=...;
Ray3fray=...;
SurfaceInteraction3fsi=scene->ray_intersect(ray,active);

// Array of pointers if Float is an array
BSDFPtrbsdf=si.bsdf();

// Dr.Jit is able to dispatch method calls involving arrays of pointers
bsdf->eval(...,active);

```
Copy to clipboard
More information on vectorized method calls is provided in the [Dr.Jit documentation](https://enoki.readthedocs.io/en/master/calls.html).
## Variant-specific code[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/variants_cpp.html#variant-specific-code "Link to this heading")
The C++17 `if constexpr` statement is often used throughout the codebase to restrict code fragments to specific variants. For instance the following C++ snippet converts a spectrum to an XYZ tristimulus value, which crucially depends on the color representation of the variant being compiled.
```
Ray3fray=...;
Maskactive=...;
Spectrumresult=compute_stuff(ray,active);

Color3fxyz;
ifconstexpr(is_monochromatic_v<Spectrum>)
xyz=result.x();
elseifconstexpr(is_rgb_v<Spectrum>)
xyz=srgb_to_xyz(result,active);
else
xyz=spectrum_to_xyz(result,ray.wavelengths,active);

```
Copy to clipboard
Since `if constexpr` is resolved at compile-time, this branch does not cause any runtime overheads. Another useful feature of `if constexpr` is that it suppresses compilation errors in disabled branches. This makes it possible to write generic code that could potentially produce compilation errors when expressed using ordinary (non-`constexpr`) `if` statements (for example, by accessing a member of a class that may not exist in all variants).
Mitsuba provides various _type-traits_ such as `is_monochromatic_v` to query variant-specific properties. They can be found in `include/mitsuba/core/traits.h`.
* * *
