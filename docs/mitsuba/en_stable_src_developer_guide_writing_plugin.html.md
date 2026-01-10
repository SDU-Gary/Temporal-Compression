---
url: https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html
crawled_at: 2025-11-13T17:15:55.985002
title: https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# C++ Plugins & Macros[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html#c-plugins-macros "Link to this heading")
This section provides an overview of the “skeleton” of a basic plugin definition, including an explanation of the roles of the various `MI_*` macros. These macros are responsible for importing types, instantiating variants, and providing run-time type information.
## Example code[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html#example-code "Link to this heading")
Consider the following hypothetical plugin `MyPlugin`:
```
NAMESPACE_BEGIN(mitsuba)

template<typenameFloat,typenameSpectrum>
classMyPlugin:publicPluginInterface<Float,Spectrum>{
public:
MI_IMPORT_BASE(PluginInterface,m_some_member,some_method)
MI_IMPORT_TYPES()

MyPlugin();

Spectrumfoo(...,Maskactive)constoverride{
MI_MASKED_FUNCTION(ProfilerPhase::MyEval,active)
// ...
}

/// Indicate the name of this class (for logging)
MI_DECLARE_CLASS(MyPlugin)
};

/// Implement RTTI data structures
MI_EXPORT_PLUGIN(MyPlugin)
NAMESPACE_END(mitsuba)

```
Copy to clipboard
## Macros[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html#macros "Link to this heading")
###  MI_MASK_ARGUMENT(mask)[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html#mi-mask-argument-mask "Link to this heading")
This macro typically occurs at the beginning of a function that takes a mask as an argument.
```
voidmy_method(...,Maskactive){
MI_MASK_ARGUMENT(active);
}

```
Copy to clipboard
Masking is not really needed on scalar variants: if a function is called, we assume that `active=true`. Masking can unfortunately decrease performance in this case due to the generation of extra branches. Here, `MI_MASK_ARGUMENT` therefore expands to
```
ifconstexpr(is_scalar_v<Float>)
active=true;

```
Copy to clipboard
which turns the mask argument into a compile-time constant on scalar targets, allowing the compiler to optimize away the undesired branches. `MI_MASK_ARGUMENT` is also part of the next macro.
###  MI_MASKED_FUNCTION(phase, mask)[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html#mi-masked-function-phase-mask "Link to this heading")
This macro builds on the `MI_MASK_ARGUMENT` macro and typically occurs at the beginning of a function that takes a mask as an argument.
```
voidmy_method(...,Maskmask){
MI_MASKED_FUNCTION(ProfilerPhase::MyMethod,active);
}

```
Copy to clipboard
Masking is not really needed on scalar variants: if a function is called, we assume that `active=true`. Masking can unfortunately decrease performance in this case due to the generation of extra branches. Here, `MI_MASKED_FUNCTION` macro expands to
```
ifconstexpr(is_scalar_v<Float>)
active=true;
ScopedPhase_(ProfilerPhase::MyMethod);

```
Copy to clipboard
which turns the mask argument into a compile-time constant on scalar targets, allowing the compiler to optimize away the undesired branches.
Mitsuba ships with a powerful sampling profiler that facilitates tracking down hot-spots during rendering. The last line of this macro (`ScopedPhase`) informs this profiler that we are currently executing a function that belongs to the profiler phase `phase`.
###  MI_IMPORT_BASE(Name, …)[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html#mi-import-base-name "Link to this heading")
Because most Mitsuba classes are templates, attributes and methods of parent classes are not visible by default. They can be imported explicitly via `using Base::some_method;` statements, but writing many such statements is tiresome. The variadic macro `MI_IMPORT_BASE` expands into arbitrarily many such `using` statements. For example,
```
MI_IMPORT_BASE(Name,m_some_member,some_method)

```
Copy to clipboard
expands to
```
usingBase=Name<Float,Spectrum>;
usingBase::m_some_member;
usingBase::some_method;

```
Copy to clipboard
###  MI_IMPORT_CORE_TYPES()[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html#mi-import-core-types "Link to this heading")
This macro will generate a sequence of `using` declarations to import the Mitsuba core types (e.g. `Vector{1/2/3}{i/u/f/d}`, `Point{1/2/3}{i/u/f/d}`, …). They are automatically inferred from the definition of `Float`.
Note
A type named `Float` must exist preceding evaluation of this macro.
For example,
```
usingFloat=float;

MI_IMPORT_CORE_TYPES()

// expands to:

// ...
usingPoint2f=Point<Float,2>;
usingPoint3f=Point<Float,3>;
// ...
usingBoundingBox3f=BoundingBox<Point3f>;
// ...

```
Copy to clipboard
###  MI_IMPORT_TYPES(…)[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html#mi-import-types "Link to this heading")
This macro invokes `MI_IMPORT_CORE_TYPES()` and furthermore imports rendering-related types, such as `Ray3f`, `SurfaceInteraction3f`, `BSDF`, etc. These templated aliases will depend on the preceding declaration of the `Float` and `Spectrum`.
It is also possible to pass other types as arguments, for which templated aliases will be created:
```
usingFloat=float;
usingSpectrum=Spectrum<Float,4>;

MI_IMPORT_TYPES(MyType1,MyType2)

// expands to:

MI_IMPORT_CORE_TYPES()
// ...
usingRay3f=Ray<Point<Float,3>,Spectrum>;
// ...
usingSurfaceInteraction3f=SurfaceInteraction<Float,Spectrum>;
// ...
usingMyType1=MyType1<Float,Spectrum>;// alias for the optional parameters
usingMyType2=MyType2<Float,Spectrum>;

```
Copy to clipboard
###  MI_DECLARE_CLASS(Name)[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html#mi-declare-class-name "Link to this heading")
This macro should be invoked within the class declaration of the plugin. The provided `Name` parameter is picked up by log messages, warnings, or errors that are triggered by the plugin.
###  MI_EXPORT_PLUGIN(Name)[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html#mi-export-plugin-name "Link to this heading")
This macro will explicitly instantiate all enabled variants of a plugin:
```
MI_EXPORT_PLUGIN(Name)

// expands to:

templateclassMI_EXPORTName<float,Color<float,1>>// scalar_mono
templateclassMI_EXPORTName<float,Spectrum<float,4>>// scalar_spectral
// ...

```
Copy to clipboard
* * *
