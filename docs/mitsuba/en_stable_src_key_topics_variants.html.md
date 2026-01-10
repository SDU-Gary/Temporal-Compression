---
url: https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html
crawled_at: 2025-11-13T17:16:13.017405
title: https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Choosing variants[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html#choosing-variants "Link to this heading")
Mitsuba 3 is a retargetable rendering system that provides a set of different system “ _variants_ ” that change elementary aspects of simulation—they can for instance replace the representation of color to support monochromatic, RGB, spectral, or even polarized illumination. Similarly, the numerical representation underlying the simulation can be exchanged to perform renderings using a higher amount of precision, process many light paths at once on the GPU, or it can be mathematically differentiated to solve inverse problems. All variants are automatically created from a single generic codebase. Each variant is associated with an identifying name that is composed of several parts:
[![../../_images/variant.svg](https://mitsuba.readthedocs.io/en/stable/_images/variant.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/variant.svg)
As many as 60 different variants of the renderer are presently available, shown in the list below.
> **Show/hide available variants**
>   * scalar_mono
>   * scalar_mono_double
>   * scalar_mono_polarized
>   * scalar_mono_polarized_double
>   * scalar_rgb
>   * scalar_rgb_double
>   * scalar_rgb_polarized
>   * scalar_rgb_polarized_double
>   * scalar_spectral
>   * scalar_spectral_double
>   * scalar_spectral_polarized
>   * scalar_spectral_polarized_double
>   * llvm_mono
>   * llvm_mono_double
>   * llvm_mono_polarized
>   * llvm_mono_polarized_double
>   * llvm_rgb
>   * llvm_rgb_double
>   * llvm_rgb_polarized
>   * llvm_rgb_polarized_double
>   * llvm_spectral
>   * llvm_spectral_double
>   * llvm_spectral_polarized
>   * llvm_spectral_polarized_double
>   * llvm_ad_mono
>   * llvm_ad_mono_double
>   * llvm_ad_mono_polarized
>   * llvm_ad_mono_polarized_double
>   * llvm_ad_rgb
>   * llvm_ad_rgb_double
>   * llvm_ad_rgb_polarized
>   * llvm_ad_rgb_polarized_double
>   * llvm_ad_spectral
>   * llvm_ad_spectral_double
>   * llvm_ad_spectral_polarized
>   * llvm_ad_spectral_polarized_double
>   * cuda_mono
>   * cuda_mono_double
>   * cuda_mono_polarized
>   * cuda_mono_polarized_double
>   * cuda_rgb
>   * cuda_rgb_double
>   * cuda_rgb_polarized
>   * cuda_rgb_polarized_double
>   * cuda_spectral
>   * cuda_spectral_double
>   * cuda_spectral_polarized
>   * cuda_spectral_polarized_double
>   * cuda_ad_mono
>   * cuda_ad_mono_double
>   * cuda_ad_mono_polarized
>   * cuda_ad_mono_polarized_double
>   * cuda_ad_rgb
>   * cuda_ad_rgb_double
>   * cuda_ad_rgb_polarized
>   * cuda_ad_rgb_polarized_double
>   * cuda_ad_spectral
>   * cuda_ad_spectral_double
>   * cuda_ad_spectral_polarized
>   * cuda_ad_spectral_polarized_double
> 

However, when installing Mitsuba on your system with `pip`, only a subset of those variants will be installed.
>   * scalar_rgb
>   * scalar_spectral
>   * scalar_spectral_polarized
>   * llvm_ad_rgb
>   * llvm_ad_mono
>   * llvm_ad_mono_polarized
>   * llvm_ad_spectral
>   * llvm_ad_spectral_polarized
>   * cuda_ad_rgb
>   * cuda_ad_mono
>   * cuda_ad_mono_polarized
>   * cuda_ad_spectral
>   * cuda_ad_spectral_polarized
> 

This is to avoid the burden of downloading massive binaries, but those should be enough to get you started with Mitsuba 3. For advanced applications that require another variant, you will need to compile it yourself. For this please refer to the [documentation on compiling the system from source](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html#sec-compiling).
## Part 1: Computational backend[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html#part-1-computational-backend "Link to this heading")
The computational backend controls how basic arithmetic operations like additions or multiplications are realized by the system. The following choices are available:
  * The `scalar` backend performs computation on the CPU using normal floating point arithmetic similar to older versions of Mitsuba. The renderer processes individual rays at a time. This mode is the easiest to understand and therefore preferred for fixing compilation errors and debugging the renderer.
  * The `cuda` backend offloads computation to the GPU using [Dr.Jit’s](https://github.com/mitsuba-renderer/drjit) just-in-time (JIT) compiler that transforms computation into CUDA kernels. Using this backend, each operation typically operates on millions of inputs at the same time. Mitsuba then becomes what is known as a _wavefront path tracer_ and delegates ray tracing on the GPU to NVIDIA’s OptiX library. Note that this requires a relatively recent NVIDIA GPU: ideally _Turing_ or newer. The older _Pascal_ architecture is also supported but tends to be slower because it lacks ray tracing hardware acceleration.
  * `llvm`: Similar to the `cuda` backend, the computation required to render a scene is just-in-time compiled using `Dr.Jit` to parallel CPU kernels that process many rays at the same time. This uses the LLVM compiler framework, which is detected and loaded at runtime. If you don’t have a NVIDIA GPU, this mode is a great alternative to the `cuda` backend.


An appealing aspect of the `llvm` and `cuda` modes, is that they expose _vectorized_ Python interfaces that operate on arbitrarily large set of inputs. This means that millions of ray tracing operations or BSDF evaluations can be performed with a single Python function call, enabling efficient prototyping within Python or Jupyter notebooks without costly iteration over many elements.
## Part 2: Automatic differentiation[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html#part-2-automatic-differentiation "Link to this heading")
It is possible to add the `_ad` suffix to enable automatic differentiation for both `cuda` and `llvm` modes. In which case the backend will furthermore propagates derivative information through the simulation, which is a crucial ingredient for solving _inverse problems_ using rendering algorithms.
The following shows an example from [[NDVZJ19](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id11 "Merlin Nimier-David, Delio Vicini, Tizian Zeltner, and Wenzel Jakob. Mitsuba 2: a retargetable forward and inverse renderer. Transactions on Graphics \(Proceedings of SIGGRAPH Asia\), December 2019. URL: https://rgl.epfl.ch/publications/NimierDavidVicini2019Mitsuba2.")]. Here, Mitsuba 3 is used to compute the height profile of a transparent glass panel that refracts red, green, and blue light in such a way as to reproduce a specified color image.
[![../../_images/caustic.jpg](https://mitsuba.readthedocs.io/en/stable/_images/caustic.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/caustic.jpg)
The main use case of the `_ad` backends is _differentiable rendering_ , which interprets the rendering algorithm as a function f(x) that converts an input x (the scene description) into an output y (the rendering). This function f is then mathematically differentiated to obtain dydx, providing a first-order approximation of how a desired change in the output y (the rendering) can be achieved by changing the inputs x (the scene description). Together with a differentiable _objective function_ g(y) that quantifies the suitability of tentative scene parameters and a gradient-based optimization algorithm, a differentiable renderer can be used to solve complex inverse problems involving light.
[![../../_images/autodiff.jpg](https://mitsuba.readthedocs.io/en/stable/_images/autodiff.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/autodiff.jpg)
The documentations provides several applied examples on differentiable and inverse rendering.
## Part 3: Color representation[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html#part-3-color-representation "Link to this heading")
The next part determines how Mitsuba represents color information. The following choices are available:
  * `mono` completely disables the concept of color, which is useful when simulating scenes that are inherently monochromatic (e.g. illumination due to a laser). This mode is great for writing testcases where color is simply not relevant. When an input scene provides color information, mono mode automatically converts it to grayscale.
  * `rgb` mode selects an RGB-based color representation. This is a reasonable default choice and matches the typical behavior of the previous generation of Mitsuba. On the flipside, RGB mode can be a poor approximation of how color works in the real world. Please click on the following for a longer explanation.
> **Issues involving RGB-based rendering (click to expand)**
> **Problematic aspects of RGB-based color representations:** A RGB rendering algorithm frequently performs two color-related operations: component-wise addition to combine different sources of light, and component-wise multiplication of RGB color vectors to model interreflection. While addition is fine, RGB multiplication turns out to be a nonsensical operation, that can give very different answers depending on the underlying RGB color space.
> Suppose we are rendering a scene in an sRGB color space, where a green light with radiance [0,0,1] reflects from a very green surface with albedo [0,0,1]. The component-wise multiplication [0,0,1]⊗[0,0,1]=[0,0,1] tells us that no light is absorbed by the surface. So good so far.
> Let’s now switch to a larger color space named _Rec. 2020_. That same green color is no longer at the extreme of the color gamut but lies somewhere inside.
> [![../../_images/rgb-mode-issue.svg](https://mitsuba.readthedocs.io/en/stable/_images/rgb-mode-issue.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/rgb-mode-issue.svg)
> For simplicity, let’s suppose it has coordinates [0,0,12]. Now, the same calculation [0,0,12]⊗[0,0,12]=[0,0,14] tells us that half of the light is absorbed by the surface, which illustrates the problem with RGB multiplication. The solution to this problem is to multiply colors in the spectral domain instead.
  * Finally, `spectral` mode switches to a fully spectral color representation spanning the visible range (360…830nm). The wavelength domain is simply treated as yet another dimension of the space of light paths over which the rendering algorithm must integrate.
This improves accuracy especially in scenarios where measured spectral data is available. Consider for example the two Cornell box renderings below: on the left side, the spectral reflectance data of all materials is first converted to RGB and rendered using the `scalar_rgb` variant, producing a deceivingly colorful image. In contrast, the `scalar_spectral` variant that correctly accounts for the spectral characteristics, produces a more muted coloration.
[![../../_images/variants_cbox_rgb.jpg](https://mitsuba.readthedocs.io/en/stable/_images/variants_cbox_rgb.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/variants_cbox_rgb.jpg)
RGB Mode[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html#id4 "Link to this image")
[![../../_images/variants_cbox_spectral.jpg](https://mitsuba.readthedocs.io/en/stable/_images/variants_cbox_spectral.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/variants_cbox_spectral.jpg)
Spectral Mode[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html#id5 "Link to this image")
Note that Mitsuba still generates RGB output images by default even when spectral mode is active. It is also important to note that many existing Mitsuba scenes only specify RGB color information. Spectral Mitsuba can still render such scenes – in this case, it determines plausible smooth spectra corresponding to the specified RGB colors [[JH19](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id9 "Wenzel Jakob and Johannes Hanika. A low-dimensional function space for efficient spectral upsampling. Computer Graphics Forum \(Proceedings of Eurographics\), March 2019. URL: https://rgl.epfl.ch/publications/Jakob2019Spectral.")]. We also recommend taking a look at the [Spectral film plugin](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_films.html#film-specfilm) which is able to output spectral multichannel output images.


## Part 4: Polarization[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html#part-4-polarization "Link to this heading")
If desired, Mitsuba 3 can keep track of the full polarization state of light. Polarization refers to the property that light is an electromagnetic wave that oscillates perpendicularly to the direction of travel. This oscillation can take on infinitely many different shapes—the following images show examples of _horizontal_ and _elliptical_ polarization.
[![../../_images/polarization_wave_variations.svg](https://mitsuba.readthedocs.io/en/stable/_images/polarization_wave_variations.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/polarization_wave_variations.svg)
Because humans do not perceive polarization, accounting for it is usually not necessary to render realistic images. However, polarization is easily observed using a variety of measurement devices and cameras, and it tends to provide a wealth of information about the material and shape of visible objects. For this reason, polarization is a powerful tool for solving inverse problems, and this is one of the reasons why we chose to support it in Mitsuba 3. Note that accounting for polarization comes at a cost—roughly a 1.5-2X increase in rendering time.
Inside the light transport simulation, _Stokes vectors_ are used to parameterize the elliptical shape of the transverse oscillations, and _Mueller matrices_ are used to compute the effect of surface scattering on the polarization [[Col93](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id2 "Edward Collett. Polarized light : fundamentals and application. Marcel Dekker New York, 1993. ISBN 0824787293.")]. For more details regarding the implementation of the polarized rendering modes, please refer to the [Polarization](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#developer-guide-polarization) section in the developer guide.
## Part 5: Precision[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html#part-5-precision "Link to this heading")
Mitsuba 3 normally relies on single precision (32 bit) arithmetic, but double precision (64 bit) is optionally available adding the `_double` suffix to the variant name. We find this particularly helpful for debugging: whether or not an observed problem arises due to floating point imprecisions can normally be determined after switching to double precision. Note that Embree and OptiX don’t support double precision, hence ray-tracing operations will run in reduced (single) precision in those modes. The only way to use precision for everything including ray tracing is to render on the CPU (`scalar` or `llvm`) and disable Embree in CMake. Also note that double precision arithmetic runs with greatly reduced throughput (1/64th of FP32) on recent NVIDIA GPUs.
* * *
