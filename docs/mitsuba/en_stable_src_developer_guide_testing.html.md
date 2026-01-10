---
url: https://mitsuba.readthedocs.io/en/stable/src/developer_guide/testing.html
crawled_at: 2025-11-13T17:15:03.106410
title: https://mitsuba.readthedocs.io/en/stable/src/developer_guide/testing.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/testing.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/testing.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Testing[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/testing.html#testing "Link to this heading")
To run the test suite, simply invoke `pytest`:
```
# or to run a single test file
pytest
```
Copy to clipboard
If you want to skip the execution of the slower tests, you can do so by running the test suite with the following flag.
```
'not slow'

```
Copy to clipboard
The build system also exposes a `pytest` target that executes `setpath` and parallelizes the test execution.
Copy to clipboard
## Testing multiple variants[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/testing.html#testing-multiple-variants "Link to this heading")
The system provides a variety of Python fixture to automatically run unit tests on subsets of the available variants. Those should be passed a first argument to the test function like in the following example:
```
# This test will run on all available variants
deftest_hello_world(variants_all):
    print(f'Hello {mi.variant()}')
    assert True

```
Copy to clipboard
Here is the list of available fixtures:
  * variants_all: execute on all variants
  * variants_all_scalar: execute on all `scalar_*` variants
  * variants_all_rgb: execute on all `*_rgb` variants
  * variants_all_spectral: execute on all `*_spectral` variants
  * variants_all_backends_once: execute the test for at least on variant per backend
  * variants_all_ad_rgb: execute on all `*_ad_rgb` variants
  * variants_all_ad_spectral: execute on all `*_ad_spectral` variants
  * variants_any_scalar: execute on one `scalar_*` variant if available
  * variants_any_llvm: execute on one `llvm_*` variant if available
  * variants_any_cuda: execute on one `cuda_*` variant if available
  * variants_vec_backends_once: execute on one `cuda_*` and one `llvm_*` variant if available
  * variants_vec_backends_once_rgb: execute on one `cuda_*_rgb` and one `llvm_*_rgb` variant if available
  * variants_vec_backends_once_spectral: execute on one `cuda_*_spectral` and one `llvm_*_spectral` variant if available
  * variants_vec_rgb: execute on `cuda_*_rgb` and `llvm_*_rgb` variants
  * variants_vec_spectral: execute on `cuda_*_spectral` and `llvm_*_spectral` variants


## Chi^2 tests[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/testing.html#chi-2-tests "Link to this heading")
The `mitsuba.chi2` module implements the Pearson’s chi-square test for testing goodness of fit of a distribution to a known reference distribution.
The implementation specifically compares a Monte Carlo sampling strategy on a 2D (or lower dimensional) space against a reference distribution obtained by numerically integrating a probability density function over grid in the distribution’s parameter domain.
This is used extensively throughout the test suite to valid the implementation of `BSDFs`, `Emitters`, and other sampling code.
It is possible to test your own sampling code in the following way:
```
importmitsubaasmi
mi.set_variant('llvm_rgb')

# some sampling code
defmy_sample(sample):
    return mi.warp.square_to_cosine_hemisphere(sample)

# the corresponding probability density function
defmy_pdf(p):
    return mi.warp.square_to_cosine_hemisphere_pdf(p)

chi2 = mi.ChiSquareTest(
    domain=mi.SphericalDomain(),
    sample_func=my_sample,
    pdf_func=my_pdf,
    sample_dim=2
)

assert chi2.run()

```
Copy to clipboard
In case of failure, the target density and histogram were written to `chi2_data.py` which can simply be run to plot the data:
Copy to clipboard
The `mitsuba.chi2` module also provides a set of `Adapter` functions which can be used to wrap different plugins (e.g. `BSDF`, `Emitter`, …) in order to test them:
```
importmitsubaasmi
importdrjitasdr

mi.set_variant('llvm_rgb')

xml = """<float name="alpha" value="0.5"/>
         <boolean name="sample_visible" value="false"/>
         <string name="distribution" value="ggx"/>
      """
wi = dr.normalize([0.2, -0.6, -0.5])
sample_func, pdf_func = mi.BSDFAdapter("roughdielectric", xml, wi=wi)

chi2 = mi.ChiSquareTest(
    domain=mi.SphericalDomain(),
    sample_func=sample_func,
    pdf_func=pdf_func,
    sample_dim=3
)

assert chi2.run()

# Forces the chi2 test to dump the plotting script (optional)
chi2._dump_tables()

```
Copy to clipboard
Here is the figure generated by the `chi2_data.py` script from the example above:
[![../../_images/chi2_example.png](https://mitsuba.readthedocs.io/en/stable/_images/chi2_example.png) ](https://mitsuba.readthedocs.io/en/stable/_images/chi2_example.png)
The plot on the left shows the density function generated by numerically integrating the analytical `pdf()` method of a `roughdielectric` BSDF with an incoming vector coming from inside. Most of the energy leaves the surface (upper half of the plot) while some energy gets reflected back inside the surface (lower half of the plot).
The middle plot shows the same density function but this time computed as a histogram of sampled directions resulting from the `sample()` method of the `roughdielectric` BSDF.
The right plot shows the difference between the two density functions. The sampling routine of the BSDF being stochastic, it is expected to see a mix of negative and positive values as the histogram is still noisy. The main role of the `ChiSquareTest` is to decide whether the observed deviation is within the range of random noise, or whether there are systematic biases that should lead to a test failure.
For more information, see [`mitsuba.chi2.ChiSquareTest`](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.chi2.ChiSquareTest "mitsuba.chi2.ChiSquareTest").
## Rendering test suite and Z-test[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/testing.html#rendering-test-suite-and-z-test "Link to this heading")
On top of test _unit tests_ , the framework implements a mechanism that automatically renders a set of test scenes and applies the [Z-test](https://en.wikipedia.org/wiki/Z-test) to compare the resulting images and some reference images.
Those tests are really useful to reveal bugs at the interaction between the individual components of the renderer.
The test scenes are rendered using all the different enabled variants of the renderer, ensuring for instance that the `scalar_rgb` renders match the `cuda_rgb` renders.
To only run the rendering test suite, use the following command:
Copy to clipboard
One can easily add a scene to the `resources/data/tests/scenes/` folder to add it to the rendering test suite. Then, the missing reference images can be generated using the following command:
Copy to clipboard
[**Simplify infrastructure** with MongoDB Atlas, the leading developer data platform](https://server.ethicalads.io/proxy/click/9504/019a7c7f-25d6-7110-a928-4a1b78392ccd/)
[Ads by EthicalAds](https://www.ethicalads.io/advertisers/topics/data-science/?ref=ea-text)
* * *
