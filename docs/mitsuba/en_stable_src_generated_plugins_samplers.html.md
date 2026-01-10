---
url: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html
crawled_at: 2025-11-13T17:13:31.847476
title: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Samplers[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#samplers "Link to this heading")
When rendering an image, Mitsuba 3 has to solve a high-dimensional integration problem that involves the geometry, materials, lights, and sensors that make up the scene. Because of the mathematical complexity of these integrals, it is generally impossible to solve them analytically — instead, they are solved numerically by evaluating the function to be integrated at a large number of different positions referred to as samples. Sample generators are an essential ingredient to this process: they produce points in a (hypothetical) infinite dimensional hypercube [0,1]∞ that constitute the canonical representation of these samples.
To do its work, a rendering algorithm, or integrator, will send many queries to the sample generator. Generally, it will request subsequent 1D or 2D components of this infinite-dimensional _point_ and map them into a more convenient space (for instance, positions on surfaces). This allows it to construct light paths to eventually evaluate the flow of light through the scene.
## Independent sampler (independent)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#independent-sampler-independent "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
sample_count | integer | Number of samples per pixel (Default: 4) |   
seed | integer | Seed offset (Default: 0) |   
The independent sampler produces a stream of independent and uniformly distributed pseudorandom numbers. Internally, it relies on the [PCG32 random number generator](https://www.pcg-random.org/) by Melissa O’Neill.
This is the most basic sample generator; because no precautions are taken to avoid sample clumping, images produced using this plugin will usually take longer to converge. Looking at the figures below where samples are projected onto a 2D unit square, we see that there are both regions that don’t receive many samples (i.e. we don’t know much about the behavior of the function there), and regions where many samples are very close together (which likely have very similar values), which will result in higher variance in the rendered image.
This sampler is initialized using a deterministic procedure, which means that subsequent runs of Mitsuba should create the same image. In practice, when rendering with multiple threads and/or machines, this is not true anymore, since the ordering of samples is influenced by the operating system scheduler. Although these should be absolutely negligible, with relative errors on the order of the machine epsilon (6⋅10−8) in single precision.
[![../../_images/independent_1024_samples.svg](https://mitsuba.readthedocs.io/en/stable/_images/independent_1024_samples.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/independent_1024_samples.svg)
1024 samples projected onto the first two dimensions.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id4 "Link to this image")
[![../../_images/independent_64_samples_and_proj.svg](https://mitsuba.readthedocs.io/en/stable/_images/independent_64_samples_and_proj.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/independent_64_samples_and_proj.svg)
64 samples projected onto the first two dimensions and their projection on both 1D axis (top and right plot).[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id5 "Link to this image")
XMLPython
```
<samplertype="independent">
<integername="sample_count"value="64"/>
</sampler>

```
Copy to clipboard
```
'type': 'independent',
'sample_count': '64'

```
Copy to clipboard
## Stratified sampler (stratified)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#stratified-sampler-stratified "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
sample_count | integer | Number of samples per pixel. This number should be a square number (Default: 4) |   
seed | integer | Seed offset (Default: 0) |   
jitter | boolean | Adds additional random jitter withing the stratum (Default: True) |   
The stratified sample generator divides the domain into a discrete number of strata and produces a sample within each one of them. This generally leads to less sample clumping when compared to the independent sampler, as well as better convergence.
[![../../_images/sampler_independent_16spp.jpg](https://mitsuba.readthedocs.io/en/stable/_images/sampler_independent_16spp.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/sampler_independent_16spp.jpg)
Independent sampler - 16 samples per pixel[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id6 "Link to this image")
[![../../_images/sampler_stratified_16spp.jpg](https://mitsuba.readthedocs.io/en/stable/_images/sampler_stratified_16spp.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/sampler_stratified_16spp.jpg)
Stratified sampler - 16 samples per pixel[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id7 "Link to this image")
[![../../_images/stratified_1024_samples.svg](https://mitsuba.readthedocs.io/en/stable/_images/stratified_1024_samples.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/stratified_1024_samples.svg)
1024 samples projected onto the first two dimensions which are well distributed if we compare to the independent sampler.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id8 "Link to this image")
[![../../_images/stratified_64_samples_and_proj.svg](https://mitsuba.readthedocs.io/en/stable/_images/stratified_64_samples_and_proj.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/stratified_64_samples_and_proj.svg)
64 samples projected in 2D and on both 1D axis (top and right plot). Every strata contains a single sample creating a good distribution when projected in 2D. Projections on both 1D axis still exhibit sample clumping which will result in higher variance, for instance when sampling a thin stretched rectangular area light.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id9 "Link to this image")
XMLPython
```
<samplertype="stratified">
<integername="sample_count"value="64"/>
</sampler>

```
Copy to clipboard
```
'type': 'stratified',
'sample_count': '4'

```
Copy to clipboard
## Correlated Multi-Jittered sampler (multijitter)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#correlated-multi-jittered-sampler-multijitter "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
sample_count | integer | Number of samples per pixel. The sampler may internally choose to slightly increase this value to create a subdivision into strata that has an aspect ratio close to one. (Default: 4) |   
seed | integer | Seed offset (Default: 0) |   
jitter | boolean | Adds additional random jitter withing the substratum (Default: True) |   
This plugin implements the methods introduced in Pixar’s tech memo [[Ken13](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id5)].
Unlike the previously described stratified sampler, multi-jittered sample patterns produce samples that are well stratified in 2D but also well stratified when projected onto one dimension. This can greatly reduce the variance of a Monte-Carlo estimator when the function to evaluate exhibits more variation along one axis of the sampling domain than the other.
This sampler achieves this by first placing samples in a canonical arrangement that is stratified in both 2D and 1D. It then shuffles the x-coordinate of the samples in every columns and the y-coordinate in every rows. Fortunately, this process doesn’t break the 2D and 1D stratification. Kensler’s method further reduces sample clumpiness by correlating the shuffling applied to the columns and the rows.
[![../../_images/sampler_independent_16spp.jpg](https://mitsuba.readthedocs.io/en/stable/_images/sampler_independent_16spp.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/sampler_independent_16spp.jpg)
Independent sampler - 16 samples per pixel[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id10 "Link to this image")
[![../../_images/sampler_multijitter_16spp.jpg](https://mitsuba.readthedocs.io/en/stable/_images/sampler_multijitter_16spp.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/sampler_multijitter_16spp.jpg)
Correlated Multi-Jittered sampler - 16 samples per pixel[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id11 "Link to this image")
[![../../_images/multijitter_1024_samples.svg](https://mitsuba.readthedocs.io/en/stable/_images/multijitter_1024_samples.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/multijitter_1024_samples.svg)
1024 samples projected onto the first two dimensions.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id12 "Link to this image")
[![../../_images/multijitter_64_samples_and_proj.svg](https://mitsuba.readthedocs.io/en/stable/_images/multijitter_64_samples_and_proj.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/multijitter_64_samples_and_proj.svg)
64 samples projected onto the first two dimensions and their projection on both 1D axis (top and right plot). As expected, the samples are well stratified both in 2D and 1D.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id13 "Link to this image")
XMLPython
```
<samplertype="multijitter">
<integername="sample_count"value="64"/>
</sampler>

```
Copy to clipboard
```
'type': 'multijitter',
'sample_count': '64'

```
Copy to clipboard
## Orthogonal Array sampler (orthogonal)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#orthogonal-array-sampler-orthogonal "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
sample_count | integer | Number of samples per pixel. This value has to be the square of a prime number. (Default: 4) |   
strength | integer | Orthogonal array’s strength (Default: 2) |   
seed | integer | Seed offset (Default: 0) |   
jitter | boolean | Adds additional random jitter withing the substratum (Default: True) |   
This plugin implements the Orthogonal Array sampler generator introduced by Jarosz et al. [[JEK+19](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id7 "Wojciech Jarosz, Afnan Enayet, Andrew Kensler, Charlie Kilpatrick, and Per Christensen. Orthogonal array sampling for Monte Carlo rendering. Computer Graphics Forum \(Proceedings of EGSR\), 38\(4\):135–147, July 2019. doi:10/gf6rx5.")]. It generalizes correlated multi-jittered sampling to higher dimensions by using _orthogonal arrays (OAs)_. An OA of strength s has the property that projecting the generated samples to any combination of s dimensions will always result in a well stratified pattern. In other words, when s=2 (default value), the high-dimensional samples are simultaneously stratified in all 2D projections as if they had been produced by correlated multi-jittered sampling. By construction, samples produced by this generator are also well stratified when projected on both 1D axis.
This sampler supports OA of strength other than 2, although this isn’t recommended as the stratification of 2D projections of those samples wouldn’t be ensured anymore.
[![../../_images/sampler_independent_25spp.jpg](https://mitsuba.readthedocs.io/en/stable/_images/sampler_independent_25spp.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/sampler_independent_25spp.jpg)
Independent sampler - 25 samples per pixel[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id14 "Link to this image")
[![../../_images/sampler_orthogonal_25spp.jpg](https://mitsuba.readthedocs.io/en/stable/_images/sampler_orthogonal_25spp.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/sampler_orthogonal_25spp.jpg)
Orthogonal Array sampler - 25 samples per pixel[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id15 "Link to this image")
[![../../_images/orthogonal_1369_samples.svg](https://mitsuba.readthedocs.io/en/stable/_images/orthogonal_1369_samples.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/orthogonal_1369_samples.svg)
1369 samples projected onto the first two dimensions.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id16 "Link to this image")
[![../../_images/orthogonal_49_samples_and_proj.svg](https://mitsuba.readthedocs.io/en/stable/_images/orthogonal_49_samples_and_proj.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/orthogonal_49_samples_and_proj.svg)
49 samples projected onto the first two dimensions and their projection on both 1D axis (top and right plot). The pattern is well stratified in both 2D and 1D projections. This is true for every pair of dimensions of the high-dimensional samples.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id17 "Link to this image")
XMLPython
```
<samplertype="orthogonal">
<integername="sample_count"value="4"/>
</sampler>

```
Copy to clipboard
```
'type': 'orthogonal',
'sample_count': '4'

```
Copy to clipboard
## Low discrepancy sampler (ldsampler)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#low-discrepancy-sampler-ldsampler "Link to this heading")
This plugin implements a simple hybrid sampler that combines aspects of a Quasi-Monte Carlo sequence with a pseudorandom number generator based on a technique proposed by Kollig and Keller [[KK02](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id6 "Thomas Kollig and Alexander Keller. Efficient Multidimensional Sampling. Computer Graphics Forum, 21\(3\):557-563, 2002. doi:10.1111/1467-8659.00706.")]. It is a good and fast general-purpose sample generator. Other QMC samplers exist that can generate even better distributed samples, but this comes at a higher cost in terms of performance. This plugin is based on Mitsuba 1’s default sampler (also called ldsampler).
Roughly, the idea of this sampler is that all of the individual 2D sample dimensions are first filled using the same (0, 2)-sequence, which is then randomly scrambled and permuted using a shuffle network. The name of this plugin stems from the fact that, by construction, (0, 2)-sequences achieve a low [star discrepancy](https://en.wikipedia.org/wiki/Low-discrepancy_sequence), which is a quality criterion on their spatial distribution.
[![../../_images/sampler_independent_16spp.jpg](https://mitsuba.readthedocs.io/en/stable/_images/sampler_independent_16spp.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/sampler_independent_16spp.jpg)
Independent sampler - 16 samples per pixel[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id18 "Link to this image")
[![../../_images/sampler_ldsampler_16spp.jpg](https://mitsuba.readthedocs.io/en/stable/_images/sampler_ldsampler_16spp.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/sampler_ldsampler_16spp.jpg)
Low-discrepancy sampler - 16 samples per pixel[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id19 "Link to this image")
[![../../_images/ldsampler_1024_samples.svg](https://mitsuba.readthedocs.io/en/stable/_images/ldsampler_1024_samples.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/ldsampler_1024_samples.svg)
1024 samples projected onto the first two dimensions.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id20 "Link to this image")
[![../../_images/ldsampler_64_samples_and_proj.svg](https://mitsuba.readthedocs.io/en/stable/_images/ldsampler_64_samples_and_proj.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/ldsampler_64_samples_and_proj.svg)
A projection of the first 64 samples onto the first two dimensions and their projection on both 1D axis (top and right plot). The 1D stratification is perfect as this sampler doesn’t add additional random perturbation to the sample positions.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id21 "Link to this image")
[![../../_images/ldsampler_1024_samples_dim_32.svg](https://mitsuba.readthedocs.io/en/stable/_images/ldsampler_1024_samples_dim_32.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/ldsampler_1024_samples_dim_32.svg)
1024 samples projected onto the 32th and 33th dimensions, which look almost identical. However, note that the points have been scrambled to reduce correlations between dimensions.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id22 "Link to this image")
[![../../_images/ldsampler_64_samples_and_proj_dim_32.svg](https://mitsuba.readthedocs.io/en/stable/_images/ldsampler_64_samples_and_proj_dim_32.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/ldsampler_64_samples_and_proj_dim_32.svg)
A projection of the first 64 samples onto the 32th and 33th dimensions.[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_samplers.html#id23 "Link to this image")
XMLPython
```
<samplertype="ldsampler">
<integername="sample_count"value="64"/>
</sampler>

```
Copy to clipboard
```
'type': 'ldsampler',
'sample_count': '64'

```
Copy to clipboard
* * *
