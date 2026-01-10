---
url: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_rfilters.html
crawled_at: 2025-11-13T17:17:05.447720
title: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_rfilters.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_rfilters.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_rfilters.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Reconstruction filters[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_rfilters.html#reconstruction-filters "Link to this heading")
Image reconstruction filters are responsible for converting a series of radiance samples generated jointly by the sampler and integrator into the final output image that will be written to disk at the end of a rendering process. This section gives a brief overview of the reconstruction filters that are available in Mitsuba. There is no universally superior filter, and the final choice depends on a trade-off between sharpness, ringing, and aliasing, and computational efficiency.
Desirable properties of a reconstruction filter are that it sharply captures all of the details that are displayable at the requested image resolution, while avoiding aliasing and ringing. Aliasing is the incorrect leakage of high-frequency into low-frequency detail, and ringing denotes oscillation artifacts near discontinuities, such as a light-shadow transition.
## Box filter (box)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_rfilters.html#box-filter-box "Link to this heading")
This is the fastest, but also about the worst possible reconstruction filter, since it is prone to severe aliasing. It is included mainly for completeness, though some rare situations may warrant its use.
XMLPython
```
<rfiltertype="box"/>

```
Copy to clipboard
```
'type': 'box',

```
Copy to clipboard
## Tent filter (tent)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_rfilters.html#tent-filter-tent "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
radius | float | Specifies the radius of the tent function (Default: 1.0) |   
Simple tent (triangular) filter. This reconstruction filter never suffers from ringing and usually causes less aliasing than a naive box filter. When rendering scenes with sharp brightness discontinuities, this may be useful; otherwise, negative-lobed filters may be preferable (e.g. Mitchell-Netravali or Lanczos Sinc).
XMLPython
```
<rfiltertype="tent">
<floatname="radius"value="1.25"/>
</rfilter>

```
Copy to clipboard
```
'type': 'tent',
'radius': 1.25,

```
Copy to clipboard
## Gaussian filter (gaussian)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_rfilters.html#gaussian-filter-gaussian "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
stddev | float | Specifies the standard deviation (Default: 0.5) |   
This is a windowed Gaussian filter with configurable standard deviation. It often produces pleasing results, and never suffers from ringing, but may occasionally introduce too much blurring.
When no reconstruction filter is explicitly requested, this is the default choice in Mitsuba.
XMLPython
```
<rfiltertype="gaussian">
<floatname="stddev"value="0.25"/>
</rfilter>

```
Copy to clipboard
```
'type': 'gaussian',
'stddev': 0.25

```
Copy to clipboard
## Mitchell filter (mitchell)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_rfilters.html#mitchell-filter-mitchell "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
A | float | A parameter in the original paper (Default: 1/3) |   
B | float | B parameter in the original paper (Default: 1/3) |   
Separable cubic spline reconstruction filter by Mitchell and Netravali [[MN88](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id20 "Don P. Mitchell and Arun N. Netravali. Reconstruction filters in computer-graphics. SIGGRAPH Comput. Graph., 22\(4\):221–228, June 1988.")]. This is often a good compromise between sharpness and ringing.
XMLPython
```
<rfiltertype="mitchell">
<floatname="A"value="0.25"/>
<floatname="B"value="0.55"/>
</rfilter>

```
Copy to clipboard
```
'type': 'mitchell',
'A': 0.25,
'B': 0.55

```
Copy to clipboard
## Catmull-Rom filter (catmullrom)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_rfilters.html#catmull-rom-filter-catmullrom "Link to this heading")
Special version of the Mitchell-Netravali filter with constants B and C configured to match the Catmull-Rom spline. It usually does a better job at at preserving sharp features at the cost of more ringing.
XMLPython
```
<rfiltertype="catmullrom"/>

```
Copy to clipboard
```
'type': 'catmullrom',

```
Copy to clipboard
## Lanczos filter (lanczos)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_rfilters.html#lanczos-filter-lanczos "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
lobes | integer | Sets the desired number of filter side-lobes. The higher, the closer the filter will approximate an optimal low-pass filter, but this also increases ringing. Values of 2 or 3 are common (Default: 3) |   
This is a windowed version of the theoretically optimal low-pass filter. It is generally one of the best available filters in terms of producing sharp high-quality output. Its main disadvantage is that it produces strong ringing around discontinuities, which can become a serious problem when rendering bright objects with sharp edges (a directly visible light source will for instance have black fringing artifacts around it). This is also the computationally slowest reconstruction filter.
XMLPython
```
<rfiltertype="lanczos">
<integername="lobes"value="4"/>
</rfilter>

```
Copy to clipboard
```
'type': 'lanczos',
'lobes': 4

```
Copy to clipboard
[**Build and run** apps in over 115 regions with MongoDB Atlas, the database for every enterprise.](https://server.ethicalads.io/proxy/click/9506/019a7c81-0187-7bc1-9ebe-bae87983ea39/)
[Ads by EthicalAds](https://www.ethicalads.io/advertisers/topics/frontend-web/?ref=ea-text)
* * *
