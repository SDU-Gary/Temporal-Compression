---
url: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_phase.html
crawled_at: 2025-11-13T17:14:14.279333
title: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_phase.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_phase.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_phase.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Phase functions[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_phase.html#phase-functions "Link to this heading")
This section contains a description of all implemented medium scattering models, which are also known as phase functions. These are very similar in principle to surface scattering models (or BSDFs), and essentially describe where light travels after hitting a particle within the medium. Currently, only the most commonly used models for smoke, fog, and other homogeneous media are implemented.
## Isotropic phase function (isotropic)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_phase.html#isotropic-phase-function-isotropic "Link to this heading")
This phase function simulates completely uniform scattering, where all directionality is lost after a single scattering interaction. It does not have any parameters.
XMLPython
```
<phasetype="isotropic"/>

```
Copy to clipboard
```
'type': 'isotropic'

```
Copy to clipboard
## Henyey-Greenstein phase function (hg)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_phase.html#henyey-greenstein-phase-function-hg "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
g | float | This parameter must be somewhere in the range -1 to 1 (but not equal to -1 or 1). It denotes the _mean cosine_ of scattering interactions. A value greater than zero indicates that medium interactions predominantly scatter incident light into a similar direction (i.e. the medium is _forward-scattering_), whereas values smaller than zero cause the medium to be scatter more light in the opposite direction. | P, ∂, D  
This plugin implements the phase function model proposed by Henyey and Greenstein [[HG41](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id19 "L.G. Henyey and J.L. Greenstein. Diffuse radiation in the galaxy. The Astrophysical Journal, 93:70–83, 1941.")]. It is parameterizable from backward- (g<0) through isotropic- (g=0) to forward (g>0) scattering.
XMLPython
```
<phasetype="hg">
<floatname="g"value="0.1"/>
</phase>

```
Copy to clipboard
```
'type': 'hg',
'g': 0.1

```
Copy to clipboard
## SGGX phase function (sggx)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_phase.html#sggx-phase-function-sggx "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
S | volume | A volume containing the SGGX parameters. The phase function is parametrized by six values \\(S_{xx}\\), \\(S_{yy}\\), \\(S_{zz}\\), \\(S_{xy}\\), \\(S_{xz}\\) and \\(S_{yz}\\) (see below for their meaning). The parameters can either be specified as a [constvolume](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_volumes.html#volume-constvolume) with six values or as a [gridvolume](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_volumes.html#volume-gridvolume) with six channels. | P, ∂  
This plugin implements the SGGX phase function [[HDCD15](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id18 "Eric Heitz, Jonathan Dupuy, Cyril Crassin, and Carsten Dachsbacher. The sggx microflake distribution. ACM Trans. Graph. \(Proc. SIGGRAPH\), July 2015. URL: https://doi.org/10.1145/2766988, doi:10.1145/2766988.")]. The SGGX phase function is an anisotropic microflake phase function [[JMA+10](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id8 "Wenzel Jakob, Jonathan T. Moon, Adam Arbree, Kavita Bala, and Steve Marschner. A radiative transfer framework for rendering materials with anisotropic structure. ACM Trans. Graph. \(Proc. SIGGRAPH\), 29\(10\):53:1–53:13, July 2010. doi:10.1145/1778765.1778790.")]. This phase function can be useful to model fibers or surface-like structures using volume rendering. The SGGX distribution is the distribution of normals (NDF) of a 3D ellipsoid. It is parametrized by a symmetric, positive definite matrix \\(S\\).
Due to it’s symmetry, the matrix \\(S\\) is fully specified by providing the entries \\(S_{xx}\\), \\(S_{yy}\\), \\(S_{zz}\\), \\(S_{xy}\\), \\(S_{xz}\\) and \\(S_{yz}\\). It is the responsibility of the user to ensure that these parameters describe a valid positive definite matrix.
XMLPython
```
<phasetype='sggx'>
<volumetype="gridvolume"name="S">
<stringname="filename"value="volume.vol"/>
</volume>
</phase>

```
Copy to clipboard
```
'type': 'sggx',
'S': {
    'type': 'gridvolume',
    'filename': 'volume.vol'
}

```
Copy to clipboard
## Blended phase function (blendphase)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_phase.html#blended-phase-function-blendphase "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
weight | float or texture | A floating point value or texture with values between zero and one. The extreme values zero and one activate the first and second nested phase function respectively, and in-between values interpolate accordingly. (Default: 0.5) | P, ∂  
(Nested plugin) | phase | Two nested phase function instances that should be mixed according to the specified blending weight | P, ∂  
This plugin implements a _blend_ phase function, which represents linear combinations of two phase function instances. Any phase function in Mitsuba 3 (be it isotropic, anisotropic, micro-flake …) can be mixed with others in this manner. This is of particular interest when mixing components in a participating medium (_e.g._ accounting for the presence of aerosols in a Rayleigh-scattering atmosphere). The association of nested Phase plugins with the two positions in the interpolation is based on the alphanumeric order of their identifiers.
XMLPython
```
<phasetype="blendphase">
<floatname="weight"value="0.5"/>
<phasename="phase_0"type="isotropic"/>
<phasename="phase_1"type="hg">
<floatname="g"value="0.2"/>
</phase>
</phase>

```
Copy to clipboard
```
'type': 'blendphase',
'weight': 0.5,
'phase_0': {
    'type': 'isotropic'
},
'phase_1': {
    'type': 'hg',
    'g': 0.2
}

```
Copy to clipboard
## Lookup table phase function (tabphase)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_phase.html#lookup-table-phase-function-tabphase "Link to this heading")
Parameter | Type | Description | Flags  
---|---|---|---  
values | string | A comma-separated list of phase function values parametrized by the cosine of the scattering angle. | P, ∂, D  
This plugin implements a generic phase function model for isotropic media parametrized by a lookup table giving values of the phase function as a function of the cosine of the scattering angle.
Notes
  * The scattering angle cosine is here defined as the dot product of the incoming and outgoing directions, where the incoming, resp. outgoing direction points _toward_ , resp. _outward_ the interaction point.
  * From this follows that \\(\cos \theta = 1\\) corresponds to forward scattering.
  * Lookup table points are regularly spaced between -1 and 1.
  * Phase function values are automatically normalized.


## Rayleigh phase function (rayleigh)[¶](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_phase.html#rayleigh-phase-function-rayleigh "Link to this heading")
Scattering by particles that are much smaller than the wavelength of light (e.g. individual molecules in the atmosphere) is well-approximated by the Rayleigh phase function. This plugin implements an unpolarized version of this scattering model (_i.e._ the effects of polarization are ignored). This plugin is useful for simulating scattering in planetary atmospheres.
This model has no parameters.
XMLPython
```
<phasetype="rayleigh"/>

```
Copy to clipboard
```
'type': 'rayleigh'

```
Copy to clipboard
[**AI and large language models (LLMs)** are revolutionizing the way businesses use and process data.](https://server.ethicalads.io/proxy/click/9501/019a7c7e-66f2-7c71-a6ea-367d3fefef85/)
[Ads by EthicalAds](https://www.ethicalads.io/advertisers/?ref=ea-text)
* * *
  *[P]: This parameter will be exposed as a scene parameter
  *[∂]: This parameter is differentiable
  *[D]: This parameter might introduce discontinuities. Therefore it requires special handling during differentiation to prevent bias)
