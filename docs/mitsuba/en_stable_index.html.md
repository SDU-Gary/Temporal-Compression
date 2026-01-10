---
url: https://mitsuba.readthedocs.io/en/stable/index.html
crawled_at: 2025-11-13T17:11:33.981994
title: https://mitsuba.readthedocs.io/en/stable/index.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/index.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[![_images/banner_01.jpg](https://mitsuba.readthedocs.io/en/stable/_images/banner_01.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/banner_01.jpg)
# Getting started[¶](https://mitsuba.readthedocs.io/en/stable/index.html#getting-started "Link to this heading")
Mitsuba 3 is a research-oriented rendering system for forward and inverse light-transport simulation. It consists of a small set of core libraries and a wide variety of plugins that implement functionality ranging from materials and light sources to complete rendering algorithms. Mitsuba 3 strives to retain scene compatibility with its predecessors: [Mitsuba 0.6](https://github.com/mitsuba-renderer/mitsuba) and [Mitsuba 2](https://github.com/mitsuba-renderer/mitsuba2). However, in most other respects, it is a completely new system following a different set of goals.
## Installation[¶](https://mitsuba.readthedocs.io/en/stable/index.html#installation "Link to this heading")
Mitsuba 3 can be installed via pip from [PyPI](https://pypi.org/project/mitsuba/). This is the recommended method of installation.
Copy to clipboard
This command will also install Dr.Jit on your system if not already available.
See the [developer guide](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html#sec-compiling) for complete instructions on building from the git source tree.
When using the [Windows Subsystem for Linux 2 (WSL2)](https://learn.microsoft.com/en-us/windows/wsl/compare-versions#whats-new-in-wsl-2), you must follow the [linked instructions](https://mitsuba.readthedocs.io/en/stable/src/optix_setup.html#optix-wsl2) to enable hardware-accelerated ray tracing on NVIDIA GPUs.
### Requirements[¶](https://mitsuba.readthedocs.io/en/stable/index.html#requirements "Link to this heading")
  * `Python >= 3.8`
  * (optional) For computation on the GPU: `NVidia driver >= 535`
  * (optional) For vectorized / parallel computation on the CPU: `LLVM >= 11.1`


## Hello World![¶](https://mitsuba.readthedocs.io/en/stable/index.html#hello-world "Link to this heading")
You should now be all setup to render your first scene with Mitsuba 3. Running the Mitsuba 3 code below will render the famous Cornell Box scene and write the rendered image to a file on disk.
```
importmitsubaasmi

mi.set_variant('scalar_rgb')

img = mi.render(mi.load_dict(mi.cornell_box()))

mi.Bitmap(img).write('cbox.exr')

```
Copy to clipboard
## Quickstart[¶](https://mitsuba.readthedocs.io/en/stable/index.html#quickstart "Link to this heading")
For the new users, we put together absolute beginner’s tutorials for both Dr.Jit and Mitsuba.
Dr.Jit quickstart
[![_images/drjit-logo.png](https://mitsuba.readthedocs.io/en/stable/_images/drjit-logo.png) ](https://mitsuba.readthedocs.io/en/stable/_images/drjit-logo.png)
[src/quickstart/drjit_quickstart.html](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html)
Mitsuba quickstart
[![_images/mitsuba-logo.png](https://mitsuba.readthedocs.io/en/stable/_images/mitsuba-logo.png) ](https://mitsuba.readthedocs.io/en/stable/_images/mitsuba-logo.png)
[src/quickstart/mitsuba_quickstart.html](https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html)
## Video tutorials[¶](https://mitsuba.readthedocs.io/en/stable/index.html#video-tutorials "Link to this heading")
The following [YouTube playlist](https://www.youtube.com/playlist?list=PLI9y-85z_Po6da-pyTNGTns2n4fhpbLe5) contains various video tutorials related to Mitsuba 3 and Dr.Jit, perfect to get you started with those two libraries.
## Citation[¶](https://mitsuba.readthedocs.io/en/stable/index.html#citation "Link to this heading")
When using Mitsuba 3 in academic projects, please cite:
```
@software{jakob2022mitsuba3,
title={Mitsuba 3 renderer},
author={Wenzel Jakob and Sébastien Speierer and Nicolas Roussel and Merlin Nimier-David and Delio Vicini and Tizian Zeltner and Baptiste Nicolet and Miguel Crespo and Vincent Leroy and Ziyi Zhang},
note={https://mitsuba-renderer.org},
version={3.0.1},
year=2022,
}

```
Copy to clipboard
* * *
