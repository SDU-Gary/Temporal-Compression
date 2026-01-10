---
url: https://mitsuba.readthedocs.io/en/stable/src/developer_guide.html
crawled_at: 2025-11-13T17:14:05.004674
title: https://mitsuba.readthedocs.io/en/stable/src/developer_guide.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/developer_guide.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/developer_guide.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[![../_images/banner_05.jpg](https://mitsuba.readthedocs.io/en/stable/_images/banner_05.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/banner_05.jpg)
# Developer’s Guide[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide.html#developer-s-guide "Link to this heading")
## Overview[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide.html#overview "Link to this heading")
This section is addressed to the users interested in modifying the core of the system or even contributing to the codebase.
New developers will want to begin by thoroughly reading the documentation of Dr.Jit before looking at any Mitsuba code. Dr.Jit is a Just-In-Time compiler that constitutes the foundation of Mitsuba 3. It drives the code transformations that enable systematic vectorization and automatic differentiation of the renderer.
## Code structure[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide.html#code-structure "Link to this heading")
The Mitsuba codebase is split into 3 basic support folders:
  * The core folder (in `src/core`) implements basic functionality such as cross-platform file and bitmap I/O, data structures, scheduling, as well as logging and plugin management.
  * The rendering folder (in `src/render`) contains abstractions needed to load and represent scenes containing light sources, shapes, materials, and participating media.
  * The python folder (in `src/python`) contains components of the system that are written in Python, and which access Mitsuba through bindings. This includes statistical tests (Chi^2, etc.) and tooling for differentiable rendering.


All other folders in `src` implement Mitsuba 3 plugins such as `bsdf`, `shapes`, etc.
## Coding style[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide.html#coding-style "Link to this heading")
We’ve essentially imported Python’s [PEP 8](https://peps.python.org/pep-0008/) into the C++ side (which does not specify a recommended naming convention), ensuring that code that uses functionality from both languages looks natural.
## Contributing[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide.html#contributing "Link to this heading")
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome. If you are brand new to Mitsuba or open-source development, we recommend going through the GitHub “issues” tracker to find issues that interest you.
## Going further[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide.html#going-further "Link to this heading")
  * [Compiling the system](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html)
  * [Writing documentation](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/documentation.html)
  * [Variants in C++](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/variants_cpp.html)
  * [C++ Plugins & Macros](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/writing_plugin.html)
  * [Testing](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/testing.html)


* * *
