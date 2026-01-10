---
url: https://mitsuba.readthedocs.io/en/stable/src/developer_guide/documentation.html
crawled_at: 2025-11-13T17:16:04.358337
title: https://mitsuba.readthedocs.io/en/stable/src/developer_guide/documentation.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/documentation.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/documentation.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Writing documentation[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/documentation.html#writing-documentation "Link to this heading")
Mitsuba uses a multi-stage documentation generation process that combines C++ docstring extraction, plugin documentation generation, and Sphinx-based HTML generation. This guide explains how the system works and how to build documentation.
## Prerequisites[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/documentation.html#prerequisites "Link to this heading")
Install required Python packages:
Copy to clipboard
## Documentation sources[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/documentation.html#documentation-sources "Link to this heading")
Documentation comes from several sources:
  1. **C++ headers** (`include/mitsuba/{core,render}/*.h`): API documentation extracted via docstrings
  2. **C++ plugin sources** (`src/{bsdfs,shapes,emitters,...}/*.cpp`): Plugin descriptions and parameters
  3. **RST files** (`docs/src/`): User guides, tutorials, and manual content
  4. **Jupyter notebooks** (`docs/tutorials/`): Interactive tutorials rendered with nbsphinx


## Build process overview[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/documentation.html#build-process-overview "Link to this heading")
The complete documentation build requires multiple steps in a specific order:
```
# Extract C++ docstrings → include/mitsuba/python/docstr.h
ninja# Build main library and Python bindings
ninja# Generate API reference documentation
ninja# Build final HTML documentation

```
Copy to clipboard
## Detailed build steps[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/documentation.html#detailed-build-steps "Link to this heading")
  1. **Docstring extraction** (`ninja docstrings`): Parses C++ headers in `include/mitsuba/` using `pybind11_mkdoc` to generate `include/mitsuba/python/docstr.h` for Python bindings.
  2. **Main build** (`ninja`): Compiles the C++ library, plugins, and Python bindings with embedded docstrings—required before generating API documentation.
  3. **API documentation** (`ninja mkdoc-api`): Introspects Python modules to generate API reference in `build/html_api/`.
  4. **Main documentation** (`ninja mkdoc`): Builds the complete documentation website in `build/html/` by running plugin extraction, processing notebooks, and combining all sources.


## Notebook tutorials[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/documentation.html#notebook-tutorials "Link to this heading")
We are using the [nbsphinx](https://nbsphinx.readthedocs.io/) Sphinx extension to render our tutorials in the online documentation.
The thumbnail of a notebook in the gallery can be the output image of a cell in the notebook. For this, simply add the following to the metadata of that cell:
```
{
"nbsphinx-thumbnail":{}
}

```
Copy to clipboard
In order to hide a cell of a notebook in the documentation, add the following to the metadata of that cell:
```
{
"nbsphinx":"hidden"
}

```
Copy to clipboard
* * *
