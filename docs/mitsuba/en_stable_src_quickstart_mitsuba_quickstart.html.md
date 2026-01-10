---
url: https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html
crawled_at: 2025-11-13T17:14:22.579875
title: https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Download notebook ](https://raw.githubusercontent.com/mitsuba-renderer/mitsuba-tutorials/master/quickstart/mitsuba_quickstart.ipynb) [ Download data ](https://d38rqfq1h7iukm.cloudfront.net/scenes/tutorials/scenes.zip)
# Mitsuba quickstart[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html#Mitsuba-quickstart "Link to this heading")
## Overview[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html#Overview "Link to this heading")
In this tutorial, you will render your very first image using Mitsuba 3!
🚀 **You will learn how to:**
  * Import Mitsuba in Python and set the “variant”
  * Load a scene from disk
  * Render a scene
  * Write a rendered image to disk


## Importing Mitsuba[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html#Importing-Mitsuba "Link to this heading")
After [installing](https://mitsuba.readthedocs.io/en/latest/index.html#installation) the Mitsuba Python module, it can be imported as follows:
Copy to clipboard
```
importmitsubaasmi

```
Copy to clipboard
We use the short alias `mi` for `mitsuba` to improve code readability.
Mitsuba is a _retargetable_ system that supports a variety of different computational backends (e.g., GPU, CPU), color representations (e.g., RGB, spectral or polarized) and floating point precisions (single and double precision). We call a specific combination of these attributes a _variant_ of the renderer. For more information about the supported variants, please refer to the [dedicated section](https://mitsuba.readthedocs.io/en/latest/src/key_topics/variants.html) in the documentation.
In brief the system implements the following computational backends:
  * `scalar`: runs on CPU, using normal floating point arithmetic, processing individual rays at a time
  * `llvm`: runs on CPU, automatically parallelized over cores and vector units
  * `cuda`: runs on NVidia GPU in parallel


Under the hood, we essentially compile a different renderer for each variant. Therefore, using any components of the system requires first setting the desired variant using the [set_variant()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.set_variant) function. Only then function calls or class instantiations can be routed to a specific underlying implementation. For most use cases, it is sufficient to set the variant once at the beginning of the program. For example, if your project consists of a `my_script.py` file with some helper functions in a `my_utils.py`, you most likely only want to specify the variant at the beginning of the execution in `my_script.py`. It is possible to switch the variant at any point of the execution, but this should typically not be necessary. Plugins and objects created using different variants are not compatible (e.g., it’s not possible to load a scene in a GPU variant and then render it on the CPU).
Using the [variants()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.variants) function, it is possible to list all Mitsuba variants installed on your system:
Copy to clipboard
```
mi.variants()

```
Copy to clipboard
Copy to clipboard
```
['scalar_rgb', 'llvm_ad_rgb']

```
Copy to clipboard
For this tutorial, we will use the simplest variant: `scalar_rgb`. As the name implies, computations will be performed in a scalar-fashion (i.e., not vectorized) on the CPU and the light transport simulation will operate on RGB color values. The code in the rest of the tutorial is agnostic to the variant and could for example also be run using a GPU variant of the system. We will discuss these and other variants in later tutorials.
Copy to clipboard
```
mi.set_variant("scalar_rgb")

```
Copy to clipboard
📑 **Note**
If your are new to the Mitsuba Python API, is it important to remember that you can use the `help()` function on Mitsuba classes and functions to read the well-documented API reference. Additionally, Mitsuba 3 has support for autocomplete with most modern IDEs to make things easier for new users.
## Loading a scene[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html#Loading-a-scene "Link to this heading")
In this tutorial, we are going to load a Mitsuba scene from a file on disk. Mitsuba scenes are described using a simple and general [XML-based format](https://mitsuba.readthedocs.io/en/latest/src/key_topics/scene_format.html).
A [few example scenes](https://rgl.s3.eu-central-1.amazonaws.com/scenes/tutorials/scenes.zip) can be downloaded clicking the “Download data” button at the top of the tutorial page. We can then load a scene from an XML file using the [load_file()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.load_file) function:
Copy to clipboard
```
scene = mi.load_file("../scenes/cbox.xml")

```
Copy to clipboard
## Rendering a scene[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html#Rendering-a-scene "Link to this heading")
Once loaded into memory, a scene can be rendered using the [render()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.render) function. The `render()` function has a variety of optional arguments. We can for example pass the desired number of samples per pixel (SPP).
Copy to clipboard
```
image = mi.render(scene, spp=256)

```
Copy to clipboard
The render function returns the generated image as a tensor ([mi.TensorXf](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.TensorXf), similar to a NumPy array) in linear RGB color space. The tensor class interfaces seamlessly with functions expecting NumPy arrays. For example, we can display the image using `matplotlib`.
Copy to clipboard
```
importmatplotlib.pyplotasplt

plt.axis("off")
plt.imshow(image ** (1.0 / 2.2)); # approximate sRGB tonemapping

```
Copy to clipboard
```
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

```
Copy to clipboard
![../../_images/src_quickstart_mitsuba_quickstart_14_1.png](https://mitsuba.readthedocs.io/en/stable/_images/src_quickstart_mitsuba_quickstart_14_1.png)
## Writing an image to file[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html#Writing-an-image-to-file "Link to this heading")
The [mi.util.write_bitmap()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.util.write_bitmap) function allows to save an image to disk and supports multiple file formats. If a low dynamic range (LDR) format is selected (e.g., PNG), this function tonemaps the image to the sRGB color space before saving.
Copy to clipboard
```
mi.util.write_bitmap("my_first_render.png", image)
mi.util.write_bitmap("my_first_render.exr", image)

```
Copy to clipboard
Bravo! 🎉 You have now successfully rendered your first image!
## See also[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html#See-also "Link to this heading")
  * [mitsuba.set_variant()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.set_variant)
  * [mitsuba.load_file()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.load_file)
  * [mitsuba.render()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.render)
  * [mitsuba.util.write_bitmap()](https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.Bitmap.write)


[**AI and large language models (LLMs)** are revolutionizing the way businesses use and process data.](https://server.ethicalads.io/proxy/click/9501/019a7c7e-66f2-7c71-a6ea-367d3fefef85/)
[Ads by EthicalAds](https://www.ethicalads.io/advertisers/?ref=ea-text)
* * *
