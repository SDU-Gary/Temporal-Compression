---
url: https://mitsuba.readthedocs.io/en/stable/src/optix_setup.html
crawled_at: 2025-11-13T17:10:19.165486
title: https://mitsuba.readthedocs.io/en/stable/src/optix_setup.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/optix_setup.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/optix_setup.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Mitsuba on WSL 2[¶](https://mitsuba.readthedocs.io/en/stable/src/optix_setup.html#mitsuba-on-wsl-2 "Link to this heading")
Mitsuba uses the [NVIDIA OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) framework for hardware-accelerated ray tracing. While OptiX is not yet officially supported on the [Windows Subsystem for Linux 2 (WSL 2)](https://learn.microsoft.com/en-us/windows/wsl/compare-versions#whats-new-in-wsl-2), it _is_ possible to get it to run in practice. The following instructions are based on an NVIDIA [forum post by @dhart](https://forums.developer.nvidia.com/t/problem-running-optix-7-6-in-wsl/239355/8) Use them at your own risk.
  * Determine your driver version using the `nvidia-smi` command. This is a number such as `560.94` shown in the middle of the first row.
  * Go to the [NVIDA driver webpage](https://www.nvidia.com/en-us/drivers) to download a similar driver version for Linux (64 bit). You may have to click on the “New feature branch” tab to find newer driver versions.
  * Following this step, you should have file named `NVIDIA-Linux-x86_64-*.run`. Move it to your WSL home directory but _do not install it_. Instead, merely extract its contents within a WSL session using the following command:


```
$ bash
```
Copy to clipboard
Create a symbolic link that exposes the already installed CUDA driver to runtime loading:
```
$ ln
```
Copy to clipboard
Next, copy-paste and run the following command:
```
$ mkdir&&&&&&&&&&&&&&"C:\Windows\System32\lxss\lib"

```
Copy to clipboard
This will open two Explorer windows: one to a system path containing internal WSL driver files (`C:\Windows\System32\lxss\lib`), and another to a newly created `driver-dist` directory containing files that need to be copied to `C:\Windows\System32\lxss\lib`. Perform this copy manually using Explorer and overwrite existing files if present. It will warn you that this is dangerous, and you will need to give permission.
Close all WSL windows, and enter the following command in a `cmd.exe` or PowerShell session:
```
C:\Users\...> wsl --shutdown

```
Copy to clipboard
Following this, OptiX should be usable within WSL.
Warning
Using CUDA and OptiX through WSL degrades performance. Please do not collect performance data within WSL, since the results will not be representative.
[**GenAI apps + MongoDB Atlas** You don't need a separate database to start building GenAI-powered apps.](https://server.ethicalads.io/proxy/click/9508/019a7c7a-d0a0-7b81-8950-8fef81e775c9/)
[Ads by EthicalAds](https://www.ethicalads.io/advertisers/topics/data-science/?ref=ea-text)
* * *
