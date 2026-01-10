---
url: https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html
crawled_at: 2025-11-13T17:10:43.008195
title: https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Compiling the system[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html#compiling-the-system "Link to this heading")
## Cloning the repository[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html#cloning-the-repository "Link to this heading")
Compiling Mitsuba 3 from scratch requires recent versions of CMake (at least **3.9.0**) and Python (at least **3.8**). Further platform-specific dependencies and compilation instructions are provided below for each operating system. Some additional steps are required for GPU-based backends that are described at the end of this section.
Mitsuba depends on several external dependencies, and its repository directly refers to specific versions of them using a Git feature called _submodules_. Cloning Mitsuba’s repository will recursively fetch these dependencies, which are subsequently compiled using a single unified build system. This dramatically reduces the number steps needed to set up the renderer compared to previous versions of Mitsuba.
Most of Mitsuba’s active development happens on the `master` Git branch. We therefore recommend using the `stable` branch which points to the most recent release.
For all of this to work out properly, you will have to specify the `--recursive` flag when cloning the repository:
Copy to clipboard
If you already cloned the repository and forgot to specify this flag, it’s possible to fix the repository in retrospect using the following command:
Copy to clipboard
**Staying up-to-date**
Unfortunately, pulling from the main repository won’t automatically keep the submodules in sync, which can lead to various problems. The following command installs a git alias named `pullall` that automates these two steps.
```
'!f(){ git pull "$@" && git submodule update --init --recursive; }; f'

```
Copy to clipboard
Afterwards, simply write
Copy to clipboard
to fetch the latest version of Mitsuba 3.
## Configuring mitsuba.conf[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html#configuring-mitsuba-conf "Link to this heading")
Mitsuba 3 variants are specified in the file mitsuba.conf. This file can be found in the build directory and will be created when executing CMake the first time.
Open mitsuba.conf in your favorite text editor and scroll down to the declaration of the enabled variants (around line 86):
Copy to clipboard
The default file specifies two scalar variants that you may wish to extend according to your requirements and the explanations given above. Note that `scalar_spectral` can be removed, but `scalar_rgb` _must_ currently be part of the list as some core components of Mitsuba depend on it. In addition, at least one `ad`-enabled variant must also be compiled. If Mitsuba is launched from the command line without any specific mode parameter, the first variant of the list below will be used.
You may also wish to change the _Python default_ variant that is executed if no variant is explicitly specified (this must be one of the entries of the `enabled` list):
Copy to clipboard
The remainder of this file lists the C++ types defining the available variants and can safely be ignored.
TLDR: If you plan to use Mitsuba from Python, we recommend adding one of `llvm_ad_rgb` or `llvm_ad_spectral` for CPU rendering, or one of `cuda_ad_rgb` or `cuda_ad_spectral` for differentiable GPU rendering.
Warning
Note that compilation time and compilation memory usage is roughly proportional to the number of enabled variants, hence including many of them (more than five) may not be advisable. Also note that the `scalar_rgb` and _at least one AD variant_ is mandatory.
Warning
Mitsuba 3 also generates corresponding [Python stub files](https://typing.readthedocs.io/en/latest/spec/distributing.html#stub-files) during compilation. The process involves selecting one of the available variants to extract the relevant type information. However, these stub files have to be variant-agnostic and hence certain combinations of variants won’t be allowed. For example, including just `scalar_rgb`, `scalar_spectral` and `llvm_ad_rgb` creates ambiguity as to which variant we should select to generate the Python stubs. In short, if a disallowed combination of variants is selected, a compilation error will report what variant should be added to remove any ambiguity.
## Linux[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html#linux "Link to this heading")
The build process under Linux requires several external dependencies that are easily installed using the system-provided package manager (e.g., apt-get under Ubuntu).
Note that recent Linux distributions include two different compilers that can both be used for C++ software development. [GCC](https://gcc.gnu.org) is typically the default, and [Clang](https://clang.llvm.org) can be installed optionally. During the development of this project, we encountered many issues with GCC (mis-compilations, compiler errors, segmentation faults), and strongly recommend that you use Clang instead.
To fetch all dependencies and Clang, enter the following commands on Ubuntu:
```
# Install recent versions build tools, including Clang
sudo# Install libraries for image I/O
sudo# Install required Python packages
sudo
```
Copy to clipboard
Additional packages are required to run the included test suite or to generate HTML documentation (see [Developer guide](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/documentation.html#sec-writing-documentation)). If those are interesting to you, also enter the following commands:
```
# For running tests
sudo
```
Copy to clipboard
Next, ensure that two environment variables CC and CXX are exported. You can either run these two commands manually before using CMake or—even better—add them to your ~/.bashrc file. This ensures that CMake will always use the correct compiler.
```
exportCC=clang-17exportCXX=clang++-17

```
Copy to clipboard
If you installed another version of Clang, the version suffix of course has to be adjusted. Now, compilation should be as simple as running the following from inside the mitsuba3 root directory:
```
# Create a directory where build products are stored
mkdircd
```
Copy to clipboard
**Tested versions**
The above procedure will likely work on many different flavors of Linux (with slight adjustments for the package manager and package names). We have mainly worked with software environments listed below, and our instructions should work without modifications in those cases.
**Focal**
  * Ubuntu 20.04
  * g++ 9.4.0
  * cmake 3.16.3
  * ninja 1.10.0
  * python 3.8.10

|  **Jammy**
  * Ubuntu 22.04
  * clang 17.0.6
  * cmake 3.22.1
  * ninja 1.10.1
  * python 3.10.12

|  **Noble**
  * Ubuntu 24.04
  * g++ 13.2.0
  * cmake 3.28.3
  * ninja 1.11.1
  * python 3.12.3

  
---|---|---  
## Windows[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html#windows "Link to this heading")
On Windows, a recent version of [Visual Studio 2022](https://visualstudio.microsoft.com/vs/) is required. Some tools such as git, CMake, or Python might need to be installed manually. Mitsuba’s build system _requires_ access to Python >= 3.8 even if you do not plan to use Mitsuba’s python interface.
From the root `mitsuba3` directory, the build can be configured with:
```
# To be safe, explicitly ask for the 64 bit version of Visual Studio
cmake"Visual Studio 17 2022"
```
Copy to clipboard
Afterwards, open the generated `mitsuba.sln` file in the build folder and proceed building as usual from within Visual Studio. You will probably also want to set the build mode to _Release_ there.
It is also possible to directly build from the terminal running the following command:
Copy to clipboard
**Tested version**
  * Windows 10
  * Visual Studio 17 2022 (Community Edition)
  * MSVC 19.41.34123.0
  * cmake 3.28.1 (64bit)
  * git 2.34.1 (64bit)
  * Python 3.11.1 (64bit)


## macOS[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html#macos "Link to this heading")
On macOS, you will need to install Xcode, CMake, and [Ninja](https://ninja-build.org/). Additionally, running the Xcode command line tools once might be necessary:
Copy to clipboard
Note that the default Python version installed with macOS is not compatible with Mitsuba 3, and a more recent version (at least 3.8) needs to be installed (e.g. via [Miniconda 3](https://docs.conda.io/en/latest/miniconda.html) or [Homebrew](https://brew.sh/)).
Now, compilation should be as simple as running the following from inside the `mitsuba3` root directory:
```
cd
```
Copy to clipboard
**Tested version**
  * macOS Big Sur 11.5.2
  * AppleClang 13.2.0.0.1.1638488800
  * Xcode 12.0.5
  * cmake 3.24.2
  * Python 3.9.5


## Running Mitsuba[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html#running-mitsuba "Link to this heading")
Once Mitsuba is compiled, run the `setpath.sh/.bat/.ps1` script in your build directory to configure environment variables (`PATH/PYTHONPATH`) that are required to run Mitsuba.
```
# On Linux / Mac OS
source# On Windows (cmd)
C:/.../mitsuba3/build/Release># On Windows (powershell)
C:/.../mitsuba3/build/Release>\setpath.ps1

```
Copy to clipboard
Mitsuba can then be used to render scenes by typing
Copy to clipboard
where `scene.xml` is a Mitsuba scene file. Alternatively,
Copy to clipboard
renders with a specific variant that was previously enabled in mitsuba.conf. Call `mitsuba --help` to print additional information about the various possible command line options.
## GPU variants[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html#gpu-variants "Link to this heading")
Variants of Mitsuba that run on the GPU (e.g. cuda_rgb, cuda_ad_spectral, etc.) will try to dynamically load the CUDA shared libraries from your system. There is no need to manually install any specific version of CUDA.
Make sure to have an up-to-date GPU driver if the framework fails to compile the GPU variants of Mitsuba. The minimum requirement is currently v535.
By default, Mitsuba is also able to resolve the OptiX API itself, and therefore does not rely on the `optix.h` header file. The `MI_USE_OPTIX_HEADERS` CMake flag can be used to turn off this feature if a developer wants to experiment with parts of the OptiX API not yet exposed to the framework.
## Embree[¶](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html#embree "Link to this heading")
By default, Mitsuba’s `scalar` and `llvm` backends use Intel’s Embree library for ray tracing instead of the builtin kd-tree in Mitsuba 3. To change this behavior, invoke CMake with the `-DMI_ENABLE_EMBREE=0` parameter or use a visual CMake tool like `cmake-gui` or `ccmake` to flip the value of this parameter. Embree tends to be faster but lacks some features such as support for double precision ray intersection.
* * *
