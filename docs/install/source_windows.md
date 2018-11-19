# Build from source on Windows

Build a TensorFlow *pip* package from source and install it on Windows.

Note: We already provide well-tested, pre-built [TensorFlow packages](/docs/tensorflow/install/pip)
for Windows systems.

## Setup for Windows

Install the following build tools to configure your Windows development
environment.

### Install Python and the TensorFlow package dependencies

Install a
[Python 3.5.x or Python 3.6.x 64-bit release for Windows](https://www.python.org/downloads/windows/).
Select *pip* as an optional feature and add it to your `%PATH%` environmental variable.

Install the TensorFlow *pip* package dependencies:

    pip3 install six numpy wheel    
    pip3 install keras_applications==1.0.5 --no-deps    
    pip3 install keras_preprocessing==1.0.3 --no-deps    


The dependencies are listed in the
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py">`setup.py`</a>
file under `REQUIRED_PACKAGES`.

### Install Bazel

[Install Bazel](https://docs.bazel.build/versions/master/install-windows.html),
the build tool used to compile TensorFlow.

Add the location of the Bazel executable to your `%PATH%` environment variable.

### Install MSYS2

[Install MSYS2](https://www.msys2.org/) for the bin tools needed to
build TensorFlow. If MSYS2 is installed to `C:\msys64`, add
`C:\msys64\usr\bin` to your `%PATH%` environment variable. Then, using `cmd.exe`,
run:
    
    pacman -S git patch unzip

### Install Visual C++ Build Tools 2015

Install the *Visual C++ build tools 2015*. This comes with *Visual Studio 2015*
but can be installed separately:

1. Go to the [Visual Studio downloads](https://visualstudio.microsoft.com/vs/older-downloads/),
2. Select *Redistributables and Build Tools*,
3. Download and install:
   - *Microsoft Visual C++ 2015 Redistributable Update 3*
   - *Microsoft Build Tools 2015 Update 3*

Note: TensorFlow is tested against the *Visual Studio 2015 Update 3*.

### Install GPU support (optional)

See the Windows [GPU support](/docs/tensorflow/install/gpu) guide to install the drivers and additional
software required to run TensorFlow on a GPU.


### Download the TensorFlow source code

Use [Git](https://git-scm.com/) to clone the
[TensorFlow repository](https://github.com/tensorflow/tensorflow)
(`git` is installed with MSYS2):

    git clone https://github.com/tensorflow/tensorflow.git    
    cd tensorflow    

The repo defaults to the `master` development branch. You can also checkout a
[release branch](https://github.com/tensorflow/tensorflow/releases)
to build:

    git checkout branch_name  # r1.9, r1.10, etc.

Key Point: If you're having build problems on the latest development branch, try
a release branch that is known to work.


## Configure the build

Configure your system build by running the following at the root of your
TensorFlow source tree:
    
    python ./configure.py


This script prompts you for the location of TensorFlow dependencies and asks for
additional build configuration options (compiler flags, for example). The
following shows a sample run of `python ./configure.py` (your session may differ):

#### View sample configuration session

    python ./configure.py
    Starting local Bazel server and connecting to it...
    ................
    You have bazel 0.15.0 installed.
    Please specify the location of python. [Default is C:\python36\python.exe]: 
    
    Found possible Python library paths:
      C:\python36\lib\site-packages
    Please input the desired Python library path to use.  Default is [C:\python36\lib\site-packages]
    
    Do you wish to build TensorFlow with CUDA support? [y/N]: <b>Y</b>
    CUDA support will be enabled for TensorFlow.
    
    Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]:
    
    Please specify the location where CUDA 9.0 toolkit is installed. Refer to README.md for more details. [Default is C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0]:
    
    Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: <b>7.0</b>
    
    Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0]: <b>C:\tools\cuda</b>
    
    Please specify a list of comma-separated Cuda compute capabilities you want to build with.
    You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
    Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 3.5,7.0]: <b>3.7</b>
    
    Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is /arch:AVX]: 
    
    Would you like to override eigen strong inline for some C++ compilation to reduce the compilation time? [Y/n]:
    Eigen strong inline overridden.
    
    Configuration finished

### Configuration options

For [GPU support](/docs/tensorflow/install/gpu), specify the versions of CUDA and cuDNN. If your
system has multiple versions of CUDA or cuDNN installed, explicitly set the
version instead of relying on the default. `./configure.py` creates symbolic links
to your system's CUDA libraries—so if you update your CUDA library paths, this
configuration step must be run again before building.

Note: Starting with TensorFlow 1.6, binaries use AVX instructions which may not
run on older CPUs.


## Build the pip package

### Bazel build

#### CPU-only

Use `bazel` to make the TensorFlow package builder with CPU-only support:
    
    bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

#### GPU support

To make the TensorFlow package builder with GPU support:
    
    bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

#### Bazel build options

Building TensorFlow from source can use a lot of RAM. If your system is
memory-constrained, limit Bazel's RAM usage with: `--local_resources 2048,.5,1.0`.

If building with GPU support, add `--copt=-nvcc_options=disable-warnings`
to suppress nvcc warning messages.

### Build the package

The `bazel build` command creates an executable named `build_pip_package`—this
is the program that builds the `pip` package. For example, the following builds a
`.whl` package in the `C:/tmp/tensorflow_pkg` directory:
    
    bazel-bin\tensorflow\tools\pip_package\build_pip_package C:/tmp/tensorflow_pkg

Although it is possible to build both CUDA and non-CUDA configs under the
same source tree, we recommend running `bazel clean` when switching between
these two configurations in the same source tree.

### Install the package

The filename of the generated `.whl` file depends on the TensorFlow version and
your platform. Use `pip3 install` to install the package, for example:

    pip3 install C:/tmp/tensorflow_pkg/tensorflow-<var>version</var>-cp36-cp36m-win_amd64.whl

Success: TensorFlow is now installed.


## Build using the MSYS shell

TensorFlow can also be built using the MSYS shell. Make the changes listed
below, then follow the previous instructions for the Windows native command line
(`cmd.exe`).

### Disable MSYS path conversion

MSYS automatically converts arguments that look like Unix paths to Windows paths,
and this doesn't work with `bazel`. (The label `//foo/bar:bin` is considered a
Unix absolute path since it starts with a slash.)

    export MSYS_NO_PATHCONV=1
    export MSYS2_ARG_CONV_EXCL="*"

### Set your PATH {:.hide-from-toc}

Add the Bazel and Python installation directories to your `$PATH` environmental
variable. If Bazel is installed to `C:\tools\bazel.exe`, and Python to
`C:\Python36\python.exe`, set your `PATH` with:

    # Use Unix-style with ':' as separator
    export PATH="/c/tools:$PATH"
    export PATH="/c/Python36:$PATH"

For GPU support, add the CUDA and cuDNN bin directories to your `$PATH`:

    export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0/bin:$PATH"    
    export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0/extras/CUPTI/libx64:$PATH"    
    export PATH="/c/tools/cuda/bin:$PATH"    

## Tested build configurations

### CPU

<table>
<tr><th>Version</th><th>Python version</th><th>Compiler</th><th>Build tools</th></tr>
<tr><td>tensorflow-1.11.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td></tr>
<tr><td>tensorflow-1.10.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td></tr>
<tr><td>tensorflow-1.9.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td></tr>
<tr><td>tensorflow-1.8.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td></tr>
<tr><td>tensorflow-1.7.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td></tr>
<tr><td>tensorflow-1.6.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td></tr>
<tr><td>tensorflow-1.5.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td></tr>
<tr><td>tensorflow-1.4.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td></tr>
<tr><td>tensorflow-1.3.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td></tr>
<tr><td>tensorflow-1.2.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td></tr>
<tr><td>tensorflow-1.1.0</td><td>3.5</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td></tr>
<tr><td>tensorflow-1.0.0</td><td>3.5</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td></tr>
</table>

### GPU

<table>
<tr><th>Version</th><th>Python version</th><th>Compiler</th><th>Build tools</th><th>cuDNN</th><th>CUDA</th></tr>
<tr><td>tensorflow_gpu-1.11.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Bazel 0.15.0</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow_gpu-1.10.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow_gpu-1.9.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow_gpu-1.8.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow_gpu-1.7.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow_gpu-1.6.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow_gpu-1.5.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow_gpu-1.4.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>6</td><td>8</td></tr>
<tr><td>tensorflow_gpu-1.3.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>6</td><td>8</td></tr>
<tr><td>tensorflow_gpu-1.2.0</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>5.1</td><td>8</td></tr>
<tr><td>tensorflow_gpu-1.1.0</td><td>3.5</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>5.1</td><td>8</td></tr>
<tr><td>tensorflow_gpu-1.0.0</td><td>3.5</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>5.1</td><td>8</td></tr>
</table>
