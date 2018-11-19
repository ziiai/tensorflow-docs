### Available packages

*   `tensorflow` —Current release for CPU-only _(recommended for beginners)_
*   `tensorflow-gpu` —Current release with [GPU support](/docs/tensorflow/install/gpu) _(Ubuntu and Windows)_
*   `tf-nightly` —Nightly build for CPU-only _(unstable)_
*   `tf-nightly-gpu` —Nightly build with [GPU support](/docs/tensorflow/install/gpu) _(unstable, Ubuntu and Windows)_

### System requirements

*   Ubuntu 16.04 or later (64-bit)
*   macOS 10.12.6 (Sierra) or later (64-bit) _(no GPU support)_
*   Windows 7 or later (64-bit) _(Python 3 only)_
*   Raspbian 9.0 or later

### Hardware requirements

*   Starting with TensorFlow 1.6, binaries use [AVX instructions](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX) which may not run on older CPUs.
*   Read the [GPU support guide](/docs/tensorflow/install/gpu) to set up a CUDA®-enabled GPU card on Ubuntu or Windows.

1\. Install the Python development environment on your system
-------------------------------------------------------------

Check if your Python environment is already configured:

Python2

    python --version

Python3(Requires Python 3.4, 3.5, or 3.6)

    python3 --version

If these packages are already installed, skip to the next step.  
Otherwise, install [Python](https://www.python.org/), the [pip package manager](https://pip.pypa.io/en/stable/installing/), and [Virtualenv](https://virtualenv.pypa.io/en/stable/):

### Ubuntu

    sudo apt update

### mac OS

Install using the [Homebrew](https://brew.sh/) package manager:

    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

### Windows (Only for Python3)

Install the _Microsoft Visual C++ 2015 Redistributable Update 3_. This comes with _Visual Studio 2015_ but can be installed separately:

1.  Go to the [Visual Studio downloads](https://visualstudio.microsoft.com/vs/older-downloads/),
2.  Select _Redistributables and Build Tools_,
3.  Download and install the _Microsoft Visual C++ 2015 Redistributable Update 3_.

Install the _64-bit_ [Python 3 release for Windows](https://www.python.org/downloads/windows/) (select `pip` as an optional feature).

    pip3 install -U pip virtualenv

### Raspberry Pi

Requirements for the [Raspbian](https://www.raspberrypi.org/downloads/raspbian/) operating system:

    sudo apt update

### Other

    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

2\. Create a virtual environment (recommended)
----------------------------------------------

Python virtual environments are used to isolate package installation from the system.

### Ubuntu / mac OS

Create a new virtual environment by choosing a Python interpreter and making a `./venv` directory to hold it:

Python2

    virtualenv --system-site-packages -p python2.7 ./venv

Python3

    virtualenv --system-site-packages -p python3 ./venv


Activate the virtual environment using a shell-specific command:

    source ./venv/bin/activate  # sh, bash, ksh, or zsh

When virtualenv is active, your shell prompt is prefixed with `(venv)`.

Install packages within a virtual environment without affecting the host system setup. Start by upgrading `pip`:

    pip install --upgrade pip

And to exit virtualenv later:

    deactivate  # don't exit until you're done using TensorFlow

### Windows(Only for Python3)

Create a new virtual environment by choosing a Python interpreter and making a `./venv` directory to hold it:

    virtualenv --system-site-packages -p python3 ./venv

Activate the virtual environment:

    .\venv\Scripts\activate

Install packages within a virtual environment without affecting the host system setup. Start by upgrading `pip`:

    pip install --upgrade pip

And to exit virtualenv later:

    deactivate  # don't exit until you're done using TensorFlow

### Conda

While we recommend the TensorFlow-provided _pip_ package, a _community-supported_ [Anaconda package](https://anaconda.org/conda-forge/tensorflow) is available.

Create a new virtual environment by choosing a Python interpreter and making a `./venv` directory to hold it:

Python2

    conda create -n venv pip python=2.7

Python3

    conda create -n venv pip python=3.6  # select python version

Activate the virtual environment:

    source activate venv

Within the virtual environment, install the TensorFlow pip package using its [complete URL](#package-location):

    pip install --ignore-installed --upgrade packageURL

And to exit virtualenv later:

    source deactivate

3\. Install the TensorFlow pip package
--------------------------------------

Choose one of the following TensorFlow packages to install [from PyPI](https://pypi.org/project/tensorflow/):

*   `tensorflow` —Current release for CPU-only _(recommended for beginners)_
*   `tensorflow-gpu` —Current release with [GPU support](/docs/tensorflow/install/gpu) _(Ubuntu and Windows)_
*   `tf-nightly` —Nightly build for CPU-only _(unstable)_
*   `tf-nightly-gpu` —Nightly build with [GPU support](/docs/tensorflow/install/gpu) _(unstable, Ubuntu and Windows)_

Package dependencies are automatically installed. These are listed in the [`setup.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py) file under `REQUIRED_PACKAGES`.

### Virtualenv install

    pip install --upgrade tensorflow

Verify the install:

    python -c "import tensorflow as tf; print(tf.\_\_version\_\_)"

### System install

Python2

    pip install --user --upgrade tensorflow  # install in $HOME

Verify the install:

    python -c "import tensorflow as tf; print(tf.\_\_version\_\_)"

Python3

    pip3 install --user --upgrade tensorflow  # install in $HOME

Verify the install:

    python3 -c "import tensorflow as tf; print(tf.\_\_version\_\_)"

**Success:** TensorFlow is now installed. Read the [tutorials](/docs/tensorflow/tutorials) to get started.

<a id="package-location"></a>
## Package location

A few installation mechanisms require the URL of the TensorFlow Python package. The value you specify depends on your Python version.

<table>
  <tr><th>Version</th><th>URL</th></tr>
  <tr class="alt"><td colspan="2">Linux</td></tr>
  <tr>
    <td>Python 2.7 CPU-only</td>
    <td class="devsite-click-to-copy">https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.11.0-cp27-none-linux_x86_64.whl</td>
  </tr>
  <tr>
    <td>Python 2.7 GPU&nbsp;support</td>
    <td class="devsite-click-to-copy">https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.11.0-cp27-none-linux_x86_64.whl</td>
  </tr>
  <tr>
    <td>Python 3.4 CPU-only</td>
    <td class="devsite-click-to-copy">https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.11.0-cp34-cp34m-linux_x86_64.whl</td>
  </tr>
  <tr>
    <td>Python 3.4 GPU&nbsp;support</td>
    <td class="devsite-click-to-copy">https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.11.0-cp34-cp34m-linux_x86_64.whl</td>
  </tr>
  <tr>
    <td>Python 3.5 CPU-only</td>
    <td class="devsite-click-to-copy">https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.11.0-cp35-cp35m-linux_x86_64.whl</td>
  </tr>
  <tr>
    <td>Python 3.5 GPU&nbsp;support</td>
    <td class="devsite-click-to-copy">https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.11.0-cp35-cp35m-linux_x86_64.whl</td>
  </tr>
  <tr>
    <td>Python 3.6 CPU-only</td>
    <td class="devsite-click-to-copy">https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.11.0-cp36-cp36m-linux_x86_64.whl</td>
  </tr>
  <tr>
    <td>Python 3.6 GPU&nbsp;support</td>
    <td class="devsite-click-to-copy">https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.11.0-cp36-cp36m-linux_x86_64.whl</td>
  </tr>

  <tr class="alt"><td colspan="2">macOS (CPU-only)</td></tr>
  <tr>
    <td>Python 2.7</td>
    <td class="devsite-click-to-copy">https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.11.0-py2-none-any.whl</td>
  </tr>
  <tr>
    <td>Python 3.4, 3.5, 3.6</td>
    <td class="devsite-click-to-copy">https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.11.0-py3-none-any.whl</td>
  </tr>

  <tr class="alt"><td colspan="2">Windows</td></tr>
  <tr>
    <td>Python 3.5 CPU-only</td>
    <td class="devsite-click-to-copy">https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.11.0-cp35-cp35m-win_amd64.whl</td>
  </tr>
  <tr>
    <td>Python 3.5 GPU&nbsp;support</td>
    <td class="devsite-click-to-copy">https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.11.0-cp35-cp35m-win_amd64.whl</td>
  </tr>
  <tr>
    <td>Python 3.6 CPU-only</td>
    <td class="devsite-click-to-copy">https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.11.0-cp36-cp36m-win_amd64.whl</td>
  </tr>
  <tr>
    <td>Python 3.6 GPU&nbsp;support</td>
    <td class="devsite-click-to-copy">https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.11.0-cp36-cp36m-win_amd64.whl</td>
  </tr>
</table>
