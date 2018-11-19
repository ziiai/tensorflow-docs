# 通过 pip 安装 TensorFlow

### 可用的安装包

*   `tensorflow` — 仅支持 CPU 的当前发布版本 _(推荐初学者使用)_
*   `tensorflow-gpu` — [支持 GPU](/docs/tensorflow/install/gpu) 的当前版本 _(Ubuntu and Windows)_
*   `tf-nightly` — 仅支持 CPU 的快速迭代版本 _(不稳定)_
*   `tf-nightly-gpu` — [支持 GPU](/docs/tensorflow/install/gpu) 的快速迭代版本 _(不稳定, Ubuntu 和 Windows)_

### 系统要求

*   Ubuntu 16.04 及以上 (64 位)
*   macOS 10.12.6 (Sierra) 及以上 (64-bit) _(无 GPU 支持)_
*   Windows 7 及以上 (64 位) _(仅 Python 3)_
*   Raspbian 9.0 及以上

### 硬件要求

*   TensorFlow 从 1.6 版本开始, TensorFlow 执行文件使用 [AVX 指令集](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX)，可能会在一些老式的 CPU 里运行不了。
*   如果要在 Ubuntu 或者 Windows 中使用 GPU 显卡加速，请参阅 [GPU 指南](/docs/tensorflow/install/gpu)。

1\. 安装 Python
-------------------------------------------------------------

检查 Python 环境是否配置正确：

Python2

    python --version

Python3(要求 Python 3.4, 3.5, or 3.6)

    python3 --version

如果安装好了就跳过这一步，不然就一个个安装吧！ [Python](https://www.python.org/)、[pip package manager](https://pip.pypa.io/en/stable/installing/) 还有 [Virtualenv](https://virtualenv.pypa.io/en/stable/)：

### Ubuntu

    sudo apt update

### mac OS

使用 [Homebrew](https://brew.sh/) 安装：

    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

### Windows (仅 Python3)

安装 _Microsoft Visual C++ 2015 Redistributable Update 3_，这个包含在 _Visual Studio 2015_ 里但是可以单独安装：

1.  进入 [Visual Studio 下载](https://visualstudio.microsoft.com/vs/older-downloads/)，
2.  选择 _Redistributables and Build Tools_，
3.  下载并安装 _Microsoft Visual C++ 2015 Redistributable Update 3_。

安装 _64 位_ [Python 3 release for Windows](https://www.python.org/downloads/windows/) (安装时选中 `pip` 选项).

    pip3 install -U pip virtualenv

### 树莓派

要求 [Raspbian](https://www.raspberrypi.org/downloads/raspbian/) 操作系统：

    sudo apt update

### 其他系统

    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

2\. 创建虚拟环境 (推荐)
----------------------------------------------

Python 虚拟环境一般用来与系统中的 Python 环境相隔离。

### Ubuntu / mac OS

创建一个新的虚拟环境时要先选择一个 Python 解释器并且要创建一个文件夹（一般是 `./venv`）来存放虚拟环境的文件:

Python2

    virtualenv --system-site-packages -p python2.7 ./venv

Python3

    virtualenv --system-site-packages -p python3 ./venv


通过以下命令来激活虚拟环境：

    source ./venv/bin/activate  # sh, bash, ksh, or zsh

当虚拟环境激活时，命令行前面会有 `(venv)` 字样。

在虚拟环境内任意安装扩展包都不会影响系统中的配置。先升级 `pip`:

    pip install --upgrade pip

如果要退出虚拟环境:

    deactivate  # don't exit until you're done using TensorFlow

### Windows(仅 Python3)

创建一个新的虚拟环境时要先选择一个 Python 解释器并且要创建一个文件夹（一般是 `./venv`）来存放虚拟环境的文件:

    virtualenv --system-site-packages -p python3 ./venv

激活虚拟环境：

    .\venv\Scripts\activate

在虚拟环境内任意安装扩展包都不会影响系统中的配置。先升级 `pip`:

    pip install --upgrade pip

如果要退出虚拟环境:

    deactivate  # don't exit until you're done using TensorFlow

### Conda

除了 _pip_ 安装包之外，还有_社区支持的_ [Anaconda 安装包](https://anaconda.org/conda-forge/tensorflow)可用。

创建一个新的虚拟环境时要先选择一个 Python 解释器并且要创建一个文件夹（一般是 `./venv`）来存放虚拟环境的文件:

Python2

    conda create -n venv pip python=2.7

Python3

    conda create -n venv pip python=3.6  # select python version

激活虚拟环境：

    source activate venv

在虚拟环境里，要用[完整的 URL](#package-location) 去安装：

    pip install --ignore-installed --upgrade packageURL

如果要退出虚拟环境:

    source deactivate

3\. 安装 TensorFlow pip 安装包
--------------------------------------

选择下列之一的包安装：

*   `tensorflow` — 仅支持 CPU 的当前发布版本 _(推荐初学者使用)_
*   `tensorflow-gpu` — [支持 GPU](/docs/tensorflow/install/gpu) 的当前版本 _(Ubuntu 和 Windows)_
*   `tf-nightly` — 仅支持 CPU 的快速迭代版本 _(不稳定)_
*   `tf-nightly-gpu` — [支持 GPU](/docs/tensorflow/install/gpu) 的快速迭代版本 _(不稳定, Ubuntu 和 Windows)_

依赖包将会被自动安装，你可以在 [`setup.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py) 文件的 `REQUIRED_PACKAGES` 里找到全部的依赖包。

### Virtualenv 安装

    pip install --upgrade tensorflow

验证安装:

    python -c "import tensorflow as tf; print(tf.\_\_version\_\_)"

### 在系统里安装

Python2

    pip install --user --upgrade tensorflow  # install in $HOME

验证安装:

    python -c "import tensorflow as tf; print(tf.\_\_version\_\_)"

Python3

    pip3 install --user --upgrade tensorflow  # install in $HOME

验证安装:

    python3 -c "import tensorflow as tf; print(tf.\_\_version\_\_)"

> **成功：** TensorFlow 已经安装好了。 从 [教程](/docs/tensorflow/tutorials) 开始你的 TensorFlow 之旅吧！

<a id="package-location"></a>
## 安装包地址

有些安装机制需要 Python 安装包的 URL，你可以根据你的系统和 Python 版本去选择你要安装的。

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
