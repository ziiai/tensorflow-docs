# Docker 中安装

[Docker](https://docs.docker.com/install/) 使用*容器*来创建虚拟环境，这样就很好地隔离了 TensorFlow。
TensorFlow 运行在可与主机共享资源（访问目录、使用 GPU、联网等等）。
[TensorFlow Docker 镜像](https://hub.docker.com/r/tensorflow/tensorflow/)已在每个发行版本中测试过。

Docker 是一种最为简单的在 Linux 上安装 [GPU 版本](/docs/tensorflow/install/gpu) TensorFlow 的选择。仅需要安装
[NVIDIA® GPU 驱动](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver)即可，
*NVIDIA® CUDA® Toolkit* 都省了。

## TensorFlow Docker 安装要求

1. [安装 Docker](https://docs.docker.com/install/)。
2. 如果实在 Linux 安装 GPU 版本， 先安装 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)。

> 如果想不带 `sudo`，需要创建 `docker` 用户组并添加你要使用的用户，请参阅
[Linux 安装后续步骤](https://docs.docker.com/install/linux/linux-postinstall/)。


## 下载 TensorFlow Docker 镜像

官方镜像在
[tensorflow/tensorflow](https://hub.docker.com/r/tensorflow/tensorflow/)仓库里。 [镜像版本](https://hub.docker.com/r/tensorflow/tensorflow/tags/)如下：

<table>
  <tr><th>Tag</th><th>Description</th></tr>
  <tr><td><code>latest</td><td>The latest release of TensorFlow CPU binary image. Default.</td></tr>
  <tr><td><code>nightly</td><td>Nightly builds of the TensorFlow image. (unstable)</td></tr>
  <tr><td><code><em>version</em></td><td>Specify the <em>version</em> of the TensorFlow binary image, for example: <em>1.11</em></td></tr>
  <tr class="alt"><td colspan="2">Tag variant</td></tr>
  <tr><td><code><em>tag</em>-devel<code></td><td>The specified <em>tag</em> release and the source code.</td></tr>
  <tr><td><code><em>tag</em>-gpu<code></td><td>The specified <em>tag</em> release with GPU support. (<a href="#gpu_support">See below</a>)</td></tr>
  <tr><td><code><em>tag</em>-py3<code></td><td>The specified <em>tag</em> release with Python 3 support.</td></tr>
  <tr><td><code><em>tag</em>-gpu-py3<code></td><td>The specified <em>tag</em> release with GPU and Python 3 support.</td></tr>
  <tr><td><code><em>tag</em>-devel-py3<code></td><td>The specified <em>tag</em> release with Python 3 support and the source code.</td></tr>
  <tr><td><code><em>tag</em>-devel-gpu<code></td><td>The specified <em>tag</em> release with GPU support and the source code.</td></tr>
  <tr><td><code><em>tag</em>-devel-gpu-py3<code></td><td>The specified <em>tag</em> release with GPU and Python 3 support, and the source code.</td></tr>
</table>

比如，你可以这样下载：

    docker pull tensorflow/tensorflow                    # latest stable release
    docker pull tensorflow/tensorflow:nightly-devel-gpu  # nightly dev release w/ GPU support

## 开启 TensorFlow Docker 容器

输入以下内容使用 TensorFlow 容器:

    docker run [-it] [--rm] [-p <em>hostPort</em>:<em>containerPort</em>] tensorflow/tensorflow[:<em>tag</em>] [<em>command</em>]

详情请参见 [docker 运行参考](https://docs.docker.com/engine/reference/run/).

### CPU 版本镜像示例

来确认下是否安装成功。 Docker 会第一次执行时下载 TensorFlow 镜像：

    docker run -it --rm tensorflow/tensorflow \
    python -c "import tensorflow as tf; print(tf.__version__)"


> 安装成功！可以去看[教程](/docs/tensorflow/tutorials)了。

我们来看看其他操作。在 TensorFlow 容器中打开 命令行：

    docker run -it tensorflow/tensorflow bash


在容器内，你也可以打开 `python` 会话并且进行相关操作。

可以指定主机目录和容器工作目录（`-v hostDir:containerDir -w workDir`）去执行 TensorFlow 程序：

    docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow python ./script.py

如果在容器内创建文件的话将会出现一些权限问题，最好是在主机里编辑好。

在 python3 的 nightly 版本里开启 [Jupyter Notebook](https://jupyter.org/) 服务器：

    docker run -it -p 8888:8888 tensorflow/tensorflow:nightly-py3

打开网页：`http://127.0.0.1:8888/?token=...`

## GPU 支持

安装 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 登录 Docker 容器。`nvidia-docker` 仅支持 Linux，详情请参见[平台支持 FAQ](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#platform-support)。

检查是否支持 GPU：

    lspci | grep -i nvidia


确认 `nvidia-docker` 的安装:

    docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi


> 注意：`nvidia-docker` v1 使用 `nvidia-docker` 别名， v2 使用 `docker --runtime=nvidia`.

### GPU 版本示例

下载 GPU 版本：

    docker run --runtime=nvidia -it --rm tensorflow/tensorflow:latest-gpu \
    python -c "import tensorflow as tf; print(tf.contrib.eager.num_gpus())"


这里可能会花一点时间。你可以使用 `docker exec` 将容器重复使用。

使用最新 TensorFlow GPU 镜像在容器内使用命令行：

    docker run --runtime=nvidia -it tensorflow/tensorflow:latest-gpu bash

> 安装成功！可以去看[教程](/docs/tensorflow/tutorials)了。
