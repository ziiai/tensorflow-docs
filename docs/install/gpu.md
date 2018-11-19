# GPU 支持

TensorFlow GPU 版本需要有相关的驱动和软件库的支持。为了简化安装和避免引起冲突我们还是建议你使用
[包含 GPU 支持的 TensorFlow Docker 镜像](/docs/tensorflow/install/docker) (仅 Linux)，这样的话只需要设置
[NVIDIA® GPU 驱动](https://www.nvidia.com/drivers)就行了。

## 硬件要求

要有支持 GPU 的显卡设备：

* 计算力 3.5 及以上的 NVIDIA® GPU 显卡。请参阅 [CUDA GPU 显卡](https://developer.nvidia.com/cuda-gpus)列表。

## 软件要求

系统要装以下 NVIDIA® 软件:

* [NVIDIA® GPU drivers](https://www.nvidia.com/drivers) —CUDA 9.0 要求 384.x 或更高。
* [CUDA® Toolkit](https://developer.nvidia.com/cuda-zone) —TensorFlow 支持 CUDA 9.0。
* [CUPTI](http://docs.nvidia.com/cuda/cupti/) 包含在 CUDA Toolkit 里。
* [cuDNN SDK](https://developer.nvidia.com/cudnn) (>= 7.2)
* *(可选)* [NCCL 2.2](https://developer.nvidia.com/nccl) 多 GPU 支持。
* *(可选)* [TensorRT 4.0](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)
  改善某些性能。

## Linux 设置

下面的 `apt` 命令能很方便地在 Ubuntu 上安装 NVIDIA 软件。但如果你是 [源码安装](/docs/tensorflow/install/source)的话，
你需要手动安装上述的软件，还有考虑一下使用
`-devel` [TensorFlow Docker 镜像](/docs/tensorflow/install/docker)。

安装 CUDA® Toolkit 自带的 [CUPTI](http://docs.nvidia.com/cuda/cupti/)，并将其安装目录添加到 `$LD_LIBRARY_PATH` 环境变量：

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

如果是算力 3.0 的 GPU 或者不同版本 NVIDIA 库 的话，请参阅 [Linux 上源码安装](/docs/tensorflow/install/source)指南。

### Install CUDA with apt

对于 Ubuntu 16.04 或者其他基于 Debian 的 Linux 系统，先添加 NVIDIA 包仓库再用 `apt` 安装 CUDA。

注意: `apt` installs the NVIDIA libraries and headers to locations that make
it difficult to configure and debug build issues.


# 添加 NVIDIA 包仓库
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
    sudo apt install ./cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
    sudo apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
    sudo apt update

# 安装 CUDA 及其他工具，包括可选的 NCCL 2.x
    sudo apt install cuda9.0 cuda-cublas-9-0 cuda-cufft-9-0 cuda-curand-9-0 \
    cuda-cusolver-9-0 cuda-cusparse-9-0 libcudnn7=7.2.1.38-1+cuda9.0 \
    libnccl2=2.2.13-1+cuda9.0 cuda-command-line-tools-9-0

# 可选: 安装 TensorRT 运行时 (必须在 CUDA 安装好之后)
    sudo apt update
    sudo apt install libnvinfer4=4.1.2-1+cuda9.0
    
## Windows 设置

参考上文的 [硬件要求](#hardware_requirements) 和
[软件要求](#software_requirements)，还有
[Windows CUDA® 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).

务必确保所安装的软件版本要匹配。如果没有 `cuDNN64_7.dll` 文件，TensorFlow 将不会运行。如果要用不同的版本，请参考 [Windows 源码安装](/docs/tensorflow/install/source_windows)。

将 CUDA、CUPTI、和 cuDNN 安装目录添加到 `%PATH%` 环境变量中。比如 CUDA Toolkit 安装在
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0` ， cuDNN 在
`C:\tools\cuda`，按照下列更新 `%PATH%`：

    SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin;%PATH%
    SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\extras\CUPTI\libx64;%PATH%
    SET PATH=C:\tools\cuda\bin;%PATH%