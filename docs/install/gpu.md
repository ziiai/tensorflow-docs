# GPU support

TensorFlow GPU support requires an assortment of drivers and libraries. To
simplify installation and avoid library conflicts, we recommend using a
[TensorFlow Docker image with GPU support](/docs/tensorflow/install/docker) (Linux only). This setup
only requires the [NVIDIA® GPU drivers](https://www.nvidia.com/drivers).

## Hardware requirements

The following GPU-enabled devices are supported:

* NVIDIA® GPU card with CUDA® Compute Capability 3.5 or higher. See the list of
  [CUDA-enabled GPU cards](https://developer.nvidia.com/cuda-gpus).

## Software requirements

The following NVIDIA® software must be installed on your system:

* [NVIDIA® GPU drivers](https://www.nvidia.com/drivers) —CUDA 9.0 requires 384.x or higher.
* [CUDA® Toolkit](https://developer.nvidia.com/cuda-zone) —TensorFlow supports CUDA 9.0.
* [CUPTI](http://docs.nvidia.com/cuda/cupti/) ships with the CUDA Toolkit.
* [cuDNN SDK](https://developer.nvidia.com/cudnn) (>= 7.2)
* *(Optional)* [NCCL 2.2](https://developer.nvidia.com/nccl) for multiple GPU support.
* *(Optional)* [TensorRT 4.0](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)
  to improve latency and throughput for inference on some models.

## Linux setup

The `apt` instructions below are the easiest way to install the required NVIDIA
software on Ubuntu. However, if [building TensorFlow from source](/docs/tensorflow/install/source),
manually install the software requirements listed above, and consider using a
`-devel` [TensorFlow Docker image](/docs/tensorflow/install/docker) as a base.

Install [CUPTI](http://docs.nvidia.com/cuda/cupti/) which ships with
the CUDA® Toolkit. Append its installation directory to the `$LD_LIBRARY_PATH`
environmental variable:

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

For a GPU with CUDA Compute Capability 3.0, or different versions of the
NVIDIA libraries, see the [Linux build from source](/docs/tensorflow/install/source) guide.

### Install CUDA with apt

For Ubuntu&nbsp;16.04—and possibly other Debian-based Linux distros—add the
NVIDIA package repository and use `apt` to install CUDA.

Caution: `apt` installs the NVIDIA libraries and headers to locations that make
it difficult to configure and debug build issues.


# Add NVIDIA package repository
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
    sudo apt install ./cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
    sudo apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
    sudo apt update

# Install CUDA and tools. Include optional NCCL 2.x
    sudo apt install cuda9.0 cuda-cublas-9-0 cuda-cufft-9-0 cuda-curand-9-0 \
    cuda-cusolver-9-0 cuda-cusparse-9-0 libcudnn7=7.2.1.38-1+cuda9.0 \
    libnccl2=2.2.13-1+cuda9.0 cuda-command-line-tools-9-0

# Optional: Install the TensorRT runtime (must be after CUDA install)
    sudo apt update
    sudo apt install libnvinfer4=4.1.2-1+cuda9.0
    
## Windows setup

See the [hardware requirements](#hardware_requirements) and
[software requirements](#software_requirements) listed above. Read the
[CUDA® install guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).

Make sure the installed NVIDIA software packages match the versions listed above. In
particular, TensorFlow will not load without the `cuDNN64_7.dll` file. To use a
different version, see the [Windows build from source](/docs/tensorflow/install/source_windows) guide.

Add the CUDA, CUPTI, and cuDNN installation directories to the `%PATH%`
environmental variable. For example, if the CUDA Toolkit is installed to
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0` and cuDNN to
`C:\tools\cuda`, update your `%PATH%` to match:

    SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin;%PATH%
    SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\extras\CUPTI\libx64;%PATH%
    SET PATH=C:\tools\cuda\bin;%PATH%