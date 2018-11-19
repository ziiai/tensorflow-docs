# 安装 TensorFlow

TensorFlow 通过以下 64 位系统的测试:

- Ubuntu 16.04 及之后的版本
- Windows 7 及之后的版本
- macOS 10.12.6 (Sierra) 及之后的版本 (无 GPU 支持)
- Raspbian 9.0 及之后的版本

## 下载安装包

官方为 Ubuntu、Windows、macOS 和树莓派等系统提供了 `pip` 安装包，用户可通过 `pip` 包管理器进行安装。
GPU 版本需要 [支持 CUDA® 的 GPU 显卡](/docs/tensorflow/install/gpu)。

[阅读 pip 安装指南](/docs/tensorflow/install/pip)

```
# Current release for CPU-only
pip install tensorflow

# Nightly build for CPU-only (unstable)
pip install tf-nightly

# GPU package for CUDA-enabled GPU cards
pip install tensorflow-gpu

# Nightly build with GPU support (unstable)
pip install tf-nightly-gpu
```

## 运行 TensorFlow 容器

官方已经配置好了可直接使用的[TensorFlow Docker 镜像](https://hub.docker.com/r/tensorflow/tensorflow/)，
这种运行在虚拟环境中的[Docker](https://docs.docker.com/install/)容器是一种再简单不过的使用
[GPU 版本](/docs/tensorflow/install/gpu)的方法了。

```
docker pull tensorflow/tensorflow                  # Download latest image
docker run -it -p 8888:8888 tensorflow/tensorflow  # Start a Jupyter notebook server
```

[阅读 Docker 安装指南](/docs/tensorflow/install/docker)

## Google Colab: 让你轻松学习和使用 TensorFLow

不需要安装 —— 直接在浏览器打开 <a href="https://colab.research.google.com/notebooks/welcome.ipynb" target="_blank">Colaboratory</a>
就可以运行 <a href="/docs/tensorflow/tutorials">TensorFlow 教程</a>。Google Colab 是谷歌的一项致力于传播机器学习教育和研究的研究项目，是一个已经配置好的 Jupyter Notebook 云编程环境。
<a href="https://www.ziiai.com/blog/765" target="_blank">阅读博客</a>。

## 创建你的第一个机器学习应用
在网页和移动端创建和部署 TensorFlow 模型。

### <a href="https://js.tensorflow.org" target="_blank">网络开发者</a>
TensorFlow.js：基于 WebGL 加速的、在浏览器和 Node.js 中运行、训练和部署机器学习模型地 JavaScript 库。
### <a href="https://tensorflow.google.cn/lite/" target="_blank">移动开发者</a>
TensorFlow Lite ：为移动和嵌入式设备设计的轻量级解决方案。