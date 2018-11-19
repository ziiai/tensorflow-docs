# Install TensorFlow

TensorFlow is tested and supported on the following 64-bit systems:

- Ubuntu 16.04 or later
- Windows 7 or later
- macOS 10.12.6 (Sierra) or later (no GPU support)
- Raspbian 9.0 or later

## Download a package

Install TensorFlow with Python's <em>pip</em> package manager.
Official packages available for Ubuntu, Windows, macOS, and the Raspberry Pi.
GPU packages require a [CUDA®-enabled GPU card](/docs/tensorflow/install/gpu)

[Read the pip install guide](/docs/tensorflow/install/pip)

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

## Run a TensorFlow container
The [TensorFlow Docker images](https://hub.docker.com/r/tensorflow/tensorflow/)
are already configured to run TensorFlow. A [Docker](https://docs.docker.com/install/)
container runs in a virtual environment and is the easiest way to set up [GPU support](/docs/tensorflow/install/gpu).

```
docker pull tensorflow/tensorflow                  # Download latest image
docker run -it -p 8888:8888 tensorflow/tensorflow  # Start a Jupyter notebook server
```

[Read the Docker install guide](/docs/tensorflow/install/docker)

## Google Colab: An easy way to learn and use TensorFlow

No install necessary—run the <a href="/docs/tensorflow/tutorials">TensorFlow tutorials</a>
directly in the browser with <a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Colaboratory</a>,
a Google research project created to help disseminate machine learning
education and research. It's a Jupyter notebook environment that requires
no setup to use and runs entirely in the cloud.
<a href="https://medium.com/tensorflow/colab-an-easy-way-to-learn-and-use-tensorflow-d74d1686e309" class="external">Read the blog post</a>.

## Build your first ML app
Create and deploy TensorFlow models on web and mobile.

### <a href="https://js.tensorflow.org" target="_blank">Web developers</a>
TensorFlow.js is a WebGL accelerated, JavaScript library to train anddeploy ML models in the browser and for Node.js.
### <a href="https://tensorflow.google.cn/lite/" target="_blank">Mobile developers</a>
TensorFlow Lite is lightweight solution for mobile and embedded devices.