
##### Copyright 2018 The TensorFlow Authors.


```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Eager execution basics

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/tutorials/eager/eager_basics"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/eager/eager_basics.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/eager/eager_basics.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

This is an introductory tutorial for using TensorFlow. It will cover:

* Importing required packages
* Creating and using Tensors
* Using GPU acceleration
* Datasets

## Import TensorFlow

To get started, import the `tensorflow` module and enable eager execution.
Eager execution enables a more interactive frontend to TensorFlow, the details of which we will discuss much later.


```
import tensorflow as tf

tf.enable_eager_execution()
```

## Tensors

A Tensor is a multi-dimensional array. Similar to NumPy `ndarray` objects, `Tensor` objects have a data type and a shape. Additionally, Tensors can reside in accelerator (like GPU) memory. TensorFlow offers a rich library of operations ([tf.add](https://www.tensorflow.org/api_docs/python/tf/add), [tf.matmul](https://www.tensorflow.org/api_docs/python/tf/matmul), [tf.linalg.inv](https://www.tensorflow.org/api_docs/python/tf/linalg/inv) etc.) that consume and produce Tensors. These operations automatically convert native Python types. For example:



```
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello world"))

# Operator overloading is also supported
print(tf.square(2) + tf.square(3))
```

Each Tensor has a shape and a datatype


```
x = tf.matmul([[1]], [[2, 3]])
print(x.shape)
print(x.dtype)
```

The most obvious differences between NumPy arrays and TensorFlow Tensors are:

1. Tensors can be backed by accelerator memory (like GPU, TPU).
2. Tensors are immutable.

### NumPy Compatibility

Conversion between TensorFlow Tensors and NumPy ndarrays is quite simple as:

* TensorFlow operations automatically convert NumPy ndarrays to Tensors.
* NumPy operations automatically convert Tensors to NumPy ndarrays.

Tensors can be explicitly converted to NumPy ndarrays by invoking the `.numpy()` method on them.
These conversions are typically cheap as the array and Tensor share the underlying memory representation if possible. However, sharing the underlying representation isn't always possible since the Tensor may be hosted in GPU memory while NumPy arrays are always backed by host memory, and the conversion will thus involve a copy from GPU to host memory.


```
import numpy as np

ndarray = np.ones([3, 3])

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)


print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())
```

## GPU acceleration

Many TensorFlow operations can be accelerated by using the GPU for computation. Without any annotations, TensorFlow automatically decides whether to use the GPU or CPU for an operation (and copies the tensor between CPU and GPU memory if necessary). Tensors produced by an operation are typically backed by the memory of the device on which the operation executed. For example:


```
x = tf.random_uniform([3, 3])

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))
```

### Device Names

The `Tensor.device` property provides a fully qualified string name of the device hosting the contents of the tensor. This name encodes many details, such as an identifier of the network address of the host on which this program is executing and the device within that host. This is required for distributed execution of a TensorFlow program. The string ends with `GPU:<N>` if the tensor is placed on the `N`-th GPU on the host.



### Explicit Device Placement

The term "placement" in TensorFlow refers to how individual operations are assigned (placed on) a device for execution. As mentioned above, when there is no explicit guidance provided, TensorFlow automatically decides which device to execute an operation, and copies Tensors to that device if needed. However, TensorFlow operations can be explicitly placed on specific devices using the `tf.device` context manager. For example:


```
def time_matmul(x):
  %timeit tf.matmul(x, x)

# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random_uniform([1000, 1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)

# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)
```

## Datasets

This section demonstrates the use of the [`tf.data.Dataset` API](https://www.tensorflow.org/guide/datasets) to build pipelines to feed data to your model. It covers:

* Creating a `Dataset`.
* Iteration over a `Dataset` with eager execution enabled.

We recommend using the `Dataset`s API for building performant, complex input pipelines from simple, re-usable pieces that will feed your model's training or evaluation loops.

If you're familiar with TensorFlow graphs, the API for constructing the `Dataset` object remains exactly the same when eager execution is enabled, but the process of iterating over elements of the dataset is slightly simpler.
You can use Python iteration over the `tf.data.Dataset` object and do not need to explicitly create an `tf.data.Iterator` object.
As a result, the discussion on iterators in the [TensorFlow Guide](https://www.tensorflow.org/guide/datasets) is not relevant when eager execution is enabled.

### Create a source `Dataset`

Create a _source_ dataset using one of the factory functions like [`Dataset.from_tensors`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensors), [`Dataset.from_tensor_slices`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices) or using objects that read from files like [`TextLineDataset`](https://www.tensorflow.org/api_docs/python/tf/data/TextLineDataset) or [`TFRecordDataset`](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset). See the [TensorFlow Guide](https://www.tensorflow.org/guide/datasets#reading_input_data) for more information.


```
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# Create a CSV file
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
Line 2
Line 3
  """)

ds_file = tf.data.TextLineDataset(filename)
```

### Apply transformations

Use the transformations functions like [`map`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map), [`batch`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch), [`shuffle`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle) etc. to apply transformations to the records of the dataset. See the [API documentation for `tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) for details.


```
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

ds_file = ds_file.batch(2)
```

### Iterate

When eager execution is enabled `Dataset` objects support iteration.
If you're familiar with the use of `Dataset`s in TensorFlow graphs, note that there is no need for calls to `Dataset.make_one_shot_iterator()` or `get_next()` calls.


```
print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)

print('\nElements in ds_file:')
for x in ds_file:
  print(x)
```
