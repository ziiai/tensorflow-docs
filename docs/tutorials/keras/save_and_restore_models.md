# 保存和恢复模型

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/tutorials/keras/save_and_restore_models"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/save_and_restore_models.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/save_and_restore_models.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

模型进度可在训练期间和之后保存。这意味着，您可以从上次暂停的地方继续训练模型，避免训练时间过长。此外，可以保存意味着您可以分享模型，而他人可以对您的工作成果进行再创作。发布研究模型和相关技术时，大部分机器学习从业者会分享以下内容：

- 用于创建模型的代码，以及
- 模型的训练权重或参数

分享此类数据有助于他人了解模型的工作原理并尝试使用新数据自行尝试模型。

注意：请谨慎使用不可信的代码 - TensorFlow 模型就是代码。有关详情，请参阅[安全地使用 TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md)。

### 选项

您可以通过多种不同的方法保存 TensorFlow 模型，具体取决于您使用的 API。本指南使用的是 [tf.keras](https://www.tensorflow.org/guide/keras)，它是一种用于在 TensorFlow 中构建和训练模型的高阶 API。要了解其他方法，请参阅 TensorFlow [保存和恢复](https://www.tensorflow.org/guide/saved_model)指南或[在 Eager 中保存](https://www.tensorflow.org/guide/eager#object_based_saving)。


## 设置

### 安装和导入

安装并导入 TensorFlow 和依赖项：


```
!pip install h5py pyyaml 
```

### 获取示例数据集

我们将使用 [MNIST dataset](http://yann.lecun.com/exdb/mnist/)训练模型，以演示如何保存权重。要加快演示运行速度，请仅使用前 1000 个样本：


```
from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

tf.__version__
```


```
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
```

### 定义模型

我们来构建一个简单的模型，以演示如何保存和加载权重。


```
# Returns a short sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  
  model.compile(optimizer=tf.train.AdamOptimizer(), 
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  
  return model


# Create a basic model instance
model = create_model()
model.summary()
```

## 在训练期间保存检查点

主要用例是，在训练期间或训练结束时自动保存检查点。这样一来，您便可以使用经过训练的模型，而无需重新训练该模型，或从上次暂停的地方继续训练，以防训练过程中断。

`tf.keras.callbacks.ModelCheckpoint` 是执行此任务的回调。该回调需要几个参数来配置检查点。

### 检查点回调用法

训练模型，并将 `ModelCheckpoint` 回调传递给该模型：


```
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

model.fit(train_images, train_labels,  epochs = 10, 
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])  # pass callback to training
```

上述代码将创建一个 TensorFlow 检查点文件集合，这些文件在每个周期结束时更新：


```
!ls {checkpoint_dir}
```

创建一个未经训练的全新模型。仅通过权重恢复模型时，您必须有一个与原始模型架构相同的模型。由于模型架构相同，因此我们可以分享权重（尽管是不同的模型实例）。

现在，重新构建一个未经训练的全新模型，并用测试集对其进行评估。未训练模型的表现有很大的偶然性（准确率约为 10%）：


```
model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
```

然后从检查点加载权重，并重新评估：


```
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```

### 检查点回调选项

该回调提供了多个选项，用于为生成的检查点提供独一无二的名称，以及调整检查点创建频率。

训练一个新模型，每隔 5 个周期保存一次检查点并设置唯一名称：


```
# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

model = create_model()
model.fit(train_images, train_labels,
          epochs = 50, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose=0)
```

现在，看一下生成的检查点（按修改日期排序）：

```
! ls {checkpoint_dir}
```


```
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
```

注意：默认的 TensorFlow 格式仅保存最近的 5 个检查点。

要进行测试，请重置模型并加载最新的检查点：

```
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```

## 这些是什么文件？

上述代码将权重存储在[检查点](https://www.tensorflow.org/guide/saved_model#save_and_restore_variables)格式的文件集合中，这些文件仅包含经过训练的权重（采用二进制格式）。检查点包括：
- 包含模型权重的一个或多个分片。 
- 指示哪些权重存储在哪些分片中的索引文件。

如果您仅在一台机器上训练模型，则您将有 1 个后缀为 `.data-00000-of-00001` 的分片。

## 手动保存权重

在上文中，您了解了如何将权重加载到模型中。

手动保存权重的方法同样也很简单，只需使用 `Model.save_weights` 方法即可。

```
# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```

## 保存整个模型

整个模型可以保存到一个文件中，其中包含权重值、模型配置乃至优化器配置。这样，您就可以为模型设置检查点，并稍后从完全相同的状态继续训练，而无需访问原始代码。

在 Keras 中保存完全可正常使用的模型非常有用，您可以在([HDF5](https://js.tensorflow.org/tutorials/import-keras.html)中加载它们，然后在网络浏览器中训练和运行它们。

### 存为 HDF5 文件

Keras 使用 [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) 标准提供基本的保存格式。对于我们来说，可将保存的模型视为一个二进制 blob。


```
model = create_model()

# You need to use a keras.optimizer to restore the optimizer state from an HDF5 file.
model.compile(optimizer=keras.optimizers.Adam(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# Save entire model to a HDF5 file
model.save('my_model.h5')
```

现在，从该文件重新创建模型：


```
# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('my_model.h5')
new_model.summary()
```

检查其准确率：

```
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```

此技巧可保存以下所有内容：

- 权重值
- 模型配置（架构）
- 优化器配置

Keras 通过检查架构来保存模型。目前，它无法保存 TensorFlow 优化器（来自 `tf.train`）。使用此类优化器时，您需要在加载模型后对其进行重新编译，使优化器的状态变松散。


### 存为 `saved_model`

注意: 这种保存 `tf.keras` 模型的方法为试验性的，之后可能会改变。

创建一个新模型:

```
model = create_model()

model.fit(train_images, train_labels, epochs=5)
```

创建一个 `saved_model`: 


```
saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")
```

保存的模型将会放在一个时间戳命名的文件夹下:


```
!ls saved_models/
```

从保存的模型中加载：


```
new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
new_model
```

运行恢复的模型：


```
# The optimizer was not restored, re-attach a new one.
new_model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```

## 后续学习计划

这些就是使用 `tf.keras` 保存和加载模型的快速指南。

* [tf.keras 指南](https://www.tensorflow.org/guide/keras)详细介绍了如何使用 `tf.keras` 保存和加载模型。

* 请参阅[在 Eager 中保存](https://www.tensorflow.org/guide/eager#object_based_saving)，了解如何在 Eager Execution 期间保存模型。

* [保存和恢复](https://www.tensorflow.org/guide/saved_model)指南介绍了有关 TensorFlow 保存的低阶详细信息。


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

```
#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
