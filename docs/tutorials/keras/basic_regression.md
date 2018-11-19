# 预测房价：回归

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/tutorials/keras/basic_regression"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_regression.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_regression.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

在回归问题中，我们的目标是预测连续值的输出，如价格或概率。不妨将此问题与分类问题进行对比，在分类问题中，我们的目标是预测离散标签（例如，某张照片中包含苹果还是橙子）。

此笔记本会构建一个模型，用于预测 20 世纪 70 年代中期波士顿郊区房价的中间值。为此，我们将为该模型提供一些关于波士顿郊区的数据点，如犯罪率和当地的房产税率。

此示例使用的是 tf.keras API，有关详情，请参阅[本指南](https://www.tensorflow.org/guide/keras)。


```
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)
```

## 波士顿房价数据集

此[数据集](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)可以直接在 TensorFlow 中访问。下载并随机化处理训练集：


```
boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]
```

### 样本和特征

此数据集比我们到目前为止使用的其他数据集小得多：它共有 506 个样本，拆分为 404 个训练样本和 102 个测试样本：

```
print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features
```

该数据集包含 13 个不同的特征：

- 人均犯罪率。
- 占地面积超过 25000 平方英尺的住宅用地所占的比例。
- 非零售商业用地所占的比例（英亩/城镇）。
- 查尔斯河虚拟变量（如果大片土地都临近查尔斯河，则为 1；否则为 0）。
- 一氧化氮浓度（以千万分之一为单位）。
- 每栋住宅的平均房间数。
- 1940 年以前建造的自住房所占比例。
- 到 5 个波士顿就业中心的加权距离。
- 辐射式高速公路的可达性系数。
- 每 10000 美元的全额房产税率。
- 生师比（按城镇统计）。
- 1000 * (Bk - 0.63) ** 2，其中 Bk 是黑人所占的比例（按城镇统计）。
- 较低经济阶层人口所占百分比。

以上每个输入数据特征都有不同的范围。一些特征用介于 0 到 1 之间的比例表示，另外一些特征的范围在 1 到 12 之间，还有一些特征的范围在 0 到 100 之间，等等。真实的数据往往都是这样，了解如何探索和清理此类数据是一项需要加以培养的重要技能。

要点：作为建模者兼开发者，需要考虑如何使用这些数据，以及模型预测可能会带来哪些潜在益处和危害。类似这样的模型可能会加深社会偏见，扩大社会差异。某个特征是否与您想要解决的问题相关，或者是否会引入偏见？要了解详情，请参阅 [机器学习公平性](https://developers.google.com/machine-learning/fairness-overview/)。


```
print(train_data[0])  # Display sample features, notice the different scales
```

使用 [pandas](https://pandas.pydata.org) 库在格式规范的表格中显示数据集的前几行：


```
import pandas as pd

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
df.head()
```

### 标签

标签是房价（以千美元为单位）。（您可能会注意到 20 世纪 70 年代中期的房价。）

```
print(train_labels[0:10])  # Display first 10 entries
```

## 标准化特征

建议标准化使用不同比例和范围的特征。对于每个特征，用原值减去特征的均值，再除以标准偏差即可：

```
# Test data is *not* used when calculating the mean and std

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data[0])  # First training sample, normalized
```

虽然在未进行特征标准化的情况下，模型可能会收敛，但这样做会增加训练难度，而且使生成的模型更加依赖于在输入中选择使用的单位。

## 创建模型

我们来构建模型。在此教程中，我们将使用 `Sequential` 模型，该模型包含两个密集连接隐藏层，以及一个返回单个连续值的输出层。由于我们稍后要再创建一个模型，因此将模型构建步骤封装在函数 `build_model` 中。

```
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()
```

## 训练模型

对该模型训练 500 个周期，并将训练和验证准确率记录到 `history` 对象中。

```
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])
```

使用存储在 `history` 对象中的统计数据可视化模型的训练进度。我们希望根据这些数据判断：对模型训练多长时间之后它会停止优化。

```
import matplotlib.pyplot as plt


def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])

plot_history(history)
```

此图显示，在大约 200 个周期之后，模型几乎不再出现任何改进。我们更新一下 `model.fit` 方法，以便在验证分数不再提高时自动停止训练。我们将使用一个回调来测试每个周期的训练状况。如果模型在一定数量的周期之后没有出现任何改进，则自动停止训练。
您可以点击
[此处](https://www.tensorflow.org/versions/master/api_docs/python/tf/keras/callbacks/EarlyStopping)
详细了解此回调。

```
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)
```

此图显示平均误差约为 2500 美元。这是一个出色的模型吗？如果某些标签只是 15000 美元，那么 2500 美元的误差也不算小。

现在看一下模型在测试集上的表现如何：


```
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))
```

## 预测

最后，使用测试集中的数据预测某些房价：


```
test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])

```


```
error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")
```

## 总结

此笔记本介绍了几个处理回归问题的技巧。

- 均方误差 (MSE) 是用于回归问题的常见损失函数（与分类问题不同）。
- 同样，用于回归问题的评估指标也与分类问题不同。常见回归指标是平均绝对误差 (MAE)。
- 如果输入数据特征的值具有不同的范围，则应分别缩放每个特征。
- 如果训练数据不多，则选择隐藏层较少的小型网络，以避免出现过拟合。
- 早停法是防止出现过拟合的实用技术。


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
