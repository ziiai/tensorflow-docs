# 开始使用 TensorFlow

TensorFlow 是一个用于研究和生产的开放源代码机器学习库。TensorFlow 提供了各种 API，可供初学者和专家在桌面、移动、网络和云端环境下进行开发。请参阅以下几部分，了解如何开始使用。 

## 学习和使用机器学习

高阶 Keras API 提供了用于创建和训练深度学习模型的构造块。请先查看以下适合初学者的笔记本示例，然后阅读
[TensorFlow Keras 指南](../guide/keras)

- [基本分类](./keras/basic_classification)
- [文本分类](./keras/basic_text_classification)
- [回归](./keras/basic_regression)
- [过拟合和欠拟合](./keras/overfit_and_underfit)
- [保存与加载](./keras/save_and_restore_models)

[阅读 Keras 指南](../guide/keras)

```python
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation=tf.nn.relu),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```

<!-- dynamic if request.tld != 'cn' -->
<!-- <a class="colab-button" target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/_index.ipynb">Run in a <span>Notebook</span></a> -->

## 研究和实验

Eager Execution 提供了由运行定义的命令式高级操作接口。通过自动微分编写自定义层、前向传播和训练循环。请先查看下面的笔记本，然后阅读 [eager execution 指南](../guide/eager).

- [Eager execution 基础](./eager/eager_basics)

- [自动微分和梯度记录](./eager/automatic_differentiation)

- [自定义训练：基础](./eager/custom_training)

- [自定义层](./eager/custom_layers)

- [自定义训练：演示](./eager/custom_training_walkthrough)


[阅读 eager execution 指南](../guide/eager)

## 生产环境

Estimator 可在生产环境中用多台机器训练大型模型。TensorFlow 提供了一组预创建的 Estimator 来实现常见的机器学习算法。请参阅 [Estimators 指南](../guide/estimators).

- [使用 Estimator 构建线性模型](/tutorials/estimators/linear)
- [使用 Estimator 进行宽度与深度学习](https://github.com/tensorflow/models/tree/master/official/wide_deep)
- [提升树](https://github.com/tensorflow/models/tree/master/official/boosted_trees)
- [如何使用 TF-Hub 构建简单的文本分类器](/hub/tutorials/text_classification_with_tf_hub)
- [使用 Estimator 构建卷积神经网络](/tutorials/estimators/cnn)

[阅读 Estimators 指南](/guide/estimators)

## Google Colab：学习和使用 TensorFlow 的一种简单方法

[Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)
是一个 Google 研究项目，旨在帮助传播机器学习培训和研究成果。它是一个 Jupyter 笔记本环境，不需要进行任何设置就可以使用，并且完全在云端运行。[阅读博文](https://medium.com/tensorflow/colab-an-easy-way-to-learn-and-use-tensorflow-d74d1686e309)。 

### 构建首个机器学习应用

创建和部署网页版和移动版 TensorFlow 模型。

#### [网络开发者](https://js.tensorflow.org)

TensorFlow.js 是一个采用 WebGL 加速技术的 JavaScript 库，用于在浏览器中针对 Node.js 训练和部署机器学习模型。 

#### [移动开发者](https://tensorflow.org/lite/)

TensorFlow Lite 是针对移动设备和嵌入式设备提供的精简解决方案。
