# 使用显式核方法改进线性模型

> 注意：本文档使用的是已弃用的 `tf.estimator` 版本，该版本具有不同的接口。此外，它还使用了其他 `contrib` 方法，这些方法的 [API 可能不稳定](/docs/tensorflow/guide/version_compat#not_covered)。

在本教程中，我们演示了结合使用（显式）核方法与线性模型可以如何大幅提高线性模型的预测质量，并且不会显著增加训练和推理时间。与双核方法不同，就训练/推理时间和内存要求而言，显式（原始）核方法能够根据训练数据集的大小灵活调整。

目标读者：虽然我们简要概述了与显式核方法相关的概念，但本教程主要面向至少已掌握核方法和支持向量机 (SVM) 基础知识的读者。如果您刚开始接触核方法，请参阅以下任一资源，简单了解一下核方法。

* 如果您有很强的数学背景，请参阅：[机器学习中的核方法](https://arxiv.org/pdf/math/0701907.pdf)
* [维基百科“核方法”页面](https://en.wikipedia.org/wiki/Kernel_method)

目前，TensorFlow 仅支持密集特征的显式核映射；TensorFlow 将在后续版本中提供对稀疏特征的支持。

本教程使用 `tf.contrib.learn`（TensorFlow 的高阶机器学习 API）Estimator 构建我们的机器学习模型。如果您不熟悉此 API，不妨通过 [Estimator guide](/docs/tensorflow/guide/estimators)
着手了解。我们将使用 MNIST 数据集。本教程包含以下步骤：

- 加载和准备 MNIST 数据，以用于分类。
- 构建一个简单的线性模型，训练该模型，并用评估数据对其进行评估。
- 将线性模型替换为核化线性模型，重新训练它，并重新进行评估。

## 加载和准备用于分类的 MNIST 数据

运行以下实用程序命令，以加载 MNIST 数据集：

```python
data = tf.contrib.learn.datasets.mnist.load_mnist()
```
上述方法会加载整个 MNIST 数据集（包含 7 万个样本），然后将数据集拆分为训练数据（5.5 万）、验证数据（5 千）和测试数据（1 万）。拆分的每个数据集均包含一个图像 NumPy 数组（形状为 [sample_size, 784]）以及一个标签 NumPy 数组（形状为 [sample_size, 1]）。在本教程中，我们仅分别使用训练数据和验证数据训练和评估模型。

要将数据馈送到 `tf.contrib.learn Estimator`，将数据转换为张量会很有帮助。为此，我们将使用 `input function` 将操作添加到 TensorFlow 图，该图在执行时会创建要在下游使用的小批次张量。有关输入函数的更多背景知识，请参阅
[this section on input functions](/docs/tensorflow/guide/premade_estimators#create_input_functions).
这一部分。在本示例中，我们不仅会将 NumPy 数组转换为张量，还将使用 `tf.train.shuffle_batch` 操作指定 batch_size 以及是否在每次执行 input_fn 操作时都对输入进行随机化处理（在训练期间，随机化处理通常会加快收敛速度）。以下代码段是加载和准备数据的完整代码。在本示例中，我们使用大小为 256 的小批次数据集进行训练，并使用整个样本（5 千个条目）进行评估。您可以随意尝试不同的批次大小。

```python
import numpy as np
import tensorflow as tf

def get_input_fn(dataset_split, batch_size, capacity=10000, min_after_dequeue=3000):

  def _input_fn():
    images_batch, labels_batch = tf.train.shuffle_batch(
        tensors=[dataset_split.images, dataset_split.labels.astype(np.int32)],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        enqueue_many=True,
        num_threads=4)
    features_map = {'images': images_batch}
    return features_map, labels_batch

  return _input_fn

data = tf.contrib.learn.datasets.mnist.load_mnist()

train_input_fn = get_input_fn(data.train, batch_size=256)
eval_input_fn = get_input_fn(data.validation, batch_size=5000)

```

## 训练一个简单的线性模型

现在，我们可以使用 MNIST 数据集训练一个线性模型。我们将使用 `tf.contrib.learn.LinearClassifier` Estimator，并用 10 个类别表示 10 个数字。输入特征会形成一个 784 维密集向量，指定方式如下：

```python
image_column = tf.contrib.layers.real_valued_column('images', dimension=784)
```

用于构建、训练和评估 LinearClassifier Estimator 的完整代码如下所示：

```python
import time

# Specify the feature(s) to be used by the estimator.
image_column = tf.contrib.layers.real_valued_column('images', dimension=784)
estimator = tf.contrib.learn.LinearClassifier(feature_columns=[image_column], n_classes=10)

# Train.
start = time.time()
estimator.fit(input_fn=train_input_fn, steps=2000)
end = time.time()
print('Elapsed time: {} seconds'.format(end - start))

# Evaluate and report metrics.
eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
print(eval_metrics)
```
下表总结了使用评估数据评估的结果。

metric        | value
:------------ | :------------
loss          | 0.25 to 0.30
accuracy      | 92.5%
training time | ~25 seconds on my machine

> 注意：指标会因各种因素而异。

除了调整（训练）批次大小和训练步数之外，您还可以微调一些其他参数。例如，您可以更改用于最小化损失的优化方法，只需明确从可用优化器集合中选择其他优化器即可。例如，以下代码构建的 LinearClassifier Estimator 使用了 Follow-The-Regularized-Leader (FTRL) 优化策略，并采用特定的学习速率和 L2 正则化。


```python
optimizer = tf.train.FtrlOptimizer(learning_rate=5.0, l2_regularization_strength=1.0)
estimator = tf.contrib.learn.LinearClassifier(
    feature_columns=[image_column], n_classes=10, optimizer=optimizer)
```

无论参数的值如何，线性模型可在此数据集上实现的准确率上限约为 **93%**。

## 结合使用显式核映射和线性模型。

线性模型在 MNIST 数据集上的错误率相对较高（约 7%）表明输入数据不是可线性分隔的。我们将使用显式核映射减少分类错误。

**直觉：** 大概的原理是，使用非线性映射将输入空间转换为其他特征空间（可能是更高维度的空间，其中转换的特征几乎是可线性分隔的），然后对映射的特征应用线性模型。如下图所示：

<div style="text-align:center">
<img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/kernel_mapping.png" />
</div>

<!-- todo latex (by jingxiongzhu)
### 技术详情

在本示例中，我们将使用 Rahimi 和 Recht 所著的论文
["Random Features for Large-Scale Kernel Machines"](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)
（大型核机器的随机特征）中介绍的随机傅里叶特征来映射输入数据。随机傅里叶特征通过以下映射将向量 \\(\mathbf{x} \in \mathbb{R}^d\\) 映射到 \\(\mathbf{x'} \in \mathbb{R}^D\\)：

$$
RFFM(\cdot): \mathbb{R}^d \to \mathbb{R}^D, \quad
RFFM(\mathbf{x}) =  \cos(\mathbf{\Omega} \cdot \mathbf{x}+ \mathbf{b})
$$

where \\(\mathbf{\Omega} \in \mathbb{R}^{D \times d}\\),
\\(\mathbf{x} \in \mathbb{R}^d,\\) \\(\mathbf{b} \in \mathbb{R}^D\\) and the
cosine is applied element-wise.

In this example, the entries of \\(\mathbf{\Omega}\\) and \\(\mathbf{b}\\) are
sampled from distributions such that the mapping satisfies the following
property:

$$
RFFM(\mathbf{x})^T \cdot RFFM(\mathbf{y}) \approx
e^{-\frac{\|\mathbf{x} - \mathbf{y}\|^2}{2 \sigma^2}}
$$

The right-hand-side quantity of the expression above is known as the RBF (or
Gaussian) kernel function. This function is one of the most-widely used kernel
functions in Machine Learning and implicitly measures similarity in a different,
much higher dimensional space than the original one. See
[Radial basis function kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)
for more details.
-->

### 核分类器

`tf.contrib.kernel_methods.KernelLinearClassifier` 是预封装的 `tf.contrib.learn` Estimator，集显式核映射和线性模型的强大功能于一身。其构造函数与 LinearClassifier Estimator 的构造函数几乎完全相同，但前者还可以指定要应用到分类器使用的每个特征的一系列显式核映射。以下代码段演示了如何将 LinearClassifier 替换为 KernelLinearClassifier。


```python
# Specify the feature(s) to be used by the estimator. This is identical to the
# code used for the LinearClassifier.
image_column = tf.contrib.layers.real_valued_column('images', dimension=784)
optimizer = tf.train.FtrlOptimizer(
   learning_rate=50.0, l2_regularization_strength=0.001)


kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
  input_dim=784, output_dim=2000, stddev=5.0, name='rffm')
kernel_mappers = {image_column: [kernel_mapper]}
estimator = tf.contrib.kernel_methods.KernelLinearClassifier(
   n_classes=10, optimizer=optimizer, kernel_mappers=kernel_mappers)

# Train.
start = time.time()
estimator.fit(input_fn=train_input_fn, steps=2000)
end = time.time()
print('Elapsed time: {} seconds'.format(end - start))

# Evaluate and report metrics.
eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
print(eval_metrics)
```
传递到 `KernelLinearClassifier` 的唯一额外参数是一个字典，表示从 feature_columns 到要应用到相应特征列的核映射列表的映射。以下行指示分类器先使用随机傅里叶特征将初始的 784 维图像映射到 2000 维向量，然后在转换的向量上应用线性模型：

```python
kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
  input_dim=784, output_dim=2000, stddev=5.0, name='rffm')
kernel_mappers = {image_column: [kernel_mapper]}
estimator = tf.contrib.kernel_methods.KernelLinearClassifier(
   n_classes=10, optimizer=optimizer, kernel_mappers=kernel_mappers)
```
请注意 `stddev` 参数。它是近似 RBF 核的标准偏差 (

)，可以控制用于分类的相似性指标。`stddev` 通常通过微调超参数确定。

下表总结了运行上述代码的结果。我们可以通过增加映射的输出维度以及微调标准偏差，进一步提高准确率。

metric        | value
:------------ | :------------
loss          | 0.10
accuracy      | 97%
training time | ~35 seconds on my machine


### 标准偏差

分类质量与标准偏差的值密切相关。下表显示了分类器在具有不同标准偏差值的评估数据上达到的准确率。最优值为标准偏差=5.0。注意标准偏差值过小或过大会如何显著降低分类的准确率。

stddev | eval accuracy
:----- | :------------
1.0    | 0.1362
2.0    | 0.4764
4.0    | 0.9654
5.0    | 0.9766
8.0    | 0.9714
16.0   | 0.8878

### 输出维度

直观地来讲，映射的输出维度越大，两个映射向量的内积越逼近核，这通常意味着分类准确率越高。换一种思路就是，输出维度等于线性模型的权重数；此维度越大，模型的“自由度”就越高。不过，超过特定阈值后，输出维度的增加只能让准确率获得极少的提升，但却会导致训练时间更长。下面的两个图表展示了这一情况，分别显示了评估准确率与输出维度和训练时间之间的函数关系。

![image](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/acc_vs_outdim.png)
![image](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/acc-vs-trn_time.png)


## 总结

显式核映射结合了非线性模型的预测能力和线性模型的可扩展性。与传统的双核方法不同，显式核方法可以扩展到数百万或数亿个样本。使用显式核映射时，请注意以下提示：

- 随机傅立叶特征对具有密集特征的数据集尤其有效。
- 核映射的参数通常取决于数据。模型质量与这些参数密切相关。通过微调超参数可找到最优值。
- 如果您有多个数值特征，不妨将它们合并成一个多维特征，然后向合并后的向量应用核映射。

