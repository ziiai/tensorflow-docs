#  Estimator 的数据集

`tf.data` 模块包含一系列类，可让您轻松地加载数据、操作数据并通过管道将数据传送到模型中。本文档通过两个简单的示例来介绍该 API：

- 从 Numpy 数组中读取内存中的数据。
- 从 csv 文件中读取行。


<!-- TODO(markdaoust): Add links to an example reading from multiple-files
(image_retraining), and a from_generator example. -->

## 基本输入

要开始使用 `tf.data`，最简单的方法是从数组中提取切片。

 [预创建的 Estimator](/docs/tensorflow/guide/premade_estimators)一章介绍了 [`iris_data.py`](https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py) 中的以下 `train_input_fn`，它可以通过管道将数据传输到 Estimator 中：

``` python
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
```

我们来详细了解一下。

### 参数

此函数需要三个参数。要求所赋值为“数组”的参数能够接受可通过 `numpy.array` 转换成数组的几乎任何值。其中存在一个例外，即对 `Datasets` 有特殊意义的 `tuple`，稍后我们会发现这一点。

- `features`：包含原始输入特征的 `{'feature_name':array}` 字典（或 `DataFrame`）。
- `labels`：包含每个样本的标签的数组。
- `batch_size`：表示所需批次大小的整数。

在 [`premade_estimator.py`](https://github.com/tensorflow/models/blob/master/samples/core/get_started/premade_estimator.py) 中，我们使用 `iris_data.load_data()` 函数检索了鸢尾花数据。您可以运行该函数并解压结果，如下所示：

``` python
import iris_data

# Fetch the data
train, test = iris_data.load_data()
features, labels = train
```

然后，我们使用类似以下内容的行将此数据传递给了输入函数：

``` python
batch_size=100
iris_data.train_input_fn(features, labels, batch_size)
```

下面我们详细介绍一下 `train_input_fn()`。

### 切片

首先，此函数会利用 `tf.data.Dataset.from_tensor_slices` 函数创建一个代表数组切片的 `tf.data.Dataset`。系统会在第一个维度内对该数组进行切片。例如，一个包含 MNIST 训练数据的数组的形状为 `(60000, 28, 28)`。将该数组传递给 `from_tensor_slices` 会返回一个包含 60000 个切片的 `Dataset` 对象，其中每个切片都是一个 28x28 的图像。

返回此 `Dataset` 的代码如下所示：

``` python
train, test = tf.keras.datasets.mnist.load_data()
mnist_x, mnist_y = train

mnist_ds = tf.data.Dataset.from_tensor_slices(mnist_x)
print(mnist_ds)
```
这段代码将输出以下行，显示数据集中条目的[形状](/docs/tensorflow/guide/tensors#shapes)和[类型](/docs/tensorflow/guide/tensors#data_types)。请注意，`Dataset` 不知道自己包含多少条目。

``` None
<TensorSliceDataset shapes: (28,28), types: tf.uint8>
```

上面的 Dataset 表示一组简单的数组，但实际的数据集要比这复杂得多。Dataset 可以透明地处理字典或元组（或
[`namedtuple`](https://docs.python.org/2/library/collections.html#collections.namedtuple)）的任何嵌套组合。

例如，在将鸢尾花 `features` 转换为标准 Python 字典后，您可以将数组字典转换为字典 `Dataset`，如下所示：

``` python
dataset = tf.data.Dataset.from_tensor_slices(dict(features))
print(dataset)
```
``` None
<TensorSliceDataset

  shapes: {
    SepalLength: (), PetalWidth: (),
    PetalLength: (), SepalWidth: ()},

  types: {
      SepalLength: tf.float64, PetalWidth: tf.float64,
      PetalLength: tf.float64, SepalWidth: tf.float64}
>
```

我们可以看到，如果 `Dataset` 包含结构化元素，则 `Dataset` 的 `shapes` 和 `types` 将采用同一结构。此数据集包含标量（类型均为 `tf.float64`）字典。

鸢尾花 `train_input_fn` 的第一行使用相同的功能，但添加了另一层结构。它会创建一个包含 `(features_dict, label)` 对的数据集。

以下代码显示标签是类型为 `int64` 的标量：

``` python
# Convert the inputs to a Dataset.
dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
print(dataset)
```
```
<TensorSliceDataset
    shapes: (
        {
          SepalLength: (), PetalWidth: (),
          PetalLength: (), SepalWidth: ()},
        ()),

    types: (
        {
          SepalLength: tf.float64, PetalWidth: tf.float64,
          PetalLength: tf.float64, SepalWidth: tf.float64},
        tf.int64)>
```

### 操作

目前，`Dataset` 会按固定顺序迭代数据一次，并且一次仅生成一个元素。它需要进一步处理才可用于训练。幸运的是，`tf.data.Dataset` 类提供了更好地准备训练数据的方法。输入函数的下一行就利用了其中的几种方法：

``` python
# Shuffle, repeat, and batch the examples.
dataset = dataset.shuffle(1000).repeat().batch(batch_size)
```

`shuffle` 方法使用一个固定大小的缓冲区，在条目经过时随机化处理条目。在这种情况下，`buffer_size` 大于 `Dataset` 中样本的数量，确保数据完全被随机化处理（鸢尾花数据集仅包含 150 个样本）。

`repeat` 方法会在结束时重启 `Dataset`。要限制周期数量，请设置 `count` 参数。

`batch` 方法会收集大量样本并将它们堆叠起来以创建批次。这为批次的形状增加了一个维度。新的维度将添加为第一个维度。以下代码对之前的 MNIST `Dataset` 使用 `batch` 方法。这样会产生一个包含表示 `(28,28)` 图像堆叠的三维数组的 `Dataset`：

``` python
print(mnist_ds.batch(100))
```

``` none
<BatchDataset
  shapes: (?, 28, 28),
  types: tf.uint8>
```
请注意，该数据集的批次大小是未知的，因为最后一个批次具有的元素数量会减少。

在 `train_input_fn` 中，经过批处理之后，`Dataset` 包含元素的一维向量，其中每个标量之前如下所示：

```python
print(dataset)
```
```
<TensorSliceDataset
    shapes: (
        {
          SepalLength: (?,), PetalWidth: (?,),
          PetalLength: (?,), SepalWidth: (?,)},
        (?,)),

    types: (
        {
          SepalLength: tf.float64, PetalWidth: tf.float64,
          PetalLength: tf.float64, SepalWidth: tf.float64},
        tf.int64)>
```


### 返回

此时，`Dataset` 包含 `(features_dict, labels)` 对。这是 `train` 和 `evaluate` 方法的预期格式，因此 `input_fn` 会返回相应的数据集。

使用 `predict` 方法时，可以/应该忽略 `labels`。

<!--
  TODO(markdaoust): link to `input_fn` doc when it exists
-->


## 读取 CSV 文件

`Dataset` 类最常见的实际用例是流式传输磁盘上文件中的数据。`tf.data` 模块包含各种文件阅读器。我们来看看如何使用 `Dataset` 解析 csv 文件中的 Iris 数据集。

对 `iris_data.maybe_download` 函数的以下调用会根据需要下载数据，并返回所生成文件的路径名：

``` python
import iris_data
train_path, test_path = iris_data.maybe_download()
```
[`iris_data.csv_input_fn`](https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py) 函数包含使用 `Dataset` 解析 csv 文件的备用实现。

我们来了解一下如何构建从本地文件读取数据且兼容 Estimator 的输入函数。

### 构建 Dataset

我们先构建一个 `TextLineDataset` 对象，实现一次读取文件中的一行数据。然后，我们调用 `skip` 方法来跳过文件的第一行，此行包含标题，而非样本：

``` python
ds = tf.data.TextLineDataset(train_path).skip(1)
```

### 构建 csv 行解析器

我们先构建一个解析单行的函数。

以下 `iris_data.parse_line` 函数会使用 `tf.decode_csv` 函数和一些简单的 Python 代码来完成此任务：

为了生成必要的 `(features, label)` 对，我们必须解析数据集中的每一行。以下 `_parse_line` 函数会调用 `tf.decode_csv`，以将一行解析为特征和标签两个部分。由于 Estimator 需要将特征表示为字典，因此我们依靠 Python 的内置 `dict` 和 `zip` 函数来构建此字典。特征名称是该字典的键。然后，我们调用字典的 `pop` 方法以从特征字典中移除标签字段：

``` python
# Metadata describing the text columns
COLUMNS = ['SepalLength', 'SepalWidth',
           'PetalLength', 'PetalWidth',
           'label']
FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0]]
def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, FIELD_DEFAULTS)

    # Pack the result into a dictionary
    features = dict(zip(COLUMNS,fields))

    # Separate the label from the features
    label = features.pop('label')

    return features, label
```

### 解析行

数据集提供很多用于在通过管道将数据传送到模型的过程中处理数据的方法。最常用的方法是 `map`，它会对 `Dataset` 的每个元素应用转换。

`map` 方法会接受 `map_func` 参数，此参数描述了应该如何转换 `Dataset` 中的每个条目。

<div style="width:80%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/map.png">
</div>
<div style="text-align: center">
 map 方法运用 `map_func` 来转换 Dataset 中的每个条目。 
</div>

因此，为了在从 csv 文件中流式传出行时对行进行解析，我们将 `_parse_line` 函数传递给 `map` 方法：

``` python
ds = ds.map(_parse_line)
print(ds)
```
``` None
<MapDataset
shapes: (
    {SepalLength: (), PetalWidth: (), ...},
    ()),
types: (
    {SepalLength: tf.float32, PetalWidth: tf.float32, ...},
    tf.int32)>
```

现在，数据集包含 `(features, label)` 对，而不是简单的标量字符串。

`iris_data.csv_input_fn` 函数的剩余部分与 `iris_data.train_input_fn` 函数完全相同，后者在[基本输入](#basic_input)部分中进行了介绍。

### 试试看

此函数可用于替换 `iris_data.train_input_fn`。可使用此函数馈送 Estimator，如下所示：

``` python
train_path, test_path = iris_data.maybe_download()

# All the inputs are numeric
feature_columns = [
    tf.feature_column.numeric_column(name)
    for name in iris_data.CSV_COLUMN_NAMES[:-1]]

# Build the estimator
est = tf.estimator.LinearClassifier(feature_columns,
                                    n_classes=3)
# Train the estimator
batch_size = 100
est.train(
    steps=1000,
    input_fn=lambda : iris_data.csv_input_fn(train_path, batch_size))
```

Estimator 要求 `input_fn` 不接受任何参数。为了不受此限制约束，我们使用 `lambda` 来获取参数并提供所需的接口。

## 总结

`tf.data` 模块提供一系列类和函数，可用于轻松从各种来源读取数据。此外，`tf.data` 还提供简单而又强大的方法，用于应用各种标准和自定义转换。

现在，您已经基本了解了如何高效地将数据加载到 Estimator 中。接下来，请查看下列文档：


* [创建自定义 Estimator](/docs/tensorflow/guide/custom_estimators)：展示了如何自行构建自定义 `Estimator` 模型。
* [低阶 API 简介](/docs/tensorflow/guide/low_level_intro#datasets)：展示了如何使用 TensorFlow 的低阶 API 直接尝试 `tf.data.Datasets`。
* [导入数据](/docs/tensorflow/guide/datasets)：详细介绍了 `Datasets` 的其他功能。

