#  导入数据

借助 `tf.data` API，您可以根据简单的可重用片段构建复杂的输入管道。例如，图片模型的管道可能会汇聚分布式文件系统中的文件中的数据、对每个图片应用随机扰动，并将随机选择的图片合并成用于训练的批次。文本模型的管道可能包括从原始文本数据中提取符号、根据对照表将其转换为嵌入标识符，以及将不同长度的序列组合成批次数据。使用 `tf.data` API 可以轻松处理大量数据、不同的数据格式以及复杂的转换。

`tf.data` API 在 TensorFlow 中引入了两个新的抽象类：

- `tf.data.Dataset` 表示一系列元素，其中每个元素包含一个或多个 Tensor 对象。例如，在图像管道中，元素可能是单个训练样本，具有一对表示图像数据和标签的张量。可以通过两种不同的方式来创建数据集：

    - 创建**来源**（例如 `Dataset.from_tensor_slices()`），以通过一个或多个 `tf.Tensor` 对象构建数据集。

    - 应用**转换**（例如 `Dataset.batch()`），以通过一个或多个 `tf.data.Dataset` 对象构建数据集。

- `tf.data.Iterator` 提供了从数据集中提取元素的主要方法。`Iterator.get_next()` 返回的操作会在执行时生成 `Dataset` 的下一个元素，并且此操作通常充当输入管道代码和模型之间的接口。最简单的迭代器是“单次迭代器”，它与特定的 `Dataset` 相关联，并对其进行一次迭代。要实现更复杂的用途，您可以通过 `Iterator.initializer` 操作使用不同的数据集重新初始化和参数化迭代器，这样一来，您就可以在同一个程序中对训练和验证数据进行多次迭代（举例而言）。

## 基本机制

本指南的这一部分介绍了创建不同种类的 `Dataset` 和 `Iterator` 对象的基础知识，以及如何从这些对象中提取数据。

要启动输入管道，您必须定义来源。例如，要通过内存中的某些张量构建 `Dataset`，您可以使用 `tf.data.Dataset.from_tensors()` 或 `tf.data.Dataset.from_tensor_slices()`。或者，如果您的输入数据以推荐的 TFRecord 格式存储在磁盘上，那么您可以构建 `tf.data.TFRecordDataset`。

一旦有了 `Dataset` 对象，可以将其转换为新的 `Dataset`，方法是链接 `tf.data.Dataset` 对象上的方法调用。例如，您可以应用单元素转换，例如 `Dataset.map()`（为每个元素应用一个函数），也可以应用多元素转换（例如 Dataset.batch()）。要了解转换的完整列表，请参阅 `tf.data.Dataset` 的文档。

使用 `Dataset` 中值的最常见方法是构建迭代器对象。通过此对象，可以一次访问数据集中的一个元素（例如通过调用 `Dataset.make_one_shot_iterator()`）。`tf.data.Iterator` 提供了两个操作：Iterator.initializer，您可以通过此操作（重新）初始化迭代器的状态；以及 `Iterator.get_next()`，此操作返回对应于有符号下一个元素的 `tf.Tensor` 对象。根据您的使用情形，您可以选择不同类型的迭代器，下文介绍了具体选项。

### 数据集结构

一个数据集包含多个元素，每个元素的结构都相同。一个元素包含一个或多个 `tf.Tensor` 对象，这些对象称为组件。每个组件都有一个 `tf.DType`，表示张量中元素的类型；以及一个 `tf.TensorShape，表示每个元素（可能部分指定）的静态形状。您可以通过 Dataset.output_types` 和 `Dataset.output_shapes` 属性检查数据集元素各个组件的推理类型和形状。这些属性的嵌套结构映射到元素的结构，此元素可以是单个张量、张量元组，也可以是张量的嵌套元组。例如：

```python
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"
```

为元素的每个组件命名通常会带来便利性，例如，如果它们表示训练样本的不同特征。除了元组之外，还可以使用 `collections.namedtuple` 或将字符串映射到张量的字典来表示 `Dataset` 的单个元素。

```python
dataset = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
```

`Dataset` 转换支持任何结构的数据集。在使用 `Dataset.map()`、`Dataset.flat_map()` 和 `Dataset.filter()` 转换时（这些转换会对每个元素应用一个函数），元素结构决定了函数的参数：

```python
dataset1 = dataset1.map(lambda x: ...)

dataset2 = dataset2.flat_map(lambda x, y: ...)

# Note: Argument destructuring is not available in Python 3.
dataset3 = dataset3.filter(lambda x, (y, z): ...)
```

### 创建迭代器

构建了表示输入数据的 `Dataset` 后，下一步就是创建 `Iterator` 来访问该数据集中的元素。`tf.data` API 目前支持下列迭代器，复杂程度逐渐增大：

- 单次，
- 可初始化，
- 可重新初始化，以及
- 可馈送。

单次迭代器是最简单的迭代器形式，仅支持对数据集进行一次迭代，不需要显式初始化。单次迭代器可以处理基于队列的现有输入管道支持的几乎所有情况，但它们不支持参数化。以 `Dataset.range()` 为例：

```python
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(100):
  value = sess.run(next_element)
  assert i == value
```

> 注意：目前，单次迭代器是唯一易于与 Estimator 搭配使用的类型。

您需要先运行显式 `iterator.initializer` 操作，然后才能使用**可初始化**迭代器。虽然有些不便，但它允许您使用一个或多个 `tf.placeholder()` 张量（可在初始化迭代器时馈送）参数化数据集的定义。继续以 `Dataset.range()` 为例：

```python
max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Initialize an iterator over a dataset with 10 elements.
sess.run(iterator.initializer, feed_dict={max_value: 10})
for i in range(10):
  value = sess.run(next_element)
  assert i == value

# Initialize the same iterator over a dataset with 100 elements.
sess.run(iterator.initializer, feed_dict={max_value: 100})
for i in range(100):
  value = sess.run(next_element)
  assert i == value
```

**可重新初始化**迭代器可以通过多个不同的 `Dataset` 对象进行初始化。例如，您可能有一个训练输入管道，它会对输入图片进行随机扰动来改善泛化；还有一个验证输入管道，它会评估对未修改数据的预测。这些管道通常会使用不同的 `Dataset` 对象，这些对象具有相同的结构（即每个组件具有相同类型和兼容形状）。

```python
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `training_dataset`
# or `validation_dataset` here, because they are compatible.
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# Run 20 epochs in which the training dataset is traversed, followed by the
# validation dataset.
for _ in range(20):
  # Initialize an iterator over the training dataset.
  sess.run(training_init_op)
  for _ in range(100):
    sess.run(next_element)

  # Initialize an iterator over the validation dataset.
  sess.run(validation_init_op)
  for _ in range(50):
    sess.run(next_element)
```

**可馈送**迭代器可以与 `tf.placeholder` 一起使用，以选择所使用的 `Iterator`（在每次调用 `tf.Session.run` 时）（通过熟悉的 `feed_dict` 机制）。它提供的功能与可重新初始化迭代器的相同，但在迭代器之间切换时不需要从数据集的开头初始化迭代器。例如，以上面的同一训练和验证数据集为例，您可以使用 `tf.data.Iterator.from_string_handle` 定义一个可让您在两个数据集之间切换的可馈送迭代器：

```python
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.data.Dataset.range(50)

# A feedable iterator is defined by a handle placeholder and its structure. We
# could use the `output_types` and `output_shapes` properties of either
# `training_dataset` or `validation_dataset` here, because they have
# identical structure.
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

# You can use feedable iterators with a variety of different kinds of iterator
# (such as one-shot and initializable iterators).
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

# The `Iterator.string_handle()` method returns a tensor that can be evaluated
# and used to feed the `handle` placeholder.
training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

# Loop forever, alternating between training and validation.
while True:
  # Run 200 steps using the training dataset. Note that the training dataset is
  # infinite, and we resume from where we left off in the previous `while` loop
  # iteration.
  for _ in range(200):
    sess.run(next_element, feed_dict={handle: training_handle})

  # Run one pass over the validation dataset.
  sess.run(validation_iterator.initializer)
  for _ in range(50):
    sess.run(next_element, feed_dict={handle: validation_handle})
```

### 使用迭代器中的值

`Iterator.get_next()` 方法返回一个或多个 `tf.Tensor` 对象，这些对象对应于迭代器有符号的下一个元素。每次评估这些张量时，它们都会获取底层数据集中下一个元素的值。（请注意，与 TensorFlow 中的其他有状态对象一样，调用 Iterator.get_next() 并不会立即使迭代器进入下个状态。您必须在 TensorFlow 表达式中使用此函数返回的 `tf.Tensor` 对象，并将该表达式的结果传递到 `tf.Session.run()`，以获取下一个元素并使迭代器进入下个状态。）

如果迭代器到达数据集的末尾，则执行 `Iterator.get_next()` 操作会产生 `tf.errors.OutOfRangeError`。在此之后，迭代器将处于不可用状态；如果需要继续使用，则必须对其重新初始化。

```python
dataset = tf.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Typically `result` will be the output of a model, or an optimizer's
# training operation.
result = tf.add(next_element, next_element)

sess.run(iterator.initializer)
print(sess.run(result))  # ==> "0"
print(sess.run(result))  # ==> "2"
print(sess.run(result))  # ==> "4"
print(sess.run(result))  # ==> "6"
print(sess.run(result))  # ==> "8"
try:
  sess.run(result)
except tf.errors.OutOfRangeError:
  print("End of dataset")  # ==> "End of dataset"
```

一种常见模式是将“训练循环”封装在 `try-except` 块中：

```python
sess.run(iterator.initializer)
while True:
  try:
    sess.run(result)
  except tf.errors.OutOfRangeError:
    break
```

如果数据集的每个元素都具有嵌套结构，则 `Iterator.get_next()` 的返回值将是一个或多个 `tf.Tensor` 对象，这些对象具有相同的嵌套结构：

```python
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()

sess.run(iterator.initializer)
next1, (next2, next3) = iterator.get_next()
```

请注意，`next1`、`next2` 和 `next3` 是由同一个操作/节点（通过 `Iterator.get_next()` 创建）生成的张量。因此，评估其中任何一个张量都会使所有组件的迭代器进入下个状态。典型的迭代器消耗方会在一个表达式中包含所有组件。

### 保存迭代器状态

`tf.contrib.data.make_saveable_from_iterator` 函数通过迭代器创建一个 `SaveableObject`，该对象可用于保存和恢复迭代器（实际上是整个输入管道）的当前状态。这样创建的可保存对象可以添加到 `tf.train.Saver` 变量列表或 `tf.GraphKeys.SAVEABLE_OBJECTS` 集合中，以便采用与 `tf.Variable` 相同的方式进行保存和恢复。请参阅
[保存和恢复](/docs/tensorflow/guide/saved_model)，详细了解如何保存和恢复变量。

```python
# Create saveable object from iterator.
saveable = tf.contrib.data.make_saveable_from_iterator(iterator)

# Save the iterator state by adding it to the saveable objects collection.
tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
saver = tf.train.Saver()

with tf.Session() as sess:

  if should_checkpoint:
    saver.save(path_to_checkpoint)

# Restore the iterator state.
with tf.Session() as sess:
  saver.restore(sess, path_to_checkpoint)
```

## 读取输入数据

### 使用 NumPy 数组

如果您的所有输入数据都适合存储在内存中，则根据输入数据创建 `Dataset` 的最简单方法是将它们转换为 `tf.Tensor` 对象，并使用 `Dataset.from_tensor_slices()`。

```python
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

dataset = tf.data.Dataset.from_tensor_slices((features, labels))
```

请注意，上面的代码段会将 `features` 和 `labels` 数组作为 `tf.constant()` 指令嵌入在 TensorFlow 图中。这样非常适合小型数据集，但会浪费内存，因为会多次复制数组的内容，并可能会达到 `tf.GraphDef` 协议缓冲区的 2GB 上限。

作为替代方案，您可以根据 `tf.placeholder()` 张量定义 `Dataset`，并在对数据集初始化 `Iterator` 时馈送 NumPy 数组。

```python
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
dataset = ...
iterator = dataset.make_initializable_iterator()

sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})
```

### 使用 TFRecord 数据

`tf.data` API 支持多种文件格式，因此您可以处理那些不适合存储在内存中的大型数据集。例如，TFRecord 文件格式是一种面向记录的简单二进制格式，很多 TensorFlow 应用采用此格式来训练数据。通过 `tf.data.TFRecordDataset` 类，您可以将一个或多个 TFRecord 文件的内容作为输入管道的一部分进行流式传输。

```python
# Creates a dataset that reads all of the examples from two files.
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
```

`TFRecordDataset` 初始化程序的 `filenames` 参数可以是字符串、字符串列表，也可以是字符串 `tf.Tensor`。因此，如果您有两组分别用于训练和验证的文件，则可以使用 `tf.placeholder(tf.string)` 来表示文件名，并使用适当的文件名初始化迭代器：

```python
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)  # Parse the record into tensors.
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()

# You can feed the initializer with the appropriate filenames for the current
# phase of execution, e.g. training vs. validation.

# Initialize `iterator` with training data.
training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

# Initialize `iterator` with validation data.
validation_filenames = ["/var/data/validation1.tfrecord", ...]
sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
```

### 消耗文本数据

很多数据集都是作为一个或多个文本文件分布的。`tf.data.TextLineDataset` 提供了一种从一个或多个文本文件中提取行的简单方法。给定一个或多个文件名，`TextLineDataset` 会为这些文件的每行生成一个字符串值元素。像 `TFRecordDataset` 一样，`TextLineDataset` 将接受 `filenames`（作为 `tf.Tensor`），因此您可以通过传递 `tf.placeholder(tf.string)` 进行参数化。

```python
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]
dataset = tf.data.TextLineDataset(filenames)
```

默认情况下，`TextLineDataset` 会生成每个文件的每一行，这可能是不可取的（例如，如果文件以标题行开头或包含注释）。可以使用 `Dataset.skip()` 和 `Dataset.filter()` 转换来移除这些行。为了将这些转换分别应用于每个文件，我们使用 `Dataset.flat_map()` 为每个文件创建一个嵌套的 `Dataset`。

```python
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]

dataset = tf.data.Dataset.from_tensor_slices(filenames)

# Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
# and then concatenate their contents sequentially into a single "flat" dataset.
# * Skip the first line (header row).
# * Filter out lines beginning with "#" (comments).
dataset = dataset.flat_map(
    lambda filename: (
        tf.data.TextLineDataset(filename)
        .skip(1)
        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))
```

### 使用 CSV 数据

CSV 文件格式是用于以纯文本格式存储表格数据的常用格式。`tf.contrib.data.CsvDataset` 类提供了一种从符合 [RFC 4180](https://tools.ietf.org/html/rfc4180).
的一个或多个 CSV 文件中提取记录的方法。给定一个或多个文件名以及默认列表后，`CsvDataset` 将生成一个元素元组，元素类型对应于为每个 CSV 记录提供的默认元素类型。像 `TFRecordDataset` 和 `TextLineDataset` 一样，`CsvDataset` 将接受 `filenames`（作为 `tf.Tensor`），因此您可以通过传递 `tf.placeholder(tf.string)` 进行参数化。

```
# Creates a dataset that reads all of the records from two CSV files, each with
# eight float columns
filenames = ["/var/data/file1.csv", "/var/data/file2.csv"]
record_defaults = [tf.float32] * 8   # Eight required float columns
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)
```

如果某些列为空，则可以提供默认值而不是类型。

```
# Creates a dataset that reads all of the records from two CSV files, each with
# four float columns which may have missing values
record_defaults = [[0.0]] * 8
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)
```

默认情况下，`CsvDataset` 生成文件的每一列或每一行，这可能是不可取的；例如，如果文件以应忽略的标题行开头，或如果输入中不需要某些列。可以分别使用 `header` 和 `select_cols` 参数移除这些行和字段。

```
# Creates a dataset that reads all of the records from two CSV files with
# headers, extracting float data from columns 2 and 4.
record_defaults = [[0.0]] * 2  # Only provide defaults for the selected columns
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[2,4])
```
<!--
TODO(mrry): Add these sections.

### Consuming from a Python generator
-->

## 使用 `Dataset.map()` 预处理数据

`Dataset.map(f)` 转换通过将指定函数 `f` 应用于输入数据集的每个元素来生成新数据集。此转换基于 `map()` 函数（通常应用于函数式编程语言中的列表和其他结构）。函数 `f` 会接受表示输入中单个元素的 `tf.Tensor` 对象，并返回表示新数据集中单个元素的 `tf.Tensor` 对象。此函数的实现使用标准的 TensorFlow 指令将一个元素转换为另一个元素。

本部分介绍了如何使用 `Dataset.map()` 的常见示例。

### 解析 `tf.Example` 协议缓冲区消息

许多输入管道都从 TFRecord 格式的文件中提取 `tf.train.Example` 协议缓冲区消息（例如这种文件使用 `tf.python_io.TFRecordWriter` 编写而成）。每个 `tf.train.Example` 记录都包含一个或多个“特征”，输入管道通常会将这些特征转换为张量。

```python
# Transforms a scalar string `example_proto` into a pair of a scalar string and
# a scalar integer, representing an image and its label, respectively.
def _parse_function(example_proto):
  features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
              "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features["image"], parsed_features["label"]

# Creates a dataset that reads all of the examples from two files, and extracts
# the image and label features.
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
```

### 解码图片数据并调整其大小

在用真实的图片数据训练神经网络时，通常需要将不同大小的图片转换为通用大小，这样就可以将它们批处理为具有固定大小的数据。

```python
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

# A vector of filenames.
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0, 37, ...])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
```

### 使用 `tf.py_func()` 应用任意 Python 逻辑

为了确保性能，我们建议您尽可能使用 TensorFlow 指令预处理数据。不过，在解析输入数据时，调用外部 Python 库有时很有用。为此，请在 `Dataset.map()` 转换中调用 `tf.py_func()` 指令。

```python
import cv2

# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_py_function(filename, label):
  image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
  return image_decoded, label

# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
labels = [0, 37, 29, 1, ...]

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(
    lambda filename, label: tuple(tf.py_func(
        _read_py_function, [filename, label], [tf.uint8, label.dtype])))
dataset = dataset.map(_resize_function)
```

<!--
TODO(mrry): Add this section.

### Handling text data with unusual sizes
-->

## 批处理数据集元素

### 简单的批处理

最简单的批处理形式是将数据集中的 `n` 个连续元素堆叠为一个元素。`Dataset.batch()` 转换正是这么做的，它与 `tf.stack()` 运算符具有相同的限制（被应用于元素的每个组件）：即对于每个组件 `i`，所有元素的张量形状都必须完全相同。

```python
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])
```

### 使用填充批处理张量

上述方法适用于具有相同大小的张量。不过，很多模型（例如序列模型）处理的输入数据可能具有不同的大小（例如序列的长度不同）。为了解决这种情况，可以通过 `Dataset.padded_batch()` 转换来指定一个或多个会被填充的维度，从而批处理不同形状的张量。

```python
dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                               #      [5, 5, 5, 5, 5, 0, 0],
                               #      [6, 6, 6, 6, 6, 6, 0],
                               #      [7, 7, 7, 7, 7, 7, 7]]
```

您可以通过 `Dataset.padded_batch()` 转换为每个组件的每个维度设置不同的填充，并且可以采用可变长度（在上面的示例中用 `None` 表示）或恒定长度。也可以替换填充值，默认设置为 `0`。

<!--
TODO(mrry): Add this section.

### Dense ragged -> tf.SparseTensor
-->

## 训练工作流程

### 处理多个周期

`tf.data` API 提供了两种主要方式来处理同一数据的多个周期。

要迭代数据集多个周期，最简单的方法是使用 `Dataset.repeat()` 转换。例如，要创建一个将其输入重复 10 个周期的数据集：

```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.repeat(10)
dataset = dataset.batch(32)
```

应用不带参数的 `Dataset.repeat()` 转换将无限次地重复输入。`Dataset.repeat()` 转换将其参数连接起来，而不会在一个周期结束和下一个周期开始时发出信号。

如果您想在每个周期结束时收到信号，则可以编写在数据集结束时捕获 `tf.errors.OutOfRangeError` 的训练循环。此时，您可以收集关于该周期的一些统计信息（例如验证错误）。

```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Compute for 100 epochs.
for _ in range(100):
  sess.run(iterator.initializer)
  while True:
    try:
      sess.run(next_element)
    except tf.errors.OutOfRangeError:
      break

  # [Perform end-of-epoch calculations here.]
```

### 随机重排输入数据

`Dataset.shuffle()` 转换会使用类似于 `tf.RandomShuffleQueue` 的算法随机重排输入数据集：它会维持一个固定大小的缓冲区，并从该缓冲区统一地随机选择下一个元素。

```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.repeat()
```

### 使用高阶 API

`tf.train.MonitoredTrainingSession` API 简化了在分布式设置下运行 TensorFlow 的很多方面。`MonitoredTrainingSession` 使用 `tf.errors.OutOfRangeError` 表示训练已完成，因此要将其与 `tf.data` API 结合使用，我们建议使用 `Dataset.make_one_shot_iterator()`。例如：

```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.repeat(num_epochs)
iterator = dataset.make_one_shot_iterator()

next_example, next_label = iterator.get_next()
loss = model_function(next_example, next_label)

training_op = tf.train.AdagradOptimizer(...).minimize(loss)

with tf.train.MonitoredTrainingSession(...) as sess:
  while not sess.should_stop():
    sess.run(training_op)
```

要在 `input_fn` 中使用 `Dataset`（`input_fn` 属于 `tf.estimator.Estimator`），我们还建议使用 `Dataset.make_one_shot_iterator()`。例如：

```python
def dataset_input_fn():
  filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
  dataset = tf.data.TFRecordDataset(filenames)

  # Use `tf.parse_single_example()` to extract data from a `tf.Example`
  # protocol buffer, and perform any additional per-record preprocessing.
  def parser(record):
    keys_to_features = {
        "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
        "date_time": tf.FixedLenFeature((), tf.int64, default_value=""),
        "label": tf.FixedLenFeature((), tf.int64,
                                    default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    image = tf.image.decode_jpeg(parsed["image_data"])
    image = tf.reshape(image, [299, 299, 1])
    label = tf.cast(parsed["label"], tf.int32)

    return {"image_data": image, "date_time": parsed["date_time"]}, label

  # Use `Dataset.map()` to build a pair of a feature dictionary and a label
  # tensor for each example.
  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(32)
  dataset = dataset.repeat(num_epochs)

  # Each element of `dataset` is tuple containing a dictionary of features
  # (in which each value is a batch of values for that feature), and a batch of
  # labels.
  return dataset
```
