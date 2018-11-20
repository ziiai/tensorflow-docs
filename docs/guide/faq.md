# 常见问题解答

本文档解答了关于 TensorFlow 的一些常见问题。如果您的问题并未在本文中得到解答，建议您访问 TensorFlow 的[社区资源](/docs/tensorflow/about/index)
，也许能找到答案。

## 功能和兼容性

#### 我是否可以在多台计算机上运行分布式训练？

是的！TensorFlow 在 0.8 版中增加了
[对分布式计算的支持](/docs/tensorflow/deploy/distributed)
。TensorFlow 现在支持在一台或多台计算机上使用多个设备（CPU 和 GPU）。

#### TensorFlow 是否支持 Python 3？

从 0.6.0 版发布之时起（2015 年 12 月初），我们便支持 Python 3.3+。

## 构建 TensorFlow 图

另请参阅
[关于构建图的 API 文档](/docs/tensorflow/api_guides/python/framework).

#### 为什么 `c = tf.matmul(a, b)` 不立即执行矩阵乘法？

在 TensorFlow Python API 中，`a`、`b` 和 `c` 是 `tf.Tensor` 对象。`Tensor` 对象是指令结果的符号句柄，但它实际上并不存放指令的输出值。相反，TensorFlow 鼓励用户以数据流图的形式构建复杂表达式（例如整个神经网络及其梯度）。然后，您将整个数据流图（或它的子图）的计算部分分流给 `tf.Session`，相比逐个执行操作，此方法能够更加高效地执行整个计算过程。

#### 设备是怎样命名的？

对于 CPU 设备，支持的设备名称是 `"/device:CPU:0"`（或 `"/cpu:0"`）；对于第 i 个 GPU 设备，支持的设备名称是 `"/device:GPU:i"`（或 `"/gpu:i"`）。

#### 如何在特定设备上分配指令？

要在设备上分配一组操作，请在 `with tf.device(name):` 上下文中创建这些操作。请参阅关于
[搭配 GPU 使用 TensorFlow](/docs/tensorflow/guide/using_gpu)
的操作方法文档，详细了解 TensorFlow 如何将操作分配给设备；另请参阅
[CIFAR-10 教程](/docs/tensorflow/tutorials/images/deep_cnn)
，查看使用了多个 GPU 的示例模型。


## 运行 TensorFlow 计算

另请参阅
[关于运行图的 API 文档](/docs/tensorflow/api_guides/python/client).

#### 什么是供给数据 (feeding) 和占位符？

数据馈送是 TensorFlow Session API 中的一种机制，该机制允许您在运行时为一个或多个张量替换不同的值。`feed_dict` 是 `tf.Session.run` 的一个参数，该参数是一个字典，可将 `tf.Tensor` 对象映射至 NumPy 数组（以及其他一些类型），这些数组将在执行某个步的过程中用作这些张量的值。

#### What is the difference between `Session.run()` and `Tensor.eval()`?

如果 `t` 是 `tf.Tensor` 对象，则 `tf.Tensor.eval` 是 `tf.Session.run` 的简写形式（其中 `sess` 是当前的 `tf.get_default_session`）。以下两段代码是等效的：

```python
# Using `Session.run()`.
sess = tf.Session()
c = tf.constant(5.0)
print(sess.run(c))

# Using `Tensor.eval()`.
c = tf.constant(5.0)
with tf.Session():
  print(c.eval())
```

在第二个示例中，会话充当了
[上下文管理器](https://docs.python.org/2.7/reference/compound_stmts.html#with),
，这会导致该会话在 `with` 代码块的生命周期内被安装为默认会话。对于简单的使用情形（比如单元测试），使用上下文管理器可让代码变得更加简洁；如果您的代码要处理多个图和会话，明确调用 `Session.run()` 可能是一种更加直观的方法。

#### 会话是否有生命周期？中间张量呢？

会话可以拥有资源，比如 `tf.Variable`、`tf.QueueBase` 和 `tf.ReaderBase`。这些资源有时可能会占用大量内存，并且通过调用 `tf.Session.close` 关闭会话后，这些资源将被释放。

在调用 `Session.run()` 的过程中创建的中间张量会在调用结束时或结束之前被释放。

#### 运行时是否会并行处理图执行的多个环节？

TensorFlow 运行时会在多个不同的维度上并行处理图执行的多个环节：

* 单个操作具有并行实现（在 CPU 中使用多个核或在 GPU 中使用多个线程）。
* TensorFlow 图中的独立节点可在多台设备上并行运行，因此可以
  [使用多个 GPU 加快 CIFAR-10 训练](/docs/tensorflow/tutorials/images/deep_cnn).
* Session API 允许多个步并发进行（即并行调用 `tf.Session.run`）。这使得运行时可以获得更高的吞吐量，前提是单个步不会耗尽计算机的所有资源。

#### TensorFlow 支持哪些客户端语言？

TensorFlow 旨在支持多种客户端语言。目前最受支持的客户端语言是 Python。此外，也为 C++、Java 和 Go 提供了用于执行和构建图的试验性接口。

TensorFlow 还具有
[基于 C 语言的客户端 API](https://www.tensorflow.org/code/tensorflow/c/c_api.h)
，方便为更多的客户端语言建立支持。我们诚邀您贡献新的语言绑定件。

开源社区创建和支持的其他各种语言（比如 [C#](https://github.com/migueldeicaza/TensorFlowSharp), [Julia](https://github.com/malmaud/TensorFlow.jl), [Ruby](https://github.com/somaticio/tensorflow.rb) 和 [Scala](https://github.com/eaplatanios/tensorflow_scala)）的绑定以 C API 为基础，而 C API 由 TensorFlow 维护人员负责提供支持。

#### DTensorFlow 是否会使用我的计算机上可用的所有设备（GPU 和 CPU）？

TensorFlow 支持多个 GPU 和 CPU。请参阅关于
[搭配 GPU 使用 TensorFlow](/docs/tensorflow/guide/using_gpu)
的操作方法文档，详细了解 TensorFlow 如何将操作分配给设备；另请参阅
[CIFAR-10 教程](/docs/tensorflow/tutorials/images/deep_cnn)，查看使用了多个 GPU 的示例模型。

请注意，TensorFlow 仅使用计算能力高于 3.5 的 GPU 设备。

#### 使用读取器或队列时，为什么 `Session.run()` 挂起？

`tf.ReaderBase` 和 `tf.QueueBase` 类提供了特殊的操作，这些操作可以在有输入（或有界队列中有可用空间）之前实现屏蔽。这些操作使您可以构建复杂的输入管道，代价则是 TensorFlow 计算的复杂性有所增加。请参阅有关使用 `QueueRunner` 对象驱动队列和读取器的操作方法文档，详细了解如何使用这些操作。

## 变量

另请参阅有关[变量](/docs/tensorflow/guide/variables)的操作方法文档和变量的 API 文档。

#### 什么是变量的生命周期？

在会话中为变量首次运行 `tf.Variable.initializer` 操作时，即会创建该变量。运行 `tf.Session.close` 后，将销毁该变量。

#### 并发访问变量时，变量会表现出怎样的行为？

变量允许并发执行读取和写入指令。如果在对变量进行更新的同时读取变量，从变量读取的值可能会发生改变。默认情况下，系统允许对变量分配并发指令，并且指令不会互斥。要在向变量赋值时获得锁，请将 `use_locking=True` 传递给 `tf.Variable.assign`。

## 张量形状

另请参阅 `tf.TensorShape`。

#### 如何确定 Python 中的张量形状？

在 TensorFlow 中，张量同时具有静态（推理）形状和动态（真实）形状。可使用 `tf.Tensor.get_shape` 方法读取静态形状：此形状是从用于创建该张量的推理得出的，可能只是部分完成。如果静态形状未完全定义，则可以通过评估 `tf.shape(t)` 来确定 `Tensor t` 的动态形状。

#### `x.set_shape()` 和 `x = tf.reshape(x)` 之间的区别是什么？

`tf.Tensor.set_shape` 方法会更新 `Tensor` 对象的静态形状，通常用于在无法直接推理形状信息时提供额外的形状信息。它不会更改张量的动态形状。

`tf.reshape` 操作可创建具有不同动态形状的新张量。

#### 如何构建可以处理大小不同的批次数据的图？

通常，构建可以处理大小不同的批次数据的图很有用，以便将同一代码用于（小）批次训练和单实例推理。生成的图可另存为协议缓冲区并导入到其他程序中。

构建具有可变大小的图时，请务必谨记：不要将批量大小编码为 Python 常量，而应使用符号 `Tensor` 来表示批量大小。以下提示可能对您有所帮助：

- 使用 `batch_size = tf.shape(input)[0]` 从名为 `input` 的 `Tensor` 提取批次大小信息，并将其存储在名为 `batch_size` 的 `Tensor` 中。

- 使用 `tf.reduce_mean` 而非 `tf.reduce_sum(...) / batch_size`。


## TensorBoard

#### 如何直观展示 TensorFlow 图？

请参阅[图的直观展示教程](/docs/tensorflow/guide/graph_viz)。

#### W要将数据发送到 TensorBoard，最简单的方法是什么？

将总结 op 添加到您的 TensorFlow 图中，然后将这些总结写入日志目录。然后，使用以下命令启动 TensorBoard：

    python tensorflow/tensorboard/tensorboard.py --logdir=path/to/log-directory

有关详情，请参阅
[总结和 TensorBoard 教程](/docs/tensorflow/guide/summaries_and_tensorboard)。

#### 每次启动 TensorBoard 时，我都会看到一个网络安全弹出窗口！

您可以通过标记 --host=localhost 将 TensorBoard 更改为在本地主机上提供服务，而非在“0.0.0.0”上提供服务。执行此操作后，安全警告便不会再出现。

## 扩展 TensorFlow

请参阅有关[将新指令添加到 TensorFlow](/docs/tensorflow/extend/adding_an_op) 的操作方法文档。

#### 我的数据采用了自定义格式。如何使用 TensorFlow 读取该数据？

目前主要有三种方法来处理使用自定义格式的数据。

最简单的方法是使用 Python 编写解析代码，将数据转换为 NumPy 数组，然后使用 `tf.data.Dataset.from_tensor_slices` 根据内存中的数据创建输入管道。

如果您的数据无法加载到内存中，请尝试在数据集管道中执行解析。首先选择适当的文件读取器，例如 `tf.data.TextLineDataset`，然后在数据集上映射适当的操作以转换数据集。首选预定义的 TensorFlow 操作，例如 `tf.decode_raw`、`tf.decode_csv`、`tf.parse_example` 或 `tf.image.decode_png`。

如果无法使用内置的 TensorFlow 操作轻松解析数据，可考虑将数据离线转换为易于解析的格式，例如 `TFRecord`。

要自定义解析行为，最高效的方法是
[添加使用 C++ 编写的新操作](/docs/tensorflow/extend/adding_an_op)
并使用该操作解析数据格式。有关相关操作步骤的更多信息，请参阅
[新数据格式处理指南](/docs/tensorflow/extend/new_data_formats)。


## 其他

#### TensorFlow 的编码样式规范是什么？

TensorFlow Python API 遵守
[PEP8](https://www.python.org/dev/peps/pep-0008/)
规范。*具体来讲，对于类，我们使用 `CamelCase` 名称，对于函数、方法和属性，我们使用 `snake_case` 名称。我们也遵循
[Google Python 样式指南](https://google.github.io/styleguide/pyguide.html).

TensorFlow C++ 代码库遵循
[Google C++ 样式指南](https://google.github.io/styleguide/cppguide.html).

（*有一个例外情况：我们使用 2 空格缩进，而非 4 空格缩进。）

