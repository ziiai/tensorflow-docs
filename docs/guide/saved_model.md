# 保存和恢复

`tf.train.Saver` 类提供了保存和恢复模型的方法。通过 `tf.saved_model.simple_save` 函数可轻松地保存适合投入使用的模型。 Estimator 会自动保存和恢复 `model_dir` 中的变量。

## 保存和恢复变量

TensorFlow [Variables](../guide/variables.md) 是表示由程序操作的共享持久状态的最佳方法。tf.train.Saver 构造函数会针对图中所有变量或指定列表的变量将 save 和 restore 操作添加到图中。Saver 对象提供了运行这些操作的方法，并指定写入或读取检查点文件的路径。

Saver 会恢复已经在模型中定义的所有变量。如果您在不知道如何构建图的情况下加载模型（例如，如果您要编写用于加载各种模型的通用程序），那么请阅读本文档后面的
[保存和恢复模型概述](#models) 部分。

TensorFlow 将变量保存在二进制检查点文件中，这类文件会将变量名称映射到张量值。

注意：TensorFlow 模型文件是代码。请注意不可信的代码。详情请参阅[安全地使用 TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md)。

### 保存变量

创建 `Saver`（使用 `tf.train.Saver()`）来管理模型中的所有变量。例如，以下代码段展示了如何调用 `tf.train.Saver.save` 方法以将变量保存到检查点文件中：

```python
# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in path: %s" % save_path)
```

### 恢复变量

`tf.train.Saver` 对象不仅将变量保存到检查点文件中，还将恢复变量。请注意，当您恢复变量时，您不必事先将其初始化。例如，以下代码段展示了如何调用 `tf.train.Saver.restore` 方法以从检查点文件中恢复变量：

```python
tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
```

> 注意：并没有名为 /tmp/model.ckpt 的实体文件。它是为检查点创建的文件名的前缀。用户仅与前缀（而非检查点实体文件）互动。

### 选择要保存和恢复的变量

如果您没有向 `tf.train.Saver()` 传递任何参数，则 `Saver` 会处理图中的所有变量。每个变量都保存在创建变量时所传递的名称下。

在检查点文件中明确指定变量名称的这种做法有时会非常有用。例如，您可能已经使用名为`"weights"`的变量训练了一个模型，而您想要将该变量的值恢复到名为`"params"`的变量中。

有时候，仅保存或恢复模型使用的变量子集也会很有裨益。例如，您可能已经训练了一个五层的神经网络，现在您想要训练一个六层的新模型，并重用该五层的现有权重。您可以使用 `Saver` 只恢复这前五层的权重。

您可以通过向 `tf.train.Saver()` 构造函数传递以下任一内容，轻松指定要保存或加载的名称和变量：

- 变量列表（将以其本身的名称保存）。
- Python 字典，其中，键是要使用的名称，键值是要管理的变量。

继续前面所示的保存/恢复示例：

```python
tf.reset_default_graph()
# Create some variables.
v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)

# Add ops to save and restore only `v2` using the name "v2"
saver = tf.train.Saver({"v2": v2})

# Use the saver object normally after that.
with tf.Session() as sess:
  # Initialize v1 since the saver will not.
  v1.initializer.run()
  saver.restore(sess, "/tmp/model.ckpt")

  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
```

注意：

- 如果要保存和恢复模型变量的不同子集，您可以根据需要创建任意数量的 `Saver` 对象。同一个变量可以列在多个 `Saver` 对象中，变量的值只有在 `Saver.restore()` 方法运行时才会更改。

- 如果您在会话开始时仅恢复一部分模型变量，则必须为其他变量运行初始化操作。详情请参阅 `tf.variables_initializer`。

- 要检查某个检查点中的变量，您可以使用 [`inspect_checkpoint`](https://www.tensorflow.org/code/tensorflow/python/tools/inspect_checkpoint.py)库，尤其是 `print_tensors_in_checkpoint_file` 函数。

- 默认情况下，`Saver` 会针对每个变量使用 `tf.Variable.name` 属性的值。但是，当您创建 `Saver` 对象时，您可以选择为检查点文件中的变量选择名称。


### 检查某个检查点中的变量

我们可以使用 `inspect_checkpoint` 库快速检查某个检查点中的变量。

继续前面所示的保存/恢复示例：

```python
# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)

# tensor_name:  v1
# [ 1.  1.  1.]
# tensor_name:  v2
# [-1. -1. -1. -1. -1.]

# print only tensor v1 in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v1', all_tensors=False)

# tensor_name:  v1
# [ 1.  1.  1.]

# print only tensor v2 in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v2', all_tensors=False)

# tensor_name:  v2
# [-1. -1. -1. -1. -1.]
```


<a name="models"></a>
## 保存和恢复模型

使用 `SavedModel` 保存和加载模型 - 变量、图和图的元数据。`SavedModel` 是一种独立于语言且可恢复的神秘序列化格式，使较高级别的系统和工具可以创建、使用和转换 TensorFlow 模型。TensorFlow 提供了多种与 `SavedModel` 交互的方式，包括 `tf.saved_model` API、`tf.estimator.Estimator` 和命令行界面。


## 构建和加载 SavedModel

### 简单保存

创建 `SavedModel` 的最简单方法是使用 `tf.saved_model.simple_save` 函数：

```python
simple_save(session,
            export_dir,
            inputs={"x": x, "y": y},
            outputs={"z": z})
```

这样可以配置 `SavedModel`，使其能够通过
[TensorFlow serving](/serving/serving_basic) 进行加载，并支持
[Predict API](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/predict.proto)。
要访问 `classify` API、`regress` API 或 `multi-inference` API，请使用手动 `SavedModel builder` API 或 `tf.estimator.Estimator`。

### 手动构建 SavedModel

如果您的用例不在 `tf.saved_model.simple_save` 涵盖范围内，请使用手动 `builder` API 创建 `SavedModel`。

`tf.saved_model.builder.SavedModelBuilder` 类提供了保存多个 `MetaGraphDef` 的功能。`MetaGraph` 是一种数据流图，并包含相关变量、资源和签名。`MetaGraphDef` 是 `MetaGraph` 的协议缓冲区表示法。签名是一组与图有关的输入和输出。

如果需要将资源保存并写入或复制到磁盘，则可以在首次添加 `MetaGraphDef` 时提供这些资源。如果多个 `MetaGraphDef` 与同名资源相关联，则只保留首个版本。

必须使用用户指定的标签对每个添加到 `SavedModel` 的 `MetaGraphDef` 进行标注。这些标签提供了一种方法来识别要加载和恢复的特定 `MetaGraphDef`，以及共享的变量和资源子集。这些标签一般会标注 `MetaGraphDef` 的功能（例如服务或训练），有时也会标注特定的硬件方面的信息（如 GPU）。

例如，以下代码展示了使用 `SavedModelBuilder` 构建 `SavedModel` 的典型方法：

```python
export_dir = ...
...
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph_and_variables(sess,
                                       [tag_constants.TRAINING],
                                       signature_def_map=foo_signatures,
                                       assets_collection=foo_assets,
                                       strip_default_attrs=True)
...
# Add a second MetaGraphDef for inference.
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph([tag_constants.SERVING], strip_default_attrs=True)
...
builder.save()
```

<a name="forward_compatibility"></a>
#### 通过 `strip_default_attrs=True` 确保向前兼容性

只有在操作集没有变化的情况下，遵循以下指南才能带来向前兼容性。

`SavedModelBuilder` 类允许用户控制在将元图添加到 SavedModel 软件包时，是否必须从
[`NodeDefs`](../extend/tool_developers/index.md#nodes)
剥离默认值属性。`SavedModelBuilder.add_meta_graph_and_variables` 和 `SavedModelBuilder.add_meta_graph` 方法都接受控制此行为的布尔标记 `strip_default_attrs`。

如果 `strip_default_attrs` 为 `False`，则导出的 `tf.MetaGraphDef` 将在其所有 `tf.NodeDef` 实例中具有设为默认值的属性。这样会破坏向前兼容性并出现一系列事件，例如：

- 现有的操作 (`Foo`) 会更新为在版本 101 中包含具有默认 (`bool`) 的新属性 (`T`)。
- 诸如“训练方二进制文件”之类的模型提供方将此更改（版本 101）提交给 `OpDef` 并重新导出使用操作 `Foo` 的现有模型。
- 运行较旧二进制文件（版本 100）的模型使用方（例如 [Tensorflow Serving](https://tensorflow.google.cn/serving)）在操作 `Foo` 中没有属性 `T`，但会尝试导入此模型。模型使用方无法在使用操作 `Foo` 的 `NodeDef` 中识别属性 `T`，因此无法加载模型。
- 通过将 `strip_default_attrs` 设置为 `True`，模型提供方可以剥离 `NodeDefs` 中任何具有默认值的属性。这有助于确保新添加的属性（具有默认值）不会导致早期的模型使用方无法加载使用较新的训练二进制文件重新生成的模型。

详情请参阅[兼容性指南](./version_compat.md)。

### 加载 Python 版 SavedModel

Python 版的 `SavedModel` 加载器为 `SavedModel` 提供加载和恢复功能。`load` 指令需要以下信息：

- 要在其中恢复图定义和变量的会话。
- 用于标识要加载的 `MetaGraphDef` 的标签。
- `SavedModel` 的位置（目录）。

加载后，作为特定 `MetaGraphDef` 的一部分提供的变量、资源和签名子集将恢复到提供的会话中。


```python
export_dir = ...
...
with tf.Session(graph=tf.Graph()) as sess:
  tf.saved_model.loader.load(sess, [tag_constants.TRAINING], export_dir)
  ...
```


### 加载 C++ 版 SavedModel

C++ 版的 `SavedModel`
[加载器](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/loader.h)
提供了一个可从某个路径加载 `SavedModel` 的 API（同时允许 `SessionOptions` 和 `RunOptions`）。您必须指定与要加载的图相关联的标签。`SavedModel` 加载后的版本称为 `SavedModelBundle`，其中包含 `MetaGraphDef` 和加载时所在的会话。

```c++
const string export_dir = ...
SavedModelBundle bundle;
...
LoadSavedModel(session_options, run_options, export_dir, {kSavedModelTagTrain},
               &bundle);
```

### 在 TensorFlow Serving 中加载和提供 SavedModel

您可以使用 TensorFlow Serving Model Server 二进制文件轻松加载和提供 SavedModel。参阅[此处](https://www.tensorflow.org/serving/setup#installing_using_apt-get)的说明，了解如何安装服务器，或根据需要创建服务器。

一旦您的 Model Server 就绪，请运行以下内容：   
```
tensorflow_model_server --port=port-numbers --model_name=your-model-name --model_base_path=your_model_base_path
```
将 port 和 model_name 标记设为您所选的值。`model_base_path` 标记应为基本目录，每个版本的模型都放置于以数字命名的子目录中。如果您的模型只有一个版本，只需如下所示地将其放在子目录中即可：
* 将模型放入 `/tmp/model/0001`
* 将 `model_base_path` 设为 `/tmp/model`

将模型的不同版本存储在共用基本目录的子目录中（以数字命名）。例如，假设基本目录是 `/tmp/model`。如果您的模型只有一个版本，请将其存储在 `/tmp/model/0001` 中。如果您的模型有两个版本，请将第二个版本存储在 `/tmp/model/0002` 中，以此类推。将 `--model-base_path` 标记设为基本目录（在本例中为 `/tmp/model`）。TensorFlow Model Server 将在该基本目录的最大编号的子目录中提供模型。

### 标准常量

SavedModel 为各种用例构建和加载 TensorFlow 图提供了灵活性。对于最常见的用例，SavedModel 的 API 在 Python 和 C++ 中提供了一组易于重复使用且便于在各种工具中共享的常量。

#### 标准 MetaGraphDef 标签

您可以使用标签组唯一标识保存在 SavedModel 中的 `MetaGraphDef`。常用标签的子集规定如下：

* [Python](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/tag_constants.py)
* [C++](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/tag_constants.h)


#### 标准 SignatureDef 常量

[**SignatureDef**](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/meta_graph.proto)
是一个协议缓冲区，用于定义图所支持的计算的签名。常用的输入键、输出键和方法名称定义如下：

* [Python](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/signature_constants.py)
* [C++](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/signature_constants.h)

## 搭配 Estimator 使用 SavedModel

训练 `Estimator` 模型之后，您可能需要使用该模型创建服务来接收请求并返回结果。您可以在本机运行此类服务，或在云端部署该服务。

要准备一个训练过的 Estimator 以供使用，您必须以标准 SavedModel 格式导出它。本节介绍了如何进行以下操作：

- 指定可以投入使用的输出节点和相应的 [API](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto)（Classify API、Regress API 或 Predict API）。
- 将模型导出为 `SavedModel` 格式。
- 从本地服务器提供模型并请求预测。


### 准备提供输入

在训练期间，[`input_fn()`](../guide/premade_estimators.md#input_fn) 会提取数据，并准备好数据以供模型使用。在提供服务期间，类似地，`serving_input_receiver_fn()` 接受推理请求，并为模型做好准备。该函数具有以下用途：

- 在投入使用系统将向其发出推理请求的图中添加占位符。
- 添加将数据从输入格式转换为模型所预期的特征 Tensor 所需的任何额外操作。

该函数返回一个 `tf.estimator.export.ServingInputReceiver` 对象，该对象会将占位符和生成的特征 `Tensor` 打包在一起。

典型的模式是推理请求以序列化 `tf.Example` 的形式到达，因此 `serving_input_receiver_fn()` 创建单个字符串占位符来接收它们。`serving_input_receiver_fn()` 接着也负责解析 `tf.Example`（通过向图中添加 `tf.parse_example` 操作）。

在编写此类 `serving_input_receiver_fn()` 时，您必须将解析规范传递给 `tf.parse_example`，告诉解析器可能会遇到哪些特征名称以及如何将它们映射到 `Tensor`。解析规范采用字典的形式，即从特征名称映射到 `tf.FixedLenFeature`、`tf.VarLenFeature` 和 `tf.SparseFeature`。请注意，此解析规范不应包含任何标签或权重列，因为这些列在服务时间将不可用（与 `input_fn()` 在训练时使用的解析规范相反）。

然后结合如下：

```py
feature_spec = {'foo': tf.FixedLenFeature(...),
                'bar': tf.VarLenFeature(...)}

def serving_input_receiver_fn():
  """An input receiver that expects a serialized tf.Example."""
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[default_batch_size],
                                         name='input_example_tensor')
  receiver_tensors = {'examples': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
```

`tf.estimator.export.build_parsing_serving_input_receiver_fn` 效用函数提供了适用于普遍情况的输入接收器。

> 注意：在使用 Predict API 和本地服务器训练要投入使用的模型时，并不需要解析步骤，因为该模型将接收原始特征数据。

即使您不需要解析或其他输入处理，也就是说，如果服务系统直接提供特征 `Tensor`，您仍然必须提供一个 `serving_input_receiver_fn()` 来为特征 `Tensor` 创建占位符并在其中传递占位符。`tf.estimator.export.build_raw_serving_input_receiver_fn` 效用函数实现了这一功能。

如果这些效用函数不能满足您的需求，您可以自由编写 `serving_input_receiver_fn()`。可能需要此方法的一种情况是，如果您训练的 `input_fn()` 包含某些必须在服务时间重演的预处理逻辑。为了减轻训练服务倾斜的风险，我们建议将这种处理封装在一个函数内，此函数随后将从 `input_fn()` 和 `serving_input_receiver_fn()` 两者中被调用。

请注意，`serving_input_receiver_fn()` 也决定了签名的输入部分。也就是说，在编写 `serving_input_receiver_fn()` 时，必须告诉解析器哪些有哪些签名可能出现，以及如何将它们映射到模型的预期输入。相反，签名的输出部分由模型决定。

<a name="specify_outputs"></a>
### 指定自定义模型的输出

编写自定义 `model_fn` 时，必须填充 `export_outputs` 元素（属于 `tf.estimator.EstimatorSpec` 返回值）。这是 `{name: output}` 描述在服务期间输出和使用的输出签名的词典。

在进行单一预测的通常情况下，该词典包含一个元素，而且 `name` 不重要。在一个多头模型中，每个头部都由这个词典中的一个条目表示。在这种情况下，`name` 是一个您所选择的字符串，用于在服务时间内请求特定头部。

每个 `output` 值必须是一个 `ExportOutput` 对象，例如 `tf.estimator.export.ClassificationOutput`、`tf.estimator.export.RegressionOutput` 或 `tf.estimator.export.PredictOutput`。

这些输出类型直接映射到
[TensorFlow Serving API](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto)，并确定将支持哪些请求类型。

Note: 注意：在多头情况下，系统将为从 model_fn 返回的 export_outputs 字典的每个元素生成 SignatureDef，这些元素都以相同的键命名。这些 SignatureDef 仅在输出（由相应的 ExportOutput 条目提供）方面有所不同。输入始终是由 serving_input_receiver_fn 提供的。推理请求可以按名称指定头部。一个头部必须使用
[`signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY`](https://www.tensorflow.org/code/tensorflow/python/saved_model/signature_constants.py)
命名，表示在推理请求没有指定 SignatureDef 时将提供哪一个 SignatureDef。

<a name="perform_export"></a>
### 导出 Estimator

要导出已训练的 Estimator，请调用 `tf.estimator.Estimator.export_savedmodel` 并提供导出基本路径和 `serving_input_receiver_fn`。

```py
estimator.export_savedmodel(export_dir_base, serving_input_receiver_fn,
                            strip_default_attrs=True)
```

该方法通过以下方式构建新图：首先调用 `serving_input_receiver_fn()` 以获得特征 `Tensor`，然后调用此 Estimator 的 `model_fn()`，以基于这些特征生成模型图。它会重新启动 `Session`，并且默认情况下会将最近的检查点恢复到它（如果需要，可以传递不同的检查点）。最后，它在给定的 `export_dir_base`（即 `export_dir_base/<timestamp>`）下面创建一个带时间戳的导出目录，并将 `SavedModel` 写入其中，其中包含从此会话中保存的单个 `MetaGraphDef`。

> 注意：您负责对先前的导出操作进行垃圾回收。否则，连续导出将在 `export_dir_base` 下累积垃圾资源。

### 在本地提供导出的模型

对于本地部署，您可以使用
[TensorFlow Serving](https://github.com/tensorflow/serving)提供模型，TensorFlow Serving 是一个开源项目，用于加载 SavedModel 并将其公开为
[gRPC](https://www.grpc.io/) 服务。

首先，[安装 TensorFlow Serving](https://github.com/tensorflow/serving)。

然后构建并运行本地模型服务器，用上面导出的指向 SavedModel 的路径替换 `$export_dir_base`：

```sh
bazel build //tensorflow_serving/model_servers:tensorflow_model_server
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_base_path=$export_dir_base
```

现在您有一台服务器在端口 9000 上通过 gRPC 监听推理请求了！


### 从本地服务器请求预测

服务器根据
[PredictionService](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto#L15)
gRPC API 服务定义对 gRPC 请求做出响应（嵌套协议缓冲区在各种
[neighboring files](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis)中定义）。

根据 API 服务定义，gRPC 框架以各种语言生成客户端库，提供对 API 的远程访问。在使用 Bazel 构建工具的项目中，这些库是自动构建的，并通过以下关联项提供（以 Python 为例）：

```build
  deps = [
    "//tensorflow_serving/apis:classification_proto_py_pb2",
    "//tensorflow_serving/apis:regression_proto_py_pb2",
    "//tensorflow_serving/apis:predict_proto_py_pb2",
    "//tensorflow_serving/apis:prediction_service_proto_py_pb2"
  ]
```

然后，Python 客户端代码可以导入这些库：

```py
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import regression_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
```

> 注意：prediction_service_pb2 将服务定义为一个整体，因此始终是必需的。然而一个典型的客户端只需要 classification_pb2、regression_pb2 和 predict_pb2 中的一个，取决于所做请求的类型。

通过在协议缓冲区聚集请求数据并将其传递给服务存根即可完成 gRPC 请求的发送。请注意观察请求协议缓冲区是如何创建为空区的，然后是如何通过
[生成的协议缓冲区 API](https://developers.google.com/protocol-buffers/docs/reference/python-generated) 填充的。

```py
from grpc.beta import implementations

channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = classification_pb2.ClassificationRequest()
example = request.input.example_list.examples.add()
example.features.feature['x'].float_list.value.extend(image[0].astype(float))

result = stub.Classify(request, 10.0)  # 10 secs timeout
```

本例中返回的结果是一个 `ClassificationResponse` 协议缓冲区。

这是一个概括性示例；详情请参阅 [Tensorflow Serving](../deploy/index.md)
文档和[示例](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example)。

> 注意：ClassificationRequest 和 RegressionRequest 包含一个 tensorflow.serving.Input 协议缓冲区，该协议缓冲区又包含一系列 tensorflow.Example 协议缓冲区。而与之不同的是，PredictRequest 包含从特征名称到用 TensorProto 进行编码的值的映射。相应地，使用 Classify 和 Regress API 时，TensorFlow Serving 会将序列化 tf.Example 馈送到图中，所以 serving_input_receiver_fn() 应该包含一个 tf.parse_example() 操作。但是如果使用通用 Predict API，TensorFlow Serving 会将原始特征数据馈送到图中，因此应该使用 serving_input_receiver_fn() 进行传递。


<!-- TODO(soergel): give examples of making requests against this server, using
the different Tensorflow Serving APIs, selecting the signature by key, etc. -->

<!-- TODO(soergel): document ExportStrategy here once Experiment moves
from contrib to core. -->




## 使用 CLI 检查并执行 SavedModel

您可以使用 SavedModel 命令行界面 (CLI) 检查并执行 `SavedModel`。例如，您可以使用 CLI 检查模型的 `SignatureDef`。CLI 让您能够快速确认输入的
[Tensor dtype 和形状](../guide/tensors.md)是否与模型匹配。此外，如果您想测试模型，可以使用 CLI 进行健全性检查，方法是传入各种格式的示例输入（如 Python 表达式），然后获取输出。


### 安装 SavedModel CLI

一般来说，您可以通过以下两种方式之一安装 TensorFlow：

- 通过安装预构建的 TensorFlow 二进制文件。
- 通过从源代码构建 TensorFlow。

如果您通过预构建的 TensorFlow 二进制文件安装了 TensorFlow，则您的系统上已经安装 SavedModel CLI（路径名称为：`bin\saved_model_cli`）。

如果您从源代码构建 TensorFlow，则必须运行以下附加命令来构建 `saved_model_cli`：

```
$ bazel build tensorflow/python/tools:saved_model_cli
```

### 命令概述

SavedModel CLI 在 `SavedModel` 中 `MetaGraphDef` 上支持以下两个命令：

- `show`，显示在 `SavedModel` 中 `MetaGraphDef` 上的计算。
- `run`，在 `MetaGraphDef` 上运行计算。



### `show` 命令

`SavedModel` 包含一个或多个 `MetaGraphDef`，由其标签集进行标识。要提供模型，您可能想知道每个模型中的 `SignatureDef` 是什么类型的，它们的输入和输出是什么。`show` 命令可让您按层次顺序检查 `SavedModel` 的内容。语法如下：

```
usage: saved_model_cli show [-h] --dir DIR [--all]
[--tag_set TAG_SET] [--signature_def SIGNATURE_DEF_KEY]
```

例如，以下命令显示 SavedModel 中所有可用的 MetaGraphDef 的标签集：

```
$ saved_model_cli show --dir /tmp/saved_model_dir
The given SavedModel contains the following tag-sets:
serve
serve, gpu
```

以下命令会显示 `MetaGraphDef` 中所有可用的 `SignatureDef` 键：

```
$ saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve
The given SavedModel `MetaGraphDef` contains `SignatureDefs` with the
following keys:
SignatureDef key: "classify_x2_to_y3"
SignatureDef key: "classify_x_to_y"
SignatureDef key: "regress_x2_to_y3"
SignatureDef key: "regress_x_to_y"
SignatureDef key: "regress_x_to_y2"
SignatureDef key: "serving_default"
```

如果 `MetaGraphDef` 在标记集中有多个标记，则您必须指定所有标记，每个标记用英文逗号分隔。例如：

```none
$ saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve,gpu
```

要显示特定 `SignatureDef` 的所有输入和输出 `TensorInfo`，请将 `SignatureDef` 键传递给 `signature_def` 选项。当您想知道输入张量的键值、`dtype` 和形状以便后续执行计算图时，这会非常有用。例如：

```
$ saved_model_cli show --dir \
/tmp/saved_model_dir --tag_set serve --signature_def serving_default
The given SavedModel SignatureDef contains the following input(s):
  inputs['x'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1)
      name: x:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['y'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1)
      name: y:0
Method name is: tensorflow/serving/predict
```

要显示 SavedModel 中的所有可用信息，请使用 `--all` 选项。例如：

```none
$ saved_model_cli show --dir /tmp/saved_model_dir --all
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['classify_x2_to_y3']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: x2:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y3:0
  Method name is: tensorflow/serving/classify

...

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['x'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: x:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['y'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y:0
  Method name is: tensorflow/serving/predict
```


### `run` 命令

调用 `run` 命令以运行图计算、传递输入，然后显示（并可选地保存）输出。语法如下：

```
usage: saved_model_cli run [-h] --dir DIR --tag_set TAG_SET --signature_def
                           SIGNATURE_DEF_KEY [--inputs INPUTS]
                           [--input_exprs INPUT_EXPRS]
                           [--input_examples INPUT_EXAMPLES] [--outdir OUTDIR]
                           [--overwrite] [--tf_debug]
```

`run` 命令提供以下三种将输入传递给模型的方式：

- `--inputs` 选项可让您在文件中传递 NumPy ndarray。
- `--input_exprs` 选项可让您传递 Python 表达式。
- `--input_examples` 选项可让您传递 `tf.train.Example`。



#### `--inputs`

要在文件中传递输入数据，请指定 `--inputs` 选项，该选项采用以下通用格式：

```bsh
--inputs <INPUTS>
```

INPUT 采用以下格式之一：

*  `<input_key>=<filename>`
*  `<input_key>=<filename>[<variable_name>]`

您可能会传递多个 INPUT。如果您确实要传递多个输入，请使用分号分隔每个 INPUT。

`saved_model_cli` 使用 `numpy.load` 加载 `filename`。文件名可以是以下任何一种格式：

*  `.npy`
*  `.npz`
*  pickle 格式

`.npy` 文件总是包含多维数组 (numpy ndarray)。因此，当从 `.npy` 文件加载时，内容将直接分配给指定的输入张量。如果使用该 `.npy` 文件指定 `variable_name`，则 `variable_name` 将被忽略，并且系统会发出警告。

从 `.npz` (zip) 文件加载时，您可以选择指定一个 `variable_name` 来标识 `zip` 文件中针对输入张量键进行加载的变量。如果您未指定 `variable_name`，则 SavedModel CLI 将检查 `zip` 文件中是否只包含一个文件，并将为指定的输入张量键加载。

从 pickle 文件加载时，如果方括号中没有指定 `variable_name`，那么 `pickle` 文件中的任何内容都将传递到指定的输入张量键。否则，SavedModel CLI 会假设在 `pickle` 文件中存储了字典，并且与 `variable_name` 对应的值将被使用。


#### `--input_exprs`

要通过 Python 表达式传递输入，请指定 `--input_exprs` 选项。这对于您目前没有数据文件的情形而言非常有用，但最好还是用一些与模型的 `SignatureDef` 的 `dtype` 和形状匹配的简单输入来检查模型。例如：

```bsh
`<input_key>=[[1],[2],[3]]`
```

除了 Python 表达式之外，您还可以传递 numpy 函数。例如：

```bsh
`<input_key>=np.ones((32,32,3))`
```

（请注意，`numpy` 模块已可作为 `np` 提供。）


#### `--input_examples`

要将 `tf.train.Example` 作为输入进行传递，请指定 `--input_examples` 选项。对于每个输入键，它都接受一个字典列表，其中每个字典都是 `tf.train.Example` 的一个实例。不同的字典键代表不同的特征，而相应的值则是每个特征的值列表。例如：

```bsh
`<input_key>=[{"age":[22,24],"education":["BS","MS"]}]`
```

#### 保存输出

默认情况下，SavedModel CLI 将输出写入 `stdout`。如果目录传递给 `--outdir` 选项，则输出将被保存为在指定目录下以输出张量键命名的 `npy` 文件。

请使用 `--overwrite` 覆盖现有的输出文件。


#### TensorFlow debugger (tfdbg) 集成

如果设置了 `--tf_debug` 选项，则 SavedModel CLI 将使用 TensorFlow Debugger (tfdbg) 在运行 SavedModel 时观察中间张量和运行时图或子图。


#### run 的完整示例

假设：

- 您的模型只需添加 `x1` 和 `x2` 即可获得输出 `y`。
- 模型中的所有张量都具有形状 `(-1, 1)`。
- 您有两个 `npy` 文件：
    - `/tmp/my_data1.npy`，其中包含一个 NumPy ndarray `[[1], [2], [3]]`。
    - `/tmp/my_data2.npy`，其中包含另一个 NumPy ndarray `[[0.5], [0.5], [0.5]]`。

要使用模型运行这两个 `npy` 文件以获得输出 `y`，请发出以下命令：

```
$ saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def x1_x2_to_y --inputs x1=/tmp/my_data1.npy;x2=/tmp/my_data2.npy \
--outdir /tmp/out
Result for output key y:
[[ 1.5]
 [ 2.5]
 [ 3.5]]
```

让我们稍微调整一下前面的例子。这一次，不是两个 `.npy` 文件，而是一个 `.npz` 文件和一个 pickle 文件。此外，您要覆盖任何现有的输出文件。命令如下：

```
$ saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def x1_x2_to_y \
--inputs x1=/tmp/my_data1.npz[x];x2=/tmp/my_data2.pkl --outdir /tmp/out \
--overwrite
Result for output key y:
[[ 1.5]
 [ 2.5]
 [ 3.5]]
```

您可以指定 python 表达式，取代输入文件。例如，以下命令用 Python 表达式替换输入 `x2`：

```
$ saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def x1_x2_to_y --inputs x1=/tmp/my_data1.npz[x] \
--input_exprs 'x2=np.ones((3,1))'
Result for output key y:
[[ 2]
 [ 3]
 [ 4]]
```

要在开启 TensorFlow Debugger 的情况下运行模型，请发出以下命令：

```
$ saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def serving_default --inputs x=/tmp/data.npz[x] --tf_debug
```


<a name="structure"></a>
## SavedModel 目录的结构

当您以 SavedModel 格式保存模型时，TensorFlow 会创建一个由以下子目录和文件组成的 SavedModel 目录：

```bsh
assets/
assets.extra/
variables/
    variables.data-?????-of-?????
    variables.index
saved_model.pb|saved_model.pbtxt
```

其中：

- `assets` 是包含辅助（外部）文件（如词汇表）的子文件夹。资源被复制到 SavedModel 的位置，并且可以在加载特定的 `MetaGraphDef` 时被读取。
- `assets.extra` 是一个子文件夹，其中较高级别的库和用户可以添加自己的资源，这些资源与模型共存，但不会被图加载。此子文件夹不由 SavedModel 库管理。
- `variables` 是包含 `tf.train.Saver` 的输出的子文件夹。
- `saved_model.pb` 或 `saved_model.pbtxt` 是 SavedModel 协议缓冲区。它包含作为 `MetaGraphDef` 协议缓冲区的图定义。

单个 SavedModel 可以表示多个图。在这种情况下，SavedModel 中所有图共享一组检查点（变量）和资源。例如，下图显示了一个包含三个 `MetaGraphDef` 的 SavedModel，它们三个都共享同一组检查点和资源：

![SavedModel represents checkpoints, assets, and one or more MetaGraphDefs](https://github.com/ziiai/tensorflow-docs/raw/master/images/SavedModel.svg)

每个图都与一组特定的标记相关联，可在加载或恢复操作期间方便您识别。
