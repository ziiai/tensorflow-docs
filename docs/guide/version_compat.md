#  TensorFlow 版本兼容性

本文档面向需要为不同版本的 TensorFlow（无论是代码或数据）提供向后兼容性的用户，以及希望在保持兼容性的同时也能够更改 TensorFlow 的开发人员。

## 语义版本 2.0

TensorFlow 的公共 API 遵循语义版本 2.0 ([semver](http://semver.org))。每个 TensorFlow 版本的版本号都采用 `MAJOR.MINOR.PATCH` 的形式。例如，TensorFlow 版本 1.2.3 的 `MAJOR` 版本为 1，`MINOR` 版本为 2，`PATCH` 版本为 3。其中每个数字的变化包含以下含义：

* **MAJOR：** 包含的更改可能具有向后不兼容性。适用于先前的 Major 版本的代码和数据不一定适用于新的 `Major` 版本。但是，在某些情况下，或许可以将现有的 TensorFlow 图和检查点迁移到新的版本；要详细了解数据兼容性，请参阅
[图和检查点的兼容性](#compatibility_of_graphs_and_checkpoints)。

* **MINOR：** 包含的功能具有向后兼容性，速度有所提升，等等。适用于先前的 Minor 版本且仅依赖于公共 API 的代码和数据将继续适用于新的 Minor 版本。要详细了解哪些 API 是公共 API，哪些不是，请参阅
[涵盖的内容](#what_is_covered).

* **PATCH：** 包含的问题修复程序具有向后兼容性。

例如，1.0.0 版本中包含的更改相较于 0.12.1 版本来说具有向后不兼容性的。但是，1.1.1 版本相较于 1.0.0 版本是具有向后兼容性的。

## 涵盖的内容

只有 TensorFlow 的公共 API 在次要版本和补丁版本之间向后兼容。公共 API 包括：

* `tensorflow` 模块及其子模块中载述的所有 Python 函数和类，以下函数和类除外：

    * `tf.contrib` 中的函数和类
    * 名称以 _ 开头的函数和类（因为它们是私有函数和类）。请注意，`examples/` 和 `tools/` 目录下的代码不能通过 `tensorflow` Python 模块访问，因此不在兼容性保证的涵盖范围内。

    如果某符号可通过 `tensorflow` Python 模块或其子模块访问，但却没有记录在文档中，则**不**属于公共 API 的一部分。

* [C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h)。

* 以下协议缓冲区文件：
    * [`attr_value`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto)
    * [`config`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto)
    * [`event`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/event.proto)
    * [`graph`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto)
    * [`op_def`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto)
    * [`reader_base`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/reader_base.proto)
    * [`summary`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto)
    * [`tensor`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto)
    * [`tensor_shape`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_shape.proto)
    * [`types`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto)

<a name="not_covered"></a>
## 不涵盖的 API

一些 API 函数明确标记为“实验性”，可以在 Minor 版本之间以向后不兼容的方式进行更改。这些 API 包括：

- **实验性 API：** 采用 Python 的 tf.contrib 模块及其子模块，以及 C API 中的任何函数或协议缓冲区中任何明确注释为实验性字段的字段。具体指协议缓冲区中任何被称为“实验性”的字段，其所有字段及子消息会随时更改。

- **其他语言的 API：** 采用 Python 和 C 之外的其他语言开发而成的 TensorFlow API，例如以下其他语言的 API：

  - [C++](../api_guides/cc/guide.md) （通过
    [`tensorflow/cc`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/cc)中的头文件提供）。
  - [Java](../api_docs/java/reference/org/tensorflow/package-summary),
  - [Go](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go)
  - [JavaScript](https://js.tensorflow.org)

- **复合操作的细节：** Python 中的许多公共函数可展开为图中的多个原始操作，这些细节将是以 `GraphDef` 形式保存到磁盘的任何图的一部分。在 Minor 版本中，这些细节可能会发生改变。具体而言，检查图之间是否完全匹配的回归测试在 Minor 版本更新后可能会崩溃，虽然图的行为会保持不变且现有检查点仍然可以工作。

- **浮点数值细节：** 操作计算的特定浮点值可能会随时改变。用户应该仅依赖于近似准确率和数值稳定性，而不是计算出的特定位。Minor 版本和 Patch 版本中对数值公式的更改应该使准确率有所提高。但是，在机器学习中，如果特定公式的准确率有所提高，则可能会导致整个系统的准确率降低。

- **随机数字：** 由随机操作计算的特定随机数字可能会随时改变。用户应该仅依赖于近似正确的分布和统计强度，而不是计算出的特定位。但是，我们很少（或许从不）更改 Patch 版本的随机位。当然，所有这些更改都将记录在案。

- **分布式 Tensorflow 中的版本偏差：** 不支持在单个集群中运行两个不同的 TensorFlow 版本。对于线路协议的向后兼容性，我们不做任何保证。

- **错误：** 如果当前实现有明显问题，也就是说，如果当前实现与文档相矛盾，或者典型且定义明确的预期行为由于某一错误而未正确实现，那么我们保留做出向后不兼容行为（但不是 API）更改的权利。例如，如果某优化器声称会实现众所周知的优化算法，但由于某一错误而不与该算法相匹配，那么我们将修复该优化器。我们的修复可能会破坏依靠错误收敛行为的代码。版本说明中会注明这类更改。

- **错误消息：** 我们保留更改错误消息文本的权利。另外，错误类型可能会发生更改，除非该类型在文档中已指定。例如，记录为会引起 `InvalidArgument` 异常的函数将继续引起 `InvalidArgument`，但是用户可读的消息内容可能会发生改变。

## 图和检查点的兼容性

用户有时需要保存图和检查点。图描述在训练和推理期间要运行的 op 的数据流，而检查点包含图中变量已保存的张量值。

许多 TensorFlow 用户将图和经过训练的模型保存到磁盘，以便今后进行评估或进行额外训练，但最终仍会在更高版本上运行保存的图或模型。依照语义版本，用某一版本 TensorFlow 写出的任何图或检查点，都可以通过相同主要版本中更高（次要或补丁）版本的 TensorFlow 来进行加载和评估。但是，我们将继续努力，希望终有一日能够在不同的主要版本中也同样做到保持向后兼容性，以便能够在长时间跨度中使用序列化文件。

图通过 `GraphDef` 协议缓冲区序列化。为了促进（罕见）图的向后不兼容更改，每个 `GraphDef` 都有一个与 TensorFlow 版本相独立的版本号。例如，`GraphDef` 17 版本为支持 `reciprocal` 而弃用 `inv` op。语义为：

- TensorFlow 的每个版本都支持 `GraphDef` 版本间隔。这种间隔在各 Patch 版本之间保持不变，并且只在各 Minor 版本之间增长。只有 TensorFlow 的 Major 版本才会出现放弃支持 `GraphDef` 版本这种情况。

- 新创建的图将被分配最新的 `GraphDef` 版本号。

- 如果某一 TensorFlow 版本支持图的 `GraphDef` 版本，它将通过和用于生成它的 TensorFlow 版本相同的行为进行加载和评估（浮点数值细节和随机数字除外），不论 TensorFlow 的 Major 版本是什么。具体而言，所有的检查点文件都具有兼容性。

- 如果 `GraphDef` 上限在 (Minor) 版本中增加到 X，那么下限至少要六个月后才能增加到 X。例如（此处使用虚拟版本号）：
    - TensorFlow 1.2 可能支持 `GraphDef` 版本 4 至 7。
    - TensorFlow 1.3 可以添加 `GraphDef` 版本 8 并支持版本 4 至 8。
    - 至少六个月后，TensorFlow 2.0.0 可能会放弃对版本 4 至 7 的支持，而只支持版本 8。

最后，在放弃对某一 `GraphDef` 版本的支持后，我们将尝试提供相关工具，帮助用户将图自动转化为更新的受支持 `GraphDef` 版本。

## 扩展 TensorFlow 时的图和检查点兼容性

本章节讲述的内容只与对 `GraphDef` 格式做出不兼容的更改相关，例如添加 op，移除 op 或更改现有 op 的功能。对于多数用户来说，阅读上一章节已足够。

<a id="backward_forward"/>

### 向后兼容性和部分向前兼容性

我们的版本控制方案有三个要求：

- 向后兼容性，以支持加载使用旧版 TensorFlow 创建的图和检查点。
- 向前兼容性，以支持以下情况：图或检查点的提供方先于使用方升级到了更新版本的 TensorFlow。
- 能够以不兼容的方式改进 TensorFlow。例如，移除操作、添加属性和移除属性。

请注意，尽管 `GraphDef` 版本机制与 TensorFlow 版本相独立，但对 `GraphDef` 格式的向后不兼容性更改仍受限于语义版本控制。这意味着，只能在 TensorFlow `MAJOR` 版本之间（例如 `1.7` 更新到 `2.0`）移除或更改功能。此外，补丁版本之间（例如 1.x.1 更新到 1.x.2）实施向前兼容性。

为了实现向后和向前兼容性，同时为了了解何时实施格式更改，图和检查点具有描述它们何时生成的元数据。后续章节详细介绍了用于改进 `GraphDef` 版本的 TensorFlow 实现和准则。

### 独立的数据版本方案

图和检查点有不同的数据版本。这两种数据格式之间的改进速度不一样，也不同于 TensorFlow。两个版本控制方案在 `core/public/version.h` 中皆有定义。每当添加新版本时，标题中都会增加注释，详细说明所更改的内容和发生日期。

### 数据、提供方和使用方

以下几种数据版本信息有所区别： *提供方：生成数据的二进制文件。提供方拥有一个版本 (`producer`)，以及与之兼容的最低使用方版本 (`min_consumer`)。 *使用方：使用数据的二进制文件。使用方拥有一个版本 (`consumer`)，以及与之兼容的最低提供方版本 (`min_producer`)。

每段版本化数据都有 `VersionDef versions` 字段，用于记录创建数据的 `producer`、与之兼容的 `min_consumer`，以及不支持的 `bad_consumers` 版本的列表。

默认情况下，当提供方创建数据时，数据会继承提供方的 `producer` 和 `min_consumer` 版本。如果已知特定的使用方版本包含错误且必须避免这种情况发生，那么可以设置 `bad_consumers`。如果使用方要接受某份数据，则需满足以下条件：

- `consumer` >= 数据的 `min_consumer`
- 数据的 `producer` >= 使用方的 `min_producer`
- `consumer` 不在数据的 `bad_consumers` 列表中

由于提供方和使用方都来自同一个 TensorFlow 代码库，因此 `core/public/version.h` 包含一个主要数据版本（根据上下文被视为 `producer` 或 `consumer`），同时也包含 `min_consumer` 和 `min_producer`（分别为提供方和使用方所需）。具体情况如下：

- 对于 `GraphDef` 版本，我们有 `TF_GRAPH_DEF_VERSION`、`TF_GRAPH_DEF_VERSION_MIN_CONSUMER` 和 `TF_GRAPH_DEF_VERSION_MIN_PRODUCER`。
- 对于检查点版本，我们有 `TF_CHECKPOINT_VERSION`、`TF_CHECKPOINT_VERSION_MIN_CONSUMER` 和 `TF_CHECKPOINT_VERSION_MIN_PRODUCER`。

### 将具有默认值的新属性添加到现有操作中

只有在操作集没有变化的情况下，遵循以下指南才能提供向前兼容性。

- 如果需要向前兼容性，请将 `strip_default_attrs` 设置为 `True`，同时使用 `SavedModelBuilder` 类的 `add_meta_graph_and_variables` 和 `add_meta_graph` 方法或 `Estimator.export_savedmodel` 导出模型
- 这会在生成/导出模型时剥离默认值属性。这样可以确保在使用默认值时，导出的 `tf.MetaGraphDef` 不包含新的操作属性。
- 这种控制可以允许过期的使用者（例如，滞后于训练二进制文件的使用二进制文件）继续加载模型并防止在使用模型时出现中断。


### 不断改进的 GraphDef 版本

本章节介绍了如何使用此版本控制机制对 `GraphDef` 格式进行不同类型的更改。

#### 添加操作

同时向使用方和提供方添加新的操作，并且不更改任何 `GraphDef` 版本。这种类型的更改会自动向后兼容，并且不会影响向前兼容性计划，因为现有提供方脚本不会突然使用新功能。

#### 添加操作并切换现有 Python 封装容器来使用该操作

1. 实现新的使用方功能并递增 `GraphDef` 版本。
1. 如果操作能添加新功能并且不破坏更改，则更新 Python 封装容器。
1. 更改 Python 封装容器以使用新功能。请勿递增 `min_consumer`，因为不使用此操作的模型不会出错。

#### 移除或限制操作的功能

1. 修复所有提供方脚本（不是 TensorFlow 本身），以避免使用禁用的操作或功能。
1. 递增 `GraphDef` 版本并实现新的使用方功能，以便在新版本及更高版本中禁用已移除的操作或 `GraphDefs` 功能。如果可能，让 TensorFlow 停止生成具备禁用功能的 `GraphDefs`。为此，请添加 REGISTER_OP(...).Deprecated(deprecated_at_version, message)。
1. 若需向后兼容性，请等待相关 Major 版本。
1. 将 `min_producer` 增加至 (2) 中的 `GraphDef` 版本，并完全移除该功能。

#### 更改操作的功能

1. 添加一个以 `SomethingV2` 或类似形式命名的类似新操作，完成添加该操作的流程并切换现有 Python 封装容器以使用该操作。为了确保向前兼容性，请在更改 Python 封装容器时使用 compat.py 中建议的方式进行检查。
1. 移除旧的操作（由于向后兼容性，只能跟随 Major 版本更改进行）。
1. 通过增加 `min_consumer` 来排除仍使用旧版操作的使用方，重新添加旧版操作并将其作为 `SomethingV2` 的别名，并切换现有 Python 封装容器来使用该操作。
1. 完成移除 `SomethingV2` 的流程。

#### 禁用单个不安全的使用方版本

1. 提升 `GraphDef` 版本，并针对所有新的 `GraphDef` 将不良版本添加到 `bad_consumers` 列表中。如果可能，仅针对包含特定或类似操作的 `GraphDef` 将不良版本添加到 `bad_consumers` 列表中。
1. 如果现有使用方是不良版本，请尽快将其弃用。
