#  使用 TPU

本文介绍了有效使用 [Cloud TPU](https://cloud.google.com/tpu/)所需的主要 TensorFlow API，并强调了常规 TensorFlow 使用情况与 TPU 上使用情况之间的差异。

本文主要针对以下用户：

* 熟悉 TensorFlow 的 `Estimator` 和 `Dataset` API
* 可能已使用现有模型 [尝试过 Cloud TPU](https://cloud.google.com/tpu/docs/quickstart)
* 可能已浏览过示例 TPU 模型的代码
  [[1]](https://github.com/tensorflow/models/blob/master/official/mnist/mnist_tpu.py)
  [[2]](https://github.com/tensorflow/tpu/tree/master/models).
* 有兴趣将现有 `Estimator` 模型移植到 Cloud TPU 上运行

## TPUEstimator

Estimator 是 TensorFlow 的模型级抽象层。 标准 Estimators 可以在 CPU 和 GPU 上运行模型。您必须使用 tf.contrib.tpu.TPUEstimator 才能在 TPU 上运行模型。

请参阅 TensorFlow 的“使用入门”部分，了解关于如何使用 [预创建的 `Estimator`](/docs/tensorflow/guide/premade_estimators)和
[自定义 `Estimator`s](/docs/tensorflow/guide/custom_estimators)的基础知识。

`TPUEstimator` 类与 `Estimator` 类有所不同。

要维护可在 CPU/GPU 或 Cloud TPU 上运行的模型，最简单的方式是将模型的推理阶段（从输入到预测）定义在 `model_fn` 之外。然后，确保 `Estimator` 设置和 `model_fn` 的单独实现，二者均包含此推理步骤。有关此模式的示例，请对比
[tensorflow/models](https://github.com/tensorflow/models/tree/master/official/mnist) 中的 `mnist.py` 和 `mnist_tpu.py` 实现。

### 在本地运行 TPUEstimator

要创建标准 `Estimator`，您可以调用构造函数，然后将它传递给 `model_fn`，如：

```
my_estimator = tf.estimator.Estimator(
  model_fn=my_model_fn)
```

在本地计算机上使用 `tf.contrib.tpu.TPUEstimator` 所需的更改相对较少。构造函数需要两个额外的参数。应将 `use_tpu` 参数设为 `False`，并将 `tf.contrib.tpu.RunConfig` 以 `config` 参数的形式传递，如下所示：

``` python
my_tpu_estimator = tf.contrib.tpu.TPUEstimator(
    model_fn=my_model_fn,
    config=tf.contrib.tpu.RunConfig()
    use_tpu=False)
```

只需进行这一简单更改即可在本地运行 `TPUEstimator`。通过如下所示地设置命令行标记，大多数示例 TPU 模型便可在此本地模式中运行：


```
$> python mnist_tpu.py --use_tpu=false --master=''
```

> 注意：`use_tpu=False` 参数对尝试 TPUEstimator API 很有用。这并非一项完整的 TPU 兼容性测试。在 TPUEstimator 中本地成功地运行模型并不能确保模型就能在 TPU 上运行。


### 构建 `tpu.RunConfig`

虽然默认的 `RunConfig` 足够进行本地训练，但在实际使用中不能忽略以下设置。

`RunConfig` 的一种更典型设置（可切换为使用 Cloud TPU）如下：

``` python
import tempfile
import subprocess

class FLAGS(object):
  use_tpu=False
  tpu_name=None
  # Use a local temporary path for the `model_dir`
  model_dir = tempfile.mkdtemp()
  # Number of training steps to run on the Cloud TPU before returning control.
  iterations = 50
  # A single Cloud TPU has 8 shards.
  num_shards = 8

if FLAGS.use_tpu:
    my_project_name = subprocess.check_output([
        'gcloud','config','get-value','project'])
    my_zone = subprocess.check_output([
        'gcloud','config','get-value','compute/zone'])
    cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_names=[FLAGS.tpu_name],
            zone=my_zone,
            project=my_project)
    master = tpu_cluster_resolver.get_master()
else:
    master = ''

my_tpu_run_config = tf.contrib.tpu.RunConfig(
    master=master,
    evaluation_master=master,
    model_dir=FLAGS.model_dir,
    session_config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True),
    tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations,
                                        FLAGS.num_shards),
)
```

然后，您必须将 `tf.contrib.tpu.RunConfig` 传递给构造函数：

``` python
my_tpu_estimator = tf.contrib.tpu.TPUEstimator(
    model_fn=my_model_fn,
    config = my_tpu_run_config,
    use_tpu=FLAGS.use_tpu)
```

通常来说，`FLAGS` 将由命令行参数设置。要从在本地训练切换为在 Cloud TPU 上训练，您需要进行以下设置：

* 将 `FLAGS.use_tpu` 设置为 `True`
* 将 `FLAGS.tpu_name` 以便 `tf.contrib.cluster_resolver.TPUClusterResolver` 能够找到该 TPU
* 将 `FLAGS.model_dir` 设置为 Google Cloud Storage 存储分区网址 (`gs://`).


## 优化器

在 Cloud TPU 上训练时，您必须将优化器封装在 `tf.contrib.tpu.CrossShardOptimizer` 中，后者使用 `allreduce` 聚合梯度并将结果广播到各个分片（每个 TPU 核）。

`CrossShardOptimizer` 与本地训练不兼容。因此，要在本地和 Cloud TPU 上运行相同代码，请添加如下内容：

``` python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
if FLAGS.use_tpu:
  optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
```

如果您希望在模型代码中避免使用全局变量 `FLAGS`，一种方法是将优化器设为 `Estimator` 的一个参数，如下所示：

``` python
my_tpu_estimator = tf.contrib.tpu.TPUEstimator(
    model_fn=my_model_fn,
    config = my_tpu_run_config,
    use_tpu=FLAGS.use_tpu,
    params={'optimizer':optimizer})
```

## 模型函数

本节详细介绍了您必须对模型函数 (`model_fn()`) 进行哪些更改，以便让其与 `TPUEstimator` 兼容。

### 静态形状

在常规使用期间，TensorFlow 会在图的构建过程中尝试确定各个 tf.Tensor 的形状。在执行期间，任何未知形状的维度都是动态确定的。请参阅
 [Tensor 形状](/docs/tensorflow/guide/tensors.md#shape) 了解详情。

要在 Cloud TPU 上运行，TensorFlow 模型需使用 [XLA](/docs/tensorflow/performance/xla/index)
进行编译。XLA 在编译时会使用类似系统来确定形状。XLA 要求在编译时静态确定所有张量维度。所有形状都必须是常量，且不依赖于外部数据或有状态的操作（如变量或随机数生成器）。


### 总结

请勿在模型中使用 `tf.summary`。

[TensorBoard summaries](/docs/tensorflow/guide/summaries_and_tensorboard)
是查看模型内部的绝佳方式。`TPUEstimator` 会自动将基本总结的最小集记录到 `model_dir` 中的 `event` 文件。不过，在 Cloud TPU 上训练时，自定义总结目前不受支持。因此虽然 `TPUEstimator` 使用总结时仍可在本地运行，但如在 TPU 上使用，则会失败。

### 指标

在独立 `metric_fn` 中构建您的评估指标字典。

评估指标是训练模型的重要部分。Cloud TPU 全面支持这些指标，但语法略有不同。

一个标准 `tf.metrics` 会返回两个张量。一个会返回指标值的平均运行值，另一个会更新平均运行值并返回该批量的值：

```
running_average, current_batch = tf.metrics.accuracy(labels, predictions)
```

在标准 `Estimator` 中，您可以创建针对这些对的字典，并将其作为 `EstimatorSpec` 的一部分返回。

```python
my_metrics = {'accuracy': tf.metrics.accuracy(labels, predictions)}

return tf.estimator.EstimatorSpec(
  ...
  eval_metric_ops=my_metrics
)
```

在 `TPUEstimator` 中，您则应传递一个函数（该函数会返回指标字典）和一个参数张量列表，如下所示：

```python
def my_metric_fn(labels, predictions):
   return {'accuracy': tf.metrics.accuracy(labels, predictions)}

return tf.contrib.tpu.TPUEstimatorSpec(
  ...
  eval_metrics=(my_metric_fn, [labels, predictions])
)
```

### 使用 `TPUEstimatorSpec`

`TPUEstimatorSpec` 不支持钩子，且部分字段需要函数封装器。

`Estimator` 的 `model_fn` 必须返回 `EstimatorSpec`。`EstimatorSpec` 是一种命名字段的简单结构，包含所有 `tf.Tensors`（属于 Estimator 可能需要与之交互的模型）。

`TPUEstimators` 使用 `tf.contrib.tpu.TPUEstimatorSpec`。它和标准 `tf.estimator.EstimatorSpec` 之间有一些差异：


*  `eval_metric_ops` 必须封装到 `metrics_fn`中，此字段已重命名为 `eval_metrics` ([参考上文](#metrics))。
*  `tf.train.SessionRunHook` 不受支持，因此这些字段已忽略。
*  `tf.train.Scaffold`（如果使用）也必须封装在函数中。此字段已重命名为 `scaffold_fn`。

`Scaffold` 和 `Hooks` 用于高级用途，通常可以忽略。

## 输入函数

输入函数在主机（而非 Cloud TPU）上运行，因此通常保持不变。本节介绍了两项必要调整。

### Params 参数

标准 `Estimator` 的 `input_fn` 可以包含 `params` 参数；`TPUEstimator` 的 `input_fn` 必须包含 `params` 参数。要让 `Estimator` 为各个输入流副本设置批量大小，这是必须的。因此 `TPUEstimator` 的 `input_fn` 的最小签名是：

```
def my_input_fn(params):
  pass
```

其中 `params['batch-size']` 包含批量大小。

### 静态形状和批次大小

`input_fn` 生成的输入管道在 CPU 上运行。因此通常不受 XLA/TPU 环境设定的严格静态形状要求的限制。一个要求是从输入管道到 TPU 的批次数据必须具备静态形状，具体由标准 TensorFlow 形状推理算法确定。中间张量可具备动态形状。如果形状推理失败，但形状已知，则可以使用 `tf.set_shape()` 应用正确的形状。

在下面的示例中，形状推理算法失败了，但 `set_shape` 使用正确：

```
>>> x = tf.zeros(tf.constant([1,2,3])+1)
>>> x.shape

TensorShape([Dimension(None), Dimension(None), Dimension(None)])

>>> x.set_shape([2,3,4])
```

在很多情况下，批次大小是唯一未知的维度。

使用 `tf.data` 的典型输入管道通常会生成固定大小的批次。不过，最后一批有限 `Dataset` 通常较小，仅包含剩余元素。由于 `Dataset` 不知道自身的长度或有限性，因此标准 `batch` 方法无法确定是否所有批次都有自己的固定大小：

```
>>> params = {'batch_size':32}
>>> ds = tf.data.Dataset.from_tensors([0, 1, 2])
>>> ds = ds.repeat().batch(params['batch-size'])
>>> ds

<BatchDataset shapes: (?, 3), types: tf.int32>
```

最直接的解决方法是应用 `tf.contrib.data.batch_and_drop_remainder`，如下所示：

```
>>> params = {'batch_size':32}
>>> ds = tf.data.Dataset.from_tensors([0, 1, 2])
>>> ds = ds.repeat().apply(
...     tf.contrib.data.batch_and_drop_remainder(params['batch-size']))
>>> ds

 <_RestructuredDataset shapes: (32, 3), types: tf.int32>
```

顾名思义，这种方法的一个缺点是这种批量方法可能导致数据集末尾出现非整数批量。这对用于训练的无限重复数据集来说非常适用，但如果要训练确切数量的周期，则可能会出现问题。

要执行正好 1 个周期的评估，您可以手动填充批次长度并在创建 `tf.metrics` 时将填充条目的权重设为零。

## 数据集

有效使用 `tf.data.Dataset` API 至关重要，因为除非您能够快速向 TPU 提供数据，否则无法使用 Cloud TPU。有关数据集性能的详细信息，请参阅 [输入管道性能指南](/docs/tensorflow/performance/datasets_performance)。

除了最简单的实验（使用 `tf.data.Dataset.from_tensor_slices` 或其他图内数据），您需要将 `TPUEstimator` 的 `Dataset` 读取的所有数据文件存储在 Google Cloud Storage 存储分区中。

对于大多数用例，我们建议将数据转换为 TFRecord 格式并使用 `tf.data.TFRecordDataset` 读取数据。不过，这不是硬性要求，您也可以根据偏好使用其他数据集读取器（`FixedLengthRecordDataset` 或 `TextLineDataset`）。

小数据集可使用 `tf.data.Dataset.cache` 完全加载到内存中。

无论使用何种数据格式，我们强烈建议您[使用大文件](/docs/tensorflow/performance/performance_guide.md#use_large_files)（约 100MB）。这在网络化设置中尤其重要，因为打开文件的开销要大很多。

无论使用哪种读取器，使用构造函数的 `buffer_size` 参数启用缓存也十分重要。该参数按字节指定。建议使用至少几 MB (`buffer_size=8*1024*1024`)，以便在需要时提供数据。

TPU-demo repo 包含一个用于下载 imagenet 数据集并将其转换为合适格式的[教本](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py)
。 这与 repo 中包含的 imagenet
[模型](https://github.com/tensorflow/tpu/tree/master/models)共同介绍了所有这些最佳做法。


## 后续步骤

* 要详细了解如何实际设置和运行 Cloud TPU，请参阅：[Google Cloud TPU 文档](https://cloud.google.com/tpu/docs/)
* 本文并非详尽无遗。要详细了解如何创建 Cloud TPU 兼容模型，请参阅以下网页中的示例模型： [TPU 演示代码库](https://github.com/tensorflow/tpu)
* 要详细了解如何调整 TensorFlow 代码性能，请参阅：[Google Cloud TPU 性能指南](https://cloud.google.com/tpu/docs/performance-guide)
