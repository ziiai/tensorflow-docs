# 创建自定义 Estimator

本文档介绍了自定义 Estimator。具体而言，本文档介绍了如何创建自定义 Estimator 来模拟预创建的 Estimator DNNClassifier 在解决鸢尾花问题时的行为。要详细了解鸢尾花问题，请参阅 [预创建的 Estimator](/docs/tensorflow/guide/premade_estimators)这一章。

要下载和访问示例代码，请执行以下两个命令：

```shell
git clone https://github.com/tensorflow/models/
cd models/samples/core/get_started
```
在本文档中，我们将介绍 [`custom_estimator.py`](https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py)。您可以使用以下命令运行它：

```bsh
python custom_estimator.py
```

如果您时间并不充足，欢迎对比
[`custom_estimator.py`](https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py)
与
[`premade_estimator.py`](https://github.com/tensorflow/models/blob/master/samples/core/get_started/premade_estimator.py).
（位于同一个目录中）。



## 预创建的 Estimator 与自定义 Estimator

如下图所示，预创建的 Estimator 是 `tf.estimator.Estimator` 基类的子类，而自定义 Estimator 是 `tf.estimator.Estimator` 的实例：

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="display:block; margin: 0 auto"
  alt="Premade estimators are sub-classes of `Estimator`. Custom Estimators are usually (direct) instances of `Estimator`"
  src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/custom_estimators/estimator_types.png">
</div>
<div style="text-align: center">
 预创建的 Estimator 和自定义 Estimator 都是 Estimator。 
</div>

预创建的 Estimator 已完全成形。不过有时，您需要更好地控制 Estimator 的行为。这时，自定义 Estimator 就派上用场了。您可以创建自定义 Estimator 来完成几乎任何操作。如果您需要以某种不寻常的方式连接隐藏层，则可以编写自定义 Estimator。如果您需要为模型计算独特的指标，也可以编写自定义 Estimator。基本而言，如果您需要一个针对具体问题进行了优化的 Estimator，就可以编写自定义 Estimator。

模型函数（即 `model_fn`）会实现机器学习算法。采用预创建的 Estimator 和自定义 Estimator 的唯一区别是：

- 如果采用预创建的 Estimator，则有人已为您编写了模型函数。
- 如果采用自定义 Estimator，则您必须自行编写模型函数。

您的模型函数可以实现各种算法，定义各种各样的隐藏层和指标。与输入函数一样，所有模型函数都必须接受一组标准输入参数并返回一组标准输出值。正如输入函数可以利用 Dataset API 一样，模型函数可以利用 Layers API 和 Metrics API。

我们来看看如何使用自定义 Estimator 解决鸢尾花问题。快速提醒：以下是我们尝试模拟的鸢尾花模型的结构：

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="display:block; margin: 0 auto"
  alt="A diagram of the network architecture: Inputs, 2 hidden layers, and outputs"
  src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/custom_estimators/full_network.png">
</div>
<div style="text-align: center">
 我们的鸢尾花实现包含四个特征、两个隐藏层和一个对数输出层。 
</div>

## 编写输入函数

我们的自定义 Estimator 实现与我们的[预创建的 Estimator 实现](/docs/tensorflow/guide/premade_estimators)使用的是同一输入函数（来自 [`iris_data.py`](https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py)）。即：

```python
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
```

此输入函数会构建可以生成批次 `(features, labels)` 对的输入管道，其中 `features` 是字典特征。

## 创建特征列

按照 [Premade Estimators预创建的 Estimator](/docs/tensorflow/guide/premade_estimators) 和[特征列](/docs/tensorflow/guide/feature_columns)章节中详细介绍的内容，您必须定义模型的特征列来指定模型应该如何使用每个特征。无论是使用预创建的 Estimator 还是自定义 Estimator，您都要使用相同的方式定义特征列。

以下代码为每个输入特征创建一个简单的 `numeric_column`，表示应该将输入特征的值直接用作模型的输入：

```python
# Feature columns describe how to use the input.
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
```

## 编写模型函数

我们要使用的模型函数具有以下调用签名：

```python
def my_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration
```

前两个参数是从输入函数中返回的特征和标签批次；也就是说，`features` 和 `labels` 是模型将使用的数据的句柄。`mode` 参数表示调用程序是请求训练、预测还是评估。

调用程序可以将 params 传递给 Estimator 的构造函数。传递给构造函数的所有 params 转而又传递给 model_fn。在
[`custom_estimator.py`](https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py)
以下行将创建 Estimator 并设置参数来配置模型。此配置步骤与我们配置 `tf.estimator.DNNClassifier`（在
[预创建的 Estimator](/docs/tensorflow/guide/premade_estimators)中）的方式相似。

```python
classifier = tf.estimator.Estimator(
    model_fn=my_model_fn,
    params={
        'feature_columns': my_feature_columns,
        # Two hidden layers of 10 nodes each.
        'hidden_units': [10, 10],
        # The model must choose between 3 classes.
        'n_classes': 3,
    })
```

要实现一般的模型函数，您必须执行下列操作：

* [定义模型](#define_the_model).
* 分别为 [three different modes](#modes)指定其他计算：
    * [预测](#predict)
    * [评估](#evaluate)
    * [训练](#train)

## 定义模型

基本的深度神经网络模型必须定义下列三个部分：

* 一个 [输入层](https://developers.google.com/machine-learning/glossary/#input_layer)
* 一个或多个[隐藏层](https://developers.google.com/machine-learning/glossary/#hidden_layer)
* 一个 [输出层](https://developers.google.com/machine-learning/glossary/#output_layer)

### 定义输入层

在 `model_fn` 的第一行调用 `tf.feature_column.input_layer`，以将特征字典和 `feature_columns` 转换为模型的输入，如下所示：

```python
    # Use `input_layer` to apply the feature columns.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
```

上面的行会应用特征列定义的转换，从而创建模型的输入层。

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="display:block; margin: 0 auto"
  alt="A diagram of the input layer, in this case a 1:1 mapping from raw-inputs to features."
  src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/custom_estimators/input_layer.png">
</div>


### 隐藏层

如果您要创建深度神经网络，则必须定义一个或多个隐藏层。Layers API 提供一组丰富的函数来定义所有类型的隐藏层，包括卷积层、池化层和丢弃层。对于鸢尾花，我们只需调用 `tf.layers.dense` 来创建隐藏层，并使用 `params['hidden_layers']` 定义维度。在 `dense` 层中，每个节点都连接到前一层中的各个节点。下面是相关代码：

``` python
    # Build the hidden layers, sized according to the 'hidden_units' param.
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
```

* `units` 参数会定义指定层中输出神经元的数量。
* `activation` 参数会定义[激活函数](https://developers.google.com/machine-learning/glossary/#activation_function)
- 在这种情况下为 [Relu](https://developers.google.com/machine-learning/glossary/#ReLU)。

这里的变量 `net` 表示网络的当前顶层。在第一次迭代中，`net` 表示输入层。在每次循环迭代时，`tf.layers.dense` 都使用变量 `net` 创建一个新层，该层将前一层的输出作为其输入。

创建两个隐藏层后，我们的网络如下所示。为了简单起见，下图并未显示各个层中的所有单元。

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="display:block; margin: 0 auto"
  alt="The input layer with two hidden layers added."
  src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/custom_estimators/add_hidden_layer.png">
</div>

请注意，`tf.layers.dense` 提供很多其他功能，包括设置多种正则化参数的功能。不过，为了简单起见，我们只接受其他参数的默认值。

### 输出层

我们再次调用 `tf.layers.dense` 定义输出层，这次不使用激活函数：

```python
    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)
```

在这里，`net` 表示最后的隐藏层。因此，所有的层如下所示连接在一起：

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="display:block; margin: 0 auto"
  alt="A logit output layer connected to the top hidden layer"
  src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/custom_estimators/add_logits.png">
</div>
<div style="text-align: center">
 最后的隐藏层馈送到输出层。 
</div>

定义输出层时，`units` 参数会指定输出的数量。因此，通过将 `units` 设置为 `params['n_classes']`，模型会为每个类别生成一个输出值。输出向量的每个元素都将包含针对相关鸢尾花类别（山鸢尾、变色鸢尾或维吉尼亚鸢尾）分别计算的分数或“对数”。

之后，`tf.nn.softmax` 函数会将这些对数转换为概率。

## 实现训练、评估和预测

创建模型函数的最后一步是编写实现预测、评估和训练的分支代码。

每当有人调用 Estimator 的 `train`、`evaluate` 或 `predict` 方法时，就会调用模型函数。您应该记得，模型函数的签名如下所示：

``` python
def my_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys, see below
   params):  # Additional configuration
```

重点关注第三个参数 `mode`。如下表所示，当有人调用 `train`、`evaluate` 或 `predict` 时，Estimator 框架会调用模型函数并将 `mode` 参数设置为如下所示的值：

| Estimator 方法                 |    Estimator 模式 |
|:---------------------------------|:------------------|
|`tf.estimator.Estimator.train` |`tf.estimator.ModeKeys.TRAIN` |
|`tf.estimator.Estimator.evaluate`  |`tf.estimator.ModeKeys.EVAL`      |
|`tf.estimator.Estimator.predict`|`tf.estimator.ModeKeys.PREDICT` |

例如，假设您实例化自定义 Estimator 来生成名为 `classifier` 的对象。然后，您做出以下调用：

``` python
classifier = tf.estimator.Estimator(...)
classifier.train(input_fn=lambda: my_input_fn(FILE_TRAIN, True, 500))
```
然后，Estimator 框架会调用模型函数并将 `mode` 设为 `ModeKeys.TRAIN`。

模型函数必须提供代码来处理全部三个 `mode` 值。对于每个 `mode` 值，您的代码都必须返回 `tf.estimator.EstimatorSpec` 的一个实例，其中包含调用程序所需的信息。我们来详细了解各个 `mode`。

### 预测

如果调用 Estimator 的 `predict` 方法，则 `model_fn` 会收到 `mode = ModeKeys.PREDICT`。在这种情况下，模型函数必须返回一个包含预测的 `tf.estimator.EstimatorSpec`。

该模型必须经过训练才能进行预测。经过训练的模型存储在磁盘上，位于您实例化 Estimator 时建立的 `model_dir` 目录中。

此模型用于生成预测的代码如下所示：

```python
# Compute predictions.
predicted_classes = tf.argmax(logits, 1)
if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'class_ids': predicted_classes[:, tf.newaxis],
        'probabilities': tf.nn.softmax(logits),
        'logits': logits,
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
```
预测字典中包含模型在预测模式下运行时返回的所有内容。

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="display:block; margin: 0 auto"
  alt="Additional outputs added to the output layer."
  src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/custom_estimators/add_predictions.png">
</div>

`predictions` 存储的是下列三个键值对：

*   `class_ids` 存储的是类别 ID（0、1 或 2），表示模型对此样本最有可能归属的品种做出的预测
*   `probabilities` 存储的是三个概率（在本例中，分别是 0.02、0.95 和 0.03）
*   `logit` 存储的是原始对数值（在本例中，分别是 -1.3、2.6 和 -0.9）

我们通过 `predictions` 参数（属于 `tf.estimator.EstimatorSpec`）将该字典返回到调用程序。Estimator 的 `predict` 方法会生成这些字典。

### 计算损失

对于 [训练](#train)和[评估](#evaluate)，我们都需要计算模型的损失。这是要进行优化的[目标](https://developers.google.com/machine-learning/glossary/#objective)。

我们可以通过调用 `tf.losses.sparse_softmax_cross_entropy` 计算损失。当正确类别的概率（索引为 `label`）接近 1.0 时，此函数返回的值将最低，接近 0。随着正确类别的概率不断降低，返回的损失值越来越大。

此函数会针对整个批次返回平均值。

```python
# Compute loss.
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
```

### 评估

如果调用 Estimator 的 `evaluate` 方法，则 `model_fn` 会收到 `mode = ModeKeys.EVAL`。在这种情况下，模型函数必须返回一个包含模型损失和一个或多个指标（可选）的 `tf.estimator.EstimatorSpec`。

虽然返回指标是可选的，但大多数自定义 Estimator 至少会返回一个指标。TensorFlow 提供一个指标模块 `tf.metrics` 来计算常用指标。为简单起见，我们将只返回准确率。`tf.metrics.accuracy` 函数会将我们的预测值与真实值进行比较，即与输入函数提供的标签进行比较。`tf.metrics.accuracy` 函数要求标签和预测具有相同的形状。下面是对 `tf.metrics.accuracy` 的调用：

``` python
# Compute evaluation metrics.
accuracy = tf.metrics.accuracy(labels=labels,
                               predictions=predicted_classes,
                               name='acc_op')
```

针对评估返回的 `EstimatorSpec` 通常包含以下信息：

* `loss`：这是模型的损失
* `eval_metric_ops`：这是可选的指标字典。

我们将创建一个包含我们的唯一指标的字典。如果我们计算了其他指标，则将这些指标作为附加键值对添加到同一字典中。然后，我们将在 `eval_metric_ops` 参数（属于 `tf.estimator.EstimatorSpec`）中传递该字典。具体代码如下：

```python
metrics = {'accuracy': accuracy}
tf.summary.scalar('accuracy', accuracy[1])

if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=metrics)
```

`tf.summary.scalar` 会在 `TRAIN` 和 `EVAL` 模式下向 TensorBoard 提供准确率（后文将对此进行详细的介绍）。

### 训练

如果调用 Estimator 的 `train` 方法，则会调用 `model_fn` 并收到 `mode = ModeKeys.TRAIN`。在这种情况下，模型函数必须返回一个包含损失和训练操作的 `EstimatorSpec`。

构建训练操作需要优化器。我们将使用 `tf.train.AdagradOptimizer`，因为我们模仿的是 `DNNClassifier`，它也默认使用 `Adagrad`。`tf.train` 文件包提供很多其他优化器，您可以随意尝试它们。

下面是构建优化器的代码：

``` python
optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
```

接下来，我们使用优化器的 `minimize` 方法根据我们之前计算的损失构建训练操作。

`minimize` 方法还具有 `global_step` 参数。TensorFlow 使用此参数来计算已经处理过的训练步数（以了解何时结束训练）。此外，`global_step` 对于 TensorBoard 图能否正常运行至关重要。只需调用 `tf.train.get_global_step` 并将结果传递给 `minimize` 的 `global_step` 参数即可。

下面是训练模型的代码：

``` python
train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
```

针对训练返回的 `EstimatorSpec` 必须设置了下列字段：

* `loss`：包含损失函数的值。
* `train_op`：执行训练步。

下面是用于调用 `EstimatorSpec` 的代码：

```python
return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
```

模型函数现已完成。

## 自定义 Estimator

通过 Estimator 基类实例化自定义 Estimator，如下所示：

```python
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model_fn,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
        })
```
在这里，`params` 字典与 `DNNClassifier` 的关键字参数用途相同；即借助 `params` 字典，您无需修改 `model_fn` 中的代码即可配置 Estimator。

使用 Estimator 训练、评估和生成预测要用的其余代码与预创建的 Estimator 一章中的相同。例如，以下行将训练模型：

```python
# Train the Model.
classifier.train(
    input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
    steps=args.train_steps)
```

## TensorBoard

您可以在 TensorBoard 中查看自定义 Estimator 的训练结果。要查看相应报告，请从命令行启动 TensorBoard，如下所示：

```bsh
# Replace PATH with the actual path passed as model_dir
tensorboard --logdir=PATH
```

然后，通过以下网址打开 TensorBoard： [http://localhost:6006](http://localhost:6006)

所有预创建的 Estimator 都会自动将大量信息记录到 TensorBoard 上。不过，对于自定义 Estimator，TensorBoard 只提供一个默认日志（损失图）以及您明确告知 TensorBoard 要记录的信息。对于您刚刚创建的自定义 Estimator，TensorBoard 会生成以下内容：

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">

<img style="display:block; margin: 0 auto"
  alt="Accuracy, 'scalar' graph from tensorboard"
  src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/custom_estimators/accuracy.png">

<img style="display:block; margin: 0 auto"
  alt="loss 'scalar' graph from tensorboard"
  src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/custom_estimators/loss.png">

<img style="display:block; margin: 0 auto"
  alt="steps/second 'scalar' graph from tensorboard"
  src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/custom_estimators/steps_per_second.png">
</div>

<div style="text-align: center">
 TensorBoard 显示了三张图。 
</div>


简而言之，下面是三张图显示的内容：

* global_step/sec：这是一个性能指标，显示我们在进行模型训练时每秒处理的批次数（梯度更新）。

* loss：所报告的损失。

* accuracy：准确率由下列两行记录：

    * `eval_metric_ops={'my_accuracy': accuracy}`（评估期间）。
    * `tf.summary.scalar('accuracy', accuracy[1])`（训练期间）。

这些 Tensorboard 图是务必要将 `global_step` 传递给优化器的 `minimize` 方法的主要原因之一。如果没有它，模型就无法记录这些图的 `x` 坐标。

注意 `my_accuracy` 和 `loss` 图中的以下内容：

* 橙线表示训练。
* 蓝点表示评估。

在训练期间，系统会随着批次的处理定期记录摘要信息（橙线），因此它会变成一个跨越 x 轴范围的图形。

相比之下，评估在每次调用 `evaluate` 时仅在图上生成一个点。此点包含整个评估调用的平均值。它在图上没有宽度，因为它完全根据特定训练步（一个检查点）的模型状态进行评估。

如下图所示，您可以使用左侧的控件查看并选择性地停用/启用报告。

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="display:block; margin: 0 auto"
  alt="Check-boxes allowing the user to select which runs are shown."
  src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/custom_estimators/select_run.jpg">
</div>
<div style="text-align: center">
 启用或停用报告。 
</div>


## 总结

虽然使用预创建的 Estimator 可以快速高效地创建新模型，但您通常需要使用自定义 Estimator 才能实现所需的灵活性。幸运的是，预创建的 Estimator 和自定义 Estimator 采用相同的编程模型。唯一的实际区别是您必须为自定义 Estimator 编写模型函数；除此之外，其他都是相同的。

要了解详情，请务必查看：

* [官方 TensorFlow MNIST 实现](https://github.com/tensorflow/models/tree/master/official/mnist)：使用了自定义 Estimator。
* [TensorFlow 官方模型代码库](https://github.com/tensorflow/models/tree/master/official)：其中包含更多使用自定义 Estimator 的精选示例。
* [TensorBoard 视频](https://youtu.be/eBbEDRsCmv4)：介绍了 TensorBoard。
* [低阶 API 简介](/docs/tensorflow/guide/low_level_intro)：展示了如何直接使用 TensorFlow 的低阶 API 更轻松地进行调试。
