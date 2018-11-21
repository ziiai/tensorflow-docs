# TensorBoard：可视化学习

TensorFlow 可用于训练大规模深度神经网络所需的计算，使用该工具涉及的计算往往复杂而深奥。为了更方便 TensorFlow 程序的理解、调试与优化，我们发布了一套名为 TensorBoard 的可视化工具。您可以用 TensorBoard 来展现 TensorFlow 图，绘制图像生成的定量指标图以及显示附加数据（如其中传递的图像）。当 TensorBoard 完全配置好后，它将显示如下：

![MNIST TensorBoard](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/mnist_tensorboard.png "MNIST TensorBoard")

<!--
<div class="video-wrapper">
  <iframe class="devsite-embedded-youtube-video" data-video-id="eBbEDRsCmv4"
          data-autohide="1" data-showinfo="0" frameborder="0" allowfullscreen>
  </iframe>
</div>

This 30-minute tutorial is intended to get you started with simple TensorBoard
usage. It assumes a basic understanding of TensorFlow.

There are other resources available as well! The
-->
[TensorBoard GitHub](https://github.com/tensorflow/tensorboard)
中提供了其他大量关于在 TensorBoard 中使用各个信息中心的信息，包括提示和技巧以及调试信息。

## 设置

[安装 TensorFlow](../install/)。通过 pip 安装 TensorFlow 时，也会自动安装 TensorBoard。

## 数据序列化

TensorBoard 通过读取 TensorFlow 的事件文件来运行。TensorFlow 的事件文件包含运行 TensorFlow 时生成的总结数据。下面是 TensorBoard 中总结数据的一般生命周期。

首先，创建您想从中收集总结数据的 TensorFlow 图，然后再选择您想在哪个节点标注总结指令。

比如，假设您正在训练一个卷积神经网络，用于识别 MNIST 数据。您可能希望记录随着时间的推移，学习速度如何变化，以及目标函数如何变化。为了收集这些数据，您可以分别向输出学习速率和损失的节点附加 `tf.summary.scalar` 操作。然后，为每个 `scalar_summary` 分配一个有意义的 `tag`，如 `'learning rate'` 或 `'loss function'`。

或许您还希望显示特定层中激活函数的分布情况，或者显示梯度或权重的分布情况。为了收集这些数据，您可以分别向梯度输出和存储权重的变量附加 `tf.summary.histogram` 操作。

有关所有可用的总结指令的详细信息，可查看总结指令文档。

在 TensorFlow 中，只有当您运行指令时，指令才会执行，或者另一个 op 依赖于指令的输出时，指令才会运行。我们刚才创建的这些总结节点都围绕着您的图：您目前运行的 op 都不依赖于这些节点的结果。因此，为了生成总结信息，我们需要运行所有这些总结节点。这样的手动操作是枯燥而乏味的，因此可以使用 `tf.summary.merge_all` 将这些操作合并为一个操作，从而生成所有汇总数据。

然后您可以执行该合并的总结 op，它会在特定步骤将所有总结数据生成一个序列化的 `Summary protobuf` 对象。最后，要将此汇总数据写入磁盘，请将汇总 `protobuf` 传递给 `tf.summary.FileWriter`。

`FileWriter` 的构造函数中包含了参数 `logdir`。`logdir` 参数非常重要，所有事件都会写到它所指的目录下。 此外，`FileWriter` 的构造函数中可包含可选参数 `Graph`。如果 TensorBoard 接收到 `Graph` 对象，则会将图与张量形状信息一起可视化。这将使您更清楚地了解图的内涵：请参阅
[张量形状信息](../guide/graph_viz.md#tensor-shape-information).

现在您已经修改了图并具备 `FileWriter`，可以开始运行网络了！如果您愿意，可以每一步运行一次此合并的总结 op，并记录大量的训练数据。不过，可能会有一些您不需要的数据。因此，您可以考虑改为每 `n` 步运行一次合并的总结 op。

以下代码示例基于
[简单 MNIST 教程](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py)
改编，我们在其中添加了一些总结 `op`，然后每十步运行一次。如果您将其运行，然后启动 `tensorboard --logdir=/tmp/tensorflow/mnist`，您就可将统计数据可视化，如可显示训练期间权重或准确性的变化。以下是节选的代码片段，要查看完整源代码请点击
[此处](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py)。

```python
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  tf.summary.scalar('dropout_keep_probability', keep_prob)
  dropped = tf.nn.dropout(hidden1, keep_prob)

# Do not apply softmax activation yet, see below.
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

with tf.name_scope('cross_entropy'):
  # The raw formulation of cross-entropy,
  #
  # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                               reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.losses.sparse_softmax_cross_entropy on the
  # raw logit outputs of the nn_layer above.
  with tf.name_scope('total'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
      cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                      sess.graph)
test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
tf.global_variables_initializer().run()
```

我们初始化 `FileWriters` 后，在训练和测试模型时，必须向 `FileWriters` 添加总结。

```python
# Train the model, and also write summaries.
# Every 10th step, measure test-set accuracy, and write test summaries
# All other steps, run train_step on training data, & add training summaries

def feed_dict(train):
  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  if train or FLAGS.fake_data:
    xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
    k = FLAGS.dropout
  else:
    xs, ys = mnist.test.images, mnist.test.labels
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}

for i in range(FLAGS.max_steps):
  if i % 10 == 0:  # Record summaries and test-set accuracy
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    test_writer.add_summary(summary, i)
    print('Accuracy at step %s: %s' % (i, acc))
  else:  # Record train set summaries, and train
    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
    train_writer.add_summary(summary, i)
```

您已完成设置，可以使用 TensorBoard 对数据进行可视化了。

## 启动 TensorBoard

要运行 TensorBoard，请使用以下命令（或者 `python -m tensorboard.main`）

```bash
tensorboard --logdir=path/to/log-directory
```

其中，`logdir` 指向 `FileWriter` 将数据序列化的目录。如果此 `logdir` 目录下有子目录，而子目录包含基于各个运行的序列化数据，则 TensorBoard 会将所有这些运行涉及的数据都可视化。TensorBoard 运行后，请在您的网络浏览器中转到 `localhost:6006` 以查看 TensorBoard。

查看 TensorBoard 时，您会看到右上角的导航标签。每个标签代表一组可供可视化的序列化数据。

要深入了解如何使用“图”标签将图可视化，请参阅 [TensorBoard: 图的可视化](../guide/graph_viz.md).

有关更多 TensorBoard 通用使用信息，请参阅 [TensorBoard GitHub](https://github.com/tensorflow/tensorboard).
