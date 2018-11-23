#  TensorBoard：图的直观展示

TensorFlow 计算图功能十分强大，但也非常复杂。图的直观展示可帮助您理解图并对图进行调试。以下示例显示了实际的可视化效果。

![Visualization of a TensorFlow graph](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/graph_vis_animation.gif "Visualization of a TensorFlow graph")
*TensorFlow 图的可视化效果。*

要查看您自己的图，请运行 TensorBoard 并将其指向作业的日志目录，点击顶部窗格中的“图”标签，然后使用左上角的菜单选择适当的运行条目。要深入了解如何运行 TensorBoard 并确保记录所有必要信息，请参阅 [TensorBoard：直观展示学习](/docs/tensorflow/guide/summaries_and_tensorboard)。

## 名称范围和节点

典型的 TensorFlow 图可能具有数千个节点 - 数量过多，难以一次性查看，甚至难以使用标准图工具来进行布置。为了进行简化，可以对变量名称设定范围，然后，直观展示工具可以使用此信息来为图中的节点定义层次结构。默认情况下仅显示该层次结构的顶层。以下示例使用 `tf.name_scope` 在 `hidden` 名称作用域下定义了三个操作：

```python
import tensorflow as tf

with tf.name_scope('hidden') as scope:
  a = tf.constant(5, name='alpha')
  W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
  b = tf.Variable(tf.zeros([1]), name='biases')
```

这导致出现以下三个 op 名称：

* `hidden/alpha`
* `hidden/weights`
* `hidden/biases`

默认情况下，直观展示工具会将这三个名称收起到一个标作 `hidden` 的节点中。其他详细信息不会丢失。您可以双击节点或点击右上角的橙色 `+` 符号以展开节点，然后您将看到 `alpha`、`weights` 和 `biases` 三个子节点。

以下真实示例展示了一个更加复杂的节点的初始状态和展开状态。

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/pool1_collapsed.png" alt="Unexpanded name scope" title="Unexpanded name scope" />
    </td>
    <td style="width: 50%;">
      <img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/pool1_expanded.png" alt="Expanded name scope" title="Expanded name scope" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      顶级名称范围 pool_1 的初始视图。点击右上角的橙色 + 按钮或双击节点本身即可展开该节点。
    </td>
    <td style="width: 50%;">
      pool_1 名称范围的展开视图。点击右上角的橙色 - 按钮或双击节点本身即可收起该名称范围。 
    </td>
  </tr>
</table>

要让图易于观看，按名称范围对节点进行分组至关重要。如果您要构建模型，名称范围可帮助您控制生成的直观展示。**您的名称范围越好，直观展示的效果也就越棒**。

上图显示了直观展示的另一方面。TensorFlow 图具有两种连接：数据依赖关系和控制依赖关系。数据依赖关系显示两个 op 之间的张量流动，该依赖关系显示为实线箭头，而控制依赖关系显示为虚线。在展开视图中（上图右侧），所有连接均为数据依赖关系，只有连接 `CheckNumerics` 和 `control_dependency` 的虚线除外。

还有一个技巧可以简化布局。大多数 TensorFlow 图都包含几个连接至许多其他节点的节点。例如，许多节点可能对初始化步骤存在控制依赖关系。绘制 `init` 节点及其依赖项之间的所有边缘时，会产生极为杂乱的视图。

为了减轻杂乱，直观展示工具会将所有高等级节点划分到右侧的辅助区域中，并且不绘制线来表示它们的边缘。我们不使用线而使用节点图标来表示连接。分出辅助节点时通常不会移除关键信息，因为这些节点通常与簿记函数相关。请参阅
[交互](#interaction) 了解如何在主图和辅助区域之间移动节点。

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/conv_1.png" alt="conv_1 is part of the main graph" title="conv_1 is part of the main graph" />
    </td>
    <td style="width: 50%;">
      <img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/save.png" alt="save is extracted as auxiliary node" title="save is extracted as auxiliary node" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      节点 conv_1 连接到 save。请注意它右侧的 save 节点小图标。
    </td>
    <td style="width: 50%;">
       save 的级别较高，它将显示为辅助节点。与 conv_1 的连接显示为左侧的节点图标。为了进一步减轻杂乱，由于 save 有许多连接，我们仅显示前 5 个连接，并将其余连接缩写为 ... 12 more。
    </td>
  </tr>
</table>

最后一种结构简化方法是序列收起。如下所示，序列模体 (motif)（即结构相同但名称末尾的数字不同的节点）将收起到节点的单个层叠中。对于具有长序列的网络，这极大地简化了视图。对于具有层次结构的节点，双击即可展开序列。请参阅
[交互](#interaction)了解如何为一组特定的节点停用/启用序列收起。

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/series.png" alt="Sequence of nodes" title="Sequence of nodes" />
    </td>
    <td style="width: 50%;">
      <img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/series_expanded.png" alt="Expanded sequence of nodes" title="Expanded sequence of nodes" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      节点序列收起后的视图。
    </td>
    <td style="width: 50%;">
      双击后显示的展开视图的一小部分内容。
    </td>
  </tr>
</table>

直观展示工具为了改进易读性而采取的最后一项辅助措施是：为常量和总结节点使用特殊图标。下表对节点符号进行了总结：

Symbol | Meaning
--- | ---
![Name scope](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/namespace_node.png "Name scope") | *High-level* node representing a name scope. Double-click to expand a high-level node.
![Sequence of unconnected nodes](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/horizontal_stack.png "Sequence of unconnected nodes") | Sequence of numbered nodes that are not connected to each other.
![Sequence of connected nodes](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/vertical_stack.png "Sequence of connected nodes") | Sequence of numbered nodes that are connected to each other.
![Operation node](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/op_node.png "Operation node") | An individual operation node.
![Constant node](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/constant.png "Constant node") | A constant.
![Summary node](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/summary.png "Summary node") | A summary node.
![Data flow edge](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/dataflow_edge.png "Data flow edge") | Edge showing the data flow between operations.
![Control dependency edge](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/control_edge.png "Control dependency edge") | Edge showing the control dependency between operations.
![Reference edge](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/reference_edge.png "Reference edge") | A reference edge showing that the outgoing operation node can mutate the incoming tensor.

## 交互 {#interaction}

通过平移和缩放在图中导航。点击并拖动可进行平移，使用滚动手势可进行缩放。双击节点或点击它的 + 按钮可展开代表一组指令的名称范围。为了在缩放和平移时轻松跟踪当前视点，右下角提供了一个小地图。

要关闭打开的节点，请再次双击节点或点击它的 - 按钮。您也可以点击一次以选择一个节点。该节点的颜色将加深，其相关详细信息及与其连接的节点将显示在直观展示工具右上角的信息卡上。

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/infocard.png" alt="Info card of a name scope" title="Info card of a name scope" />
    </td>
    <td style="width: 50%;">
      <img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/infocard_op.png" alt="Info card of operation node" title="Info card of operation node" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      显示了 conv2 名称范围的相关详情的信息卡。输入和输出是从名称范围内的指令节点的输入和输出合并而来。没有显示名称范围的属性。 
    </td>
    <td style="width: 50%;">
      显示了 DecodeRaw 指令节点的详细信息的信息卡。除了输入和输出以外，该卡还显示与当前指令关联的设备和和属性。 
    </td>
  </tr>
</table>

TensorBoard 提供了数种方法来改变图的视觉布局。这不会改变图的计算语义，但可让网络结构变得更加清晰。通过右键点击节点或按下节点信息卡底部的按钮，您可以对它的布局做出以下更改：

- 您可在主图和辅助区域之间移动节点。
- 您可以对一系列节点取消分组，使得序列中的节点不以分组的形式显示。您同样可以对取消分组的序列进行重新分组。

选中节点也有助于您理解高级别节点。请选择任意高级别节点，随后该节点的其他连接所对应的节点图标也将被选中。此操作可帮助您轻松了解正在保存哪些节点，以及没有保存哪些节点。

点击信息卡中的节点名称即可选中该节点。如有必要，视点将自动平移以便使节点显示出来。

最后，您可以使用图例上方的颜色菜单为您的图选择两种配色方案。默认的结构视图显示了结构：当两个高级别节点具有相同的结构时，它们将显示相同的彩虹色。具有独特结构的节点显示为灰色。第二个视图显示运行不同指令的设备。名称范围根据其内部指令的设备比例来按比例着色。

下图显示了一个真实图的一部分的图解。

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/colorby_structure.png" alt="Color by structure" title="Color by structure" />
    </td>
    <td style="width: 50%;">
      <img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/colorby_device.png" alt="Color by device" title="Color by device" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      结构视图：灰色节点具有独特的结构。橙色 conv1 和 conv2 节点具有相同的结构，其他颜色相同的节点同样如此。
    </td>
    <td style="width: 50%;">
      设备视图：名称范围根据其内部指令节点的设备比例来按比例着色。此处的紫色代表 GPU，绿色代表 CPU。
    </td>
  </tr>
</table>

## 张量形状信息

当序列化 `GraphDef` 包含张量形状时，图可视化工具会使用张量维度来标记边缘，而边缘厚度反映总张量大小。要在 `GraphDef` 中包括张量形状，请在对图执行序列化时，将实际图对象（如 `sess.graph` 中所示）传递到 `FileWriter`。下图显示了包含张量形状信息的 CIFAR-10 模型：
<table width="100%;">
  <tr>
    <td style="width: 100%;">
      <img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/tensor_shapes.png" alt="CIFAR-10 model with tensor shape information" title="CIFAR-10 model with tensor shape information" />
    </td>
  </tr>
  <tr>
    <td style="width: 100%;">
      包含张量形状信息的 CIFAR-10 模型。
    </td>
  </tr>
</table>

## 运行时统计信息

通常，收集运行的运行时元数据（例如节点的总内存使用量、总计算时间和张量形状）很有用。以下代码示例节选自修改版
[Estimators MNIST 教程](/docs/tensorflow/tutorials/estimators/cnn)的训练和测试部分，在该代码段中，我们记录了汇总和运行时统计信息。请参阅
[汇总教程](/docs/tensorflow/guide/summaries_and_tensorboard#serializing-the-data)
以详细了解如何记录汇总。要查看完整的源代码，请点击[此处](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py).

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
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
```

此代码将从第 99 步开始，每隔 100 步发出一次运行时统计。

当您启动 TensorBoard 并转至“图”(Graph) 标签时，您现在将在“会话运行”(Session runs)（与添加了运行元数据的步对应）下看到一些选项。选择其中一个运行后，系统将为您显示该步时的网络快照，并将未使用的节点显示为淡出。在左侧的控件中，您将能够按总内存或总计算时间为节点着色。此外，点击节点可显示确切的总内存量、计算时间和张量输出大小。


<table width="100%;">
  <tr style="height: 380px">
    <td>
      <img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/colorby_compute_time.png" alt="Color by compute time" title="Color by compute time"/>
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/run_metadata_graph.png" alt="Run metadata graph" title="Run metadata graph" />
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/run_metadata_infocard.png" alt="Run metadata info card" title="Run metadata info card" />
    </td>
  </tr>
</table>
