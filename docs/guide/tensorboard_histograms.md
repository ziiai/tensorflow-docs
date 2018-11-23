#  TensorBoard 直方图信息中心

TensorBoard 直方图信息中心用于显示在 TensorFlow 图中某些 `Tensor` 随着时间推移而变化的分布。即，该信息中心可显示在不同时间点对应张量的许多张直方图图示。

## 基本示例

让我们从一个简单的例子切入：某个正态分布变量，其平均值随着时间推移而变化。TensorFlow 中的操作 `tf.random_normal` 可完美实现此目的。按照 TensorBoard 中的常用做法，我们将使用汇总操作（本示例中为 `tf.summary.histogram`）提取数据。要大致了解汇总方式的工作原理，请参阅
[TensorBoard 指南](/docs/tensorflow/guide/summaries_and_tensorboard).

以下代码段将生成一些包含正态分布数据的直方图汇总，其中该分布的均值将随着时间的推移而增大。

```python
import tensorflow as tf

k = tf.placeholder(tf.float32)

# Make a normal distribution, with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
# Record that distribution into a histogram summary
tf.summary.histogram("normal/moving_mean", mean_moving_normal)

# Setup a session and summary writer
sess = tf.Session()
writer = tf.summary.FileWriter("/tmp/histogram_example")

summaries = tf.summary.merge_all()

# Setup a loop and write the summaries to disk
N = 400
for step in range(N):
  k_val = step/float(N)
  summ = sess.run(summaries, feed_dict={k: k_val})
  writer.add_summary(summ, global_step=step)
```

运行代码后，我们可以通过以下命令行将数据加载到 TensorBoard 中：


```sh
tensorboard --logdir=/tmp/histogram_example
```

TensorBoard 运行时，在 Chrome 或 Firefox 加载该数据，并导航到直方图信息中心。然后我们可以即可看到正态分布数据的直方图图示。

![](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/tensorboard/histogram_dashboard/1_moving_mean.png)

`tf.summary.histogram` 接受任意大小和形状的张量，并将该张量压缩成一个由许多分箱组成的直方图数据结构，这些分箱有各种宽度和计数。例如，假设我们要将数字 `[0.5, 1.1, 1.3, 2.2, 2.9, 2.99]` 整理到不同的分箱中，我们可以创建三个分箱： * 一个分箱包含 0 到 1 之间的所有数字（会包含一个元素：0.5）， * 一个分箱包含 1 到 2 之间的所有数字（会包含两个元素：1.1 和 1.3）， * 一个分箱包含 2 到 3 之间的所有数字（会包含三个元素：2.2、2.9 和 2.99）。

TensorFlow 使用类似的方法创建分箱，但与我们的示例不同，它不创建整数分箱。对于大型稀疏数据集，可能会导致数千个分箱。相反，
[这些分箱呈指数分布，许多分箱接近 0，有较少的分箱的数值较大](https://github.com/tensorflow/tensorflow/blob/c8b59c046895fa5b6d79f73e0b5817330fcfbfc1/tensorflow/core/lib/histogram/histogram.cc#L28)。
然而，将指数分布的分箱可视化是非常艰难的。如果将高度用于为计数编码，那么即使元素数量相同，较宽的分箱所占的空间也越大。反过来推理，如果用面积为计数编码，则使高度无法比较。因此，直方图会
[将数据重新采样](https://github.com/tensorflow/tensorflow/blob/17c47804b86e340203d451125a721310033710f1/tensorflow/tensorboard/components/tf_backend/backend.ts#L400)
并分配到统一的分箱。很不幸，在某些情况下，这可能会造成假象。

直方图可视化工具中的每个切片显示单个直方图。切片是按步骤整理的；较早的切片（如，步骤 0）位于较“靠后”的位置，颜色也较深，而较晚的切片（如，步骤 400）则靠近前景，颜色也较浅。右侧的 y 轴显示步骤编号。

您可以将鼠标悬停在直方图上以查看包含更多详细信息的工具提示。例如，在下面的图片中，我们可以看到，时间步骤 176 对应的直方图的分箱位于 2.25 附近，分箱中有 177 个元素。

![](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/tensorboard/histogram_dashboard/2_moving_mean_tooltip.png)

另外，您可能会注意到，直方图切片并不总是按步数或时间间隔均匀分布。这是因为 TensorBoard 使用
[蓄水池抽样](https://en.wikipedia.org/wiki/Reservoir_sampling) 
来保留所有直方图的子集，以节省内存。蓄水池抽样保证每个样本都有相同的被包含的可能性，但是因为它是一个随机算法，选择的样本不会按步数均匀出现。

## 覆盖模式

信息中心左侧有一个控件，可以将直方图模式从“偏移”切换到“覆盖”：

![](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/tensorboard/histogram_dashboard/3_overlay_offset.png)

在“偏移”模式下，可视化旋转 45 度，以便各个直方图切片不再按时间展开，而是全部绘制在相同的 y 轴上。

![](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/tensorboard/histogram_dashboard/4_overlay.png)
现在，每个切片都是图表上的一条单独线条，y 轴显示的是每个分箱内的项目数。颜色较深的线条表示较早的步，而颜色较浅的线条表示较晚的步。同样，您可以将鼠标悬停在图表上以查看其他一些信息。

![](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/tensorboard/histogram_dashboard/5_overlay_tooltips.png)

一般来说，如果要直接比较不同直方图的计数，重叠式可视化图表将非常有用。

## 多峰分布

直方图信息中心非常适合可视化多峰分布。我们通过合并两个不同正态分布的输出构造一个简单的双峰分布。代码如下所示：

```python
import tensorflow as tf

k = tf.placeholder(tf.float32)

# Make a normal distribution, with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
# Record that distribution into a histogram summary
tf.summary.histogram("normal/moving_mean", mean_moving_normal)

# Make a normal distribution with shrinking variance
variance_shrinking_normal = tf.random_normal(shape=[1000], mean=0, stddev=1-(k))
# Record that distribution too
tf.summary.histogram("normal/shrinking_variance", variance_shrinking_normal)

# Let's combine both of those distributions into one dataset
normal_combined = tf.concat([mean_moving_normal, variance_shrinking_normal], 0)
# We add another histogram summary to record the combined distribution
tf.summary.histogram("normal/bimodal", normal_combined)

summaries = tf.summary.merge_all()

# Setup a session and summary writer
sess = tf.Session()
writer = tf.summary.FileWriter("/tmp/histogram_example")

# Setup a loop and write the summaries to disk
N = 400
for step in range(N):
  k_val = step/float(N)
  summ = sess.run(summaries, feed_dict={k: k_val})
  writer.add_summary(summ, global_step=step)
```

你已经在上面的例子中见过“均值变动”的正态分布。现在还有一个“缩小方差”分布。并排显示如下：
![](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/tensorboard/histogram_dashboard/6_two_distributions.png)

当我们合并这两个分布后，获得一张清晰地显示发散、双峰结构的图表：
![](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/tensorboard/histogram_dashboard/7_bimodal.png)

## 更多分布

为了增添乐趣，我们来生成并可视化更多的分布图，然后将它们合并到一个图表中。以下是我们要使用的代码：

```python
import tensorflow as tf

k = tf.placeholder(tf.float32)

# Make a normal distribution, with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
# Record that distribution into a histogram summary
tf.summary.histogram("normal/moving_mean", mean_moving_normal)

# Make a normal distribution with shrinking variance
variance_shrinking_normal = tf.random_normal(shape=[1000], mean=0, stddev=1-(k))
# Record that distribution too
tf.summary.histogram("normal/shrinking_variance", variance_shrinking_normal)

# Let's combine both of those distributions into one dataset
normal_combined = tf.concat([mean_moving_normal, variance_shrinking_normal], 0)
# We add another histogram summary to record the combined distribution
tf.summary.histogram("normal/bimodal", normal_combined)

# Add a gamma distribution
gamma = tf.random_gamma(shape=[1000], alpha=k)
tf.summary.histogram("gamma", gamma)

# And a poisson distribution
poisson = tf.random_poisson(shape=[1000], lam=k)
tf.summary.histogram("poisson", poisson)

# And a uniform distribution
uniform = tf.random_uniform(shape=[1000], maxval=k*10)
tf.summary.histogram("uniform", uniform)

# Finally, combine everything together!
all_distributions = [mean_moving_normal, variance_shrinking_normal,
                     gamma, poisson, uniform]
all_combined = tf.concat(all_distributions, 0)
tf.summary.histogram("all_combined", all_combined)

summaries = tf.summary.merge_all()

# Setup a session and summary writer
sess = tf.Session()
writer = tf.summary.FileWriter("/tmp/histogram_example")

# Setup a loop and write the summaries to disk
N = 400
for step in range(N):
  k_val = step/float(N)
  summ = sess.run(summaries, feed_dict={k: k_val})
  writer.add_summary(summ, global_step=step)
```
### 伽玛分布
![](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/tensorboard/histogram_dashboard/8_gamma.png)

### 均匀分布
![](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/tensorboard/histogram_dashboard/9_uniform.png)

### 泊松分布
![](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/tensorboard/histogram_dashboard/10_poisson.png)
泊松分布以整数为基础。所以，所有生成的值都完全是整数。直方图压缩将数据移动到按浮点数划分的分箱中，导致生成的图表在整数值上显示出很小的隆起，而不是锐利的尖峰。

### 全合并到一起
最后，我们可以将所有数据合并成一个有趣的曲线图。
![](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/tensorboard/histogram_dashboard/11_all_combined.png)

