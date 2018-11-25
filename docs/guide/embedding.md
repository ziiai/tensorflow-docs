# 嵌入

本文档介绍了嵌入这一概念，并且举了一个简单的例子来说明如何在 TensorFlow 中训练嵌入，此外还说明了如何使用 TensorBoard Embedding Projector 查看嵌入（
([live example](http://projector.tensorflow.org)）。前两部分适合机器学习或 TensorFlow 新手，而 Embedding Projector 指南适合各个层次的用户。

有关这些概念的另一个教程，请参阅
[《机器学习速成课程》的“嵌入”部分](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture)。

**嵌入**是从离散对象（例如字词）到实数向量的映射。例如，英语字词的 300 维嵌入可能包括：

```
blue:  (0.01359, 0.00075997, 0.24608, ..., -0.2524, 1.0048, 0.06259)
blues:  (0.01396, 0.11887, -0.48963, ..., 0.033483, -0.10007, 0.1158)
orange:  (-0.24776, -0.12359, 0.20986, ..., 0.079717, 0.23865, -0.014213)
oranges:  (-0.35609, 0.21854, 0.080944, ..., -0.35413, 0.38511, -0.070976)
```

这些向量中的各个维度通常没有固有含义，机器学习所利用的是向量的位置和相互之间的距离这些整体模式。

嵌入对于机器学习的输入非常重要。分类器（更笼统地说是神经网络）适用于实数向量。它们训练密集向量时效果最佳，其中所有值都有助于定义对象。不过，机器学习的很多重要输入（例如文本的字词）没有自然的向量表示。嵌入函数是将此类离散输入对象转换为有用连续向量的标准和有效方法。

嵌入作为机器学习的输出也很有价值。由于嵌入将对象映射到向量，因此应用可以将向量空间中的相似性（例如欧几里德距离或向量之间的角度）用作一项强大而灵活的标准来衡量对象相似性。一个常见用途是找到最近的邻点。例如，下面是采用与上述相同的字词嵌入后，每个字词的三个最近邻点和相应角度：

```
blue:  (red, 47.6°), (yellow, 51.9°), (purple, 52.4°)
blues:  (jazz, 53.3°), (folk, 59.1°), (bluegrass, 60.6°)
orange:  (yellow, 53.5°), (colored, 58.0°), (bright, 59.9°)
oranges:  (apples, 45.3°), (lemons, 48.3°), (mangoes, 50.4°)
```

这样应用就会知道，在某种程度上，苹果和橙子（相距 45.3°）的相似度高于柠檬和橙子（相距 48.3°）。

## TensorFlow 中的嵌入

要在 TensorFlow 中创建字词嵌入，我们首先将文本拆分成字词，然后为词汇表中的每个字词分配一个整数。我们假设已经完成了这一步，并且 `word_ids` 是这些整数的向量。例如，可以将“I have a cat.”这个句子拆分成 `[“I”, “have”, “a”, “cat”, “.”]`，那么相应 `word_ids` 张量的形状将是 `[5]`，并且包含 5 个整数。为了将这些字词 ID 映射到向量，我们需要创建嵌入变量并使用 `tf.nn.embedding_lookup` 函数，如下所示：

```
word_embeddings = tf.get_variable(“word_embeddings”,
    [vocabulary_size, embedding_size])
embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, word_ids)
```

完成此操作后，示例中张量 embedded_word_ids 的形状将是 [5, embedding_size]，并且包含全部 5 个字词的嵌入（密集向量）。在训练结束时，word_embeddings 将包含词汇表中所有字词的嵌入。

嵌入可以通过很多网络类型进行训练，并具有各种损失函数和数据集。例如，对于大型句子语料库，可以使用递归神经网络根据上一个字词预测下一个字词，还可以训练两个网络来进行多语言翻译。
[字词的向量表示](../tutorials/representation/word2vec.md)
教程中介绍了这些方法。

## 直观显示嵌入

TensorBoard 包括 `Embedding Projector`，这是一款可让您以交互的方式直观显示嵌入的工具。此工具可以读取模型中的嵌入，并以二维或三维方式渲染这些嵌入。

Embedding Projector 具有三个面板：

- 数据面板：位于左上方，您可以在其中选择运行、嵌入变量和数据列，以对点进行着色和标记。
- 投影面板：位于左下方，您可以在其中选择投影类型。
- 检查工具面板：位于右侧，您可以在其中搜索特定点并查看最近邻点的列表。

### 投影

Embedding Projector 提供三种方法来降低数据集的维度。

- *[t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)*：
  一种非线性不确定性算法（T 分布式随机邻点嵌入），它会尝试保留数据中的局部邻点，通常以扭曲全局结构为代价。您可以选择是计算二维还是三维投影。

- *[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)*：
  一种线性确定性算法（主成分分析），它尝试用尽可能少的维度捕获尽可能多的数据可变性。PCA 会突出数据中的大规模结构，但可能会扭曲本地邻点。Embedding Projector 会计算前 10 个主成分，您可以从中选择两三个进行查看。

- *自定义*：
  线性投影到您使用数据中的标签指定的水平轴和垂直轴上。例如，您可以通过为“左”和“右”指定文本格式来定义水平轴。Embedding Projector 会查找标签与“左”格式相匹配的所有点，并计算这些点的形心；“右”格式与此类似。穿过这两个形心的线定义了水平轴。同样地，计算与“上”和“下”文本格式相匹配的点的形心可定义垂直轴。

要查看其他实用文章，请参阅
[如何有效使用 t-SNE](https://distill.pub/2016/misread-tsne/) 和
[直观介绍主成分分析](http://setosa.io/ev/principal-component-analysis/).

### 探索

您可以使用自然的点击并拖动手势来缩放、旋转和平移，从而进行直观探索。将鼠标悬停在某个点上即可看到该点的所有
[metadata](#metadata)
。您还可以检查最近的邻点子集。点击某个点以后，右窗格中会列出最近的领点，以及到当前点的距离。投影中还会突出显示最近的邻点。

有时，将视图限制为点的子集并仅投影这些点非常有用。要执行此操作，您可以通过多种方式选择点：

    点击某个点之后，其最近的邻点也会处于选中状态。
    搜索之后，与查询匹配的点会处于选中状态。
    启用选择，点击某个点并拖动可定义选择范围。

然后，点击右侧检查工具窗格顶部的“隔离 nnn 个点”按钮。下图显示已选择 101 个点，因此用户可以点击“隔离 101 个点”：

![Selection of nearest neighbors](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/embedding-nearest-points.png "Selection of nearest neighbors")

*在字词嵌入数据集中选择“重要”一词的最近邻点*。

高级技巧：使用自定义投影进行过滤可能会非常有用。我们在下图中滤出了“政治”一词的 100 个最近邻点，并将它们投影到“最差”-“最优”向量上作为 x 轴。y 轴是随机的。这样一来，我们可以发现“想法”、“科学”、“视角”、“新闻”这些字词位于右侧，而“危机”、“暴力”和“冲突”这些字词位于左侧。

<table width="100%;">
  <tr>
    <td style="width: 30%;">
      <img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/embedding-custom-controls.png" alt="Custom controls panel" title="Custom controls panel" />
    </td>
    <td style="width: 70%;">
      <img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/embedding-custom-projection.png" alt="Custom projection" title="Custom projection" />
    </td>
  </tr>
  <tr>
    <td style="width: 30%;">
       自定义投影控件。
    </td>
    <td style="width: 70%;">
       “政治”的邻点到“最优”-“最差”向量的自定义投影。
    </td>
  </tr>
</table>

要分享您的发现，可以使用右下角的书签面板并将当前状态（包括任何投影的计算坐标）保存为小文件。接着可以将 Projector 指向一个包含一个或多个这些文件的集合，从而生成下面的面板。然后，其他用户就可以查看一系列书签。

<img src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/embedding-bookmark.png" alt="Bookmark panel" style="width:300px;">

### 元数据

如果您使用嵌入，则可能需要向数据点附加标签/图片。您可以通过生成一个元数据文件（其中包含每个点的标签），并在 Embedding Projector 的数据面板中点击“加载数据”来完成此操作。

元数据可以是标签，也可以是图片，它们存储在单独的文件中。如果是标签，则格式应该是 [TSV file](https://en.wikipedia.org/wiki/Tab-separated_values)
（制表符显示为红色），其中第一行包含列标题（以粗体显示），而后续行包含元数据值。例如：

<code>
<b>Word<span style="color:#800;">\t</span>Frequency</b><br/>
  Airplane<span style="color:#800;">\t</span>345<br/>
  Car<span style="color:#800;">\t</span>241<br/>
  ...
</code>

假设元数据文件中的行顺序与嵌入变量中的向量顺序相匹配，但标题除外。那么，元数据文件中的第 (i+1) 行对应于嵌入变量的第 i 行。如果 TSV 元数据文件仅有一列，那么不会有标题行，并且假设每行都是嵌入的标签。我们之所以包含此例外情况，是因为它与常用的“词汇表文件”格式相匹配。

要将图片用作元数据，您必须生成一个
[sprite image](https://www.google.com/webhp#q=what+is+a+sprite+image),
，其中包含小缩略图，嵌入中的每个向量都有一个小缩略图。sprite 应该按照行在前的顺序存储缩略图：将第一个数据点放置在左上方，最后一个数据点放在右下方，但是最后一行不必填充，如下所示。

<table style="border: none;">
<tr style="background-color: transparent;">
  <td style="border: 1px solid black">0</td>
  <td style="border: 1px solid black">1</td>
  <td style="border: 1px solid black">2</td>
</tr>
<tr style="background-color: transparent;">
  <td style="border: 1px solid black">3</td>
  <td style="border: 1px solid black">4</td>
  <td style="border: 1px solid black">5</td>
</tr>
<tr style="background-color: transparent;">
  <td style="border: 1px solid black">6</td>
  <td style="border: 1px solid black">7</td>
  <td style="border: 1px solid black"></td>
</tr>
</table>

点击[此链接](https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/embedding-mnist.mp4)可查看 Embedding Projector 中的一个有趣缩略图示例。


## 迷你版常见问题解答

**“嵌入”是一种操作还是一种事物？** 都是。人们一直说的是在向量空间中嵌入字词（操作），以及生成字词嵌入（事物）。两者的共同点在于嵌入这一概念，即从离散对象到向量的映射。创建或应用该映射是一种操作，但映射本身是一种事物。

**嵌入是高维度还是低维度？** 视情况而定。例如，与可包含数百万个字词和短语的向量空间相比，一个 300 维的字词和短语向量空间通常被视为低维度（且密集）空间。但从数学角度上来讲，它是高维度空间，显示的很多属性与人类直觉了解的二维和三维空间大相径庭。

**嵌入与嵌入层是否相同？** 不同。嵌入层是神经网络的一部分，而嵌入则是一个更宽泛的概念。
