# 字词的向量表示法

在本教程中，我们将介绍由
[Mikolov 等](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
提供的 word2vec 模型。该模型用于学习字词的向量表示法，称为“字词嵌入”。

## 要点

本教程旨在重点介绍在 TensorFlow 中构建 word2vec 模型时的一些有趣且重要的部分。

- 我们将先说明将字词表示为向量的动机。
- 我们会介绍模型的原理及其训练方式（用数学方法进行有效衡量）。
- 我们还会在 TensorFlow 中展示模型的简单实现。
- 最后，我们会介绍如何提高该简单版本的扩展性能。

我们会在本教程的后面部分介绍代码，但如果您想直接深入研究代码，欢迎随时查看
[tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)
中的简化实现。此基本示例包含下载某些数据、根据这些数据进行训练以及可视化结果所需的代码。在您可以自如阅读和运行基本版本后，您就可以查看
[models/tutorials/embedding/word2vec.py](https://github.com/tensorflow/models/tree/master/tutorials/embedding/word2vec.py)
中更复杂的实现，其中展示了有关如何有效使用线程将数据移到文本模型、如何在训练期间设置检查点等更高级的 TensorFlow 原则。

首先，我们来了解一下为何要学习字词嵌入。如果您是嵌入方面的行家且只想弄清楚细节部分，请自行跳过此部分。

## 动机：为什么学习字词嵌入？

图像和音频处理系统采用的是庞大的高维度数据集，对于图像数据来说，此类数据集会编码为单个原始像素强度的向量，对于音频数据来说，则编码为功率谱密度系数。对于对象识别或语音识别这样的任务，我们知道成功执行任务所需的所有信息都在数据中进行了编码（因为人类可以从原始数据执行这些任务）。不过，自然语言处理系统一直以来都将字词视为离散的原子符号，因此“cat”可能表示为 `Id537`，“dog”可能表示为 `Id143`。这些编码是任意的，并未向系统提供有关各个符号之间可能存在的关系的有用信息。这意味着模型在处理有关“狗”的数据时，几乎不可能利用到它所学的关于“猫”的知识（例如它们都属于动物、宠物，有四条腿等）。将字词表示为唯一的离散 ID 还会导致数据稀疏性，并且通常意味着我们可能需要更多数据才能成功训练统计模型。使用向量表示法可以扫除其中一些障碍。

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/audio-image-text.png" alt>
</div>

[向量空间模型](https://en.wikipedia.org/wiki/Vector_space_model)
(VSM) 在连续向量空间中表示（嵌入）字词，其中语义相似的字词会映射到附近的点（“在彼此附近嵌入”）。VSM 在 NLP 方面有着悠久而丰富的历史，但所有方法均以某种方式依赖于
[分布假设](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_Hypothesis)
，这种假设指明在相同上下文中显示的字词语义相同。利用该原则的不同方法可分为两类：基于计数的方法（例如
[潜在语义分析](https://en.wikipedia.org/wiki/Latent_semantic_analysis)
）以及预测方法（例如
[神经概率语言模型](http://www.scholarpedia.org/article/Neural_net_language_models)）。

This distinction is elaborated in much more detail by
[Baroni 等](http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf)
很详细地阐述了这两大类别的区别。简而言之：基于计数的方法会计算在大型文本语料库中，一些字词与临近字词共同出现的频率统计数据，然后将这些计数统计数据向下映射到每个字词的小型密集向量。预测模型会根据学到的小型密集嵌入向量（被视为模型的参数），直接尝试预测临近的字词。

Word2vec 是一种计算效率特别高的预测模型，用于学习原始文本中的字词嵌入。它分为两种类型：连续词袋模型 (CBOW) 和 Skip-Gram 模型（请参阅
[Mikolov 等](https://arxiv.org/pdf/1301.3781.pdf))中的第 3.1 和 3.2 部分）。从算法上看，这些模型比较相似，只是 CBOW 从源上下文字词（“the cat sits on the”）中预测目标字词（例如“mat”），而 skip-gram 则逆向而行，从目标字词中预测源上下文字词。这种调换似乎是一种随意的选择，但从统计学上来看，它有助于 CBOW 整理很多分布信息（通过将整个上下文视为一个观察对象）。在大多数情况下，这对于小型数据集来说是很有用的。但是，skip-gram 将每个上下文-目标对视为一个新的观察对象，当我们使用大型数据集时，skip-gram 似乎能发挥更好的效果。在本教程接下来的部分，我们将重点介绍 skip-gram 模型。


## 通过噪声对比训练进行扩展

神经概率语言模型一直以来都使用
[最大似然率](https://en.wikipedia.org/wiki/Maximum_likelihood)
(ML) 原则进行训练，以最大限度地提高使用
[*softmax* 函数](https://en.wikipedia.org/wiki/Softmax_function)
根据之前的字词 \\(h\\)（表示“历史”字词）正确预测出下一个字词 \\(w_t\\)（表示“目标”字词）的概率。

$$
\begin{align}
P(w_t | h) &= \text{softmax}(\text{score}(w_t, h)) \\
           &= \frac{\exp \{ \text{score}(w_t, h) \} }
             {\sum_\text{Word w' in Vocab} \exp \{ \text{score}(w', h) \} }
\end{align}
$$

where \\(\text{score}(w_t, h)\\) computes the compatibility of word \\(w_t\\)
with the context \\(h\\) (a dot product is commonly used). We train this model
by maximizing its [log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function)
on the training set, i.e. by maximizing

$$
\begin{align}
 J_\text{ML} &= \log P(w_t | h) \\
  &= \text{score}(w_t, h) -
     \log \left( \sum_\text{Word w' in Vocab} \exp \{ \text{score}(w', h) \} \right).
\end{align}
$$

This yields a properly normalized probabilistic model for language modeling.
However this is very expensive, because we need to compute and normalize each
probability using the score for all other \\(V\\) words \\(w'\\) in the current
context \\(h\\), *at every training step*.

<div style="width:60%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/softmax-nplm.png" alt>
</div>

On the other hand, for feature learning in word2vec we do not need a full
probabilistic model. The CBOW and skip-gram models are instead trained using a
binary classification objective ([logistic regression](https://en.wikipedia.org/wiki/Logistic_regression))
to discriminate the real target words \\(w_t\\) from \\(k\\) imaginary (noise) words \\(\tilde w\\), in the
same context. We illustrate this below for a CBOW model. For skip-gram the
direction is simply inverted.

<div style="width:60%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/nce-nplm.png" alt>
</div>

Mathematically, the objective (for each example) is to maximize

$$J_\text{NEG} = \log Q_\theta(D=1 |w_t, h) +
  k \mathop{\mathbb{E}}_{\tilde w \sim P_\text{noise}}
     \left[ \log Q_\theta(D = 0 |\tilde w, h) \right]$$

where \\(Q_\theta(D=1 | w, h)\\) is the binary logistic regression probability
under the model of seeing the word \\(w\\) in the context \\(h\\) in the dataset
\\(D\\), calculated in terms of the learned embedding vectors \\(\theta\\). In
practice we approximate the expectation by drawing \\(k\\) contrastive words
from the noise distribution (i.e. we compute a
[Monte Carlo average](https://en.wikipedia.org/wiki/Monte_Carlo_integration)).

This objective is maximized when the model assigns high probabilities
to the real words, and low probabilities to noise words. Technically, this is
called
[Negative Sampling](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf),
and there is good mathematical motivation for using this loss function:
The updates it proposes approximate the updates of the softmax function in the
limit. But computationally it is especially appealing because computing the
loss function now scales only with the number of *noise words* that we
select (\\(k\\)), and not *all words* in the vocabulary (\\(V\\)). This makes it
much faster to train. We will actually make use of the very similar
[noise-contrastive estimation (NCE)](https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf)
loss, for which TensorFlow has a handy helper function `tf.nn.nce_loss()`.

下面我们在实践中直观了解下相关工作原理！

## The Skip-gram Model

以下面的数据集为例

`the quick brown fox jumped over the lazy dog`

首先形成一个数据集，其中包含字词以及字词在其中出现的上下文。我们可以通过任何有意义的方式定义“上下文”，事实上人们研究了语法上下文（即当前目标字词的语法依赖项，具体示例请参阅
[Levy et al.](https://levyomer.files.wordpress.com/2014/04/dependency-based-word-embeddings-acl-2014.pdf)
）、目标左侧的字词、目标右侧的字词等。暂时我们使用 vanilla 定义，将“上下文”定义为目标字词左侧和右侧的字词窗口。使用大小为 1 的窗口，我们将获得以下数据集

`([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...`

其中包含多组 `(context, target)` 对。回想一下，`skip-gram` 会调换上下文和目标，并尝试根据其目标字词预测每个上下文字词。因此，任务变成根据“quick”预测“the”和“brown”、根据“brown”预测“quick”和“fox”，等等。这样一来，我们的数据集就变成了

`(quick, the), (quick, brown), (brown, quick), (brown, fox), ...`

其中包含多组 `(input, output)` 对。目标函数基于整个数据集进行定义，但我们通常使用
[stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
 (SGD) 进行优化，并且一次使用一个样本（或大小为 `batch_size` 的小批次样本，通常 `16 <= batch_size <= 512`）。我们来看看这个过程的一个时间步。

Let's imagine at training step \\(t\\) we observe the first training case above,
where the goal is to predict `the` from `quick`. We select `num_noise` number
of noisy (contrastive) examples by drawing from some noise distribution,
typically the unigram distribution, \\(P(w)\\). For simplicity let's say
`num_noise=1` and we select `sheep` as a noisy example. Next we compute the
loss for this pair of observed and noisy examples, i.e. the objective at time
step \\(t\\) becomes

$$J^{(t)}_\text{NEG} = \log Q_\theta(D=1 | \text{the, quick}) +
  \log(Q_\theta(D=0 | \text{sheep, quick}))$$

The goal is to make an update to the embedding parameters \\(\theta\\) to improve
(in this case, maximize) this objective function.  We do this by deriving the
gradient of the loss with respect to the embedding parameters \\(\theta\\), i.e.
\\(\frac{\partial}{\partial \theta} J_\text{NEG}\\) (luckily TensorFlow provides
easy helper functions for doing this!). We then perform an update to the
embeddings by taking a small step in the direction of the gradient. When this
process is repeated over the entire training set, this has the effect of
'moving' the embedding vectors around for each word until the model is
successful at discriminating real words from noise words.

We can visualize the learned vectors by projecting them down to 2 dimensions
using for instance something like the
[t-SNE dimensionality reduction technique](https://lvdmaaten.github.io/tsne/).
When we inspect these visualizations it becomes apparent that the vectors
capture some general, and in fact quite useful, semantic information about
words and their relationships to one another. It was very interesting when we
first discovered that certain directions in the induced vector space specialize
towards certain semantic relationships, e.g. *male-female*, *verb tense* and
even *country-capital* relationships between words, as illustrated in the figure
below (see also for example
[Mikolov et al., 2013](https://www.aclweb.org/anthology/N13-1090)).

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/linear-relationships.png" alt>
</div>

这就解释了为什么这些向量也可用作很多规范 NLP 预测任务（例如词性标注或命名实体识别）的特征（示例请参阅
[Collobert 等人在 2011 年](https://arxiv.org/abs/1103.0398)
发表的原始论文 (
[pdf](https://arxiv.org/pdf/1103.0398.pdf)
)，或者
[Turian 等人在 2010 年](https://www.aclweb.org/anthology/P10-1040)发表的后续论文）。

暂时我们先用它们绘制漂亮的图片吧！

## 构建图

图主要与嵌入相关，因此我们先定义嵌入矩阵。它其实就是一个大型随机矩阵。我们将初始化值，使其在单位立方体中保持统一。

```python
embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
```

噪声对比估计损失是基于逻辑回归模型进行定义的。为此，我们需要为词汇表中的每个字词定义权重和偏差（也称为 `output weights`，与 `input embeddings` 相对）。我们先进行定义。

```python
nce_weights = tf.Variable(
  tf.truncated_normal([vocabulary_size, embedding_size],
                      stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
```

现在参数已设置完毕，我们可以定义 skip-gram 模型图了。为简单起见，假设我们已将文本语料库与词汇表进行整合，以便每个字词都表示为一个整数（如需了解详情，请访问
[tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)
）。skip-gram 模型有两个输入。一个是表示源上下文字词的整数批次，另一个是表示目标字词的整数批次。下面我们为这些输入创建占位符节点，以便之后馈送这些数据。

```python
# Placeholders for inputs
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
```

现在，我们需要做的是查询批次中每个源字词的向量。TensorFlow 提供的便利辅助函数可简化这一操作。

```python
embed = tf.nn.embedding_lookup(embeddings, train_inputs)
```

现在，每个字词均有嵌入，我们希望尝试使用噪声对比训练目标来预测目标字词。

```python
# Compute the NCE loss, using a sample of the negative labels each time.
loss = tf.reduce_mean(
  tf.nn.nce_loss(weights=nce_weights,
                 biases=nce_biases,
                 labels=train_labels,
                 inputs=embed,
                 num_sampled=num_sampled,
                 num_classes=vocabulary_size))
```

现在我们有一个损失节点，下面需要添加计算梯度并更新参数等所需的节点。为此，我们将使用随机梯度下降法，而 TensorFlow 提供的便利辅助函数也可简化这一操作。

```python
# We use the SGD optimizer.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
```

## 训练模型

训练模型再简单不过了，只需使用 `feed_dict` 将数据推入占位符并循环地使用此新数据调用 `tf.Session.run` 即可。

```python
for inputs, labels in generate_batch(...):
  feed_dict = {train_inputs: inputs, train_labels: labels}
  _, cur_loss = session.run([optimizer, loss], feed_dict=feed_dict)
```

如需查看完整的示例代码，请访问
[tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)。

## 可视化学到的嵌入

训练结束后，我们可以使用 t-SNE 可视化学到的嵌入。

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/tsne.png" alt>
</div>

Et voila! As expected, words that are similar end up clustering nearby each
other. For a more heavyweight implementation of word2vec that showcases more of
the advanced features of TensorFlow, see the implementation in
[models/tutorials/embedding/word2vec.py](https://github.com/tensorflow/models/tree/master/tutorials/embedding/word2vec.py).

## 评估嵌入：类比推理

嵌入对于 NLP 中的各种预测任务来说非常有用。除了训练一个成熟的词性模型或命名实体模型之外，评估嵌入的一种简单方法是直接使用它们预测语法和语义关系（如 `king is to queen as father is to ?`）。这种方法称为类比推理，
[Mikolov 及其同事](https://www.aclweb.org/anthology/N13-1090).
介绍了这项任务。请从
[download.tensorflow.org](http://download.tensorflow.org/data/questions-words.txt)
下载此任务的数据集。

如需了解我们如何进行此评估，请参阅
[models/tutorials/embedding/word2vec.py](https://github.com/tensorflow/models/tree/master/tutorials/embedding/word2vec.py)
中的 `build_eval_graph()` 和 `eval()` 函数。

超参数的选择可极大影响此任务的准确率。要在此任务中实现领先的性能，我们需要用非常大型的数据集进行训练、仔细调整超参数并使用诸如对数据进行下采样等技巧，这些内容不在本教程的探讨范围内。


## 优化实现

我们的 vanilla 实现展示了 TensorFlow 的灵活性。例如，更改训练目标很简单，直接将对 `tf.nn.nce_loss()` 的调用替换成 `tf.nn.sampled_softmax_loss()` 等现成备用方案即可。如果您对损失函数有新的想法，可以在 TensorFlow 中为新目标手动编写表达式，并让优化器计算其导数。这种灵活性在机器学习模型开发的探索阶段非常宝贵，在这一阶段，我们会尝试几种不同的想法并快速迭代。

在您对模型结构感到满意后，可能有必要优化您的实现，以提高运行效率，并在更短的时间内涵盖更多数据。例如，我们在本教程中使用的简单代码在速度上会受限，因为我们使用 Python 读取和馈送数据项（在 TensorFlow 后端上，每项操作需要进行的工作都非常少）。如果您发现模型在输入数据方面存在严重瓶颈，您可能需要针对您的问题实现自定义数据读取器，如
[新数据格式](/docs/tensorflow/extend/new_data_formats)
中所述。至于 Skip-Gram 建模，我们实际上已经在
[models/tutorials/embedding/word2vec.py](https://github.com/tensorflow/models/tree/master/tutorials/embedding/word2vec.py) 中提供示例。

如果您的模型不再受 I/O 限制，但您仍希望提高性能，则可以通过编写自己的 TensorFlow 操作（如
[添加新操作](/docs/tensorflow/extend/adding_an_op)
中所述）进一步采取措施。同样，我们已在
[models/tutorials/embedding/word2vec_optimized.py](https://github.com/tensorflow/models/tree/master/tutorials/embedding/word2vec_optimized.py)
中提供 Skip-Gram 示例。欢迎对它们相互进行基准测试，以衡量它们在各个阶段的性能改善情况。

## 总结

在本教程中，我们介绍了 word2vec 模型，这是一种计算效率很高的模型，用于学习字词嵌入。我们提出了为何嵌入非常有用，讨论了有效的训练技巧，并展示了如何在 TensorFlow 中实现所有这些操作。总而言之，我们希望通过本教程传达以下信息：TensorFlow 可以为您提供早期实验所需的灵活性，以及后期自定义优化实现所需的控制力。
