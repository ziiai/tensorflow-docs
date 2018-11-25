#  使用 TensorFlow 创建大型线性模型

`tf.estimator` 提供了大量用于在 TensorFlow 中处理线性模型的工具（以及其他工具）。本文简要介绍了这些工具，并阐述了以下内容：

- 什么是线性模型。
- 为何要使用线性模型。
- Estimator 如何使您能够轻松地在 TensorFlow 中构建线性模型。
- 如何使用 Estimator 将线性模型与深度学习相结合，从而汲取二者的优势。

您可以阅读这篇概述文章，判断 Estimator 的线性模型工具是否对您有用。然后，阅读
[Estimator wide and deep learning tutorial](https://github.com/tensorflow/models/tree/master/official/wide_deep)
，并放手一试。这篇概述文章使用了此教程中的代码示例，但此教程更详细地介绍了代码。

对基本的机器学习概念以及
[Estimators](/docs/tensorflow/guide/premade_estimators)
有一定了解将有助于理解这篇概述文章。


## 什么是线性模型？

**线性模型**使用特征的单个加权和进行预测。举例来说，如果您有关于人口年龄、受教育年数和每周工作时长的[数据](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names)
，那么模型可以学习每个数值的权重，并通过加权和估算某个人的薪水。您还可以使用线性模型进行分类。

一些线性模型会将加权和转换为一种更便利的形式。例如， [**逻辑回归**](https://developers.google.com/machine-learning/glossary/#logistic_regression)
将加权和代入逻辑函数，以将输出转换为介于 0 和 1 之间的值。不过，每个输入特征仍然只有一个权重。

## 为何要使用线性模型？

近期研究已证实具有多层的更复杂神经网络具有强大的功能，为何还要使用如此简单的模型？

线性模型：

- 与深度神经网络相比，训练速度快。
- 可以在非常大的特征集上取得很好的效果。
- 可以使用无需反复调整学习速率等因素的算法进行训练。
- 可以比神经网络更轻松地进行解读和调试。您可以检查分配给每个特征的权重，确定哪些特征对预测结果的影响最大。
- 为理解机器学习提供了一个很好的起点。
- 有广泛的行业应用。


## Estimator 如何帮助您构建线性模型？

您可以在 TensorFlow 中从头开始构建线性模型，而无需借助于特殊的 API。不过，Estimator 提供了一些工具，使您可以更轻松地构建有效的大型线性模型。


### 特征列和转换

设计线性模型的主要操作包括将原始数据转换为合适的输入特征。Tensorflow 使用 `FeatureColumn` 抽象类来实现此类转换。

`FeatureColumn` 表示数据中的单个特征。`FeatureColumn` 可能表示“height”等数量，也可能表示“eye_color”等类别（值来自一组离散概率，如 {'blue'，'brown'，'green'}）。

对于“height”等连续特征和“eye_color”等类别特征，数据中的单个值可能先转换为数字序列，然后再输入到模型中。尽管如此，您还是可以通过 `FeatureColumn` 抽象类将该特征视为单个语义单元进行操作。您可以指定转换并选择要包括的特征，而无需处理馈送到模型的张量中的特定索引。

#### 稀疏列

线性模型中的类别特征通常被转换为稀疏向量，其中每个可能的值都具有对应的索引或 ID。例如，如果只有 3 种可能的眼睛颜色，您可以将“eye_color”表示为长度为 3 的向量：“brown”为 [1, 0, 0]，“blue”为 [0, 1, 0]，而“green”为 [0, 0, 1]。这些向量称为“稀疏”向量，因为当可能值的集合非常大（例如包含所有英文单词的集合）时，它们可能很长，且包含很多零。

虽然您不需要通过类别列来使用 Estimator 提供的线性模型工具，但是线性模型的优势之一是它们能够处理大型稀疏向量。稀疏特征是 Estimator 提供的线性模型工具的主要用例。

##### 编码稀疏列

`FeatureColumn` 自动将类别值转换为向量，具体代码如下所示：

```python
eye_color = tf.feature_column.categorical_column_with_vocabulary_list(
    "eye_color", vocabulary_list=["blue", "brown", "green"])
```

其中 `eye_color` 是源数据中的一列的名称。

您还可以为类别特征（您不知道此类特征的所有可能值）生成 `FeatureColumn`。对于这种情况，您将使用 `categorical_column_with_hash_bucket()`，它使用哈希函数为特征值分配索引。


```python
education = tf.feature_column.categorical_column_with_hash_bucket(
    "education", hash_bucket_size=1000)
```

##### 特征组合

由于线性模型为不同的特征分配独立权重，因此它们无法了解特定特征值组合的相对重要性。如果您有“favorite_sport”和“home_city”这两个特征，并且您尝试预测某人是否喜欢穿红色，此时线性模型将无法判断来自圣路易斯的棒球迷是否特别喜欢穿红色。

您可以通过创建新特征“favorite_sport_x_home_city”突破这个限制。对于给定用户，此特征的值只是两个源特征的值相连：例如“baseball_x_stlouis”。这种组合特征称为特征组合。

使用 `crossed_column()` 方法可轻松设置特征组合：

```python
sport_x_city = tf.feature_column.crossed_column(
    ["sport", "city"], hash_bucket_size=int(1e4))
```

#### 连续列

您可以如下所示地指定连续特征：

```python
age = tf.feature_column.numeric_column("age")
```

虽然作为单个实数的连续特征通常可以直接输入到模型中，但是 Tensorflow 也为此类列提供了有用的转换。

##### 分桶

分桶可将连续列转换为类别列。此转换使您能够在特征组合中使用连续特征，或学习特定值范围特别重要的情况。

分桶将可能的值范围划分为子范围（称为分桶）：

```python
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```

值所在的分桶便是该值的类别标签。


#### 输入函数

FeatureColumn 为模型提供输入数据规范，指示如何表示和转换数据。但它们本身不提供数据。您需要通过输入函数提供数据。

输入函数必须返回一个张量字典。每个键对应一个 `FeatureColumn` 的名称。每个键的值都是一个张量，其中包含该特征针对所有数据实例的值。请参阅
[预创建的 Estimator](/docs/tensorflow/guide/premade_estimators#input_fn)，详细了解输入函数；另请参阅
[宽度学习和深度学习教程](https://github.com/tensorflow/models/tree/master/official/wide_deep)
中的 `input_fn`，查看输入函数的示例实现。

输入函数会传递给 `train()` 和 `evaluate()` 调用（用于启动训练和测试操作），如下一部分中所述。

### 线性 Estimator

Tensorflow Estimator 类为回归模型和分类模型提供统一的训练和评估工具。它们负责训练和评估循环的细节部分，使用户可以专注于模型输入和架构。

要构建线性 Estimator，您可以使用 `tf.estimator.LinearClassifier Estimator` 或 `tf.estimator.LinearRegressor Estimator`（分别用于分类和回归）。

与所有 TensorFlow Estimator 一样，要运行 Estimator，只需执行以下操作即可：

- 实例化 Estimator 类。对于两个线性 Estimator 类，将 `FeatureColumn` 列表传递给构造函数。
- 调用 Estimator 的 `train()` 方法以对其进行训练。
- 调用 Estimator 的 `evaluate()` 方法以查看其效果。

例如：


```python
e = tf.estimator.LinearClassifier(
    feature_columns=[
        native_country, education, occupation, workclass, marital_status,
        race, age_buckets, education_x_occupation,
        age_buckets_x_race_x_occupation],
    model_dir=YOUR_MODEL_DIRECTORY)
e.train(input_fn=input_fn_train, steps=200)
# Evaluate for one step (one pass through the test data).
results = e.evaluate(input_fn=input_fn_test)

# Print the stats for the evaluation.
for key in sorted(results):
    print("%s: %s" % (key, results[key]))
```

### 宽度学习和深度学习

`tf.estimator` 模块还提供一个 Estimator 类，使您能够一同训练线性模型和深度神经网络。这种新方法结合了线性模型“记忆”关键特征的能力以及神经网络的泛化能力。请使用 `tf.estimator.DNNLinearCombinedClassifier` 创建这种“宽度学习和深度学习”模型：

```python
e = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=YOUR_MODEL_DIR,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])
```

如需了解详情，请参阅
[宽度学习和深度学习教程](https://github.com/tensorflow/models/tree/master/official/wide_deep)。
