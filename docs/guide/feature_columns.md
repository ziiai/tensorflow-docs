#  特征列

本文档详细介绍了特征列。您可以将特征列视为原始数据和 Estimator 之间的媒介。特征列内容丰富，使您可以将各种原始数据转换为 Estimator 可以使用的格式，从而可以轻松地进行实验。

在 [预创建的 Estimator](/docs/tensorflow/guide/premade_estimators)中，我们使用预创建的 Estimator (`DNNClassifier`) 训练模型，根据四个输入特征预测不同类型的鸢尾花。该示例仅创建了数值特征列（类型为 `tf.feature_column.numeric_column`）。虽然数值特征列有效地对花瓣和花萼的长度进行了建模，但真实的数据集包含各种各样的特征，其中很多特征并非数值。

<div style="width:80%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/feature_columns/feature_cloud.jpg">
</div>
<div style="text-align: center">
 一些真实特征（如经度）为数值，但很多并不是数值。
</div>

## 深度神经网络的输入

深度神经网络可以处理哪类数据？答案当然是数字（例如 `tf.float32`）。毕竟，神经网络中的每个神经元都会对权重和输入数据执行乘法和加法运算。不过，实际输入数据通常包含非数值（分类）数据。以一个可包含下列三个非数值的 `product_class` 特征为例：

* `kitchenware`
* `electronics`
* `sports`

机器学习模型一般将分类值表示为简单的矢量，其中 1 表示存在某个值，0 表示不存在某个值。例如，如果将 `product_class` 设置为 `sports`，机器学习模型通常将 `product_class` 表示为 `[0, 0, 1]`，意即：

* `0`：`kitchenware`不存在
* `0`：`electronics`不存在
* `1`：`sports`不存在

因此，虽然原始数据可以是数值或分类值，但机器学习模型会将所有特征表示为数字。

## 特征列

如下图所示，您可以通过 Estimator（鸢尾花的 `DNNClassifier`）的 `feature_columns` 参数指定模型的输入。特征列作为输入数据（由 `input_fn` 返回）与模型之间的桥梁。

<div style="width:80%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/feature_columns/inputs_to_model_bridge.jpg">
</div>
<div style="text-align: center">
特征列作为原始数据与模型所需的数据之间的桥梁。 
</div>

要创建特征列，请调用 tf.feature_column 模块的函数。本文档介绍了该模块中的九个函数。如下图所示，所有九个函数都会返回一个 `Categorical-Column` 或一个 `Dense-Column` 对象，但却不会返回 `bucketized_column`，后者继承自这两个类：

<div style="width:80%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/feature_columns/some_constructors.jpg">
</div>
<div style="text-align: center">
特征列方法分为两个主要类别和一个混合类别。 
</div>

我们来更详细地了解这些函数。

### 数值列

鸢尾花分类器会针对所有输入特征调用 `tf.feature_column.numeric_column` 函数：

  * `SepalLength`
  * `SepalWidth`
  * `PetalLength`
  * `PetalWidth`

虽然 `tf.numeric_column` 提供可选参数，但也可以在没有任何参数的情况下调用 `tf.numeric_column`（如下所示），这是一种将具有默认数据类型 (`tf.float32`) 的数值指定为模型输入的不错方式：

```python
# Defaults to a tf.float32 scalar.
numeric_feature_column = tf.feature_column.numeric_column(key="SepalLength")
```

要指定一个非默认的数值数据类型，请使用 `dtype` 参数。例如：

``` python
# Represent a tf.float64 scalar.
numeric_feature_column = tf.feature_column.numeric_column(key="SepalLength",
                                                          dtype=tf.float64)
```

默认情况下，数值列会创建单个值（标量）。使用 `shape` 参数指定另一种形状。例如：

<!--TODO(markdaoust) link to full example-->
```python
# Represent a 10-element vector in which each cell contains a tf.float32.
vector_feature_column = tf.feature_column.numeric_column(key="Bowling",
                                                         shape=10)

# Represent a 10x5 matrix in which each cell contains a tf.float32.
matrix_feature_column = tf.feature_column.numeric_column(key="MyMatrix",
                                                         shape=[10,5])
```
### 分桶列

通常，您不会直接向模型馈送数字，相反，您会根据数值范围将其值分为不同的类别。为此，请创建一个分桶列。以表示房屋建造年份的原始数据为例。我们并非以标量数值列表示年份，而是将年份分成下列四个分桶：

<div style="width:80%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/feature_columns/bucketized_column.jpg">
</div>
<div style="text-align: center">
 将年份数据分成四个分桶。 
</div>

模型将按以下方式表示这些分桶：

|日期范围 |表示为… |
|:----------|:-----------------|
|< 1960               | [1, 0, 0, 0] |
|>= 1960 但 < 1980   | [0, 1, 0, 0] |
|>= 1980 但 < 2000   | [0, 0, 1, 0] |
|>= 2000              | [0, 0, 0, 1] |

为什么要将数字（一个完全有效的模型输入）拆分为分类值？请注意，该分类将单个输入数字分成了一个四元素矢量。因此，模型现在可以学习四个单独的权重，而非仅仅一个；相比一个权重，四个权重能够创建一个内容更丰富的模型。更重要的是，借助分桶，模型能够清楚地区分不同年份类别，因为仅设置了一个元素 (1)，其他三个元素则被清除 (0)。例如，当我们仅将单个数字（年份）用作输入时，线性模型只能学习线性关系。因此，分桶为模型提供了可用于学习的额外灵活性。

以下代码演示了如何创建分桶特征：

<!--TODO(markdaoust) link to full example - housing price grid?-->
```python
# First, convert the raw input to a numeric column.
numeric_feature_column = tf.feature_column.numeric_column("Year")

# Then, bucketize the numeric column on the years 1960, 1980, and 2000.
bucketized_feature_column = tf.feature_column.bucketized_column(
    source_column = numeric_feature_column,
    boundaries = [1960, 1980, 2000])
```
请注意，指定一个三元素边界矢量可创建一个四元素分桶矢量。


### 分类标识列

可将分类标识列视为分桶列的一种特殊情况。在传统的分桶列中，每个分桶表示一系列值（例如，从 1960 年到 1979 年）。在分类标识列中，每个分桶表示一个唯一整数。例如，假设您想要表示整数范围 `[0, 4)`。也就是说，您想要表示整数 0、1、2 或 3。在这种情况下，分类标识映射如下所示：

<div style="width:80%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/feature_columns/categorical_column_with_identity.jpg">
</div>
<div style="text-align: center">
分类标识列映射。请注意，这是一种独热编码，而非二元数值编码。 
</div>

与分桶列一样，模型可以在分类标识列中学习每个类别各自的权重。例如，我们使用唯一的整数值来表示每个类别，而不是使用某个字符串来表示 `product_class`。即：

* `0="kitchenware"`
* `1="electronics"`
* `2="sport"`

调用 `tf.feature_column.categorical_column_with_identity` 以实现类别标识列。例如：

``` python
# Create categorical output for an integer feature named "my_feature_b",
# The values of my_feature_b must be >= 0 and < num_buckets
identity_feature_column = tf.feature_column.categorical_column_with_identity(
    key='my_feature_b',
    num_buckets=4) # Values [0, 4)

# In order for the preceding call to work, the input_fn() must return
# a dictionary containing 'my_feature_b' as a key. Furthermore, the values
# assigned to 'my_feature_b' must belong to the set [0, 4).
def input_fn():
    ...
    return ({ 'my_feature_a':[7, 9, 5, 2], 'my_feature_b':[3, 1, 2, 2] },
            [Label_values])
```

### 分类词汇列

我们不能直接向模型中输入字符串。相反，我们必须首先将字符串映射到数值或分类值。分类词汇列提供了一种将字符串表示为独热矢量的好方法。例如：

<div style="width:80%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/feature_columns/categorical_column_with_vocabulary.jpg">
</div>
<div style="text-align: center">
将字符串值映射到词汇列。 
</div>

我们可以看出，分类词汇列就像是分类标识列的枚举版本。TensorFlow 提供了两种不同的函数来创建分类词汇列：

* `tf.feature_column.categorical_column_with_vocabulary_list`
* `tf.feature_column.categorical_column_with_vocabulary_file`

`categorical_column_with_vocabulary_list` 根据明确的词汇表将每个字符串映射到一个整数。例如：

```python
# Given input "feature_name_from_input_fn" which is a string,
# create a categorical feature by mapping the input to one of
# the elements in the vocabulary list.
vocabulary_feature_column =
    tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature_name_from_input_fn,
        vocabulary_list=["kitchenware", "electronics", "sports"])
```

上面的函数非常简单，但它有一个明显的缺点。那就是，当词汇表很长时，需要输入的内容太多了。对于此类情况，请改为调用 `tf.feature_column.categorical_column_with_vocabulary_file`，以便将词汇放在单独的文件中。例如：

```python

# Given input "feature_name_from_input_fn" which is a string,
# create a categorical feature to our model by mapping the input to one of
# the elements in the vocabulary file
vocabulary_feature_column =
    tf.feature_column.categorical_column_with_vocabulary_file(
        key=feature_name_from_input_fn,
        vocabulary_file="product_class.txt",
        vocabulary_size=3)
```

`product_class.txt` 中的每个词汇元素应各占一行。在我们的示例中：

```None
kitchenware
electronics
sports
```

### 经过哈希处理的列

到目前为止，我们处理的示例都包含很少的类别。例如，我们的 `product_class` 示例只有 3 个类别。但是通常，类别的数量非常大，以至于无法为每个词汇或整数设置单独的类别，因为这会消耗太多内存。对于此类情况，我们可以反问自己：“我愿意为我的输入设置多少类别？”实际上，`tf.feature_column.categorical_column_with_hash_bucket` 函数使您能够指定类别的数量。对于这种类型的特征列，模型会计算输入的哈希值，然后使用模运算符将其置于其中一个 `hash_bucket_size` 类别中，如以下伪代码所示：

```python
# pseudocode
feature_id = hash(raw_feature) % hash_bucket_size
```

创建 `feature_column` 的代码可能如下所示：

``` python
hashed_feature_column =
    tf.feature_column.categorical_column_with_hash_bucket(
        key = "some_feature",
        hash_bucket_size = 100) # The number of categories
```
此时，您可能会认为：“这太疯狂了！”，这种想法很正常。毕竟，我们是将不同的输入值强制划分成更少数量的类别。这意味着，两个可能不相关的输入会被映射到同一个类别，这样一来，神经网络也会面临同样的结果。下图显示了这一困境，即厨具和运动用品都被分配到类别（哈希分桶）12：

<div style="width:80%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/feature_columns/hashed_column.jpg">
</div>
<div style="text-align: center">
 用哈希分桶表示数据。 
</div>

与机器学习中的很多反直觉现象一样，事实证明哈希技术经常非常有用。这是因为哈希类别为模型提供了一些分隔方式。模型可以使用其他特征进一步将厨具与运动用品分隔开来。

### 组合列

通过将多个特征组合为一个特征（称为特征组合），模型可学习每个特征组合的单独权重。

更具体地说，假设我们希望模型计算佐治亚州亚特兰大的房产价格。这个城市的房产价格在不同位置差异很大。在确定对房产位置的依赖性方面，将纬度和经度表示为单独的特征用处不大；但是，将纬度和经度组合为一个特征则可精确定位位置。假设我们将亚特兰大表示为一个 100x100 的矩形网格区块，按纬度和经度的特征组合标识全部 10000 个区块。借助这种特征组合，模型可以针对与各个区块相关的房价条件进行训练，这比单独的经纬度信号强得多。

下图展示了我们的计划（以红色文本显示城市各角落的纬度和经度值）：

<div style="width:80%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/feature_columns/Atlanta.jpg">
</div>
<div style="text-align: center">
 亚特兰大地图。想象一下，这张地图被分成 10000 个大小相同的区块。
</div>

为了解决此问题，我们同时使用了 `tf.feature_column.crossed_column` 函数及先前介绍的 `bucketized_column`。

<!--TODO(markdaoust) link to full example-->

``` python
def make_dataset(latitude, longitude, labels):
    assert latitude.shape == longitude.shape == labels.shape

    features = {'latitude': latitude.flatten(),
                'longitude': longitude.flatten()}
    labels=labels.flatten()

    return tf.data.Dataset.from_tensor_slices((features, labels))


# Bucketize the latitude and longitude using the `edges`
latitude_bucket_fc = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('latitude'),
    list(atlanta.latitude.edges))

longitude_bucket_fc = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('longitude'),
    list(atlanta.longitude.edges))

# Cross the bucketized columns, using 5000 hash bins.
crossed_lat_lon_fc = tf.feature_column.crossed_column(
    [latitude_bucket_fc, longitude_bucket_fc], 5000)

fc = [
    latitude_bucket_fc,
    longitude_bucket_fc,
    crossed_lat_lon_fc]

# Build and train the Estimator.
est = tf.estimator.LinearRegressor(fc, ...)
```

您可以根据下列任意内容创建一个特征组合：

- 特征名称；也就是 `input_fn` 返回的 `dict` 中的名称。
- 任意分类列，`categorical_column_with_hash_bucket` 除外（因为 `crossed_column` 会对输入进行哈希处理）。

当特征列 `latitude_bucket_fc` 和 `longitude_bucket_fc` 组合时，TensorFlow 会为每个样本创建 `(latitude_fc, longitude_fc)` 对。这会生成完整的概率网格，如下所示：

``` None
 (0,0),  (0,1)...  (0,99)
 (1,0),  (1,1)...  (1,99)
   ...     ...       ...
(99,0), (99,1)...(99, 99)
```

不同之处在于，完整的网格仅对词汇有限的输入而言可追踪。`crossed_column` 仅构建 `hash_bucket_size` 参数所请求的数字，而不是构建这个可能非常庞大的输入表。特征列通过在输入元组上运行哈希函数，然后使用 `hash_bucket_size` 进行模运算，为索引分配一个样本。

如前所述，执行哈希函数和模函数会限制类别的数量，但会导致类别冲突；也就是说，多个（纬度、经度）特征组合最终位于同一个哈希分桶中。但实际上，执行特征组合对于模型的学习能力仍具备重大价值。

有些反直觉的是，在创建特征组合时，通常仍应在模型中包含原始（未组合）特征（如前面的代码段中所示）。独立的纬度和经度特征有助于模型区分组合特征中发生哈希冲突的样本。

## 指标列和嵌入列

指标列和嵌入列从不直接处理特征，而是将分类列视为输入。

使用指标列时，我们指示 TensorFlow 完成我们在分类 `product_class` 样本中看到的确切操作。也就是说，指标列将每个类别视为独热矢量中的一个元素，其中匹配类别的值为 1，其余类别为 0：

<div style="width:80%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/feature_columns/categorical_column_with_identity.jpg">
</div>
<div style="text-align: center">
用指标列表示数据。
</div>

以下是通过调用 `tf.feature_column.indicator_column` 创建指标列的方法：

``` python
categorical_column = ... # Create any type of categorical column.

# Represent the categorical column as an indicator column.
indicator_column = tf.feature_column.indicator_column(categorical_column)
```

现在，假设我们有一百万个可能的类别，或者可能有十亿个，而不是只有三个。出于多种原因，随着类别数量的增加，使用指标列来训练神经网络变得不可行。

我们可以使用嵌入列来克服这一限制。嵌入列并非将数据表示为很多维度的独热矢量，而是将数据表示为低维度普通矢量，其中每个单元格可以包含任意数字，而不仅仅是 0 或 1。通过使每个单元格能够包含更丰富的数字，嵌入列包含的单元格数量远远少于指标列。

我们来看一个将指标列和嵌入列进行比较的示例。假设我们的输入样本包含多个不同的字词（取自仅有 81 个字词的有限词汇表）。我们进一步假设数据集在 4 个不同的样本中提供了下列输入字词：

* `"dog"`
* `"spoon"`
* `"scissors"`
* `"guitar"`

在这种情况下，下图说明了嵌入列或指标列的处理流程。

<div style="width:80%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://raw.githubusercontent.com/ziiai/tensorflow-docs/master/images/feature_columns/embedding_vs_indicator.jpg">
</div>
<div style="text-align: center">
嵌入列将分类数据存储在低于指标列的低维度矢量中。（我们只是将随机数字放入嵌入矢量中；由训练决定实际数字。） 
</div>

处理样本时，其中一个 `categorical_column_with...` 函数会将样本字符串映射到分类数值。例如，一个函数将“spoon”映射到 `[32]`。（32 是我们想象出来的，实际值取决于映射函数。）然后，您可以通过下列两种方式之一表示这些分类数值：

- 作为指标列。函数将每个分类数值转换为一个 81 元素的矢量（因为我们的词汇表由 81 个字词组成），将 1 置于分类值 (0, 32, 79, 80) 的索引处，将 0 置于所有其他位置。

- 作为嵌入列。函数将分类数值 `(0, 32, 79, 80)` 用作对照表的索引。该对照表中的每个槽位都包含一个 3 元素矢量。

嵌入矢量中的值如何神奇地得到分配？实际上，分配值在训练期间进行。也就是说，模型学习了将输入分类数值映射到嵌入矢量值以解决问题的最佳方法。嵌入列可以增强模型的功能，因为嵌入矢量从训练数据中学习了类别之间的新关系。

为什么示例中的嵌入矢量大小为 3？下面的“公式”提供了关于嵌入维度数量的一般经验法则：

```python
embedding_dimensions =  number_of_categories**0.25
```

也就是说，嵌入矢量维数应该是类别数量的 4 次方根。由于本示例中的词汇量为 81，建议维数为 3：

``` python
3 =  81**0.25
```
请注意，这只是一个一般规则；您可以根据需要设置嵌入维度的数量。

调用 `tf.feature_column.embedding_column` 来创建一个 `embedding_column`，如以下代码段所示：

``` python
categorical_column = ... # Create any categorical column

# Represent the categorical column as an embedding column.
# This means creating an embedding vector lookup table with one element for each category.
embedding_column = tf.feature_column.embedding_column(
    categorical_column=categorical_column,
    dimension=embedding_dimensions)
```

[嵌入](/docs/tensorflow/guide/embedding)是机器学习中的一个重要概念。这些信息仅仅是帮助您将其用作特征列的入门信息。

## 将特征列传递给 Estimator

如下面的列表所示，并非所有 Estimator 都支持所有类型的 `feature_columns` 参数：

* `tf.estimator.LinearClassifier` 和
  `tf.estimator.LinearRegressor`：接受所有类型的特征列。
* `tf.estimator.DNNClassifier` 和
  `tf.estimator.DNNRegressor`：只接受密集列。其他类型的列必须封装在 `indicator_column` 或 `embedding_column` 中。
* `tf.estimator.DNNLinearCombinedClassifier` 和
  `tf.estimator.DNNLinearCombinedRegressor`:
    * `linear_feature_columns` 参数接受任何类型的特征列。
    * `dnn_feature_columns` 参数只接受密集列。

## 其他资料

有关特征列的更多示例，请查看以下内容：

* [低阶 API 简介](/docs/tensorflow/guide/low_level_intro#feature_columns)展示了如何使用 TensorFlow 的低阶 API 直接尝试 `feature_columns`。
* [Estimator 宽度与深度学习教程](https://github.com/tensorflow/models/tree/master/official/wide_deep)针对各种输入数据类型使用 `feature_columns` 解决了二元分类问题。

要详细了解嵌入，请查看以下内容：

* [深度学习、NLP 和表示法](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)（Chris Olah 的博客）
* TensorFlow [Embedding Projector](http://projector.tensorflow.org)
