#  张量

正如名称所示，TensorFlow 这一框架定义和运行涉及张量的计算。张量是对矢量和矩阵向潜在的更高维度的泛化。TensorFlow 在内部将张量表示为基本数据类型的 n 维数组。

在编写 TensorFlow 程序时，您操作和传递的主要对象是 `tf.Tensor`。`tf.Tensor` 对象表示一个部分定义的计算，最终会生成一个值。TensorFlow 程序首先会构建一个 `tf.Tensor` 对象图，详细说明如何基于其他可用张量计算每个张量，然后运行该图的某些部分以获得期望的结果。

`tf.Tensor` 具有以下属性：

- 数据类型（例如 `float32`、`int32` 或 `string`）
- 形状

张量中的每个元素都具有相同的数据类型，且该数据类型一定是已知的。形状，即张量的维数和每个维度的大小，可能只有部分已知。如果其输入的形状也完全已知，则大多数操作会生成形状完全已知的张量，但在某些情况下，只能在执行图时获得张量的形状。

某些类型的张量有点特殊，TensorFlow 指南的其他部分有所介绍。以下是主要特殊张量：

- `tf.Variable`
- `tf.constant`
- `tf.placeholder`
- `tf.SparseTensor`

除了 `tf.Variable` 以外，张量的值是不变的，这意味着对于单个执行任务，张量只有一个值。然而，两次评估同一张量可能会返回不同的值；例如，该张量可能是从磁盘读取数据的结果，或是生成随机数的结果。

## 阶

`tf.Tensor` 对象的阶是它本身的维数。阶的同义词包括：**秩**、**等级**或 **n 维**。请注意，TensorFlow 中的阶与数学中矩阵的阶并不是同一个概念。如下表所示，TensorFlow 中的每个阶都对应一个不同的数学实例：

阶 | 数学实例
--- | ---
0 | 标量（只有大小）
1 | 矢量（大小和方向）
2 | 矩阵（数据表）
3 | 3 阶张量（数据立体）
n | n 阶张量（自行想象）


### 0 阶

以下摘要演示了创建 0 阶变量的过程：

```python
mammal = tf.Variable("Elephant", tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)
```

> 注意：字符串在 TensorFlow 中被视为单一项，而不是一连串字符。TensorFlow 可以有标量字符串，字符串矢量，等等。

### 1 阶

要创建 1 阶 `tf.Tensor` 对象，您可以传递一个项目列表作为初始值。例如：

```python
mystr = tf.Variable(["Hello"], tf.string)
cool_numbers  = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)
```


### 更高阶

2 阶 `tf.Tensor` 对象至少包含一行和一列：

```python
mymat = tf.Variable([[7],[11]], tf.int16)
myxor = tf.Variable([[False, True],[True, False]], tf.bool)
linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
mymatC = tf.Variable([[7],[11]], tf.int32)
```

同样，更高阶的张量由一个 n 维数组组成。例如，在图像处理过程中，会使用许多 4 阶张量，维度对应批次大小、图像宽度、图像高度和颜色通道。

``` python
my_image = tf.zeros([10, 299, 299, 3])  # batch x height x width x color
```

### 获取 `tf.Tensor` 对象的阶

要确定 `tf.Tensor` 对象的阶，需调用 `tf.rank` 方法。例如，以下方法会程序化地确定上一章节中所定义的 `tf.Tensor` 的阶：

```python
r = tf.rank(my_image)
# After the graph runs, r will hold the value 4.
```

### 引用 `tf.Tensor` 切片

由于 `tf.Tensor` 是 n 维单元数组，因此要访问 `tf.Tensor` 中的某一单元，需要指定 n 个索引。

0 阶张量（标量）不需要索引，因为其本身便是单一数字。

对于 1 阶张量（矢量），可以通过传递一个索引访问某个数字：

```python
my_scalar = my_vector[2]
```

请注意，如果想从矢量中动态地选择元素，那么在 `[]` 内传递的索引本身可以是一个标量 `tf.Tensor`。

对于 2 阶及以上的张量，情况更为有趣。对于 2 阶 `tf.Tensor`，传递两个数字会如预期般返回一个标量：


```python
my_scalar = my_matrix[1, 2]
```


而传递一个数字则会返回一个矩阵子矢量，如下所示：


```python
my_row_vector = my_matrix[2]
my_column_vector = my_matrix[:, 3]
```

符号 `:` 是 Python 切片语法，意味“不要触碰该维度”。这对更高阶的张量来说很有用，可以帮助访问其子矢量，子矩阵，甚至其他子张量。


## 形状

张量的形状是每个维度中元素的数量。TensorFlow 在图的构建过程中自动推理形状。这些推理的形状可能具有已知或未知的阶。如果阶已知，则每个维度的大小可能已知或未知。

TensorFlow 文件编制中通过三种符号约定来描述张量维度：阶，形状和维数。下表阐述了三者如何相互关联：

阶 | 形状 | 维数 | 示例    
--- | --- | --- | ---
0 | [] | 0-D | 0 维张量。标量。
1 | [D0] | 1-D | 形状为 [5] 的 1 维张量。
2 | [D0, D1] | 2-D | 形状为 [3, 4] 的 2 维张量。
3 | [D0, D1, D2] | 3-D | 形状为 [1, 4, 3] 的 3 维张量。
n | [D0, D1, ... Dn-1] | n-D | 形状为 [D0, D1, ... Dn-1] 的张量。

形状可以通过整型 Python 列表/元组或者 `tf.TensorShape` 表示。

### 获取 `tf.Tensor` 对象的形状

可以通过两种方法获取 `tf.Tensor` 的形状。在构建图的时候，询问有关张量形状的已知信息通常很有帮助。可以通过查看 `shape` 属性（属于 `tf.Tensor` 对象）获取这些信息。该方法会返回一个 `TensorShape` 对象，这样可以方便地表示部分指定的形状（因为在构建图的时候，并不是所有形状都完全已知）。

也可以获取一个将在运行时表示另一个 `tf.Tensor` 的完全指定形状的 `tf.Tensor`。为此，可以调用 `tf.shape` 操作。如此一来，您可以构建一个图，通过构建其他取决于输入 `tf.Tensor` 的动态形状的张量来控制张量的形状。

例如，以下代码展示了如何创建大小与给定矩阵中的列数相同的零矢量：

``` python
zeros = tf.zeros(my_matrix.shape[1])
```

### 改变形状：`tf.Tensor`

张量的元素数量是其所有形状大小的乘积。标量的元素数量永远是 `1`。由于通常有许多不同的形状具有相同数量的元素，因此如果能够改变 `tf.Tensor` 的形状并使其元素固定不变通常会很方便。为此，可以使用 `tf.reshape`。

以下示例演示如何重构张量：

```python
rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10])  # Reshape existing content into
                                                 # a 6x10 matrix
matrixB = tf.reshape(matrix, [3, -1])  #  Reshape existing content into a 3x20
                                       # matrix. -1 tells reshape to calculate
                                       # the size of this dimension.
matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # Reshape existing content into a
                                             #4x3x5 tensor

# Note that the number of elements of the reshaped Tensors has to match the
# original number of elements. Therefore, the following example generates an
# error because no possible value for the last dimension will match the number
# of elements.
yet_another = tf.reshape(matrixAlt, [13, 2, -1])  # ERROR!
```

## 数据类型

除维度外，张量还具有数据类型。如需数据类型的完整列表，请参阅 `tf.DType` 页面。

一个 `tf.Tensor` 只能有一种数据类型。但是，可以将任意数据结构序列化为 `string` 并将其存储在 `tf.Tensor` 中。

可以将 `tf.Tensor` 从一种数据类型转型为另一种（通过 `tf.cast`）：

``` python
# Cast a constant integer tensor into floating point.
float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)
```

要检查 `tf.Tensor` 的数据类型，请使用 `Tensor.dtype` 属性。

用 python 对象创建 `tf.Tensor` 时，可以选择指定数据类型。如果不指定数据类型，TensorFlow 会选择一个可以表示您的数据的数据类型。TensorFlow 会将 Python 整数转型为 `tf.int32`，并将 python 浮点数转型为 `tf.float32`。此外，TensorFlow 使用 Numpy 在转换至数组时使用的相同规则。

## 评估张量

计算图构建完毕后，您可以运行生成特定 `tf.Tensor` 的计算并获取分配给它的值。这对于程序调试通常很有帮助，也是 TensorFlow 的大部分功能正常运行所必需的。

评估张量最简单的方法是使用 `Tensor.eval` 方法。例如：

```python
constant = tf.constant([1, 2, 3])
tensor = constant * constant
print(tensor.eval())
```

`eval` 方法仅在默认 `tf.Session` 处于活跃状态时才起作用（详情请参阅“图和会话”）。

`Tensor.eval` 会返回一个与张量内容相同的 NumPy 数组。

有时无法在没有背景信息的情况下评估 `tf.Tensor`，因为它的值可能取决于无法获取的动态信息。例如，在没有为 `placeholder` 提供值的情况下，无法评估依赖于 `placeholder` 的张量。

``` python
p = tf.placeholder(tf.float32)
t = p + 1.0
t.eval()  # This will fail, since the placeholder did not get a value.
t.eval(feed_dict={p:2.0})  # This will succeed because we're feeding a value
                           # to the placeholder.
```

请注意，可以提供任何 `tf.Tensor`，而不仅仅是占位符。

其他模型构造可能会使评估 `tf.Tensor` 变得较为复杂。TensorFlow 无法直接评估在函数内部或控制流结构内部定义的 `tf.Tensor`。如果 `tf.Tensor` 取决于队列中的值，那么只有在某个项加入队列后才能评估 `tf.Tensor`；否则，评估将被搁置。在处理队列时，请先调用 `tf.train.start_queue_runners`，再评估任何 `tf.Tensor`。

## 输出张量

出于调试目的，您可能需要输出 `tf.Tensor` 的值。虽然
[tfdbg](/docs/tensorflow/guide/debugger) 提供高级调试支持，但 TensorFlow 也有一个操作可以直接输出 `tf.Tensor` 的值。

请注意，输出 `tf.Tensor` 时很少使用以下模式：

``` python
t = <<some tensorflow operation>>
print(t)  # This will print the symbolic tensor when the graph is being built.
          # This tensor does not have a value in this context.
```

上述代码会输出 `tf.Tensor` 对象（表示延迟计算），而不是其值。TensorFlow 提供了 `tf.Print` 操作，该操作会返回其第一个张量参数（保持不变），同时输出作为第二个参数传递的 `tf.Tensor` 集合。

要正确使用 `tf.Print`，必须使用其返回的值。请参阅下文的示例：

``` python
t = <<some tensorflow operation>>
tf.Print(t, [t])  # This does nothing
t = tf.Print(t, [t])  # Here we are using the value returned by tf.Print
result = t + 1  # Now when result is evaluated the value of `t` will be printed.
```

在评估 `result` 时，会评估所有影响 `result` 的元素。由于 `result` 依靠 `t`，而评估 `t` 会导致输出其输入（`t` 的旧值），所以系统会输出 `t`。

