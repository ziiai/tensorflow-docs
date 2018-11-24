#  变量

TensorFlow 变量是表示程序处理的共享持久状态的最佳方法。

我们使用 `tf.Variable` 类操作变量。`tf.Variable` 表示可通过对其运行操作来改变其值的张量。与 `tf.Tensor` 对象不同，`tf.Variable` 存在于单个 `session.run` 调用的上下文之外。

在 TensorFlow 内部，`tf.Variable` 会存储持久性张量。具体 op 允许您读取和修改此张量的值。这些修改在多个 `tf.Session` 之间是可见的，因此对于一个 tf.Variable，多个工作器可以看到相同的值。

## 创建变量

创建变量的最佳方式是调用 `tf.get_variable` 函数。此函数要求您指定变量的名称。其他副本将使用此名称访问同一变量，以及在对模型设置检查点和导出模型时指定此变量的值。`tf.get_variable` 还允许您重复使用先前创建的同名变量，从而轻松定义重复利用层的模型。

要使用 `tf.get_variable` 创建变量，只需提供名称和形状即可

``` python
my_variable = tf.get_variable("my_variable", [1, 2, 3])
```

这将创建一个名为“my_variable”的变量，该变量是形状为 `[1, 2, 3]` 的三维张量。默认情况下，此变量将具有 `dtypetf.float32`，其初始值将通过 `tf.glorot_uniform_initializer` 随机设置。

您可以选择为 `tf.get_variable` 指定 `dtype` 和初始化器。例如：

``` python
my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32,
  initializer=tf.zeros_initializer)
```

TensorFlow 提供了许多方便的初始化器。或者，您也可以将 `tf.Variable` 初始化为 `tf.Tensor` 的值。例如：

``` python
other_variable = tf.get_variable("other_variable", dtype=tf.int32,
  initializer=tf.constant([23, 42]))
```

请注意，当初始化器是 `tf.Tensor` 时，您不应指定变量的形状，因为将使用初始化器张量的形状。


<a name="collections"></a>
### 变量集合

由于 TensorFlow 程序的未连接部分可能需要创建变量，因此能有一种方式访问所有变量有时十分受用。为此，TensorFlow 提供了集合，它们是张量或其他对象（如 `tf.Variable` 实例）的命名列表。

默认情况下，每个 `tf.Variable` 都放置在以下两个集合中：

- `tf.GraphKeys.GLOBAL_VARIABLES` - 可以在多台设备间共享的变量，
- `tf.GraphKeys.TRAINABLE_VARIABLES` - TensorFlow 将计算其梯度的变量。

如果您不希望变量可训练，可以将其添加到 `tf.GraphKeys.LOCAL_VARIABLES` 集合中。例如，以下代码段展示了如何将名为 `my_local` 的变量添加到此集合中：

``` python
my_local = tf.get_variable("my_local", shape=(),
collections=[tf.GraphKeys.LOCAL_VARIABLES])
```

或者，您可以指定 `trainable=False`（作为 `tf.get_variable` 的参数）：

``` python
my_non_trainable = tf.get_variable("my_non_trainable",
                                   shape=(),
                                   trainable=False)
```


您也可以使用自己的集合。集合名称可为任何字符串，且您无需显式创建集合。创建变量（或任何其他对象）后，要将其添加到集合中，请调用 `tf.add_to_collection`。例如，以下代码将名为 `my_local` 的现有变量添加到名为 `my_collection_name` 的集合中：

``` python
tf.add_to_collection("my_collection_name", my_local)
```

要检索您放置在某个集合中的所有变量（或其他对象）的列表，您可以使用：

``` python
tf.get_collection("my_collection_name")
```

### 设备放置方式

与任何其他 TensorFlow 指令一样，您可以将变量放置在特定设备上。例如，以下代码段创建了名为 `v` 的变量并将其放置在第二个 GPU 设备上：

``` python
with tf.device("/device:GPU:1"):
  v = tf.get_variable("v", [1])
```

在分布式设置中，将变量放置在正确设备上尤为重要。如果不小心将变量放在工作器而不是参数服务器上，可能会严重减慢训练速度，最坏的情况下，可能会让每个工作器不断复制各个变量。为此，我们提供了 `tf.train.replica_device_setter`，它可以自动将变量放置在参数服务器中。例如：

``` python
cluster_spec = {
    "ps": ["ps0:2222", "ps1:2222"],
    "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
  v = tf.get_variable("v", shape=[20, 20])  # this variable is placed
                                            # in the parameter server
                                            # by the replica_device_setter
```

## 初始化变量

变量必须先初始化后才可使用。如果您在低级别 TensorFlow API 中进行编程（即您在显式创建自己的图和会话），则必须明确初始化变量。`tf.contrib.slim`、`tf.estimator.Estimator` 和 `Keras` 等大多数高级框架在训练模型前会自动为您初始化变量。

显式初始化在其他方面很有用。它允许您在从检查点重新加载模型时不用重新运行潜在资源消耗大的初始化器，并允许在分布式设置中共享随机初始化的变量时具有确定性。

要在训练开始前一次性初始化所有可训练变量，请调用 `tf.global_variables_initializer()`。此函数会返回一个操作，负责初始化 `tf.GraphKeys.GLOBAL_VARIABLES` 集合中的所有变量。运行此操作会初始化所有变量。例如：

``` python
session.run(tf.global_variables_initializer())
# Now all variables are initialized.
```

如果您确实需要自行初始化变量，则可以运行变量的初始化器操作。例如：

``` python
session.run(my_variable.initializer)
```


您可以查询哪些变量尚未初始化。例如，以下代码会打印所有尚未初始化的变量名称：

``` python
print(session.run(tf.report_uninitialized_variables()))
```


请注意，默认情况下，`tf.global_variables_initializer` 不会指定变量的初始化顺序。因此，如果变量的初始值取决于另一变量的值，那么很有可能会出现错误。任何时候，如果您在并非所有变量都已初始化的上下文中使用某个变量值（例如在初始化某个变量时使用另一变量的值），最好使用 `variable.initialized_value()`，而非 `variable`：

``` python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = tf.get_variable("w", initializer=v.initialized_value() + 1)
```

## 使用变量

要在 TensorFlow 图中使用 `tf.Variable` 的值，只需将其视为普通 `tf.Tensor` 即可：

``` python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = v + 1  # w is a tf.Tensor which is computed based on the value of v.
           # Any time a variable is used in an expression it gets automatically
           # converted to a tf.Tensor representing its value.
```

要为变量赋值，请使用 `assign`、`assign_add` 方法以及 `tf.Variable` 类中的友元。例如，以下就是调用这些方法的方式：

``` python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(1)
tf.global_variables_initializer().run()
sess.run(assignment)  # or assignment.op.run(), or assignment.eval()
```

大多数 TensorFlow 优化器都有专门的 op，会根据某种梯度下降算法有效地更新变量的值。请参阅 `tf.train.Optimizer`，了解如何使用优化器。

由于变量是可变的，因此及时了解任意时间点所使用的变量值版本有时十分有用。要在事件发生后强制重新读取变量的值，可以使用 `tf.Variable.read_value`。例如：

``` python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(1)
with tf.control_dependencies([assignment]):
  w = v.read_value()  # w is guaranteed to reflect v's value after the
                      # assign_add operation.
```


## 共享变量

TensorFlow 支持两种共享变量的方式：

- 显式传递 `tf.Variable` 对象。
- 将 `tf.Variable` 对象隐式封装在 `tf.variable_scope` 对象内。

虽然显式传递变量的代码非常清晰，但有时编写在其实现中隐式使用变量的 TensorFlow 函数非常方便。`tf.layers` 中的大多数功能层以及所有 `tf.metrics` 和部分其他库实用程序都使用这种方法。

变量作用域允许您在调用隐式创建和使用变量的函数时控制变量重用。作用域还允许您以分层和可理解的方式命名变量。

例如，假设我们编写一个函数来创建一个卷积/relu 层：

```python
def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)
```

此函数使用短名称 `weights` 和 `biases`，这有利于清晰区分二者。然而，在真实模型中，我们需要很多此类卷积层，而且重复调用此函数将不起作用：

``` python
input1 = tf.random_normal([1,10,10,32])
input2 = tf.random_normal([1,20,20,32])
x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape = [32])  # This fails.
```

由于期望的操作不清楚（创建新变量还是重新使用现有变量？），因此 TensorFlow 将会失败。不过，在不同作用域内调用 `conv_relu` 可表明我们想要创建新变量：

```python
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])
```

如果您想要共享变量，有两种方法可供选择。首先，您可以使用 `reuse=True` 创建具有相同名称的作用域：

``` python
with tf.variable_scope("model"):
  output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
  output2 = my_image_filter(input2)

```

您也可以调用 `scope.reuse_variables()` 以触发重用：

``` python
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
  scope.reuse_variables()
  output2 = my_image_filter(input2)

```

由于依赖于作用域的确切字符串名称可能比较危险，因此也可以根据另一作用域初始化某个变量作用域：

``` python
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
with tf.variable_scope(scope, reuse=True):
  output2 = my_image_filter(input2)

```

