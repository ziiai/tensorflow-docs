#  Eager Execution

TensorFlow 的 Eager Execution 是一种命令式编程环境，可立即评估操作，无需构建图：操作会返回具体的值，而不是构建以后再运行的计算图。这样能让您轻松地开始使用 TensorFlow 和调试模型，并且还减少了样板代码。要遵循本指南，请在交互式 `python` 解释器中运行下面的代码示例。

Eager Execution 是一个灵活的机器学习平台，用于研究和实验，可提供：

- 直观的界面 - 自然地组织代码结构并使用 Python 数据结构。快速迭代小模型和小型数据集。
- 更轻松的调试功能 - 直接调用操作以检查正在运行的模型并测试更改。使用标准 Python 调试工具进行即时错误报告。
- 自然控制流程 - 使用 Python 控制流程而不是图控制流程，简化了动态模型的规范。

Eager Execution 支持大多数 TensorFlow 操作和 GPU 加速。有关在 Eager Execution 中运行的示例集合，请参阅：
[tensorflow/contrib/eager/python/examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples)。

注意：如果启用 Eager Execution，某些模型的开销可能会增加。我们正在改进性能；如果发现问题，请
[报告错误](https://github.com/tensorflow/tensorflow/issues)
，并分享您的基准测试结果。

## 设置和基本用法

升级到最新版本的 TensorFlow：

```
$ pip install --upgrade tensorflow
```

要启动 Eager Execution，请将 `tf.enable_eager_execution()` 添加到程序或控制台会话的开头。不要将此操作添加到程序调用的其他模块。

```py
from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()
```

现在您可以运行 TensorFlow 操作了，结果将立即返回：

```py
tf.executing_eagerly()        # => True

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))  # => "hello, [[4.]]"
```

启用 Eager Execution 会改变 TensorFlow 操作的行为方式 - 现在它们会立即评估并将值返回给 Python。`tf.Tensor` 对象会引用具体值，而不是指向计算图中的节点的符号句柄。由于不需要构建稍后在会话中运行的计算图，因此使用 `print()` 或调试程序很容易检查结果。评估、输出和检查张量值不会中断计算梯度的流程。

Eager Execution 适合与 [NumPy](http://www.numpy.org/)
一起使用。NumPy 操作接受 `tf.Tensor` 参数。TensorFlow
[数学运算](https://www.tensorflow.org/api_guides/python/math_ops)
将 Python 对象和 NumPy 数组转换为 `tf.Tensor` 对象。`tf.Tensor.numpy` 方法返回对象的值作为 NumPy `ndarray`。

```py
a = tf.constant([[1, 2],
                 [3, 4]])
print(a)
# => tf.Tensor([[1 2]
#               [3 4]], shape=(2, 2), dtype=int32)

# Broadcasting support
b = tf.add(a, 1)
print(b)
# => tf.Tensor([[2 3]
#               [4 5]], shape=(2, 2), dtype=int32)

# Operator overloading is supported
print(a * b)
# => tf.Tensor([[ 2  6]
#               [12 20]], shape=(2, 2), dtype=int32)

# Use NumPy values
import numpy as np

c = np.multiply(a, b)
print(c)
# => [[ 2  6]
#     [12 20]]

# Obtain numpy value from a tensor:
print(a.numpy())
# => [[1 2]
#     [3 4]]
```

`tf.contrib.eager` 模块包含可用于 Eager Execution 和 Graph Execution 环境的符号，对编写[处理图](#work_with_graphs)的代码非常有用：

```py
tfe = tf.contrib.eager
```

## 动态控制流

Eager Execution 的一个主要好处是，在执行模型时，主机语言的所有功能都可用。因此，编写 [fizzbuzz](https://en.wikipedia.org/wiki/Fizz_buzz) 很容易（举例而言）：

```py
def fizzbuzz(max_num):
  counter = tf.constant(0)
  max_num = tf.convert_to_tensor(max_num)
  for num in range(max_num.numpy()):
    num = tf.constant(num)
    if int(num % 3) == 0 and int(num % 5) == 0:
      print('FizzBuzz')
    elif int(num % 3) == 0:
      print('Fizz')
    elif int(num % 5) == 0:
      print('Buzz')
    else:
      print(num)
    counter += 1
  return counter
```

这段代码具有依赖于张量值的条件并在运行时输出这些值。

## 构建模型

许多机器学习模型通过组合层来表示。将 TensorFlow 与 Eager Execution 结合使用时，您可以编写自己的层或使用在 `tf.keras.layers` 程序包中提供的层。

虽然您可以使用任何 Python 对象表示层，但 TensorFlow 提供了便利的基类 `tf.keras.layers.Layer`。您可以通过继承它实现自己的层：

```py
class MySimpleLayer(tf.keras.layers.Layer):
  def __init__(self, output_units):
    super(MySimpleLayer, self).__init__()
    self.output_units = output_units

  def build(self, input_shape):
    # The build method gets called the first time your layer is used.
    # Creating variables on build() allows you to make their shape depend
    # on the input shape and hence removes the need for the user to specify
    # full shapes. It is possible to create variables during __init__() if
    # you already know their full shapes.
    self.kernel = self.add_variable(
      "kernel", [input_shape[-1], self.output_units])

  def call(self, input):
    # Override call() instead of __call__ so we can perform some bookkeeping.
    return tf.matmul(input, self.kernel)
```

请使用 `tf.keras.layers.Dense` 层（而不是上面的 `MySimpleLayer`），因为它具有其功能的超集（它也可以添加偏差）。

将层组合成模型时，可以使用 `tf.keras.Sequential` 表示由层线性堆叠的模型。它非常适合用于基本模型：

```py
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, input_shape=(784,)),  # must declare input shape
  tf.keras.layers.Dense(10)
])
```

或者，通过继承 `tf.keras.Model` 将模型整理为类。这是一个本身也是层的层容器，允许 `tf.keras.Model` 对象包含其他 `tf.keras.Model` 对象。

```py
class MNISTModel(tf.keras.Model):
  def __init__(self):
    super(MNISTModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(units=10)
    self.dense2 = tf.keras.layers.Dense(units=10)

  def call(self, input):
    """Run the model."""
    result = self.dense1(input)
    result = self.dense2(result)
    result = self.dense2(result)  # reuse variables from dense2 layer
    return result

model = MNISTModel()
```

因为第一次将输入传递给层时已经设置参数，所以不需要为 `tf.keras.Model` 类设置输入形状。

`tf.keras.layers` 类会创建并包含自己的模型变量，这些变量与其层对象的生命周期相关联。要共享层变量，请共享其对象。


## Eager 训练

### 计算梯度

[自动微分](https://en.wikipedia.org/wiki/Automatic_differentiation)
对于实现机器学习算法（例如用于训练神经网络的
[反向传播](https://en.wikipedia.org/wiki/Backpropagation) ）来说很有用。在 Eager Execution 期间，请使用 `tf.GradientTape` 跟踪操作以便稍后计算梯度。

`tf.GradientTape` 是一种选择性功能，可在不跟踪时提供最佳性能。由于在每次调用期间都可能发生不同的操作，因此所有前向传播操作都会记录到“磁带”中。要计算梯度，请反向播放磁带，然后放弃。特定的 `tf.GradientTape` 只能计算一个梯度；随后的调用会抛出运行时错误。

```py
w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w

grad = tape.gradient(loss, w)
print(grad)  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)
```

下面是一个记录前向传播操作以训练简单模型的 `tf.GradientTape` 示例：

```py
# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 1000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

def prediction(input, weight, bias):
  return input * weight + bias

# A loss function using mean-squared error
def loss(weights, biases):
  error = prediction(training_inputs, weights, biases) - training_outputs
  return tf.reduce_mean(tf.square(error))

# Return the derivative of loss with respect to weight and bias
def grad(weights, biases):
  with tf.GradientTape() as tape:
    loss_value = loss(weights, biases)
  return tape.gradient(loss_value, [weights, biases])

train_steps = 200
learning_rate = 0.01
# Start with arbitrary values for W and B on the same batch of data
W = tf.Variable(5.)
B = tf.Variable(10.)

print("Initial loss: {:.3f}".format(loss(W, B)))

for i in range(train_steps):
  dW, dB = grad(W, B)
  W.assign_sub(dW * learning_rate)
  B.assign_sub(dB * learning_rate)
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(W, B)))

print("Final loss: {:.3f}".format(loss(W, B)))
print("W = {}, B = {}".format(W.numpy(), B.numpy()))
```

输出（具体数字可能会有所不同）：

```
Initial loss: 71.204
Loss at step 000: 68.333
Loss at step 020: 30.222
Loss at step 040: 13.691
Loss at step 060: 6.508
Loss at step 080: 3.382
Loss at step 100: 2.018
Loss at step 120: 1.422
Loss at step 140: 1.161
Loss at step 160: 1.046
Loss at step 180: 0.996
Final loss: 0.974
W = 3.01582956314, B = 2.1191945076
```

重播 `tf.GradientTape` 以计算梯度并将梯度应用于训练循环中。下面是来自 
[mnist_eager.py](https://github.com/tensorflow/models/blob/master/official/mnist/mnist_eager.py)
示例的摘录：

```py
dataset = tf.data.Dataset.from_tensor_slices((data.train.images,
                                              data.train.labels))
...
for (batch, (images, labels)) in enumerate(dataset):
  ...
  with tf.GradientTape() as tape:
    logits = model(images, training=True)
    loss_value = loss(logits, labels)
  ...
  grads = tape.gradient(loss_value, model.variables)
  optimizer.apply_gradients(zip(grads, model.variables),
                            global_step=tf.train.get_or_create_global_step())
```


以下示例将创建一个多层模型，该模型会对标准 MNIST 手写数字进行分类。它演示了在 Eager Execution 环境中构建可训练图的优化器和层 API。

### 训练模型

即使没有训练，也可以在 Eager Execution 中调用模型并检查输出：

```py
# Create a tensor representing a blank image
batch = tf.zeros([1, 1, 784])
print(batch.shape)  # => (1, 1, 784)

result = model(batch)
# => tf.Tensor([[[ 0.  0., ..., 0.]]], shape=(1, 1, 10), dtype=float32)
```

该示例使用了
[TensorFlow MNIST 示例](https://github.com/tensorflow/models/tree/master/official/mnist)
中的
[dataset.py 模块](https://github.com/tensorflow/models/blob/master/official/mnist/dataset.py)
，请将该文件下载到本地目录。运行以下命令以将 MNIST 数据文件下载到工作目录并准备要进行训练的 `tf.data.Dataset`：

```py
import dataset  # download dataset.py file
dataset_train = dataset.train('./datasets').shuffle(60000).repeat(4).batch(32)
```

为了训练模型，请定义损失函数以进行优化，然后计算梯度。使用优化器更新变量：

```py
def loss(model, x, y):
  prediction = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=prediction)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

x, y = iter(dataset_train).next()
print("Initial loss: {:.3f}".format(loss(model, x, y)))

# Training loop
for (i, (x, y)) in enumerate(dataset_train):
  # Calculate derivatives of the input function with respect to its parameters.
  grads = grad(model, x, y)
  # Apply the gradient to the model
  optimizer.apply_gradients(zip(grads, model.variables),
                            global_step=tf.train.get_or_create_global_step())
  if i % 200 == 0:
    print("Loss at step {:04d}: {:.3f}".format(i, loss(model, x, y)))

print("Final loss: {:.3f}".format(loss(model, x, y)))
```

输出（具体数字可能会有所不同）：

```
Initial loss: 2.674
Loss at step 0000: 2.593
Loss at step 0200: 2.143
Loss at step 0400: 2.009
Loss at step 0600: 2.103
Loss at step 0800: 1.621
Loss at step 1000: 1.695
...
Loss at step 6600: 0.602
Loss at step 6800: 0.557
Loss at step 7000: 0.499
Loss at step 7200: 0.744
Loss at step 7400: 0.681
Final loss: 0.670
```

为了加速训练，可以将计算移至 GPU：

```py
with tf.device("/gpu:0"):
  for (i, (x, y)) in enumerate(dataset_train):
    # minimize() is equivalent to the grad() and apply_gradients() calls.
    optimizer.minimize(lambda: loss(model, x, y),
                       global_step=tf.train.get_or_create_global_step())
```

### 变量和优化器

`tfe.Variable` 对象会存储在训练期间访问的可变 `tf.Tensor` 值，以更加轻松地实现自动微分。模型的参数可以作为变量封装在类中。

通过将 `tfe.Variable` 与 `tf.GradientTape` 结合使用可以更好地封装模型参数。例如，上面的自动微分示例可以重写为：

```py
class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.W = tf.Variable(5., name='weight')
    self.B = tf.Variable(10., name='bias')
  def call(self, inputs):
    return inputs * self.W + self.B

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# The loss function to be optimized
def loss(model, inputs, targets):
  error = model(inputs) - targets
  return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, [model.W, model.B])

# Define:
# 1. A model.
# 2. Derivatives of a loss function with respect to model parameters.
# 3. A strategy for updating the variables based on the derivatives.
model = Model()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

# Training loop
for i in range(300):
  grads = grad(model, training_inputs, training_outputs)
  optimizer.apply_gradients(zip(grads, [model.W, model.B]),
                            global_step=tf.train.get_or_create_global_step())
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))
```

输出（具体数字可能会有所不同）：

```
Initial loss: 69.066
Loss at step 000: 66.368
Loss at step 020: 30.107
Loss at step 040: 13.959
Loss at step 060: 6.769
Loss at step 080: 3.567
Loss at step 100: 2.141
Loss at step 120: 1.506
Loss at step 140: 1.223
Loss at step 160: 1.097
Loss at step 180: 1.041
Loss at step 200: 1.016
Loss at step 220: 1.005
Loss at step 240: 1.000
Loss at step 260: 0.998
Loss at step 280: 0.997
Final loss: 0.996
W = 2.99431324005, B = 2.02129220963
```

## 在 Eager Execution 期间将对象用于状态

使用 Graph Execution 时，程序状态（如变量）存储在全局集合中，它们的生命周期由 `tf.Session` 对象管理。相反，在 Eager Execution 期间，状态对象的生命周期由其对应的 Python 对象的生命周期决定。

### 变量是对象

在 Eager Execution 期间，变量会一直存在，直到相应对象的最后一个引用被移除，然后变量被删除。

```py
with tf.device("gpu:0"):
  v = tf.Variable(tf.random_normal([1000, 1000]))
  v = None  # v no longer takes up GPU memory
```

### 基于对象的保存

`tfe.Checkpoint` 可以将 `tfe.Variable` 保存到检查点并从中恢复：

```py
x = tf.Variable(10.)

checkpoint = tf.train.Checkpoint(x=x)  # save as "x"

x.assign(2.)   # Assign a new value to the variables and save.
save_path = checkpoint.save('./ckpt/')

x.assign(11.)  # Change the variable after saving.

# Restore values from the checkpoint
checkpoint.restore(save_path)

print(x)  # => 2.0
```

要保存和加载模型，`tfe.Checkpoint` 会存储对象的内部状态，而不需要隐藏变量。要记录 `model`、`optimizer` 和全局步的状态，请将它们传递到 `tfe.Checkpoint`：

```py
model = MyModel()
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
checkpoint_dir = ‘/path/to/model_dir’
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model,
                           optimizer_step=tf.train.get_or_create_global_step())

root.save(file_prefix=checkpoint_prefix)
# or
root.restore(tf.train.latest_checkpoint(checkpoint_dir))
```

### 面向对象的指标

`tfe.metrics` 存储为对象。通过将新数据传递给可调用对象来更新指标，并使用 `tfe.metrics.result` 方法检索结果，例如：

```py
m = tfe.metrics.Mean("loss")
m(0)
m(5)
m.result()  # => 2.5
m([8, 9])
m.result()  # => 5.5
```

#### 汇总和 TensorBoard

[TensorBoard](/docs/tensorflow/guide/summaries_and_tensorboard) 是一种可视化工具，用于了解、调试和优化模型训练过程。它使用在执行程序时编写的汇总事件。

`tf.contrib.summary` 与 Eager Execution 和 Graph Execution 环境兼容。汇总操作（如 `tf.contrib.summary.scalar`）在模型构建期间被插入。例如，要每 100 个全局步记录一次汇总：

```py
global_step = tf.train.get_or_create_global_step()
writer = tf.contrib.summary.create_file_writer(logdir)
writer.set_as_default()

for _ in range(iterations):
  global_step.assign_add(1)
  # Must include a record_summaries method
  with tf.contrib.summary.record_summaries_every_n_global_steps(100):
    # your model code goes here
    tf.contrib.summary.scalar('loss', loss)
     ...
```

## 自动微分高级内容

### 动态模型

`tf.GradientTape` 也可用于动态模型。这个
[回溯线搜索](https://wikipedia.org/wiki/Backtracking_line_search)
算法示例看起来像普通的 NumPy 代码，除了存在梯度并且可微分，尽管控制流比较复杂：

```py
def line_search_step(fn, init_x, rate=1.0):
  with tf.GradientTape() as tape:
    # Variables are automatically recorded, but manually watch a tensor
    tape.watch(init_x)
    value = fn(init_x)
  grad = tape.gradient(value, init_x)
  grad_norm = tf.reduce_sum(grad * grad)
  init_value = value
  while value > init_value - rate * grad_norm:
    x = init_x - rate * grad
    value = fn(x)
    rate /= 2.0
  return x, value
```

### 计算梯度的其他函数

`tf.GradientTape` 是用于计算梯度的强大接口，还有另一种 [Autograd](https://github.com/HIPS/autograd)样式的 API 可用于自动微分。如果只用张量和梯度函数编写数学代码，而不使用 `tfe.Variables`，则这些函数非常有用：

- `tfe.gradients_function` - 返回一个函数，该函数会计算其输入函数参数相对于其参数的导数。输入函数参数必须返回一个标量值。当返回的函数被调用时，它会返回一个 `tf.Tensor` 对象列表：输入函数的每个参数各对应一个元素。因为任何相关信息都必须作为函数参数传递，所以如果依赖于许多可训练参数，则会变得很难处理。
- `tfe.value_and_gradients_function` - 与 `tfe.gradients_function` 相似，但是当返回的函数被调用时，除了输入函数相对于其参数的导数列表之外，它还会返回输入函数的值。

在以下示例中，`tfe.gradients_function` 将 `square` 函数作为参数，并返回一个函数（计算 `square` 相对于其输入的偏导数）。如果计算输入为 `3` 时 `square` 的偏导数，`grad(3.0)` 会返回 `6`。

```py
def square(x):
  return tf.multiply(x, x)

grad = tfe.gradients_function(square)

square(3.)  # => 9.0
grad(3.)    # => [6.0]

# The second-order derivative of square:
gradgrad = tfe.gradients_function(lambda x: grad(x)[0])
gradgrad(3.)  # => [2.0]

# The third-order derivative is None:
gradgradgrad = tfe.gradients_function(lambda x: gradgrad(x)[0])
gradgradgrad(3.)  # => [None]


# With flow control:
def abs(x):
  return x if x > 0. else -x

grad = tfe.gradients_function(abs)

grad(3.)   # => [1.0]
grad(-3.)  # => [-1.0]
```

### 自定义梯度

自定义梯度是在 Eager Execution 和 Graph Execution 中覆盖梯度的一种简单方式。在正向函数中，定义相对于输入、输出或中间结果的梯度。例如，下面是在反向传播中截断梯度范数的一种简单方式：

```py
@tf.custom_gradient
def clip_gradient_by_norm(x, norm):
  y = tf.identity(x)
  def grad_fn(dresult):
    return [tf.clip_by_norm(dresult, norm), None]
  return y, grad_fn
```

自定义梯度通常用于为一系列操作提供数值稳定的梯度：

```py
def log1pexp(x):
  return tf.log(1 + tf.exp(x))
grad_log1pexp = tfe.gradients_function(log1pexp)

# The gradient computation works fine at x = 0.
grad_log1pexp(0.)  # => [0.5]

# However, x = 100 fails because of numerical instability.
grad_log1pexp(100.)  # => [nan]
```

在此处，`log1pexp` 函数可以通过自定义梯度进行分析简化。下面的实现重用了在前向传播期间计算的 `tf.exp(x)` 的值，通过消除冗余计算，变得更加高效：

```py
@tf.custom_gradient
def log1pexp(x):
  e = tf.exp(x)
  def grad(dy):
    return dy * (1 - 1 / (1 + e))
  return tf.log(1 + e), grad

grad_log1pexp = tfe.gradients_function(log1pexp)

# As before, the gradient computation works fine at x = 0.
grad_log1pexp(0.)  # => [0.5]

# And the gradient computation also works at x = 100.
grad_log1pexp(100.)  # => [1.0]
```

## 性能

在 Eager Execution 期间，计算会自动分流到 GPU。如果要控制计算运行的位置，可以将其放在 `tf.device('/gpu:0')` 块（或 CPU 等效块）中：

```py
import time

def measure(x, steps):
  # TensorFlow initializes a GPU the first time it's used, exclude from timing.
  tf.matmul(x, x)
  start = time.time()
  for i in range(steps):
    x = tf.matmul(x, x)
  # tf.matmul can return before completing the matrix multiplication
  # (e.g., can return after enqueing the operation on a CUDA stream).
  # The x.numpy() call below will ensure that all enqueued operations
  # have completed (and will also copy the result to host memory,
  # so we're including a little more than just the matmul operation
  # time).
  _ = x.numpy()
  end = time.time()
  return end - start

shape = (1000, 1000)
steps = 200
print("Time to multiply a {} matrix by itself {} times:".format(shape, steps))

# Run on CPU:
with tf.device("/cpu:0"):
  print("CPU: {} secs".format(measure(tf.random_normal(shape), steps)))

# Run on GPU, if available:
if tfe.num_gpus() > 0:
  with tf.device("/gpu:0"):
    print("GPU: {} secs".format(measure(tf.random_normal(shape), steps)))
else:
  print("GPU: not found")
```

输出（确切数字取决于硬件）：

```
Time to multiply a (1000, 1000) matrix by itself 200 times:
CPU: 1.46628093719 secs
GPU: 0.0593810081482 secs
```

`tf.Tensor` 对象可以被复制到不同的设备来执行其操作：

```py
x = tf.random_normal([10, 10])

x_gpu0 = x.gpu()
x_cpu = x.cpu()

_ = tf.matmul(x_cpu, x_cpu)    # Runs on CPU
_ = tf.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0

if tfe.num_gpus() > 1:
  x_gpu1 = x.gpu(1)
  _ = tf.matmul(x_gpu1, x_gpu1)  # Runs on GPU:1
```

### 基准

对于计算量繁重的模型（如在 GPU 上训练的
[ResNet50](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/resnet50)
），Eager Execution 性能与 Graph Execution 相当。但是对于计算量较小的模型来说，这种性能差距会越来越大，并且有很多工作要做，以便为具有大量小操作的模型优化热代码路径。


## 处理图

虽然 Eager Execution 增强了开发和调试的交互性，但 TensorFlow Graph Execution 在分布式训练、性能优化和生产部署方面具有优势。不过，编写图形代码不同于编写常规 Python 代码，并且更难以调试。

为了构建和训练由图构建的模型，Python 程序首先构建一个表示计算的图，然后调用 `Session.run` 来发送该图，以便在基于 C++ 的运行时上执行。这种方式具有以下优势：

- 使用静态 `autodiff` 进行自动微分。
- 可轻松地部署到独立于平台的服务器。
- 基于图的优化（常见的子表达式消除、常量折叠等）。
- 编译和内核融合。
- 自动分发和复制（在分布式系统上放置节点）。

部署为 Eager Execution 编写的代码更加困难：要么从模型生成图，要么直接在服务器上运行 Python 运行时和代码。

### 编写兼容的代码

为 Eager Execution 编写的相同代码在 Graph Execution 期间也会构建图。在未启用 Eager Execution 的新 Python 会话中运行相同的代码便可实现此目的。

大多数 TensorFlow 操作在 Eager Execution 期间都有效，但需要注意以下几点：

- 使用 `tf.data`（而不是队列）进行输入处理，速度更快、更简单。
- 使用面向对象的层 API，如 `tf.keras.layers` 和 `tf.keras.Model`，因为它们有明确的变量存储空间。
- 大多数模型代码在 Eager Execution 和 Graph Execution 过程中效果一样，但也有例外情况。（例如，使用 Python 控制流更改基于输入的计算的动态模型。）
- 一旦通过 `tf.enable_eager_execution` 启用了 Eager Execution，就不能将其关闭。要返回到 Graph Execution，需要启动一个新的 Python 会话。

最好同时为 Eager Execution 和 Graph Execution 编写代码。这样，既可以获得 Eager Execution 的交互式实验和可调试性功能，又能拥有 Graph Execution 的分布式性能优势。

在 Eager Execution 中编写、调试和迭代代码，然后导入模型图用于生产部署。使用 `tfe.Checkpoint` 保存和恢复模型变量，这样可在 Eager Execution 和 Graph Execution 环境之间移动模型。请参阅
[tensorflow/contrib/eager/python/examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples)
中的示例。

### 在图环境中使用 Eager Execution

使用 `tfe.py_func` 在 TensorFlow 图环境中选择性地启用 Eager Execution。在未调用 `tf.enable_eager_execution()` 的情况下使用这种方法。

```py
def my_py_func(x):
  x = tf.matmul(x, x)  # You can use tf ops
  print(x)  # but it's eager!
  return x

with tf.Session() as sess:
  x = tf.placeholder(dtype=tf.float32)
  # Call eager function in graph!
  pf = tfe.py_func(my_py_func, [x], tf.float32)
  sess.run(pf, feed_dict={x: [[2.0]]})  # [[4.0]]
```
