
##### Copyright 2018 The TensorFlow Authors.

Licensed under the Apache License, Version 2.0 (the "License");


```
#@title Licensed under the Apache License, Version 2.0 (the "License"); { display-mode: "form" }
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# AutoGraph: Easy control flow for graphs 

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/guide/autograph"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/autograph.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/guide/autograph.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

[AutoGraph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/) helps you write complicated graph code using normal Python. Behind the scenes, AutoGraph automatically transforms your code into the equivalent [TensorFlow graph code](https://www.tensorflow.org/guide/graphs). AutoGraph already supports much of the Python language, and that coverage continues to grow. For a list of supported Python language features, see the [Autograph capabilities and limitations](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/LIMITATIONS.md).

## Setup

To use AutoGraph, install the latest version of TensorFlow:


```
! pip install -U tf-nightly
```

Import TensorFlow, AutoGraph, and any supporting modules:


```
from __future__ import division, print_function, absolute_import

import tensorflow as tf
layers = tf.keras.layers
from tensorflow.contrib import autograph


import numpy as np
import matplotlib.pyplot as plt
```

We'll enable  [eager execution](https://www.tensorflow.org/guide/eager) for demonstration purposes, but AutoGraph works in both eager and [graph execution](https://www.tensorflow.org/guide/graphs) environments:


```
tf.enable_eager_execution()
```

Note: AutoGraph converted code is designed to run during graph execution. When eager exectuon is enabled, use explicit graphs (as this example shows) or `tf.contrib.eager.defun`.

## Automatically convert Python control flow

AutoGraph will convert much of the Python language into the equivalent TensorFlow graph building code. 

Note: In real applications batching is essential for performance. The best code to convert to AutoGraph is code where the control flow is decided at the _batch_ level. If making decisions at the individual _example_ level, you must index and batch the examples to maintain performance while applying the control flow logic. 

AutoGraph converts a function like:


```
def square_if_positive(x):
  if x > 0:
    x = x * x
  else:
    x = 0.0
  return x
```

To a function that uses graph building:


```
print(autograph.to_code(square_if_positive))
```

Code written for eager execution can run in a `tf.Graph` with the same results, but with the benfits of graph execution:


```
print('Eager results: %2.2f, %2.2f' % (square_if_positive(tf.constant(9.0)), 
                                       square_if_positive(tf.constant(-9.0))))
```

Generate a graph-version and call it:


```
tf_square_if_positive = autograph.to_graph(square_if_positive)

with tf.Graph().as_default():  
  # The result works like a regular op: takes tensors in, returns tensors.
  # You can inspect the graph using tf.get_default_graph().as_graph_def()
  g_out1 = tf_square_if_positive(tf.constant( 9.0))
  g_out2 = tf_square_if_positive(tf.constant(-9.0))
  with tf.Session() as sess:
    print('Graph results: %2.2f, %2.2f\n' % (sess.run(g_out1), sess.run(g_out2)))
```

AutoGraph supports common Python statements like `while`, `for`, `if`, `break`, and `return`, with support for nesting. Compare this function with the complicated graph verson displayed in the following code blocks:


```
# Continue in a loop
def sum_even(items):
  s = 0
  for c in items:
    if c % 2 > 0:
      continue
    s += c
  return s

print('Eager result: %d' % sum_even(tf.constant([10,12,15,20])))

tf_sum_even = autograph.to_graph(sum_even)

with tf.Graph().as_default(), tf.Session() as sess:
    print('Graph result: %d\n\n' % sess.run(tf_sum_even(tf.constant([10,12,15,20]))))
```


```
print(autograph.to_code(sum_even))
```

## Decorator

If you don't need easy access to the original Python function, use the `convert` decorator:


```
@autograph.convert()
def fizzbuzz(i, n):
  while i < n:
    msg = ''
    if i % 3 == 0:
      msg += 'Fizz'
    if i % 5 == 0:
      msg += 'Buzz'
    if msg == '':
      msg = tf.as_string(i)
    print(msg)
    i += 1
  return i

with tf.Graph().as_default():
  final_i = fizzbuzz(tf.constant(10), tf.constant(16))
  # The result works like a regular op: takes tensors in, returns tensors.
  # You can inspect the graph using tf.get_default_graph().as_graph_def()
  with tf.Session() as sess:
    sess.run(final_i)


```

## Examples

Let's demonstrate some useful Python language features.


### Assert

AutoGraph automatically converts the Python `assert` statement into the equivalent `tf.Assert` code:


```
@autograph.convert()
def inverse(x):
  assert x != 0.0, 'Do not pass zero!'
  return 1.0 / x

with tf.Graph().as_default(), tf.Session() as sess:
  try:
    print(sess.run(inverse(tf.constant(0.0))))
  except tf.errors.InvalidArgumentError as e:
    print('Got error message:\n    %s' % e.message)
```

### Print

Use the Python `print` function in-graph:


```
@autograph.convert()
def count(n):
  i=0
  while i < n:
    print(i)
    i += 1
  return n
    
with tf.Graph().as_default(), tf.Session() as sess:
    sess.run(count(tf.constant(5)))
```

### Lists

Append to lists in loops (tensor list ops are automatically created):


```
@autograph.convert()
def arange(n):
  z = []
  # We ask you to tell us the element dtype of the list
  autograph.set_element_type(z, tf.int32)
  
  for i in tf.range(n):
    z.append(i)
  # when you're done with the list, stack it
  # (this is just like np.stack)
  return autograph.stack(z) 


with tf.Graph().as_default(), tf.Session() as sess:
    sess.run(arange(tf.constant(10)))
```

### Nested control flow


```
@autograph.convert()
def nearest_odd_square(x):
  if x > 0:
    x = x * x
    if x % 2 == 0:
      x = x + 1
  return x

with tf.Graph().as_default():  
  with tf.Session() as sess:
    print(sess.run(nearest_odd_square(tf.constant(4))))
    print(sess.run(nearest_odd_square(tf.constant(5))))
    print(sess.run(nearest_odd_square(tf.constant(6))))
```

### While loop


```
@autograph.convert()
def square_until_stop(x, y):
  while x < y:
    x = x * x
  return x
    
with tf.Graph().as_default():  
  with tf.Session() as sess:
    print(sess.run(square_until_stop(tf.constant(4), tf.constant(100))))
```

### For loop


```
@autograph.convert()
def squares(nums):

  result = []
  autograph.set_element_type(result, tf.int64)

  for num in nums: 
    result.append(num * num)
    
  return autograph.stack(result)
    
with tf.Graph().as_default():  
  with tf.Session() as sess:
    print(sess.run(squares(tf.constant(np.arange(10)))))
```

### Break


```
@autograph.convert()
def argwhere_cumsum(x, threshold):
  current_sum = 0.0
  idx = 0
  for i in tf.range(len(x)):
    idx = i
    if current_sum >= threshold:
      break
    current_sum += x[i]
  return idx

N = 10
with tf.Graph().as_default():  
  with tf.Session() as sess:
    idx = argwhere_cumsum(tf.ones(N), tf.constant(float(N/2)))
    print(sess.run(idx))
```

## Interoperation with `tf.Keras`

Now that you've seen the basics, let's build some model components with autograph.

It's relatively simple to integrate `autograph` with `tf.keras`. 


### Stateless functions

For stateless functions, like `collatz` shown below, the easiest way to include them in a keras model is to wrap them up as a layer using `tf.keras.layers.Lambda`.


```
import numpy as np

@autograph.convert()
def collatz(x):
  x = tf.reshape(x,())
  assert x > 0
  n = tf.convert_to_tensor((0,)) 
  while not tf.equal(x, 1):
    n += 1
    if tf.equal(x%2, 0):
      x = x // 2
    else:
      x = 3 * x + 1
      
  return n

with tf.Graph().as_default():
  model = tf.keras.Sequential([
    tf.keras.layers.Lambda(collatz, input_shape=(1,), output_shape=())
  ])
  
result = model.predict(np.array([6171]))
result
```

### Custom Layers and Models

<!--TODO(markdaoust) link to full examples  or these referenced models.-->

The easiest way to use AutoGraph with Keras layers and models is to `@autograph.convert()` the `call` method. See the [TensorFlow Keras guide](https://tensorflow.org/guide/keras#build_advanced_models) for details on how to build on these classes. 

Here is a simple example of the [stochastic network depth](https://arxiv.org/abs/1603.09382) technique :


```
# `K` is used to check if we're in train or test mode.
K = tf.keras.backend

class StochasticNetworkDepth(tf.keras.Sequential):
  def __init__(self, pfirst=1.0, plast=0.5, *args,**kwargs):
    self.pfirst = pfirst
    self.plast = plast
    super().__init__(*args,**kwargs)
        
  def build(self,input_shape):
    super().build(input_shape.as_list())
    self.depth = len(self.layers)
    self.plims = np.linspace(self.pfirst, self.plast, self.depth + 1)[:-1]
    
  @autograph.convert()
  def call(self, inputs):
    training = tf.cast(K.learning_phase(), dtype=bool)  
    if not training: 
      count = self.depth
      return super(StochasticNetworkDepth, self).call(inputs), count
    
    p = tf.random_uniform((self.depth,))
    
    keeps = (p <= self.plims)
    x = inputs
    
    count = tf.reduce_sum(tf.cast(keeps, tf.int32))
    for i in range(self.depth):
      if keeps[i]:
        x = self.layers[i](x)
      
    # return both the final-layer output and the number of layers executed.
    return x, count
```

Let's try it on mnist-shaped data:


```
train_batch = np.random.randn(64, 28, 28, 1).astype(np.float32)
```

Build a simple stack of `conv` layers, in the stochastic depth model:


```
with tf.Graph().as_default() as g:
  model = StochasticNetworkDepth(
        pfirst=1.0, plast=0.5)

  for n in range(20):
    model.add(
          layers.Conv2D(filters=16, activation=tf.nn.relu,
                        kernel_size=(3, 3), padding='same'))

  model.build(tf.TensorShape((None, None, None, 1)))
  
  init = tf.global_variables_initializer()
```

Now test it to ensure it behaves as expected in train and test modes:


```
# Use an explicit session here so we can set the train/test switch, and
# inspect the layer count returned by `call`
with tf.Session(graph=g) as sess:
  init.run()
 
  for phase, name in enumerate(['test','train']):
    K.set_learning_phase(phase)
    result, count = model(tf.convert_to_tensor(train_batch, dtype=tf.float32))

    result1, count1 = sess.run((result, count))
    result2, count2 = sess.run((result, count))

    delta = (result1 - result2)
    print(name, "sum abs delta: ", abs(delta).mean())
    print("    layers 1st call: ", count1)
    print("    layers 2nd call: ", count2)
    print()
```

## Advanced example: An in-graph training loop

The previous section showed that AutoGraph can be used inside Keras layers and models. Keras models can also be used in AutoGraph code.

Since writing control flow in AutoGraph is easy, running a training loop in a TensorFlow graph should also be easy.  

This example shows how to train a simple Keras model on MNIST with the entire training process—loading batches, calculating gradients, updating parameters, calculating validation accuracy, and repeating until convergence—is performed in-graph.

### Download data


```
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
```

### Define the model


```
def mlp_model(input_shape):
  model = tf.keras.Sequential((
      tf.keras.layers.Dense(100, activation='relu', input_shape=input_shape),
      tf.keras.layers.Dense(100, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')))
  model.build()
  return model


def predict(m, x, y):
  y_p = m(tf.reshape(x, (-1, 28 * 28)))
  losses = tf.keras.losses.categorical_crossentropy(y, y_p)
  l = tf.reduce_mean(losses)
  accuracies = tf.keras.metrics.categorical_accuracy(y, y_p)
  accuracy = tf.reduce_mean(accuracies)
  return l, accuracy


def fit(m, x, y, opt):
  l, accuracy = predict(m, x, y)
  # Autograph automatically adds the necessary `tf.control_dependencies` here.
  # (Without them nothing depends on `opt.minimize`, so it doesn't run.)
  # This makes it much more like eager-code.
  opt.minimize(l)
  return l, accuracy


def setup_mnist_data(is_training, batch_size):
  if is_training:
    ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    ds = ds.shuffle(batch_size * 10)
  else:
    ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

  ds = ds.repeat()
  ds = ds.batch(batch_size)
  return ds


def get_next_batch(ds):
  itr = ds.make_one_shot_iterator()
  image, label = itr.get_next()
  x = tf.to_float(image) / 255.0
  y = tf.one_hot(tf.squeeze(label), 10)
  return x, y 
```

### Define the training loop


```
# Use `recursive = True` to recursively convert functions called by this one.
@autograph.convert(recursive=True)
def train(train_ds, test_ds, hp):
  m = mlp_model((28 * 28,))
  opt = tf.train.AdamOptimizer(hp.learning_rate)
  
  # We'd like to save our losses to a list. In order for AutoGraph
  # to convert these lists into their graph equivalent,
  # we need to specify the element type of the lists.
  train_losses = []
  autograph.set_element_type(train_losses, tf.float32)
  test_losses = []
  autograph.set_element_type(test_losses, tf.float32)
  train_accuracies = []
  autograph.set_element_type(train_accuracies, tf.float32)
  test_accuracies = []
  autograph.set_element_type(test_accuracies, tf.float32)
  
  # This entire training loop will be run in-graph.
  i = tf.constant(0)
  while i < hp.max_steps:
    train_x, train_y = get_next_batch(train_ds)
    test_x, test_y = get_next_batch(test_ds)

    step_train_loss, step_train_accuracy = fit(m, train_x, train_y, opt)
    step_test_loss, step_test_accuracy = predict(m, test_x, test_y)
    if i % (hp.max_steps // 10) == 0:
      print('Step', i, 'train loss:', step_train_loss, 'test loss:',
            step_test_loss, 'train accuracy:', step_train_accuracy,
            'test accuracy:', step_test_accuracy)
    train_losses.append(step_train_loss)
    test_losses.append(step_test_loss)
    train_accuracies.append(step_train_accuracy)
    test_accuracies.append(step_test_accuracy)
    i += 1
  
  # We've recorded our loss values and accuracies 
  # to a list in a graph with AutoGraph's help.
  # In order to return the values as a Tensor, 
  # we need to stack them before returning them.
  return (autograph.stack(train_losses), autograph.stack(test_losses),  
          autograph.stack(train_accuracies), autograph.stack(test_accuracies))
```

Now build the graph and run the training loop:


```
with tf.Graph().as_default() as g:
  hp = tf.contrib.training.HParams(
      learning_rate=0.005,
      max_steps=500,
  )
  train_ds = setup_mnist_data(True, 50)
  test_ds = setup_mnist_data(False, 1000)
  (train_losses, test_losses, train_accuracies,
   test_accuracies) = train(train_ds, test_ds, hp)

  init = tf.global_variables_initializer()
  
with tf.Session(graph=g) as sess:
  sess.run(init)
  (train_losses, test_losses, train_accuracies,
   test_accuracies) = sess.run([train_losses, test_losses, train_accuracies,
                                test_accuracies])
  
plt.title('MNIST train/test losses')
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.xlabel('Training step')
plt.ylabel('Loss')
plt.show()
plt.title('MNIST train/test accuracies')
plt.plot(train_accuracies, label='train accuracy')
plt.plot(test_accuracies, label='test accuracy')
plt.legend(loc='lower right')
plt.xlabel('Training step')
plt.ylabel('Accuracy')
plt.show()
```
