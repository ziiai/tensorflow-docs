# Mandelbrot 集合

可视化 [Mandelbrot 集合](https://en.wikipedia.org/wiki/Mandelbrot_set)
与机器学习没有任何关系，但它有趣地演示了如何将 TensorFlow 应用于普通数学。这实际上是一个很基础的可视化实现，但它也点到实处。（我们最终可能会在后期提供更详细的实现来生成更多精美的图片。）


## 基本设置

首先，我们需要导入一些库。

```python
# Import libraries for simulation
import tensorflow as tf
import numpy as np

# Imports for visualization
import PIL.Image
from io import BytesIO
from IPython.display import Image, display
```

现在，我们将定义一个函数，以便在获得迭代计数后实际显示图像。

```python
def DisplayFractal(a, fmt='jpeg'):
  """Display an array of iteration counts as a
     colorful picture of a fractal."""
  a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
  img = np.concatenate([10+20*np.cos(a_cyclic),
                        30+50*np.sin(a_cyclic),
                        155-80*np.cos(a_cyclic)], 2)
  img[a==a.max()] = 0
  a = img
  a = np.uint8(np.clip(a, 0, 255))
  f = BytesIO()
  PIL.Image.fromarray(a).save(f, fmt)
  display(Image(data=f.getvalue()))
```

## 会话和变量初始化

对于此类操作，我们通常使用交互式会话，但常规会话一样可行。

```python
sess = tf.InteractiveSession()
```

交互式会话便于我们自由混用 NumPy 和 TensorFlow。

```python
# Use NumPy to create a 2D array of complex numbers

Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Z = X+1j*Y
```

现在，我们定义和初始化 TensorFlow 张量。

```python
xs = tf.constant(Z.astype(np.complex64))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, tf.float32))
```

TensorFlow 要求您在使用变量之前明确初始化变量。

```python
tf.global_variables_initializer().run()
```

## 定义和运行计算

现在，我们指定计算的其他方面……


```python
# Compute the new values of z: z^2 + x
zs_ = zs*zs + xs

# Have we diverged with this new value?
not_diverged = tf.abs(zs_) < 4

# Operation to update the zs and the iteration count.
#
# Note: We keep computing zs after they diverge! This
#       is very wasteful! There are better, if a little
#       less simple, ways to do this.
#
step = tf.group(
  zs.assign(zs_),
  ns.assign_add(tf.cast(not_diverged, tf.float32))
  )
```

……然后运行几百步

```python
for i in range(200): step.run()
```

我们看看结果是什么。

```python
DisplayFractal(ns.eval())
```

![jpeg](https://www.tensorflow.org/images/mandelbrot_output.jpg)

还行！
