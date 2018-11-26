# 安装 C 语言版本的 TensorFlow

TensorFlow 提供了一个 C 语言的 API 来[绑定其他语言](/docs/tensorflow/extend/language_bindings)。这个 API 定义在
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h">`c_api.h`</a>里，是秉着简单性和一致性设计的，而非便利性。


## 支持的平台

TensorFlow C 语言版本支持以下系统：

* Linux, 64-bit, x86
* macOS X, 10.12.6 (Sierra) 或更高版本


## 设置

### 下载

<table>
  <tr><th>TensorFlow C library</th><th>URL</th></tr>
  <tr class="alt"><td colspan="2">Linux</td></tr>
  <tr>
    <td>Linux CPU only</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.11.0.tar.gz">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.11.0.tar.gz</a></td>
  </tr>
  <tr>
    <td>Linux GPU support</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.11.0.tar.gz">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.11.0.tar.gz</a></td>
  </tr>
  <tr class="alt"><td colspan="2">macOS</td></tr>
  <tr>
    <td>macOS CPU only</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.11.0.tar.gz">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.11.0.tar.gz</a></td>
  </tr>
</table>

### 解压

将下载好的 C 扩张库解压到 `/usr/local/lib` （或者其他你自己喜欢的地方）：

    sudo tar -xz <var>libtensorflow.tar.gz</var> -C /usr/local

### Linker

如果你解压在系统文件夹，比如 `/usr/local`，需要用 `ldconfig`配置链接：

    sudo ldconfig

如果不是的话，比如 `~/mydir`，那你就要配置到环境变量中。

### Linux

    export LIBRARY_PATH=$LIBRARY_PATH:~/mydir/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mydir/lib

### mac OS

    export LIBRARY_PATH=$LIBRARY_PATH:~/mydir/lib
    export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/mydir/lib


## 创建

### 示例程序

安装好之后，创建一个示例程序（`hello_tf.c`）：

```c
#include <stdio.h>
#include <tensorflow/c/c_api.h>

int main() {
  printf("Hello from TensorFlow C library version %s\n", TF_Version());
  return 0;
}
```

### 编译

将以上程序编译成可执行文件，执行：

    gcc hello_tf.c -o hello_tf

    ./hello_tf

执行后会输出： `Hello from TensorFlow C library version <number>`

> 配置成功！

如果配置失败，请确保`gcc` 可以访问到 TensorFlow C 扩展库。如果解压到了 `/usr/local`，那么要指定库的位置：

    gcc -I/usr/local/include -L/usr/local/lib hello_tf.c -ltensorflow -o hello_tf


## 源码安装

TensorFlow 是开源的，请参考[此说明](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/lib_package/README)进行源码安装。
