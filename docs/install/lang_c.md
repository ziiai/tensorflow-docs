# Install TensorFlow for C

TensorFlow provides a C API that can be used to build
[bindings for other languages](/docs/tensorflow/extend/language_bindings). The API is defined in
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h">`c_api.h`</a>
and designed for simplicity and uniformity rather than convenience.


## Supported Platforms

TensorFlow for C is supported on the following systems:

* Linux, 64-bit, x86
* macOS X, Version 10.12.6 (Sierra) or higher


## Setup

### Download

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

### Extract

Extract the downloaded TensorFlow C library to `/usr/local/lib` (or use a
different directory, if preferred):

    sudo tar -xz <var>libtensorflow.tar.gz</var> -C /usr/local

### Linker

If you extract the TensorFlow C library to a system directory, such as
`/usr/local`, configure the linker with `ldconfig`:

    sudo ldconfig

Or, if you extract the TensorFlow C library to a non-system directory, such as
`~/mydir`, then configure the linker environmental variables:

### Linux

    export LIBRARY_PATH=$LIBRARY_PATH:~/mydir/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mydir/lib

### mac OS

    export LIBRARY_PATH=$LIBRARY_PATH:~/mydir/lib
    export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/mydir/lib


## Build

### Example program

With the TensorFlow C library installed, create an example program with the
following source code (`hello_tf.c`):

```c
#include <stdio.h>
#include <tensorflow/c/c_api.h>

int main() {
  printf("Hello from TensorFlow C library version %s\n", TF_Version());
  return 0;
}
```

### Compile

Compile the example program to create an executable, then run:

    gcc hello_tf.c -o hello_tf</code>

    ./hello_tf</code>

The command outputs: `Hello from TensorFlow C library version <number>`

Success: The TensorFlow C library is configured.

If the program doesn't build, make sure that `gcc` can access the TensorFlow C
library. If extracted to `/usr/local`, explictly pass the library location to
the compiler:

    gcc -I/usr/local/include -L/usr/local/lib hello_tf.c -ltensorflow -o hello_tf


## Build from source

TensorFlow is open source. Read
[the instructions](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/lib_package/README){:.external}
to build TensorFlow's C library from source code.
