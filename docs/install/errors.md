# 编译和安装问题汇总

TensorFlow 使用 [GitHub issues](https://github.com/tensorflow/tensorflow/issues)
和 [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow)
追踪和记录用户在使用中遇到的问题。

以下是问题信息的解决方案或者相关讨论。如果问题还没解决你可以去以上平台搜索，如果还没有的话你就直接提问吧。

<table>
<tr><th>GitHub issue or Stack Overflow</th> <th>Error Message</th></tr>
<tr>
  <td><a href="https://stackoverflow.com/q/36159194">36159194</a></td>
  <td><code>ImportError: libcudart.so.<i>Version</i>: cannot open shared object file:
  No such file or directory</code></td>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/q/41991101">41991101</a></td>
  <td><code>ImportError: libcudnn.<i>Version</i>: cannot open shared object file:
  No such file or directory</code></td>
</tr>
<tr>
  <td><a href="http://stackoverflow.com/q/36371137">36371137</a> and
  <a href="#Protobuf31">here</a></td>
  <td><code>libprotobuf ERROR google/protobuf/src/google/protobuf/io/coded_stream.cc:207] A
  protocol message was rejected because it was too big (more than 67108864 bytes).
  To increase the limit (or to disable these warnings), see
  CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.</code></td>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/q/35252888">35252888</a></td>
  <td><code>Error importing tensorflow. Unless you are using bazel, you should
  not try to import tensorflow from its source directory; please exit the
  tensorflow source tree, and relaunch your python interpreter from
  there.</code></td>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/q/33623453">33623453</a></td>
  <td><code>IOError: [Errno 2] No such file or directory:
  '/tmp/pip-o6Tpui-build/setup.py'</tt></code>
</tr>
<tr>
  <td><a href="http://stackoverflow.com/q/42006320">42006320</a></td>
  <td><code>ImportError: Traceback (most recent call last):
  File ".../tensorflow/core/framework/graph_pb2.py", line 6, in <module>
  from google.protobuf import descriptor as _descriptor
  ImportError: cannot import name 'descriptor'</code>
  </td>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/questions/35190574">35190574</a> </td>
  <td><code>SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify
  failed</code></td>
</tr>
<tr>
  <td><a href="http://stackoverflow.com/q/42009190">42009190</a></td>
  <td><code>
  Installing collected packages: setuptools, protobuf, wheel, numpy, tensorflow
  Found existing installation: setuptools 1.1.6
  Uninstalling setuptools-1.1.6:
  Exception:
  ...
  [Errno 1] Operation not permitted:
  '/tmp/pip-a1DXRT-uninstall/.../lib/python/_markerlib' </code></td>
</tr>
<tr>
  <td><a href="http://stackoverflow.com/questions/36933958">36933958</a></td>
  <td><code>
  ...
  Installing collected packages: setuptools, protobuf, wheel, numpy, tensorflow
  Found existing installation: setuptools 1.1.6
  Uninstalling setuptools-1.1.6:
  Exception:
  ...
  [Errno 1] Operation not permitted:
  '/tmp/pip-a1DXRT-uninstall/System/Library/Frameworks/Python.framework/
   Versions/2.7/Extras/lib/python/_markerlib'</code>
  </td>
</tr>
<tr>
  <td><a href="http://stackoverflow.com/q/42006320">42006320</a></td>
  <td><code>ImportError: Traceback (most recent call last):
File ".../tensorflow/core/framework/graph_pb2.py", line 6, in <module>
from google.protobuf import descriptor as _descriptor
ImportError: cannot import name 'descriptor'</code>
  </td>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/q/33623453">33623453</a></td>
  <td><code>IOError: [Errno 2] No such file or directory:
  '/tmp/pip-o6Tpui-build/setup.py'</tt></code>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/questions/35190574">35190574</a> </td>
  <td><code>SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify
  failed</code></td>
</tr>
<tr>
  <td><a href="http://stackoverflow.com/q/42009190">42009190</a></td>
  <td><code>
  Installing collected packages: setuptools, protobuf, wheel, numpy, tensorflow
  Found existing installation: setuptools 1.1.6
  Uninstalling setuptools-1.1.6:
  Exception:
  ...
  [Errno 1] Operation not permitted:
  '/tmp/pip-a1DXRT-uninstall/.../lib/python/_markerlib' </code></td>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/q/33622019">33622019</a></td>
  <td><code>ImportError: No module named copyreg</code></td>
</tr>
<tr>
  <td><a href="http://stackoverflow.com/q/37810228">37810228</a></td>
  <td>During a <tt>pip install</tt> operation, the system returns:
  <code>OSError: [Errno 1] Operation not permitted</code>
  </td>
</tr>
<tr>
  <td><a href="http://stackoverflow.com/q/33622842">33622842</a></td>
  <td>An <tt>import tensorflow</tt> statement triggers an error such as the
  following:<code>Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python2.7/site-packages/tensorflow/__init__.py",
    line 4, in <module>
    from tensorflow.python import *
    ...
  File "/usr/local/lib/python2.7/site-packages/tensorflow/core/framework/tensor_shape_pb2.py",
    line 22, in <module>
    serialized_pb=_b('\n,tensorflow/core/framework/tensor_shape.proto\x12\ntensorflow\"d\n\x10TensorShapeProto\x12-\n\x03\x64im\x18\x02
      \x03(\x0b\x32
      .tensorflow.TensorShapeProto.Dim\x1a!\n\x03\x44im\x12\x0c\n\x04size\x18\x01
      \x01(\x03\x12\x0c\n\x04name\x18\x02 \x01(\tb\x06proto3')
  TypeError: __init__() got an unexpected keyword argument 'syntax'</code>
  </td>
</tr>
<tr>
  <td><a href="http://stackoverflow.com/q/42075397">42075397</a></td>
  <td>A <tt>pip install</tt> command triggers the following error:
<code>...<lots of warnings and errors>
You have not agreed to the Xcode license agreements, please run
'xcodebuild -license' (for user-level acceptance) or
'sudo xcodebuild -license' (for system-wide acceptance) from within a
Terminal window to review and agree to the Xcode license agreements.
...<more stack trace output>
  File "numpy/core/setup.py", line 653, in get_mathlib_info

    raise RuntimeError("Broken toolchain: cannot link a simple C program")

RuntimeError: Broken toolchain: cannot link a simple C program</code>
  </td>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/q/41007279">41007279</a></td>
  <td>
  <code>[...\stream_executor\dso_loader.cc] Couldn't open CUDA library nvcuda.dll</code>
  </td>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/q/41007279">41007279</a></td>
  <td>
  <code>[...\stream_executor\cuda\cuda_dnn.cc] Unable to load cuDNN DSO</code>
  </td>
</tr>
<tr>
  <td><a href="http://stackoverflow.com/q/42006320">42006320</a></td>
  <td><code>ImportError: Traceback (most recent call last):
File "...\tensorflow\core\framework\graph_pb2.py", line 6, in <module>
from google.protobuf import descriptor as _descriptor
ImportError: cannot import name 'descriptor'</code>
  </td>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/q/42011070">42011070</a></td>
  <td><code>No module named "pywrap_tensorflow"</code></td>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/q/42217532">42217532</a></td>
  <td>
  <code>OpKernel ('op: "BestSplits" device_type: "CPU"') for unknown op: BestSplits</code>
  </td>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/q/43134753">43134753</a></td>
  <td>
  <code>The TensorFlow library wasn't compiled to use SSE instructions</code>
  </td>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/q/38896424">38896424</a></td>
  <td>
  <code>Could not find a version that satisfies the requirement tensorflow</code>
  </td>
</tr>
<tr>
  <td><a href="http://stackoverflow.com/q/42006320">42006320</a></td>
  <td><code>ImportError: Traceback (most recent call last):
File ".../tensorflow/core/framework/graph_pb2.py", line 6, in <module>
from google.protobuf import descriptor as _descriptor
ImportError: cannot import name 'descriptor'</code>
  </td>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/q/33623453">33623453</a></td>
  <td><code>IOError: [Errno 2] No such file or directory:
  '/tmp/pip-o6Tpui-build/setup.py'</tt></code>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/questions/35190574">35190574</a> </td>
  <td><code>SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify
  failed</code></td>
</tr>
<tr>
  <td><a href="http://stackoverflow.com/q/42009190">42009190</a></td>
  <td><code>
  Installing collected packages: setuptools, protobuf, wheel, numpy, tensorflow
  Found existing installation: setuptools 1.1.6
  Uninstalling setuptools-1.1.6:
  Exception:
  ...
  [Errno 1] Operation not permitted:
  '/tmp/pip-a1DXRT-uninstall/.../lib/python/_markerlib' </code></td>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/q/33622019">33622019</a></td>
  <td><code>ImportError: No module named copyreg</code></td>
</tr>
<tr>
  <td><a href="http://stackoverflow.com/q/37810228">37810228</a></td>
  <td>During a <tt>pip install</tt> operation, the system returns:
  <code>OSError: [Errno 1] Operation not permitted</code>
  </td>
</tr>
<tr>
  <td><a href="http://stackoverflow.com/q/33622842">33622842</a></td>
  <td>An <tt>import tensorflow</tt> statement triggers an error such as the
  following:<code>Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python2.7/site-packages/tensorflow/__init__.py",
    line 4, in <module>
    from tensorflow.python import *
    ...
  File "/usr/local/lib/python2.7/site-packages/tensorflow/core/framework/tensor_shape_pb2.py",
    line 22, in <module>
    serialized_pb=_b('\n,tensorflow/core/framework/tensor_shape.proto\x12\ntensorflow\"d\n\x10TensorShapeProto\x12-\n\x03\x64im\x18\x02
      \x03(\x0b\x32
      .tensorflow.TensorShapeProto.Dim\x1a!\n\x03\x44im\x12\x0c\n\x04size\x18\x01
      \x01(\x03\x12\x0c\n\x04name\x18\x02 \x01(\tb\x06proto3')
  TypeError: __init__() got an unexpected keyword argument 'syntax'</code>
  </td>
</tr>
<tr>
  <td><a
  href="https://stackoverflow.com/questions/41293077/how-to-compile-tensorflow-with-sse4-2-and-avx-instructions">41293077</a></td>
  <td><code>W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow
  library wasn't compiled to use SSE4.1 instructions, but these are available on
  your machine and could speed up CPU computations.</code></td>
</tr>
<tr>
  <td><a href="http://stackoverflow.com/q/42013316">42013316</a></td>
  <td><code>ImportError: libcudart.so.8.0: cannot open shared object file:
  No such file or directory</code></td>
</tr>
<tr>
  <td><a href="http://stackoverflow.com/q/42013316">42013316</a></td>
  <td><code>ImportError: libcudnn.5: cannot open shared object file:
  No such file or directory</code></td>
</tr>
<tr>
  <td><a href="http://stackoverflow.com/q/35953210">35953210</a></td>
  <td>Invoking `python` or `ipython` generates the following error:
  <code>ImportError: cannot import name pywrap_tensorflow</code></td>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/questions/45276830">45276830</a></td>
  <td><code>external/local_config_cc/BUILD:50:5: in apple_cc_toolchain rule
  @local_config_cc//:cc-compiler-darwin_x86_64: Xcode version must be specified
  to use an Apple CROSSTOOL.</code>
  </td>
</tr>
<tr>
  <td><a href="https://stackoverflow.com/q/47080760">47080760</a></td>
  <td><code>undefined reference to `cublasGemmEx@libcublas.so.9.0'</code></td>
</tr>
<tr>
  <td><a href="https://github.com/tensorflow/tensorflow/issues/22512">22512</a></td>
  <td><code>ModuleNotFoundError: No module named 'tensorflow.python._pywrap_tensorflow_internal'</code></td>
</tr>
<tr>
  <td><a href="https://github.com/tensorflow/tensorflow/issues/22512">22512</a>, <a href="https://github.com/tensorflow/tensorflow/issues/22794">22794</a></td>
  <td><code>ImportError: DLL load failed: The specified module could not be found.</code></td>
</tr>
</table>
