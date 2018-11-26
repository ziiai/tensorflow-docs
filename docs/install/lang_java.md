# 安装 Java 语言的 TensorFlow

TensorFlow 可提供在 Java 程序中使用的 API。这些 API 特别适合用于加载以 Python 语言创建的模型并在 Java 应用中运行这些模型。本指南将介绍如何安装
[Java API](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary)
并在 Java 应用中使用 TensorFlow。

警告：TensorFlow Go API 不在 TensorFlow [API 稳定性保障](/docs/tensorflow/guide/version_compat)的涵盖范围内。


## 支持的平台

本指南介绍如何安装适用于 Java 的 TensorFlow。虽然这些说明可能也适用于其他配置，但我们只在满足以下要求的计算机上验证过这些说明（而且我们只支持在此类计算机上按这些说明操作）：

* Ubuntu 16.04 或更高版本； 64-bit, x86
* macOS 10.12.6 (Sierra) 或更高版本
* Windows 7 or higher;或更高版本； 64-bit, x86

针对 Android 的安装说明位于单独的 [Android TensorFlow 支持](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android)页面中。安装完成后，请查看这个适用于 Android 的[完整 TensorFlow 示例](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android)。


## 在 Maven 项目中使用 TensorFlow

如果您的项目使用 [Apache Maven](https://maven.apache.org)，请将以下内容添加到项目的 `pom.xml` 以使用 TensorFlow Java API：

```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>tensorflow</artifactId>
  <version>1.11.0</version>
</dependency>
```

### GPU 支持

如果您的 Linux 系统搭载了 NVIDIA® GPU 且您的 TensorFlow Java 程序需要 GPU 加速，请将以下内容添加到项目的 `pom.xml`：

```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>libtensorflow</artifactId>
  <version>1.11.0</version>
</dependency>
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>libtensorflow_jni_gpu</artifactId>
  <version>1.11.0</version>
</dependency>
```

### 示例程序 

首先在项目的 `pom.xml` 文件添加 TensorFlow 依赖：

```xml
<project>
  <modelVersion>4.0.0</modelVersion>
  <groupId>org.myorg</groupId>
  <artifactId>hellotensorflow</artifactId>
  <version>1.0-SNAPSHOT</version>
  <properties>
    <exec.mainClass>HelloTensorFlow</exec.mainClass>
	<!-- The sample code requires at least JDK 1.7. -->
	<!-- The maven compiler plugin defaults to a lower version -->
	<maven.compiler.source>1.7</maven.compiler.source>
	<maven.compiler.target>1.7</maven.compiler.target>
  </properties>
  <dependencies>
    <dependency>
	  <groupId>org.tensorflow</groupId>
	  <artifactId>tensorflow</artifactId>
	  <version>1.11.0</version>
	</dependency>
  </dependencies>
</project>
```

创建源文件 （`src/main/java/HelloTensorFlow.java`）：

```java
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

public class HelloTensorFlow {
  public static void main(String[] args) throws Exception {
	try (Graph g = new Graph()) {
	  final String value = "Hello from " + TensorFlow.version();

	  // Construct the computation graph with a single operation, a constant
	  // named "MyConst" with a value "value".
	  try (Tensor t = Tensor.create(value.getBytes("UTF-8"))) {
	    // The Java API doesn't yet include convenience functions for adding operations.
		g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();
	  }

	  // Execute the "MyConst" operation in a Session.
	  try (Session s = new Session(g);
	      // Generally, there may be multiple output tensors,
		  // all of them must be closed to prevent resource leaks.
		  Tensor output = s.runner().fetch("MyConst").run().get(0)) {
	    System.out.println(new String(output.bytesValue(), "UTF-8"));
	  }
    }
  }
}
```

编译和执行：

    mvn -q compile exec:java  # Use -q to hide logging

输出： `Hello from <version>`

成功！


## 在 JDK 中使用 TensorFlow

TensorFlow can be used with the JDK through the Java Native Interface (JNI).

### 下载

1. 下载 JAR 文件： [libtensorflow.jar](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-1.11.0.jar)
2. 下载并解压匹配系统和处理器的 Java Native Interface (JNI) 文件：

<table>
  <tr><th>JNI version</th><th>URL</th></tr>
  <tr class="alt"><td colspan="2">Linux</td></tr>
  <tr>
    <td>Linux CPU only</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.11.0.tar.gz">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.11.0.tar.gz</a></td>
  </tr>
  <tr>
    <td>Linux GPU support</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.11.0.tar.gz">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.11.0.tar.gz</a></td>
  </tr>
  <tr class="alt"><td colspan="2">macOS</td></tr>
  <tr>
    <td>macOS CPU only</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.11.0.tar.gz">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.11.0.tar.gz</a></td>
  </tr>
  <tr class="alt"><td colspan="2">Windows</td></tr>
  <tr>
    <td>Windows CPU only</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.11.0.zip">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.11.0.zip</a></td>
  </tr>
  <tr>
    <td>Windows GPU support</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-windows-x86_64-1.11.0.zip">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-windows-x86_64-1.11.0.zip</a></td>
  </tr>
</table>

注意：在 Windows 中，本地库 （`tensorflow_jni.dll`）在运行时需要 `msvcp140.dll` 文件。 参见 [Windows 源码安装](/docs/tensorflow/install/source_windows)。


### 编译

编译使用 TensorFlow 的 Java 程序时，下载的 `.jar` 必须是 `classpath` 的一部分。例如，您可以使用 `-cp` 编译标志将下载的 `.jar` 添加到 `classpath` 中，如下所示：

    javac -cp libtensorflow-1.11.0.jar HelloTensorFlow.java

### 执行

要运行依赖 TensorFlow 的 Java 程序，请确保 JVM 能访问以下两个文件：

- 下载的 `.jar` 文件
- 提取的 JNI 库

### Linux / mac OS

    java -cp libtensorflow-1.11.0.jar:. -Djava.library.path=./jni HelloTensorFlow

### Windows
    java -cp libtensorflow-1.11.0.jar;. -Djava.library.path=jni HelloTensorFlow

输出： `Hello from <version>`

成功！


## 源码安装

TensorFlow 是开源的，请参考[此说明](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/java/README)进行源码安装。
