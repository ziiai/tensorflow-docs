# 安装 Go 语言的 TensorFlow

TensorFlow 提供在 Go 程序中使用的 API。这些 API 特别适合用于加载以 Python 语言创建的模型并在 Go 应用中运行这些模型。本指南将介绍如何安装和设置
[Go API](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go)。

警告：TensorFlow Go API 不在 TensorFlow [API 稳定性保障](/docs/tensorflow/guide/version_compat)的涵盖范围内。


## 支持的平台

本指南介绍如何安装适用于 Go 的 TensorFlow。虽然这些说明可能也适用于其他配置，但我们只在满足以下要求的计算机上验证过这些说明（而且我们只支持在此类计算机上按这些说明操作）：

* Linux, 64-bit, x86
* macOS X, 10.12.6 (Sierra) 或更高版本


## 设置

### TensorFlow C 库

需要先安装好 [TensorFlow C 语言库](/docs/tensorflow/install/lang_c)。

### 下载

安装 TensorFlow C 库后，按以下方式调用 go get 以下载正确的软件包及其依赖项：

    go get github.com/tensorflow/tensorflow/tensorflow/go

按照以下方式调用 go test 以验证适用于 Go 的 TensorFlow 安装结果：

    go test github.com/tensorflow/tensorflow/tensorflow/go


## 创建

### 示例程序 

创建 `hello_tf.go` 文件：

```
package main

import (
    tf "github.com/tensorflow/tensorflow/tensorflow/go"
    "github.com/tensorflow/tensorflow/tensorflow/go/op"
    "fmt"
)

func main() {
    // Construct a graph with an operation that produces a string constant.
    s := op.NewScope()
    c := op.Const(s, "Hello from TensorFlow version " + tf.Version())
    graph, err := s.Finalize()
    if err != nil {
        panic(err)
    }

    // Execute the graph in a session.
    sess, err := tf.NewSession(graph, nil)
    if err != nil {
        panic(err)
    }
    output, err := sess.Run(nil, []tf.Output{c}, nil)
    if err != nil {
        panic(err)
    }
    fmt.Println(output[0].Value())
}
```

### 运行

执行：

    go run hello_tf.go

程序输出：`Hello from TensorFlow version <number>`

成功！

程序也可能会生成以下形式的多条警告消息，您可以忽略：

    W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library
    wasn't compiled to use *Type* instructions, but these are available on your
    machine and could speed up CPU computations.

## 源码安装

TensorFlow 是开源的，请参考[此说明](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/README)进行源码安装。
