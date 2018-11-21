#  TensorFlow 指南

本单元中的文档深入介绍了 TensorFlow 的工作原理。这些单元包括：

## 高阶 API

  * [Keras](/docs/tensorflow/guide/keras)，用于构建和训练深度学习模型的 TensorFlow 高阶 API。
  * [Eager Execution](/docs/tensorflow/guide/eager)，一个以命令方式编写 TensorFlow 代码的 API，就像使用 NumPy 一样。
  * [Estimators](/docs/tensorflow/guide/estimators)，一个高阶 API，可以提供已准备好执行大规模训练和生产的完全打包的模型。
  * [导入数据](/docs/tensorflow/guide/datasets)，简单的输入管道，用于将您的数据导入 TensorFlow 程序。

## Estimator

* [Estimator](/docs/tensorflow/guide/estimators)，了解如何将 Estimator 用于机器学习。
* [预创建的 Estimator](/docs/tensorflow/guide/premade_estimators)，预创建的 Estimator 的基础知识。
* [检查点](/docs/tensorflow/guide/checkpoints)，保存训练进度并从您停下的地方继续。
* [特征列](/docs/tensorflow/guide/feature_columns)，在不对模型做出更改的情况下处理各种类型的输入数据。
* [Estimator 的数据集](/docs/tensorflow/guide/datasets_for_estimators)，使用 `tf.data` 输入数据。
* [创建自定义 Estimator](/docs/tensorflow/guide/custom_estimators)，编写自己的 Estimator。

## 加速器

  * [使用 GPU](/docs/tensorflow/guide/using_gpu) - 介绍了 TensorFlow 如何将操作分配给设备，以及如何手动更改此类分配。
  * [使用 TPU](/docs/tensorflow/guide/using_tpu) - 介绍了如何修改 `Estimator` 程序以便在 TPU 上运行。

## 低阶 API

  * [简介](/docs/tensorflow/guide/low_level_intro) - 介绍了如何使用高阶 API 之外的低阶 TensorFlow API 的基础知识。
  * [张量](/docs/tensorflow/guide/tensors) - 介绍了如何创建、操作和访问张量（TensorFlow 中的基本对象）。
  * [变量](/docs/tensorflow/guide/variables) - 详细介绍了如何在程序中表示共享持久状态。
  * [图和会话](/docs/tensorflow/guide/graphs) - 介绍了以下内容：
      * 数据流图：这是 TensorFlow 将计算表示为操作之间的依赖关系的一种表示法。
      * 会话：TensorFlow 跨一个或多个本地或远程设备运行数据流图的机制。如果您使用低阶 TensorFlow API 编程，请务必阅读并理解本单元的内容。如果您使用高阶 TensorFlow API（例如 Estimator 或 Keras）编程，则高阶 API 会为您创建和管理图和会话，但是理解图和会话依然对您有所帮助。 
  * [保存和恢复](/docs/tensorflow/guide/saved_model) - 介绍了如何保存和恢复变量及模型。

## 机器学习概念

  * [嵌入](/docs/tensorflow/guide/embedding) - 介绍了“嵌入”这一概念，并且举了一个简单的例子来说明如何在 TensorFlow 中训练嵌入，此外还说明了如何使用 TensorBoard Embedding Projector 查看嵌入。

## 调试

  * [TensorFlow 调试程序](/docs/tensorflow/guide/debugger) - 介绍了如何使用 TensorFlow 调试程序 (tfdbg)。

## TensorBoard

TensorBoard 是一款实用工具，能够直观地展示机器学习的各个不同方面。以下指南介绍了如何使用 TensorBoard：

  * [TensorBoard：可视化学习过程 ](/docs/tensorflow/guide/summaries_and_tensorboard) - 介绍了 TensorBoard。
  * [TensorBoard：图的可视化](/docs/tensorflow/guide/graph_viz) - 介绍了如何可视化计算图。
  * [TensorBoard 直方图信息中心](/docs/tensorflow/guide/tensorboard_histograms) - 演示了如何使用 TensorBoard 的直方图信息中心。


## 其他

  * [TensorFlow 版本兼容性](/docs/tensorflow/guide/version_compat) - 介绍了向后兼容性保证及无保证内容。
  * [常见问题解答](/docs/tensorflow/guide/faq) - 包含关于 TensorFlow 的常见问题解答。
