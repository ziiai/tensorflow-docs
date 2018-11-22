#  Estimator

本文档介绍了 Estimator - 一种可极大地简化机器学习编程的高阶 TensorFlow API。Estimator 会封装下列操作：

- 训练
- 评估
- 预测
- 导出以供使用

您可以使用我们提供的预创建的 Estimator，也可以编写自定义 Estimator。所有 Estimator（无论是预创建的还是自定义）都是基于 `tf.estimator.Estimator` 类的类。

> 注意：TensorFlow 还包含一个已弃用的 Estimator 类 `tf.contrib.learn.Estimator`，您不应该使用此类。


## Estimator 的优势

Estimator 具有下列优势：

- 您可以在本地主机上或分布式多服务器环境中运行基于 Estimator 的模型，而无需更改模型。此外，您可以在 CPU、GPU 或 TPU 上运行基于 Estimator 的模型，而无需重新编码模型。
- Estimator 简化了在模型开发者之间共享实现的过程。
- 您可以使用高级直观代码开发先进的模型。简言之，采用 Estimator 创建模型通常比采用低阶 TensorFlow API 更简单。
- Estimator 本身在 `tf.layers` 之上构建而成，可以简化自定义过程。
- Estimator 会为您构建图。
- Estimator 提供安全的分布式训练循环，可以控制如何以及何时：
    - 构建图
    - 初始化变量
    - 开始排队
    - 处理异常
    - 创建检查点文件并从故障中恢复
    - 保存 TensorBoard 的摘要

使用 Estimator 编写应用时，您必须将数据输入管道从模型中分离出来。这种分离简化了不同数据集的实验流程。


## 预创建的 Estimator

借助预创建的 Estimator，您能够在比基本 TensorFlow API 高级很多的概念层面上进行操作。由于 Estimator 会为您处理所有“管道工作”，因此您不必再为创建计算图或会话而操心。也就是说，预创建的 Estimator 会为您创建和管理 `Graph` 和 `Session` 对象。此外，借助预创建的 Estimator，您只需稍微更改下代码，就可以尝试不同的模型架构。例如，`DNNClassifier` 是一个预创建的 `Estimator` 类，它根据密集的前馈神经网络训练分类模型。


### 预创建的 Estimator 程序的结构

依赖预创建的 Estimator 的 TensorFlow 程序通常包含下列四个步骤：

1.  编写一个或多个数据集导入函数。 例如，您可以创建一个函数来导入训练集，并创建另一个函数来导入测试集。每个数据集导入函数都必须返回两个对象：

    *   一个字典，其中键是特征名称，值是包含相应特征数据的张量（或 SparseTensor）
    *   一个包含一个或多个标签的张量

    例如，以下代码展示了输入函数的基本框架：

        def input_fn(dataset):
           ...  # manipulate dataset, extracting the feature dict and the label
           return feature_dict, label

2.  定义特征列。 每个 `tf.feature_column` 都标识了特征名称、特征类型和任何输入预处理操作。例如，以下代码段创建了三个存储整数或浮点数据的特征列。前两个特征列仅标识了特征的名称和类型。第三个特征列还指定了一个 lambda，该程序将调用此 lambda 来调节原始数据：

        # Define three numeric feature columns.
        population = tf.feature_column.numeric_column('population')
        crime_rate = tf.feature_column.numeric_column('crime_rate')
        median_education = tf.feature_column.numeric_column('median_education',
                            normalizer_fn=lambda x: x - global_education_mean)

3.  实例化相关的预创建的 Estimator。 例如，下面是对名为 `LinearClassifier` 的预创建 Estimator 进行实例化的示例代码：

        # Instantiate an estimator, passing the feature columns.
        estimator = tf.estimator.LinearClassifier(
            feature_columns=[population, crime_rate, median_education],
            )

4.  调用训练、评估或推理方法。例如，所有 Estimator 都提供训练模型的 `train` 方法。

        # my_training_set is the function created in Step 1
        estimator.train(input_fn=my_training_set, steps=2000)


### 预创建的 Estimator 的优势

预创建的 Estimator 会编码最佳做法，从而具有下列优势：

*   确定计算图不同部分的运行位置以及在单台机器或多台机器上实现策略的最佳做法。
*   事件（汇总）编写和普遍有用的汇总的最佳做法。

如果您不使用预创建的 Estimator，则必须自行实现上述功能。


## 自定义 Estimator

每个 Estimator（无论是预创建还是自定义）的核心都是其模型函数，这是一种为训练、评估和预测构建图的方法。如果您使用预创建的 Estimator，则有人已经实现了模型函数。如果您使用自定义 Estimator，则必须自行编写模型函数。
[随附文档](/docs/tensorflow/guide/custom_estimators)
介绍了如何编写模型函数。


## 推荐的工作流程

我们推荐以下工作流程：

1. 假设存在合适的预创建的 Estimator，使用它构建第一个模型并使用其结果确定基准。
1. 使用此预创建的 Estimator 构建和测试整体管道，包括数据的完整性和可靠性。
1. 如果存在其他合适的预创建的 Estimator，则运行实验来确定哪个预创建的 Estimator 效果最好。
1. 可以通过构建自定义 Estimator 进一步改进模型。


## 从 Keras 模型创建 Estimator

您可以将现有的 Keras 模型转换为 Estimator。这样做之后，Keras 模型就可以利用 Estimator 的优势，例如分布式训练。调用 `tf.keras.estimator.model_to_estimator`，如下例所示：

```python
# Instantiate a Keras inception v3 model.
keras_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None)
# Compile model with the optimizer, loss, and metrics you'd like to train with.
keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy',
                          metric='accuracy')
# Create an Estimator from the compiled Keras model. Note the initial model
# state of the keras model is preserved in the created Estimator.
est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model=keras_inception_v3)

# Treat the derived Estimator as you would with any other Estimator.
# First, recover the input name(s) of Keras model, so we can use them as the
# feature column name(s) of the Estimator input function:
keras_inception_v3.input_names  # print out: ['input_1']
# Once we have the input name(s), we can create the input function, for example,
# for input(s) in the format of numpy ndarray:
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input_1": train_data},
    y=train_labels,
    num_epochs=1,
    shuffle=False)
# To train, we call Estimator's train function:
est_inception_v3.train(input_fn=train_input_fn, steps=2000)
```
请注意，Keras Estimator 的特征列名称和标签来自经过编译的对应 Keras 模型。例如，上面的 `train_input_fn` 的输入键名称可以从 `keras_inception_v3.input_names` 获得；同样，预测的输出名称可以从 `keras_inception_v3.output_names` 获得。

要了解详情，请参阅 `tf.keras.estimator.model_to_estimator` 的文档。
