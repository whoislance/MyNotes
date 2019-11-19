[TOC]

# 第一章 深度学习简介

- 深度学习和传统机器学习的区别和关系

  将传统的【人工特征提取】替换为【基础特征提取】+【多层复杂特征提取】

  $深度学习  \in 机器学习 \in 人工智能$

  ​

- 神经网络发展历史

  - 模仿神经元结构

    1943：神经网络数学模型

    1958：感知机preceptron模型

  - 反向传播算法、分布式知识表达

    1979：分布式知识表达

    1986：反向传播算法 back propagation

    1989：卷积神经网络，循环神经网络

    1991：LSTM模型(long short-term memory)

    1998：支持向量机SVM

  - 云计算、GPU

    2012：深度学习deep learning

    ​

- 深度学习应用

  - 计算机视觉

    OCR：光学字符识别 optical character recognition

  - 语音识别

  - 自然语言处理

  - 人机博弈

    蒙特卡罗树搜索MCTS：Monte Carlo tree search 

    ​

- 深度学习工具

  | 工具         | 维护人员     | 语言              |
  | ------------ | ------------ | ----------------- |
  | Caffe        | 伯克利       | C++,PYTHON,MATLAB |
  | CNTK         | 微软         | PYTHON  C++       |
  | TensorFlow   | 谷歌         | C++,PYTHON        |
  | Theano       | 蒙特卡尔大学 | PYTHON            |
  | Torch        | 脸书         | C，Lua            |
  | PaddlePaddle | 百度         | C++,PYTHON        |




# 第二章 TensorFlow环境搭建

- 依赖包

  - **Protocol Buffer**： 处理结构化数据的工具

    序列化后得到的是二进制流（不是可读的字符串）；

    使用时先定义数据的格式(schema)，还原数据需要使用该格式，定义格式的文件保存在.proto文件中；

    message中定义每一个属性的类型的名字，类型：布尔型，整数型，实数型，字符型

  - **Bazel**：自动化构建工具

    py_binary：编译为可执行文件

    py_library：编译python测试程序

    py_test：将python程序编译成库函数共其他py_binary和py_test调用

- 测试样例

  ```python
  >>> import tensorflow as tf
  >>> a = tf.constant([1,2],name="a")
  >>> b = tf.constant([1,2],name="b")
  >>> result = a + b
  >>> sess = tf.Session()
  >>> sess.run(result)
  array([2, 4])
  ```

  ​


# 第三章 TensorFlow入门

### 计算模型：计算图

获取当前默认的计算图：`tf.get_default_graph()`

生成新的计算图：`tf.Graph()`

指定运算设备：`tf.Graph().device('/gpu:0')`

张量所属的计算图：`a.graph`

- 集合collection

  可通过`tf.add_to_collection`将资源加入集合中，再通过`tf.get_collection`获取一个集合中所有资源。

### 数据模型：张量

- tensor概念

  零阶张量：标量 scalar

  一阶张量：向量 vector

  n阶张量：n维数组

  张量不保存数字，而是保存得到数字的计算过程

- 张量的属性

  - 名字 name

    形式：`node:src_output`，node是节点名称，src_output表示来自节点的第几个输出

  - 维度 shape

  - 类型 type

    - 实数：tf.float32, tf.float64
    - 整数：tf.int8, tf.int16, tf.int32, tf.int64, tf.unint8
    - 布尔型：tf.bool
    - 复数：tf.complex64, tf.complex128

- 张量的操作

  计算一个张量的取值：`tf.Tensor.eval()`

  `sess.run(result)` 等价于 `result.eval(session=sess)`

### 运行模型：会话

会话拥有并管理TensorFlow运行时的所有资源。

- 会话模式

  - 第一种：明确调用会话生成函数和关闭会话函数

    ```python
    sess = tf.Session()
    sess.run(...)
    sess.close()
    ```

  - 为解决异常退出的资源释放问题，采用如下方法：

    ```python
    with tf.Session() as sess:
    	sess.run(...)
    #不需要再调用close()函数
    ```

- 配置会话

  ```
  config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
  sess1 = tf.Session(config=config)
  ```

  可以

### 实现神经网络