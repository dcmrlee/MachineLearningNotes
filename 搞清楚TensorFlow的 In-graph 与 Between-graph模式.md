
## 背景知识

TensorFlow作为一块机器学习框架，功能非常强大，经过几年的发展，已经越来越复杂了，框架本身也做过很多优化，省去了算法工程师很多工作。但是，如果想用好TensorFlow也需要对其本身有一些深入的了解。本文重点想解释清楚TensorFlow中的In-graph与Between-graph两种机制。

**In-graph与Between-graph两种机制，其实指的是TensorFlow进行分布式训练中，两种构造计算图的方式，在编码的时候能看到明显的区别，这种区别同时也会带来计算图执行过程中的一些差异。**

想要理解这两种机制，需要一定的背景知识：

+ 计算图的概念：简单的理解，你的模型定义成什么样，模型的参数（变量）是哪些，损失函数是什么，梯度如何计算等等
+ 变量与设备：简单的理解，模型中的参数，参数放在什么设备上计算（GPU还是CPU）
+ 分布式模式：这个比较重要，下面详细解释下

## TensorFlow分布式模式

**熟悉的朋友可以跳过**

#### 基本工作方式
这里将经典的参数服务器架构，两种核心的角色：ps和worker。基本工作方式：ps（Parameter Server）顾名思义是参数服务器，用于更新模型中的参数，worker用于训练模型中的参数并把训练完的参数发给ps，ps收集到每个worker的参数后，进行汇总更新（例如求均值），更新完后将新的参数发送给worker，worker拿到新的参数后继续新的训练，如此往复直到收敛。详情可以查看：[TensorFlow架构](https://www.tensorflow.org/extend/architecture)

#### 运行时概念
在分布式运行是，可以指定ps在哪些机器上，worker指定在哪些机器上运行，如下：
```
cluster = tf.train.ClusterSpec({
    "worker": [
        "worker0.example.com:2222",
        "worker1.example.com:2222",
        "worker2.example.com:2222"
    ],
    "ps": [
        "ps0.example.com:2222",
        "ps1.example.com:2222"
    ]})
```
每个角色（ps 或 worker）运行时，都需要绑定一个叫server的对象，保证所有的角色，都指导对方的存在，同时也能区分不同的角色
```
server = tf.train.Server(cluster, job_name="ps", task_index=0)
worker = tf.train.Server(cluster, job_name="worker", task_index=0)
worker = tf.train.Server(cluster, job_name="worker", task_index=1)
```
## In Graph解释
## Between Graph解释
## 具体两种模式的代码举例
