
## 背景知识

TensorFlow作为一块机器学习框架，功能非常强大，经过几年的发展，已经越来越复杂了，框架本身也做过很多优化，省去了算法工程师很多工作。但是，如果想用好TensorFlow也需要对其本身有一些深入的了解。本文重点想解释清楚TensorFlow中的In-graph与Between-graph两种机制。

**In-graph与Between-graph两种机制，其实指的是TensorFlow进行分布式训练中，两种构造计算图的方式，在编码的时候能看到明显的区别，这种区别同时也会带来计算图执行过程中的一些差异。**

想要理解这两种机制，需要一定的背景知识：

+ 计算图的概念：简单的理解，你的模型定义成什么样，模型的参数（变量）是哪些，损失函数是什么，梯度如何计算等等
+ 变量、设备、角色：简单的理解，模型中的参数，参数放在什么设备上计算（GPU还是CPU），每个角色有哪些变量
+ 分布式模式：这个比较重要，下面详细解释下

## TensorFlow分布式模式

**熟悉的朋友可以跳过**

#### 基本工作方式
这里将经典的参数服务器架构，两种核心的角色：ps和worker。基本工作方式：ps（Parameter Server）顾名思义是参数服务器，用于更新模型中的参数，worker用于训练模型中的参数并把训练完的参数发给ps，ps收集到每个worker的参数后，进行汇总更新（例如求均值），更新完后将新的参数发送给worker，worker拿到新的参数后继续新的训练，如此往复直到收敛。详情可以查看：[TensorFlow架构](https://www.tensorflow.org/extend/architecture)

#### 运行时概念
在分布式运行时，可以指定ps在哪些机器上，worker指定在哪些机器上运行，可以理解为定义这次分布式运行的集群，如下：
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
每个角色（ps 或 worker）运行时，都需要绑定一个叫server的对象，保证所有的角色，都知道集群中还有哪些角色的存在，同时也能区分不同的角色
```
server = tf.train.Server(cluster, job_name="ps", task_index=0)
worker = tf.train.Server(cluster, job_name="worker", task_index=0)
worker = tf.train.Server(cluster, job_name="worker", task_index=1)
```

#### 把参数分配给角色

参考如下的代码，通过with.device函数来分配变量
```
with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(...)
  biases_1 = tf.Variable(...)

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(...)
  biases_2 = tf.Variable(...)

with tf.device("/job:worker/task:7"):
  input, labels = ...
  layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
  logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
  # ...
  train_op = ...
```
上面的代码，在ps中定义模型变量，两个ps的task，将计算密集的操作放到worker的task上。TensorFlow在计算前向操作时，将参数从ps传给worker，在应用梯度时将参数从worker传给ps

#### 数据并行

multiple tasks in a worker job training the same model on different mini-batches of data ,updating shared parameters hosted in one or more tasks in a ps job

#### 执行过程
写好代码 --> client提交代码 --> master --> 启动 ps和worker开始进行分布式计算

---
有了上述概念就比较好理解了


## In Graph解释

一个client，构造**一个计算图**，其中的参数定义在ps上，将模型中计算复杂的操作，**拷贝多份**定义在不同的worker上

支持同步和异步训练

## Between Graph解释

多个client，都定义了相似的计算图，每个worker上有一个完整且相同的计算图

支持同步和异步训练


## 具体两种模式的代码举例

常用的In Graph编程模式（同步）
```
with tf.Graph().as_default():
    input, labels = get_inputs_and_labels()
    global_step = global_step = tf.get_variable('global_step', [],
                                initializer=tf.constant_initializer(0), trainable=False)
    ...
    grads = []
    for i in range(gpu_nums):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('model_%d' % i) as scope:
                # define your model
                ...
                grads.append(grad)
    # compute average grads
    # applying grads
```

常用的Between Graph编程模式（同步）
```
# Create any optimizer to update the variables, say a simple SGD:
opt = GradientDescentOptimizer(learning_rate=0.1)

# Wrap the optimizer with sync_replicas_optimizer with 50 replicas: at each
# step the optimizer collects 50 gradients before applying to variables.
# Note that if you want to have 2 backup replicas, you can change
# total_num_replicas=52 and make sure this number matches how many physical
# replicas you started in your job.
opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=50,
                               total_num_replicas=50)
                               
# Now you can call `minimize()` or `compute_gradients()` and
# `apply_gradients()` normally
training_op = opt.minimize(total_loss, global_step=self.global_step)

# You can create the hook which handles initialization and queues.
sync_replicas_hook = opt.make_session_run_hook(is_chief)

with training.MonitoredTrainingSession(
    master=workers[worker_id].target, is_chief=is_chief,
    hooks=[sync_replicas_hook]) as mon_sess:
  while not mon_sess.should_stop():
    mon_sess.run(training_op)
```


## 参考资料
+ [TF Distributed](https://www.tensorflow.org/deploy/distributed)
+ [In Graph Example Code](https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py)
+ [Between Graph Example Code](https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer)
