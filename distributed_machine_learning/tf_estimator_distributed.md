# 如何写tf.estimator的分布式代码

主要说三点
- 普通tensorflow分布式
- tf.estimator单机
- tf.estimator分布式


### 普通分布式
一般我们运行分布式的时候，采用的是下面这种方式：
```
ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")

# Create a cluster from the parameter server and worker hosts.
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

# Create and start a server for the local task.
server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)

if FLAGS.job_name == "ps":
  server.join()
elif FLAGS.job_name == "worker":
  ...
  # define model and loss, then start train
```
接收的命令行参数有：--ps_hosts、--worker_hosts、--job_name、--task_index

tf.estimator是tensorflow提供的高级API，也支持分布式。下面详细介绍下从单机版的，以及改造成分布式版的，并且能够兼容一般的分布式的传参。

### tf.estimator单机
下面写一个非常简单的线性回归模型，核心代码如下
```
feature_columns = [
        tf.feature_column.numeric_column(key="feature1"),
        tf.feature_column.numeric_column(key="feature2"),
]
model_dir = "/tmp/model_dir"
model = tf.estimator.LinearRegressor(feature_columns=feature_columns, model_dir=model_dir)

train_spec = tf.estimator.TrainSpec(input_fn=input_train_fn, max_steps=STEPS)
eval_spec = tf.estimator.EvalSpec(input_fn=input_test_fn)

tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
```
调用的是tf.estimator.train_and_evaluate()函数进行训练和eval，调用之前会定义TrainSpec、EvalSpec。

### tf.estimator分布式
首先这个分布式启动，会依赖一个叫TF_CONFIG的环境变量，每个角色启动前都会读取这个环境变量的值，来设置分布式相关的参数。
TF_CONFIG包括两个核心字段：cluster和task。
cluster分别为三个角色：chief、ps、worker。chief只能有一个。
举几个例子：
```
# chief的TF_CONFIG设置
TF_CONFIG='{
    "cluster": {
        "chief": ["host0:2222"],
        "worker": ["host1:2222", "host2:2222", "host3:2222"],
        "ps": ["host4:2222", "host5:2222"]
    },
    "task": {"type": "chief", "index": 0}
}'

# worker的TF_CONFIG设置
TF_CONFIG='{
    "cluster": {
        "chief": ["host0:2222"],
        "worker": ["host1:2222", "host2:2222", "host3:2222"],
        "ps": ["host4:2222", "host5:2222"]
    },
    "task": {"type": "worker", "index": 0}
}'

# ps的TF_CONFIG设置
TF_CONFIG='{
    "cluster": {
        "chief": ["host0:2222"],
        "worker": ["host1:2222", "host2:2222", "host3:2222"],
        "ps": ["host4:2222", "host5:2222"]
    },
    "task": {"type": "ps", "index": 0}
}'
```

下面将上面单机版tf.estimator改造成分布式的，并且兼容一般分布式的传参方式。默认一般分布式传参worker_hosts的第一个为chief。
```
# 分布式添加的代码(start)
ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")
job_name = FLAGS.job_name
task_index = int(FLAGS.task_index)

cluster = {'chief': [worker_hosts[0]],
               'ps': ps_hosts,
               'worker': worker_hosts[1:]}
if job_name == 'worker' and task_index == 0:
    job_name = 'chief'
if job_name == 'worker' and task_index > 0:
    task_index = task_index - 1
os.environ['TF_CONFIG'] = json.dumps({
        'cluster': cluster,
        'task': {'type': job_name, 'index': task_index}
})
# 分布式添加的代码(end)

# 单机版代码
feature_columns = [
        tf.feature_column.numeric_column(key="feature1"),
        tf.feature_column.numeric_column(key="feature2"),
]
model_dir = "/tmp/model_dir"
model = tf.estimator.LinearRegressor(feature_columns=feature_columns, model_dir=model_dir)

train_spec = tf.estimator.TrainSpec(input_fn=input_train_fn, max_steps=STEPS)
eval_spec = tf.estimator.EvalSpec(input_fn=input_test_fn)

tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
```
可以看出，只需要添加分布式相关的代码即可将单机版的，轻松改造成分布式版来运行


### 参考资料
https://www.tensorflow.org/deploy/distributed

https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate
