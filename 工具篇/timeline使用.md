## timeline工具可以做什么

对Tensorflow进行性能剖析是十分有用的，通过性能剖析可以了解什么操作更花费时间

下面两种使用方法均可

### 使用方法1
自己控制

```
from tensorflow.python.client import timeline
...

with tf.Session() as sess:
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    ...
    for i in range(num_steps):
        sess.run(train_op,
                 feed_dict=feed_dict,
                 options=options,
                 run_metadata=run_metadata)
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline-%d.json' % i, 'w') as f:
            f.write(chrome_trace)
```
### 使用方法2

集成到高级API中
```
from tensorflow.python.client import timeline

timeline_dir = './timeline'

hooks = [
    tf.train.ProfilerHook(output_dir=timeline_dir, save_steps=100),
]

with tf.train.MonitoredTrainingSession(hooks=hooks) as mon_sess:
    mon_sess.run(train_op, feed_dict=feed_dict)
```

### 查看

上述方法都会将timeline的内容保存到最后的json文件中，可以使用chrome浏览器查看
打开浏览器，输入chrome://tracing  , 点击load按钮，选择相应的json文件即可查看

