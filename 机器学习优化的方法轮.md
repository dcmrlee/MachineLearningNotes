## 输入管道流优化

通常模型都需要从磁盘读取输入，预处理，然后才输入给模型中的网络进行训练。例如，一般的图片输入流是下面这样：
load image from disk -> decode jpeg into a tensor -> corp and pad -> flip and distort -> batch

**不要让输入流成为瓶颈**

常用的做法：
- 将数据流操作放到CPU上面进行，解放GPU
- 用tf.data API，避免用feed_dict
- 用TFRecord
- 数据格式: NCHW for GPU, NHWC for CPU
