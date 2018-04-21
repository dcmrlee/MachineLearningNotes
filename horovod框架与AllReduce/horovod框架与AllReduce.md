## 机器学习分布式与多GPU卡

在ML中，常常用多GPU卡来提升训练效率，也就是我们常说的单机多卡、多机多卡。
其中的主要原理很简单，快速解释下，假设在一块GPU上训练，一个step跑batch size=100需要1秒钟，训练1000 steps则需要1000秒（忽略其他时间）。
那么如果在两块GPU卡上训练，通过ps、worker数据并行的模式（假设网络开销可以忽略），一个step还是batch size=100，但是这时候每块GPU跑的mini batch=50，
那么训练1000 steps只需要500秒（忽略其他时间）

上面的情况很理想，现实中，往往是模型特别大，网络带宽有限的情况，在worker上计算完成，将模型参数同步到ps上的过程中，瓶颈就出现在网络带宽上，这时候容易
出现训练加速比越来越差，随着GPU的数量增多

如何解决这个问题，就可以尝试下面的AllReduce方案了

## AllReduce模式

简单的来说，AllReduce把所有的机器组成一个环，通过MPI的Scatter-Reduce、ALlgather两个阶段，完成模型参数的同步，把每两台机器之间的带宽都利用起来了，而不是像ps、worker那样，网络瓶颈在ps这端。

具体的方式，可以参考百度写的论文（Bringing HPC Techniques to Deep Learning），原理还是很好理解的。

AllReduce的优势就是在模型特别大、网络带宽有限的情况下，训练成本不随着GPU的数量增加而增加，所以可以放心的扩展GPU的卡数，达到scale的效果。这种情况下，
ps的方式就做不到

AllReduce的劣势就是对抗failover不太好，如果任意一个worker失败，参数同步需要重新做，而ps方式只需要某个worker重新计算再传给ps

## Horovod框架

Horovod框架是Uber开源的AllReduce框架

跑了下官方提供的mnist例子，发现在模型小、数据量小的时候，还是尽量别用，因为网络开销是大头。如图：
![单GPU一个step耗时](./images/single_gpu.jpg)

## 参考资料
