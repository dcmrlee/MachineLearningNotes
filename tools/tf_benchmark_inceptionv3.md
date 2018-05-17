# TensorFlow Benchmark performance of Inception-V3

## Preparation
1. benchmark code: https://github.com/tensorflow/benchmarks
2. Machine Info:
    - CPU: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz, 40 core
    - MEM: 256G
    - DISK: CephFS
    - Net: 10Gb/s
    - GPU: P40
        - Topo: ![P40 Topo](/images/p40_gpu_topo.jpg)
3. Model(Inception-V3) Info:
    - Nums of param: 23M
    - Memory of params: 23M * float32 = 92MB
    - Memory of Derivatives: 92MB

## Experiments
