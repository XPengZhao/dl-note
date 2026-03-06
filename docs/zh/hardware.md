# 硬件

## GPU、NPU、GPGPU

- **GPU（Graphics Processing Unit，图形处理器）**：一种高度并行、通用的处理器（SIMT/SIMD），面向高吞吐场景设计，适用于图形、线性代数和大多数机器学习任务。灵活且可编程（CUDA/ROCm）。

- **NPU（Neural Processing Unit，神经网络处理器）**：面向神经网络算子的专用加速器（如 matmul/conv、激活、池化）。通常通过固定功能或半可编程数据流，优先优化能效与时延。

- **GPGPU（General-Purpose computing on Graphics Processing Units，通用 GPU 计算）**：指利用 GPU 的大规模并行硬件执行非图形任务（科学计算/机器学习/信号处理等）。你不是在绘制三角形，而是在运行计算 kernel，让同一种操作并行作用于大量数据元素（SIMT/SIMD），从而在数据并行问题上获得高吞吐。

## HBM、GDDR

GDDR（Graphics Double Data Rate）和 HBM（High Bandwidth Memory）是用于 GPU 与高性能计算系统的两类不同显存技术，主要差异在于架构、带宽与能效。GDDR 采用传统平面布局，显存芯片分布在 GPU 周围，通过相对较窄的总线和高频率获得带宽。HBM 则将多层 DRAM 进行 3D 堆叠，并通过 TSV（硅通孔）与中介层（interposer）连接处理器，形成超宽内存接口，在更低频率下实现更高带宽和更低功耗。因此，HBM 在性能功耗比上更优，常见于数据中心与 AI 加速器；GDDR 因成本更低、集成更简单，仍是消费级 GPU 的主流选择。

## GPU 配置

锁步

### 显式指定 GPU

```python
torch.cuda.set_device(0,1)

## or you can set the env variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

```

也可以在运行时直接设置环境变量：

```bash
CUDA_VISIBLE_DEVICES=0,1 python script.py
```

**对比：**

如果你希望在脚本启动前就限制可见 GPU，使用 `CUDA_VISIBLE_DEVICES` 最直接。

如果你希望根据代码逻辑在运行中动态切换当前设备，则需要 `torch.cuda.set_device()`。
