# 推理系统中的数据搬运与通信

推理性能不只由模型做了多少数学计算决定，也由为了支撑这些计算，系统必须搬运多少状态决定。在实践里，很多 serving 瓶颈并不是因为设备理论 FLOPs 不够，而是因为权重、activations、KV 状态，或者跨设备张量必须以不合适的粒度、从不合适的存储层级被读取、写回或交换。

这一页把数据搬运当作一个系统对象来介绍。目标是在讨论 KV cache 压力、分布式通信或 decode 低效这些具体问题之前，先建立一套干净的心智模型。

## 跨内存层级的数据搬运

第一步，是不要把数据搬运理解得过于狭义。推理中的 data movement 不只是从存储加载模型文件，或者把输入从 CPU 拷到 GPU。这些操作当然存在，但真正落在性能关键路径上的，通常是 kernel 执行过程中运行时状态在 memory hierarchy 之间被反复搬运。

一个足够有用的简化路径是：

`storage / host memory -> GPU HBM -> on-chip cache / registers -> compute units`

在多 GPU 执行中，还要再补上一条路径：

`GPU A HBM <-> GPU B HBM`

这个抽象是有意保持简化的。目标不是复述 GPU 微架构的每一个细节，而是识别出：为了让硬件真正开始计算，数据必须跨越哪些边界。

### Storage 与 Host Memory

最外层包含模型文件、驻留在 CPU 上的预处理结果、tokenized inputs，以及各种 host-side runtime metadata。它们在系统运行上当然重要，但在一个结构合理的在线 serving 主循环里，通常并不是 steady-state 的主要瓶颈。它们更重要的意义在于：定义了进入 device execution 的入口路径。

### GPU HBM

HBM 是 GPU 推理阶段最主要的工作内存。模型权重、activations 和 KV cache 通常都驻留在这里。HBM 的带宽远高于 host memory，但它依然比片上存储更慢、距离计算也更远。当人们说 serving workload 是 memory-bound 时，很多时候说的就是这一层的压力过大：太多有用工作依赖反复读取或写回驻留在 HBM 中的状态。

### On-Chip Cache 与 Registers

再往里，是更贴近执行的 on-chip caches、shared memory、register files，以及其它为活跃 kernel 提供短延迟工作集的存储结构。它们容量远小于 HBM，但访问代价低得多。一个 kernel 是否高效，往往取决于数据能否在这里被复用，而不是一遍遍重新从 HBM 拉取。

### Compute Units

Tensor cores、CUDA cores 或其它执行单元负责真正完成算术运算。只有当所需的操作数已经被搬运到它们可消费的位置和形式时，它们才能开始推进。这也是为什么 data movement 不是“独立于计算之外的另一个问题”，而是计算本身的一部分：每一步算术执行都受制于必要状态是否已经靠近 compute units。

## 为什么层级重要

同样数量的 FLOPs，如果依赖的数据来自不同层级，最终体现出来的性能可能完全不同。

- 如果数据能在片上存储中复用，那么有效供数代价就较低。
- 如果 kernel 需要反复从 HBM 拉取大块状态，带宽压力就会上升。
- 如果张量必须跨 GPU 交换，那么通信和同步就会进入关键路径。
- 如果执行路径过于频繁地回退到 host-device 交互，延迟就会显著增加。

这就是为什么数据搬运值得被单独拿出来讲。不是所有 bytes 的代价都一样。它们的系统成本取决于跨越的是哪一层边界、跨越得有多频繁，以及在再次搬运之前能否先被充分复用。

## 一个更实用的分类

从这套分层视角出发，推理系统里最常见的数据流动可以先分成四类：

- `host -> device`：输入、预处理输出，以及少量控制路径传输
- `HBM -> on-chip working set`：kernel 读取的 weights、activations 和 KV 状态
- `on-chip -> HBM`：写回的 outputs、更新后的 activations，以及新物化出的 KV 状态
- `GPU <-> GPU`：hidden states、partial reductions、expert traffic 或其它分布式张量

对单 GPU serving 来说，真正主导性能的通常不是 host 到 device 的传输，而是内层执行循环中对 HBM 的反复访问。到了多 GPU serving 场景，跨设备数据搬运则会在本地 memory hierarchy 之外，再引入一类新的瓶颈。

## 相关页面

- [Roofline Model for Inference](roofline.md) 解释为什么 bytes moved 会决定性能 ceiling。
- [KV Cache](kv-cache.md) 是重复状态搬运如何塑造 decode 效率的一个核心例子。
- [Parallelism](parallelism.md) 讨论当数据搬运超出单设备范围后，如何演化成通信问题。
