# Collective Communication

集合通信（collective communication）让一组 GPU 共同完成数据分发、收集或归约。例如，张量并行中每张 GPU 算出一部分结果后，可以通过 All-Reduce 求和，让所有参与者得到完整结果。

## 基本操作

通信组中的每个参与者有一个编号，称为 **rank**。下面用四张 GPU 对应 rank 0～3，归约操作均采用逐元素求和。各操作的输入输出定义见 [NCCL 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)。

### Broadcast

Broadcast 将指定 rank（root）上的数据复制给组内所有 rank。例如，以 rank 0 为 root，将它的 `[1, 2]` 广播后，四个 rank 都得到 `[1, 2]`。

### All-Gather

All-Gather 收集各 rank 的数据，按 rank 顺序拼接，并将完整结果交给每个 rank。

| Rank | 输入 | 输出 |
| --- | --- | --- |
| 0 | `[1, 2]` | `[1, 2, 3, 4, 5, 6, 7, 8]` |
| 1 | `[3, 4]` | `[1, 2, 3, 4, 5, 6, 7, 8]` |
| 2 | `[5, 6]` | `[1, 2, 3, 4, 5, 6, 7, 8]` |
| 3 | `[7, 8]` | `[1, 2, 3, 4, 5, 6, 7, 8]` |

它适用于各 rank 持有不同分片，而后续计算需要完整张量的情况。数据只做拼接，不做求和。

### All-Reduce 与 Reduce-Scatter

这两个操作都先对各 rank 的对应位置求和，区别在于结果如何分布。

| Rank | 输入 | All-Reduce 输出 | Reduce-Scatter 输出 |
| --- | --- | --- | --- |
| 0 | `[1, 2, 3, 4]` | `[1111, 2222, 3333, 4444]` | `[1111]` |
| 1 | `[10, 20, 30, 40]` | `[1111, 2222, 3333, 4444]` | `[2222]` |
| 2 | `[100, 200, 300, 400]` | `[1111, 2222, 3333, 4444]` | `[3333]` |
| 3 | `[1000, 2000, 3000, 4000]` | `[1111, 2222, 3333, 4444]` | `[4444]` |

**All-Reduce** 让每个 rank 得到完整的求和结果。**Reduce-Scatter** 则将求和结果分片，每个 rank 只得到自己负责的一段。如果只有指定 root 需要完整结果，可以使用 **Reduce**。

Reduce-Scatter 后再执行 All-Gather，可以得到与 All-Reduce 相同的结果。如果下一步计算可以直接使用分片，就可以保留 Reduce-Scatter 的输出，省去立即收集完整张量的通信。

### All-to-All

All-to-All 按目的 rank 重新分发数据。下面每个符号表示一个数据块，下标表示目的 rank：

| Rank | 输入 | 输出 |
| --- | --- | --- |
| 0 | `[a₀, a₁, a₂, a₃]` | `[a₀, b₀, c₀, d₀]` |
| 1 | `[b₀, b₁, b₂, b₃]` | `[a₁, b₁, c₁, d₁]` |
| 2 | `[c₀, c₁, c₂, c₃]` | `[a₂, b₂, c₂, d₂]` |
| 3 | `[d₀, d₁, d₂, d₃]` | `[a₃, b₃, c₃, d₃]` |

每个 rank 向其他 rank 发送不同的数据，也从其他 rank 接收发给自己的数据。与 All-Gather 不同，各 rank 最终得到的内容不同。

## 推理中的通信

### Tensor Parallelism

考虑线性层 $Y=XW$。沿输入维度切分到四张 GPU 后，各 rank 计算自己的部分结果：

$$
Y_r = X_rW_r,\qquad Y = \sum_{r=0}^{3}Y_r
$$

如果后续计算要求每个 rank 持有完整的 $Y$，就执行 All-Reduce。如果后续计算接受分片结果，则可以用 Reduce-Scatter。

沿输出维度切分时，各 rank 得到的是不同输出特征，不需要相加。只有后续计算需要完整输出时，才通过 All-Gather 拼接。通信操作由下一步需要的数据布局决定，具体切分方式见[并行策略](parallelism.md)。

### Expert Parallelism

MoE 的专家分布在不同 GPU 上。路由确定 token 要访问的专家后，需要将 token 表示发送到专家所在的 rank。专家计算完成，再将输出送回原 rank，按路由权重合并。

这对应 All-to-All 式的数据交换。不同专家收到的 token 数可能不同，因此实际实现需要处理可变的数据量。热门专家所在的 rank 可能接收更多数据、执行更多计算，使其他 rank 等待其结果。

流水线并行的阶段交接则通常是点对点发送与接收：上一阶段把激活交给下一阶段，无需整个通信组共同交换。

## 通信开销

### 消息大小与通信轮数

一次通信包含启动开销和数据传输时间。用 $\alpha$ 表示每轮固定延迟，$B$ 表示有效链路带宽，$S$ 表示该轮传输的字节数，可以粗略写成：

$$
T_{\mathrm{round}} \approx \alpha + \frac{S}{B}
$$

小消息中，固定延迟占比较大。低并发 decode 每轮处理的 token 少，但仍需反复执行层内通信，因此即使传输量不大，也可能明显增加延迟。大消息中，传输项更重要，链路带宽和实际流量对耗时影响更大。[NVIDIA 的推理通信分析](https://developer.nvidia.com/blog/optimizing-for-low-latency-communication-in-inference-workloads-with-jax-and-xla/)

### Ring 与 Tree

同一个 All-Reduce 可以采用不同算法。Ring 将 rank 组织成环，分块传递数据，可以分为 Reduce-Scatter 和 All-Gather 两个阶段。每个阶段需要 $p-1$ 轮，其中 $p$ 是 rank 数。GPU 越多，环上传递的轮数越多。

Tree 沿树形结构归约，再分发结果，通信层数随 rank 数呈对数增长。算法选择需要同时考虑轮数、链路利用率和拓扑，不能只比较总字节数。NCCL 会根据通信规模与硬件选择实现，参见 [NCCL 算法与协议调优](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/)。

这里的 $B$ 是 GPU 互联的有效带宽，不是 HBM 带宽。同样的 rank 数和张量大小，放在一台机器内或分布到多个节点，经过的链路和共享带宽不同，通信耗时也会不同。

### 与计算重叠

通信期间可以执行不依赖其结果的计算。例如，一个 microbatch 向下一流水线阶段发送激活时，当前阶段可以处理另一个 microbatch。

如果下一算子需要 All-Reduce 的结果，就必须等所需结果就绪。异步发起通信只能让调用方继续执行，能否缩短整体耗时，还取决于是否有独立计算可做，以及通信和计算是否争用 GPU 资源。分析一次推理时，应关注未被计算覆盖的通信时间。
