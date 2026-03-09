# AI Infra 预备知识

## 指标

### 延迟

延迟是处理请求并生成输出 token 所需的时间。在 LLM 推理中，延迟通常通过两个主要指标来衡量：

#### 首 token 时间（TTFT）

首 token 时间（TTFT）是指从推理服务器收到请求，到第一个输出 token 被生成并返回给客户端之间的延迟。它衡量模型开始响应的速度，是流式 LLM 系统中最核心的用户感知延迟指标。形式化定义为：

$$
\text{TTFT} = t_{\text{first token emitted}} - t_{\text{request received}}
$$

TTFT 主要反映 prefill 阶段（处理输入上下文）以及调度和系统开销。它不包含生成其余输出 token 所需的时间。更细致地，TTFT 可分解为：

$$
\small
\begin{aligned}
\text{TTFT} = t_{\text{queue}} + t_{\text{input processing}} + t_{\text{prefill compute}} + t_{\text{first decode step (optional)}} + t_{\text{framework overhead}}
\end{aligned}
$$

其中：

-  $t_{\text{queue}}$ 是请求在被调度器接纳执行前的排队时间。
- $t_{\text{input processing}}$ 包括 tokenization、多模态预处理（例如图像或视频编码）以及请求级归一化或格式化。
- $t_{\text{prefill compute}}$ 是对全部输入 token 执行前向计算所需时间（序列长度为 $L$ 时复杂度为 $O(L)$），在该过程中会填充 KV cache 并得到 prompt 末位置的 logits。
- $t_{\text{first decode step (optional)}}$ 对应某些将 prefill 与 generation 调度分离的实现中的额外一次 decode step。在这些系统中，首个输出 token 由后续 decode pass 产生，而不是直接由 prefill logits 产生。在优化实现（vLLM 的较新版本）中，这一步可能与 prefill 融合，因此可忽略不计。
- $t_{\text{framework overhead}}$ 表示额外成本，例如 CUDA kernel 启动、同步、内存分配、采样（softmax 与 top-k/top-p）以及首个流式 token 的网络传输。

在长上下文输入下，TTFT 通常由 $t_{\text{prefill compute}}$ 主导，其余项贡献相对较小。

#### 每输出 token 时间（TPOT）

每输出 token 时间（TPOT）衡量在首 token 发出之后，平均生成每个 token 所需的延迟。它刻画自回归模型的稳态 decode 性能，并直接决定流式输出速度。形式化地，对于生成 $N$ 个输出 token 的请求：

$$
\text{TPOT} = \frac{t_\text{last token emitted} - t_\text{first token emitted}}{N - 1}
$$

等价地，若端到端总延迟为 $T_{\text{total}}$，

$$
\text{TPOT} = \frac{T_{\text{total}} - \text{TTFT}}{N - 1}
$$

在稳态解码中，每个输出 token 需要：

$$
\text{TPOT} = t_{\text{decode compute}} + t_{\text{communication}} + t_{\text{scheduling}} + t_{\text{framework overhead}}
$$

其中：

- $t_{\text{decode compute}}$ 是使用 KV cache 对单个 token 执行前向计算的时间（每步在序列长度维度上的复杂度为 $O(1+\text{num speculative tokens})$，但与模型规模成正比）。
- $t_{\text{communication}}$ 表示跨 GPU 同步与数据交换成本，例如 tensor parallel 的集合通信（all-reduce、all-gather）、MoE 模型中的 expert 路由与聚合（all-to-all、dispatch/combine），以及由调度或内存管理触发的主机-设备内存传输（H2D/D2H）。
- $t_{\text{scheduling}}$ 包括运行时调度开销，例如 continuous batching、请求协调、准入控制，以及 KV cache 分配、分页或内存管理。
- $t_{\text{framework overhead}}$ 包括辅助执行成本，例如采样（softmax、top-k/top-p）、kernel 启动、同步、输入输出编组、结果后处理以及 token 流式传输。

与 TTFT 不同，TPOT 不需要对完整 prompt 重新计算。但它仍取决于有效上下文长度，因为每步 decode 都需要对全部已缓存 token 执行 attention。因此 TPOT 通常随总上下文长度（prompt + 已生成 token）近似线性增长，不过其单 token 成本显著小于 prefill。

### 吞吐

#### 每秒 token 数（TPS）

每秒 token 数（TPS）用于衡量 LLM 推理系统的吞吐，即在给定服务配置下每秒可处理或生成多少 token。与 TTFT 和 TPOT 这类在请求级定义的延迟指标不同，TPS 是系统级指标，反映负载下的整体服务能力。在时间区间 $\Delta t$ 内，TPS 可表示为：

$$
\text{TPS} = \frac{N_\text{tokens}}{\Delta t}
$$

很多情况下，TPS 特指 decode 吞吐，即只统计生成的输出 token：

$$
\text{TPS}_\text{decode} = \frac{N_\text{output tokens}}{\Delta t}
$$

对于单序列顺序解码，它大致是 TPOT 的倒数：

$$
\text{TPS}_\text{per request} \approx \frac{1}{\text{TPOT}}
$$

但在具有 continuous batching 和并发解码的实际服务系统中，总吞吐会随有效 batch 大小 $B$ 扩展。对于单个副本，系统吞吐可近似为

$$
\text{TPS}_\text{replica} \approx \frac{B}{\text{TPOT}}
$$

若有 $R$ 个数据并行（DP）副本，

$$
\text{TPS}_\text{total} \approx R \times \frac{B}{\text{TPOT}}
$$

需要注意的是，对于长上下文或多模态负载，prompt 处理可能占据总计算的很大比例。在这种情况下，通常需要考虑端到端吞吐，它同时计入 prompt 与输出 token：

$$
\text{TPS}_\text{end-to-end} = \frac{N_\text{prompt tokens} + N_\text{output tokens}}{\Delta t}
$$

总体而言，TPS 受模型规模、并行策略、有效 batch 大小、kernel 效率、互联带宽，以及 prefill 与 decode 负载相对占比的影响。

## 调度

### Chunked Prefill

Chunked prefill 是一种面向长上下文推理的 serving 侧优化。它不再用一次 prefill pass 处理整个 prompt，而是将其切分为多个较小的 chunk，并按增量方式依次执行。若输入序列长度为 $L$，chunk 大小为 $C$，那么 prefill 阶段可被分解为大约

$$
N_{\text{chunks}} = \left\lceil \frac{L}{C} \right\rceil
$$

个子请求。对于第 $j$ 个 chunk，模型只处理该 chunk 对应的 token 区间，并将得到的 key 和 value 追加到 KV cache 中，然后再进入下一个 chunk。

从系统角度看，这样做的主要动机在于 prefill 和 decode 的执行特征明显不同。长 prefill 请求通常更宽、计算更重，而 decode 则由大量细粒度、对延迟敏感的迭代构成。如果运行时把一个很长的 prompt 当作单个整体 prefill 来执行，该请求可能会在较长时间内占据 GPU，并拖慢不相关的 decode 工作。Chunked prefill 将 prompt 处理的调度粒度变得更细，使运行时能够在长 prefill 请求、decode step 以及其他较短 prompt 之间进行穿插调度。实践中，它主要是一种在 continuous batching 下改善公平性与响应性的优化。

Chunked prefill 还可以降低 prefill 阶段的峰值运行时显存。当 prompt 一次性处理时，运行时往往需要为完整序列分配更大的临时激活、attention workspace 或其他中间 buffer。分块之后，这些瞬时 buffer 更接近按 chunk 大小而不是按完整 prompt 长度增长，因此在超长输入下有助于避免 OOM。但它不会降低最终的 KV cache 占用：只要所有 prompt token 都需要保留在上下文中，它们对应的 key 和 value 仍然必须存储。也就是说，chunked prefill 缓解的是 prefill 过程中的瞬时显存压力，而不是稳态的 KV cache 需求。

它改变的是调度特性和峰值内存行为，而不是模型语义。在 attention mask 和数值计算一致的前提下，最终得到的 KV cache 和输出 logits 与未分块的 prefill 是相同的。其代价在于，chunk 太小时会引入更多 kernel 启动和调度开销，而 chunk 太大时又会逐渐退化为普通的整体 prefill。从概念上看，chunked prefill 属于 serving runtime，而不是模型结构本身。


## KV Cache

### 内存开销

KV cache 大小计算为：

$$
\text{KV size} = 2 \times \text{num layers} \times \text{num heads} \times \text{seq len} \times \text{head dim}
$$

假设：

- 32 层、80 个头、head_dim = 128、seq_len = 4K。
- FP16 KV cache：2 × 32 × 80 × 4K × 128 × 2 bytes = **约 5.2 GB** 每样本。
- C8（int8 KV cache）：**约减半，即约 2.6 GB** 每样本。

## Attention

在 transformer 模型中，attention 模块计算：

$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

每个 token 会产生：

- **Q（Query）** - 来自当前 token 的隐藏状态
- **K（Key）**、**V（Value）** - 来自上下文（此前 token）并存储在 **KV cache** 中

**MHA**、**MQA** 和 **GQA** 的区别在于：相对于 Query 头数，使用了多少组 Key/Value 投影。

### **MHA — Multi-Head Attention**

每个 attention 头都有各自独立的投影：

$$
Q_i=X W_{Q, i}, \quad K_i=X W_{K, i}, \quad V_i=X W_{V, i}
$$

- Q 头数 = K 头数 = V 头数

### **MQA — Multi-Query Attention**

所有头 **共享一组** K 和 V 投影：

$$
K_{\text {shared }}=X W_K, \quad V_{\text {shared }}=X W_V
$$

- **多个 Q 头**，但 **只有一对共享 K/V**。
- 推理时，所有头复用相同的 K/V。
- KV cache 大小可减少约 #heads 倍。
- **LLaMA 2 7B**、**Mistral** 及多数优化后的 LLM 使用 MQA 以降低内存占用。

### **GQA — Grouped-Query Attention**

Query 被分组，每组共享各自的 K/V：

$$
\text { Group } g: \quad Q_{g, *}, K_g, V_g
$$

- 例如有 8 个 query 头但只有 2 组，则每组 4 个 Q 头共享一组 K/V。
- **K/V 头数 < Q 头数**。
- **LLaMA 2 70B**、**Mixtral**、**GPT-J-6B** 等使用 GQA 以提升内存效率。

## Parallelism

### 数据并行（Data Parallelism, DP）

Data Parallelism（DP）是机器学习系统中最传统、也最直观的分布式执行形式。它的核心思想很简单：在多个设备上实例化同一个模型，同时将输入工作负载分片到这些实例上。每个 worker 在不同的数据子集上执行相同的计算图。从这个意义上讲，它只改变了计算到硬件的映射方式。

由于参数是复制而不是切分，DP 不会降低单个设备上的内存占用。这一特性界定了它的实际适用范围：当模型本身已经可以放入单卡显存时，DP 很有效；但如果模型本身超过单设备容量，DP 就不够用。因此，它的主要价值在于通过副本扩展吞吐和服务并发。

在训练中，DP 常用于增大有效 batch size 并提升硬件利用率。每个 worker 处理 mini-batch 的一个分片，计算本地梯度，然后参与同步，使所有模型副本在优化步之后保持数值一致。在推理中，同样适用这种副本机制，但不再有梯度交换。取而代之的是，每个 worker 处理独立的请求子集或 token batch。因此在推理负载下，DP 应主要被理解为一种流量分发机制，而不是模型切分机制。

在混合部署中，DP 往往与 Tensor Parallelism（TP）和 Expert Parallelism（EP）并存。在这种配置下，“DP replica” 不应被字面地理解为单张 GPU。一个 DP 单元本身可能是一个多 GPU 组，其内部执行由 TP 或 EP 组织。从这个视角看，DP 定义的是工作负载复制的外层结构，而 TP 与 EP 决定每个副本内部如何执行计算。

#### 与 DP 相关的集合通信

与 DP 相关的通信行为，本质上取决于系统是在训练还是推理。

在分布式训练中，每个 DP worker 基于本地 mini-batch 分片计算梯度。由于每个 worker 都维护同一套参数，这些梯度必须在每个 step 后同步，以保证副本间一致性。用于这一目的的标准集合通信原语是对梯度执行 all-reduce。在一些实现中，也会用 reduce-scatter 后接 all-gather 来实现同样效果，但目标不变：聚合所有 DP rank 的梯度贡献，并在每个 worker 上得到一致的参数更新。

相比之下，推理不涉及反向传播或优化器 step。每个 DP rank 服务于独立的一部分请求流，因此 DP 本身不会引入强制性的逐层集合通信。在推理场景下，它主要充当调度与流量切分机制，而不是同步密集型执行策略。

即便如此，基于 DP 的推理系统也并非完全无通信。运行时仍需要系统级协调，例如请求分发、负载均衡、准入控制与结果收集。实践中，这类协调常依赖 all-gather 等通信模式，在各 rank 间交换元数据、状态或调度信息。这些操作属于服务框架层，而非模型数值执行本身。

在混合 MoE 部署中，DP 在推理阶段通常只引入有限通信开销，而每个 DP 单元内部的并行机制往往更“重通信”。Tensor Parallelism 通常依赖 TP 组内的 all-reduce 或 all-gather 等集合通信，而 Expert Parallelism 会引入跨设备的 token 路由以访问不同专家。因此，分析混合系统中的通信时，应区分外层 DP 维度与内层模型并行维度：DP 负责工作负载分发，而主导通信成本的通常是 TP 或 EP。

### 张量并行（Tensor Parallelism, TP）

张量并行（TP）通过对单层计算中的权重张量进行切分，将计算分布到多个设备上，而不是在每个 rank 上复制完整的层参数。在大规模 Transformer 模型中，这种方式通常应用于注意力层和 MLP 层中的投影矩阵。其核心目标是降低单设备的参数与激活内存占用，从而能够运行那些在单个加速器上无法容纳的超大规模模型层。

考虑一个线性变换：

$$
Y = X W,
$$

其中 $X \in \mathbb{R}^{B \times H}$，$W \in \mathbb{R}^{H \times O}$，$Y \in \mathbb{R}^{B \times O}$。在 TP 中，权重矩阵 $W$ 会在 $p$ 个 rank 构成的进程组上进行切分。常见的两种切分方式是按列切分（column-wise sharding）和按行切分（row-wise sharding）。

在列切分的 TP 中，输出维度 $O$ 被划分，因此每个 rank 存储

$$
W_i \in \mathbb{R}^{H \times O/p}
$$

每个 rank 计算一个局部输出

$$
Y_i = X W_i \in \mathbb{R}^{B \times O/p}
$$

这对应于最终输出张量的一个切片。这种模式在 Transformer 的前馈网络（feed-forward block）的第一层投影中非常常见，因为这些部分输出通常可以保持分布式状态，直到后续某个需要同步的阶段才进行合并。

在行切分的 TP 中，输入维度 $H$ 被划分，因此每个 rank 存储

$$
W_i \in \mathbb{R}^{H/p \times O}
$$

输入 $X$ 也必须按照相同方式进行切分，每个 rank 计算对最终输出的一个部分贡献：

$$
Y_i = X_i W_i \in \mathbb{R}^{B \times O}
$$

由于这些结果都是对同一个输出张量的加性贡献，因此所有 rank 的结果需要进行求和才能恢复完整的 $Y$。因此，行切分的 TP 天然需要一个归约（reduction）步骤。

在 Transformer 的实现中，TP 通常会通过在相邻投影层之间交替使用通信较轻和通信较重的布局来设计。例如，在两层 MLP 结构中，第一层线性变换通常采用列切分以扩展隐藏维度，而第二层则使用行切分将其投影回原始隐藏维度。这种组合可以避免在每个 rank 上物化完整的中间激活，同时尽量延迟通信，使大部分计算保持在本地完成。

TP 的主要优势在于能够扩展模型的宽度，而无需在每个设备上复制全部参数。然而，其效率在很大程度上取决于本地矩阵乘计算与跨 rank 通信之间的平衡。随着 TP 并行度的增加，每个 rank 的 GEMM 规模会变小，而通信在整体执行时间中的占比则会上升。因此，TP 通常只在紧密连接的设备组中使用，例如同一节点内通过高带宽互联（如 NVLink）连接的 GPU。

#### 与 TP 相关的集合通信（Collective Communication）

TP 中的通信模式由张量的切分方式以及本地计算产生的是独立输出切片还是部分求和结果所决定。在实际系统中，TP 主要依赖三种集合通信操作：all-gather、reduce-scatter 和 all-reduce。这些通信操作不仅用于数据交换，同时也定义了张量并行执行过程中的同步边界。

在列切分的线性层中，每个 rank 生成的是输出特征的一部分。如果下一个算子能够直接消费这种分布式形式的激活，则不需要立即进行通信。这也是列切分在实践中常用的原因之一。然而，如果后续操作需要完整的激活张量，那么所有 rank 必须执行一次 all-gather 来重建完整输出：

$$
Y = [Y_0, Y_1, \ldots, Y_{p-1}]
$$

因此，all-gather 通常用于恢复在特征维度上被拆分的张量。

在行切分的线性层中，每个 rank 只计算最终输出的一部分贡献。全局输出为：

$$
Y = \sum_{i=0}^{p-1} Y_i
$$

这需要在 TP 组内进行求和操作。如果每个 rank 都需要完整的输出，则该操作可以表示为 all-reduce。在一些优化实现中，这个通信步骤会与后续的张量切分需求融合，并通过 reduce-scatter 实现，从而在完成求和的同时将输出重新分布为分片形式，以供下一阶段计算使用。

从系统角度来看，TP 通信的成本受到消息大小、集合通信算法、进程组拓扑结构以及通信与计算的重叠程度等因素的影响。由于 TP 的集合通信是在每一层内部以较高频率触发的，其性能对通信延迟非常敏感。因此，TP 通常只在少量且拓扑接近的设备之间使用。若 TP 组跨越较慢的网络链路，集合通信延迟很容易成为性能瓶颈，从而抵消参数切分带来的收益。

从整体角度看，TP 本质上是将一次大型矩阵乘法拆分为多个较小的本地 GEMM 计算，并辅以结构化的集合通信操作。因此，其性能取决于两个因素之间的权衡：一方面是降低单设备内存压力与计算规模带来的好处，另一方面是 all-gather、reduce-scatter 和 all-reduce 等通信操作所带来的额外开销。


## Kernel

## Overhead

一个tensor在steam1中准备好了，stream2要用，是还需要一个D2D copy嘛，这个时间大概是多久？和stream2 上准备的tensor stream2直接用差了多少？



## Sampling
### Temperature

温度 $T$ 控制概率分布的“尖锐度”。给定模型 logits $z_i$，温度采样会在 softmax 前对其重缩放：

$$
p_i=\frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

- $T<1$ 时更尖锐（峰值更高），模型更“自信”、随机性更低；$T>1$ 时更平坦（更均匀），模型更具探索性、多样性更高。
- 温度会保持 logits 的顺序，因为除以正数不会改变排序（保序）。如果使用 top-k = 1（即 greedy），温度完全不起作用。
- 当温度 $T \rightarrow 0$ 时，softmax 分布会变得任意尖锐并收敛到 greedy（argmax）采样。例如在 $T=10^{-6}$ 时，最高 logit 会主导分布，因此在实践中几乎等价于 greedy。
- 当 $T \rightarrow \infty$ 时，分布会趋于均匀（所有 token 概率相同）。

### Top-k Sampling

在计算概率（温度处理后）后，按概率排序，仅保留概率最高的 k 个 token。然后对 $p_i^{\prime}$ 归一化得到最终采样分布。

$$
\begin{aligned}
S_k & =\text { top-k tokens by } p_i \\
p_i^{\prime} & = \begin{cases}\frac{p_i}{\sum_{j \in S_k} p_j} & i \in S_k \\
0 & \text { otherwise }\end{cases}
\end{aligned}
$$

然后从该截断分布中采样。

- 小 k（例如 1-10）：更确定、更安全（模型几乎不会选择罕见词）。
- 大 k（例如 50-100）：更有变化，但选出离题词的风险更高。

### Top-p（nucleus）Sampling

“nucleus” 的字面含义是核心、中心或本质核心部分。与固定 k 不同，我们选择累计概率 $\geq p$ 的最小 token 集合。

$$
\begin{aligned}
S_p&=\left\{i: \sum_{j \in S_p} p_j \geq p\right\} \\
p_i^{\prime}&= \begin{cases}\frac{p_i}{\sum_{j \in S_p} p_j} & i \in S_p \\
0 & \text { otherwise }\end{cases}
\end{aligned}
$$

**与 top-k 的关键区别：**

Top-p 会随不确定性自适应。Top-k 看的是排名（前 k 个），Top-p 看的是概率质量（分布核心质量）。

- 如果模型很确定（一个 token 占主导），保留的 token 会很少。
- 如果模型不确定（很多 token 概率接近），nucleus 集合会扩大。

### Min-p

Min-p 会过滤掉相对 top-1 来说“过于不可能”的 token。这是较新的方法，但越来越流行（Mistral、Gemini 等在用）。它不固定 $k$ 或 $p$，而是保留相对 top-1 token 超过最小概率阈值的 token：

$$
S_{\min }=\left\{i: p_i \geq \min -\mathrm{p} \times p_{\max }\right\}
$$

如果 min-p = 0.1，则任何概率至少达到最可能 token 的 10% 的 token 都会被保留。随后归一化：

$$
p_i^{\prime} = \frac{p_i}{\sum_{j \in S_{\min}} p_j}
$$

- Min-p 与 top-p 一样会随上下文自适应，但更简单。它意味着候选 token 数量会随模型当前预测而变化。模型越确定，候选集越小；模型越不确定，候选集越大。
- 在分布较平坦时，可避免过早截断一些看起来排名低但仍合理的 token。
- 对长尾词表通常有更好的数值稳定性。

### 组合使用

通常按以下顺序组合使用：Temperature → Softmax → (top-k / top-p / min-p) → Renormalize → Sample

当 top-k 和 top-p 同时启用时，最终保留的是二者交集，也就是更小的那个集合。

### 示例代码

代码片段来自 [Omniinfer Sampler](https://gitee.com/omniai/omniinfer/blob/master/omni/adaptors/vllm/sample/sampler.py)。

<details>
<summary>Top-k 和 Top-p 实现</summary>

```python
def apply_top_k_top_p(
    logits_or_prob: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
    is_logits: bool,
) -> torch.Tensor:
    if p is None:
        if k is not None:
            logits_or_prob = apply_top_k_only(logits_or_prob, k, is_logits)
        if is_logits:
            probs = logits_or_prob.softmax(dim=-1, dtype=torch.float32)
        else:
            probs = logits_or_prob / logits_or_prob.sum(dim=-1, keepdim=True)
        return probs, None

    logits_or_prob_sort, logits_or_prob_idx = logits_or_prob.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_or_prob_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_or_prob_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_or_prob_sort < top_k_mask
        logits_or_prob_sort.masked_fill_(top_k_mask, -float("inf") if is_logits else 0)

    # Apply top-p.
    if is_logits:
        probs_sort = logits_or_prob_sort.softmax(dim=-1)
    else:
        probs_sort = logits_or_prob_sort / logits_or_prob_sort.sum(dim=-1, keepdim=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    # at least one
    top_p_mask[:, -1] = False
    probs_sort.masked_fill_(top_p_mask, 0)
    probs = probs_sort / probs_sort.sum(dim=-1, keepdim=True)
    return probs, logits_or_prob_idx
```

```python
def apply_top_k_only(
    logits_or_prob: torch.Tensor,
    k: torch.Tensor,
    is_logits: bool,
) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits_or_prob.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits_or_prob.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits_or_prob.masked_fill_(logits_or_prob < top_k_mask, -float("inf") if is_logits else float(0))
    return logits_or_prob
```
</details>



## Speculvative decoding

### Speculative Decoding 加速

Speculative decoding 通过更小的草稿模型 $M_d$ 来加速目标模型 $M_t$ 的推理。每个解码周期中，草稿模型先提出 $\gamma$ 个 token，再由目标模型并行验证。平均每 token 延迟为：

$$
L = \frac{T_{\text{draft}} + T_{\text{target}}}{\tau}.
$$

其中，$T_{\text{draft}}$ 是生成草稿 token 的时间，$T_{\text{target}}$ 是验证开销，$\tau \in [1, \gamma + 1]$ 是每周期期望接受的 token 数（包含目标模型产生的 bonus token）。令 $L_\text{target}$ 表示 $M_t$ 自回归逐 token 延迟，则速度提升为：

$$
\eta = \frac{L_\text{target}}{L}.
$$

#### 加速来自哪里？

对于 speculative decoding 的目标侧验证，可用一个紧凑且直观的模型表示：

$$
T_{\text {target }}(n) \approx \alpha_t(n)+n \cdot \bar{\beta}_t(n),
$$

其中 $n= B(\gamma + 1)$ 表示在服务 batch $B$ 中并行验证的 token 总数，$\bar{\beta}_t(n)$ 是宽度为 $n$ 时的有效逐 token 验证成本（秒/token）。在这个分解里，$\alpha_t(n)$ 聚合了在中等范围内对 $n$ 弱依赖的调用级开销，包括 kernel 启动与同步延迟、框架分发/图调度开销、每次调用的记账成本（如 attention metadata 构建与 paged-attention 索引设置），以及分布式或 MoE 场景下的 collective 启动/同步。项 $n \cdot \bar{\beta}_t(n)$ 则聚合了与处理 token 数近似成比例增长的工作：QKV/MLP GEMM、attention 计算、KV-cache 读写与逐元素操作。对 MoE 而言，dispatch/gather 的载荷搬运主要计入线性项，而 collective 启动更自然地归于开销项。

关键是，$\bar{\beta}_t(n)$ 不是常数：当 $n$ 从低填充区增大时，GEMM 形状更友好、occupancy 提高、cache 复用改善，从而提高有效算术强度与利用率，并降低 $\bar{\beta}_t(n)$。这种下降通常会持续到接近硬件上限（受 KV/cache 流量带宽或 tensor-core 算力限制）后趋于平坦，此时 $\bar{\beta}_t(n)$ 接近由设备决定的下限，总时间也会近似线性随 $n$ 增长。

在固定 $\gamma$ 和固定期望接受长度 $\tau$ 下，目标侧对每个“已接受 token 延迟”的贡献为：

$$
\frac{T_{\text {target }}(n)}{\tau} \approx \frac{\alpha_t(n)}{\tau} + \frac{n}{\tau} \bar{\beta}_t(n).
$$

加速的关键直觉是：在低 batch/低填充区，验证过程可让 $\bar{\beta}_t(n)$ 显著小于目标模型基线自回归逐 token 成本，同时 $\alpha_t(n)$ 会被 $\tau$ 摊销。一个获益条件是（暂时忽略 draft 成本）：

$$
\frac{T_{\text {target }}(n)}{\tau} < L_\text{target}(B).
$$

令 $p = \tau / (\gamma + 1)$ 表示每个提议 token 的期望接受率。决定加速效果的主导因子之一是：

$$
\frac{\bar{\beta}_t(n)}{p}  < \bar{\beta}_t(B).
$$


$\scriptstyle\blacksquare$ **它改变了目标模型工作的粒度。** 在自回归解码中，目标模型每生成一个 token 都执行一次前向，形成一长串严格串行的小步。对 GPU 而言，单 token decode 往往受内存/时延主导：每步相对于访问的状态量（KV cache 读写、小 kernel、同步与启动开销）来说计算量有限。因此在 batch-1 或低并发服务中，硬件利用率通常较差。即使模型理论 FLOPs 很高，实际逐 token 吞吐也可能受内存带宽和启动延迟而非算力上限约束。

Speculative decoding 改变了目标计算形态。目标模型不再把每步开销支付 $(\gamma+1)$ 次，而是每周期支付一次来验证 $\gamma$ 个提议 token。Speculative decoding 将大量细碎、时延主导的目标步，转成更少但更宽的验证步。

$\scriptstyle\blacksquare$ **验证成本随 $\gamma$ 的增长是次线性的。** 如果验证 $\gamma$ 个 token 的成本精确等于 $(\gamma + 1) \cdot L_\text{target}$，speculative decoding 就不会有帮助，因为这只是做了同样的工作再加上 draft 开销。实际中的速度提升来自 $T_\text{target}(\gamma)$ 更接近“一次稍宽的 decode”，而不是 $\gamma$ 次独立 decode。原因包括：(1) 固定开销（kernel 启动、同步）被摊销；(2) 更高的有效算术强度与硬件利用率。更宽的验证会放大 GEMM/attention kernel 规模，提升 occupancy 与 cache 复用，因此目标模型验证 pass 往往可获得更高 FLOP/byte（L2/HBM）与更高 tensor-core 效率。

#### 为什么在大服务 batch 下加速会变小（甚至为负）？

在 batch-1 或低并发服务下，基线自回归 decode 利用率较差，speculative verification 仍有较大优化空间。随着服务 batch 增大，基线 decode 本身已更宽、更高效，speculative decoding 能利用的次线性空间变小。当系统接近算力饱和（compute-bound）时，$T_\text{target}(\gamma)$ 会更接近对 $\gamma$ 线性增长，而 speculative decoding 可能收益很小，甚至出现负加速，因为它引入了 draft 计算与额外控制流。常见失败场景包括：

- 接受率低：$\tau$ 小，摊销失败。

- draft 太慢：$\frac{T_\text{draft}}{\tau}$ 吃掉收益。

- 服务已算力饱和：验证近线性，可优化空间很小。

- 实现开销大：拒绝记账、同步、KV 处理和采样逻辑若实现不够紧凑，会抹去理论收益。




<p class="doc-reference-label"><strong>Reference</strong></p>

<p class="doc-reference">[1] Chen J, Liang Y, Liu Z. DFlash: Block Diffusion for Flash Speculative Decoding. arXiv preprint arXiv:2602.06036, 2026.</p>


###  Rejection Sampler

若目标是采样 $x \sim p(x)$，可先从 $q(x)$ 采样 $x$：当 $q(x) \leq p(x)$ 时保留；当 $q(x) > p(x)$ 时，以概率 $1 - \frac{p(x)}{q(x)}$ 拒绝该样本，并改为从调整分布 $p^{\prime} = \text{norm}(\max(0, p(x) - q(x)))$ 重新采样 $x$。

**NOTE:** 对任意分布 $p(x)$ 与 $q(x)$，通过 rejection sampling（基于 $p(x)$ 和 $q(x)$）得到的 token 分布，与直接从 $p(x)$ 采样得到的分布完全一致。证明见附录 A [1]。

Rejection sampling 的思路是将目标分布 $p(x)$ 分解为两部分：

-  接受分支：基于比率 $\frac{p(x)}{q(x)}$ 接受 token，其中 $q(x)$ 是提议（draft）分布。

$$
  p(\text{accepted}, x = x^{\prime}) = q(x^{\prime}) \cdot \min\left(\frac{p(x^{\prime})}{q(x^{\prime})}, 1\right) = \min(p(x^{\prime}), q(x^{\prime}))
$$

- 拒绝分支：token 被拒后从调整分布 $p^{\prime}(x)$ 重采样。所有从 $q(x)$ 采样 token 被拒绝的概率为：

$$
\begin{aligned}
    p(\text{rejected}) &= \sum_{x} q(x) \left(1 - \min\left(\frac{p(x)}{q(x)}, 1\right)\right) \\
    &= \sum_{x} \max(0, q(x) - p(x)) \\
\end{aligned}
$$

$$
\begin{aligned}
    p(\text{rejected}) &= \sum_{x} q(x) \left(1 - \min\left(\frac{p(x)}{q(x)}, 1\right)\right) \\
    &= \sum_{x} \max(0, q(x) - p(x)) \\
\end{aligned}
$$



**示例代码**

代码片段来自 [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py)。

<details>
<summary>Triton 中的拒绝采样 kernel</summary>

```python
# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            if NO_DRAFT_PROBS:
                draft_prob = 1
            else:
                draft_prob = tl.load(
                    draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                )
            target_prob = tl.load(
                target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
            )
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            # NOTE(woosuk): While the draft probability should never be 0,
            # we check it to avoid NaNs. If it happens to be 0, we reject.
            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                # Accept.
                token_id = draft_token_id
            else:
                # Reject. Use recovered token.
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )
```

</details>

<details>
<summary>Triton 中的重采样 kernel</summary>


```python
@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    vocab_offset = tl.arange(0, PADDED_VOCAB_SIZE)
    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
        prob = tl.load(
            target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=((vocab_offset < vocab_size) & (vocab_offset != draft_token_id)),
            other=0,
        )
    else:
        draft_prob = tl.load(
            draft_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0,
        )
        target_prob = tl.load(
            target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0,
        )
        prob = tl.maximum(target_prob - draft_prob, 0)
        # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
        # `tl.argmax` will select the maximum value.

    q = tl.load(
        q_ptr + req_idx * vocab_size + vocab_offset,
        mask=vocab_offset < vocab_size,
        other=float("-inf"),
    )
    recovered_id = tl.argmax(prob / q, axis=-1)
    tl.store(output_token_ids_ptr + start_idx + pos, recovered_id)

```
</details>

**Reference**

[[1]](https://arxiv.org/pdf/2211.17192) Y Leviathan, M Kalman, Y Matias, "Fast Inference from Transformers via Speculative Decoding", ICML 2023.


## LLM 预训练目标

从零训练的大语言模型，通常是在大规模文本语料上用最大似然估计（MLE）进行优化。

给定从数据分布 $p_\text{data}$ 采样的 token 序列 $(t_1, t_2, \ldots, t_n)$，模型定义如下自回归分解：

$$
p_\theta(t_1, t_2, \ldots, t_n) = \prod_{i=1}^n p_\theta(t_i | t_1, t_2, \ldots, t_{i-1})
$$

训练目标是最大化数据的对数似然，这等价于最小化负对数似然（NLL）：

$$
\mathcal{L}(\theta) = - \mathbb{E}_{(t_1, t_2, \ldots, t_i) \sim p_\text{data}} \left[ \log p_\theta(t_i | t_1, t_2, \ldots, t_{i-1}) \right]
$$

在实践中，这个损失实现为以 one-hot 目标计算的 token 级交叉熵：

$$
\mathcal{L} = -\frac{1}{N} \sum_{\text{tokens}} \log p_\theta(t_i | t_1, t_2, \ldots, t_{i-1})
$$

该目标对应于最小化经验数据分布与模型分布之间的交叉熵；等价地，也是在最小化 $KL(p_\text{data} || p_\theta)$。
