# KV Cache

本页关注长上下文 serving 背后的对象和内存语义：KV cache 的规模估算、block/page 组织、prefix 复用，以及会改变 KV 开销的 attention 结构。

这个主题的重要性在于：长上下文 serving 的主要瓶颈往往不再只是模型权重，而是保存 attention 历史状态所需的 KV 内存。一旦 prompt 变长或者并发变高，KV cache 很快就会成为最主要的系统对象。它决定了多少请求能同时保持活跃，决定了什么时候会触发 preemption，也直接影响 DP 和 TP 这样的部署选择。

## KV 内存

### 内存开销

KV cache 大小可近似写为：

第一个需要回答的问题是：为什么 KV cache 会增长得这么快。原因在于它会随着上下文中保留的 token 数增长，并且每一层都要为这些 token 保存 key 和 value，因此即便是看起来不算极端的序列长度，也会迅速转化为很大的显存占用。

$$
\text{KV size} = 2 \times \text{num layers} \times \text{num heads} \times \text{seq len} \times \text{head dim}
$$

例如：

- 32 层、80 个头、`head_dim = 128`、`seq_len = 4K`
- FP16 KV cache：$2 \times 32 \times 80 \times 4K \times 128 \times 2$ bytes，约为 5.2 GB 每样本
- int8 KV cache：约减半，即约 2.6 GB 每样本

### Request、Token、Block 与 Page 的关系

在 serving 系统中，request 是业务层对象，token 是序列层对象，而 KV block 是运行时管理 KV cache 的基本内存对象。一个 request 对应一条逻辑 token 序列；随着 prefill 和 decode 推进，序列中的每个 token 都会在每一层生成对应的 key 和 value，并被追加到该 request 的 KV cache 中。若运行时设置的 block 大小为 $B$ 个 token，那么长度为 $L$ 的请求大约需要

$$
N_{\text{blocks}} = \left\lceil \frac{L}{B} \right\rceil
$$

个 KV block 来容纳其上下文状态。

因此，一个 request 并不是直接映射到一段单体的连续 KV 内存，而是映射到一串 block：

$$
\text{request} \rightarrow \text{token sequence} \rightarrow \text{KV blocks}
$$

最后一个 block 往往不会被完全填满，因此 block 大小本身决定了尾部浪费的粒度。block 较小时，内存利用更细；block 较大时，block table 更短、管理更简单，但缓存复用粒度也更粗。

在 vLLM 语境里，page 更多是对这种组织方式的描述，而不是独立于 block 的另一层对象。所谓 paged KV cache，指的是一个 request 的 KV 状态不再需要映射到一段物理连续的大内存，而是像分页内存一样由多个固定大小的 block 组成。因此，page 强调的是“分页化组织”，block 强调的是“实际分配单元”。

### 为什么需要 PagedAttention

PagedAttention 要解决的核心问题不是 block 内部如何存储，而是一个会持续增长的 request 应如何由多个 block 组成其逻辑连续的 KV 地址空间。

其背后的问题是，朴素的连续内存分配方式与真实 serving 负载的演化方式并不匹配。请求长度不同、decode 过程中还会继续增长，而且不同请求结束的时间也不同。如果运行时始终要求为每个请求保留一大段连续空间，就会很快出现碎片化和昂贵的数据搬移。

第一，它避免了为每个 request 预留一整段连续显存。如果 KV cache 必须整体连续，那么长短请求混跑时很容易产生外部碎片：系统可能总空闲显存仍然足够，却找不到一段足够大的连续区域容纳新请求。

第二，它避免了请求增长时的大规模数据搬移。自回归 decode 会不断向上下文末尾追加 token。如果采用连续大块分配，一个 request 增长到原空间无法容纳时，就需要将已有 KV 整体复制到更大的连续区域。PagedAttention 则只需为该 request 再挂接新的 block。

第三，它使共享、回收和调度都自然落在 block 粒度上。prefix cache 复用的是完整 block，内存回收和调度抢占也以 block 为单位进行。

### Block 内连续，Block 间非连续

PagedAttention 并不意味着 GPU 需要以完全随机的方式读取 KV。更准确地说，它实现的是“块内连续，块间可非连续”的组织。运行时通过 block table 维护从逻辑序列位置到物理 block 的映射关系；attention kernel 再依据这张映射表找到当前 request 对应的一串 physical blocks，并按 block 为单位执行读取和计算。

因此，PagedAttention 引入的是 block 级别的间接寻址，而不是 token 级别的完全离散访问。block 内部仍可以采用适合向量化和高带宽读写的规则布局，而不连续性主要体现在 request 所使用的 block 列表上。

### Prefix Cache 与 Block 粒度复用

在 vLLM 中，automatic prefix caching 的复用对象不是单个 token，而是完整 block。设 block 大小为 $B$，若两个请求共享前缀长度为 $L_{\text{shared}}$，则理论上最多只有

$$
\left\lfloor \frac{L_{\text{shared}}}{B} \right\rfloor
$$

个完整 block 能被直接复用。未填满的尾块通常不能直接作为缓存命中对象。因此，block 大小不仅影响内存分配粒度，也影响前缀缓存的复用粒度。

### Preemption 的含义

在 vLLM 中，preemption 指的是：当 GPU 上可用的 KV block 不足以容纳当前调度决策时，运行时临时中断某些正在服务的 request，并回收或迁移它们占用的 KV 资源，以便为其他 request 腾出空间。它本质上是一种内存压力下的调度退让机制，而不是模型数值层面的运算。

之所以要单独理解 preemption，是因为它标记了系统从健康运行区间进入内存受压区间的边界。服务即使在 preemption 发生后仍可能继续运行，但一旦频繁发生，往往意味着当前并发已经超过了这套 KV 预算能够高效支撑的范围。

从系统现象上看，preemption 往往意味着当前配置已经接近或撞上 KV 容量墙。典型触发因素包括：并发 request 数过高、单请求上下文过长、输出长度过大、TP 切分不足导致单卡 KV 预算偏小，或者 `max-num-seqs`、`max-num-batched-tokens` 设得过于激进。

## Attention 与 KV 效率

在 transformer 模型中，attention 模块计算：

$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

每个 token 会产生：

- **Q（Query）**，来自当前 token 的隐藏状态
- **K（Key）** 与 **V（Value）**，来自上下文并存储在 KV cache 中

MHA、MQA 和 GQA 的主要区别在于：相对于 Query 头数，使用了多少组 Key/Value 投影。

把这些 attention 变体放在 KV Cache 这一页讨论，是因为它们不仅是架构差异，也会直接改变 serving 成本。在推理场景中，MHA、MQA 和 GQA 决定了每个 token 需要保存多少 KV 状态，从而直接影响可支持的上下文长度和并发规模。

### MHA：Multi-Head Attention

每个 attention 头都有各自独立的投影：

$$
Q_i = X W_{Q, i}, \quad K_i = X W_{K, i}, \quad V_i = X W_{V, i}
$$

- Q、K、V 的头数相同。

### MQA：Multi-Query Attention

所有头共享一组 K 和 V 投影：

$$
K_{\text{shared}} = X W_K, \quad V_{\text{shared}} = X W_V
$$

- 有多个 Q 头，但只有一对共享的 K/V。
- 推理时，所有头复用相同的 K/V。
- 相对于标准 MHA，KV cache 占用显著更低。

### GQA：Grouped-Query Attention

Query 被分组，每组共享各自的 K/V：

$$
\text{Group } g: \quad Q_{g, *}, K_g, V_g
$$

- K/V 头数少于 Q 头数。
- GQA 在 MHA 的表达能力和 MQA 的内存效率之间做折中。

## 相关页面

- [指标](metrics.md) 定义 TTFT、TPOT 和 TPS。
- [推理运行时](serving-runtime.md) 讨论当 KV 内存成为瓶颈时的调度行为。
- [并行策略](parallelism.md) 讨论 DP/TP 如何改变显存预算、吞吐和通信。
