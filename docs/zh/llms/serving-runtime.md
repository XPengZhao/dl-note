# 推理运行时

本页关注模型结构固定之后的运行时机制，即 serving 系统如何调度请求、管理内存并在负载下维持稳定性。重点不是模型语义本身，而是运行时对延迟、吞吐和显存压力的影响。

把运行时单独拿出来讲的动机在于，很多 serving 瓶颈并不是模型结构本身带来的，而是部署之后才出现的。即便模型完全不变，系统仍然必须决定请求如何接纳、长 prompt 如何与 decode 交织执行，以及内存压力出现时如何处理。运行时技术之所以值得研究，正是因为它们会在真实负载下改变“同一个模型”的实际行为。

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

### 准入控制与稳定区间

在实践中，serving 的稳定性很大程度上取决于运行时的准入控制参数，例如 `max-num-seqs`、`max-num-batched-tokens` 以及运行时可使用的 GPU 显存预算。这些参数共同决定了 engine 能接纳多少并发工作，以及系统距离 KV 容量墙还有多远。

准入控制的重要性在于，朴素地把并发开到最大通常会适得其反。系统一旦进入持续 preemption、swap 或严重排队的区间，表面上的并发更高，实际吞吐和延迟却可能同时变差。

工程上的目标通常不是把理论并发开到最大，而是找到一个稳定区间：GPU 利用率较高，同时又不会持续触发 preemption、swap 或明显的尾延迟抖动。因此，运行时调优不能脱离 KV cache 来看。batching 策略、调度行为和内存压力本身就是耦合的。

### 跨 Stream 的 Tensor 复用

如果一个 tensor 在某个 CUDA stream 上生成，而后在另一个 stream 上被消费，默认情况下并不需要额外的 device-to-device copy。只要程序通过 event 或 stream 同步建立了正确的先后关系，同一块存储就可以被跨 stream 复用。此时主要成本来自同步以及由此损失的重叠执行，而不是张量本身必须被复制一份。

这个话题有价值，是因为 stream 级执行经常被误解。很多人看到工作跨 stream 流动，会先猜测代价来自 copy；但在不少情况下，真正的成本其实是依赖建立、同步以及重叠执行能力的损失。

只有当程序显式创建新 buffer、改变 layout，或者在不同设备之间移动数据时，额外 copy 才是必要的。因此，从运行时分析的角度看，跨 stream 的 tensor 使用更像是依赖管理问题，而不是张量 payload 必然复制的问题。
