# 推理系统中的 Roofline Model

延迟和吞吐指标告诉我们用户最终看到了什么，但它们本身并不能解释为什么某个 serving 配置在继续调优后还是提不上去。一个系统可能表现为 TPOT 很高、TPS 很低，或者 batch 变大后扩展性依然很差，但这些现象背后仍然有一个核心问题没有被回答：当前 workload 主要受限于算力，还是受限于数据搬运？`Roofline model` 正是用来回答这个问题的一种紧凑方法。

在推理场景里，这一点尤其重要，因为不同阶段对硬件的施压方式并不相同。Prefill 和 decode 可以运行在同一个模型、同一张 GPU 上，却落在完全不同的性能区间里。很多实际优化，包括 batching、KV 压缩、量化和 speculative decoding，本质上都可以理解为试图把 workload 从一个更低的 ceiling 推向一个更高的 ceiling。

## 为什么 Roofline 重要

当人们说某个推理优化“提升了利用率”时，这句话往往还不够完整。利用率提高，可能来自完全不同的原因：

- 同样的数据搬运量上做了更多计算
- 为得到同样的数学结果，搬运的数据更少了
- 固定运行时开销被更宽的一步摊薄了
- 原来的主瓶颈被解除，系统转而受另一个 ceiling 限制

`Roofline model` 的价值，就在于它为这些情况提供了一套共同语言。它不会替代 request-level 分析、scheduler 分析或通信分析，但它提供了一个从硬件第一性原理出发的设备侧性能上界视角。

## 基本模型

经典 roofline model 把可达到性能与两个硬件上限联系起来：

- 峰值计算吞吐
- 峰值内存带宽

记：

- $P$ 为实际达到的性能，单位可写为 FLOPs/s
- $P_{\max}$ 为峰值计算吞吐
- $B_{\max}$ 为峰值内存带宽，单位为 bytes/s
- $I$ 为 arithmetic intensity，即每搬运 1 byte 数据对应的 FLOPs

则可达到性能满足：

$$
P \le \min \left(P_{\max},\; I \cdot B_{\max}\right)
$$

这个表达式把系统分成两个典型区间。

### Memory-Bound 区间

当 arithmetic intensity $I$ 较小时，$I \cdot B_{\max}$ 会低于峰值算力，此时性能主要受限于数据搬运速度：

$$
P \approx I \cdot B_{\max}
$$

在这个区间里，单纯提高理论算力通常帮助不大，除非带宽本身提高了，或者 workload 的 arithmetic intensity 提高了。

### Compute-Bound 区间

当 arithmetic intensity 足够高时，由带宽导出的上界会上升到峰值算力之上：

$$
I \cdot B_{\max} \ge P_{\max}
$$

这时 workload 更接近计算 roof：

$$
P \approx P_{\max}
$$

在这个区间里，性能提升更依赖 kernel 效率、tensor core 利用率和并行执行质量，而不只是减少内存流量。

## Arithmetic Intensity

Arithmetic intensity 是连接 workload 结构与硬件上限的桥梁：

$$
I = \frac{\text{FLOPs}}{\text{Bytes moved}}
$$

这里的重点并不是为每一个 kernel 都精确算出一个完美值，而是判断某个 workload 相对于它必须读写的数据量，到底做了多少数学计算。

Arithmetic intensity 较低的 workload 往往会：

- 大量时间耗在等待内存流量上
- 难以从更高的峰值 FLOPs 中直接受益
- 更依赖减少 bytes moved 或提升数据复用的技术

Arithmetic intensity 较高的 workload 往往会：

- 更容易用好 tensor cores 或 SIMD 单元
- 更直接地受益于更强的 kernel 和更多可用算力
- 在 batch 或 step width 增加时，不那么快出现收益展平

## 把 Roofline 映射到推理

Roofline 在 AI Infra 里的价值，不在于那张抽象图本身，而在于它解释了为什么不同推理阶段会有完全不同的性能表现。

### Prefill

Prefill 会成批处理整个 prompt。序列长度更大，矩阵形状更宽，硬件在单次执行中通常也能拿到更多并行工作。因此，prefill 往往比 decode 更接近 compute-bound 一侧，尤其是在 prompt 较长或 batch 较大时。

这并不意味着 prefill 总是 compute-bound。小 batch、多模态预处理开销，或分布式通信压力都可能让其它 ceiling 更早成为主导。更稳妥的说法是：prefill 通常比 decode 更有利于计算单元被充分利用。

### Decode

Decode 每次只执行一步自回归迭代。每一步都很窄、对延迟敏感，而且会反复访问之前所有 token 积累下来的 KV 状态。因此在实际 serving 中，decode 更容易落入 memory-bound 或 overhead-dominated 区间。

这也是在线推理常常让人感觉反直觉的原因：数学上更“小”的阶段，反而可能成为系统级主瓶颈。每个 decode step 的总计算量虽然比 prefill 小得多，但由于工作过窄、串行性太强，设备往往并没有被真正喂饱。

## Roofline 能解释什么，不能解释什么

Roofline 很有用，但它并不是完整的 serving 模型。

它比较擅长解释：

- 一个设备侧 workload 主要是受算力约束还是受内存流量约束
- 为什么更宽的 batch shape 往往能带来更好的性能
- 为什么有些优化在 decode 中更有效，而在 prefill 中不明显
- 为什么减少 bytes moved 有时和减少 FLOPs 一样重要

它不能完整解释：

- request queueing 和 scheduler delay
- 多副本之间 prefix routing 碎片化
- kernel 视角之外的 host-device synchronization 开销
- 分布式执行中的跨 GPU 通信 ceiling
- 应用层 fan-out 和编排开销

因此，更合理的使用方式是：把 roofline 当作更大 request lifecycle 里的一个局部性能模型，而不是用它去替代整个 lifecycle 分析。

## 用 Roofline 理解常见推理优化

一旦有了 roofline 视角，很多优化点就更容易归类了。

### KV Cache 与内存流量

Decode 往往需要读取大量缓存下来的 attention 状态，而单步新引入的计算却相对有限，这会把 arithmetic intensity 向下拉。KV 量化、更好的 cache layout，以及更高效的 block 复用之所以重要，正是因为它们可以在已经接近带宽 ceiling 的区间里降低内存压力或提高有效复用。

### Batching 与更宽的执行形状

更大的有效 batch 往往能提高利用率，因为 kernel 会更宽，固定开销也更容易被摊薄。从 roofline 角度看，batching 往往会把 workload 推向一个更有利于计算单元发挥的区间，并让实际性能点更接近上方的 envelope。

这也解释了为什么 batching 的收益最终会递减。一旦 workload 已经接近当前的主 roof，再继续加宽，边际收益就会明显变小。

### Quantization

量化有时被说成是计算优化，有时又被说成是内存优化。在推理场景里，它往往两者都是，但主收益取决于原始瓶颈在哪里。如果 workload 本来就是 memory-bound，那么更小的 activations、weights 或 KV tensors 往往能直接降低带宽压力和容量压力；如果 workload 已经接近 compute roof，那么除非低精度 kernel 也显著提升了实际计算吞吐，否则最终收益可能远小于理论想象。

### Speculative Decoding

更合适的理解方式，是把 speculative decoding 看成对 target model 工作粒度的一次改变。普通自回归 decode 中，target model 需要执行很多次窄而串行的 step；而 speculative decoding 会把多个候选 token 的验证合并成更宽的一步。

它可能带来收益，主要有两个相关原因：

- 固定开销被更多有效工作摊薄了
- 更宽的 verification step 往往比许多孤立的窄 step 更容易让硬件跑得高效

但这并不意味着 speculative decoding 可以被简化成“提高 arithmetic intensity”这一句话。更准确的说法是：它减少了 target-side 昂贵的串行 decode 次数，并且在基线 decode 路径本来就喂不饱硬件时，可能改善 target-side utilization。当 serving batch 本来就已经很大、target 路径已经接近其有效 roof 时，这种收益就会明显缩小。

## 超出单设备 Roofline 的部分

经典 roofline model 最自然的适用对象是一台设备或一类 kernel。真实推理系统往往还会撞上更高一层的 ceiling。

在分布式或 MoE serving 中，实际吞吐不仅可能受 HBM 带宽和单设备算力约束，还可能受以下因素限制：

- all-reduce 或 all-to-all 带宽
- collective start-up latency
- pipeline bubbles
- scheduler 层面的准入和 preemption 行为

因此，roofline 更适合被介绍成“第一层有用的 ceiling model”，而不是最终模型。它回答的是局部计算究竟更偏 bandwidth-limited 还是 compute-limited；而更广义的系统分析还需要继续判断：通信、runtime policy 或 workload shaping 是否施加了一个更低的 ceiling。

## 实际分析时该问什么

当一个推理优化声称自己能提速时，用 roofline 视角去读它，最先应该问的是四个问题：

1. 它减少的是 FLOPs、减少的是 bytes moved，还是主要在摊薄固定开销？
2. 它改善的是 prefill、decode，还是两者都有？
3. 当前基线 workload 是 memory-bound、compute-bound，还是被更高层的系统 ceiling 卡住？
4. 这个优化是解除了当前主瓶颈，还是只是把瓶颈转移到了别处？

这些问题往往比抽象地讨论某个技巧“好不好”更有用。一个 serving 优化只有在真正抬高了当前 workload 所受的活动 ceiling 时，才算在系统里产生了实际价值。

## 相关页面

- [指标](metrics.md) 定义 TTFT、TPOT 和 TPS，它们描述的是用户可见现象。
- [KV Cache](kv-cache.md) 解释为什么 decode 往往伴随明显的内存压力。
- [推理运行时](serving-runtime.md) 讨论 roofline 本身无法捕获的调度与准入效应。
- [解码与采样](decoding.md) 将 speculative decoding 作为改变 decode 工作形状的一个具体例子展开。
