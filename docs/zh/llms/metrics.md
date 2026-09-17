# 指标

本页定义后续 AI Infra 笔记中会反复使用的延迟和吞吐指标。这些指标是运行时行为与用户可感知性能之间最直接的接口。

先讲指标的动机很简单：大多数 serving 优化本身没有意义，只有当它改变了用户体验或系统容量时才有意义。指标提供了描述这种变化的语言。没有 TTFT、TPOT 和 TPS，就很难说明一个调度策略究竟是改善了响应速度，还是只是把延迟转移到了别处；也很难判断一个内存优化到底是增加了并发，还是只是改变了资源分配方式。

## 延迟

延迟是处理请求并生成输出 token 所需的时间。在 LLM 推理中，延迟通常通过两个主要指标来衡量。

之所以要把延迟拆成多个指标，而不是只看一个端到端数字，是因为不同技术优化的是不同阶段的瓶颈。有些技术主要改善模型开始响应的速度，有些技术主要改善稳态生成速度。

### 首 token 时间（TTFT）

首 token 时间（TTFT）是指从推理服务器收到请求，到第一个输出 token 被生成并返回给客户端之间的延迟。它衡量模型开始响应的速度，是流式 LLM 系统中最重要的用户感知指标之一：

$$
\text{TTFT} = t_{\text{first token emitted}} - t_{\text{request received}}
$$

TTFT 主要反映 prefill 阶段以及调度和框架开销。一个常用分解方式是：

$$
\small
\begin{aligned}
\text{TTFT} = t_{\text{queue}} + t_{\text{input processing}} + t_{\text{prefill compute}} + t_{\text{first decode step (optional)}} + t_{\text{framework overhead}}
\end{aligned}
$$

其中：

- $t_{\text{queue}}$ 是请求在被调度器接纳执行前的排队时间。
- $t_{\text{input processing}}$ 包括 tokenization、多模态预处理以及请求级格式化。
- $t_{\text{prefill compute}}$ 是对全部输入 token 执行前向计算并填充 KV cache 的时间。
- $t_{\text{first decode step (optional)}}$ 出现在将 prefill 与首个 decode step 分离的实现中。
- $t_{\text{framework overhead}}$ 包括 kernel 启动、同步、内存分配、采样和流式传输等额外开销。

在长上下文输入下，TTFT 通常由 $t_{\text{prefill compute}}$ 主导。

### 每输出 token 时间（TPOT）

每输出 token 时间（TPOT）衡量在首 token 发出之后，平均生成每个 token 所需的延迟。它刻画自回归模型的稳态 decode 性能。对于生成 $N$ 个输出 token 的请求：

$$
\text{TPOT} = \frac{t_\text{last token emitted} - t_\text{first token emitted}}{N - 1}
$$

等价地，若端到端总延迟为 $T_{\text{total}}$：

$$
\text{TPOT} = \frac{T_{\text{total}} - \text{TTFT}}{N - 1}
$$

在稳态解码中，每个输出 token 的时间可分解为：

$$
\text{TPOT} = t_{\text{decode compute}} + t_{\text{communication}} + t_{\text{scheduling}} + t_{\text{framework overhead}}
$$

其中：

- $t_{\text{decode compute}}$ 是利用 KV cache 完成一次 decode step 的前向计算时间。
- $t_{\text{communication}}$ 表示跨 GPU 的集合通信、expert 路由，以及由运行时行为触发的主机-设备数据交换。
- $t_{\text{scheduling}}$ 包括 batching、协调、准入控制与 KV cache 管理。
- $t_{\text{framework overhead}}$ 包括采样、kernel 启动、同步和 token 流式输出等辅助开销。

与 TTFT 不同，TPOT 不需要重新计算完整 prompt，但它仍取决于有效上下文长度，因为每个 decode step 都需要对全部已缓存 token 执行 attention。

## 吞吐

### 每秒 token 数（TPS）

每秒 token 数（TPS）用于衡量系统吞吐，即在给定 serving 配置下每秒可处理或生成多少 token。在时间区间 $\Delta t$ 内：

TPS 的动机来自容量规划。当系统需要同时服务很多请求时，核心问题不再只是“单个请求多快结束”，而是“当前部署每秒到底能稳定吞下多少工作量”。

$$
\text{TPS} = \frac{N_\text{tokens}}{\Delta t}
$$

很多情况下，TPS 特指 decode 吞吐：

$$
\text{TPS}_\text{decode} = \frac{N_\text{output tokens}}{\Delta t}
$$

对于单条顺序解码请求，它近似是 TPOT 的倒数：

$$
\text{TPS}_\text{per request} \approx \frac{1}{\text{TPOT}}
$$

在具有 continuous batching 和并发解码的实际系统中，总吞吐会随有效 batch 大小 $B$ 扩展。对于单个副本：

$$
\text{TPS}_\text{replica} \approx \frac{B}{\text{TPOT}}
$$

若有 $R$ 个数据并行副本：

$$
\text{TPS}_\text{total} \approx R \times \frac{B}{\text{TPOT}}
$$

对于长上下文或多模态负载，也常常需要考虑端到端吞吐：

$$
\text{TPS}_\text{end-to-end} = \frac{N_\text{prompt tokens} + N_\text{output tokens}}{\Delta t}
$$

总体而言，TPS 受模型规模、并行策略、有效 batch 大小、kernel 效率、互联带宽，以及 prefill 与 decode 负载相对占比的影响。
