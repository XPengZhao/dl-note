# 并行策略

本页整理 LLM 系统中最常见的并行维度，并重点从推理场景解释它们各自改变了什么。最核心的区分在于：外层维度通常复制工作负载，内层维度通常切分模型执行。

写这一页的动机在于，部署决策经常被表述成 `DP2TP2`、`TP4` 这种配置标签，但这些标签背后其实对应着不同的系统问题。有些并行方式主要解决吞吐不足，有些主要解决单副本显存不够，还有一些虽然理论上能扩展，却会引入足够大的通信开销以至于收益被抵消。不把这些问题先讲清楚，并行配置就很难真正分析。

## 数据并行（Data Parallelism, DP）

Data Parallelism（DP）是机器学习系统中最传统、也最直观的分布式执行形式。它的核心思想很简单：在多个设备上实例化同一个模型，同时将输入工作负载分片到这些实例上。每个 worker 在不同的数据子集上执行相同的计算图。从这个意义上讲，它只改变了计算到硬件的映射方式。

DP 在推理中的动机很直接：当单个副本已经放得下模型，但吞吐仍然不足以吸收请求流量时，最自然的扩容方式就是增加更多副本来并行服务请求。

由于参数是复制而不是切分，DP 不会降低单个设备上的内存占用。这一特性界定了它的实际适用范围：当模型本身已经可以放入单卡显存时，DP 很有效；但如果模型本身超过单设备容量，DP 就不够用。因此，它的主要价值在于通过副本扩展吞吐和服务并发。

在训练中，DP 常用于增大有效 batch size 并提升硬件利用率。每个 worker 处理 mini-batch 的一个分片，计算本地梯度，然后参与同步，使所有模型副本在优化步之后保持数值一致。在推理中，同样适用这种副本机制，但不再有梯度交换。取而代之的是，每个 worker 处理独立的请求子集或 token batch。因此在推理负载下，DP 应主要被理解为一种流量分发机制，而不是模型切分机制。

在混合部署中，DP 往往与 Tensor Parallelism（TP）和 Expert Parallelism（EP）并存。在这种配置下，DP replica 不应被字面地理解为单张 GPU。一个 DP 单元本身可能是一个多 GPU 组，其内部执行由 TP 或 EP 组织。从这个视角看，DP 定义的是工作负载复制的外层结构，而 TP 与 EP 决定每个副本内部如何执行计算。

### 与 DP 相关的集合通信

与 DP 相关的通信行为，本质上取决于系统是在训练还是推理。

在分布式训练中，每个 DP worker 基于本地 mini-batch 分片计算梯度。由于每个 worker 都维护同一套参数，这些梯度必须在每个 step 后同步，以保证副本间一致性。用于这一目的的标准集合通信原语是对梯度执行 all-reduce。在一些实现中，也会用 reduce-scatter 后接 all-gather 来实现同样效果，但目标不变：聚合所有 DP rank 的梯度贡献，并在每个 worker 上得到一致的参数更新。

相比之下，推理不涉及反向传播或优化器 step。每个 DP rank 服务于独立的一部分请求流，因此 DP 本身不会引入强制性的逐层集合通信。在推理场景下，它主要充当调度与流量切分机制，而不是同步密集型执行策略。

即便如此，基于 DP 的推理系统也并非完全无通信。运行时仍需要系统级协调，例如请求分发、负载均衡、准入控制与结果收集。实践中，这类协调常依赖 all-gather 等通信模式，在各 rank 间交换元数据、状态或调度信息。这些操作属于服务框架层，而非模型数值执行本身。

在混合 MoE 部署中，DP 在推理阶段通常只引入有限通信开销，而每个 DP 单元内部的并行机制往往更重通信。Tensor Parallelism 通常依赖 TP 组内的 all-reduce 或 all-gather 等集合通信，而 Expert Parallelism 会引入跨设备的 token 路由以访问不同专家。因此，分析混合系统中的通信时，应区分外层 DP 维度与内层模型并行维度：DP 负责工作负载分发，而主导通信成本的通常是 TP 或 EP。

## 张量并行（Tensor Parallelism, TP）

张量并行（TP）通过对单层计算中的权重张量进行切分，将计算分布到多个设备上，而不是在每个 rank 上复制完整的层参数。在大规模 Transformer 模型中，这种方式通常应用于注意力层和 MLP 层中的投影矩阵。其核心目标是降低单设备的参数与激活内存占用，从而能够运行那些在单个加速器上无法容纳的超大规模模型层。

TP 的动机与 DP 不同。它通常在单个加速器无法舒适容纳模型，或者减少单卡权重占用能够为激活和 KV cache 腾出更多空间时才真正变得有价值。

### Row Parallel 与 Column Parallel

考虑线性变换：

$$
Y=XW,
$$

其中 $X\in\mathbb{R}^{N\times H_{\mathrm{in}}}$，$W\in\mathbb{R}^{H_{\mathrm{in}}\times H_{\mathrm{out}}}$，$Y\in\mathbb{R}^{N\times H_{\mathrm{out}}}$。$N$ 可以表示本轮参与计算的 token 数量。Row Parallel 和 Column Parallel 描述的是逻辑权重矩阵 $W$ 的切分方向。

PyTorch 的 `Linear` 通常把权重保存为 $[H_{\mathrm{out}},H_{\mathrm{in}}]$，前向计算实际使用 $W^\mathsf{T}$。因此，代码中参数张量的行列方向看起来可能与数学定义相反。判断切分方式时，更可靠的标准是看它切分输入维度还是输出维度。

| 布局 | $W$ 的逻辑切分 | 输入布局 | 每个 rank 的输出 | 常见前向通信 |
| --- | --- | --- | --- | --- |
| Column Parallel | 切分 $H_{\mathrm{out}}$ | $X$ 通常复制 | 不同的输出特征切片 | 通常无通信，需要完整输出时 all-gather |
| Row Parallel | 切分 $H_{\mathrm{in}}$ | $X$ 按输入特征切分 | 同一个完整输出的部分贡献 | all-reduce，或 sequence parallel 下 reduce-scatter |

Column Parallel 沿输出维度切分 $W$：

$$
W=[W^{(0)},W^{(1)},\ldots,W^{(p-1)}],
\qquad
W^{(r)}\in\mathbb{R}^{H_{\mathrm{in}}\times H_{\mathrm{out}}/p}.
$$

每个 rank 接收相同的 $X$，只计算一部分输出特征：

$$
Y^{(r)}=XW^{(r)}
\in\mathbb{R}^{N\times H_{\mathrm{out}}/p}.
$$

这些局部结果是 $Y$ 的不同切片，不需要彼此求和：

$$
Y=[Y^{(0)},Y^{(1)},\ldots,Y^{(p-1)}].
$$

如果下一个算子能够直接消费分片后的输出，Column Parallel 后不需要通信。只有后续计算要求每个 rank 都获得完整 $Y$ 时，才需要通过 all-gather 拼接这些切片。

Row Parallel 沿输入维度切分 $W$。输入 $X$ 也沿相同维度切分：

$$
X=[X^{(0)},X^{(1)},\ldots,X^{(p-1)}],
$$

$$
W=
\begin{bmatrix}
W^{(0)}\\
W^{(1)}\\
\vdots\\
W^{(p-1)}
\end{bmatrix},
\qquad
W^{(r)}\in\mathbb{R}^{H_{\mathrm{in}}/p\times H_{\mathrm{out}}}.
$$

每个 rank 计算：

$$
Z^{(r)}=X^{(r)}W^{(r)}
\in\mathbb{R}^{N\times H_{\mathrm{out}}}.
$$

$Z^{(r)}$ 不是最终输出的特征切片，而是每个输出元素的一部分贡献。完整结果为：

$$
Y=\sum_{r=0}^{p-1}Z^{(r)}.
$$

如果后续计算要求每个 rank 都持有完整 $Y$，这里需要一次 all-reduce。如果系统希望下一阶段沿 token 维度保持 sequence-parallel 状态，则可以使用 reduce-scatter，在完成求和的同时把不同 token 分给不同 rank。

### Transformer 中的成对布局

Transformer 通常把 Column Parallel 和 Row Parallel 成对使用，使两个线性层之间的中间激活保持分片状态。这样可以避免先 all-gather，再为下一层重新切分。

Attention 中的典型路径为：

```text
完整 hidden state
    -> Column Parallel QKV projection
    -> 每个 rank 得到一部分 attention heads
    -> 本地 attention
    -> Row Parallel output projection
    -> all-reduce
    -> 完整 hidden state
```

Column Parallel QKV 投影按输出 heads 切分。每个 rank 可以直接在本地 heads 上计算 attention，不需要先恢复全部 heads。输出投影再以 Row Parallel 方式消费这些本地结果。各 rank 经过输出投影得到的是同一个 hidden state 的部分贡献，因此通过 all-reduce 求和。

这里相加的不是不同 attention heads 的原始输出。设拼接后的多头输出为：

$$
A=[A^{(0)},A^{(1)},\ldots,A^{(p-1)}],
$$

输出投影权重按输入维度切分为：

$$
W_O=
\begin{bmatrix}
W_O^{(0)}\\
W_O^{(1)}\\
\vdots\\
W_O^{(p-1)}
\end{bmatrix}.
$$

分块矩阵乘法满足：

$$
AW_O=\sum_{r=0}^{p-1}A^{(r)}W_O^{(r)}.
$$

all-reduce 合并的是各 rank 经过 $W_O^{(r)}$ 投影后的部分结果。这与单卡上先拼接全部 heads，再执行完整输出投影在数学上等价。

FFN 使用相同的配对方式：

```text
完整 hidden state
    -> Column Parallel gate/up projection
    -> 分片的 intermediate state
    -> 本地激活函数
    -> Row Parallel down projection
    -> all-reduce
    -> 完整 hidden state
```

第一层投影沿 intermediate dimension 切分。激活函数可以在每个 rank 的局部切片上独立执行。`down_proj` 随后计算完整 hidden state 的部分贡献，并通过 all-reduce 恢复完整输出。Attention 和 FFN 都只在成对布局的末端通信一次，而不需要在两个投影之间物化完整的中间激活。

### TP 的通信边界

只看推理前向计算，Column Parallel 本身不必然对应 all-gather。它产生的是输出特征切片，能否省略通信取决于下一算子是否支持这种布局。Row Parallel 通常对应 all-reduce，因为各 rank 产生的是同一输出的部分和。启用 sequence parallel 后，Row Parallel 末端也可能使用 reduce-scatter。

TP 的性能取决于本地矩阵乘与集合通信之间的平衡。随着 TP degree 增大，每个 rank 的权重和 GEMM 规模都会减小，但每层 Attention 和 FFN 末端的同步仍位于执行关键路径。TP 通常部署在同一节点或高带宽互联的设备组内。若 TP group 跨越较慢的网络，集合通信延迟很容易抵消参数切分带来的收益。

### 推理视角下的解释

在 serving 场景中，DP 和 TP 往往承担不同职责。

- DP 的主要作用是通过增加副本数来扩展吞吐。
- TP 的主要作用是通过切分参数来改善单副本的模型适配性和显存预算，但代价是引入层内集合通信。
- 增大 DP 会复制更多权重副本，也可能打散 prefix cache 的局部性。
- 增大 TP 会降低单卡权重占用并扩大单副本的 KV 预算，但 TP 过大时通信可能成为瓶颈。

因此，推理部署中的并行决策通常是在“副本级并发能力”和“单副本内存余量”之间做权衡。

## 案例：GLM-5.2 的 TP8 与 EP8

在 GLM-5.2 这类稀疏 MoE 模型中，TP 的实际边界比标准 Transformer 更复杂。下面的映射对应当前 vLLM 中 `GlmMoeDsaForCausalLM` 的实现，该实现复用了 DeepSeek-V2/V3 风格的 MLA 与稀疏注意力路径。这里考虑一个由 8 个 rank 组成的单副本，同时配置 `TP8` 并在同一组 rank 上启用 Expert Parallelism（EP）。

8 个 TP rank 会处理同一条请求和同一个 token batch。TP 不是请求级并行：它切分一次 forward 内部的部分模型张量与算子，而每条请求仍会占用整个 TP group。

| 模块 | TP8 下的放置方式 | 主要通信 |
| --- | --- | --- |
| Query heads 与 MLA B 投影 | 在 8 个 rank 间切分 | 本地 attention 后归约输出 |
| Attention 输出投影 | Row-sharded | TP all-reduce |
| Dense MLP gate/up 投影 | Column-sharded | 扩展边界通常无通信 |
| Dense MLP down 投影 | Row-sharded | TP all-reduce |
| Token embedding 与 LM head | 按词表切分 | 按 logits 处理需要进行 gather 或归约 |
| RMSNorm、RoPE 与残差运算 | 复制 | 无通信或与相邻算子融合 |
| MLA A 投影 | 复制 | 无通信 |
| DSA indexer | 大部分计算复制 | 与索引选择相关的同步 |
| 开启 EP 后的 routed experts | 按 EP rank 切分专家 | Token all-to-all 与结果回传 |
| MLA 与 indexer KV 状态 | 每个 TP rank 保留完整 token 范围 | TP 不沿序列维切分 |

### MLA Attention

设模型共有 $H$ 个 query heads，TP degree 为 $p=8$，则每个 rank 计算

$$
H_{\mathrm{local}} = \frac{H}{8}
$$

个 query heads。在 vLLM 中，`q_b_proj` 和 `kv_b_proj` 使用 column-parallel 投影，其输出维度被切分，每个 rank 只生成本地 heads 所需的 Q/K/V 特征。随后，attention kernel 在本地 heads 上执行。输出投影使用 row-parallel 布局，各 rank 产生的部分结果通过 all-reduce 求和，恢复完整 hidden state 后再进入下一个 block。

MLA 在按 head 展开的 B 投影之前，还包含低秩 A 投影。当前实现中的 `q_a_proj` 和 `kv_a_proj_with_mqa` 都是 replicated linear。每个 rank 保存相同的 A 投影权重，并重复执行这部分计算。因此，TP8 切分了按 head 组织的 B 投影和本地 attention，但不会把所有 attention 计算都缩小为原来的八分之一。

### Dense MLP 层

Dense MLP 使用标准的成对 TP 布局。Gate 与 up 投影融合为 column-parallel linear，沿 intermediate dimension 切分；down 投影使用 row-parallel 布局，并通过 TP 归约合并各 rank 的部分输出。两个投影之间的中间激活可以保持分片状态，从而避免在扩展边界执行不必要的 all-gather。

这种布局降低了每个 rank 的 MLP 参数占用和 GEMM 规模。随着 TP degree 增大，本地 GEMM 逐渐变窄，而 down projection 后的归约仍位于执行关键路径，因此扩展效率会逐步下降。

### DSA Indexer 与 KV 状态

稀疏注意力的 indexer 是按 head 切分模式中的一个例外。当前实现中，indexer 的 query 投影采用复制布局，融合后的 key/weight 投影也显式关闭了 TP。因此，各 rank 会执行大致相同的 indexer 投影和 top-$k$ 选择。

KV 内存也存在相同边界。MLA 为每个缓存 token 保存一个压缩 latent vector。每个 TP rank 上的本地 query heads 都需要访问完整历史 token 范围，因此各 rank 都会为该范围保存压缩 MLA 状态；indexer cache 同样覆盖完整序列。TP8 不会把一段 160k token 的 KV 历史自动变成 8 份彼此独立的 20k token 分片。

因此，TP 与 Context Parallelism 解决的是不同问题。TP 切分 heads 和投影矩阵，DCP/CP 则沿 sequence 或 KV 维度切分。TP8 可以通过降低单 rank 权重占用，为 KV cache 留出更多 HBM，但不会仅凭 TP 就把 8 张 GPU 的 KV 容量合并为 8 倍上下文长度。

### Expert Parallelism 下的 MoE 层

启用 EP 后，routed experts 的放置规则与 attention 和 dense MLP 不同。Router 采用复制布局，以便各 rank 得到一致的路由决策；物理 routed experts 则分布到 EP group 中。Token 会被发送到持有所选专家的 rank，在本地完成 expert GEMM，再返回原执行 rank。

对 `TP8 + EP8` 而言，整体执行结构为：

- attention 投影与 heads 使用 TP8；
- dense MLP 层使用 TP8；
- routed experts 使用 EP8 进行专家切分；
- router、MLA A 投影和大部分 indexer 计算保持复制；
- shared experts 通常仍采用 tensor-parallel 路径，除非后端将其融合进 MoE 实现。

同一个 Transformer layer 中由此存在两类通信：attention 与 dense projection 触发 TP collectives，routed MoE 则触发 token all-to-all。提高 TP 或 EP degree 只会改变该并行维度负责的部分；复制算子和通信边界仍然存在，因此单请求延迟不会随 GPU 数量线性下降。

### Serving 层面的影响

上述放置方式带来以下运行特征：

- 一条请求会占用 8 张 GPU，TP8 不会提供 8 个独立请求槽位。
- Attention heads 和大矩阵投影在每个 rank 上的计算规模减小，但复制的 MLA 与 indexer 计算限制了扩展效率。
- 每个 decode step 都需要执行细粒度 TP collectives，因此低并发 decode 通常比 prefill 更难随 TP 扩展。
- EP 减少 routed-expert 参数复制，但 token routing 会引入 all-to-all，并受到专家负载不均衡的影响。
- MLA KV 容量仍然受单 rank 约束。若要增加 active context 容量，需要为每个 rank 留出更多 HBM、降低 KV 精度、采用 sequence/context parallelism，或者使用能够处理 active KV state 的 offloading 方案。

AWQ 等权重量化会改变分片矩阵的存储形式和执行 kernel，但不会从根本上改变上述并行放置关系。

## 一个 8-GPU 推理例子

理解 DP 和 TP，最直接的方式是固定硬件，只改变 8 张 GPU 被组织成多少个副本。假设有一台 8-GPU 服务器，节点内互联较快，那么常见的几种组织方式是：

- `DP8TP1`：8 个单 GPU 独立副本
- `DP4TP2`：4 个副本，每个副本用 2 张 GPU
- `DP2TP4`：2 个副本，每个副本用 4 张 GPU
- `DP1TP8`：1 个副本，占用全部 8 张 GPU

这个对比只有在这些布局都放得下目标模型和上下文窗口时才有意义。如果 `TP1` 下模型或 KV 状态已经放不下，那么高 DP 的方案就不是实际可选项。

这里最关键的区别是：DP 把 GPU 花在“更多独立副本”上，而 TP 把 GPU 花在“把单个副本做大”上。因此，哪种方案更优，本质上取决于当前工作负载更受哪一面约束：单请求延迟、单副本内存余量，还是多请求并发。

这里还有一个在长上下文 serving 里很关键、但很容易被忽略的点：TP 改变的不只是单个副本的计算模式。由于它降低了单卡权重占用，它也可能给同一个副本腾出更多 KV cache 空间。更大的 KV 预算意味着这个副本可以同时容纳更多长请求、降低 preemption 风险，并形成更大的稳定 batch。也就是说，TP 有时提升吞吐，不是因为单个 decode step 本身更快，而是因为副本终于能挂住足够多的 active work，把 GPU 真正打满。

因此，在 8 张 GPU 上做推理调优时，通常是在同时平衡三个问题：

- 节点应该暴露多少个独立副本？
- 每个副本需要多大的 KV 余量？
- 当前 workload 的瓶颈到底是通信、内存压力，还是 active work 不够？

### Prefill 很长：`40k + 1k`

这个场景的主导成本是吃入超长 prompt。真正昂贵的是 prefill，因为系统需要先处理很长的输入上下文，decode 反而不是主导项。

- 在低并发下，TP 往往更有吸引力，因为多个 GPU 可以协同完成同一个长 prefill。`DP8TP1` 虽然副本多，但如果同时只有一两个请求，绝大多数副本都会闲着，而某一张 GPU 独自承担全部 prompt 成本。
- 在高并发下，DP 又会重新变得有竞争力，因为许多长 prefill 可以分散到不同副本上并行推进。如果每个请求都已经能稳定放下，`DP8TP1` 或 `DP4TP2` 往往比很大的 TP 组具有更高的总吞吐。
- 在这个场景里保留一定 TP，通常不是为了让 decode 更快，而是为了给单副本争取更大的内存余量。超长 prompt 会迅速消耗 KV 预算，而适度 TP 能让更长上下文真正变成可运行配置。
- 这个内存效应也会直接改变并发能力。面对超长 prompt 时，`TP1` 可能迫使每个副本只敢接很小的 active batch，才能避免 preemption。适度增加 TP 往往能扩大单副本的 KV 预算，使它在同一时刻驻留更多 prefill 请求，从而把真实吞吐拉起来。

因此，prefill 很重的 serving 往往容易落在中间方案，例如 `DP4TP2`。当 `TP1` 的内存已经太紧，而 `TP4` 或 `TP8` 又牺牲了太多副本数时，这类折中方案通常更稳。

### Decode 很长：`1k + 10k`

这个场景的主导成本是长时间的自回归生成。prompt 不长，但系统要重复执行大量 decode steps。

- 在高并发下，如果模型本身放得下，DP 往往是最强的选择。`DP8TP1` 最大化了独立 decoder 的数量，也避免了每个 decode step 都要支付 TP 层内集合通信。
- 在低并发下，过大的 TP 组通常也帮不上太多，除非它们是内存所必需的。单请求 decode 的每一步都很窄，把它摊到很多 GPU 上时，同步开销常常增长得比本地计算缩短得更快。
- 如果输出很长，以至于 KV 压力成了主导瓶颈，那么适度 TP 仍可能间接变好，因为它扩大了单副本可承载的 KV 预算，降低了 preemption 风险。但这本质上是内存收益，不是纯算力收益。
- 这一点在工程上很重要。decode-heavy 服务从单步看往往算得不重，但如果每个副本始终被 KV 卡住，它就永远积累不起足够多的 active sequences，也很难把设备打到 compute-bound。这时增加一定 TP，虽然引入了通信，却可能通过提升稳定 decode 并发带来更高的真实吞吐。

这也是为什么在线 decode-heavy serving 往往倾向于在内存允许的前提下尽量多做 DP。一旦模型已经放得下、KV 也稳定，继续增加 TP 通常是在用通信换掉吞吐。

### Prefill 和 Decode 都很长：`10k + 10k`

当输入和输出都很长时，单纯用 “DP 好” 或 “TP 好” 都不够。系统既要承受长上下文带来的 prefill 成本，也要承受长输出尾巴带来的 decode 压力。

- 在小并发或中等并发下，像 `DP4TP2` 这样的混合布局往往是比较稳的折中。它一方面保留了多个独立副本，另一方面又给每个副本比 `TP1` 更多的内存余量。
- 在大并发下，判断标准通常是谁先撞墙。如果 `DP8TP1` 已经开始 preempt，或者为了不爆 KV 不得不把准入并发压得很低，那么增加 TP 反而可能提升真实吞吐，即使它引入了通信。如果 `DP8TP1` 本身已经很稳定，它通常仍然保有最高的总吞吐。
- `DP1TP8` 一般只有在单个副本确实需要整台机器时才合理，例如模型本身很大，或者目标上下文窗口极端大。否则它牺牲的外层并发通常太多。

这也是“纸面并发”和“有效并发”差别最大的场景。`DP8TP1` 从副本数上看并发更高，但如果每个副本为了守住 KV 预算只能把 admitted batch 压得很小，整台机器仍然可能打不到 compute-bound。副本数更少但更大的 TP 组，有时反而能维持更多真正有用的 in-flight work。

### 权衡总结

对一台 8-GPU 服务器上的推理来说，常见的经验规律通常是：

- 更多 DP 在“独立请求足够多，能把多个副本都喂满”时更有利。
- 更多 TP 在“单副本太受内存限制”或“单个请求太重，需要多张 GPU 协作”时更有利。
- 更多 TP 有时也会在“超长请求让 `TP1` 副本的 KV 预算太紧，导致挂不住足够 active work”时更有利。此时 TP 买到的是内存带来的并发，而不是单步算力加速。
- prefill 很重、并发又不高时，系统更容易被推向一定程度的 TP。
- decode 很重、并发又很高时，系统更容易被推向更多 DP。
- 输入输出都长的场景往往落在中间地带，`DP4TP2` 或 `DP2TP4` 往往比两端更容易稳定。

因此，推理部署里真正重要的问题通常不是抽象地问“DP 和 TP 谁更好”，而是先问：在这套硬件和这类 workload 上，系统最先撞到的墙到底是什么，是单请求延迟、单副本 KV 容量，还是整体并发能力。DP 和 TP 本质上是在用两种不同方式移动这堵墙。
