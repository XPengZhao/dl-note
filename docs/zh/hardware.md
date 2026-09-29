# Hardware

## 1. Overview

现在讨论 AI Infra，已经很难只看一张 GPU。一个模型真正跑起来，需要计算芯片执行矩阵运算，需要 HBM 持续提供权重和中间数据，需要多颗加速器之间交换 activation 和梯度，还需要 CPU、PCIe、NIC、存储以及更上层的集群网络共同完成数据搬运。再往外扩展，供电、散热和封装又决定了这些硬件最终能够以什么规模组合在一起。

从系统结构上看，可以把 AI 硬件粗略分成几个层次。最里面是 **GPU、NPU 或其他 accelerator**，负责主要计算；紧挨着它的是 **HBM 和片上 memory hierarchy**，决定数据能否及时送到计算单元。多颗 accelerator 通过 **scale-up interconnect** 组成更大的计算域，再进一步形成 supernode 或 rack-scale system；多个计算域之间则通过 **scale-out network** 连接成更大的训练或推理集群。CPU、PCIe、NIC 和 storage 位于这些层次之间，负责控制、数据交换和 I/O。

这些硬件并不是彼此独立地决定性能。一个 workload 可能拥有很高的理论 FLOPS，却受限于 HBM bandwidth。也可能单卡计算很快，却在 tensor parallel 或 MoE 中等待通信；继续增加 accelerator 数量后，瓶颈还可能从计算迁移到网络、内存容量，甚至供电和散热。很多 AI 系统问题，本质上都是同一个问题：**当前 critical path 上，计算和数据搬运分别花了多少时间？**

因此，这篇文章沿着 **chip → memory → server → interconnect → supernode → cluster** 的路径逐层展开。对每一层，我们都会关注三个问题：它在系统中负责什么、哪些硬件指标真正影响 AI workload，以及当模型规模、batch size、context length 或并行方式发生变化时，瓶颈会如何迁移。


## 2. Compute

对今天的大模型来说，计算的主体是矩阵乘法。Transformer 中的 QKV projection、attention projection 和 MLP，以及 MoE 中的 expert computation，都包含大量 GEMM。因此，理解一颗 AI accelerator 的计算能力，可以先抓住两个问题：**矩阵算得有多快，以及数据能否持续喂给计算单元。**

###2.1 AI Accelerators

#### 2.1.1 GPU

GPU 最初面向图形渲染设计，其中大量像素和顶点可以同时处理，这推动了 GPU 朝着高并行吞吐的方向发展。今天的 GPU 已经是一种通用并行计算平台：一颗芯片包含大量计算单元，可以让成千上万个线程同时执行。神经网络恰好具有很强的数据并行性，矩阵乘法中的大量乘加操作可以同时展开，因此 GPU 很早就成为深度学习训练和推理的主要计算平台。

现代 GPU 内部通常同时存在两类重要的计算路径。CUDA Core 一类的通用计算单元负责标量、向量和各种普通算术操作，而 Tensor Core 这样的 matrix engine 专门执行小块矩阵乘加。Transformer 中的 QKV projection、MLP 和 MoE expert 等主要计算最终都会落到大量 GEMM 上，因此它们可以大量使用 Tensor Core；normalization、element-wise operation、routing、sampling 等计算则继续依赖更通用的执行单元。GPU 因此可以在同一颗芯片上覆盖模型中的多种计算模式。

GPU 的另一个重要特征是较强的 programmability。硬件通过大量线程、warp 调度和 memory hierarchy 隐藏计算与数据访问延迟，软件则可以通过 CUDA 等编程模型定义新的 kernel。模型结构发生变化时，很多新算子可以通过修改软件实现，而不需要重新设计芯片。这种灵活性对 AI 尤其重要，因为模型架构、数据类型和推理算法仍在快速变化。

这种通用性也意味着 GPU 需要为调度、通用执行、缓存和复杂控制逻辑投入相当多的芯片面积与功耗。对于计算模式更加固定的 neural network workload，如果进一步限制支持的数据类型、算子和执行方式，就可以把更多硬件资源集中到 matrix engine、片上 memory 和数据搬运上，从而追求更高的 compute density 和 energy efficiency。这也是 TPU、NPU 以及其他 specialized AI accelerators 试图探索的方向。


#### 2.1.2 Specialized AI Accelerators

##### TPU

沿着更强 specialization 的方向，Google 的 TPU 是一个很典型的例子。TPU 是专门面向 machine learning workload 设计的 ASIC，其核心任务就是高效执行神经网络中的矩阵运算。相比需要覆盖广泛 workload 的 GPU，TPU 可以把更多硬件资源直接围绕 matrix computation 和相应的数据流组织。

一颗 TPU 的核心计算单元包含 Matrix Multiply Unit（MXU）、vector unit 和 scalar unit。MXU 承担主要的矩阵计算，内部由大量 multiply-accumulate 单元组成 systolic array；vector unit 处理 activation、softmax 等计算，scalar unit 则负责控制流和地址计算。这个划分和现代 GPU 中“matrix engine + general-purpose execution unit”的思路有些相似，但 TPU 把整个架构进一步围绕机器学习 workload 收紧。

Systolic array 最值得理解的地方是它的数据流。矩阵数据进入阵列后，会在相邻的 multiply-accumulate 单元之间逐级传递，同时不断累积 partial sum。这样，同一份数据进入阵列后可以连续参与多次计算，不需要每次乘加都重新访问外部 memory。Google 对 TPU 的架构描述中也强调，参数从 HBM 送入 MXU 后，矩阵乘法过程中的中间结果沿阵列传递，从而减少计算过程中的 memory access。

TPU 展示了 specialized accelerator 的一个基本思路：**当主要 workload 已经比较明确时，可以围绕最重要的计算和数据移动路径重新组织硬件。** 这样可以把更多芯片面积和功耗投入 matrix engine、片上数据传输和低精度计算，从而提高目标 workload 上的 compute density 和 energy efficiency。代价则是硬件适用范围会更集中，因此 compiler 和软件栈需要把模型有效映射到这些固定的计算与数据流上。

##### Ascend NPU

Ascend 是另一类典型的 specialized AI accelerator。它的基本计算单元称为 **AI Core**，内部将不同类型的计算进一步拆分为 Cube、Vector 和 Scalar。Cube Unit 主要负责矩阵乘加，Vector Unit 处理向量计算，Scalar Unit 则负责控制流和地址计算。这样的划分直接对应了神经网络中的几类主要 workload：大规模矩阵计算交给 Cube，element-wise、activation 等操作交给 Vector，控制和调度则由 Scalar 完成。

Ascend 中很值得关注的一点是 **compute 和 data movement 被一起设计**。AI Core 内部除了计算单元，还包含 L1 Buffer、L0A/L0B/L0C、Unified Buffer 等多级 local memory，以及专门负责不同存储层级之间数据搬运的 MTE。以矩阵乘法为例，输入矩阵会进入 L0A 和 L0B，Cube Unit 完成矩阵运算后，中间结果保存在 L0C。这样，计算过程中频繁使用的数据可以停留在距离计算单元更近的位置，减少对 global memory 的反复访问。

不同代 Ascend NPU 还会调整这些计算单元之间的组织方式。例如在部分较新的架构中，Cube 和 Vector 被分别部署到独立的 AIC 和 AIV core 上，各自拥有 Scalar Unit，可以独立执行不同类型的程序。这说明 specialized accelerator 的设计并不只有“增加更多矩阵单元”这一条路径，还可以通过重新划分计算资源、local memory 和 data path，让不同类型的 AI 算子获得更适合自己的执行方式。

如果说 TPU 展示了如何围绕 matrix computation 和 systolic dataflow 构建高度专用的执行路径，那么 Ascend NPU 展示了另一种思路：**将 matrix、vector、scalar 和 data movement 显式拆分，再通过软件把一个 AI workload 映射到这些不同的硬件资源上。** 两者的具体实现不同，但目标是一致的——让神经网络中最常见的计算和数据移动获得更高的执行效率。

##### Cerebras Wafer-Scale Engine

Cerebras 走了一条更激进的路线：直接扩大一颗 accelerator 的物理边界。传统芯片会把一片 wafer 切成许多独立 die，再通过 package、board 和高速 interconnect 把多颗芯片连接起来；Cerebras 的 Wafer-Scale Engine（WSE）则保留整片 wafer，并通过跨越 reticle boundary 的片上互联把不同区域连接成一个统一的计算 fabric。这样，原本需要跨 package 或跨设备完成的一部分数据交换，可以留在 silicon 上完成。

Wafer-scale integration 的价值来自 **compute、memory 和 communication 被放到了同一个巨大计算域中**。以 WSE-3 为例，一片 wafer 集成约 900,000 个 AI cores 和 44 GB distributed SRAM，片上 memory bandwidth 达到 21 PB/s。每个计算区域都有靠近 compute core 的 local SRAM，数据通过二维片上 fabric 在不同 core 之间显式传递。对于大量需要频繁访问权重和中间结果的 AI workload，这种设计能够缩短数据移动距离，并提供远高于片外 memory 的局部带宽。

把整片 wafer 做成一个处理器也带来了传统芯片很少遇到的问题。单个小 die 即使出现制造缺陷，也可以在封装前直接丢弃；wafer-scale processor 无法要求整片 wafer 上所有区域都完美，因此必须在架构层面处理 defect tolerance。Cerebras 在 compute fabric 和通信路径中加入冗余，通过绕过失效区域仍然形成完整的逻辑计算阵列。同时，整片 wafer 的 power delivery、cooling 和 packaging 也需要围绕这种尺寸重新设计。

因此，WSE 展示的是另一种 accelerator specialization：**通过扩大单个高速计算域，把更多 compute、SRAM 和 communication 留在 silicon 内部，从体系结构层面减少跨芯片的数据移动。** 它也揭示了 AI hardware 的一个重要趋势——随着模型越来越依赖大规模并行，性能不仅取决于单个计算单元有多快，也越来越取决于能够以多低的代价把多少计算和 memory 连接在一起。


### 2.2 Matrix Engines and GEMM

以一个线性层为例，

\[
Y = XW,
\]

其中 \(X\in\mathbb{R}^{M\times K}\)，\(W\in\mathbb{R}^{K\times N}\)。对 LLM decode 来说，\(M\) 往往与当前参与计算的 token 数有关；batch 较小时，\(M\) 可能很小，而 \(K\) 和 \(N\) 通常很大。现代 accelerator 会把这样的矩阵拆成大量小 tile，再交给 Tensor Core 或类似的 matrix engine 并行执行。

这也是为什么 AI 芯片拥有很高的理论 FLOPS，却不意味着任意 GEMM 都能达到这个数字。矩阵太窄时，可以同时执行的工作有限，计算单元可能无法被完全填满；矩阵规模增大后，更多计算可以并行执行，同时同一份权重也能够被更多 token 复用，硬件利用率通常会提高。这也是 batch size 会显著影响 LLM 推理效率的一个重要原因。

矩阵计算之外，GPU 仍然需要执行 normalization、sampling、element-wise operation、routing 等大量非 GEMM 算子。这些操作可能由普通 CUDA Core、vector unit 或其他执行单元完成。因此，一个模型即使大部分 FLOPs 来自 GEMM，端到端 latency 也不一定完全由 matrix engine 决定。

### 2.3 Peak FLOPS vs. Effective Performance

硬件规格表最常出现的指标是 TFLOPS 或 PFLOPS，但这个数字描述的通常是特定数据类型和特定执行条件下的理论峰值。例如，同一颗 accelerator 在 FP32、BF16、FP8 或更低精度下可能具有完全不同的峰值吞吐；如果规格使用了 structured sparsity，数字还可能进一步提高。因此，在比较两颗芯片之前，至少需要确认 **precision、dense/sparse、accumulation type 和统计口径是否一致**。

更重要的是，实际性能取决于有多少理论算力真正被 workload 使用。可以简单写成

\[
P_{\mathrm{effective}}
=
U_{\mathrm{compute}}\cdot P_{\mathrm{peak}},
\]

其中 \(U_{\mathrm{compute}}\) 表示有效计算利用率。这个量会受到矩阵形状、batch size、kernel implementation、数据类型以及 memory traffic 等因素共同影响。一个理论峰值更高的 accelerator，如果 workload 无法提供足够大的计算规模，或者大部分时间都在等待数据和通信，最终未必更快。

因此，看 AI accelerator 时，峰值 FLOPS 只能回答“这颗芯片最多能算多快”。真正需要进一步问的是：**模型能否把这些计算单元填满，数据能否及时送到计算单元，以及计算完成后是否还要等待其他设备。** 后两个问题分别会把我们带到下一章的 memory hierarchy，以及后面的 interconnect 和 networking。







