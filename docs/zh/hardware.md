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

#### 2.2.1 Matrix Engines

现代 AI accelerator 通常同时包含 scalar、vector 和 matrix 三类计算单元。Scalar unit 一次处理少量标量操作，适合地址计算、控制流等任务；vector unit 可以对一组数据同时执行相同运算，常用于 activation、normalization 和 element-wise operation。Transformer 中占据大部分计算量的 linear layer 和 MoE expert 则主要由 matrix engine 执行。

Matrix engine 针对的核心操作是矩阵乘加，例如

\[
D = AB + C.
\]

矩阵乘法本身可以展开成大量 multiply-accumulate（MAC）。如果完全依靠普通 scalar 或 vector instruction 执行，需要发出大量独立指令并不断读取和写回中间结果。Matrix engine 会把一小块矩阵作为一个整体交给专门的数据通路，在硬件内部同时完成大量 MAC。NVIDIA 的 Tensor Core、TPU 的 MXU 和 Ascend 的 Cube Unit 虽然实现方式不同，都体现了这种设计思路。

以 Tensor Core 为例，软件通常不会要求它直接计算完整的 \(M\times N\) 矩阵，而是提交较小的 matrix multiply-accumulate operation。大量这样的操作再组合成完整 GEMM。这样做的价值在于，数据进入 matrix engine 后，可以在内部计算路径中连续参与多次乘加，控制和指令开销也能够被大量 MAC 分摊。对于以 GEMM 为主的 neural network workload，这种专用数据通路能够提供远高于普通通用计算单元的矩阵吞吐。

因此，现代 AI accelerator 的计算能力可以理解为两部分的配合：**matrix engine 承担大规模、规则的矩阵计算，scalar 和 vector unit 处理剩余的通用算子。** 拥有大量 matrix engine 仍然只提供了计算能力的上限；这些单元能否真正被填满，还取决于 GEMM 的矩阵形状以及任务如何被切分到硬件上，这也是下一节要讨论的问题。

#### 2.2.2 GEMM Shape and Tiling

考虑一个最常见的 linear layer：

\[
Y=XW,
\qquad
X\in\mathbb{R}^{M\times K},
\quad
W\in\mathbb{R}^{K\times N}.
\]

这次 GEMM 需要大约 \(2MNK\) FLOPs，其中 \(K\) 和 \(N\) 通常由模型维度决定，而 \(M\) 往往对应这次 forward 中一起处理的 token 数。对于 LLM prefill，多个 prompt token 可以同时进入 linear layer，因此 \(M\) 通常较大；decode 时每个请求每一步只有一个新 token，\(M\) 基本随 batch size 增长。这也是为什么同一个模型在 prefill 和 decode 阶段，会表现出非常不同的计算效率。

GPU 不会把整个 \(M\times N\) 输出矩阵一次交给一个计算单元，而是将它划分成许多更小的 **tiles**。例如一个 thread block 负责一个 \(M_{\mathrm{tile}}\times N_{\mathrm{tile}}\) 的输出区域，再沿着 \(K\) 维分块读取 \(X\) 和 \(W\)，不断执行 matrix multiply-accumulate。这样，大 GEMM 会产生大量彼此独立的 tiles，可以同时分配到不同的 SM 上执行。矩阵较小时，能够产生的 tiles 也更少，即使每个 Tensor Core 本身很快，整块 GPU 仍然可能没有足够的并行工作可以执行。

Tiling 还会带来两个容易被忽略的量化效应。如果 \(M\) 或 \(N\) 不能被 tile size 整除，边缘 tile 中只有部分位置是真正有效的，但硬件仍需要执行这个 tile，这称为 **tile quantization**。即使 tile 本身都很满，总 tile 数也可能无法刚好填满 GPU：假设一次可以并行执行 \(S\) 个 tiles，而 GEMM 最终产生 \(S+1\) 个，那么最后那个 tile 还需要额外启动一轮计算，其余大部分计算单元却处于空闲状态，这就是 **wave quantization**。因此，GEMM latency 并不一定随着 FLOPs 平滑增长，有时矩阵尺寸只增加一点，就可能因为多出一个 tile 或一个 wave 而出现明显的 latency 跳变。

所以，矩阵“更大”本身并不是重点，关键是它是否产生了足够多、足够规整的计算工作。Prefill 的大 \(M\) 通常能够产生更多 tiles，把更多计算单元同时利用起来；小 batch decode 的 \(M\) 很窄，则更容易受到 tile parallelism 和 wave quantization 的限制。这也解释了 AI Infra 中一个很常见的现象：**增加 batch size 或一次处理更多 token 后，计算量虽然增加了，执行时间却可能增长得远慢于计算量。** 接下来还需要考虑另一个原因——更大的 GEMM 同时能够带来更好的数据复用。

#### 2.2.3 Data Reuse and Arithmetic Intensity

GEMM 的效率还取决于另一件事：**搬进来的数据能够参与多少次计算。** 仍然考虑

\[
Y=XW,
\qquad
X\in\mathbb{R}^{M\times K},
\quad
W\in\mathbb{R}^{K\times N}.
\]

完成这次矩阵乘法需要大约 \(2MNK\) FLOPs，但真正的执行时间还取决于 \(X\)、\(W\) 和中间结果需要在不同 memory level 之间搬动多少数据。通常用 **arithmetic intensity** 描述这种关系：

\[
I=
\frac{\text{FLOPs}}
{\text{Bytes Moved}}.
\]

\(I\) 越高，说明每搬运一个 byte 的数据能够完成更多计算。

如果暂时只考虑 HBM traffic，并假设 \(X\)、\(W\) 各读取一次、\(Y\) 写回一次，每个元素占 \(s\) bytes，那么一个理想化的 GEMM arithmetic intensity 可以写成

\[
I
\approx
\frac{2MNK}
{s(MK+KN+MN)}.
\]

这个式子可以直观看出 \(M\) 为什么重要。Decode 中 \(M\) 很小时，庞大的权重矩阵 \(W\) 被读进来后只服务少量 token，每读取一次权重只能完成有限计算；随着 batch size 或同时处理的 token 数增加，同一份 \(W\) 可以被更多行的 \(X\) 复用，计算量增长得比权重读取量更快，因此 arithmetic intensity 随之提高。

实际 GPU 会进一步通过 tiling 把这种 reuse 留在更靠近计算单元的位置。一个 tile 的 \(X\) 或 \(W\) 被加载到 register、shared memory 或其他片上存储后，可以参与多次 matrix multiply-accumulate，再去处理下一块数据。这里需要区分 **理论上存在的数据复用** 和 **硬件真正实现的数据复用**：矩阵形状、tile size、cache behavior 和 kernel implementation 都会决定有多少数据最终仍然需要从更远的 memory hierarchy 重新读取。

因此，扩大 GEMM 的工作规模通常同时产生两种收益：更多 tiles 提供更高的并行度，而更高的数据复用又提高 arithmetic intensity。当前者不足时，计算单元没有被填满；后者不足时，计算单元可能在等待数据。只有两者都足够高，GEMM 才有机会接近 accelerator 的峰值计算能力。这也自然引出了下一节的问题：**规格表上的 Peak FLOPS，究竟在什么条件下才能真正转化成有效性能？**

### 2.3 Peak FLOPS vs. Effective Performance

#### 2.3.1 Precision and Peak FLOPS

硬件规格中的 FLOPS（floating-point operations per second）描述的是单位时间能够完成多少浮点运算。对于矩阵乘法中最常见的 fused multiply-add（FMA），一次乘法和一次加法通常计作两次 floating-point operations。因此，如果一颗 accelerator 每秒能够执行 \(N_{\mathrm{FMA}}\) 次这样的操作，其理论计算吞吐可以写成

\[
P_{\mathrm{peak}}
=
2N_{\mathrm{FMA}}.
\]

同一颗 accelerator 往往会给出多组完全不同的 peak throughput，例如 FP32、TF32、BF16、FP16 和 FP8。原因很直接：precision 越低，一个数需要的 bit 越少，同样的芯片面积和数据通路通常能够并行处理更多元素。以 matrix engine 为例，一条面向低精度数据设计的计算路径可以在一个周期内完成更多 multiply-accumulate，因此通常有

\[
P_{\mathrm{FP8}}
>
P_{\mathrm{BF16}}
>
P_{\mathrm{FP32}}.
\]

更低的 precision 同时也减少了数据量。相同数量的参数和 activation 使用 BF16 时只需要 FP32 一半的存储空间，FP8 又进一步减半，因此 HBM、cache 和片上数据通路可以在相同时间内搬运更多元素。低精度计算带来的收益于是同时出现在 **compute throughput** 和 **memory traffic** 两侧，这也是现代 AI accelerator 越来越强调 BF16、FP8 甚至更低精度计算的重要原因。

不过，“FP8 compute”并不意味着整条计算链路中的所有数据都以 FP8 完成运算。Matrix multiplication 经常使用较低精度的 input 执行乘法，再使用更高精度保存和累加 partial sum，例如低精度 multiply 配合 FP16 或 FP32 accumulation。这样可以在提高吞吐的同时控制长序列乘加带来的数值误差。因此，阅读一项 peak FLOPS 时，需要同时确认 **input precision、accumulation precision 和实际使用的计算单元**。只有这些口径一致，两颗 accelerator 的峰值算力才具有直接可比性。

#### 2.3.2 Dense, Sparse, and Tensor Core Throughput

即使 precision 相同，一颗 accelerator 的规格表中也可能出现多组完全不同的 peak throughput。以 NVIDIA GPU 为例，普通 FP32 CUDA Core、Tensor Core dense GEMM 和 Tensor Core sparse GEMM 使用的是不同的执行路径，因此对应的峰值算力也不同。Tensor Core 针对规则的 matrix multiply-accumulate 提供更高吞吐，而普通计算单元还需要支持更广泛的 arithmetic 和 control operation。因此，看到一个很高的 TFLOPS 数字时，首先需要确认它描述的是哪条计算路径。

Sparse throughput 又引入了另一层区别。以 NVIDIA Ampere 及后续架构支持的 structured sparsity 为例，对于一些数据类型，硬件可以利用 **2:4 sparsity**：每连续四个 weight 中至少有两个为零，非零值和对应 metadata 被压缩后送入 Sparse Tensor Core。硬件只对保留下来的非零元素执行实际乘加，因此在满足这种结构约束时，同一套 matrix hardware 可以完成大约两倍的等效 dense computation。 这也是为什么规格表中的 sparse Tensor Core throughput 经常明显高于对应的 dense throughput。

这里的关键在于，**sparse FLOPS 描述的是满足特定 sparsity pattern 时的有效计算能力，并不代表任意稀疏模型都能获得同样的加速。** 模型权重首先需要满足硬件支持的结构，例如 2:4 sparsity，kernel 和软件栈也必须真正使用对应的 sparse execution path；普通 unstructured sparsity 并不会自动映射到 Sparse Tensor Core。以 2:4 为例，NVIDIA 的实现通过只保存和计算非零值来减少权重 footprint 和相关 bandwidth，并在支持的 Tensor Core 路径上提高理论吞吐。

因此，比较两颗 accelerator 的“算力”时，至少需要同时对齐 **precision、accumulation precision、dense/sparse 口径，以及实际使用的 compute path**。例如一个 BF16 dense Transformer，应该首先比较 BF16 dense matrix throughput；直接拿另一颗芯片的 FP8 sparse peak 与它相比，得到的数字几乎没有实际意义。规格表给出的始终是某种特定条件下的硬件上限，下一步真正需要回答的是：实际 workload 最终能够利用其中多少。

#### 2.3.3 From Peak FLOPS to Achieved Throughput

Peak FLOPS 给出了 accelerator 在特定 precision 和计算路径下的理论上限，实际 workload 能达到的计算吞吐则可以写成

\[
P_{\mathrm{achieved}}
=
\frac{\text{executed FLOPs}}
{\text{execution time}},
\]

进一步定义相对于峰值算力的利用率

\[
U_{\mathrm{compute}}
=
\frac{P_{\mathrm{achieved}}}{P_{\mathrm{peak}}}.
\]

这个比例通常远小于 1，而且会随着 workload 改变。同一颗 GPU、同一个模型，仅仅改变 batch size、sequence length 或 GEMM shape，就可能得到完全不同的 \(U_{\mathrm{compute}}\)。

前面讨论的几个机制都会影响这个比例。矩阵较窄时，tile 数量不足，matrix engine 无法全部填满；tile 或 wave 没有很好对齐时，一部分计算资源会处于空闲；arithmetic intensity 较低时，计算单元又可能需要等待数据从 memory hierarchy 中搬进来。可以粗略地把实际计算性能理解为同时受到两个上限约束：

\[
P_{\mathrm{achieved}}
\lesssim
\min
\left(
P_{\mathrm{peak}},
\;
I\cdot BW_{\mathrm{memory}}
\right),
\]

其中 \(I\) 是 arithmetic intensity，\(BW_{\mathrm{memory}}\) 是可用 memory bandwidth。前者对应 compute ceiling，后者对应数据供给能够支撑的计算速度。这也是 Roofline Model 最核心的直觉：提高 Peak FLOPS 只有在 workload 能够提供足够高 arithmetic intensity 时，才会真正转化成性能提升。

端到端模型性能还会进一步低于单个 GEMM 所能达到的吞吐。Transformer 中除了 matrix multiplication，还有 normalization、softmax、sampling、routing、memory copy 等操作；多卡运行时还会加入 collective communication 和 synchronization。即使 GEMM 已经接近 Tensor Core 的峰值，其他部分仍然可能处在 critical path 上。因此，分析硬件性能时，需要从单个 kernel 的 achieved throughput 一直看到完整模型的 execution timeline。

这里也要区分不同意义上的 utilization。监控工具显示的“GPU utilization”通常表示一段时间内 GPU 是否有 kernel 在执行，并不能直接说明 Tensor Core 已经达到多少 Peak FLOPS。一块 GPU 可以长期处于 busy 状态，同时只达到很低的理论计算吞吐。真正评价 accelerator 是否被有效使用，需要结合 **achieved FLOPS、matrix shape、memory bandwidth、kernel breakdown 和端到端 latency** 一起看。规格表告诉我们硬件的上限，而 workload 决定了最终能够接近这个上限多少。