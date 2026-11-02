# Collective Communication

Collective communication lets a group of GPUs distribute, gather, or reduce data together. In tensor parallelism, for example, each GPU can compute a partial result and use All-Reduce to sum the contributions so that every participant receives the complete result.

## Basic Operations

Each participant in a communication group has an index called a **rank**. The examples below use four GPUs, mapped to ranks 0–3. All reductions use elementwise summation. Operation definitions are documented in the [NCCL guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html).

### Broadcast

Broadcast copies data from a designated rank, the root, to every rank in the group. If rank 0 broadcasts `[1, 2]`, all four ranks receive `[1, 2]`.

### All-Gather

All-Gather collects data from each rank, concatenates it in rank order, and gives the complete result to every rank.

| Rank | Input | Output |
| --- | --- | --- |
| 0 | `[1, 2]` | `[1, 2, 3, 4, 5, 6, 7, 8]` |
| 1 | `[3, 4]` | `[1, 2, 3, 4, 5, 6, 7, 8]` |
| 2 | `[5, 6]` | `[1, 2, 3, 4, 5, 6, 7, 8]` |
| 3 | `[7, 8]` | `[1, 2, 3, 4, 5, 6, 7, 8]` |

This is useful when ranks hold different shards but the next computation needs the full tensor. The data are concatenated, without summation.

### All-Reduce and Reduce-Scatter

Both operations sum corresponding elements across ranks. They differ in how the result is distributed.

| Rank | Input | All-Reduce output | Reduce-Scatter output |
| --- | --- | --- | --- |
| 0 | `[1, 2, 3, 4]` | `[1111, 2222, 3333, 4444]` | `[1111]` |
| 1 | `[10, 20, 30, 40]` | `[1111, 2222, 3333, 4444]` | `[2222]` |
| 2 | `[100, 200, 300, 400]` | `[1111, 2222, 3333, 4444]` | `[3333]` |
| 3 | `[1000, 2000, 3000, 4000]` | `[1111, 2222, 3333, 4444]` | `[4444]` |

**All-Reduce** gives every rank the complete sum. **Reduce-Scatter** partitions the sum, giving each rank its assigned segment. If only a designated root needs the complete result, use **Reduce**.

Reduce-Scatter followed by All-Gather produces the same result as All-Reduce. If the next computation can consume shards directly, it can retain the Reduce-Scatter output and avoid immediately gathering the full tensor.

### All-to-All

All-to-All redistributes data by destination rank. Each symbol below represents a data block, with its subscript identifying the destination:

| Rank | Input | Output |
| --- | --- | --- |
| 0 | `[a₀, a₁, a₂, a₃]` | `[a₀, b₀, c₀, d₀]` |
| 1 | `[b₀, b₁, b₂, b₃]` | `[a₁, b₁, c₁, d₁]` |
| 2 | `[c₀, c₁, c₂, c₃]` | `[a₂, b₂, c₂, d₂]` |
| 3 | `[d₀, d₁, d₂, d₃]` | `[a₃, b₃, c₃, d₃]` |

Each rank sends different data to other ranks and receives the data addressed to it. Unlike All-Gather, the ranks end up with different contents.

## Communication in Inference

### Tensor Parallelism

Consider a linear layer $Y=XW$. After partitioning its input dimension across four GPUs, each rank computes a partial result:

$$
Y_r = X_rW_r,\qquad Y = \sum_{r=0}^{3}Y_r
$$

Use All-Reduce if every rank needs the complete $Y$ for subsequent computation. Use Reduce-Scatter if the subsequent computation accepts sharded results.

When partitioning the output dimension, ranks produce different output features that do not need summation. All-Gather is needed only when subsequent computation requires the complete output. The next operation's required data layout determines the communication. See [Parallelism](parallelism.md) for partitioning details.

### Expert Parallelism

MoE experts reside on different GPUs. After routing selects experts for a token, its representation must be sent to the ranks hosting those experts. Expert outputs then return to the originating rank and are combined using the routing weights.

This is an All-to-All-style exchange. Experts can receive different token counts, so implementations must handle variable data sizes. Ranks hosting popular experts may receive more data and perform more computation, leaving other ranks waiting for their results.

Pipeline stage boundaries usually use point-to-point sends and receives: one stage passes activations to the next, without requiring the entire group to exchange data collectively.

## Communication Cost

### Message Size and Communication Rounds

Communication includes startup overhead and data transfer time. Let $\alpha$ denote the fixed latency per round, $B$ the effective link bandwidth, and $S$ the bytes transferred in that round. A rough estimate is:

$$
T_{\mathrm{round}} \approx \alpha + \frac{S}{B}
$$

Fixed latency accounts for a larger share of small-message communication. Low-concurrency decode processes few tokens per step but still repeats communication within layers, so even small transfers can add noticeable latency. For large messages, transfer time matters more, making link bandwidth and actual traffic more influential. See [NVIDIA's inference communication analysis](https://developer.nvidia.com/blog/optimizing-for-low-latency-communication-in-inference-workloads-with-jax-and-xla/).

### Ring and Tree

The same All-Reduce can use different algorithms. Ring arranges ranks in a cycle and passes data in chunks through Reduce-Scatter and All-Gather phases. Each phase takes $p-1$ rounds, where $p$ is the rank count. More GPUs mean more rounds around the ring.

Tree reduces data along a tree and distributes the result, with a number of communication levels that grows logarithmically with the rank count. Algorithm selection depends on rounds, link utilization, and topology, rather than total bytes alone. NCCL selects implementations according to communication size and hardware. See [NCCL algorithm and protocol tuning](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/).

Here, $B$ is effective GPU interconnect bandwidth, not HBM bandwidth. With the same rank count and tensor size, placement within one machine or across multiple nodes changes the links traversed and the shared bandwidth, affecting communication time.

### Overlap with Computation

Communication can overlap with computation that does not depend on its result. For example, while one microbatch sends activations to the next pipeline stage, the current stage can process another microbatch.

An operator that needs an All-Reduce result must wait until the required result is ready. Launching communication asynchronously lets the caller continue, but reducing total time requires independent work and depends on contention for GPU resources. When analyzing inference, focus on communication time that computation does not cover.
