# Parallelism

Deployment decisions are often framed as configuration choices such as `DP2TP2` versus `TP4`, but those labels are only shorthand for deeper systems trade-offs. Some forms of parallelism primarily solve a throughput problem, some solve a memory-fit problem, and some introduce communication overhead large enough to erase their theoretical benefit. Parallelism becomes much easier to reason about once those underlying problems are separated.

## Data Parallelism (DP)

Data Parallelism (DP) is the most conventional and intuitive form of distributed execution in machine learning systems. Its central idea is simple: the same model is instantiated on multiple devices, while the input workload is partitioned across those instances. Each worker executes an identical computation graph on a different subset of data. In this sense, it changes only the mapping of computation to hardware.

When one model replica is already memory-feasible but cannot absorb enough request load, the natural way to increase system capacity is to add more replicas.

Because the parameters are duplicated rather than partitioned, DP does not reduce the memory footprint on any individual device. This property defines its practical scope: DP is effective when the model already fits within the memory budget of a single GPU, but it is insufficient when the model itself exceeds per-device capacity. Its primary value therefore lies in scaling throughput and serving concurrency through replication.

In training, DP is commonly used to increase effective batch size and improve hardware utilization. Each worker processes a shard of the mini-batch, computes local gradients, and then participates in synchronization so that all model copies remain numerically consistent after the optimization step. In inference, the same replication principle still applies, but without gradient exchange. Instead, each worker handles an independent subset of requests or token batches. Under inference workloads, DP should therefore be understood mainly as a mechanism for traffic distribution rather than model partitioning.

In hybrid deployments, DP often coexists with Tensor Parallelism (TP) and Expert Parallelism (EP). Under such configurations, a DP replica should not be interpreted too literally as a single GPU. Rather, one DP unit may itself be a multi-GPU group whose internal execution is organized by TP or EP. From this perspective, DP defines the outer structure of workload replication, while TP and EP determine how computation is carried out within each replica.

### The Collective Communication Related to DP

The communication behavior associated with DP depends fundamentally on whether the system is performing training or inference.

During distributed training, each DP worker computes gradients from its local mini-batch shard. Since every worker maintains the same parameter set, these gradients must be synchronized after each step to preserve consistency across replicas. The standard collective primitive for this purpose is all-reduce over gradients. In some implementations, the same effect is realized through reduce-scatter followed by all-gather, but the objective remains unchanged: to aggregate gradient contributions from all DP ranks and produce identical parameter updates on every worker.

By contrast, inference does not involve backward propagation or optimizer steps. Each DP rank serves an independent portion of the request stream, so DP itself introduces no mandatory per-layer collective communication. Under inference, it functions primarily as a scheduling and traffic-partitioning mechanism rather than a synchronization-intensive execution strategy.

That said, a DP-based inference system is not entirely communication-free. The runtime still requires system-level coordination for request dispatch, load balancing, admission control, and result collection. In practice, such coordination often relies on communication patterns such as all-gather to exchange metadata, state, or scheduling information across ranks. These operations belong to the serving framework rather than the model's numerical execution.

In hybrid MoE deployments, DP usually contributes limited communication overhead during inference, while the parallel mechanisms inside each DP unit are often more communication-intensive. Tensor Parallelism typically relies on collectives such as all-reduce or all-gather within a TP group, whereas Expert Parallelism introduces token routing across devices that host different experts. Accordingly, communication analysis in a hybrid system should separate the outer DP dimension from the inner model-parallel dimensions: DP governs workload distribution, while the dominant communication cost usually comes from TP or EP.

## Tensor Parallelism (TP)

TP partitions the computation of a single layer across multiple devices by sharding the layer's weight tensors, rather than replicating the full layer on every rank. In large transformer models, this is typically applied to the projection matrices in attention and MLP blocks. The core objective is to reduce the per-device memory footprint and enable execution of layers whose parameters or activations would otherwise exceed the capacity of a single accelerator.

TP becomes interesting when one accelerator cannot comfortably host the model or when reducing per-device weight footprint creates more room for activations and KV cache inside a single replica.

Consider a linear transformation

$$
Y = X W,
$$

where $X \in \mathbb{R}^{B \times H}$, $W \in \mathbb{R}^{H \times O}$, and $Y \in \mathbb{R}^{B \times O}$. Under TP, $W$ is divided across a process group of $p$ ranks. Two common decompositions are column-wise sharding and row-wise sharding.

In column-wise TP, the output dimension $O$ is partitioned, so each rank stores

$$
W_i \in \mathbb{R}^{H \times O/p}
$$

Each rank computes a partial output

$$
Y_i = X W_i \in \mathbb{R}^{B \times O/p}
$$

which corresponds to a slice of the final output tensor. This pattern is natural for the first projection in transformer feed-forward blocks, because the partial outputs can often remain distributed until a later synchronization point.

In row-wise TP, the input dimension $H$ is partitioned, so each rank stores

$$
W_i \in \mathbb{R}^{H/p \times O}
$$

The input $X$ must then be partitioned consistently, and each rank computes a partial contribution to the full output:

$$
Y_i = X_i W_i \in \mathbb{R}^{B \times O}
$$

Since these are additive contributions to the same output tensor, the ranks must sum them to recover $Y$. This makes row-wise TP naturally coupled with a reduction step.

For transformer layers, TP is usually designed so that adjacent projections alternate between communication-light and communication-heavy layouts. A standard example is the two-layer MLP block. The first linear layer expands the hidden state and is often implemented with column-wise TP, while the second projects back to the hidden dimension using row-wise TP. This pairing avoids unnecessary materialization of full intermediate activations on every rank and keeps most of the computation local until the boundary where the distributed partial results must be merged.

The main advantage of TP is that it scales model width without requiring full parameter replication. However, its efficiency depends strongly on the balance between local matrix multiplication and inter-rank communication. As TP degree increases, per-rank GEMM sizes become smaller and communication becomes a larger fraction of the step time. For this reason, TP is usually effective only within a tightly coupled device group, such as GPUs connected by high-bandwidth intra-node interconnects.

### The Collective Communication Related to TP

The communication pattern in TP is determined by how tensors are sharded and whether the local computation produces disjoint output slices or partial sums. In practice, TP relies primarily on three collective patterns: all-gather, reduce-scatter, and all-reduce. Their role is not incidental; they define the synchronization boundaries of tensor-parallel execution.

In a column-wise partitioned linear layer, each rank produces a distinct shard of the output features. If the next operator can consume the activation in sharded form, no immediate communication is required. This is one of the main reasons column-wise sharding is attractive. However, when a subsequent operation expects the full activation tensor, the ranks must perform an all-gather to assemble

$$
Y = [Y_0, Y_1, \ldots, Y_{p-1}]
$$

Thus, all-gather is associated with reconstructing a feature dimension that was split across ranks.

In a row-wise partitioned linear layer, each rank computes only a partial contribution to the same output tensor. The global result is

$$
Y = \sum_{i=0}^{p-1} Y_i.
$$

This requires a summation across the TP group. Conceptually, this is an all-reduce if every rank needs the full output. In optimized implementations, the communication may instead be fused with downstream sharding requirements and expressed as a reduce-scatter, which both sums the partial results and redistributes the output into shards for the next stage.

At the systems level, the cost of TP communication is shaped by message size, collective algorithm, process-group topology, and overlap with computation. Since TP collectives are invoked at fine granularity inside individual layers, their latency sensitivity is high. This is why TP is typically constrained to a small number of nearby devices: the communication volume may be moderate, but the synchronization frequency is high. Once the TP group spans slower links, collective latency can dominate the runtime and erase the gains from parameter sharding. A practical interpretation is that TP converts a single large matrix multiplication into several smaller local GEMMs plus structured collectives. Its performance therefore depends on whether the reduction in memory pressure and per-rank compute fits outweighs the repeated cost of all-gather, reduce-scatter, and all-reduce across the tensor-parallel group.

### Inference Interpretation

For serving workloads, DP and TP usually play different roles.

- DP primarily scales throughput by creating more replicas that can serve independent requests.
- TP primarily improves model fit and per-replica memory budget by sharding parameters, but it introduces intra-layer collectives.
- Increasing DP duplicates model weights and may fragment prefix-cache locality across replicas.
- Increasing TP reduces per-device weight memory and can enlarge the effective KV budget of one replica, but too much TP may turn communication into the bottleneck.

This is why inference deployment decisions are often phrased as a trade-off between replica-level concurrency and per-replica memory headroom.
