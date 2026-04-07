# KV Cache

In long-context serving, the main bottleneck is often not model weights but the memory required to keep attention state alive. As prompts grow and concurrency rises, KV cache quickly becomes the dominant systems object: it limits how many requests can stay active, determines when preemption occurs, and strongly shapes deployment choices such as DP versus TP. Many serving optimizations exist precisely because this memory budget is tight.

## KV Memory

### Memory Cost

The first question is why KV cache becomes expensive so quickly. The answer is that it scales with every token that remains in context and with every transformer layer, so even moderate sequence lengths can translate into multi-gigabyte memory footprints per request.

The KV-cache size can be approximated as:

$$
\text{KV size} = 2 \times \text{num layers} \times \text{num heads} \times \text{seq len} \times \text{head dim}
$$

Assume:

- 32 layers, 80 heads, `head_dim = 128`, `seq_len = 4K`
- FP16 KV cache: $2 \times 32 \times 80 \times 4K \times 128 \times 2$ bytes $\approx 5.2$ GB per sample
- int8 KV cache: approximately half of that, or about 2.6 GB per sample

## Logical vs Physical View

Understanding KV cache requires separating the logical object seen by the model from the physical object managed by the runtime. Logically, a request is a growing token sequence with saved attention state. Physically, that state is stored as many fixed-size memory objects that can be allocated, mapped, reused, and reclaimed over time.

### Requests, Tokens, Blocks, and Pages

In a serving system, a request is the application-level object, a token is the sequence-level object, and a KV block is the runtime-level memory object used to manage the KV cache. A request corresponds to one logical token sequence. As prefill and decode proceed, every token in that sequence produces keys and values at every layer, and those tensors are appended to the request's KV cache. If the runtime uses a block size of $B$ tokens, then a request of length $L$ requires approximately

$$
N_{\text{blocks}} = \left\lceil \frac{L}{B} \right\rceil
$$

KV blocks to store its context state.

Accordingly, a request is not mapped to one monolithic contiguous KV allocation. It is mapped to a list of blocks:

$$
\text{request} \rightarrow \text{token sequence} \rightarrow \text{KV blocks}
$$

The final block is often only partially filled, so the block size determines the granularity of tail waste.

In the vLLM context, page is primarily a description of this organization style rather than a distinct object above blocks. A paged KV cache means that the KV state of one request is assembled from many fixed-size blocks rather than one contiguous physical segment. In this sense, page refers to the paged organization, while block refers to the actual allocation unit.

## PagedAttention

### Why PagedAttention Is Needed

PagedAttention is not mainly about how data are laid out inside one block. Its central purpose is to define how a growing request is represented by multiple blocks while preserving a logically contiguous KV address space.

The underlying problem is that a naive contiguous-allocation design does not match the way serving workloads evolve. Requests arrive with different lengths, they grow during decoding, and they leave at different times. A memory manager that insists on large contiguous allocations will fragment quickly and force costly data movement.

First, it avoids reserving one large contiguous memory region for each request. If the KV cache for a request had to remain physically contiguous, mixed long and short requests would quickly create external fragmentation.

Second, it avoids expensive data movement when a request grows. During autoregressive decoding, new tokens are appended to the context one by one. Under a contiguous-allocation scheme, a request that outgrows its original region would require copying its existing KV tensors to a larger contiguous region. Under PagedAttention, the runtime only needs to attach new blocks.

Third, it makes sharing, eviction, and scheduling naturally block-granular. Prefix caching reuses complete blocks, and memory reclamation or scheduler-driven eviction also operates at block granularity.

### Contiguous Within a Block, Non-Contiguous Across Blocks

PagedAttention does not imply that the GPU reads KV tensors in a fully random way. A more precise description is contiguous within a block and potentially non-contiguous across blocks. The runtime maintains a block table that maps logical sequence positions to physical blocks. The attention kernel then uses this mapping to locate the physical blocks associated with a request and performs reads and computation block by block.

The indirection introduced by PagedAttention is therefore block-level rather than token-level. A block can still use a regular memory layout favorable to vectorized access and high-bandwidth reads, while non-contiguity appears mainly in the list of physical blocks that together form one request.

## KV Manager

KV cache describes what is stored; a KV manager describes how that storage is allocated, mapped, reused, reclaimed, and protected under load. Once serving enters the long-context or high-concurrency regime, this manager effectively becomes one of the central runtime components.

### Block Table and Allocation Granularity

Because one request is mapped to a list of physical blocks rather than one contiguous region, the runtime needs metadata that records this mapping. The block table serves this role: it translates logical sequence positions into the physical blocks that actually hold the KV tensors.

The choice of block size matters because it sets the allocation granularity of the manager. Small blocks reduce tail waste and allow finer-grained reuse, but they make the block table longer and can increase management overhead. Large blocks simplify metadata and can be friendlier to some kernels, but they waste more space at the tail and make reuse coarser.

### A Concrete Example

Suppose the block size is 16 tokens and one request currently has 53 tokens in context. The runtime will need

$$
\left\lceil \frac{53}{16} \right\rceil = 4
$$

blocks to store its KV state. A logical view might look like:

```text
request tokens:
[t1 ... t16] [t17 ... t32] [t33 ... t48] [t49 ... t53]
```

The corresponding block table could then point to four physical blocks:

```text
logical block 0 -> physical block 42
logical block 1 -> physical block 107
logical block 2 -> physical block 18
logical block 3 -> physical block 203
```

The important point is that the physical blocks do not need to be contiguous. The first three logical blocks are full, while the last one is only partially filled. That last block is where tail waste appears. If another request shares the first 32 tokens exactly, then the first two full blocks may be reused directly by prefix caching, while the partially filled tail block cannot usually be reused as a full-block cache hit.

### Prefix Caching Is Reuse at Block Granularity

In vLLM, automatic prefix caching does not reuse arbitrary token ranges. It reuses complete blocks. Let the block size be $B$. If two requests share a common prefix of length $L_{\text{shared}}$, then at most

$$
\left\lfloor \frac{L_{\text{shared}}}{B} \right\rfloor
$$

complete blocks can be reused directly. A partially filled tail block usually cannot be treated as a cache hit. Therefore, the block size affects not only memory-allocation granularity but also prefix-cache reuse granularity.

From the perspective of the manager, prefix caching is a reuse policy over previously materialized blocks. It improves performance only when requests share enough aligned prefix structure for those blocks to be looked up and kept alive efficiently.

### Preemption, Reclamation, and Memory Pressure

In vLLM, preemption means that when the GPU no longer has enough available KV blocks to satisfy the current scheduling decision, the runtime temporarily interrupts one or more in-flight requests and reclaims or migrates the KV resources they occupy so that other requests can proceed. It is fundamentally a scheduling response to memory pressure rather than a numerical property of the model.

Preemption marks the boundary between a healthy operating regime and a memory-stressed one. A system may remain alive after preemption begins, but repeated preemption usually indicates that concurrency has exceeded what the current KV budget can support efficiently.

At the systems level, preemption usually indicates that the configuration is close to or already hitting the KV-capacity wall. Typical triggers include excessive request concurrency, very long contexts, large output lengths, insufficient tensor parallelism leading to a small per-GPU KV budget, or overly aggressive settings of `max-num-seqs` or `max-num-batched-tokens`.

### Interaction with the Scheduler

The KV manager and the scheduler are tightly coupled. Admission-control knobs such as `max-num-seqs` and `max-num-batched-tokens` are not only scheduler parameters; they are also indirect controls on how much pressure is placed on KV allocation and reclamation. In practice, many serving pathologies that look like scheduler issues are actually symptoms of the KV manager being forced into a memory-stressed operating regime.

## Attention Variants and KV Cost

In transformer models, the attention module computes:

$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

Every token produces:

- **Q (Query)**, derived from the current token's hidden state
- **K (Key)** and **V (Value)**, derived from the context and stored in the KV cache

The difference among MHA, MQA, and GQA lies in how many Key/Value projections are used relative to the number of Query heads.

In inference, MHA, MQA, and GQA are not only architectural details; they directly change how much KV state must be stored per token, which in turn changes feasible context length and concurrency.

### MHA: Multi-Head Attention

Each attention head has its own set of projections:

$$
Q_i = X W_{Q, i}, \quad K_i = X W_{K, i}, \quad V_i = X W_{V, i}
$$

- The number of Q, K, and V heads is the same.

### MQA: Multi-Query Attention

All heads share one K projection and one V projection:

$$
K_{\text{shared}} = X W_K, \quad V_{\text{shared}} = X W_V
$$

- There are multiple Q heads but only one shared K/V pair.
- The same K/V tensors are reused by all heads during inference.
- KV-cache size is reduced substantially relative to standard MHA.

### GQA: Grouped-Query Attention

Queries are split into groups, and each group shares its own K/V:

$$
\text{Group } g: \quad Q_{g, *}, K_g, V_g
$$

- K/V heads are fewer than Q heads.
- GQA is a compromise between the representational flexibility of MHA and the memory efficiency of MQA.

## Related

- [Metrics](metrics.md) defines TTFT, TPOT, and TPS.
- [Serving Runtime](serving-runtime.md) covers scheduler behavior once KV memory becomes the bottleneck.
- [Parallelism](parallelism.md) explains how DP and TP change memory budget, throughput, and communication.
