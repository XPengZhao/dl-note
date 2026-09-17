# Serving Runtime

Many serving bottlenecks appear only after deployment. Even when the model is fixed, the system still has to decide how requests are admitted, how long prompts are interleaved with decode work, and how memory pressure is handled in steady state. This is why runtime deserves to be discussed separately from model architecture: under real workloads, scheduling and memory management can substantially change the effective behavior of the same model.

## Scheduling

### Chunked Prefill

Chunked prefill is a serving-side optimization for long-context inference. Instead of processing the entire prompt in one prefill pass, the runtime splits it into smaller chunks and executes them incrementally. If the input length is $L$ and the chunk size is $C$, the prefill stage is decomposed into roughly

$$
N_{\text{chunks}} = \left\lceil \frac{L}{C} \right\rceil
$$

sub-requests. For chunk $j$, the model processes only the assigned token range and appends the resulting keys and values to the KV cache before moving to the next chunk.

A long prefill request is wide and compute-heavy, whereas decode consists of many small latency-sensitive iterations. If a very long prompt is executed as one monolithic prefill, it can occupy the GPU for a long interval and delay unrelated decode work. Chunked prefill reduces the scheduling granularity of prompt processing, allowing the runtime to interleave long-prefill requests with decode steps or shorter prompts from other users. In practice, this is mainly a fairness and responsiveness optimization under continuous batching.

Chunked prefill can also reduce peak runtime memory during prefill. When the prompt is processed all at once, the runtime may need larger temporary activations, attention workspaces, or other intermediate buffers for the full sequence. With chunking, these transient buffers scale with the chunk size rather than the full prompt length, which can help avoid OOM for very long inputs. However, it does not reduce the final KV cache footprint: if all prompt tokens remain in context, their keys and values still have to be stored. In other words, chunked prefill can lower transient prefill memory, but not the steady-state KV cache requirement.

This changes scheduling and peak-memory behavior rather than model semantics. The final KV cache and output logits are identical to those of an unchunked prefill, assuming the same attention mask and numerics. The trade-off is that very small chunks increase launch and scheduling overhead, while very large chunks recover the behavior of ordinary monolithic prefill. Conceptually, chunked prefill belongs to the serving runtime rather than the model architecture.

### Admission Control and Stability

In practice, serving stability is largely determined by admission-control knobs such as `max-num-seqs`, `max-num-batched-tokens`, and the effective GPU memory budget exposed to the runtime. These parameters define how much concurrent work the engine may admit before it approaches the KV-capacity wall.

Naively maximizing concurrency often backfires. A deployment that admits too much work can cross into a regime dominated by preemption, swapping, or long queueing delays, which lowers effective throughput and makes latency unpredictable.

The tuning objective is usually not to maximize theoretical concurrency at any cost. A more useful target is the stable operating region in which the runtime sustains high GPU utilization without repeatedly triggering preemption, swapping, or large tail-latency spikes. This is why runtime tuning is inseparable from KV-cache analysis: batching policy, scheduler behavior, and memory pressure are coupled.

### Cross-Stream Tensor Reuse

If a tensor is produced on one CUDA stream and consumed on another, a device-to-device copy is not required by default. The same storage can be reused across streams as long as the program establishes the correct ordering, typically through events or stream synchronization. The main cost is therefore synchronization and any lost overlap, not mandatory duplication of the tensor.

This topic matters because stream-level execution is often misunderstood. When engineers observe work crossing stream boundaries, it is easy to assume that the main penalty is copying, whereas in many cases the real issue is serialization, dependency tracking, or lost overlap.

An explicit copy becomes necessary only when the program intentionally creates a new buffer, changes layout, or moves data between devices. From a runtime-analysis perspective, cross-stream communication is usually a question of dependency management rather than a question of whether a new tensor payload must be copied.
