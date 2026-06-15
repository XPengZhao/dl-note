# Roofline Model for Inference

Latency and throughput metrics tell us what users observe, but they do not by themselves explain why a serving configuration stops improving. A system may show high TPOT, low TPS, or weak scaling under larger batch sizes, yet those symptoms still leave one core question unresolved: is the workload limited mainly by compute or by data movement? The roofline model is a compact way to answer that question.

For inference, this matters because not all stages stress hardware in the same way. Prefill and decode can run on the same model and the same GPU while sitting in very different performance regimes. Many practical optimizations, including batching, KV compression, quantization, and speculative decoding, are best understood as attempts to move the workload away from one limiting roof and toward a higher one.

## Why Roofline Matters

When people say an inference optimization "improves utilization," the statement is often incomplete. Utilization can improve for different reasons:

- the workload performs more math per byte moved
- the workload moves fewer bytes for the same mathematical result
- fixed runtime overhead is amortized over a wider step
- the original bottleneck is removed and a different one becomes dominant

The roofline model gives a common language for separating these cases. It does not replace request-level analysis, scheduler analysis, or communication analysis. It provides a first-principles upper-bound view of device-side performance.

## The Basic Model

The classical roofline model relates achieved performance to two hardware limits:

- peak compute throughput
- peak memory bandwidth

Let:

- $P$ be achieved performance in FLOPs/s
- $P_{\max}$ be peak compute throughput
- $B_{\max}$ be peak memory bandwidth in bytes/s
- $I$ be arithmetic intensity in FLOPs per byte moved

Then the attainable performance is bounded by:

$$
P \le \min \left(P_{\max},\; I \cdot B_{\max}\right)
$$

This expression separates two regimes.

### Memory-Bound Regime

When arithmetic intensity $I$ is small, the product $I \cdot B_{\max}$ stays below peak compute. In this region, performance is limited mainly by how fast data can be moved:

$$
P \approx I \cdot B_{\max}
$$

Increasing raw compute capability alone does not help much here unless bandwidth or arithmetic intensity also improves.

### Compute-Bound Regime

When arithmetic intensity is high enough, the bandwidth-derived ceiling rises above peak compute:

$$
I \cdot B_{\max} \ge P_{\max}
$$

The workload then approaches the compute roof:

$$
P \approx P_{\max}
$$

In this region, performance gains come mainly from improving kernel efficiency, tensor-core utilization, or parallel execution quality rather than from reducing memory traffic alone.

## Arithmetic Intensity

Arithmetic intensity is the bridge between workload structure and hardware limits:

$$
I = \frac{\text{FLOPs}}{\text{Bytes moved}}
$$

The point is not to compute this value with perfect precision for every kernel. The point is to reason about whether a workload performs a large amount of math relative to the memory it must read and write.

A workload with low arithmetic intensity tends to:

- spend much of its time waiting on memory traffic
- show limited gains from higher peak FLOPs alone
- benefit from techniques that reduce bytes moved or improve reuse

A workload with high arithmetic intensity tends to:

- make better use of tensor cores or SIMD units
- benefit more directly from better kernels and more available compute
- flatten less quickly when the batch or step width increases

## Mapping Roofline to Inference

The main value of roofline in AI Infra is not the abstract diagram. The main value is that it explains why different inference stages behave differently.

### Prefill

Prefill processes the prompt in bulk. Sequence length is large, matrix shapes are wider, and the hardware usually has more parallel work available in one pass. As a result, prefill is often closer to the compute-bound side than decode, especially when prompt length or batch size is substantial.

This does not mean prefill is always compute-bound. Under small batches, multimodal preprocessing overhead, or distributed communication pressure, other ceilings can dominate. The safer statement is that prefill is usually more compute-favorable than decode.

### Decode

Decode runs one autoregressive step at a time. Each step is narrow, latency-sensitive, and repeatedly touches KV state accumulated from all previous tokens. In practical serving, decode is therefore much more likely to fall into a memory-bound or overhead-dominated regime.

This is why online serving often feels counterintuitive: the mathematically smaller phase can become the operational bottleneck. Decode may perform less total work per step than prefill, yet achieve worse hardware efficiency because the work is too narrow and too sequential to fully utilize the device.

## What Roofline Explains and What It Does Not

Roofline is useful, but it is not the whole serving story.

It explains well:

- whether a device-side workload is primarily constrained by compute or by memory traffic
- why wider batch shapes often improve performance
- why some optimizations help more in decode than in prefill
- why reducing bytes moved can matter as much as reducing FLOPs

It does not fully explain:

- request queueing and scheduler delay
- prefix-routing fragmentation across replicas
- host-device synchronization overhead outside the kernel view
- inter-GPU communication ceilings in distributed execution
- application-layer fan-out and orchestration overhead

So the right way to use roofline is as a local performance model within a larger request lifecycle, not as a substitute for the lifecycle itself.

## Reading Common Inference Optimizations Through Roofline

Once the roofline view is in place, many optimizations become easier to classify.

### KV Cache and Memory Traffic

Decode often reads large amounts of cached attention state relative to the amount of new computation introduced by one token step. That pushes arithmetic intensity downward. Techniques such as KV quantization, better cache layout, or more efficient block reuse matter because they reduce memory pressure or improve effective reuse in a regime that is already close to a bandwidth ceiling.

### Batching and Wider Execution

Larger effective batches usually improve utilization because they make kernels wider and amortize fixed overhead. In roofline terms, batching can move a workload toward a more compute-favorable regime by improving reuse and making the achieved performance point climb closer to the upper envelope.

The same logic also explains why batching eventually shows diminishing returns. Once the workload is already near the relevant roof, further widening helps less.

### Quantization

Quantization is sometimes described as a compute optimization and sometimes as a memory optimization. In inference, it is often both, but the dominant benefit depends on the original bottleneck. If the workload is memory-bound, smaller activations, weights, or KV tensors can reduce bandwidth pressure and capacity pressure. If the workload is already close to the compute roof, the realized gain may be much smaller unless the lower-precision kernels also increase achieved compute throughput.

### Speculative Decoding

Speculative decoding is best understood as a change in target-model work granularity. In ordinary autoregressive decode, the target model performs many narrow sequential steps. In speculative decoding, the target model verifies multiple proposed tokens in a wider step.

This can help for two related reasons:

- fixed overhead is amortized over more useful work
- the wider verification step often uses the hardware more efficiently than many isolated narrow steps

That does not mean speculative decoding should be reduced to a single phrase such as "it increases arithmetic intensity." A more accurate statement is that it reduces the number of expensive target-side sequential steps and may improve target-side utilization if the baseline decode path is underfilled. The gain shrinks when the baseline serving batch is already large and the target path is already close to its effective roof.

## Beyond Single-Device Roofline

The classical roofline model is most natural for one device or one kernel family. Real inference systems often hit additional ceilings above that level.

In distributed or MoE serving, achieved throughput may be bounded not only by HBM bandwidth and on-device compute, but also by:

- all-reduce or all-to-all bandwidth
- collective start-up latency
- pipeline bubbles
- scheduler-level admission and preemption behavior

For that reason, roofline should be introduced as the first useful ceiling model, not the final one. It tells us whether the local computation is bandwidth-limited or compute-limited. Then the broader system analysis asks whether communication, runtime policy, or workload shaping imposes an even lower ceiling.

## Practical Questions to Ask

When an inference optimization claims to improve performance, a roofline-oriented reading starts with four questions:

1. Does it reduce FLOPs, reduce bytes moved, or mainly amortize fixed overhead?
2. Does it help prefill, decode, or both?
3. Is the baseline workload memory-bound, compute-bound, or dominated by a higher-level system ceiling?
4. Does the optimization remove the current bottleneck, or merely shift the bottleneck elsewhere?

Those questions are often more useful than asking whether a technique is "good" in the abstract. A serving optimization is valuable only if it lifts the active ceiling for the workload that actually matters.

## Related

- [Metrics](metrics.md) defines TTFT, TPOT, and TPS, which describe the user-visible symptoms.
- [KV Cache](kv-cache.md) explains why decode often experiences strong memory pressure.
- [Serving Runtime](serving-runtime.md) explains scheduler and admission effects that roofline alone does not capture.
- [Decoding and Sampling](decoding.md) discusses speculative decoding as one concrete example of changing decode-side work shape.
