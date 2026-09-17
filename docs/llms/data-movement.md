# Data Movement and Communication in Inference

Inference performance is shaped not only by how much math a model performs, but also by how much state must be moved to support that math. In practice, many serving bottlenecks appear not because the device lacks theoretical FLOPs, but because weights, activations, KV state, or cross-device tensors must be fetched, written back, or exchanged at the wrong granularity or from the wrong level of the memory hierarchy.

This page introduces data movement as a systems object. The goal is to build a clean mental model before discussing specific cases such as KV cache pressure, distributed communication, or decode-side inefficiency.

## Data Movement Across Memory Levels

The first step is to avoid interpreting data movement too narrowly. Inference data movement is not only about loading a model file from storage or copying an input from CPU to GPU. Those operations exist, but the performance-critical path is usually the repeated movement of runtime state across the memory hierarchy while kernels are executing.

A useful simplified path is:

`storage / host memory -> GPU HBM -> on-chip cache / registers -> compute units`

In multi-GPU execution, one more path becomes important:

`GPU A HBM <-> GPU B HBM`

This abstraction is intentionally simple. The point is not to reproduce every detail of GPU microarchitecture. The point is to identify the boundaries across which data must travel before the hardware can consume it.

### Storage and Host Memory

The outermost layer includes model files, CPU-resident preprocessing results, tokenized inputs, and any host-side runtime metadata. These objects matter operationally, but they are usually not the dominant steady-state bottleneck in a well-structured online serving loop. Their main importance is that they define the entry path into device execution.

### GPU HBM

HBM is the main working memory of GPU inference. Model weights, activations, and KV cache usually reside here. HBM provides much higher bandwidth than host memory, but it is still far slower and farther away than on-chip storage. When people say a workload is memory-bound in serving, they are often referring to pressure at this level: too much useful work depends on repeatedly reading or writing HBM-resident state.

### On-Chip Cache and Registers

Closer to execution sit on-chip caches, shared memory, register files, and other short-latency working storage used by active kernels. These structures are much smaller than HBM but much cheaper to access. Kernel efficiency often depends on whether data can be reused here instead of being fetched again from HBM.

### Compute Units

Tensor cores, CUDA cores, or other execution units perform the actual arithmetic. They cannot make progress until the required operands have reached a form and location they can consume. This is why data movement is part of computation rather than a separate concern: every arithmetic step is gated by whether the necessary state is already near the compute units.

## Why the Hierarchy Matters

The same number of FLOPs can lead to very different realized performance depending on where the supporting data come from.

- If data are reused from on-chip storage, the effective supply cost is low.
- If kernels repeatedly fetch large state from HBM, bandwidth pressure rises.
- If tensors must cross GPUs, communication and synchronization become part of the critical path.
- If the execution path falls back to host-device interaction too often, latency increases sharply.

This is the systems reason that data movement deserves to be discussed explicitly. Not all bytes are equally expensive. The cost depends on which boundary they cross, how often they cross it, and whether they can be reused before being moved again.

## A Practical Classification

From this layered view, the most common movement categories in inference are:

- `host -> device`: inputs, preprocessing outputs, and occasional control-path transfers
- `HBM -> on-chip working set`: weights, activations, and KV state fetched by kernels
- `on-chip -> HBM`: outputs, updated activations, and newly materialized KV state written back
- `GPU <-> GPU`: hidden states, partial reductions, expert traffic, or other distributed tensors

For single-GPU serving, the dominant issue is often not host-to-device transfer but repeated HBM access during the inner execution loop. For multi-GPU serving, cross-device movement adds a new class of bottleneck on top of the local memory hierarchy.

## Related

- [Roofline Model for Inference](roofline.md) explains why bytes moved matter for performance ceilings.
- [KV Cache](kv-cache.md) is one of the most important examples of state whose repeated movement shapes decode efficiency.
- [Parallelism](parallelism.md) discusses when movement extends beyond one device and becomes communication.
