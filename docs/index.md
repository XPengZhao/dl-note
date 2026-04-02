# DL Notes

A focused knowledge base for modern deep learning systems, from hardware fundamentals to LLM serving and neural rendering.

## Start Here


1. [Hardware](hardware.md) - GPU/NPU basics, memory systems, and practical GPU configuration
2. [Tensor Operations](tensor.md) - core tensor manipulations, activation functions, and CUDA Graph notes
3. [AI Infra Metrics](llms/metrics.md) - TTFT, TPOT, and TPS
4. [KV Cache](llms/kv-cache.md) - KV memory semantics, paged blocks, and prefix reuse
5. [Serving Runtime](llms/serving-runtime.md) - chunked prefill, admission control, and runtime-side stability
6. [Parallelism](llms/parallelism.md) - DP and TP from the perspective of memory, throughput, and communication
7. [Decoding and Sampling](llms/decoding.md) - sampling policies and speculative decoding
8. [Training Objective](llms/training-objective.md) - autoregressive pre-training objective
9. [Models](llms/models.md) - model-specific notes (Qwen3-Omni, DFlash) and practical serving commands
10. [Neural Graphics](neural-graphics.md) - NeRF and Flow Matching foundations

## Documentation Map

### Systems and Infrastructure

- [Hardware](hardware.md)
- [Tensor Operations](tensor.md)

### AI Infra

- [Metrics](llms/metrics.md)
- [KV Cache](llms/kv-cache.md)
- [Serving Runtime](llms/serving-runtime.md)
- [Parallelism](llms/parallelism.md)
- [Decoding and Sampling](llms/decoding.md)
- [Training Objective](llms/training-objective.md)

### Models

- [Models](llms/models.md)

### Graphics and Generative Modeling

- [Neural Graphics](neural-graphics.md)

## Scope

This site emphasizes practical understanding:

- concise theory with equations where useful
- implementation-minded notes and runnable snippets
- serving and performance considerations for real workloads
