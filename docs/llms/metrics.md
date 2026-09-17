# Metrics

Latency and throughput metrics are the main interface between runtime behavior and user-visible serving performance. Most serving optimizations are not meaningful in isolation: a technique matters only if it changes what users experience or how much work the system can sustain. TTFT, TPOT, and TPS provide the vocabulary for describing that change. Without them, it becomes difficult to explain whether a scheduling policy improves responsiveness, whether a memory optimization really increases concurrency, or whether a deployment trade-off helps latency at the expense of throughput.

## Latency

Latency is the time taken to process a request and generate output tokens. In LLM inference, latency is often measured with two main metrics.

The reason latency is split into multiple metrics is that a single end-to-end number hides different bottlenecks. Some techniques primarily improve how fast the model starts responding, whereas others improve steady-state generation speed.

### Time to First Token (TTFT)

Time to First Token (TTFT) is the latency between when a request is received by the inference server and when the first output token is generated and returned to the client. It measures how quickly the model begins responding and is one of the most important user-perceived metrics in streaming systems:

$$
\text{TTFT} = t_{\text{first token emitted}} - t_{\text{request received}}
$$

TTFT mainly captures the prefill phase plus scheduling and framework overhead. A useful decomposition is:

$$
\small
\begin{aligned}
\text{TTFT} = t_{\text{queue}} + t_{\text{input processing}} + t_{\text{prefill compute}} + t_{\text{first decode step (optional)}} + t_{\text{framework overhead}}
\end{aligned}
$$

where:

- $t_{\text{queue}}$ is the waiting time before the scheduler admits the request.
- $t_{\text{input processing}}$ includes tokenization, multimodal preprocessing, and request formatting.
- $t_{\text{prefill compute}}$ is the forward pass over all input tokens, during which the KV cache is populated.
- $t_{\text{first decode step (optional)}}$ appears in runtimes that separate prefill from the first decode step.
- $t_{\text{framework overhead}}$ includes kernel launch latency, synchronization, allocation, sampling, and streaming overhead.

For long-context inputs, TTFT is usually dominated by $t_{\text{prefill compute}}$.

### Time per Output Token (TPOT)

Time per Output Token (TPOT) measures the average latency required to generate each token after the first token has been emitted. It characterizes steady-state decode performance. For a request that generates $N$ output tokens:

$$
\text{TPOT} = \frac{t_\text{last token emitted} - t_\text{first token emitted}}{N - 1}
$$

Equivalently, if total end-to-end latency is $T_{\text{total}}$:

$$
\text{TPOT} = \frac{T_{\text{total}} - \text{TTFT}}{N - 1}
$$

In steady-state decoding, each output token incurs:

$$
\text{TPOT} = t_{\text{decode compute}} + t_{\text{communication}} + t_{\text{scheduling}} + t_{\text{framework overhead}}
$$

where:

- $t_{\text{decode compute}}$ is the forward pass for one decode step using the KV cache.
- $t_{\text{communication}}$ includes cross-GPU collectives, expert routing, and host-device transfers triggered by runtime behavior.
- $t_{\text{scheduling}}$ includes batching, coordination, admission control, and KV-cache management.
- $t_{\text{framework overhead}}$ includes sampling, launches, synchronization, and token-streaming overhead.

Unlike TTFT, TPOT does not recompute the full prompt. However, it still depends on effective context length because each decode step attends to all cached tokens.

## Throughput

### Tokens per Second (TPS)

Tokens per Second (TPS) measures system throughput, that is, how many tokens are processed or generated per second under a given serving configuration. Over a time interval $\Delta t$:

Once a service must support many concurrent users, the central question is no longer only how fast one request finishes, but how much aggregate work one deployment can sustain before queues begin to grow.

$$
\text{TPS} = \frac{N_\text{tokens}}{\Delta t}
$$

In many cases, TPS refers specifically to decode throughput:

$$
\text{TPS}_\text{decode} = \frac{N_\text{output tokens}}{\Delta t}
$$

For a single sequential request, it is approximately the inverse of TPOT:

$$
\text{TPS}_\text{per request} \approx \frac{1}{\text{TPOT}}
$$

In practical serving systems with continuous batching and concurrent decoding, aggregate throughput scales with the effective batch size $B$. For a single replica:

$$
\text{TPS}_\text{replica} \approx \frac{B}{\text{TPOT}}
$$

and with $R$ data-parallel replicas:

$$
\text{TPS}_\text{total} \approx R \times \frac{B}{\text{TPOT}}
$$

For long-context or multimodal workloads, it is also useful to consider end-to-end throughput:

$$
\text{TPS}_\text{end-to-end} = \frac{N_\text{prompt tokens} + N_\text{output tokens}}{\Delta t}
$$

Overall, TPS is influenced by model size, parallelism strategy, effective batch size, kernel efficiency, interconnect bandwidth, and the balance between prefill and decode.
