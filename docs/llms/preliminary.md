# AI Infra Preliminary

## Metrics

### Latency

Latency is the time taken to process a request and generate output tokens. In LLM inference, latency is often measured in two main metrics:

#### Time to first token (TTFT)

Time to First Token (TTFT) is the latency between when a request is received by the inference server and when the first output token is generated and returned to the client. It measures how quickly the model begins responding, and is a primary user-perceived latency metric in streaming LLM systems. Formally:

$$
\text{TTFT} = t_{\text{first token emitted}} - t_{\text{request received}}
$$

TTFT mainly captures the prefill phase (processing the input context) plus scheduling and system overhead. It does not include the time required to generate the rest of the output tokens. In details TTFT can be decomposed into:

$$
\small
\begin{aligned}
\text{TTFT} = t_{\text{queue}} + t_{\text{input processing}} + t_{\text{prefill compute}} + t_{\text{first decode step (optional)}} + t_{\text{framework overhead}}
\end{aligned}
$$

where:

-  $t_{\text{queue}}$ is the time spent waiting in the scheduler before the request is admitted for execution.
- $t_{\text{input processing}}$ includes tokenization, multimodal preprocessing (e.g., image or video encoding), and any request-level normalization or formatting.
- $t_{\text{prefill compute}}$ is the time required to run the forward pass over all input tokens (complexity is $O(L)$ in sequence length $L$), during which the KV cache is populated and the logits at the final prompt position are produced.
- $t_{\text{first decode step (optional)}}$ accounts for an additional single-step decode iteration in implementations that separate prefill and generation scheduling. In such systems, the first output token is emitted during a subsequent decode pass rather than directly from the prefill logits. In optimized implementations (recent versions of vLLM), this step may be fused with prefill and therefore negligible.
- $t_{\text{framework overhead}}$ captures auxiliary costs such as CUDA kernel launches, synchronization, memory allocation, sampling (softmax and top-k/top-p), and network transmission of the first streamed token.

For long-context inputs, TTFT is typically dominated by $t_{\text{prefill compute}}$, with the remaining terms contributing comparatively small overhead.

#### Time per output token (TPOT)

Time per Output Token (TPOT) measures the average latency required to generate each token after the first token has been emitted. It characterizes the steady-state decode performance of an autoregressive model and directly determines streaming speed. Formally, for a request generating $N$ output tokens:

$$
\text{TPOT} = \frac{t_\text{last token emitted} - t_\text{first token emitted}}{N - 1}
$$

Equivalently, if total end-to-end latency is $T_{\text{total}}$,

$$
\text{TPOT} = \frac{T_{\text{total}} - \text{TTFT}}{N - 1}
$$

In steady-state decoding, each output token requires:

$$
\text{TPOT} = t_{\text{decode compute}} + t_{\text{communication}} + t_{\text{scheduling}} + t_{\text{framework overhead}}
$$

where:

- $t_{\text{decode compute}}$ is the forward pass for a single token using the KV cache (complexity per step is $O(1+\text{num speculative tokens})$ in sequence length but proportional to model size).
- $t_{\text{communication}}$ captures the cost of cross-GPU synchronization and data exchange, such as tensor-parallel collectives (all-reduce, all-gather), expert routing and aggregation in MoE models (all-to-all, dispatch/combine), as well as any host–device memory transfers (H2D/D2H) triggered by scheduling or memory management.
- $t_{\text{scheduling}}$ includes runtime scheduling overhead such as continuous batching, request coordination, admission control, and KV cache allocation, paging, or memory management.
- $t_{\text{framework overhead}}$ includes auxiliary execution costs such as sampling (softmax, top-k/top-p), kernel launches, synchronization, input/output marshaling, post-processing of results, and token streaming.

Unlike TTFT, TPOT does not require recomputation over the full prompt. However, it still depends on the effective context length, since each decode step performs attention over all cached tokens. Therefore, TPOT scales approximately linearly with the total context length (prompt + generated tokens), though the per-token cost is significantly smaller than prefill.

### Throughput

#### Tokens per Second (TPS)

Tokens per Second (TPS) measures the throughput of an LLM inference system, i.e., how many tokens are processed or generated per second under a given serving configuration. Unlike TTFT and TPOT, which are latency metrics defined at the request level, TPS is a system-level metric that reflects overall serving capacity under load. Over a time interval over a time interval $\Delta t$, TPS can be expressed as:

$$
\text{TPS} = \frac{N_\text{tokens}}{\Delta t}
$$

In many cases, TPS refers specifically to decode throughput, counting only generated output tokens:

$$
\text{TPS}_\text{decode} = \frac{N_\text{output tokens}}{\Delta t}
$$

and for a single sequence decoded sequentially, it is approximately the inverse of TPOT:

$$
\text{TPS}_\text{per request} \approx \frac{1}{\text{TPOT}}
$$

However, in practical serving systems with continuous batching and concurrent decoding, aggregate throughput scales with the effective batch size $B$. For a single replica, the system throughput can be approximated as

$$
\text{TPS}_\text{replica} \approx \frac{B}{\text{TPOT}}
$$

and with $R$ data-parallel (DP) replicas,

$$
\text{TPS}_\text{total} \approx R \times \frac{B}{\text{TPOT}}
$$

Importantly, for long-context or multimodal workloads, prompt processing can contribute a substantial fraction of total computation. In such cases, it is often necessary to consider end-to-end throughput, which accounts for both prompt and generated tokens:

$$
\text{TPS}_\text{end-to-end} = \frac{N_\text{prompt tokens} + N_\text{output tokens}}{\Delta t}
$$

Overall, TPS is influenced by model size, parallelism strategy, effective batch size, kernel efficiency, interconnect bandwidth, and the relative balance between prefill and decode workloads.


## KV Cache

### Memory cost

The KV cache size is calculated as:

$$
\text{KV size} = 2 \times \text{num layers} \times \text{num heads} \times \text{seq len} \times \text{head dim}
$$

Assume:

- 32 layers, 80 heads, head_dim = 128, seq_len = 4K.
- FP16 KV cache: 2 × 32 × 80 × 4K × 128 × 2 bytes = **~5.2 GB** per sample.
- C8 (int8 KV cache): **half that → ~2.6 GB** per sample.

## Attention

In transformer models, the attention module computes:

$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

Every token produces:

- **Q (Query)** — derived from the current token’s hidden state
- **K (Key)**, **V (Value)** — derived from the context (previous tokens) and stored in the **KV cache**

The difference between **MHA**, **MQA**, and **GQA** lies in **how many Key/Value projections** are used relative to the number of Query heads.

### **MHA — Multi-Head Attention**

Each attention head has its own set of projections:

$$
Q_i=X W_{Q, i}, \quad K_i=X W_{K, i}, \quad V_i=X W_{V, i}
$$

- Q heads = K heads = V heads

### **MQA — Multi-Query Attention**

All heads **share one** K and V projection:

$$
K_{\text {shared }}=X W_K, \quad V_{\text {shared }}=X W_V
$$

- **Multiple Q heads**, but **only one pair of shared K/V**.
- During inference, the same K/V are reused for all heads.
- KV cache size reduced by ~#heads×
- **LLaMA 2 7B**, **Mistral**, and most optimized LLMs use MQA to cut memory usage.

### **GQA — Grouped-Query Attention**

Queries are split into groups, and each group shares its own K/V:

$$
\text { Group } g: \quad Q_{g, *}, K_g, V_g
$$

- Suppose you have 8 query heads but only 2 groups → 4 Q heads per group share one K/V.
- **K/V heads < Q heads**.
- **LLaMA 2 70B**, **Mixtral**, **GPT-J-6B**, etc. use GQA for better memory efficiency.

## Sampling
### Temperature

Temperature $T$ controls "sharpness" of the probability distribution. Given model logits $z_i$, temperature sampling rescales them before softmax:

$$
p_i=\frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

- $T<1$ Sharper (more peaked). Model becomes confident, less random. $T>1$ Flatter (more uniform). Model becomes exploratory, more diverse.
- Temperature keeps the order of logits because dividing by a positive number does not change ranking (rank-preserving). If you use top-k = 1 (i.e., greedy), temperature has no effect at all.
- With a temperature $T \rightarrow 0$, the softmax distribution becomes arbitrarily sharp and converges to greedy (argmax) sampling. For example, at $T=10^{-6}$, the highest logit dominates the distribution, so sampling is effectively greedy in practice.
- At $T \rightarrow \infty$, the distribution becomes uniform (all tokens equally likely).

### Top-k Sampling

After computing probabilities (after temperature), sort tokens by probability, and keep only the k highest. Then renormalize $p_i^{\prime}$ to get final distribution for sampling.

$$
\begin{aligned}
S_k & =\text { top-k tokens by } p_i \\
p_i^{\prime} & = \begin{cases}\frac{p_i}{\sum_{j \in S_k} p_j} & i \in S_k \\
0 & \text { otherwise }\end{cases}
\end{aligned}
$$

Then you sample from that truncated distribution.

- Small k (e.g., 1–10): deterministic and safe (model never picks rare words).
- Large k (e.g., 50–100): more variety but higher risk of off-topic words.

### Top-p (nucleus) Sampling

“nucleus” literally means core, center, or the essential central part of something. Instead of a fixed k, we choose the smallest set of tokens whose cumulative probability $\geq p$.

$$
\begin{aligned}
S_p&=\left\{i: \sum_{j \in S_p} p_j \geq p\right\} \\
p_i^{\prime}&= \begin{cases}\frac{p_i}{\sum_{j \in S_p} p_j} & i \in S_p \\
0 & \text { otherwise }\end{cases}
\end{aligned}
$$

**Key difference vs top-k:**

Top-p adapts to uncertainty. Top-k is about rank (k highest scores). Top-p is about mass (the core mass of the distribution).

- If the model is confident (one token dominates), very few tokens are kept.
- If it’s uncertain (many tokens have similar probabilities), the nucleus expands.

### Min-p

Min-p filters out tokens that are too improbable relative to the top-1. This is newer but increasingly popular (used in Mistral, Gemini, etc.). Instead of fixing $k$ or $p$, you keep tokens above a minimum probability threshold relative to the top-1 token:

$$
S_{\min }=\left\{i: p_i \geq \min -\mathrm{p} \times p_{\max }\right\}
$$

If min-p = 0.1, then any token whose probability is at least 10% of the most likely token is kept.Then normalize:

$$
p_i^{\prime} = \frac{p_i}{\sum_{j \in S_{\min}} p_j}
$$

- Min-p adaptive to context like top-p, but simpler. It means the number of candidate tokens changes depending on what the model is predicting. If the model is confident, the candidate set is small. If the model is uncertain, the candidate set becomes larger.
- Avoids cutting off plausible low-rank tokens when the distribution is flat.
- More numerically stable for long-tail vocabularies.

### Put them together
You usually combine these in a specific order: Temperature → Softmax → (top-k / top-p / min-p) → Renormalize → Sample

When both top-k and top-p are enabled, you always keep the intersection, which is the smaller of the two sets.

### Example code

Code snippet from [Omniinfer Sampler](https://gitee.com/omniai/omniinfer/blob/master/omni/adaptors/vllm/sample/sampler.py).

<details>
<summary>Top-k and Top-p implementation</summary>

```python
def apply_top_k_top_p(
    logits_or_prob: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
    is_logits: bool,
) -> torch.Tensor:
    if p is None:
        if k is not None:
            logits_or_prob = apply_top_k_only(logits_or_prob, k, is_logits)
        if is_logits:
            probs = logits_or_prob.softmax(dim=-1, dtype=torch.float32)
        else:
            probs = logits_or_prob / logits_or_prob.sum(dim=-1, keepdim=True)
        return probs, None

    logits_or_prob_sort, logits_or_prob_idx = logits_or_prob.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_or_prob_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_or_prob_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_or_prob_sort < top_k_mask
        logits_or_prob_sort.masked_fill_(top_k_mask, -float("inf") if is_logits else 0)

    # Apply top-p.
    if is_logits:
        probs_sort = logits_or_prob_sort.softmax(dim=-1)
    else:
        probs_sort = logits_or_prob_sort / logits_or_prob_sort.sum(dim=-1, keepdim=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    # at least one
    top_p_mask[:, -1] = False
    probs_sort.masked_fill_(top_p_mask, 0)
    probs = probs_sort / probs_sort.sum(dim=-1, keepdim=True)
    return probs, logits_or_prob_idx
```

```python
def apply_top_k_only(
    logits_or_prob: torch.Tensor,
    k: torch.Tensor,
    is_logits: bool,
) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits_or_prob.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits_or_prob.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits_or_prob.masked_fill_(logits_or_prob < top_k_mask, -float("inf") if is_logits else float(0))
    return logits_or_prob
```
</details>

## Speculvative decoding


###  Rejection Sampler

To sample $x \sim p(x)$,  we instead sample $x \sim q(x)$, keeping it if $q(x) \leq p(x)$, and in case $q(x) > p(x)$, we reject the
sample with probability $1 - \frac{p(x)}{q(x)}$ and resample $x$  from an adjusted distribution $p^{\prime} = \text{norm}(\max(0, p(x) - q(x)))$ instead.

**NOTE:** For for any distributions $p(x)$ and $q(x)$, the tokens sampled via rejection sampling from $p(x)$ and $q(x)$ are distributed identically to those sampled from $p(x)$ alone. The proof is in Appendix A [1].

Rejection sampling works by decomposing the target distribution $p(x)$ into two parts:

-  The accept branch, where tokens are accepted based on the ratio $\frac{p(x)}{q(x)}$, where $q(x)$ is the proposal (draft) distribution.

$$
  p(\text{accepted}, x = x^{\prime}) = q(x^{\prime}) \cdot \min\left(\frac{p(x^{\prime})}{q(x^{\prime})}, 1\right) = \min(p(x^{\prime}), q(x^{\prime}))
$$

- The reject branch, where tokens are rejected and resampled from the adjusted distribution $p^{\prime}(x)$. The probability of all tokens being sampled from $q(x)$ being rejected as:

$$
\begin{aligned}
    p(\text{rejected}) &= \sum_{x} q(x) \left(1 - \min\left(\frac{p(x)}{q(x)}, 1\right)\right) \\
    &= \sum_{x} \max(0, q(x) - p(x)) \\
\end{aligned}
$$

$$
\begin{aligned}
    p(\text{rejected}) &= \sum_{x} q(x) \left(1 - \min\left(\frac{p(x)}{q(x)}, 1\right)\right) \\
    &= \sum_{x} \max(0, q(x) - p(x)) \\
\end{aligned}
$$



**Example Code**

Code snippet from [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py).

<details>
<summary>Rejection sampling kernel in Triton</summary>

```python
# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            if NO_DRAFT_PROBS:
                draft_prob = 1
            else:
                draft_prob = tl.load(
                    draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                )
            target_prob = tl.load(
                target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
            )
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            # NOTE(woosuk): While the draft probability should never be 0,
            # we check it to avoid NaNs. If it happens to be 0, we reject.
            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                # Accept.
                token_id = draft_token_id
            else:
                # Reject. Use recovered token.
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )
```

</details>

<details>
<summary>Resample kernel in Triton</summary>


```python
@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    vocab_offset = tl.arange(0, PADDED_VOCAB_SIZE)
    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
        prob = tl.load(
            target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=((vocab_offset < vocab_size) & (vocab_offset != draft_token_id)),
            other=0,
        )
    else:
        draft_prob = tl.load(
            draft_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0,
        )
        target_prob = tl.load(
            target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0,
        )
        prob = tl.maximum(target_prob - draft_prob, 0)
        # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
        # `tl.argmax` will select the maximum value.

    q = tl.load(
        q_ptr + req_idx * vocab_size + vocab_offset,
        mask=vocab_offset < vocab_size,
        other=float("-inf"),
    )
    recovered_id = tl.argmax(prob / q, axis=-1)
    tl.store(output_token_ids_ptr + start_idx + pos, recovered_id)

```
</details>

**Reference**

[[1]](https://arxiv.org/pdf/2211.17192) Y Leviathan, M Kalman, Y Matias, "Fast Inference from Transformers via Speculative Decoding", ICML 2023.


## Pre-training Objective of LLMs

A large language model trained from scratch is optimized using maximum likelihood estimation (MLE) over a large text corpus.

Given a sequence of tokens $(t_1, t_2, \ldots, t_n)$ sampled from the data distribution $p_\text{data}$ the model defines an autoregressive factorization:

$$
p_\theta(t_1, t_2, \ldots, t_n) = \prod_{i=1}^n p_\theta(t_i | t_1, t_2, \ldots, t_{i-1})
$$

The training objective is to maximize the log-likelihood of the data, which is equivalent to minimizing the negative log-likelihood (NLL):

$$
\mathcal{L}(\theta) = - \mathbb{E}_{(t_1, t_2, \ldots, t_i) \sim p_\text{data}} \left[ \log p_\theta(t_i | t_1, t_2, \ldots, t_{i-1}) \right]
$$

In practice, this loss is implemented as token-level cross-entropy with one-hot targets:

$$
\mathcal{L} = -\frac{1}{N} \sum_{\text{tokens}} \log p_\theta(t_i | t_1, t_2, \ldots, t_{i-1})
$$

This objective corresponds to minimizing the cross-entropy between the empirical data distribution and the model distribution. or equivalently, minimizing $KL(p_\text{data} || p_\theta)$.
