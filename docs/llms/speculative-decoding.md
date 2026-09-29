# Speculative Decoding

Autoregressive decode performs one small, strictly sequential step at a time, and those steps often underutilize modern accelerators. Speculative decoding addresses this bottleneck by replacing many narrow target-model steps with fewer, wider verification steps that amortize runtime overhead.

### Speculative Decoding Speedup

Speculative decoding accelerates inference of a target model $M_t$ using a smaller draft model $M_d$. In each decoding cycle, the draft model proposes $\gamma$ tokens, which are verified in parallel by the target model. The average per-token latency is:

$$
L = \frac{T_{\text{draft}} + T_{\text{target}}}{\tau}
$$

where $T_{\text{draft}}$ is the time spent generating draft tokens, $T_{\text{target}}$ is the cost of verification, and $\tau \in [1, \gamma + 1]$ is the expected number of accepted tokens per cycle, including the bonus token produced by the target model. Let $L_\text{target}$ denote the autoregressive per-token latency of $M_t$. The resulting speedup is:

$$
\eta = \frac{L_\text{target}}{L}
$$

#### Where Does the Speedup Come From?

A compact and intuitive model for target-side verification in speculative decoding is:

$$
T_{\text{target}}(n) \approx \alpha_t(n) + n \cdot \bar{\beta}_t(n)
$$

where $n = B(\gamma + 1)$ denotes the total number of tokens verified in parallel across the serving batch $B$, and $\bar{\beta}_t(n)$ is the effective per-token verification cost achieved at width $n$. In this decomposition, $\alpha_t(n)$ collects per-call overheads that are only weakly dependent on $n$ over a moderate range, including kernel launch and synchronization latency, framework dispatch overhead, attention-metadata construction, paged-attention index setup, and collective start-up in distributed or MoE settings. The term $n \cdot \bar{\beta}_t(n)$ aggregates work that scales approximately linearly with the number of processed tokens, such as QKV and MLP GEMMs, attention computation, KV-cache reads and writes, and elementwise operations.

Crucially, $\bar{\beta}_t(n)$ is not constant. As $n$ increases from an underfilled regime, GEMM shapes become more favorable, occupancy increases, and cache reuse improves. This raises effective arithmetic intensity and lowers the achieved per-token verification cost. The improvement continues until the hardware approaches a roofline limit, after which the benefit flattens and the total verification time becomes closer to linear in $n$.

Under fixed $\gamma$ and fixed expected accepted length $\tau$, the target-side contribution to latency per accepted token is:

$$
\frac{T_{\text{target}}(n)}{\tau} \approx \frac{\alpha_t(n)}{\tau} + \frac{n}{\tau} \bar{\beta}_t(n)
$$

Ignoring the draft cost for the moment, a useful win condition is:

$$
\frac{T_{\text{target}}(n)}{\tau} < L_\text{target}(B)
$$

Let $p = \tau / (\gamma + 1)$ denote the expected acceptance rate per proposed token. A dominant factor for speedup is then:

$$
\frac{\bar{\beta}_t(n)}{p} < \bar{\beta}_t(B)
$$

$\scriptstyle\blacksquare$ **It changes the granularity of the target model's work.** In autoregressive decoding, the target model runs one forward pass per generated token. That produces a long chain of strictly sequential small steps. On GPUs, single-token decode is often memory or latency dominated: each step performs limited math relative to the amount of state it touches, including KV-cache reads and writes, small kernels, synchronization, and launch overhead. As a result, hardware utilization is frequently poor in batch-1 or low-concurrency serving. Even if the model's theoretical FLOPs are high, the achieved throughput per token can be limited by memory bandwidth and launch latency rather than compute.

Speculative decoding changes the shape of the target computation. Instead of paying the per-step overhead $(\gamma + 1)$ times, the target model pays it once per cycle to verify $\gamma$ proposed tokens. In this way, speculative decoding converts many tiny, latency-dominated target steps into fewer, wider verification steps.

$\scriptstyle\blacksquare$ **Verification cost grows sublinearly with $\gamma$.** If verifying $\gamma$ tokens cost exactly $(\gamma + 1) \cdot L_\text{target}$, speculative decoding would never help because it would just add draft overhead to the same amount of target work. The practical speedup comes from $T_\text{target}(\gamma)$ being closer to one slightly wider decode step than to $\gamma$ independent decode steps. Fixed overheads are amortized, and the larger verification width generally improves occupancy, cache reuse, and tensor-core efficiency.

#### Why Speedup Shrinks at Large Serving Batch

In batch-1 or low-concurrency serving, baseline autoregressive decode has poor utilization, so speculative verification still has room to improve efficiency. As serving batch increases, baseline decode already becomes wider and more efficient, so speculative decoding has less sublinearity to exploit. Once the system is close to compute saturation, $T_\text{target}(\gamma)$ becomes closer to linear in $\gamma$, and speculative decoding may provide little benefit or even a negative speedup because it introduces draft compute and extra control flow.

Common failure cases include:

- Low acceptance rate: $\tau$ is small, so amortization fails.
- Slow draft model: $\frac{T_\text{draft}}{\tau}$ consumes the gain.
- Already compute-saturated serving: verification is near-linear, so little headroom remains.
- Large implementation overhead: rejection bookkeeping, synchronization, KV handling, and sampling logic can erase theoretical wins if they are not engineered tightly.

<p class="doc-reference-label"><strong>Reference</strong></p>

<p class="doc-reference">[1] Chen J, Liang Y, Liu Z. DFlash: Block Diffusion for Flash Speculative Decoding. arXiv preprint arXiv:2602.06036, 2026.</p>

### Rejection Sampler

To sample $x \sim p(x)$, we instead sample $x \sim q(x)$, keeping it if $q(x) \leq p(x)$. When $q(x) > p(x)$, the sample is rejected with probability $1 - \frac{p(x)}{q(x)}$, and $x$ is resampled from an adjusted distribution

$$
p^{\prime} = \text{norm}(\max(0, p(x) - q(x)))
$$

**Note:** For any distributions $p(x)$ and $q(x)$, the token distribution obtained by rejection sampling is identical to the distribution obtained by sampling directly from $p(x)$. A proof appears in Appendix A of [1].

Rejection sampling works by decomposing the target distribution $p(x)$ into two branches:

- The accept branch, where tokens are accepted according to the ratio $\frac{p(x)}{q(x)}$, with $q(x)$ as the proposal distribution.

$$
p(\text{accepted}, x = x^{\prime}) =
q(x^{\prime}) \cdot \min\left(\frac{p(x^{\prime})}{q(x^{\prime})}, 1\right)
= \min(p(x^{\prime}), q(x^{\prime}))
$$

- The reject branch, where rejected tokens are resampled from the adjusted distribution $p^{\prime}(x)$.

$$
\begin{aligned}
p(\text{rejected})
&= \sum_{x} q(x) \left(1 - \min\left(\frac{p(x)}{q(x)}, 1\right)\right) \\
&= \sum_{x} \max(0, q(x) - p(x))
\end{aligned}
$$

### Example Code

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
