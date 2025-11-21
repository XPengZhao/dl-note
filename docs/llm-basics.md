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