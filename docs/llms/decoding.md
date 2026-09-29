# Decoding and Sampling

This page covers sampling: how the runtime turns logits into tokens. Speculative decoding is in [Speculative Decoding](speculative-decoding.md).

## Sampling

Next-token probabilities alone do not determine the behavior of an interactive model. The serving system still has to choose whether outputs should be conservative, diverse, stable, or exploratory, and different sampling rules expose different trade-offs among those goals.

### Temperature

Temperature $T$ controls the sharpness of the probability distribution. Given model logits $z_i$, temperature sampling rescales them before softmax:

$$
p_i=\frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

- $T < 1$ makes the distribution sharper and less random; $T > 1$ makes it flatter and more exploratory.
- Temperature preserves the order of logits because dividing by a positive constant does not change the ranking. If `top-k = 1` and decoding is effectively greedy, temperature has no effect.
- As $T \rightarrow 0$, the softmax becomes arbitrarily sharp and converges to greedy sampling.
- As $T \rightarrow \infty$, the distribution approaches uniform.

### Top-k Sampling

After computing probabilities, sort tokens by probability and keep only the $k$ highest. Then renormalize $p_i^{\prime}$ to obtain the final sampling distribution:

$$
\begin{aligned}
S_k & = \text{top-k tokens by } p_i \\
p_i^{\prime} & = \begin{cases}
\frac{p_i}{\sum_{j \in S_k} p_j} & i \in S_k \\
0 & \text{otherwise}
\end{cases}
\end{aligned}
$$

Then sample from the truncated distribution.

- Small $k$ values make decoding more deterministic and conservative.
- Large $k$ values increase diversity but also increase the chance of off-topic tokens.

### Top-p (Nucleus) Sampling

Instead of fixing $k$, top-p chooses the smallest set of tokens whose cumulative probability mass reaches a threshold $p$:

$$
\begin{aligned}
S_p & = \left\{i : \sum_{j \in S_p} p_j \geq p \right\} \\
p_i^{\prime} & = \begin{cases}
\frac{p_i}{\sum_{j \in S_p} p_j} & i \in S_p \\
0 & \text{otherwise}
\end{cases}
\end{aligned}
$$

The key difference from top-k is that top-p adapts to model uncertainty.

- If the model is confident, the retained set is small.
- If the model is uncertain, the retained set expands.

### Min-p

Min-p filters out tokens that are too improbable relative to the top-1 token. Instead of fixing $k$ or $p$, it keeps tokens whose probability stays above a threshold relative to the maximum probability:

$$
S_{\min }=\left\{i: p_i \geq \min -\mathrm{p} \times p_{\max }\right\}
$$

If `min-p = 0.1`, any token whose probability is at least 10% of the most likely token is retained. The probabilities are then renormalized:

$$
p_i^{\prime} = \frac{p_i}{\sum_{j \in S_{\min}} p_j}
$$

- Like top-p, min-p adapts to context.
- It can preserve plausible lower-ranked tokens when the distribution is relatively flat.
- It is often numerically convenient for long-tail vocabularies.

### Putting Them Together

The usual composition order is:

`Temperature -> Softmax -> (top-k / top-p / min-p) -> Renormalize -> Sample`

When both top-k and top-p are enabled, the surviving candidate set is their intersection.

### Example Code

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
    logits_or_prob.masked_fill_(
        logits_or_prob < top_k_mask,
        -float("inf") if is_logits else float(0),
    )
    return logits_or_prob
```
</details>
