# 解码与采样

本页讨论采样：运行时如何把 logits 变成输出 token。Speculative decoding 见 [Speculative Decoding](speculative-decoding.md)。

## 采样

采样之所以重要，是因为仅有 next-token 概率还不足以决定交互式模型的行为。系统仍然必须决定输出是更保守还是更多样，是更稳定还是更探索，不同采样规则正是在这些目标之间暴露不同权衡。

### Temperature

温度 $T$ 控制概率分布的尖锐度。给定模型 logits $z_i$，温度采样会在 softmax 前对其重缩放：

$$
p_i=\frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

- $T < 1$ 时分布更尖锐，随机性更低；$T > 1$ 时分布更平坦，探索性更高。
- 温度不会改变 logits 的排序，因为除以正数不会改变相对顺序。如果 `top-k = 1`，也就是 greedy 采样，温度基本不起作用。
- 当 $T \rightarrow 0$ 时，softmax 分布会收敛到 greedy。
- 当 $T \rightarrow \infty$ 时，分布趋于均匀。

### Top-k Sampling

在计算概率之后，按概率排序，仅保留概率最高的 $k$ 个 token，然后对 $p_i^{\prime}$ 重新归一化：

$$
\begin{aligned}
S_k & = \text{top-k tokens by } p_i \\
p_i^{\prime} & = \begin{cases}
\frac{p_i}{\sum_{j \in S_k} p_j} & i \in S_k \\
0 & \text{otherwise}
\end{cases}
\end{aligned}
$$

然后从该截断分布中采样。

- 较小的 $k$ 会让输出更确定、更保守。
- 较大的 $k$ 会提升多样性，但也会增加偏题 token 的概率。

### Top-p（Nucleus）Sampling

与固定 $k$ 不同，top-p 选择累计概率质量达到阈值 $p$ 的最小 token 集合：

$$
\begin{aligned}
S_p & = \left\{i : \sum_{j \in S_p} p_j \geq p \right\} \\
p_i^{\prime} & = \begin{cases}
\frac{p_i}{\sum_{j \in S_p} p_j} & i \in S_p \\
0 & \text{otherwise}
\end{cases}
\end{aligned}
$$

它与 top-k 的关键区别在于会根据模型不确定性自适应调整候选集大小。

- 如果模型很确定，保留的 token 很少。
- 如果模型不确定，候选集会自动扩大。

### Min-p

Min-p 会过滤掉相对 top-1 来说过于不可能的 token。它不固定 $k$ 或 $p$，而是保留那些概率至少达到 top-1 token 一定比例的 token：

$$
S_{\min }=\left\{i: p_i \geq \min -\mathrm{p} \times p_{\max }\right\}
$$

如果 `min-p = 0.1`，则任何概率至少达到最可能 token 的 10% 的 token 都会被保留。随后归一化：

$$
p_i^{\prime} = \frac{p_i}{\sum_{j \in S_{\min}} p_j}
$$

- 和 top-p 一样，min-p 会随上下文自适应。
- 在分布较平坦时，它更容易保留一些虽然排名较低但仍合理的 token。
- 对长尾词表通常也更数值稳定。

### 组合方式

常见的组合顺序是：

`Temperature -> Softmax -> (top-k / top-p / min-p) -> Renormalize -> Sample`

当 top-k 和 top-p 同时启用时，最终保留的是二者交集。

### 示例代码

代码片段来自 [Omniinfer Sampler](https://gitee.com/omniai/omniinfer/blob/master/omni/adaptors/vllm/sample/sampler.py)。

<details>
<summary>Top-k 和 Top-p 实现</summary>

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
