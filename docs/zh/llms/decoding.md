# 解码与采样

本页关注 decode 阶段的两类问题：一类是采样策略如何塑造输出分布，另一类是 speculative decoding 这类 decode 侧加速技术如何改变生成时的系统行为。

把这两个主题放在一起的动机是，decode 同时有模型侧和系统侧两层含义。一方面，运行时必须决定如何把 logits 变成输出 token，这会直接影响质量、多样性和确定性；另一方面，decode 又是 serving 中最核心的稳态循环，因此 decode 侧优化也会直接改变吞吐和延迟。

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

## Speculative Decoding

Speculative decoding 的动机来自一个经典 serving 瓶颈：自回归 decode 每次只执行一个很小、严格串行的步骤，而这些步骤往往无法充分利用现代加速器。它的核心思路是把许多窄小的目标模型步骤，替换成更少但更宽的验证步骤，从而摊薄运行时开销。

### Speculative Decoding 加速

Speculative decoding 通过更小的草稿模型 $M_d$ 来加速目标模型 $M_t$ 的推理。每个解码周期中，草稿模型先提出 $\gamma$ 个 token，再由目标模型并行验证。平均每 token 延迟为：

$$
L = \frac{T_{\text{draft}} + T_{\text{target}}}{\tau}
$$

其中，$T_{\text{draft}}$ 是生成草稿 token 的时间，$T_{\text{target}}$ 是验证开销，$\tau \in [1, \gamma + 1]$ 是每周期期望接受的 token 数，包含目标模型产生的 bonus token。令 $L_\text{target}$ 表示 $M_t$ 自回归逐 token 延迟，则速度提升为：

$$
\eta = \frac{L_\text{target}}{L}
$$

#### 加速来自哪里

对于 speculative decoding 的目标侧验证，可用一个紧凑且直观的模型表示：

$$
T_{\text{target}}(n) \approx \alpha_t(n) + n \cdot \bar{\beta}_t(n)
$$

其中 $n = B(\gamma + 1)$ 表示在服务 batch $B$ 中并行验证的 token 总数，$\bar{\beta}_t(n)$ 是宽度为 $n$ 时的有效逐 token 验证成本。在这个分解里，$\alpha_t(n)$ 聚合了对 $n$ 弱依赖的调用级开销，包括 kernel 启动与同步延迟、框架分发开销、attention metadata 构建、paged-attention 索引准备，以及分布式或 MoE 场景下的 collective 启动。项 $n \cdot \bar{\beta}_t(n)$ 则聚合了与处理 token 数近似成比例增长的工作，例如 QKV/MLP GEMM、attention 计算、KV-cache 读写与逐元素操作。

关键是，$\bar{\beta}_t(n)$ 并不是常数。当 $n$ 从低填充区间增大时，GEMM 形状更友好、occupancy 更高、cache 复用更好，因此有效算术强度上升，逐 token 验证成本下降。这种下降会持续到系统接近硬件上限，之后收益趋于平坦，总时间也会更接近随 $n$ 线性增长。

在固定 $\gamma$ 和固定期望接受长度 $\tau$ 下，目标侧对每个已接受 token 延迟的贡献为：

$$
\frac{T_{\text{target}}(n)}{\tau} \approx \frac{\alpha_t(n)}{\tau} + \frac{n}{\tau} \bar{\beta}_t(n)
$$

暂时忽略 draft 成本时，一个有用的获益条件是：

$$
\frac{T_{\text{target}}(n)}{\tau} < L_\text{target}(B)
$$

令 $p = \tau / (\gamma + 1)$ 表示每个提议 token 的期望接受率，则影响加速效果的主导因子之一是：

$$
\frac{\bar{\beta}_t(n)}{p} < \bar{\beta}_t(B)
$$

$\scriptstyle\blacksquare$ **它改变了目标模型工作的粒度。** 在自回归解码中，目标模型每生成一个 token 都执行一次前向，形成一长串严格串行的小步。对 GPU 而言，单 token decode 往往是内存或时延主导的：每步相对于访问的状态量来说计算量有限，包括 KV cache 读写、小 kernel、同步和启动开销。因此在 batch-1 或低并发服务中，硬件利用率通常较差。

Speculative decoding 改变了目标计算形态。目标模型不再把每步开销支付 $(\gamma + 1)$ 次，而是每周期支付一次来验证 $\gamma$ 个提议 token。换句话说，它把大量细碎、时延主导的目标步，转换成更少但更宽的验证步。

$\scriptstyle\blacksquare$ **验证成本随 $\gamma$ 的增长是次线性的。** 如果验证 $\gamma$ 个 token 的成本精确等于 $(\gamma + 1) \cdot L_\text{target}$，speculative decoding 就不会有帮助，因为这只是做了同样的目标工作再加上 draft 开销。实际中的速度提升来自 $T_\text{target}(\gamma)$ 更接近一次稍宽的 decode，而不是 $\gamma$ 次独立 decode。固定开销被摊销，更宽的验证也通常会改善 occupancy、cache 复用和 tensor-core 效率。

#### 为什么在大服务 Batch 下收益会变小

在 batch-1 或低并发服务下，基线自回归 decode 的利用率较差，speculative verification 仍有较大优化空间。随着服务 batch 增大，基线 decode 本身已经更宽、更高效，speculative decoding 可利用的次线性空间会减小。当系统接近算力饱和时，$T_\text{target}(\gamma)$ 会更接近对 $\gamma$ 线性增长，此时 speculative decoding 收益很小，甚至可能为负，因为它引入了 draft 计算和额外控制流。

常见失败场景包括：

- 接受率低：$\tau$ 太小，摊销失败。
- draft 太慢：$\frac{T_\text{draft}}{\tau}$ 吃掉收益。
- 服务已接近算力饱和：验证过程近似线性，可优化空间很小。
- 实现开销大：拒绝记账、同步、KV 处理和采样逻辑会抹去理论收益。

<p class="doc-reference-label"><strong>Reference</strong></p>

<p class="doc-reference">[1] Chen J, Liang Y, Liu Z. DFlash: Block Diffusion for Flash Speculative Decoding. arXiv preprint arXiv:2602.06036, 2026.</p>

### Rejection Sampler

若目标是采样 $x \sim p(x)$，可以先从 $q(x)$ 采样 $x$。当 $q(x) \leq p(x)$ 时保留；当 $q(x) > p(x)$ 时，以概率 $1 - \frac{p(x)}{q(x)}$ 拒绝该样本，并改为从调整分布

$$
p^{\prime} = \text{norm}(\max(0, p(x) - q(x)))
$$

中重新采样。

**Note：** 对任意分布 $p(x)$ 与 $q(x)$，通过 rejection sampling 得到的 token 分布，与直接从 $p(x)$ 采样得到的分布完全一致。证明见文献 [1] 的附录 A。

Rejection sampling 的思路是将目标分布 $p(x)$ 分解为两个分支：

- 接受分支：基于比率 $\frac{p(x)}{q(x)}$ 接受 token，其中 $q(x)$ 是提议分布。

$$
p(\text{accepted}, x = x^{\prime}) =
q(x^{\prime}) \cdot \min\left(\frac{p(x^{\prime})}{q(x^{\prime})}, 1\right)
= \min(p(x^{\prime}), q(x^{\prime}))
$$

- 拒绝分支：被拒绝的 token 从调整分布 $p^{\prime}(x)$ 中重采样。

$$
\begin{aligned}
p(\text{rejected})
&= \sum_{x} q(x) \left(1 - \min\left(\frac{p(x)}{q(x)}, 1\right)\right) \\
&= \sum_{x} \max(0, q(x) - p(x))
\end{aligned}
$$

### 示例代码

代码片段来自 [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py)。

<details>
<summary>Triton 中的拒绝采样 kernel</summary>

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
<summary>Triton 中的重采样 kernel</summary>

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
