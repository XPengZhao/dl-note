# AI Infra 预备知识

## 指标

### 延迟（Latency）

延迟是指处理请求并生成输出 token 所需的时间。在 LLM 推理中，通常关注两个核心延迟指标：

#### 首 token 时间（TTFT）

TTFT（Time to First Token）表示从推理服务接收到请求，到第一个输出 token 被生成并返回给客户端的时间间隔。它衡量模型“开始响应”有多快，是流式 LLM 系统中最直接的用户感知延迟指标。形式化地：

$$
\text{TTFT} = t_{\text{first token emitted}} - t_{\text{request received}}
$$

TTFT 主要覆盖 prefill 阶段（处理输入上下文）以及调度与系统开销，不包含后续 token 的生成时间。通常可分解为：

$$
\small
\begin{aligned}
\text{TTFT} = t_{\text{queue}} + t_{\text{input processing}} + t_{\text{prefill compute}} + t_{\text{first decode step (optional)}} + t_{\text{framework overhead}}
\end{aligned}
$$

其中：

- $t_{\text{queue}}$：请求进入执行前在调度器中的排队时间。
- $t_{\text{input processing}}$：输入处理开销，包括分词、多模态预处理（如图像/视频编码）和请求规范化。
- $t_{\text{prefill compute}}$：对全部输入 token 做前向计算的时间（序列长度 $L$ 下复杂度约为 $O(L)$），同时建立 KV cache，并得到提示词末位置的 logits。
- $t_{\text{first decode step (optional)}}$：某些实现将 prefill 和 decode 调度分离时，首 token 可能要在下一次 decode step 才发出；优化实现中此项可与 prefill 融合，开销很小。
- $t_{\text{framework overhead}}$：框架额外开销，如 CUDA kernel 启动、同步、内存分配、采样（softmax/top-k/top-p）以及首 token 网络传输。

在长上下文场景下，TTFT 往往由 $t_{\text{prefill compute}}$ 主导，其余项通常是次要开销。

#### 每输出 token 时间（TPOT）

TPOT（Time per Output Token）衡量首 token 发出后，平均每个后续 token 的生成延迟。它表征自回归解码的稳态性能，并直接决定流式输出速度。对生成 $N$ 个输出 token 的请求：

$$
\text{TPOT} = \frac{t_\text{last token emitted} - t_\text{first token emitted}}{N - 1}
$$

等价地，若总端到端时延为 $T_{\text{total}}$：

$$
\text{TPOT} = \frac{T_{\text{total}} - \text{TTFT}}{N - 1}
$$

在稳态 decode 中，每个 token 的时间可写为：

$$
\text{TPOT} = t_{\text{decode compute}} + t_{\text{communication}} + t_{\text{scheduling}} + t_{\text{framework overhead}}
$$

其中：

- $t_{\text{decode compute}}$：利用 KV cache 的单 token 前向计算。
- $t_{\text{communication}}$：跨卡通信与数据交换（如 TP all-reduce/all-gather，MoE all-to-all，以及 H2D/D2H 传输）。
- $t_{\text{scheduling}}$：运行时调度开销（连续 batching、请求协调、准入控制、KV 分配/分页/管理）。
- $t_{\text{framework overhead}}$：采样、kernel 启动、同步、I/O 封送、后处理和 token streaming 等辅助开销。

与 TTFT 不同，TPOT 不需重算完整 prompt；但由于 decode 仍要对全部缓存上下文做注意力计算，其成本仍受有效上下文长度影响，通常随总上下文（提示词 + 已生成）近似线性增长。

### 吞吐（Throughput）

#### 每秒 token 数（TPS）

TPS（Tokens per Second）衡量系统在给定服务配置下每秒处理/生成 token 的能力。与 TTFT、TPOT 这种请求级指标不同，TPS 是系统级指标，反映整体承载能力。在时间窗口 $\Delta t$ 内：

$$
\text{TPS} = \frac{N_\text{tokens}}{\Delta t}
$$

很多场景下 TPS 特指 decode 吞吐（只统计输出 token）：

$$
\text{TPS}_\text{decode} = \frac{N_\text{output tokens}}{\Delta t}
$$

对单序列顺序 decode，通常有：

$$
\text{TPS}_\text{per request} \approx \frac{1}{\text{TPOT}}
$$

在连续 batching 并发场景中，系统吞吐会随有效 batch 大小 $B$ 增长。单副本近似为：

$$
\text{TPS}_\text{replica} \approx \frac{B}{\text{TPOT}}
$$

若有 $R$ 个数据并行（DP）副本：

$$
\text{TPS}_\text{total} \approx R \times \frac{B}{\text{TPOT}}
$$

对于长上下文或多模态任务，prompt 处理可能占据可观算力，此时应关注端到端吞吐：

$$
\text{TPS}_\text{end-to-end} = \frac{N_\text{prompt tokens} + N_\text{output tokens}}{\Delta t}
$$

总体上，TPS 受模型规模、并行策略、有效 batch、大核效率、互联带宽以及 prefill/decode 负载比例共同影响。

## KV Cache

### 显存开销

KV cache 大小计算公式：

$$
\text{KV size} = 2 \times \text{num layers} \times \text{num heads} \times \text{seq len} \times \text{head dim}
$$

假设：

- 32 层、80 个头、head_dim = 128、seq_len = 4K。
- FP16 KV cache：2 × 32 × 80 × 4K × 128 × 2 bytes = **约 5.2 GB/样本**。
- C8（int8 KV cache）：约为一半，即 **约 2.6 GB/样本**。

## 注意力

在 Transformer 中，注意力模块计算：

$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

每个 token 会产生：

- **Q（Query）**：来自当前 token 的隐状态。
- **K（Key）**、**V（Value）**：来自上下文（历史 token），并存入 **KV cache**。

**MHA、MQA、GQA** 的主要区别是：相对于 Query 头数，Key/Value 投影是如何共享的。

### **MHA — 多头注意力**

每个注意力头都有独立投影：

$$
Q_i=X W_{Q, i}, \quad K_i=X W_{K, i}, \quad V_i=X W_{V, i}
$$

- Q 头数 = K 头数 = V 头数。

### **MQA — 多查询注意力**

所有头共享同一组 K、V 投影：

$$
K_{\text {shared }}=X W_K, \quad V_{\text {shared }}=X W_V
$$

- Q 头可以很多，但只有一组共享 K/V。
- 推理时各头复用同一 K/V。
- KV cache 显存约可按头数倍数下降。
- **LLaMA 2 7B**、**Mistral** 等模型常用 MQA 以降低内存开销。

### **GQA — 分组查询注意力**

将 Query 头分组，每组共享一组 K/V：

$$
\text { Group } g: \quad Q_{g, *}, K_g, V_g
$$

- 例如 8 个 Q 头、2 组时，每组 4 个 Q 头共享一组 K/V。
- **K/V 头数 < Q 头数**。
- **LLaMA 2 70B**、**Mixtral**、**GPT-J-6B** 等常用 GQA，在效果与效率间折中。

## 采样

### 温度（Temperature）

温度 $T$ 控制概率分布的“尖锐程度”。给定 logits $z_i$，温度采样先缩放再 softmax：

$$
p_i=\frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

- $T<1$：分布更尖锐，输出更确定、随机性更低。
- $T>1$：分布更平坦，探索性更强、多样性更高。
- 温度是保序变换（不会改变 logits 排序）。若 top-k = 1（贪心），温度不会产生实际影响。
- 当 $T \rightarrow 0$ 时，分布趋近 argmax（贪心）。
- 当 $T \rightarrow \infty$ 时，分布趋近均匀。

### Top-k Sampling

在温度处理后按概率排序，仅保留前 k 个 token，并重归一化得到采样分布：

$$
\begin{aligned}
S_k & =\text { top-k tokens by } p_i \\
p_i^{\prime} & = \begin{cases}\frac{p_i}{\sum_{j \in S_k} p_j} & i \in S_k \\
0 & \text { otherwise }\end{cases}
\end{aligned}
$$

再从截断后的分布中采样。

- 小 k（如 1–10）：更稳定、更保守。
- 大 k（如 50–100）：更有多样性，但偏题风险更高。

### Top-p（Nucleus）Sampling

“nucleus”指概率质量的核心集合。不同于固定 k，top-p 选择“累计概率达到阈值 $p$ 的最小 token 集合”。

$$
\begin{aligned}
S_p&=\left\{i: \sum_{j \in S_p} p_j \geq p\right\} \\
p_i^{\prime}&= \begin{cases}\frac{p_i}{\sum_{j \in S_p} p_j} & i \in S_p \\
0 & \text { otherwise }\end{cases}
\end{aligned}
$$

**与 top-k 的关键区别：**

top-k 是按排名截断，top-p 是按概率质量截断，因此能自适应模型不确定性。

- 当模型很确定（头部 token 占比高）时，保留 token 很少。
- 当模型不确定（概率较分散）时，集合会自动扩大。

### Min-p

Min-p 按相对 top-1 的最小概率阈值过滤 token（近年较常见，如 Mistral、Gemini 等）。其规则不是固定 $k$ 或累计概率 $p$，而是保留所有满足：

$$
S_{\min }=\left\{i: p_i \geq \min -\mathrm{p} \times p_{\max }\right\}
$$

若 min-p = 0.1，则保留概率至少为 top-1 概率 10% 的 token。之后再归一化：

$$
p_i^{\prime} = \frac{p_i}{\sum_{j \in S_{\min}} p_j}
$$

- 与 top-p 一样可自适应上下文，但规则更直接。
- 在分布较平坦时，不容易错误截断低排名但合理的 token。
- 对长尾词表通常更稳定。

### 组合使用顺序

实际中常按以下顺序组合：

Temperature → Softmax → (top-k / top-p / min-p) → Renormalize → Sample

当 top-k 与 top-p 同时开启时，最终候选集是两者交集（更小的集合）。

### 示例代码

代码来自 [Omniinfer Sampler](https://gitee.com/omniai/omniinfer/blob/master/omni/adaptors/vllm/sample/sampler.py)。

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
    logits_or_prob.masked_fill_(logits_or_prob < top_k_mask, -float("inf") if is_logits else float(0))
    return logits_or_prob
```
</details>

## Speculative Decoding

### Speculative Decoding 的加速机理

Speculative decoding 使用较小的草稿模型 $M_d$ 来加速目标模型 $M_t$ 的推理。每个解码周期中，草稿模型先提出 $\gamma$ 个 token，再由目标模型并行验证。平均每 token 延迟可写为：

$$
L = \frac{T_{\text{draft}} + T_{\text{target}}}{\tau}.
$$

其中 $T_{\text{draft}}$ 是草稿生成开销，$T_{\text{target}}$ 是验证开销，$\tau \in [1, \gamma + 1]$ 是每周期期望接受长度（含 bonus token）。设 $L_\text{target}$ 为目标模型自回归单 token 延迟，则速度提升为：

$$
\eta = \frac{L_\text{target}}{L}.
$$

#### 加速来自哪里？

对目标侧验证时间可使用一个简洁模型：

$$
T_{\text {target }}(n) \approx \alpha_t(n)+n \cdot \bar{\beta}_t(n),
$$

其中 $n= B(\gamma + 1)$ 为 batch $B$ 下并行验证 token 总数，$\bar{\beta}_t(n)$ 为在宽度 $n$ 下的等效单 token 验证成本。$\alpha_t(n)$ 聚合了弱依赖 $n$ 的调用级固定开销（kernel 启动、同步、框架调度、attention metadata/paged-attention 索引构建、分布式/MoE 的 collective 启动等）。$n \cdot \bar{\beta}_t(n)$ 对应近似线性增长的工作量（QKV/MLP GEMM、attention、KV 读写、逐元素算子等）。

关键点是：$\bar{\beta}_t(n)$ 并非常数。随着 $n$ 从低利用区增大，GEMM 形状更友好、占用率和 cache 复用提升，通常会降低单位 token 成本；直到接近硬件上限后趋于平坦。

在固定 $\gamma$ 与固定期望接受长度 $\tau$ 下，目标侧对“每个已接受 token 延迟”的贡献近似：

$$
\frac{T_{\text {target }}(n)}{\tau} \approx \frac{\alpha_t(n)}{\tau} + \frac{n}{\tau} \bar{\beta}_t(n).
$$

可获得加速的直观条件（先忽略 draft 成本）是：

$$
\frac{T_{\text {target }}(n)}{\tau} < L_\text{target}(B).
$$

令 $p = \tau / (\gamma + 1)$ 表示提议 token 的期望接受率，一个关键因子是：

$$
\frac{\bar{\beta}_t(n)}{p}  < \bar{\beta}_t(B).
$$

$\scriptstyle\blacksquare$ **它改变了目标模型工作的粒度。** 自回归解码每次只处理 1 个 token，形成大量串行小步，常受内存访问/启动时延限制。Speculative decoding 将多个小步验证合并为更少但更宽的验证步，从而提升利用率。

$\scriptstyle\blacksquare$ **验证开销随 $\gamma$ 的增长常呈次线性。** 如果验证 $\gamma$ 个 token 的成本严格等于 $(\gamma+1)\cdot L_\text{target}$，则不会加速。现实中由于固定开销摊销、算术强度提升、占用率与 cache 命中改善，验证成本通常低于线性外推。

#### 为什么大 batch 下加速会变小，甚至为负？

在 batch=1 或低并发时，基线自回归 decode 利用率低，speculative 仍有较大优化空间。随着 batch 增大，基线本身变“更宽更高效”，speculative 能额外榨取的次线性收益减少。若系统已接近计算饱和，验证时间会更接近线性，同时又增加了 draft 与控制流开销，可能导致负收益。常见失败场景：

- 接受率低：$\tau$ 小，摊销不足。
- draft 太慢：$\frac{T_\text{draft}}{\tau}$ 吃掉收益。
- 服务已算力饱和：验证近线性，剩余空间很小。
- 实现开销过高：拒绝采样记账、同步、KV 管理、采样逻辑等吞噬理论收益。

<p class="doc-reference-label"><strong>参考文献</strong></p>

<p class="doc-reference">[1] Chen J, Liang Y, Liu Z. DFlash: Block Diffusion for Flash Speculative Decoding. arXiv preprint arXiv:2602.06036, 2026.</p>

### Rejection Sampler

若目标是采样 $x \sim p(x)$，可先从提议分布 $q(x)$ 采样：当 $q(x) \leq p(x)$ 时接受；当 $q(x) > p(x)$ 时，以概率 $1 - \frac{p(x)}{q(x)}$ 拒绝，并改为从调整分布 $p^{\prime} = \text{norm}(\max(0, p(x) - q(x)))$ 重新采样。

**说明：** 对任意分布 $p(x)$ 与 $q(x)$，基于拒绝采样流程得到的 token 分布与直接从 $p(x)$ 采样一致（证明见附录 A [1]）。

拒绝采样可看成将目标分布 $p(x)$ 拆成两部分：

- 接受分支：按 $\frac{p(x)}{q(x)}$ 比率接受（$q(x)$ 为草稿分布）。

$$
  p(\text{accepted}, x = x^{\prime}) = q(x^{\prime}) \cdot \min\left(\frac{p(x^{\prime})}{q(x^{\prime})}, 1\right) = \min(p(x^{\prime}), q(x^{\prime}))
$$

- 拒绝分支：被拒绝后从调整分布 $p^{\prime}(x)$ 重采样。所有从 $q(x)$ 采样 token 被拒绝的概率：

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

**示例代码**

代码来自 [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py)。

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

**参考文献**

[[1]](https://arxiv.org/pdf/2211.17192) Y Leviathan, M Kalman, Y Matias, "Fast Inference from Transformers via Speculative Decoding", ICML 2023.

## LLM 的预训练目标

从零训练的大语言模型通常通过最大似然估计（MLE）在大规模语料上进行优化。

给定从数据分布 $p_\text{data}$ 采样得到的 token 序列 $(t_1, t_2, \ldots, t_n)$，模型定义如下自回归分解：

$$
p_\theta(t_1, t_2, \ldots, t_n) = \prod_{i=1}^n p_\theta(t_i | t_1, t_2, \ldots, t_{i-1})
$$

训练目标是最大化数据对数似然，等价于最小化负对数似然（NLL）：

$$
\mathcal{L}(\theta) = - \mathbb{E}_{(t_1, t_2, \ldots, t_i) \sim p_\text{data}} \left[ \log p_\theta(t_i | t_1, t_2, \ldots, t_{i-1}) \right]
$$

在实践中，通常实现为 token 级 one-hot 交叉熵：

$$
\mathcal{L} = -\frac{1}{N} \sum_{\text{tokens}} \log p_\theta(t_i | t_1, t_2, \ldots, t_{i-1})
$$

该目标等价于最小化经验数据分布与模型分布之间的交叉熵；也可视作最小化 $KL(p_\text{data} || p_\theta)$。
