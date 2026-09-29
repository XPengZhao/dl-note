# Speculative Decoding

## 1. Motivation

![Draft 逐个写出 token，target 一次 forward 验证到第一处拒绝](../../assets/images/speculative-decoding/sd-architecture.svg){ width="80%" }

大模型生成文本时，token 是一个接一个产生的。比如模型已经生成到 **once**，接下来要得到 **upon → a → time → there**。由于后一个 token 依赖前一个 token，target model 必须连续执行四次 decode，才能生成这四个 token。问题在于，decode 的每一步其实只生成一个 token。尤其在 batch 较小时，一次 forward 的计算量并不大，但仍然需要读取已有的 KV cache、启动 GPU kernel，并完成同步。因此，生成四个 token 的代价基本就是四次 target forward的时间. Speculative decoding 的想法很直接：既然 target 每次只生成一个 token 很贵，能不能先让一个更小、更快的模型猜几个 token，再让 target 一次检查完？

图中，draft model 先连续生成 **upon → a → time → there**。随后 target 不再逐个生成这些 token，而是在一次 forward 中同时计算这几个位置的预测结果，并从左到右进行验证。在这个例子里，**upon、a、time** 都被接受，到 **there** 时第一次出现拒绝，因此验证在这里停止。这样，原本需要 target 连续执行四次的工作，被压缩成了一次 target forward，再加上若干次成本更低的 draft forward。这也是 speculative decoding 的核心：**用便宜的 draft 生成候选，再利用 target 对多个候选 token 进行并行验证，以减少昂贵的 target decode 次数。** Draft 猜得越准，一次 target forward 能确认的 token 越多，加速效果也就越明显。


## 2. Vanilla Speculative Decoding

### 2.1 Draft

Speculative decoding 的第一步，是让一个更小的 draft model 先向前生成若干个候选 token。继续使用前面的例子，当前 prefix 以 **once** 结尾。如果 draft length 为 \(K=4\)，draft 会连续生成$\text{upon}\rightarrow\text{a}\rightarrow\text{time}\rightarrow\text{there}.$ 形式化地，令当前 prefix 为 \(y\)，draft model 的条件分布为 \(p_d\)。第 \(i\) 个候选 token 按照

$$
x_i\sim p_d(\cdot\mid y,x_{<i})
$$

依次生成，其中 \(x_{<i}=(x_1,\ldots,x_{i-1})\)。因此，draft 阶段本身仍然是 autoregressive 的：生成 \(K\) 个候选 token，依然需要 $K$ 次 draft forward。 这看起来似乎没有减少 forward 次数，关键在于 draft 通常远小于 target。Speculative decoding 实际上是在用 $K$ 次便宜的 draft forward 去换取后面若干次昂贵的 target forward。Draft 生成的 token 也只是候选，真正决定它们能否进入最终输出的仍然是 target。

因此，一个好的 draft 并不需要完全复现 target，只需要同时做到两件事：**生成得足够快，并且经常猜中 target 接下来会接受的 token。** Draft 太大，候选更准但自身开销更高；draft 太小，虽然便宜，却可能频繁被拒绝。这个 trade-off 会直接决定 speculative decoding 最终能获得多少加速。

### 2.2 Verification

Draft 得到 \(K\) 个候选 token 后，target 不需要像正常 decoding 那样再逐个生成。因为整段 candidate sequence 已经给定，target 可以把它们一次性送入模型，在一次 forward 中同时得到每个位置的条件分布

$$
p_t(\cdot\mid y),\quad
p_t(\cdot\mid y,x_1),\quad \ldots,\quad
p_t(\cdot\mid y,x_{<K}).
$$

虽然这些位置在语义上仍然存在先后依赖，但它们的输入 token 已经由 draft 提供，因此 target 可以像 prefill 一样并行计算。回到前面的例子，draft 给出 **upon → a → time → there**，target 只需一次 forward，就能得到这四个位置对应的预测分布。

接下来从左到右验证候选 token。对于第 \(i\) 个 draft token \(x_i\)，其接受概率为

$$
\alpha_i=
\min\left(
1,
\frac{p_t(x_i\mid y,x_{<i})}
     {p_d(x_i\mid y,x_{<i})}
\right).
$$

如果 \(x_i\) 被接受，就继续检查下一个；一旦某个位置被拒绝，它右边的候选全部失效，因为这些 token 都是在被拒绝的 token 条件下生成的。于是图中的 **upon、a、time** 可以连续通过，而 **there** 被拒绝，当前 speculative step 也就在这里结束。

Verification 的收益正来自这里：**target 用一次较宽的 forward，同时检查多个 draft token，再把可以接受的部分一次写入输出。** 如果一次 verification 平均能接受多个 token，就相当于用一次 target forward 替代了多次普通 decode。


### 2.3 Rejection and Resampling

如果第 \(i\) 个候选 token \(x_i\) 被拒绝，那么从 \(x_{i+1}\) 开始的候选都会被丢弃，因为它们是在 \(x_i\) 已经出现的条件下生成的。此时需要由 target 在第 \(i\) 个位置重新生成一个 token，然后结束这一轮 speculative decoding。

这里不能简单地重新从 \(p_t\) 采样。原因是：我们已经知道 \(x_i\) 没有通过前面的 acceptance test，这个 rejection 本身已经改变了当前位置的条件分布。如果忽略这一信息再次直接采样 \(p_t\)，某些 token 会被重复赋予概率，最终生成分布就不再严格等价于 target model。

正确的做法是从 target 与 draft 的**剩余概率质量**中采样：

$$
p_{\mathrm{res}}(x)
=
\frac{
\left[p_t(x)-p_d(x)\right]_+
}{
\sum_{x'}
\left[p_t(x')-p_d(x')\right]_+
},
\qquad
[a]_+=\max(a,0).
$$

直观来看，draft 已经通过候选过程“用掉”了一部分概率质量；当候选被拒绝时，只需要由 target 补上 draft 没有覆盖到的部分。正是 acceptance rule 和 residual resampling 的配合，使 speculative decoding 虽然先让 draft 猜测多个 token，最终得到的采样分布仍然与直接从 target autoregressive decoding 完全一致。


## 3. Why Is Speculative Decoding Exact?

Speculative decoding 改变了 token 的生成过程，但不会改变最终的采样分布。关键在于：draft 负责提出 candidate，acceptance rule 保留 target 与 draft 重叠的概率质量，而 rejection 后的 residual sampling 再补上 target 剩余的部分。

先看一个候选 token \(x\)。Draft 以 \(p_d(x)\) 的概率提出它，并以

$$
\alpha(x)=
\min\left(
1,\frac{p_t(x)}{p_d(x)}
\right)
$$

的概率接受。因此，\(x\) 通过 draft proposal 并最终被接受的概率为

$$
p_d(x)\alpha(x)
=
\min\left(p_d(x),p_t(x)\right).
$$

也就是说，acceptance 恰好保留了 \(p_d\) 和 \(p_t\) 之间重叠的概率质量。对于 target 比 draft 更偏好的 token，这部分还不够，缺少的概率正是

$$
\left[p_t(x)-p_d(x)\right]_+.
$$

当 candidate 被拒绝后，residual distribution 正是按照这部分剩余概率重新采样。因此，对任意 token \(x\)，最终得到它的总概率为

$$
\min\left(p_d(x),p_t(x)\right)
+
\left[p_t(x)-p_d(x)\right]_+
=
p_t(x).
$$

所以，无论一个 token 是由 draft 提出后被接受，还是在 rejection 后通过 residual sampling 得到，最终分布都严格等于 target distribution。这个结论在每一个 decoding position 上都成立，因此 speculative decoding 在加速生成的同时，仍然保持与原始 target autoregressive sampling 完全相同的输出分布。

### 4.1 Acceptance Length

Speculative decoding 的加速首先取决于：一次 target verification 能让生成序列向前推进多少个 token。假设 draft length 为 \(K\)，并且前 \(m\) 个 draft token 被连续接受，那么这一轮至少可以确认这 \(m\) 个 token；在标准 speculative decoding 中，target 还会在 verification 的末尾额外产生一个 token，因此这一轮通常能够推进

$$
m+1
$$

个 token。我们将一次 speculative step 实际推进的 token 数称为 **acceptance length**，其平均值记为

$$
A=\mathbb{E}[\text{tokens advanced per speculative step}].
$$

以 \(K=4\) 为例，如果 draft 给出 **upon → a → time → there**，其中前三个 token 被接受、第四个被拒绝，那么 target 会在第四个位置重新采样一个 token。这一轮最终向前推进 4 个 token，因此 acceptance length 为 4。相反，如果第一个 draft token 就被拒绝，那么这一轮只能推进 1 个 token，此时 speculative decoding 几乎没有利用到并行 verification 的优势。

这里需要区分 **acceptance rate** 和 **acceptance length**。Acceptance rate 描述单个 draft token 被接受的比例，而 acceptance length 直接描述一次昂贵的 target verification 能换来多少个输出 token。对于实际推理性能，后者通常更加重要：**一次 verification 推进得越远，需要执行的 target forward 就越少。** 下一节可以据此建立一个简单的 speedup model，分析 acceptance length、draft cost 和 verification cost 如何共同决定最终加速比。

### 4.2 A Simple Speedup Model

有了 acceptance length，就可以粗略估算 speculative decoding 为什么会更快。假设普通 autoregressive decoding 的一次 target decode 耗时为 \(T_{\mathrm{target}}\)。如果生成 \(A\) 个 token，需要执行 \(A\) 次 target forward，因此时间约为

$$
T_{\mathrm{AR}}
=
A\,T_{\mathrm{target}}.
$$

对于 speculative decoding，假设 draft length 为 \(K\)。一轮 speculative step 需要先执行 \(K\) 次 draft forward，再执行一次 target verification，因此其耗时可以写成

$$
T_{\mathrm{spec}}
=
K T_{\mathrm{draft}}
+
T_{\mathrm{verify}}(K).
$$

如果这一轮平均能够推进 \(A\) 个 token，那么对应的理想化 speedup 为

$$
S
=
\frac{A\,T_{\mathrm{target}}}
{K T_{\mathrm{draft}}+T_{\mathrm{verify}}(K)}.
$$

这个公式也直接说明了 speculative decoding 的三个核心目标：让一次 verification 接受尽可能多的 token，即增大 \(A\)；让 draft 足够轻量，减小 \(T_{\mathrm{draft}}\)；同时控制一次多-token verification 的额外开销 \(T_{\mathrm{verify}}(K)\)。因此，**高 acceptance 并不自动意味着高 speedup**。如果为了提高 acceptance 使用了过大的 draft，或者 verification 本身变得很贵，最终节省下来的 target decode 时间仍然可能抵不过新增的开销。

### 4.3 What Actually Determines the Speedup?

从上一节的公式可以看出，speculative decoding 的性能主要由三个因素决定：acceptance length \(A\)、draft cost \(T_{\mathrm{draft}}\) 和 verification cost \(T_{\mathrm{verify}}(K)\)。其中最直接的是 \(A\)。一次 verification 能接受的 token 越多，就越能摊薄 target forward 的成本；如果 draft 经常在前几个位置就被拒绝，那么即使 verification 能一次处理很多 candidate，大部分计算也没有真正转化成有效输出。

Draft length \(K\) 也存在明显的 trade-off。增大 \(K\) 给了一次 verification 接受更多 token 的机会，但同时需要更多 draft forward，而且越靠后的 candidate 越容易因为前面某个 token 被拒绝而直接作废。因此，\(K\) 并不是越大越好。类似地，更大的 draft model 往往能提高 acceptance，但自身生成 candidate 的成本也更高。一个好的 drafter 追求的不是单纯更准确，而是以尽可能低的代价获得较长的 acceptance length。

Verification 本身也不是免费的。Target 虽然只执行一次 forward，但需要同时处理 \(K\) 个 candidate token，因此

$$
T_{\mathrm{verify}}(K) > T_{\mathrm{target}}
$$

通常是正常的。Speculative decoding 能够加速，是因为一次较宽的 verification 仍然可能比连续执行多次单-token decode 更划算。这个优势还会受到 batch size、KV cache 访问和 GPU utilization 的影响：例如 batch 较大时，普通 target decoding 已经能更充分地利用 GPU，speculative decoding 可以获得的额外收益往往会缩小。

因此，评价一个 speculative decoding 方法时，只看 acceptance rate 并不够。真正需要关心的是：**每次 target verification 最终推进了多少 token，以及为了得到这些 token 额外付出了多少 drafting 和 verification 成本。** 后面的各种 speculative decoding 方法，本质上都在优化这个 trade-off。


## 加速比

Target 自回归 decode 时，每个请求每步只多一个 token。同时解码的请求数是 batch $B$，这一步的延迟写成 $L_{\mathrm{target}}(B)$。该请求已有的 KV cache 仍要读进来，kernel 也要启动并同步。$B$ 小的时候矩阵乘法很窄，这些开销在这一次 forward 里占得更多。

一个周期的时间为 draft 的 $T_{\mathrm{draft}}$ 加上 target 验证的 $T_{\mathrm{target}}$。每个写出 token 分到

$$
L=\frac{T_{\mathrm{draft}}+T_{\mathrm{target}}}{\tau}.
$$

与基线一步相比，加速比为

$$
\eta=\frac{L_{\mathrm{target}}(B)}{L}.
$$

$\eta>1$ 等价于

$$
T_{\mathrm{draft}}+T_{\mathrm{target}}<\tau\,L_{\mathrm{target}}(B).
$$

## 验证步的时间

该 forward 的 token 宽度为 $n=B(\gamma+1)$。写成

$$
T_{\mathrm{target}}(n)\approx\alpha_t(n)+n\cdot\bar\beta_t(n).
$$

$\alpha_t(n)$ 是随 $n$ 变化较慢的一次调用开销：kernel 启动与同步、attention metadata、paged KV 索引、collective 启动。$n\bar\beta_t(n)$ 是 QKV、MLP、attention 和 KV 读写。$n$ 较小时 $\bar\beta_t(n)$ 随宽度下降。接近设备算力或带宽上限后不再明显下降，$T_{\mathrm{target}}(n)$ 更接近对 $n$ 线性。

每个写出 token 分到的 target 时间为

$$
\frac{T_{\mathrm{target}}(n)}{\tau}
\approx\frac{\alpha_t(n)}{\tau}+\frac{n}{\tau}\bar\beta_t(n).
$$

令 $p=\tau/(\gamma+1)$，则 $n/\tau=B/p$。基线写成 $L_{\mathrm{target}}(B)\approx\alpha_t(B)+B\bar\beta_t(B)$。只比较随 token 数增长的一项，target 侧变小的条件是

$$
\frac{B}{p}\bar\beta_t(n)<B\bar\beta_t(B),
$$

即

$$
\frac{\bar\beta_t(n)}{p}<\bar\beta_t(B).
$$

$p<1$ 使左端放大 $1/p$。宽度从 $B$ 增到 $n$，只有 $\bar\beta_t(n)$ 的下降超过这个因子，这一项才变小。$B$ 大到 $\bar\beta_t(B)$ 接近平台时，$\bar\beta_t(n)/\bar\beta_t(B)$ 接近 1，只要 $p<1$ 该不等式就不成立。$a_i$ 小则 $\tau$ 接近 1，$T_{\mathrm{draft}}/\tau$ 与 $\alpha_t(n)/\tau$ 变大。拒绝记账、KV 回滚和采样若进入 $\alpha_t(n)$，$\alpha_t(n)/\tau$ 可以大于模型 forward 本身。

## 实现

接受与重采样的实现来自 [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py)。接受条件是 $p(x)/q(x)\ge u$，$u$ 为均匀随机数。`NO_DRAFT_PROBS` 将该 token 的 draft 概率取为 1，条件变成 $p(x)\ge u$。`is_greedy` 的请求在入口返回。重采样核在 $\max(p-q,0)$ 上做 Gumbel-max。代码里的 `q` 是这组随机数，不是提议分布 $q_i$。

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

<p class="doc-reference-label"><strong>Reference</strong></p>

<p class="doc-reference">[1] Y. Leviathan, M. Kalman, Y. Matias. Fast Inference from Transformers via Speculative Decoding. ICML, 2023. <a href="https://arxiv.org/pdf/2211.17192">arXiv:2211.17192</a>.</p>

