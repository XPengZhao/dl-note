# Qwen3-4B DSpark 训练消融

## 实验设置

以 Qwen3-4B 为冻结的Target模型，训练五层 DSpark draft。目标模型的 token embedding 与 LM head 保持冻结。各实验训练 2,616 个optimizer steps，约一个 epoch，占完整scheduler的 10%。具体设置如下。

| **项目** | 设置 |
|---|---|
| **训练数据** | Open PerfectBlend；经目标模型重新生成回复并缓存特征，共 1,339,875 个样本 |
| **Regen 采样** | Non-thinking；temperature 0.7，top-p 0.8，top-k 20，min-p 0；每次回复最多生成 4096 tokens（不含 prompt） |
| **最大序列长度** | 4096 tokens（完整对话，含 prompt与response）。 |
| **Draft 架构** | 5 层；隐藏维度 2560；FFN 中间维度 9728；GQA：32 个 Q heads、8 个 KV heads，head dim 128 |
| **Aux 层** | `[1, 9, 17, 25, 33]` |
| **Block size** | 7；Backbone 输入为 `[anchor, MASK × 6]` |
| **Anchor 采样** | 每个样本最多 512 个 |
| **监督** | 仅监督 response |
| **Markov head** | Vanilla，rank 256 |
| **Batch size** | 4 GPUs × micro-batch 1 × 梯度累积 128；global batch 512 |
| **学习率** | 峰值 \(6\times10^{-4}\)；warmup 比例 0.04；完整计划为 26,160 步 |
| **Loss** | \(0.1L_{\mathrm{CE}}+0.9L_{\mathrm{L1}}+L_{\mathrm{conf}}\)；位置权重为 \(\exp(-i/4)\)，\(i=0,\ldots,6\) |
| **训练步数** | 2,616 optimizer step，约一个 epoch |
| **总参数量** | GQA baseline：1,393,133,569（约 1.393B；含冻结的 embedding 与 LM head，不含 target backbone） |
| **可训练参数量** | GQA baseline：615,221,249（约 615.2M） |

### 指标

第 \(i\) 个 draft 位置上，从 \(q_i\) 采样、按 \(p_i\) 做拒绝采样的接受率为

$$
a_i
=\sum_v\min\bigl(p_i(v),q_i(v)\bigr)
=1-\tfrac12\lVert p_i-q_i\rVert_1.
$$

训练接受长度

$$
\tau
=1+\sum_{i=0}^{6}\prod_{j=0}^{i}a_j.
$$


### 消融条件

各消融相对上表默认配置只改一项。

<span class="ablation-sq">\(\blacksquare\)</span>**Tau loss.** 附加

$$
L_{\tau}=\frac{8-\tau}{7},
$$

系数 0.1。CE 与 L1 仍按位置衰减，该项不衰减。

<span class="ablation-sq">\(\blacksquare\)</span>**MLA.** Attention 换为 MLA：32 个 query heads，每头 QK 为 64 content + 64 RoPE，V 为 128，KV latent 512。相对 GQA（32Q / 8KV，head dim 128），可训练参数 \(612.1\mathrm{M}\) vs \(615.2\mathrm{M}\)（\(-0.51\%\)）。压缩 KV 每层每 token 为 \(512+64=576\) 个数，GQA 为 \(8\times128\times2=2048\)，约 \(3.6\times\)。

<span class="ablation-sq">\(\blacksquare\)</span>**SWA.** 叠在 MLA 上。History 相对 anchor 截成 \([\max(0,a-W+1),a)\)，至多 \(W-1\) 个 token，不随块内 query 滑动；块内 7位仍双向可见。\(W\in\{1024,512,128\}\)，相对最大长度 4096，history KV 约为 \(1/4\)、\(1/8\)、\(1/32\)。

<span class="ablation-sq">\(\blacksquare\)</span>**Context-only.** 去掉当前 draft block 的 bi-directional K/V，只保留严格早于 anchor 的 target context。

<span class="ablation-sq">\(\blacksquare\)</span>**All-mask.** Backbone 输入改为七个 MASK，保留 block 内双向 attention。Markov head 第一个生成draft仍接收 anchor token logits bias。

<span class="ablation-sq">\(\blacksquare\)</span>**三层 aux.** Auxiliary layers 改为 \([1,17,33]\)。

<span class="ablation-sq">\(\blacksquare\)</span>**Full-sequence.** Prompt 与 response 同时作为监督和 anchor 候选。

## 指标总览

| **实验** | CE ↓ | 分布 L1 ↓ | Token acc ↑ | τ ↑ | Δτ |
| --- | ---: | ---: | ---: | ---: | ---: |
| **GQA baseline** | 1.1191 | 0.4550 | 74.35% | 4.9591 | 0.00% |
| **GQA + tau loss** | 1.1260 | 0.4546 | 74.34% | 4.9860 | +0.54% |
| **MLA** | 1.1402 | 0.4623 | 74.00% | 4.9368 | -0.45% |
| **MLA + SWA1024** | 1.1571 | 0.4690 | 73.65% | 4.9008 | -1.17% |
| **MLA + SWA512** | 1.1853 | 0.4787 | 73.16% | 4.8523 | -2.15% |
| **MLA + SWA128** | 1.2830 | 0.5140 | 71.35% | 4.6668 | -5.89% |
| **GQA context-only** | 1.1795 | 0.4767 | 73.30% | 4.8718 | -1.76% |
| **GQA all-mask** | 1.1860 | 0.4780 | 73.51% | 4.8791 | -1.61% |
| **GQA 三层 aux** | 1.1676 | 0.4735 | 73.36% | 4.8837 | -1.52% |


## 结果分析

### Tau loss


末段 τ 从 4.9591 提高到 4.9860，增加 0.0270，即约 0.54%。与此同时，token accuracy 几乎不变：74.3522% → 74.3407%。CE 也基本不变，而 distribution L1 略有改善。

位置上，收益随 draft depth 增大：

* position 1 overlap：约 **+0.045 pp**
* position 7 overlap：约 **+0.403 pp**

这说明 tau loss 的主要作用并非改善首 token top-1 prediction，而是提高后续位置的 distribution alignment。对于 speculative decoding，这一区别很重要：token accuracy 很难反映多个连续 draft token 的联合接受行为，而 τ 会显式累积前缀接受概率。

收益也主要出现在训练中后期。810–1000 步时：

* baseline：3.9087
* tau loss：3.8614

到 1610–1800 步：

* baseline：4.7117
* tau loss：4.7488

一个可能的原因来自 τ 本身的 prefix-product 结构。对较后位置 \(a_i\) 的梯度会乘以前面多个位置的 overlap；训练早期各位置预测较弱时，这些乘积较小，因此多步目标对后位置的有效梯度也较弱。随着基础 CE/L1 训练提高前缀 overlap，tau objective 才逐渐产生更明显的作用。

因此，当前结果更像是 **tau loss 在基础 draft 已经较强后进一步优化 multi-token alignment**，而不是替代原始 token-level objective。

## Anchor information and block communication affect different positions

![位置 overlap 相对 baseline 的差异](../../assets/images/qwen3-4b-dspark-ablation/position-deltas.png)

两个 ablation 对 block 内信息采用了不同的删除方式。

### Context-only

Context-only 移除当前 draft block 的 K/V，只允许模型读取严格早于 anchor 的 target context。首位置仍保留 anchor embedding residual，Markov head 也仍能看到训练序列中的前驱 token。

末段 τ 下降 **1.76%**。

位置上：

* position 1：−0.12 pp
* position 2：−1.56 pp
* position 7：−0.79 pp

首位置几乎不受影响，而后续位置明显下降，说明 **block 内通信主要服务于多步 draft prediction**。

### All-mask

All-mask 将 backbone 输入从 `[anchor, MASK×6]` 改为 `[MASK×7]`，但仍保留 block 内双向 attention；Markov head 仍接收 anchor 或训练时前驱 token。

末段 τ 下降 **1.61%**。

位置上：

* position 1：−1.65 pp
* position 7：−0.04 pp

对应 CE 的变化也高度集中在首位置：

* position 1：0.6036 → 0.6978
* position 7：1.9763 → 1.9983

因此 anchor information 与 block communication 呈现出明显不同的作用位置：

* **anchor embedding 主要帮助 block 的第一个预测；**
* **block 内信息交换主要帮助后续 draft positions。**

all-mask 在后续位置基本恢复，也说明这些位置可以从历史 target context、block 内其他 hidden states 以及 Markov predecessor 中重新获得部分信息。

目前还无法区分后续位置的收益究竟来自一般的 bidirectional block interaction，还是主要来自第一个位置对 anchor 信息的广播。一个直接的对照是只允许所有 query 读取当前 block 的第一个 K/V，同时保持历史 context 完全一致。

## Context length matters more than GQA vs. MLA

MLA 的末段 τ 为 4.9368，相比 GQA baseline 仅下降 **0.45%**。考虑到两者参数量只差 0.51%，当前结果表明，将 GQA 替换为这一版 MLA 对 draft quality 的影响较小。

但在 MLA 内缩短 target history 后，性能呈明显单调下降：

| **Context window** | τ | 相对 MLA |
| --- | ---: | ---: |
| **Unlimited** | 4.9368 | — |
| **1024** | 4.9008 | −0.73% |
| **512** | 4.8523 | −1.71% |
| **128** | 4.6668 | −5.47% |

当前 SWA 仅限制 anchor 之前的 target context：

$$
[\max(0,a-W+1),a),
$$

不改变 block 内双向 attention。

这一组结果比 GQA/MLA 差异更明显：**attention parameterization 本身影响较小，而历史 context range 对 draft prediction 十分敏感。**

尤其是 SWA128，其损失并不局限于第一位置，而会持续到整个 block。也就是说，Markov predecessor 和 block 内 hidden-state communication 无法完全补偿长历史信息。

这表明 DSpark 的 draft prediction 并不是纯局部 continuation。即使 target 已经提供了 anchor 和训练时前驱 token，较远的上下文仍能显著降低预测不确定性。

这一结果也意味着，对 speculative draft 做 context compression 时，**简单截断 sequence history 可能比修改 attention representation 更危险**。如果系统目标是降低 KV cache 或 attention 开销，SWA1024 目前是更合理的起点：相对 MLA 的训练 τ 损失只有 0.73%，远低于 512 和 128。

需要注意的是，当前 MLA 实现仍使用展开后的 K/V attention/cache，并未形成完整的 compressed latent cache 推理路径。因此该实验只说明其训练质量接近 GQA，还不能推出实际的 memory 或 throughput 优势。

## Auxiliary features help later predictions

baseline 使用 target backbone 的五层 auxiliary features：

$$
[1,9,17,25,33].
$$

将其减少为：

$$
[1,17,33]
$$

后，输入投影参数从 32.77M 降至 19.66M，总 trainable parameters 减少 13.11M，即约 **2.13%**。

代价是：

* τ：−1.52%
* token accuracy：约 −0.99 pp
* position 1 overlap：−0.42 pp
* position 7 overlap：−1.15 pp

损失随位置加深而扩大，说明 intermediate target representations 对较远 draft positions 更有价值。一个合理解释是：后续位置需要从固定 anchor 附近推断更远的未来状态，多层 target features 提供了比单一高层 representation 更丰富的局部和语义信息。

从当前数字看，这一压缩的 trade-off 并不明显有利。总参数只减少约 2%，但 τ 损失达到 1.5%。如果三层 aux 不能显著降低实际 feature bandwidth、projection latency 或 cache footprint，就很难仅从参数量上证明其价值。

另外，当前训练仍复用五层离线缓存并在读取后切片，因此该实验本身并未降低缓存文件大小或原始 I/O。

## Full-sequence changes the training distribution

Full-sequence run 的末段结果为：

* CE：1.4346
* token accuracy：68.58%
* τ：4.7750

这些数值明显低于 response-only baseline，但二者并非同分布比较。

Full-sequence 将 prompt 和 response 都加入监督和 anchor 候选，因此训练指标包含了不同类型的位置。固定每个样本最多 512 个 anchor 后，加入 prompt positions 还会改变 response anchors 获得的监督预算。

因此，该结果目前能说明的是：

**将训练范围扩展到 full sequence 会显著改变 optimizer 实际看到的位置分布。**

它是否提高或降低最终 response speculative decoding quality，需要在固定的 response-only validation positions 上重新评估。同时应记录 prompt / response anchor 数量，判断是否存在监督预算被 prompt 稀释的问题。


## Takeaways

这组消融给出了几个比较明确的信号。

**1. 长历史上下文很重要。**
GQA → MLA 只损失 0.45% τ，而 MLA → SWA128 损失 5.47%。对当前 DSpark，context range 比 attention parameterization 更敏感。

**2. Anchor 与 block 内通信承担不同功能。**
移除 anchor 主要损伤第一个预测位置；移除 block K/V 主要损伤后续位置。DSpark 的多步预测同时依赖初始 anchor information 和 block 内的信息传播。

**3. Token accuracy 不足以描述 speculative draft quality。**
Tau loss 几乎没有改变 top-1 accuracy，却主要改善了后续位置 overlap。直接优化 multi-token objective 可能捕获 CE / token accuracy 看不到的收益。

**4. Tau loss 有正向信号，但收益仍小。**
当前末段 τ 提升约 0.54%，且主要在训练后期出现。最重要的下一步是同实现 baseline + 实际 MAL，而不是继续比较 training loss。

**5. Auxiliary feature compression 目前缺少足够的系统收益。**
三层 aux 仅减少 2.13% 总参数，却损失约 1.52% τ。只有在实际 feature bandwidth 或 latency 明显下降时，这一取舍才可能成立。

从下一轮实验优先级看，最值得补的是：**SpecForge baseline vs. tau-loss 的同实现 MAL 对比、block-first-only attention、以及 MLA+SWA1024 的实际 KV/cache 与 decode latency。**
