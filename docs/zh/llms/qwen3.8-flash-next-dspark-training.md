# Qwen3.8-Flash-Next DSpark 训练实验

以 Qwen3.8-Flash-Next 为目标模型（Target），冻结其参数，训练三层 DSpark 草稿模型（draft）。各实验使用相同的数据和隐藏状态缓存，比较不同结构的收敛情况、候选选择和接受长度。

## 实验设置

各组实验共用以下训练设置，结构改动和额外损失在对应章节说明。

| **项目** | 设置 |
| --- | --- |
| **Target** | Qwen3.8-Flash-Next；Target backbone、token embedding 与 LM head 冻结 |
| **训练数据** | PerfectBlend，经 Target 重新生成回复，由 DeepSpec 缓存特征 |
| **离线 cache** | Aux hidden states 与 `target_last_hidden_states`；后者经冻结的 LM head 得到 teacher 分布 |
| **验证集** | GSM8K 256 条问题对应的重新生成回复及 hidden cache |
| **最大序列长度** | 4096 tokens，包含 prompt 与 response |
| **Draft 架构** | 3 层；隐藏维度 2560；FFN 中间维度 9728；GQA：24 个 Q heads、2 个 KV heads，head dim 256 |
| **Attention 设置** | 三层均为 full attention，无滑动窗口；Q/K RMSNorm 与 RoPE，RoPE theta 为 \(10^6\) |
| **Aux 层** | `[45, 46, 47]`，按配置中的零基层号记录 |
| **Block size** | 7；Baseline 输入为 `[anchor, MASK × 6]` |
| **Anchor 采样** | 每个样本最多 512 个 |
| **监督** | Response-only |
| **Markov head** | Vanilla，rank 256 |
| **Batch size** | 4 GPUs × micro-batch 4 × 梯度累积 32；global batch 512 |
| **学习率** | 峰值 \(3\times10^{-4}\)；warmup 比例 0.04 |
| **训练计划** | 完整 scheduler 为 26,040 optimizer steps；约 2,604 步 / epoch |
| **Baseline loss** | \(0.1L_{\mathrm{CE}}+0.9L_{\mathrm{L1}}+L_{\mathrm{conf}}\)；位置权重 \(\exp(-i/4)\)，\(i=0,\ldots,6\) |
| **精度与随机种子** | BF16；seed 42 |
| **Checkpoint** | 本组对照实验每 651 步保存一次 |

学习率按 26,040 步调度。一个 epoch 的实验训练到第 2,604 步，使用同一调度的前段。从已有 checkpoint 继续训练时，分别记录此前的训练步数和本次新增步数。

## 数据集

### 来源与回复重生成

训练使用 [Open PerfectBlend Regenerated with Qwen3.8-Flash-Next](https://huggingface.co/datasets/xpzhao11/open-perfectblend-qwen38-flash-next-regen)。输入（prompt）来自 [mlabonne/open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend)，回复（response）由 `Qwen/Qwen3.8-Flash-Next` 重新生成。各组实验使用同一份重生成数据。

### 生成参数

回复由 `Qwen/Qwen3.8-Flash-Next` 在 non-thinking 模式下生成，采样参数为 temperature 0.7、top-p 0.8、top-k 20、min-p 0，最大新生成长度为 4096 tokens。

训练序列的 4096 tokens 上限包含 prompt 和 response。

### 样本规模与长度分布

重生成数据共包含 **1,349,859 条对话记录**、1,790,298 次 assistant 回复。其中，204,494 条记录包含多次回复，占 15.15%。

使用 Qwen3.8-Flash-Next tokenizer 统计每条对话记录截断前的长度。Prompt 累计所有非 assistant 消息的正文，response 累计所有 assistant 回复的正文。总长度按 non-thinking 对话模板编码，包含特殊 token，不追加用于开始新回复的提示。

- **Prompt**：平均 134.9 tokens，中位数 72，P95 为 495，P99 为 973。
- **Response**：平均 1,166.9 tokens，中位数 554，P95 为 4,242，P99 为 7,975。
- **总长度**：平均 1,320.4 tokens，中位数 681，P95 为 4,621，P99 为 8,226。

总长度超过 4096 tokens 的记录有 **111,429 条，占 8.25%**。多轮记录中的 response 累计全部回复，因此可以超过单次生成的长度上限。


<!-- 待补：有效训练样本数、GPU 型号、cache 版本，以及各 run 的最终配置。 -->

## 评估指标

### 离线指标

给定相同的参考前缀，记 Target 的概率分布（teacher 分布）为 \(p_i\)，draft 的概率分布为 \(q_i\)。两者的分布重叠率为

$$
a_i=\sum_v\min\bigl(p_i(v),q_i(v)\bigr)
=1-\frac12\lVert p_i-q_i\rVert_1.
$$

每个有效 block 的概率接受长度定义为

$$
\tau_{\mathrm{prob}}
=1+\sum_{i=0}^{6}\prod_{j=0}^{i}a_j,
$$

不足 7 个有效位置时，将无效位置的 \(a_i\) 置零，再对有效 block 求平均。各位置的 `accept_rate@i` 按有效 token 数求平均，各 token 等权。离线计算使用参考 token 作为前驱（teacher forcing）。在线推理的平均接受长度（MAL）则由实际生成和 Target 验证得到。

前缀重排器（Prefix reranker）的 `train/tau_probabilistic` 和 `eval/tau_probabilistic` 使用重排后的分布：对 top-16 候选的修正分数做 softmax，候选之外的概率置零。Teacher 使用原始全词表分布。重排前的 DSpark 分布也在全词表上归一化。

| **指标** | 含义 |
| --- | --- |
| **`rerank_tau_gain`** | 同一 checkpoint、同一批 anchor 上，重排后与重排前的 \(\tau_{\mathrm{prob}}\) 之差 |
| **`rerank_target_agreement_gain`** | 重排前后选中 teacher argmax 的比例之差；0.02 表示增加 2 个百分点 |
| **`reranker/candidate_recall`** | Teacher argmax 落在 Markov top-16 内的比例 |
| **`rerank_teacher_forced_prefix_gain`** | 与参考 token 序列连续匹配的前缀长度变化，仍在 teacher-forced 输入下计算 |

`rerank_tau_gain` 比较同一模型重排前后的结果，包含分数修正、top-16 截断和重新归一化的影响。两种训练方法的整体效果以相同训练步数的独立 baseline 为参照。

### 推理指标

推理评测记录 MAL、每轮生成与验证的耗时，以及 tokens/s。对照实验使用相同的测试集、采样方式、并发数、张量并行度（TP）、最大长度和每轮候选 token 数。

目前在线评测覆盖 Prefix reranker 的 greedy 推理。Engram 和 Draft Memory 的 vLLM 集成与在线评测仍待完成。

<!-- 待补：离线 eval 的固定配置、训练末段统计窗口、线上启动参数和评测命令。 -->

## 实验分组

| **实验** | 初始化与更新范围 | 比较的问题 |
| --- | --- | --- |
| **Baseline** | 从头训练原 DSpark | 原结构随训练步数增加的收敛情况 |
| **Engram-Markov** | 从头联合训练 draft 与 Engram 门控投影 | Markov head 的 n-gram 条件对接受长度的影响 |
| **Engram-MASK：门控残差** | 从头联合训练 draft、输入投影与逐槽标量门 | 保留 MASK embedding 的门控注入效果 |
| **Engram-MASK：直接替换** | 从头联合训练 draft 与输入投影 | 内容相关输入替代固定 MASK 的效果 |
| **Draft Memory** | 从头训练 draft 与 memory adapter，仅第二阶段回传 | 上一轮 draft 状态对新 anchor 下续写的作用 |
| **Prefix reranker：联合训练** | 从头联合训练 draft 与 reranker | 加入前缀重排后，相对独立 baseline 的效果 |
| **Prefix reranker：冻结基线** | 加载 baseline step 2604，仅训练新加的 reranker | 固定原模型后，重排模块能改善多少 |

<!-- 待补：Carryover teacher forcing 的实验设置与结果。 -->

## Baseline 架构

Baseline 采用标准 DSpark 结构。三层 draft 主干网络一次生成各预测位置的隐藏状态，LM head 将其映射为初始 logits。Markov head 根据前驱 token 添加偏置，confidence head 预测对应位置的分布重叠率。

![DSpark baseline：左为并行 draft 与顺序选择，右为首个预测位置的 Markov 修正及 confidence 分支](../../assets/images/qwen3.8-flash-next-dspark/dspark-baseline.svg){ width="100%" }

图左以 D 为本轮起始 token（anchor），展示 E – I 的生成，虚线表示当前 token 的选择依赖前一个 token。右图展开预测 E 时的 Markov 修正与 confidence 分支。实际训练的 block size 为 7，包含一个 anchor 和六个 MASK。

### DSpark 主干网络

设 block size 为 \(B\)，token embedding 为 \(E(\cdot)\)。对于前缀 `[... A B C]` 与新 anchor D，draft 输入为

$$
X=[E(D),\underbrace{E(\mathrm{MASK}),\ldots,E(\mathrm{MASK})}_{B-1\text{ 个槽}}],
\qquad B=7.
$$

MASK 槽共享输入 embedding，并使用不同的位置编号。anchor 槽对应 \(h_1\)，用于预测 E，后续槽依次预测 F – K。对 anchor 之前的每个位置 \(j\)，拼接 Target 第 45、46、47 层的隐藏状态，经投影与归一化得到上下文特征：

$$
c_j=\operatorname{RMSNorm}\!\left(
W_{\mathrm{aux}}[H_j^{45};H_j^{46};H_j^{47}]
\right),
\qquad
W_{\mathrm{aux}}:\mathbb R^{7680}\rightarrow\mathbb R^{2560}.
$$

\(W_{\mathrm{aux}}\) 是可训练的无偏置投影。图中的 Target 上下文截止 C，anchor D 通过自身 embedding 输入。

三层主干网络均采用 GQA，每层包含 24 个 query heads 和 2 个 K/V heads，head dimension 为 256，即每 12 个 query heads 共享一组 K/V。三层均使用 full attention。对于一个 anchor block，每个 query 可读取 anchor 之前全部有效的 Target 上下文，以及自身 block 中的所有输入槽。

每层的 query 来自当前 draft 隐藏状态，上下文特征与当前 block 隐藏状态分别经过该层的 K/V 投影，沿序列维拼接后在同一次 attention 中读取。三层共享 \(c_j\)，但各层的 Q/K/V 投影参数独立。Draft 状态通过 attention、残差连接和 MLP 逐层更新，最终经 RMSNorm 得到 \(h_1,\ldots,h_B\)，再由冻结的 LM head 并行生成全词表初始 logits \(\ell_1^0,\ldots,\ell_B^0\)。

### Markov 顺序修正

令 \(t_0=D\)，\(t_i\) 为第 \(i\) 个生成位置的 token。Markov head 将前驱 token 映射为低秩表示 \(m_i\)，再投影为词表偏置 \(b_i\)：

$$
\begin{aligned}
m_i&=W_1[t_{i-1}]\in\mathbb R^{256},\\
b_i&=W_2m_i,\\
\ell_i&=\ell_i^0+b_i,\qquad
q_i=\operatorname{softmax}(\ell_i).
\end{aligned}
$$

\(W_1\) 与 \(W_2\) 均参与训练。偏置 \(b_i\) 由前驱 token 确定，上下文信息由 \(\ell_i^0\) 提供。图右的第一步对应 \(m_1=W_1[D]\) 与 \(b_1=W_2m_1\)。Greedy 选择全词表中分数最高的 token，即 \(\hat t_i=\arg\max_v\ell_i(v)\)，图示结果为 E。

训练使用参考前驱 token，所有位置的 Markov 修正可并行计算。推理使用实际选出的前驱 token，依次查表、修正 logits 并选择下一个 token。主干网络的隐藏状态在整轮内保持不变。

### Confidence head 与训练目标

Confidence 分支读取 draft 隐藏状态 \(h_i\) 与词表投影之前的 Markov 表示 \(m_i\)，通过线性层与 sigmoid 得到预测值：

$$
s_i=w_c^\top[h_i;m_i]+b_c,
\qquad
\hat a_i=\sigma(s_i).
$$

输入维度为 \(2560+256=2816\)。监督目标为相同参考前缀下 teacher 分布 \(p_i\) 与修正后 draft 分布 \(q_i\) 的重叠率 \(a_i=\sum_v\min(p_i(v),q_i(v))\)。该标签停止梯度，以二元交叉熵训练 confidence head。图右的 \(\hat a_1\) 用作辅助预测，E 仍由修正后的 logits 选择，CE、L1 使用原有位置权重。

基础训练目标为 \(0.1L_{\mathrm{CE}}+0.9L_{\mathrm{L1}}+L_{\mathrm{conf}}\)。CE 监督数据中记录的下一个 token，L1 减小 teacher 与 draft 全词表分布的差异，confidence loss 拟合分布重叠率。三项损失均在有效 response 位置按位置衰减权重求平均。

离线 aux 缓存提供模型输入，Target 最后一层的隐藏状态经冻结 LM head 生成用于监督和评估的 teacher 分布。

## Engram 增强 Markov head

Engram-Markov 在前驱 token 的低秩表示中加入 n-gram 特征，由当前 draft 隐藏状态控制其权重。主干网络与 MASK 输入保持不变。

![Engram 增强 Markov head：左为整块生成，右为以 D 为 anchor 预测 E 的第一步](../../assets/images/qwen3.8-flash-next-dspark/engram-markov.svg){ width="100%" }

图右展开以 D 为 anchor 预测 E 的第一步。Engram 根据截至 D 的前缀 `[... A B C D]` 查表，得到特征 \(e(D)\)，经投影与门控后加入 \(W_1[D]\)，再修正 logits。

### 门控与特征注入

从 1 开始编号生成位置，令 \(t_0\) 为 anchor，\(h_i\in\mathbb R^{2560}\) 和 \(\ell_i^0\) 分别为第 \(i\) 个预测位置的 draft 隐藏状态与初始 logits。\(e_{i-1}\in\mathbb R^{2560}\) 是根据截至前驱 token \(t_{i-1}\) 的前缀查表得到的原始 Engram 特征，由 n-gram 查表结果拼接而成，随后进行 K/V 投影。

Draft 隐藏状态提供 query，Engram 特征提供 key 和 value。记 \(N\) 为无可训练 affine 参数的 RMSNorm，Markov rank 为 \(r=256\)，则

$$
\begin{aligned}
\bar e_{i-1}&=N(e_{i-1}),\\
\mathbf q_i&=N\!\left(W_QN(h_i)\right),\\
\mathbf k_i&=N(W_K\bar e_{i-1}),\qquad
\mathbf v_i=W_V\bar e_{i-1},\\
s_i&=\frac{\mathbf q_i^\top\mathbf k_i}{\sqrt r},\\
g_i&=\sigma\!\left(\operatorname{sign}(s_i)
\sqrt{\max(|s_i|,10^{-6})}\right).
\end{aligned}
$$

\(g_i\) 由当前位置的一对 query 和 key 计算得到，用来缩放 Engram value。各位置独立计算门值。增强后的 Markov 表示和最终 logits 为

$$
m_i=W_1[t_{i-1}]+g_i\mathbf v_i,
\qquad
\ell_i=\ell_i^0+W_2m_i.
$$

其中 \(W_1[t_{i-1}]\) 保留原有前驱 token 表示，\(W_2\) 将相加后的 256 维表示映射为全词表偏置。Confidence head 同样读取增强后的 \(m_i\)。

三个无 bias 投影 \(W_Q,W_K,W_V\) 均为 \(2560\rightarrow256\)，新增 \(3\times2560\times256=1{,}966{,}080\) 个 draft 参数。\(W_V\) 零初始化，使初始 Engram 残差为零。在 backbone 和原 Markov 参数相同的条件下，初始输出与 vanilla Markov 一致。

### 训练与推理

离线训练读取参考序列对应的 Engram 特征。预测 \(t_i\) 时使用前驱位置 \(t_{i-1}\) 的缓存，各位置的门值与 Markov 修正可并行计算。

推理随实际生成前缀逐 token 查表：读取 D 位置的特征预测 E，再根据包含 E 的前缀查表预测 F。上述新增参数量计入 draft 的投影矩阵，Engram 表另占存储空间，查表和特征传输也增加推理耗时。

## Engram 门控增强 MASK 输入

门控 Engram-MASK 在原 MASK embedding 上加入前缀特征残差，以各槽的可训练标量控制注入强度。门值零初始化，后续仍使用 vanilla Markov head。

![Engram 门控增强 MASK：共享的 pre-anchor 特征经逐槽 tanh 门缩放后，与原 MASK embedding 相加](../../assets/images/qwen3.8-flash-next-dspark/engram-gate-mask.svg){ width="100%" }

图中 anchor 为 D，Engram 特征来自截至 C 的前缀 `[... A B C]`。特征投影后由各 MASK 槽共享，anchor 保留原 token embedding。

### 门控残差

令 \(a\) 为 anchor 在原序列中的位置，\(e_{a-1}\in\mathbb R^{2560}\) 为其前一个位置的原始 Engram 特征。\(P\in\mathbb R^{2560\times2560}\) 是无偏置的可训练投影，投影后使用无可训练仿射参数的 RMSNorm。Block 输入为

$$
\begin{aligned}
z_a&=\operatorname{RMSNorm}(Pe_{a-1}),\\
u_{a,0}&=E(x_a),\\
u_{a,j}&=E(\mathrm{MASK})+\tanh(\theta_j)\,z_a,
\qquad j=1,\ldots,6,\\
X_a&=[u_{a,0},u_{a,1},\ldots,u_{a,6}].
\end{aligned}
$$

每个 MASK 槽有一个可训练标量 \(\theta_j\)，由所有样本和 block 共享。输入内容通过 \(z_a\) 影响特征，门值由槽位置决定。\(\tanh(\theta_j)\in(-1,1)\)，可以正向或反向缩放残差。

所有 \(\theta_j\) 均零初始化，初始 \(u_{a,j}=E(\mathrm{MASK})\)，与相同权重的 baseline 一致。投影 \(P\) 随机初始化，在门值离开零后开始获得任务梯度。新增参数为 \(2560^2+6=6{,}553{,}606\) 个，包括一个共享投影和六个标量门。

### 训练与推理

训练对每个 anchor 读取 \(a-1\) 位置的 Engram 缓存，缺少有效前驱或 block 无效时置零。六个槽共享投影结果，保留各自的位置编号、RoPE 与块内双向 attention。投影和标量门都由基础训练损失优化。

推理每轮读取一次 anchor 之前的特征，经投影与门控后输入 draft，整块生成期间保持不变。该特征可以在新 anchor 产生前查表得到，因此可尝试与 Target 计算并行执行。

## Engram 替换 MASK 输入

直接替换版本用 anchor 之前的 Engram 特征投影替代 MASK embedding，省去标量门，其余结构沿用 baseline。

![Engram 替换 MASK：pre-anchor 特征经投影和归一化后填充并行 draft 的输入槽](../../assets/images/qwen3.8-flash-next-dspark/engram-fix-mask.svg){ width="100%" }

图中以 D 为 anchor，读取截至 C 的特征 \(e(C)\)。D 使用原 token embedding，后续槽使用同一 Engram 投影。

### 输入替换

令 \(a\) 为 anchor 在原序列中的位置，\(e_{a-1}\in\mathbb R^{2560}\) 为其前一个位置的原始 Engram 特征。无偏置线性投影 \(P\in\mathbb R^{2560\times2560}\) 将其映射到 draft 输入空间，随后使用无可训练仿射参数的 RMSNorm：

$$
z_a=\operatorname{RMSNorm}(Pe_{a-1}),
\qquad
X_a=[E(x_a),\underbrace{z_a,\ldots,z_a}_{6\text{ 个槽}}].
$$

六个槽共享 \(z_a\)，位置编号为 \(a+1,\ldots,a+6\)，沿用 RoPE 和块内双向 attention。

新增参数为投影矩阵 \(P\)，共 \(2560^2=6{,}553{,}600\) 个。该投影随机初始化，从训练开始就替换原有 MASK 输入。

### 训练与推理

特征选取与门控版一致：训练读取 \(a-1\) 位置的缓存，缺少有效前驱时置零。新增投影由基础任务损失直接优化。

推理每轮查表并投影一次，所有替换槽共享 \(z_a\)，整轮内保持不变。相比门控版，直接替换省去逐槽缩放与残差相加，保留查表、特征传输、投影和归一化。

## Draft Memory

Draft Memory 保留上一轮 draft 对后续位置的隐藏表示，供下一轮预测使用。训练分两次执行同一个 draft 模型，复用已有 Target 隐藏状态缓存。

![DSpark Draft Memory：第一遍筛选后缀 hidden，第二遍在各层 attention 中读取 memory](../../assets/images/qwen3.8-flash-next-dspark/draft-memory.svg){ width="100%" }

### 两阶段训练

第一阶段不输入 memory，也不保留梯度，计算 draft 最终 RMSNorm 后的隐藏表示。LM head 与 Markov head 使用参考前驱 token 选择分数最高的 token，再找到预测与记录序列首次不匹配的位置。首次不匹配之前的前驱与 greedy 生成一致，因此可以用这次并行计算确定新的 anchor。

图中旧 anchor 为 C，Target 上下文截止 B，参考续写为 D、E、F、G、H。若前两个预测匹配，第三个预测为 X，则以参考 token F 为新 anchor，将上下文扩展至 E。对应已匹配位置及首次不匹配位置的 \(h_1,h_2,h_3\) 被移除，仅保留 \(h_4,h_5\)，供第二阶段预测 G、H、I、J、K。

训练按记录序列确定拒绝位置。Memory 直接取自旧 anchor 与 MASK 输入生成的隐藏状态，选出 X 后不重新计算这些状态。

第二阶段输入新 anchor、对应的 Target 上下文和保留的 memory，使用参考前驱 token 计算基础训练损失。梯度由这一阶段回传，更新共享 draft 和 memory 适配器。

### Memory 筛选与读取

令旧 anchor 的绝对位置为 \(a\)，block size 为 \(B\)，第 \(i\) 个输出 hidden 为 \(h_i^{\mathrm{old}}\)，其中 \(i=1,\ldots,B\)。该 hidden 的旧 query 位置为 \(a+i-1\)，预测的 token 位置为 \(a+i\)。若连续匹配了 \(m<B\) 个 token，则

$$
a'=a+m+1,\qquad
\mathcal I=\{m+2,\ldots,B\}.
$$

新 anchor 为 \(a'\)。后缀索引集合 \(\mathcal I\) 中的隐藏表示经共享的无偏置线性投影和带可训练尺度参数的 RMSNorm 转换为 memory：

$$
M_i=\operatorname{RMSNorm}_{\gamma}
\left(P\,\operatorname{stopgrad}(h_i^{\mathrm{old}})\right),
\qquad i\in\mathcal I.
$$

\(P\in\mathbb R^{2560\times2560}\) 随机初始化，\(\gamma\) 为归一化尺度参数。Adapter 由投影和归一化组成，共增加 \(2560^2+2560=6{,}556{,}160\) 个参数。第一阶段的隐藏表示停止梯度，\(P\) 与 \(\gamma\) 由第二阶段的损失优化。

Memory 的 RoPE 位置采用预测位置 \(a+i\)，即旧 query 位置编号加一。因此，首条保留状态对齐新 anchor 后的第一个预测位置 \(a'+1\)。图中 \(h_4,h_5\) 分别对齐 G、H。

Adapter 计算一次，得到的同一组 \(M\) 供三层 draft 使用。每层的 query 来自当前 draft 隐藏状态。将正确上下文、memory 和当前 draft 隐藏状态拼接，再通过该层原有的投影得到 K/V。省略归一化与 RoPE 后，可写为

$$
\operatorname{Attn}_{\ell}
\left(
Q_{\ell}(X_{\ell}),
K_{\ell}([H_{\mathrm{ctx}};M;X_{\ell}]),
V_{\ell}([H_{\mathrm{ctx}};M;X_{\ell}])
\right).
$$

其中 \(H_{\mathrm{ctx}}\) 为投影后的正确前缀特征，\(X_\ell\) 为当前层的 draft 隐藏状态。每个 query 在同一次 attention 中读取本 anchor 的有效 memory、当前 draft block 和 anchor 之前的正确上下文。

若整块全部匹配，则转移至记录序列的 bonus 位置，memory 为空。缺少后续监督或跨越回答边界的样本跳过。其余样本保留全部有效后缀。

### 训练与推理

相比 baseline，每步训练多一次无梯度的 draft 前向计算，以及首次错误定位、memory 投影和额外 K/V 的计算。比较训练效率时，同时记录训练步数和实际耗时。

在线推理为每个请求保存上一轮 draft 的最终隐藏表示。Target 验证后，移除已接受位置和首次拒绝位置的状态，将剩余后缀按预测位置用于下一轮。各层通过自身投影生成 memory K/V。

训练用参考序列确定拒绝位置，memory 来自未输入 memory 的第一阶段。在线推理由 Target 验证确定拒绝位置，并连续复用上一轮的隐藏状态，因此可能累积更早轮次的 memory 信息。

## Prefix reranker

### 设计动机

原始 Markov head 根据单个前驱 token 计算偏置。Prefix reranker 将当前 block 内已选出的整个前缀用于候选评分。

Prefix reranker 通过块内因果 Transformer 编码 anchor 与已选 token，再结合当前 draft 隐藏状态，对 Markov top-16 候选进行残差打分。

![DSpark prefix reranker：左为整块生成，右为第三个位置的候选重排](../../assets/images/qwen3.8-flash-next-dspark/prefix-reranker.svg){ width="100%" }

图右展开第三个预测位置：前缀 `[D, E, F]` 编码为 \(r_3\)，与 \(h_3\) 和候选特征共同生成修正分数。图示候选 G 在残差修正后超过 H。

### 候选与前缀表示

设 \(t_0\) 为 anchor，\(t_i\) 为第 \(i\) 个生成位置的 token。训练时使用参考 token，推理时使用当前轮已选 token。记主干网络输出为 \(h_i\in\mathbb{R}^{2560}\)，冻结 LM head 生成的初始 logits 为 \(\ell_i^0\)。

复用 Markov head 的两张秩为 256 的 token 表 \(E_{\mathrm{in}}\) 与 \(E_{\mathrm{out}}\)，得到原 DSpark 分数

$$
\ell_i(v)=\ell_i^0(v)+E_{\mathrm{in}}(t_{i-1})^\top E_{\mathrm{out}}(v),
\qquad
C_i=\operatorname{Top16}(\ell_i).
$$

候选集合由 Markov 修正后的分数确定。因此，teacher argmax 的候选覆盖率构成重排选择准确率的上界。

Prefix Transformer 的输入为 \([t_0,\ldots,t_{i-1}]\)。各 token 经共享的 \(E_{\mathrm{in}}\) 查表、线性投影并加上可学习的块内位置 embedding。一个 pre-norm 因果 Transformer 层生成前缀表示 \(r_i\)。其宽度为 256，包含 4 个 attention heads，FFN 为 \(256\rightarrow1024\rightarrow256\)，使用 SiLU，输出经过 LayerNorm。

前缀编码范围限于当前 block，历史上下文由 \(h_i\) 提供。

### 残差打分

将 \(h_i\) 投影到 256 维，与 \(r_i\) 拼接，再经过 MLP 和输出投影形成查询向量 \(u_i\)。候选 token 则从共享的 \(E_{\mathrm{out}}\) 查表并投影为 \(k_v\)：

$$
\begin{aligned}
u_i&=W_o\,\operatorname{SiLU}\!\left(W_q[W_hh_i;r_i]+\beta_q\right),\\
k_v&=W_cE_{\mathrm{out}}(v),\\
\Delta_i(v)&=\frac{u_i^\top k_v}{\sqrt{256}},\\
s_i(v)&=\ell_i(v)+\Delta_i(v),\qquad v\in C_i.
\end{aligned}
$$

最终选择为 \(\hat t_i=\arg\max_{v\in C_i}s_i(v)\)。Prefix Transformer 编码前缀一次，候选通过查询向量与投影特征的内积并行评分。

\(W_o\) 零初始化，并列分数优先保留原 DSpark 的选择，因此初始 greedy 选择与原模型一致。概率重叠指标使用 top-16 内归一化后的分布。

### 训练目标与更新范围

训练沿用现有隐藏状态缓存。每个 block 的前驱输入右移一位，以 anchor 开头，其后为参考序列中的 token。Prefix Transformer 使用因果掩码，一次前向计算编码所有位置。

记 \(p_i\) 为缓存参考前缀对应的 teacher 分布，重排监督标签为

$$
v_i^\star=\arg\max_v p_i(v).
$$

排序标签取 teacher 概率最高的 token。记 \(\mathcal H\) 为有效且 \(v_i^\star\in C_i\) 的位置集合，在这些位置计算候选内的排序 CE：

$$
L_{\mathrm{rank}}
=-\frac{
\sum_{i\in\mathcal H} w_i
\log\frac{\exp s_i(v_i^\star)}
{\sum_{v\in C_i}\exp s_i(v)}
}{
\sum_{i\in\mathcal H}w_i
}.
$$

其中 \(w_i=\exp(-(i-1)/4)\)，求和覆盖 batch 中各 block 的有效命中位置。没有命中位置时，该项为零。联合训练保留重排前的原始 DSpark 目标：

$$
L=0.1L_{\mathrm{CE}}+0.9L_{\mathrm{L1}}+L_{\mathrm{conf}}+L_{\mathrm{rank}}.
$$

原始 DSpark 损失监督全部有效位置，排序损失监督 teacher argmax 进入 top-16 的位置。联合训练时，两项损失共同更新主干网络和共享 Markov 参数。

冻结基线实验从训练 2,604 步的 baseline 开始，固定主干网络、Markov 和 confidence head，只训练 reranker 的新增参数，并重新初始化优化器与学习率调度。曲线横轴为此前训练步数加上本次新增步数，学习率按本次新增步数调度。

### 推理开销

主干网络每轮只运行一次。第 \(i\) 个位置利用上一个已选 token 更新 Prefix Transformer 的本地 K/V，得到 \(r_i\)，并计算 Markov 修正后的 top-16 候选及其残差分数。选出的 token 进入下一位置，整块候选最后交给 Target 验证。

Prefix K/V 在当前 block 内增长，每轮重新清空。每个 token 的选择依次执行前缀编码、候选筛选和残差评分。

## 结果总览

已记录 Prefix reranker 与独立 baseline 在 step 7812 的 GSM8K 全集评测，包含 thinking 与 non-thinking 两种设置。其余结构的在线结果待补。

### Prefix reranker 在线评测

2026 年 10 月 5 日，使用 `qwen38-prefix-reranker-step7812` 在 vLLM 中进行多轮 draft 生成与 Target 验证。测试集为 GSM8K test split 全部 1,319 条问题。训练期间的离线验证使用 GSM8K 256 条问题对应的重新生成回复和隐藏状态缓存。

| **项目** | 设置 |
| --- | --- |
| **Target** | Qwen3.8-Flash-Next |
| **Draft checkpoint** | `qwen38-prefix-reranker-step7812` |
| **接口与 prompt** | `/v1/chat/completions`；5-shot 示例与待回答问题拼接为单条 user message |
| **Thinking 设置** | 对比开启 thinking 与 `enable_thinking=False`；后者与 DeepSpec cache 的 non-thinking 设置一致 |
| **Target 采样** | Temperature 0 |
| **Draft 选择** | Greedy；每轮 7 个 proposal；关闭 adaptive verification |
| **并行与并发** | Target TP 4；服务端 `max_num_seqs=8`；客户端最大并发 256 |
| **长度限制** | 模型最大上下文 8,192 tokens；每题最多生成 1,024 tokens |
| **计分** | 使用本地 `gsm8k_eval.py` 的答案提取与计分逻辑 |

接受统计由服务端累计计数得到，测试前清零。逐位置接受率的分母为总 draft 轮数，表示一轮中前 \(i\) 个候选 token 连续被接受的比例。

| **指标** | Thinking | Non-thinking |
| --- | --- | --- |
| **GSM8K accuracy** | 76.4% | 96.8916%（1,278 / 1,319） |
| **Invalid rate** | 5.6% | 0% |
| **Draft rounds** | 131,596 | 38,100 |
| **Drafted tokens** | 921,172 | 266,700 |
| **Accepted draft tokens** | 294,863 | 154,796 |
| **MAL** | 3.2407 | 5.0629 |
| **Draft token acceptance rate** | 32.01% | 58.04% |
| **评测总耗时** | 151.655 s | 51.6538 s |
| **Questions/s** | 8.697 | 25.5354 |
| **总输出 tokens** | 426,057 | 191,738 |
| **整体输出吞吐** | 2,809.391 tokens/s | 3,711.9851 tokens/s |

整体输出吞吐为总输出 token 数除以评测总耗时。MAL 按 vLLM 的统计方式计算，每轮包含一个 bonus 或纠正 token：

$$
\mathrm{MAL}=1+\frac{154{,}796}{38{,}100}=5.0629.
$$

| **Proposal 位置** | Thinking 连续接受计数 | Thinking 接受率 | Non-thinking 连续接受计数 | Non-thinking 接受率 |
| --- | --- | --- | --- | --- |
| **1** | 89,759 | 68.21% | 35,616 | 93.48% |
| **2** | 62,427 | 47.44% | 31,325 | 82.22% |
| **3** | 46,642 | 35.44% | 26,918 | 70.65% |
| **4** | 36,031 | 27.38% | 22,033 | 57.83% |
| **5** | 27,063 | 20.57% | 17,209 | 45.17% |
| **6** | 19,551 | 14.86% | 12,817 | 33.64% |
| **7** | 13,390 | 10.18% | 8,878 | 23.30% |

从 thinking 改为 non-thinking 后，MAL 从 3.2407 提高到 5.0629，第一位置接受率从 68.21% 提高到 93.48%。Non-thinking 与训练缓存的设置一致，本次评测中的接受率也更高。两种设置下的 baseline 对照见下文。

Non-thinking 的总输出量更少，整体输出吞吐更高，两者共同缩短了评测耗时。Thinking 在每题最多生成 1,024 tokens 的设置下，无效回答比例为 5.6%。

Temperature 0 时，第一位置接受表示 draft token 与 Target argmax 一致，离线 `reranker/target_accuracy@0` 也使用这一判据。在线评测使用实际生成前缀，离线评测使用参考前缀。

### Non-thinking 下的 baseline 对照

独立 baseline 使用 `qwen38-baseline-lr3e4-step7812`，对应训练 checkpoint `qwen38-flash-next-baseline-lr3e4-step7812`。两组均在 `dspark-v1.4-prefix-reranker` 分支运行，使用相同的 non-thinking、temperature 0、5-shot、长度限制、TP、并发和每轮候选数量。Baseline 按自身配置加载，使用原有 DSpark 的 anchor 位置设置和顺序 Markov 修正。

| **指标** | Baseline | Prefix reranker |
| --- | --- | --- |
| **GSM8K accuracy** | 97.0% | 96.8916% |
| **Invalid rate** | 0% | 0% |
| **Draft rounds** | 38,812 | 38,100 |
| **Drafted tokens** | 271,684 | 266,700 |
| **Accepted draft tokens** | 154,726 | 154,796 |
| **MAL** | 4.9866 | 5.0629 |
| **Draft token acceptance rate** | 56.95% | 58.04% |
| **评测总耗时** | 51.224 s | 51.6538 s |
| **Questions/s** | 25.750 | 25.5354 |
| **总输出 tokens** | 192,395 | 191,738 |
| **整体输出吞吐** | 3,755.970 tokens/s | 3,711.9851 tokens/s |

此前 non-thinking 评测的结果文件路径为 `/public/workspace/dspark/logs/eval-qwen38-flash-next/qwen38-baseline-lr3e4-step7812.json`。

| **Proposal 位置** | Baseline 连续接受计数 | Baseline 接受率 | Reranker 接受率 | 接受率变化（百分点） |
| --- | --- | --- | --- | --- |
| **1** | 36,173 | 93.20% | 93.48% | +0.28 |
| **2** | 32,162 | 82.87% | 82.22% | −0.65 |
| **3** | 27,308 | 70.36% | 70.65% | +0.29 |
| **4** | 22,090 | 56.92% | 57.83% | +0.91 |
| **5** | 16,909 | 43.57% | 45.17% | +1.60 |
| **6** | 12,028 | 30.99% | 33.64% | +2.65 |
| **7** | 8,056 | 20.76% | 23.30% | +2.54 |

与独立 baseline 相比，联合训练 reranker 的 MAL 增加约 0.0763，相对提高约 1.53%。第一位置的接受率基本不变，第二位置略降，提升主要出现在第 5–7 个位置。

两组准确率基本持平。本次测量中，reranker 的总耗时增加约 0.84%，整体输出吞吐降低约 1.17%。接受长度略有增加，推理速度没有提高，耗时差异还需通过重复测量确认。

### Thinking 下的 baseline 对照

Baseline 使用 `qwen38-baseline-lr3e4-step7812`，已训练 3 个 epoch（7,812 步）。本次开启 thinking，在 GSM8K test split 全部 1,319 条问题上进行 5-shot 评测，每轮生成 7 个候选 token。以下与上文 reranker 的 thinking 结果并列。

| **指标** | Baseline | Prefix reranker |
| --- | --- | --- |
| **GSM8K accuracy** | 76.0% | 76.4% |
| **Invalid rate** | 7.4% | 5.6% |
| **Draft rounds** | 133,824 | 131,596 |
| **Drafted tokens** | 936,768 | 921,172 |
| **Accepted draft tokens** | 285,612 | 294,863 |
| **MAL** | 3.1342 | 3.2407 |
| **Draft token acceptance rate** | 30.49% | 32.01% |
| **评测总耗时** | 146.356 s | 151.655 s |
| **Questions/s** | 9.012 | 8.697 |
| **总输出 tokens** | 419,107 | 426,057 |
| **整体输出吞吐** | 2,863.617 tokens/s | 2,809.391 tokens/s |

Baseline 的 MAL 按接受 token 总数和 draft 轮数计算：

$$
\mathrm{MAL}=1+\frac{285{,}612}{133{,}824}=3.1342.
$$

| **Proposal 位置** | Baseline 连续接受计数 | Baseline 接受率 | Reranker 接受率 | 接受率变化（百分点） |
| --- | --- | --- | --- | --- |
| **1** | 88,152 | 65.87% | 68.21% | +2.34 |
| **2** | 61,293 | 45.80% | 47.44% | +1.64 |
| **3** | 45,902 | 34.30% | 35.44% | +1.14 |
| **4** | 35,276 | 26.36% | 27.38% | +1.02 |
| **5** | 25,658 | 19.17% | 20.57% | +1.40 |
| **6** | 17,739 | 13.26% | 14.86% | +1.60 |
| **7** | 11,592 | 8.66% | 10.18% | +1.52 |

Thinking 下，reranker 的 MAL 比 baseline 高约 0.1065，相对提高约 3.40%，各位置的连续接受率均较高。准确率从 76.0% 到 76.4%，无效回答比例从 7.4% 到 5.6%。这两次测量中，reranker 的总耗时高约 3.62%，整体输出吞吐低约 1.89%。

Baseline 从 thinking 改为 non-thinking 后，MAL 从 3.1342 提高到 4.9866，准确率从 76.0% 到 97.0%，无效回答比例从 7.4% 到 0%。两种模型都在与训练缓存一致的 non-thinking 设置下取得更高的接受长度。

本次 thinking 结果文件为 `/public/workspace/dspark/logs/eval-qwen38-flash-next/qwen38-baseline-lr3e4-step7812.json`。

<!--
待汇总的结果表与分析提纲，暂不渲染。

训练与离线验证表按相同 checkpoint step、相同 eval cache 和 anchor 设置补齐。冻结基线实验同时记录起始 checkpoint 与新增训练步数。

| **实验** | 起始 checkpoint | 新增 optimizer steps | Eval \(\tau_{\mathrm{prob}}\) | Candidate recall@16 | Target agreement gain | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| **Baseline** | 从头训练 | 待汇总 | 待汇总 | 待汇总 | — | 相同预算对照 |
| **Engram-Markov** | 从头训练 | 待汇总 | 待汇总 | — | — | 全词表 Markov 修正，无候选重排 |
| **Engram-MASK：门控残差** | 从头训练 | 待汇总 | 待汇总 | — | — | 保留 MASK embedding，逐槽零初始化 tanh 门 |
| **Engram-MASK：直接替换** | 从头训练 | 待汇总 | 待汇总 | — | — | Pre-anchor Engram 直接替换六个 MASK 输入 |
| **Draft Memory** | 从头训练 | 待汇总 | 待汇总 | — | — | 两遍共享 draft，仅第二遍回传，三层读取后缀 memory |
| **Reranker：联合训练** | 从头训练 | 待汇总 | 待汇总 | 待汇总 | 待汇总 | Gain 相对本模型重排前 |
| **Reranker：冻结基线** | Baseline step 2604 | 待汇总 | 待汇总 | 待汇总 | 待汇总 | Backbone 与 Markov 固定 |

| **推理实验** | Proposal 数量 | MAL | Cycle latency | Tokens/s |
| --- | --- | --- | --- | --- |
| **Baseline** | 待记录 | 待汇总 | 待汇总 | 待汇总 |
| **Reranker** | 待记录 | 待汇总 | 待汇总 | 待汇总 |

## 结果分析

### Baseline 收敛

待补：1 epoch、后续 checkpoint 与完整训练计划的离线验证结果。训练曲线采用明确的统计窗口，区分瞬时读数和末段均值。

### 联合训练与冻结基线

待补：联合训练相对独立 baseline 的差异，以及冻结实验相对固定起点的增量。分别讨论候选分布的变化和重排本身的变化。

### 候选覆盖与位置收益

待补：各位置的 candidate recall、target agreement 和连续前缀匹配。区分改正首次错误与改正首次错误之后的 token，后者不一定增加连续接受长度。

### 推理收益

待补：固定评测条件下的真实 MAL 与延迟，检查接受长度的增量是否覆盖新增串行开销。

## 总结

待结果统一汇总后填写。分别记录已观察到的收益、无收益或负收益，不由内部 rerank gain 直接推断完整模型优于 baseline。
-->
