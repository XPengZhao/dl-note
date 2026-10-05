# Qwen3.8-Flash-Next DSpark 训练实验

以 Qwen3.8-Flash-Next 为冻结的 Target，训练三层 DSpark draft。各结构变体复用同一份离线数据与 hidden cache，比较其对收敛、候选选择和接受长度的影响。与 [Qwen3-4B 消融](qwen3-4b-dspark-ablation.md) 相比，本组使用不同的 Target、draft 深度和 aux 层，实验数值不作跨模型直接比较。

## 实验设置

各组实验采用下表所列基础训练设置，结构变体、附加目标与参数更新范围在各节分别说明。

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

学习率调度按完整的 26,040 步定义。一个 epoch 的阶段性实验在第 2,604 步停止，仍沿用完整调度的前段。对于从已有 checkpoint 初始化的实验，分别报告预训练步数与新增优化步数。

除冻结基线实验外，aux projector、draft backbone、Markov、confidence head 与各变体的新增参数联合优化。Target 相关参数和离线缓存保持固定。Engram 与 Draft Memory 沿用 baseline 损失，Prefix reranker 的附加排序目标和冻结策略在对应小节定义。

<!-- 待补：训练样本数、regen 参数、GPU 型号、cache 版本，以及各 run 的最终配置。 -->

## 评估口径

### 离线指标

在同一参考前缀下，记 teacher 分布为 \(p_i\)，draft 分布为 \(q_i\)。分布重叠为

$$
a_i=\sum_v\min\bigl(p_i(v),q_i(v)\bigr)
=1-\frac12\lVert p_i-q_i\rVert_1.
$$

每个有效 block 的概率接受长度定义为

$$
\tau_{\mathrm{prob}}
=1+\sum_{i=0}^{6}\prod_{j=0}^{i}a_j,
$$

不足 7 个有效位置时，无效位置的 \(a_i\) 置零，\(\tau_{\mathrm{prob}}\) 对有效 block 聚合。各位置的 `accept_rate@i` 按有效 token 数聚合，不使用损失的位置衰减权重。这些指标在缓存的参考前缀上以 teacher forcing 计算，区别于多轮推理中测得的 mean acceptance length（MAL）。

对 reranker，`train/tau_probabilistic` 和 `eval/tau_probabilistic` 使用**重排后**的分布：在 top-16 候选上对修正分数做 softmax，候选之外的概率置零。Teacher 仍使用原始全词表概率，不在候选集合内重新归一化。重排前的 DSpark 分布则在全词表上归一化。

| **指标** | 含义 |
| --- | --- |
| **`rerank_tau_gain`** | 同一 checkpoint、同一批 anchor 上，重排后与重排前的 \(\tau_{\mathrm{prob}}\) 之差 |
| **`rerank_target_agreement_gain`** | 重排前后选中 teacher argmax 的比例之差；0.02 表示增加 2 个百分点 |
| **`reranker/candidate_recall`** | Teacher argmax 落在 Markov top-16 内的比例 |
| **`rerank_teacher_forced_prefix_gain`** | 与参考 token 序列连续匹配的前缀长度变化，仍在 teacher-forced 输入下计算 |

`rerank_tau_gain` 衡量同一模型内部的重排增量，包含残差修正、top-16 截断与重新归一化的共同影响。完整方法的收益则以相同训练预算下的独立 baseline 为参照，二者分别报告。

### 推理指标

推理指标包括真实 Target verification 下的 MAL、draft–verify cycle latency 和 tokens/s。对照条件固定为相同的测试集、采样方式、并发数、TP、最大长度和 proposal 数量。

当前 Engram 的实际前缀查表与服务链路尚未完成集成及在线评估，查表与 Target 计算重叠的调度也未验证。Draft Memory 尚未接入 vLLM 多轮推理。Prefix reranker 的采样接口仅支持 greedy，其随机采样收益不能由离线概率重叠推断。

<!-- 待补：离线 eval 的固定配置、训练末段统计窗口、线上启动参数和评测命令。 -->

## 实验分组

| **实验** | 初始化与更新范围 | 比较的问题 |
| --- | --- | --- |
| **Baseline** | 从头训练原 DSpark | 原结构随训练预算增加的收敛水平 |
| **Engram-Markov** | 从头联合训练 draft 与 Engram 门控投影 | Markov head 的 n-gram 条件对接受长度的影响 |
| **Engram-MASK：门控残差** | 从头联合训练 draft、输入投影与逐槽标量门 | 保留 MASK embedding 的门控注入效果 |
| **Engram-MASK：直接替换** | 从头联合训练 draft 与输入投影 | 内容相关输入替代固定 MASK 的效果 |
| **Draft Memory** | 从头训练 draft 与 memory adapter，仅第二阶段回传 | 上一轮 draft 状态对新 anchor 下续写的作用 |
| **Prefix reranker：联合训练** | 从头联合训练 draft 与 reranker | 前缀条件化重排相对独立 baseline 的收益 |
| **Prefix reranker：冻结基线** | 加载 baseline step 2604，仅训练新加的 reranker | 固定 proposal 能力后，重排模块能提供多少增量 |

<!-- 待补：Carryover teacher forcing 的实验设置与结果。 -->

## Baseline 架构

Baseline 采用标准DSpark结构。三层 draft backbone一次生成各预测位置的hidden states，LM head 将其映射为初始 logits。Markov head 根据前驱 token 添加logits bias，confidence head 独立估计对应位置的分布重叠率。

![DSpark baseline：左为并行 draft 与顺序选择，右为首个预测位置的 Markov 修正及 confidence 分支](../../assets/images/qwen3.8-flash-next-dspark/dspark-baseline.svg){ width="100%" }

图左以 D 为 anchor，展示E – I的生成，虚线表示相邻位置的选择依赖。右图展开预测E时的 Markov 修正与 confidence 分支。实际训练时 block size 为 7，包含一个 anchor 和六个MASK。

### Dspark Backbone

设 block size 为 \(B\)，token embedding 为 \(E(\cdot)\)。对于前缀 `[... A B C]` 与新 anchor D，draft 输入为

$$
X=[E(D),\underbrace{E(\mathrm{MASK}),\ldots,E(\mathrm{MASK})}_{B-1\text{ 个槽}}],
\qquad B=7.
$$

MASK 槽共享输入 embedding，并使用不同的位置编号。anchor 槽对应 \(h_1\)，用于预测 E，后续槽依次预测 F – K。对 anchor 之前的每个位置 \(j\)，拼接 Target 第 45、46、47 层的隐藏状态，经投影与归一化得到 context 特征：

$$
c_j=\operatorname{RMSNorm}\!\left(
W_{\mathrm{aux}}[H_j^{45};H_j^{46};H_j^{47}]
\right),
\qquad
W_{\mathrm{aux}}:\mathbb R^{7680}\rightarrow\mathbb R^{2560}.
$$

\(W_{\mathrm{aux}}\) 是可训练的无偏置投影。图中的 Target context 截止 C，anchor D 仅通过自身 embedding 输入，不提供其对应位置及未来位置的 Target hidden。

三层 backbone 均采用 GQA，每层包含 24 个 query heads 和 2 个 K/V heads，head dimension 为 256，即每 12 个 query heads 共享一组 K/V。三层均配置为 full attention，不启用 SWA。对于一个 anchor block，每个 query 可读取 anchor 之前全部有效的 Target context，以及自身 block 中的所有输入槽。

每层的 query 来自当前 draft 隐藏状态，context 特征与当前 block 隐藏状态分别经过该层的 K/V 投影，沿序列维拼接后在同一次 attention 中读取。三层共享 \(c_j\)，但各层的 Q/K/V 投影参数独立。Draft 状态通过 attention、残差连接和 MLP 逐层更新，最终经 RMSNorm 得到 \(h_1,\ldots,h_B\)，再由冻结的 LM head 并行生成全词表初始 logits \(\ell_1^0,\ldots,\ell_B^0\)。



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

\(W_1\) 与 \(W_2\) 均参与训练。偏置 \(b_i\) 仅显式依赖前驱 token，不读取 \(h_i\)，上下文信息由 \(\ell_i^0\) 提供。图右的第一步对应 \(m_1=W_1[D]\) 与 \(b_1=W_2m_1\)。Greedy 选择为 \(\hat t_i=\arg\max_v\ell_i(v)\)，图示结果为 E。该修正作用于全词表，不限制候选集合。

训练采用 teacher forcing，所有位置的 Markov 修正可并行计算。推理使用实际选出的前驱 token，依次完成查表、logits 修正与 token 选择。图左的虚线仅连接相邻位置的选择过程，backbone 隐藏状态在整轮内保持不变。

### Confidence head 与训练目标

Confidence 分支读取 draft 隐藏状态 \(h_i\) 与词表投影之前的 Markov 表示 \(m_i\)，通过线性层与 sigmoid 得到预测值：

$$
s_i=w_c^\top[h_i;m_i]+b_c,
\qquad
\hat a_i=\sigma(s_i).
$$

输入维度为 \(2560+256=2816\)。监督目标为同一参考前缀下 teacher 分布 \(p_i\) 与修正后 draft 分布 \(q_i\) 的重叠率 \(a_i=\sum_v\min(p_i(v),q_i(v))\)。该连续标签停止梯度，以二元交叉熵优化 confidence head。图右的 \(\hat a_1\) 是独立辅助输出，不参与 E 的选择，也不改变 CE、L1 的位置权重。

基础训练目标为 \(0.1L_{\mathrm{CE}}+0.9L_{\mathrm{L1}}+L_{\mathrm{conf}}\)。CE 监督记录的 next token，L1 对齐 teacher 与 draft 的全词表分布，confidence loss 拟合分布重叠率。三项损失均按有效 response 位置及位置衰减权重聚合。

离线 aux cache 提供模型输入，Target 最后一层的隐藏状态经冻结 LM head 生成 teacher 分布，仅用于监督与评估。

## Engram 增强 Markov head

Engram-Markov 在前驱 token 的低秩表示中引入 n-gram 特征，并由当前 draft 隐藏状态控制注入强度。Backbone 与 MASK 输入保持不变。

![Engram 增强 Markov head：左为整块生成，右为以 D 为 anchor 预测 E 的第一步](../../assets/images/qwen3.8-flash-next-dspark/engram-markov.svg){ width="100%" }

图右展开以 D 为 anchor 预测 E 的第一步。Engram lookup 读取截至 D 的前缀 `[... A B C D]`，返回特征 \(e(D)\)，与 \(W_1[D]\) 融合后修正 logits。

### 门控与特征注入

从 1 开始编号生成位置，令 \(t_0\) 为 anchor，\(h_i\in\mathbb R^{2560}\) 和 \(\ell_i^0\) 分别为第 \(i\) 个预测位置的 draft hidden 与初始 logits。记 \(e_{i-1}\in\mathbb R^{2560}\) 为截至前驱 token \(t_{i-1}\) 的前缀查表得到的 raw Engram 特征，即 n-gram lookup 拼接后、尚未经过 K/V 投影的表示。

Draft hidden 提供 query，Engram 特征提供 key 和 value。记 \(N\) 为无可训练 affine 参数的 RMSNorm，Markov rank 为 \(r=256\)，则

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

\(g_i\) 是每个位置一个标量，用来缩放该位置的 Engram value。这是单个 query–key 对的门控，不包含跨位置的 softmax attention。增强后的 Markov 表示和最终 logits 为

$$
m_i=W_1[t_{i-1}]+g_i\mathbf v_i,
\qquad
\ell_i=\ell_i^0+W_2m_i.
$$

其中 \(W_1[t_{i-1}]\) 保留原有 unigram 路径，\(W_2\) 将融合后的 256 维表示映射为全词表偏置。Confidence head 同样读取增强后的 \(m_i\)。

三个无 bias 投影 \(W_Q,W_K,W_V\) 均为 \(2560\rightarrow256\)，新增 \(3\times2560\times256=1{,}966{,}080\) 个 draft 参数。\(W_V\) 零初始化，使初始 Engram 残差为零。在 backbone 和原 Markov 参数相同的条件下，初始输出与 vanilla Markov 一致。

### 训练与推理

离线训练读取与参考序列对齐的 Engram 特征。预测 \(t_i\) 时使用前驱位置 \(t_{i-1}\) 的缓存，各位置的门控与 Markov 修正可并行计算。

推理时，lookup 随实际生成前缀逐 token 更新：读取 D 位置的特征预测 E，再基于包含 E 的前缀查表以预测 F。该路径不能沿用分叉后的参考序列缓存。上述参数量仅计入 draft 投影，Engram 表、特征传输和逐 token 查表构成额外存储与时延开销。

## Engram 门控增强 MASK 输入

门控 Engram-MASK 在原 MASK embedding 上加入前缀特征残差，以各槽的可训练标量控制注入强度。门值零初始化，后续仍使用 vanilla Markov head。

![Engram 门控增强 MASK：共享的 pre-anchor 特征经逐槽 tanh 门缩放后，与原 MASK embedding 相加](../../assets/images/qwen3.8-flash-next-dspark/engram-gate-mask.svg){ width="100%" }

图中 anchor 为 D，Engram 特征来自截至 C 的前缀 `[... A B C]`。特征投影后由各 MASK 槽共享，anchor 保留原 token embedding。

### 门控残差

令 \(a\) 为 anchor 在原序列中的位置，\(e_{a-1}\in\mathbb R^{2560}\) 为 pre-anchor 的 raw Engram 特征。\(P\in\mathbb R^{2560\times2560}\) 是无 bias 的可训练投影，RMSNorm 位于投影之后，且不含可训练 affine 参数。Block 输入为

$$
\begin{aligned}
z_a&=\operatorname{RMSNorm}(Pe_{a-1}),\\
u_{a,0}&=E(x_a),\\
u_{a,j}&=E(\mathrm{MASK})+\tanh(\theta_j)\,z_a,
\qquad j=1,\ldots,6,\\
X_a&=[u_{a,0},u_{a,1},\ldots,u_{a,6}].
\end{aligned}
$$

\(\theta_j\) 是第 \(j\) 个相对 MASK 槽的可训练标量，跨样本和 block 共享。门值不随输入内容变化，特征依赖由 \(z_a\) 提供。\(\tanh(\theta_j)\in(-1,1)\) 允许正向或反向缩放残差，区别于 Engram-Markov 的内容相关门控。

所有 \(\theta_j\) 均零初始化，因此初始 \(u_{a,j}=E(\mathrm{MASK})\)。在原有 draft 权重相同的条件下，初始前向与 baseline 一致。投影 \(P\) 随机初始化，但门值为零时，它从任务损失获得的梯度也为零。门值离开零后，投影才开始接收该路径的任务梯度。新增参数为 \(2560^2+6=6{,}553{,}606\) 个，包括一个共享投影和六个标量门。

### 训练与推理

训练对每个 anchor 读取 \(a-1\) 位置的 Engram 缓存，缺少有效前驱或 block 无效时置零。六个槽共享投影结果，保留各自的位置编号、RoPE 与块内双向 attention。投影和标量门由基础任务损失学习，不增加独立门控目标。

推理每轮只读取一次 pre-anchor 特征，经投影与门控后输入 draft，整块生成期间保持不变。该特征在新 anchor 产生前即可确定，使 host lookup 具备与 Target 计算重叠的条件。

## Engram 替换 MASK 输入

直接替换版本移除 MASK embedding 与标量门，以 pre-anchor Engram 的投影作为各槽输入，其余结构保持 baseline 配置。

![Engram 替换 MASK：pre-anchor 特征经投影和归一化后填充并行 draft 的输入槽](../../assets/images/qwen3.8-flash-next-dspark/engram-fix-mask.svg){ width="100%" }

图中以 D 为 anchor，读取截至 C 的特征 \(e(C)\)。D 通过原 token embedding 路径输入，后续槽由同一 Engram 投影填充。

### 输入替换

令 \(a\) 为 anchor 在原序列中的位置，\(e_{a-1}\in\mathbb R^{2560}\) 为该位置之前的 raw Engram 特征。一个无 bias 的线性投影 \(P\in\mathbb R^{2560\times2560}\) 将其映射到 draft 输入空间，随后进行无可训练 affine 参数的 RMSNorm：

$$
z_a=\operatorname{RMSNorm}(Pe_{a-1}),
\qquad
X_a=[E(x_a),\underbrace{z_a,\ldots,z_a}_{6\text{ 个槽}}].
$$

六个槽共享 \(z_a\)，位置编号为 \(a+1,\ldots,a+6\)，沿用 RoPE 和块内双向 attention。

新增参数为投影矩阵 \(P\)，共 \(2560^2=6{,}553{,}600\) 个。该投影随机初始化并参与训练，因此初始模型与 baseline 不等价。

### 训练与推理

特征选取与门控版一致：训练读取 \(a-1\) 位置的缓存，缺少有效前驱时置零。新增投影由基础任务损失直接优化。

推理每轮查表并投影一次，将 \(z_a\) 广播至所有替换槽，不随块内新选 token 更新。相较门控版，该路径省去逐槽缩放与残差相加，仍保留查表、特征传输、投影和归一化开销。

## Draft Memory

Draft Memory 复用上一轮 draft 对后续位置的隐藏表示，为新 anchor 下的并行预测提供跨轮条件。Memory 来源于 draft，而非 Target 在被拒绝候选路径上的状态。训练采用共享参数的两阶段前向过程，复用已有 Target hidden cache，无需额外 Target rollout 或 Engram 特征。

![DSpark Draft Memory：第一遍筛选后缀 hidden，第二遍在各层 attention 中读取 memory](../../assets/images/qwen3.8-flash-next-dspark/draft-memory.svg){ width="100%" }

### 两阶段训练

第一阶段在无 memory 条件下执行不保留梯度的 draft 前向计算，取得最终 RMSNorm 后的隐藏表示。LM head 与 Markov head 在参考前驱 token 条件下计算 argmax，并确定其与记录序列的首次不匹配位置。在此位置之前，参考前驱与 greedy draft 已选 token 一致，因此该过程与自回归 greedy 生成得到相同的首次参考错误。后续预测不参与 anchor 转移。

图中旧 anchor 为 C，Target context 截止 B，参考续写为 D、E、F、G、H。若前两个预测匹配，第三个预测为 X，则以参考 token F 为新 anchor，将 context 扩展至 E。对应已匹配位置及首次不匹配位置的 \(h_1,h_2,h_3\) 被移除，仅保留 \(h_4,h_5\)，供第二阶段预测 G、H、I、J、K。

该转移依据记录序列模拟拒绝，而非实时 Target verification。Memory 在 Markov 选择之前由旧 anchor 与 MASK 并行计算，不包含沿错误 token X 重新计算的状态，也不附加候选 token embedding 或拒绝标记。

第二阶段以新 anchor、对应的正确 Target context 和筛选后的 memory 为输入，采用 teacher forcing 计算基础任务损失。梯度仅通过该阶段回传，联合更新共享 draft 与 memory adapter。

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

\(P\in\mathbb R^{2560\times2560}\) 随机初始化，\(\gamma\) 为归一化尺度参数。Adapter 共增加 \(2560^2+2560=6{,}556{,}160\) 个参数，不引入门控。停止梯度作用于第一阶段的隐藏表示，\(P\) 与 \(\gamma\) 由第二阶段的损失优化。

Memory 的 RoPE 位置采用预测位置 \(a+i\)，即旧 query 位置编号加一。因此，首条保留状态对齐新 anchor 后的第一个预测位置 \(a'+1\)。图中 \(h_4,h_5\) 分别对齐 G、H。

Adapter 只计算一次，得到的同一组 \(M\) 供三层 draft 使用。每一层的 query 仍来自当前 draft states，K/V 则由正确 context、memory 和当前 draft states 拼接后，通过该层原有的投影得到。省略归一化与 RoPE 后，可写为

$$
\operatorname{Attn}_{\ell}
\left(
Q_{\ell}(X_{\ell}),
K_{\ell}([H_{\mathrm{ctx}};M;X_{\ell}]),
V_{\ell}([H_{\mathrm{ctx}};M;X_{\ell}])
\right).
$$

其中 \(H_{\mathrm{ctx}}\) 为投影后的正确前缀特征，\(X_\ell\) 为当前层的 draft states。Memory 作为静态 K/V 来源，与 context 和当前 block 共享同一次 attention 归一化，不增加独立 cross-attention。可见性约束限定为本 anchor 的有效 memory、当前 draft block 与 anchor 之前的正确 context，排除其他 block 及未来 Target states。

若整块全部匹配，则转移至记录序列的 bonus 位置，memory 为空。缺少后续监督或跨越回答边界的转移不参与训练。本组实验保留全部有效后缀，不使用 memory dropout。

### 训练与推理

相对 baseline，训练增加一次无梯度 draft 前向计算、首次错误定位所需的输出头计算，以及 memory 投影与扩展 attention 的开销。虽然不增加 Target 前向计算，相同步数仍不对应相同计算预算，训练效率需结合实际耗时评估。

在线推理需按请求保存上一轮 draft 的最终隐藏表示。Target verification 后，根据接受长度移除已接受状态及首次拒绝状态，将剩余后缀按预测位置对齐至下一轮。各层通过自身投影生成 memory K/V。

训练与推理仍存在两项分布差异：拒绝位置由参考序列而非实时 Target 验证确定，训练 memory 仅来自无 memory 的第一阶段，而连续推理中的源隐藏状态可能已包含更早一轮 memory 的影响。

## Prefix reranker

### 设计动机

Vanilla Markov 的偏置只显式依赖单个前驱 token，缺少对当前 block 内更长已选前缀的编码。

Prefix reranker 通过块内因果 Transformer 编码 anchor 与已选 token，再结合当前 draft 隐藏状态，对 Markov top-16 候选进行残差打分。

![DSpark prefix reranker：左为整块生成，右为第三个位置的候选重排](../../assets/images/qwen3.8-flash-next-dspark/prefix-reranker.svg){ width="100%" }

图右展开第三个预测位置：前缀 `[D, E, F]` 编码为 \(r_3\)，与 \(h_3\) 和候选特征共同生成修正分数。图示候选 G 在残差修正后超过 H。

### 候选与前缀表示

设 \(t_0\) 为 anchor，\(t_i\) 为第 \(i\) 个生成位置的 token。训练时使用参考 token，推理时使用当前轮已选 token。记 backbone 输出为 \(h_i\in\mathbb{R}^{2560}\)，冻结 LM head 生成的初始 logits 为 \(\ell_i^0\)。

复用 Markov head 的两张秩为 256 的 token 表 \(E_{\mathrm{in}}\) 与 \(E_{\mathrm{out}}\)，得到原 DSpark 分数

$$
\ell_i(v)=\ell_i^0(v)+E_{\mathrm{in}}(t_{i-1})^\top E_{\mathrm{out}}(v),
\qquad
C_i=\operatorname{Top16}(\ell_i).
$$

候选集合由 Markov 修正后的分数确定。因此，teacher argmax 的候选覆盖率构成重排选择准确率的上界。

Prefix Transformer 的输入为 \([t_0,\ldots,t_{i-1}]\)。各 token 经共享的 \(E_{\mathrm{in}}\) 查表、线性投影并加上可学习的块内位置 embedding。一个 pre-norm 因果 Transformer 层生成前缀表示 \(r_i\)。其宽度为 256，包含 4 个 attention heads，FFN 为 \(256\rightarrow1024\rightarrow256\)，使用 SiLU，输出经过 LayerNorm。

前缀编码范围限于当前 block，历史 context 由 \(h_i\) 提供。

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

\(W_o\) 零初始化，并在并列分数下优先保留原 DSpark 的选择，从而保证初始 greedy 选择一致。候选内归一化与全词表归一化仍有差异，因此这一初始化不保证概率重叠指标相同。

### 训练目标与更新范围

训练沿用现有 hidden cache。每个 block 的前驱输入右移一位，以 anchor 开头，其后为参考序列中的 token。Prefix Transformer 使用 causal mask，在一次 forward 中并行编码所有位置。

记 \(p_i\) 为 cached reference prefix 上的 teacher 分布，重排监督标签为

$$
v_i^\star=\arg\max_v p_i(v).
$$

该标签可能不同于参考序列中实际记录的 token。记 \(\mathcal H\) 为有效且 \(v_i^\star\in C_i\) 的位置集合，只在这些位置计算候选内的排序 CE：

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

原目标仍监督全部有效位置，包括 teacher argmax 未进入 top-16 的位置。联合训练中，排序损失也会更新 backbone 和共享 Markov 参数，因此内部的“重排前模型”不等同于独立训练的 baseline。

冻结基线实验以训练 2,604 步的 baseline 为起点，固定 backbone、Markov 和 confidence head，仅优化 reranker 的新增参数，并重新初始化优化器与学习率调度。曲线横轴采用累计训练步数：预训练步数加上 reranker 的新增优化步数。横轴偏移不影响后者的学习率调度。

### 推理开销

Backbone 每轮只运行一次。第 \(i\) 个位置利用上一个已选 token 更新 Prefix Transformer 的本地 K/V，得到 \(r_i\)，并计算 Markov 修正后的 top-16 候选及其残差分数。选出的 token 进入下一位置，整块候选最后交给 Target 验证。

Prefix K/V 在当前 block 内增长，每轮重新清空。新增计算位于串行选择路径，包括前缀编码、候选筛选与残差打分，不增加 Target rollout。

## 结果总览

已完成 Prefix reranker 与独立 baseline 在 step 7812 的在线 GSM8K 全集评测，并记录 reranker 的 thinking 与 non-thinking 对照。其余结构的在线结果待补。

### Prefix reranker 在线评测

2026 年 10 月 5 日，使用 `qwen38-prefix-reranker-step7812` 在 vLLM 中进行多轮 draft–verify 推理。测试集为 GSM8K test split 全部 1,319 条问题，区别于训练过程中使用的 256 条重新生成回复及 hidden cache 的离线验证集。

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

评测期间的接受统计由服务端累计计数获得，测试前计数为零。逐位置接受率的分母均为 draft rounds，表示一轮中前 \(i\) 个 proposal 连续被接受的比例，不是前 \(i-1\) 个已接受条件下的条件接受率。

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

Thinking 的 accuracy 与 invalid rate 保留原始日志的显示精度，未据此反推精确题数。

上述吞吐由总输出 token 数除以评测总耗时计算，包含并发与排队影响，不表示单请求 decode 速度。MAL 沿用 vLLM 的计数口径，包含每轮一个 bonus／纠正 token：

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

此前开启 thinking 的在线评测中，MAL 为 3.2407，第一位置接受率为 68.21%。改为 non-thinking 后，二者分别提高至 5.0629 与 93.48%。该变化表明 thinking 设置造成的生成分布差异显著影响接受统计。两次评测的生成轨迹不同，这一对比不能替代相同前缀下的端到端数值对齐，也不能用于估计 reranker 相对 baseline 的增量。

Non-thinking 的总输出量也低于 thinking，因此总耗时下降同时包含输出长度与整体输出吞吐的变化，不能全部归因于 MAL 提高。Thinking 的无效回答尚未按生成截断、答案格式或提取失败分类，当前准确率差异仅反映该长度限制与计分协议下的评测结果。

在 temperature 0 下，在线第一位置的接受判据为 draft token 与 Target argmax 一致，与离线 `reranker/target_accuracy@0` 的判据相同。在线生成与离线参考序列的前缀和 anchor 分布仍有差异，两个指标不要求数值相等。

### Non-thinking 下的 baseline 对照

独立 baseline 使用 `qwen38-baseline-lr3e4-step7812`，对应训练 checkpoint `qwen38-flash-next-baseline-lr3e4-step7812`。与 reranker 使用相同的 non-thinking、temperature 0、5-shot、长度限制、TP、并发及 proposal 设置，均在 `dspark-v1.4-prefix-reranker` 推理分支运行。Baseline 按自身配置加载，不启用 reranker，保留 DSpark 的 anchor 对齐与顺序 Markov 修正。

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

Baseline accuracy 沿用日志中三位小数的显示精度。结果文件为 `/public/workspace/dspark/logs/eval-qwen38-flash-next/qwen38-baseline-lr3e4-step7812.json`。

| **Proposal 位置** | Baseline 连续接受计数 | Baseline 接受率 | Reranker 接受率 | 接受率变化（百分点） |
| --- | --- | --- | --- | --- |
| **1** | 36,173 | 93.20% | 93.48% | +0.28 |
| **2** | 32,162 | 82.87% | 82.22% | −0.65 |
| **3** | 27,308 | 70.36% | 70.65% | +0.29 |
| **4** | 22,090 | 56.92% | 57.83% | +0.91 |
| **5** | 16,909 | 43.57% | 45.17% | +1.60 |
| **6** | 12,028 | 30.99% | 33.64% | +2.65 |
| **7** | 8,056 | 20.76% | 23.30% | +2.54 |

在这次全集对照中，reranker 的 MAL 增加约 0.0763，相对提高约 1.53%。第一位置的接受率基本不变，第二位置略降，较大的增量出现在第 5–7 个位置。该结果衡量联合训练方法相对独立 baseline 的整体差异，不等同于同一 checkpoint 内关闭重排得到的模块增量。

两组准确率基本持平。Reranker 的总耗时增加约 0.84%，整体输出吞吐降低约 1.17%，本次评测未观察到端到端推理加速。该幅度较小，单次测量尚不能区分新增串行计算开销与运行波动。当前结果支持小幅接受长度收益，但不足以证明净速度收益。

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
