# Speculative Decoding 训练

## 1. 符号定义

设当前 context 为 $x$，词表为 $\mathcal V$，其中 $y\in\mathcal V$ 表示候选下一个 token。

* $p(\cdot|x)$ 和 $q(\cdot|x)$：Target 和 Draft 的完整下一个 token 分布。
* $p(y|x)$ 和 $q(y|x)$：对于具体 token $y$，Target 和 Draft 为其分配的概率。

Draft 常见两种采样方式：

* **Greedy Draft**：选择 Draft 分布中概率最高的 token 作为 proposal，即

    $$d=\arg\max_{y\in\mathcal V}q(y|x)$$

* **Probabilistic Draft**：proposal token 按照 Draft 分布采样，即 $d\sim q(\cdot|x)$。

Draft model 的训练目标取决于 Serving 时的采样和验证方式。不同机制对应的最优训练目标也不同，下面分别讨论。

---

## 2. Serving Objective

### 2.1 Target Greedy + Draft Greedy

Target 使用 greedy decoding：

$$
y_p=\arg\max_y p(y|x)
$$

Draft 也使用 greedy decoding：

$$
y_q=\arg\max_y q(y|x)
$$

如果 validation 直接判断两个 token 是否相同，则 acceptance 为：

$$
A(x)
=
\mathbf{1}
\left[
\arg\max_y q(y|x)
=
\arg\max_y p(y|x)
\right]
$$

因此，该 validation 机制只关心 Draft 和 Target 的 **Top-1 是否一致**。也就是说，达到最优 acceptance 只需要：

$$
\arg\max q=\arg\max p
$$

并不要求完整分布满足$q=p$。


例如：

$$
p=(0.5,0.3,0.2)
$$

两个 Draft 分布分别为：

$$
q_1=(0.49,0.31,0.20)
$$

$$
q_2=(0.99,0.005,0.005)
$$

虽然 $q_1$ 和 $q_2$ 与 Target 分布 $p$ 的距离差别很大，但二者都有：

$$
\arg\max q_1
=
\arg\max q_2
=
\arg\max p
$$

因此在 greedy equality validation 下，两者的单步 acceptance 都是 1。 在 Target greedy + Draft greedy + equality validation 下，distribution matching 是达到最优 acceptance 的充分条件，但不是必要条件。真正的最优条件是 Draft 与 Target 的 Top-1 一致。


### 2.2 Target Sampling + Draft Greedy

Draft 使用 greedy decoding：

$$
d=\arg\max_y q(y|x)
$$

Target 按其分布采样：

$$
Y\sim p(\cdot|x)
$$

如果 validation 要求：

$$
Y=d
$$

则给定 context $x$ 时的期望 acceptance probability 为：

$$
A(x)
=
P(Y=d|x)
=
p(d|x)
$$

代入 Draft proposal：

$$
A(x)
=
p\left(
\arg\max_y q(y|x)
\mid x
\right)
$$

因此，该 validation 机制下的最优 Draft proposal 满足：

$$
\arg\max q
=
\arg\max p
$$

此时，给定 context $x$ 的最大期望 acceptance 为：

$$
A^*(x)
=
\max_y p(y|x)
$$

因此，即使 Draft 完全预测正确 Target 的 Top-1 token，由于 Target 本身采用 sampling，acceptance 也不一定为 1。在 Target sampling + Draft greedy + equality validation 下，distribution matching 是 acceptance-optimal 的充分条件，但不是必要条件。真正的最优条件是 Top-1 agreement。


### 2.3 Target Sampling + Draft Sampling + Equality

Target 按其分布采样：

$$
Y_p\sim p(\cdot|x)
$$

Draft 也按其分布采样：

$$
Y_q\sim q(\cdot|x)
$$

如果两次采样相互独立，并且 validation 直接判断：

$$
Y_p=Y_q
$$

则单步 acceptance probability 为：

$$
A(x)
=
P(Y_p=Y_q|x)
$$

展开得到：

$$
A(x)
=
\sum_y p(y|x)q(y|x)
$$

即：

$$
A(x)=p^\top q
$$

如果直接以 acceptance 为目标，对 Draft 分布 $q$ 进行优化：

$$
\max_q p^\top q
$$

由于该目标关于 $q$ 是线性的，最优策略是把全部概率放在 Target 概率最大的 token 上：

$$
q(y|x)
=
\delta_{y^*}(y),
\qquad
y^*=\arg\max_y p(y|x)
$$

此时最大 acceptance 为：

$$
A^*(x)
=
\max_y p(y|x)
$$

因此，在 independent sampling + equality validation 下，真正的最优 Draft distribution 是把全部概率放在 Target top-1 token 上，Distribution matching 通常不是 acceptance-optimal 的充分条件。


### 2.4 Target Sampling + Draft Sampling + Rejection Sampling

标准 speculative decoding 使用 rejection sampling，而不是直接进行 token equality checking。

Draft 按其分布采样一个 proposal token：

$$
Y\sim q(\cdot|x)
$$

假设本次采样得到的具体 token 为 $y$，则该 token 的接受概率为：

$$
\alpha(y)
=
\min\left(
1,
\frac{p(y|x)}{q(y|x)}
\right)
$$

因此，给定 context $x$ 的平均 acceptance probability 为：

$$
A(x)
=
\sum_y
q(y|x)
\min\left(
1,
\frac{p(y|x)}{q(y|x)}
\right)
$$

化简得到：

$$
A(x)
=
\sum_y
\min\left(
p(y|x),
q(y|x)
\right)
$$

当：

$$
q(\cdot|x)=p(\cdot|x)
$$

时：

$$
A(x)=1
$$

因此，在 rejection sampling validation 下，Distribution matching 与最优 acceptance 是直接对齐的；当且仅当 $q(\cdot|x)=p(\cdot|x)$ 时，可以达到 $A(x)=1$。

### 2.5 小结：两类 Serving Objective

前面的 validation 机制可以粗略归纳为两类。

#### 1. Top-1 Agreement

适用于：

- Target greedy + Draft greedy
- Target sampling + Draft greedy + equality validation

这类机制下，达到最优 acceptance 的条件是：

$$
\arg\max_y q(y|x)
=
\arg\max_y p(y|x)
$$

因此，validation 主要关心 Draft 和 Target 的 **Top-1 是否一致**，并不要求完整 distribution 一致。

#### 2. Distribution Matching

适用于：

- Standard speculative decoding with rejection sampling

此时 acceptance 取决于 Target 和 Draft distribution 的整体重叠程度，最优条件为：

$$
q(\cdot|x)=p(\cdot|x)
$$

因此，这类机制更直接要求 **distribution matching**。
## 3. Training Objectives

### 3.1 Sampled-token CE / SFT

给定 context $x$，Target 按其分布采样一个 token：

$$
Y\sim p(\cdot|x)
$$

将采样得到的 token $Y$ 作为监督信号，训练 Draft：

$$
L_{\mathrm{CE}}
=
-\log q(Y|x)
$$

对 Target sampling 取期望：

$$
\mathbb{E}_{Y\sim p(\cdot|x)}
[L_{\mathrm{CE}}]
=
-\sum_y p(y|x)\log q(y|x)
$$

这就是 Target 分布 $p$ 与 Draft 分布 $q$ 的 cross entropy：

$$
H(p,q)
=
H(p)
+
D_{\mathrm{KL}}(p\Vert q)
$$

由于 $H(p)$ 与 Draft 参数无关，因此最小化期望 CE 等价于最小化：

$$
D_{\mathrm{KL}}(p\Vert q)
$$

理论最优解为：

$$
q(\cdot|x)=p(\cdot|x)
$$

因此，sampled-token CE 的训练目标是完整的 **distribution matching**。

从这个角度看，单个 sampled-token CE：

$$
-\log q(Y|x)
$$

可以看作完整 cross-entropy objective 的 Monte Carlo estimator。


### 3.2 Soft-logit Distillation / KL

如果训练时能够获得 Target 的完整 token distribution：

$$
p(\cdot|x)
$$

则可以直接让 Draft distribution：

$$
q(\cdot|x)
$$

拟合 Target distribution。

常见目标为：

$$
L_{\mathrm{KD}}
=
D_{\mathrm{KL}}(p\Vert q)
$$

即：

$$
L_{\mathrm{KD}}
=
\sum_y
p(y|x)
\log
\frac{p(y|x)}{q(y|x)}
$$

由于 Target distribution $p$ 在训练 Draft 时是固定的，因此该目标同样等价于最小化：

$$
-\sum_y p(y|x)\log q(y|x)
$$

理论最优解为：

$$
q(\cdot|x)=p(\cdot|x)
$$

因此，Soft-logit KD 和 sampled-token CE 的理论最优目标相同，都是 **distribution matching**。

区别在于：

- Sampled-token CE 每个 context 只使用一个 Target sampled token。
- Soft-logit KD 直接利用 Target 的完整 probability distribution。

因此，Soft-logit KD 提供的监督信号更完整，也避免了 sampled-token supervision 带来的采样噪声。


### 3.3 Target Argmax CE：Top-1-oriented Training

如果 serving 时 Draft 使用 greedy decoding，并且 validation 主要关心 Top-1 agreement，可以考虑使用 Target 的 top-1 token 作为监督信号：

$$
y^*
=
\arg\max_y p(y|x)
$$

训练 Draft：

$$
L_{\mathrm{top1}}
=
-\log q(y^*|x)
$$

该 loss 会提高 Draft 对 Target top-1 token 的概率，因此更直接地推动：

$$
\arg\max q
=
\arg\max p
$$

这与 Target sampling + Draft greedy + equality validation 下的最优 serving 条件更加直接对齐。

需要注意，Target-argmax CE 并不是 speculative training 中最常见的标准目标。主流方法通常采用 sampled-token CE 或基于完整 Target distribution 的 distillation。这里将其作为一种 **Top-1-oriented surrogate objective**。

例如，假设 Target distribution 为：

$$
p=(0.40,0.35,0.25)
$$

使用 sampled-token CE 时，理论上会推动：

$$
q\rightarrow p
$$

而使用 Target argmax CE 时，监督 token 始终为 token 1，会推动：

$$
q_1\rightarrow1
$$

例如：

$$
q=(0.9,0.05,0.05)
$$

虽然该分布与 $p$ 并不接近，但 Draft greedy token 仍然是 token 1，因此在 Target sampling + Draft greedy + equality validation 下已经达到最大单步 acceptance：

$$
A^*(x)=p_1=0.4
$$

这说明 distribution matching quality 与 greedy acceptance 并不完全等价。

### 3.4 Hidden-state L1 / L2 Loss

部分 speculative training 方法还会加入 hidden-state matching loss，使 Draft 的中间表示接近 Target。

例如：

$$
L_h
=
\left\|
h_t^{\mathrm{draft}}
-
h_t^{\mathrm{target}}
\right\|_1
$$

或：

$$
L_h
=
\left\|
h_t^{\mathrm{draft}}
-
h_t^{\mathrm{target}}
\right\|_2^2
$$

其中：

- $h_t^{\mathrm{draft}}$：Draft 在位置 $t$ 预测得到的 hidden representation。
- $h_t^{\mathrm{target}}$：Target 对应位置的 hidden representation。

这类 loss 的目标是 **representation matching**，并不直接优化 token acceptance。

它通常作为 auxiliary loss，与 token-level CE / KD 一起使用，例如：

$$
L
=
L_{\mathrm{token}}
+
\lambda L_h
$$

其作用是让 Draft 的内部表示更接近 Target，从而改善后续 token prediction，尤其是在多层或多步 speculative prediction 中减少 hidden-state drift。

因此，Hidden-state L1 / L2 属于辅助训练目标：

$$
h^{\mathrm{draft}}\approx h^{\mathrm{target}}
\quad\Rightarrow\quad
\text{更好的 token prediction}
\quad\Rightarrow\quad
\text{potentially higher acceptance}
$$

需要注意，它与 acceptance 的关系是间接的，最终仍需要结合 token-level objective 进行训练。
### 3.5 Multi-token Objective

假设 Draft 一次预测 $K$ 个 speculative tokens：

$$
d_1,d_2,\ldots,d_K
$$

定义 $a_k$ 为：在前 $k-1$ 个 speculative token 已经被接受的条件下，第 $k$ 个 token 的 acceptance probability。

那么至少连续接受 $k$ 个 token 的概率为：

$$
P(L\ge k)
=
\prod_{j=1}^{k} a_j
$$

其中 $L$ 表示最终连续接受的 speculative token 数量。

因此 expected accepted length 为：

$$
\mathbb{E}[L]
=
\sum_{k=1}^{K}
P(L\ge k)
=
\sum_{k=1}^{K}
\prod_{j=1}^{k} a_j
$$

这比简单平均各位置的 acceptance：

$$
\frac{1}{K}\sum_{k=1}^{K} a_k
$$

更接近真实 serving objective，因为后续 speculative token 只有在前面 token 都被接受后才有机会被使用。

因此，early speculative positions 的 acceptance 对最终 accepted length 影响更大。

实际训练通常采用 multi-position token loss：

$$
L_{\mathrm{MTP}}
=
\sum_{k=1}^{K} w_k L_k
$$

其中：

$$
L_k
=
-\log q_k(y_{t+k}|x_t)
$$

最简单的设置是：

$$
w_k=1
$$

如果希望训练目标更贴近 expected accepted length，也可以考虑：

$$
w_1>w_2>\cdots>w_K
$$

即给予前面的 speculative positions 更高权重。

## 4. Context Distribution 与 Supervision

Speculative training 需要区分两个独立的设计维度：

- **Context distribution**：Draft 在哪些 prefix / state 上训练。
- **Supervision**：给定 context 后，Target 提供什么训练信号。

常见 supervision 包括：

- Target sampled token
- Target argmax token
- Target full distribution / logits
- Target hidden state

这两个维度可以独立组合。例如，可以使用 Target sampling rollout 产生 context，同时使用 Target argmax token 作为训练 label。

### 4.1 Dataset / Teacher Forcing

训练 context 来自 ground-truth sequence：

$$
x_t=(y_1^{GT},\ldots,y_{t-1}^{GT})
$$

这种方式简单、稳定、成本低，但 serving 时的 context 来自模型生成 trajectory，因此可能存在 train-serving distribution shift。

### 4.2 Target Greedy Rollout

Target 使用 greedy decoding 产生 trajectory：

$$
y_t=\arg\max_y p(y|x_t)
$$

得到的 context 更接近 Target 自身生成过程，但 trajectory 相对集中。

如果实际 serving 使用 sampling，例如 $T=1$，则 greedy rollout 无法覆盖大量 sampling trajectory 中可能出现的 context。

### 4.3 Target Sampling Rollout

Target 按其分布采样：

$$
y_t\sim p(\cdot|x_t)
$$

由此产生 sampling trajectory 和对应的训练 context。

如果实际 serving 的 Target 也使用 sampling，例如 $T=1$，这种方式能够更好地覆盖 serving 时可能出现的 context。

需要注意：

> 使用 Target sampling rollout 产生 context，并不意味着 supervision 也必须使用 sampled token。

在同一个 sampled context $x_t$ 上，可以采用不同的 supervision。

#### Sampled-token supervision

直接使用 rollout 中采样得到的 token $y_t$：

$$
L
=
-\log q(y_t|x_t)
$$

其期望目标是：

$$
q(\cdot|x_t)\rightarrow p(\cdot|x_t)
$$

因此更偏向 distribution matching。

#### Target argmax supervision

在同一个 context $x_t$ 上取 Target top-1 token：

$$
y_t^*
=
\arg\max_y p(y|x_t)
$$

然后训练：

$$
L
=
-\log q(y_t^*|x_t)
$$

这种方式保留了 sampling rollout 带来的 context diversity，同时将 supervision 更直接地对齐到 Target Top-1。

因此可以将两种设计理解为：

- **Sampling context + sampled-token label**：偏向 distribution matching。
- **Sampling context + argmax label**：偏向 Top-1 agreement。

### 4.4 Draft / On-policy Rollout

也可以让 Draft 自己产生 training trajectory：

$$
d_t
=
\arg\max_y q(y|x_t)
$$

然后在 Draft 实际访问到的 context $x_t$ 上查询 Target，并构造相应的 token-level 或 representation-level supervision。

这种方式能够让训练 context 更接近 Draft 在 serving 时真正访问的 states，从而缓解 rollout distribution shift。

代价是训练过程更加复杂，因为随着 Draft 参数更新，其访问到的 context distribution 也会不断变化。
## 5. Target Temp=1 + Draft Greedy 下的训练设计

考虑实际 serving 配置：

- Target 使用 temperature=1 sampling
- Draft 使用 greedy decoding
- Validation 使用 token equality
- 不使用 rejection sampling

前面已经得到，该 validation 下的最优条件是：

$$
\arg\max_y q(y|x)
=
\arg\max_y p(y|x)
$$

因此，可以分别考虑 context distribution 和 training objective。

### 5.1 Context Distribution

为了匹配 Target temperature=1 serving 时可能访问的 states，可以使用 Target sampling rollout：

$$
y_t\sim p(\cdot|x_t)
$$

得到相应的 training contexts：

$$
x_t
$$

这里 temperature=1 主要决定 **context 如何产生**，并不要求 sampled token 必须同时作为训练 label。

### 5.2 Token-level Objective

一种标准做法是直接使用 rollout 中 sampled token 进行 CE 训练：

$$
L_{\mathrm{CE}}
=
-\log q(y_t|x_t)
$$

其期望目标是：

$$
q(\cdot|x_t)
\rightarrow
p(\cdot|x_t)
$$

因此属于 distribution matching。

如果更希望训练目标直接对齐 greedy Draft 的 Top-1 acceptance，也可以在相同 context 上取 Target top-1：

$$
y_t^*
=
\arg\max_y p(y|x_t)
$$

并训练：

$$
L_{\mathrm{top1}}
=
-\log q(y_t^*|x_t)
$$

该目标更直接推动：

$$
\arg\max q
=
\arg\max p
$$

但 Target argmax CE 并不是当前 speculative training 中最常见的标准训练方式，更适合作为一种 serving-aligned alternative。

### 5.3 Auxiliary Objectives

如果能够获得 Target 的完整 distribution，可以加入 KD：

$$
L_{\mathrm{KD}}
=
D_{\mathrm{KL}}(p\Vert q)
$$

用于保留 Target distribution 中的概率和 uncertainty 信息。

如果 speculative architecture 需要预测或传递 hidden representation，还可以加入：

$$
L_h
=
\left\|
h^{\mathrm{draft}}
-
h^{\mathrm{target}}
\right\|
$$

作为 representation matching 的 auxiliary loss。

因此，一个可能的组合为：

$$
L
=
L_{\mathrm{top1}}
+
\beta L_{\mathrm{KD}}
+
\gamma L_h
$$

其中 Top-1 CE 对齐 greedy acceptance，KD 和 hidden-state loss 提供额外的 distribution / representation supervision。



## LLM 预训练目标

从零训练的大语言模型，通常是在大规模文本语料上用最大似然估计（MLE）进行优化。

其核心问题是：如何训练一个模型，使其对真实文本延续赋予更高概率。标准的自回归目标通过把语言建模转化为大规模语料上的 repeated next-token prediction 来解决这个问题。

给定从数据分布 $p_\text{data}$ 采样的 token 序列 $(t_1, t_2, \ldots, t_n)$，模型定义如下自回归分解：

$$
p_\theta(t_1, t_2, \ldots, t_n) = \prod_{i=1}^n p_\theta(t_i | t_1, t_2, \ldots, t_{i-1})
$$

训练目标是最大化数据的对数似然，这等价于最小化负对数似然（NLL）：

$$
\mathcal{L}(\theta) = - \mathbb{E}_{(t_1, t_2, \ldots, t_i) \sim p_\text{data}} \left[ \log p_\theta(t_i | t_1, t_2, \ldots, t_{i-1}) \right]
$$

在实践中，这个损失实现为以 one-hot 目标计算的 token 级交叉熵：

$$
\mathcal{L} = -\frac{1}{N} \sum_{\text{tokens}} \log p_\theta(t_i | t_1, t_2, \ldots, t_{i-1})
$$

该目标对应于最小化经验数据分布与模型分布之间的交叉熵；等价地，也是在最小化 $KL(p_\text{data} || p_\theta)$。