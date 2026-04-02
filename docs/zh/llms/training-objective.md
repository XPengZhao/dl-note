# 训练目标

本页记录大语言模型最基础的自回归训练目标。它与 serving 运行时是不同层面的问题，但作为背景知识很有必要，因为许多推理时的量最终都建立在同一套 token 级概率分解之上。

把这页放在 AI Infra 里，并不是因为做 serving 必须深入研究训练，而是因为推理时很多最基本的对象都继承自预训练目标。logits、token 概率、next-token prediction 和 sampling 的含义，最终都建立在同一套自回归概率分解上。

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
