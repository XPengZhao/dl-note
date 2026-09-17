# Training Objective

This page records the basic autoregressive objective used in pre-training large language models. It sits somewhat outside serving and systems runtime behavior, but it remains useful background because many inference-time quantities are defined with respect to the same token-level factorization. Terms such as logits, token probabilities, next-token prediction, and sampling all inherit their meaning from the autoregressive objective.

## Pre-Training Objective of LLMs

A large language model trained from scratch is optimized using maximum likelihood estimation (MLE) over a large text corpus.

The core problem is to train one model that assigns high probability to realistic continuations of text. The standard autoregressive objective solves this by reducing language modeling to repeated next-token prediction over large corpora.

Given a sequence of tokens $(t_1, t_2, \ldots, t_n)$ sampled from the data distribution $p_\text{data}$, the model defines an autoregressive factorization:

$$
p_\theta(t_1, t_2, \ldots, t_n) = \prod_{i=1}^n p_\theta(t_i | t_1, t_2, \ldots, t_{i-1})
$$

The training objective is to maximize the log-likelihood of the data, which is equivalent to minimizing the negative log-likelihood (NLL):

$$
\mathcal{L}(\theta) = - \mathbb{E}_{(t_1, t_2, \ldots, t_i) \sim p_\text{data}} \left[ \log p_\theta(t_i | t_1, t_2, \ldots, t_{i-1}) \right]
$$

In practice, this loss is implemented as token-level cross-entropy with one-hot targets:

$$
\mathcal{L} = -\frac{1}{N} \sum_{\text{tokens}} \log p_\theta(t_i | t_1, t_2, \ldots, t_{i-1})
$$

This objective corresponds to minimizing the cross-entropy between the empirical data distribution and the model distribution, or equivalently, minimizing $KL(p_\text{data} || p_\theta)$.
