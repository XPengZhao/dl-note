## KV Cache

### Memory cost

The KV cache size is calculated as:

$$
\text{KV size} = 2 \times \text{num layers} \times \text{num heads} \times \text{seq len} \times \text{head dim}
$$

Assume:

- 32 layers, 80 heads, head_dim = 128, seq_len = 4K.
- FP16 KV cache: 2 × 32 × 80 × 4K × 128 × 2 bytes = **~5.2 GB** per sample.
- C8 (int8 KV cache): **half that → ~2.6 GB** per sample.

## Attention

In transformer models, the attention module computes:

$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

Every token produces:

- **Q (Query)** — derived from the current token’s hidden state
- **K (Key)**, **V (Value)** — derived from the context (previous tokens) and stored in the **KV cache**

The difference between **MHA**, **MQA**, and **GQA** lies in **how many Key/Value projections** are used relative to the number of Query heads.

### **MHA — Multi-Head Attention**

Each attention head has its own set of projections:

$$
Q_i=X W_{Q, i}, \quad K_i=X W_{K, i}, \quad V_i=X W_{V, i}
$$

- Q heads = K heads = V heads

### **MQA — Multi-Query Attention**

All heads **share one** K and V projection:

$$
K_{\text {shared }}=X W_K, \quad V_{\text {shared }}=X W_V
$$

- **Multiple Q heads**, but **only one pair of shared K/V**.
- During inference, the same K/V are reused for all heads.
- KV cache size reduced by ~#heads×
- **LLaMA 2 7B**, **Mistral**, and most optimized LLMs use MQA to cut memory usage.

### **GQA — Grouped-Query Attention**

Queries are split into groups, and each group shares its own K/V:

$$
\text { Group } g: \quad Q_{g, *}, K_g, V_g
$$

- Suppose you have 8 query heads but only 2 groups → 4 Q heads per group share one K/V.
- **K/V heads < Q heads**.
- **LLaMA 2 70B**, **Mixtral**, **GPT-J-6B**, etc. use GQA for better memory efficiency.

## Sampling
### Temperature

Temperature $T$ controls "sharpness" of the probability distribution. Given model logits $z_i$, temperature sampling rescales them before softmax:

$$
p_i=\frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

- $T<1$ Sharper (more peaked). Model becomes confident, less random. $T>1$ Flatter (more uniform). Model becomes exploratory, more diverse.
- Temperature keeps the order of logits because dividing by a positive number does not change ranking (rank-preserving). If you use top-k = 1 (i.e., greedy), temperature has no effect at all.

### Top-k Sampling

After computing probabilities (after temperature), sort tokens by probability, and keep only the k highest. Then renormalize $p_i^{\prime}$ to get final distribution for sampling.

$$
\begin{aligned}
S_k & =\text { top-k tokens by } p_i \\
p_i^{\prime} & = \begin{cases}\sum_{j \in s_k} p_j & i \in S_k \\
0 & \text { otherwise }\end{cases}
\end{aligned}
$$

Then you sample from that truncated distribution.

- Small k (e.g., 1–10): deterministic and safe (model never picks rare words).
- Large k (e.g., 50–100): more variety but higher risk of off-topic words.

### Top-p (nucleus) Sampling

“nucleus” literally means core, center, or the essential central part of something. Instead of a fixed k, we choose the smallest set of tokens whose total probability $\geq p$.

$$
\begin{aligned}
S_p&=\left\{i: \sum_{j \in S_p} p_j \geq p\right\} \\
p_i^{\prime}&= \begin{cases}\sum_{j \in s_p p_j}^{p_i} & i \in S_p \\
0 & \text { otherwise }\end{cases}
\end{aligned}
$$

**Key difference vs top-k:**

Top-p adapts to uncertainty. Top-k is about rank (k highest scores). Top-p is about mass (the core mass of the distribution).

- If the model is confident (one token dominates), very few tokens are kept.
- If it’s uncertain (many tokens have similar probabilities), the nucleus expands.
