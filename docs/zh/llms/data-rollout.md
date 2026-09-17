# 训练数据准备

## Chat template 与训练文本

我们以conversations的形式存储训练数据，`conversations` 是一个列表，每个元素是一个对话轮次，包含 `role` 和 `content`。在训练、rollout 或 dump hidden state 时，我们会将这些对话套用 chat template，以确保与当时使用的 tokenizer 和 thinking 开关保持一致。对于跨模型族的情况，只需要更换 tokenizer，而 `content` 保持不变。

我们以

### apply_chat_




`conversations` 存 `role` / `content`。Chat template 在 rollout、dump hidden state、训练时再套，对齐当时的 tokenizer 与 thinking 开关。跨模型族只换 tokenizer，`content` 保持原样。

`apply_chat_template` 两种用法。丢掉最后一轮 assistant、`add_generation_prompt=True`，得到 completions 的续写前缀。保留完整对话、`add_generation_prompt=False`，得到训练和 dump 的整段文本。两次渲染应对齐：

\[
\texttt{template}(\text{prefix}, \texttt{gen}=1) \;+\; y \;+\; \text{eos}
\;=\;
\texttt{template}(\text{prefix}+y, \texttt{gen}=0)
\]

其中 \(y\) 是目标回复。对不齐则 loss 边界会偏。

Thinking 改的是 generation prompt。开启时前缀停在未闭合的 `<think>`，模型接着写思考、`</think>` 和可见回复。关闭时前缀通常已带 `<think></think>`，模型直接写回答。Dump 与 rollout 共用同一套 `enable_thinking`。`conversations` 走 chat 接口；已渲染的 `text` 走 completions。