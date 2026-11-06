# AI Infra

## 序言

当我在2025年3月5号第一次从Wireless Systems 调转60度车头开始接触 AI Systems 这个领域时，我对这个领域很多东西都感到十分陌生，一开始甚至对很多名词都毫无概念。当时恰逢DeepSeek V3 惊艳问世，公司上下针对昇腾上MoE模型推理优化忙成了一锅粥，在某地组织了集中攻关。而我作为一个新人，自然也被召集了过去。但是彼时彼刻，我几乎没有啥大模型推理的经验，仅有的知识也只是做Wireless期间对Transformer模型的一点训练经验，对于 AI Infra 完全陌生。一开始我印象特别深，对于很多几乎是常识性的名词，比如算子，我都不清楚含义。AI 基建时代好像是一列火车不停向前开的火车，没有时间等你停下来慢慢去理解。好在是过程中逐渐干中学，慢慢地对这个领域有了入门的理解。仅以此笔记记录下我在 AI Infra 领域的学习过程，供自己回顾，也希望能帮助到后来者。


## Overview

AI Infra 涉及的技术点很多。我刚开始干 AI Systems 的时候，就是胡子眉毛一把抓，哪里有问题就学哪里。这样学能很快解决眼前的问题，但学到的东西比较零散。回头看整个推理系统，总有种管中窥豹的感觉。

后来我想，不妨从一次日常对话入手，把这些零散的技术点串起来。假设你打开一个接入 GLM-5.2 的写作助手（Agent），在对话框里输入“你好！”。从你按下发送，到屏幕上出现第一行回答，中间发生了什么？

### 用户输入

你按下发送后，写作助手开始准备这次模型调用。它从应用配置里取出预设指令“你是一个中文写作助手。”，用这条指令告诉 GLM-5.2 该以什么身份回答。随后，它把你的“你好！”作为用户消息接在指令后面。这两条消息按顺序放进请求的 `messages` 字段。

```json
{
  "messages": [
    {"role": "system", "content": "你是一个中文写作助手。"},
    {"role": "user", "content": "你好！"}
  ]
}
```

每条消息都保留了两个信息。`content` 是这条消息的文本，`role` 标明它在对话中的角色。上例中，`system` 是应用给模型的指令，`user` 是你刚发出的消息。模型生成的回复则可以作为 `assistant` 消息保存在会话历史中。

<div class="chat-example" role="group" aria-label="写作助手对话示例">
  <div class="chat-turn chat-turn-user"><span class="chat-speaker">你</span><p>你好！</p></div>
  <div class="chat-turn"><span class="chat-speaker">写作助手</span><p>你好！你想写什么？</p></div>
  <div class="chat-turn chat-turn-user"><span class="chat-speaker">你</span><p>帮我给这篇 AI Infra 笔记写两段序言。</p></div>
</div>

你发出第二条消息时，应用把上一轮问答和这次的新要求一起放进 `messages`。

```json
{
  "messages": [
    {"role": "system", "content": "你是一个中文写作助手。"},
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！你想写什么？"},
    {"role": "user", "content": "帮我给这篇 AI Infra 笔记写两段序言。"}
  ]
}
```

到了下一轮，助手要决定把哪些历史内容带进请求。假设它刚替你写了一份草稿，你接着说“把第二段再改短一点”。助手取出上一轮包含草稿的 `assistant` 消息，再追加这次修改要求作为新的 `user` 消息。有了草稿，模型才知道“第二段”指的是哪段文字。如果应用还保存过“我喜欢短句”这样的长期偏好，助手也可以在这一轮取回它，作为补充要求放进上下文。

回到最初的“你好！”。你输入的文字很短，助手随请求附带的内容却可能很长。例如，Codex CLI 会在用户消息之前加入工作环境信息，也可以带上项目的 `AGENTS.md` 指令。[OpenAI 对首轮请求的拆解](https://openai.com/index/unrolling-the-codex-agent-loop/)

工具定义也占输入空间。Anthropic 举过一个例子：接入 5 个 MCP 服务，一次加载全部 58 个工具的定义，仅工具名称、说明和参数等内容就占约 5.5 万 token。这个数字取决于加载了哪些工具，用户的问候本身仍然只有“你好！”。[Anthropic 的工具加载示例](https://www.anthropic.com/research/advanced-tool-use)

项目指令如何加载、工具定义和调用结果分别在哪一轮进入上下文，见 [Agent 的指令与工具](agent-input.md)。

## 请求进入推理引擎

输入准备好后，写作助手向服务端的 `POST /v1/chat/completions` 接口发送请求。请求体除了 `messages`，还用 `model` 指定要调用的 `glm-5.2`，用 `stream: true` 请求流式返回。假设服务端有 A、B 两个实例，都部署了 GLM-5.2。请求先到网关，由网关的路由器（Router）选择其中一个实例。

A 刚处理过同一个写作助手的请求，B 当前排队的请求更少，该发给谁？假如这个助手有几千个 token 的固定系统指令，发给 A 有机会复用相同前缀的计算结果。发给 B 则可能更早轮到执行。缓存感知路由考虑的是两笔时间：选 A 能省下多少前缀计算，是否值得为此多排一会儿队。对于上例中很短的系统指令，缓存收益可能很小。下面假设 Router 选中 A。

请求抵达 A 后，需要有人把对话转成模型输入，安排 GPU 执行，并把生成的结果送回调用方。实例里的推理服务负责这些工作。下面以 vLLM 为例：它的 API 层接收聊天请求，执行引擎负责任务调度和模型计算。

API 层先按 GLM-5.2 的对话模板（**chat template**）渲染 `messages`。例如，用户消息会带上 `<|user|>`，生成回答的位置会带上 `<|assistant|>`。在启用思考的模板配置下，末尾还会出现 `<think>`。Tokenizer 再把渲染后的内容编码为 token ID。这些角色标记和生成提示也占输入位置，所以不能用“你好！”的字符数估算这次实际要处理的 token 数。[GLM-5.2 对话模板示例](glm-5.2-inference-path.md#对话模板与分词)

接下来，请求进入引擎的等待队列。A 可以用输入的 token 序列查找本地 **prefix cache**。假设两次请求的开头相同，且 A 仍保留着对应的完整 KV 块，本次请求就可以引用这些块，从其后的 token 继续计算。即使用户发了同一句“你好！”，只要系统指令在前面某处变了，匹配也只能延续到差异之前。Router 估计哪个副本可能命中，A 决定具体哪些块可以复用。[KV Cache](kv-cache.md)

调度器每轮都要决定两件事：哪些请求可以执行，每条请求在这一轮计算几个 token。举个小例子，假设每轮最多安排 8 个 token，已有两条请求各要做一步 decode，你的 prompt 还有 9 个输入 token 待计算，显存中也有足够的 KV 空间。在启用 chunked prefill 的情况下，一种排法是：

| 调度轮次 | 其他请求 | 你的请求 |
| --- | --- | --- |
| 第一轮 | 两条请求各处理 1 个 token | 处理前 6 个输入 token |
| 第二轮 | 若两条请求都继续生成，各处理 1 个 token | 处理剩余 3 个输入 token，得到首个输出 token |
| 第三轮 | 已结束的请求退出，新请求可以加入 | 用刚生成的 token 做一步 decode |

每轮重新组合请求，就是 **continuous batching**。同一条长输入可以分几轮完成 prefill，就是 **chunked prefill**。例子中的 8 是单轮的计算预算。另一个约束是 KV 空间：即使本轮还有计算预算，引擎也要能为新增位置分配 KV 块，才能安排相应的计算。[推理运行时](serving-runtime.md) · [vLLM 调度说明](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill)

## 从 prefill 到 decode

现在跟着你的输入看一次 GPU 前向。模型通过 embedding 为每个待处理的 token ID 取出一个向量。向量经过一层层 Transformer，在 attention 中结合当前位置可见的前文，在前馈网络中进一步变换。各层还会把历史位置的注意力状态写入 KV Cache，后续 token 可以直接读取这些状态。[GLM-5.2 推理案例](glm-5.2-inference-path.md)

具体到 GLM-5.2，前 3 层使用 dense MLP，后 75 层使用 MoE。进入 MoE 层时，模型内部也有一个 router：它根据 token 的向量选择参与计算的专家。如果专家分布在多张 GPU 上，执行引擎就要把 token 向量发往所选专家所在的卡，再取回专家输出，按路由权重合并。网关 Router 为整条请求选副本，MoE router 为一层里的每个 token 选专家。二者的决策发生在不同时间，也涉及不同的数据。[并行策略](parallelism.md) · [集合通信](collective-communication.md)

处理到完整 prompt 的最后一个位置后，模型把该位置的最终向量投影到词表，得到每个候选 token 的分数，即 **logits**。解码器按请求配置选择一个 token：贪心解码选分数最高的候选，采样则根据调整后的概率分布选取。由此产生的是回答的第一个 token。它来自 prefill 最后一步的结果。[解码与采样](decoding.md)

设刚选出的 token 为 $y_1$。进入 **decode** 后，这条请求的新输入只有 $y_1$。模型仍要走过全部层，但可以直接读取 prompt 已有的 KV，同时追加 $y_1$ 的状态。新的 logits 用于选择 $y_2$。下一轮以 $y_2$ 为输入，同样的依赖一直延续到停止条件成立。投机解码正是从这个串行依赖入手，用一次目标模型前向验证多个候选。[Speculative Decoding](speculative-decoding.md)

同样是一次前向，prefill 可以把多个输入位置一起送进矩阵乘，复用读入的权重。普通 decode 中，一条请求每轮只增加一个位置，却仍需要读取权重和历史 KV。低并发时，一轮能共同使用权重的 token 较少，数据搬运占用的时间就更值得关注。这也是服务端会把你的 decode 和其他用户的 decode 放进同一批的原因。[Roofline 模型](roofline.md)

最后回到请求里的 `stream: true`。服务端随着生成进度把 token 解码成文本片段，经网关发回 Agent，对话框才开始逐段显示文字。首个输出 token 和屏幕上的第一个字是两个测量点。后端的首 token 时间包含排队和 prefill，用户看到正文的时间还受网络、缓冲以及应用如何展示推理内容影响。回答开始后，相邻片段之间的停顿又是另一种体验。[推理指标](metrics.md)

生成结束，引擎释放这条请求持有的 KV 块引用，可复用的前缀块仍可能留在缓存里。此时你再问一句“能帮我改段文字吗？”，Agent 会把上一轮问答加入新的 `messages`。这次请求比第一次长了。如果仍命中相同的前缀，GPU 就可以接着已有状态往下算。
