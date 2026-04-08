# 模型

## LLM 时间线

[LLMs Timeline](https://llm-timeline.com/)

## Qwen3-Omni

### 概览

<div align="center">
  <img src="../assets/images/qwen3-omni-architecture.png" alt="Architecture of Qwen3-Omni" width="600">
</div>

Qwen3-Omni-30B-A3B 值得单独整理，是因为它不是“在视觉语言模型后面再挂一个语音头”这么简单。它试图把文本、图像、音频、视频的感知，以及文本和语音生成，放进同一套部署路径里。理解这个模型时，问题不只是“多模态 backbone 长什么样”，还包括“感知和实时语音生成是如何被放进同一个系统里的”。

从整体结构上看，Qwen3-Omni 采用了一个 Thinker-Talker 分工：

- **Thinker** 是主要的多模态理解与推理主干，负责消费文本、图像、音频和视频输入，并产出用于理解和文本生成的语义表示。
- **Talker** 是语音生成路径，负责自回归预测 codec tokens，再通过 `Code2Wav` 重建语音波形。

这个分工之所以重要，是因为它把 omni-modal generation 变成了一个系统设计问题，而不只是单个 backbone 的问题。模型可以在同一条路径上完成多模态推理，同时又让语音生成链路保持适合流式输出的结构。

### 主要结构特征

- Thinker 和 Talker 都采用 Mixture-of-Experts（MoE）架构。
- Thinker 里的文本主干是一个 48 层 MoE decoder，包含 128 个 experts，每个 token 激活 8 个 experts。
- 文本注意力是 **GQA**，不是 MHA，也不是 MLA：公开 config 给出的配置是 32 个 query heads、4 个 KV heads。
- Vision stack 是一个 ViT-like encoder，`depth = 27`、`hidden_size = 1152`、`num_heads = 16`，并带有 `deepstack_visual_indexes = [8, 16, 24]`。
- Audio tower 是一个 32 层 Transformer encoder，`d_model = 1280`、`encoder_attention_heads = 20`、`encoder_ffn_dim = 5120`。
- Talker 采用多 codebook 的 codec 生成路径，一共 16 个 code groups / quantizers。最终的 `Code2Wav` 不是扩散式大解码器，而是轻量级因果 ConvNet。

这些数字直接来自 Hugging Face 上公开的 [Qwen/Qwen3-Omni-30B-A3B-Instruct 配置文件](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/config.json)。

### 一个紧凑的结构总结

| 模块 | 关键结构 | 记忆点 |
| --- | --- | --- |
| Thinker 文本主干 | 48 层 MoE decoder，`hidden_size = 2048`，`32` 个 attention heads，`4` 个 KV heads，`128` 个 experts，`8` 个 active experts | 主干较窄，但 MoE 很重；文本注意力是 GQA |
| Vision encoder | ViT-like encoder，`depth = 27`，`hidden_size = 1152`，`num_heads = 16`，`patch_size = 16` | 更像 Qwen3-VL 一类视觉主干，而不是轻量 side encoder |
| Audio encoder | Transformer encoder，`encoder_layers = 32`，`d_model = 1280`，`encoder_attention_heads = 20` | 音频理解是一个单独而完整的 dense encoder |
| Talker 文本栈 | 20 层 MoE decoder，`hidden_size = 1024`，`16` 个 heads，`2` 个 KV heads | 语音生成路径有自己的 text-like stack |
| Codec 路径 | `16` 个 quantizers，`codebook_size = 2048`，`semantic_codebook_size = 4096`，`codebook_dim = 512` | 语音先生成离散 codec tokens，再还原波形 |
| Code2Wav | `code2wav_config` 中为 8 层 decoder stack | 最后的波形合成刻意做得很轻 |

### Thinker：主多模态推理主干

Thinker 是大多数人直觉里“模型本体”的部分。它承担多模态理解、通过语言主干进行推理，也是如果你只需要文本输出时最想保留的部分。

从公开 config 来看，Thinker 的文本主干参数包括：

- `num_hidden_layers = 48`
- `hidden_size = 2048`
- `num_attention_heads = 32`
- `num_key_value_heads = 4`
- `num_experts = 128`
- `num_experts_per_tok = 8`

这个组合本身很值得注意。对于一个名字里带有 “30B” 的模型来说，它的 backbone hidden size 并不大，但总容量被大量 MoE experts 撑了起来。因此，更合适的心智模型不是“一个很宽的 dense 30B 模型”，而是“一个相对较窄的多模态主干，后面挂着很重的 expert 容量”。这也解释了它为什么采用 GQA：模型希望在保留较多 query heads 的同时，让 KV 状态比 full MHA 更轻。

### Vision Stack

Vision 侧更像一个完整的视觉编码器，而不是一个轻量 projector。公开 config 中可以看到：

- `depth = 27`
- `hidden_size = 1152`
- `num_heads = 16`
- `intermediate_size = 4304`
- `patch_size = 16`
- `deepstack_visual_indexes = [8, 16, 24]`

这里的命名沿用了 CV 社区习惯，因此 `depth` 就应该读作 Transformer blocks 的层数。由于 config 里只给出了 `num_heads`，并没有单独暴露 KV-head 的配置，因此最稳妥的解释是：vision encoder 使用的是标准 MHA，而不是 GQA。

这也是为什么 Qwen3-Omni 的视觉部分不应被理解成一个“很薄的前端”。它本身就是一个规模不小的 Transformer encoder，之后再与 Thinker 的隐藏空间发生多模态融合。

### Audio Transformer（AuT）

<div align="center">
  <img src="../assets/images/aut-architecture.png" alt="Architecture of Audio Transformer" width="600">
</div>

音频是 Qwen3-Omni 与标准 VL 模型拉开差异的关键部分。它不是在文本主干前面接了一个薄薄的 adapter，而是专门配了一整套 Transformer encoder，用来把声学输入变成 Thinker 能够推理的表示。

从 config 来看，audio encoder 的关键结构是：

- `encoder_layers = 32`
- `d_model = 1280`
- `encoder_attention_heads = 20`
- `encoder_ffn_dim = 5120`

这意味着音频塔本身就是一个深且宽的 dense encoder。从结构直觉上看，它更像 Whisper 风格的 Transformer encoder，而不是一个简单 projection layer。

在前端，Qwen3-Omni 会先把音频重采样到 16 kHz，再转换为 128 通道的 mel-spectrogram，窗口为 25 ms，hop 为 10 ms。进入 attention 层之前，这些时频特征还会先经过 Conv2D 模块做下采样。这样设计的重点在于平衡两件事：既要保留足够的声学细节供理解，又不能让后续 runtime 面对过长的原始音频序列。

### AuT 的系统意义

AuT 这一部分值得保留，因为它说明了 Qwen3-Omni 的音频路径为什么在 serving 上和文本、视觉输入不一样：

- AuT 在大规模监督音频数据上从零训练。
- 仅 encoder 一侧规模就约有 `0.6B` 参数。
- 输入表示是 mel-spectrogram，而不是原始 waveform token。
- 在进入 Transformer 层前，Conv2D 模块会把时频表示下采样 `8x`，从 `100 Hz` 降到 `12.5 Hz`。
- 为了兼顾实时 prefill cache 效率与离线音频任务表现，实现中使用了动态注意力窗口的 flash attention，覆盖大约 `1` 到 `8` 秒的查询模式。

从系统角度看，这说明 Qwen3-Omni 不是把音频简单 token 化后喂给 LLM，而是先通过专门的声学编码器做压缩和结构化，再让下游多模态主干来接管。

### Talker、Codec Tokens 与 Code2Wav

Talker 这一侧，让 Qwen3-Omni 从“多模态理解模型”真正变成了“any-to-any 生成模型”。它并不是把 Thinker 的高层表示当作语音合成的最后格式化步骤，而是拥有一条独立的生成链路。

有几个细节值得记住：

- Talker 自身也是一个 20 层的 MoE decoder。
- 语音通过 **多 codebook** 方案生成，一共 `16` 个 code groups。
- `codebook_size = 2048`，`semantic_codebook_size = 4096`，`codebook_dim = 512`。
- 模型先预测离散 codec 结构，再通过 `Code2Wav` 恢复波形。

这是一个很关键的设计选择。语音生成被放在 codec 抽象之后，使得自回归主路径不必直接面对 waveform 级别的生成难度，也让流式语音输出在工程上更现实。

### 怎么把 Qwen3-Omni 放回 Qwen 家族里理解

从建模角度看，Qwen3-Omni 更适合被理解为多个完整子系统的组合，而不是一个单体 decoder：

- 一个用于推理的多模态 MoE 语言主干，
- 一个真正的视觉编码器，而不是薄图像适配器，
- 一个用于声学理解的 dense audio encoder，
- 再加上一条基于 codec prediction 的独立语音生成链路。

这也是为什么它的参数结构乍看会有点“反直觉”。模型之所以大，并不只是因为文本主干很宽，容量实际上被分散到了 MoE experts、视觉和音频编码器，以及语音生成路径上。

### 实用备注

- 如果只需要文本输出，使用 `Qwen3OmniMoeThinkerForConditionalGeneration` 可以避免把完整语音生成路径一并加载进来。
- Hugging Face 当前说明，集成模型路径下的完整音频生成目前只支持 batch size `1`。
- 像 “更像 Whisper-style” 或 “ViT-like” 这样的说法，是基于公开 config 的结构解释；真正最可信的层数和 hidden 维度信息，仍然应该以公开 config 为准。

## DFlash

```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export MODEL_PATH=/models/Qwen
export TVM_FFI_CUDA_ARCH_LIST="9.0"

## DFLASH
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server \
    --model-path $MODEL_PATH/Qwen3-Coder-30B-A3B-Instruct \
    --disable-radix-cache \
    --tp-size 4 \
    --dtype bfloat16 \
    --attention-backend fa3 \
    --context-length 32000 \
    --max-running-requests 12 \
    --port 8005 \
    --host 0.0.0.0 \
    --mem-fraction-static 0.8 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path $MODEL_PATH/Qwen3-Coder-30B-A3B-DFlash \
    --trust-remote-code
```

在相同 MTBench 服务配置下（80 条提示、输出上限 1024 token、请求速率和最大并发均设为 12，后端为 OpenAI 兼容 sglang，并开启连续使用统计），我们对比了 DFlash（draft block size = 16）与 EAGLE-3（layer-16 speculation）的 speculative decoding。两者在浅层草稿位置的接受率分布相近，但 DFlash 在更深位置上保持了更高的接受概率质量，且中后段衰减更慢。<b>结果显示 DFlash 的平均接受长度更高（3.78），优于 EAGLE-3（3.49）</b>。

<div align="center">
  <iframe src="../assets/images/dflash-accepted-length.pdf" title="Accepted Length of DFlash vs. Eagle 3" width="700" height="500"></iframe>
</div>
