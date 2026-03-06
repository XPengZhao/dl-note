# 模型

## LLM 时间线

[LLMs Timeline](https://llm-timeline.com/)

## Qwen3-Omni

### 概览

<div align="center">
  <img src="../../assets/images/qwen3-omni-architecture.png" alt="Architecture of Qwen3-Omni" width="600">
</div>

Qwen3-Omni 的主要特点：

- Thinker 和 Talker 都采用 Mixture-of-Experts（MoE）架构。
- Talker 不再使用 Thinker 的高层**文本表示**，而只依赖音频与视觉多模态特征。
   - 对文本内容而言，离散 token 与 embedding 在信息上可视为等价。
- 由于文本表示解耦，Thinker 与 Talker 可使用不同系统提示词，从而分别控制 Thinker 的回复风格和 Talker 的语音风格。
- Talker 采用多码本自回归方案：每步生成一个 codec frame，剩余残差码本由 MTP 模块生成。
- Code2Wav 采用轻量级因果 ConvNet，实现更简洁的音频合成末端流程。

### Audio Transformer（AuT）

<div align="center">
  <img src="../../assets/images/aut-architecture.png" alt="Architecture of Audio Transformer" width="600">
</div>

- Audio Transformer（AuT）是一个 attention-encoder-decoder 模型，在 2000 万小时监督音频数据上从零训练。
- AuT Encoder 规模约为 0.6B 参数。
- 滤波器组特征（filter bank features）是由波形短时分析得到的时频表示。10 ms 的帧移（hop）表示分析窗口每次前进 10 ms。
- 在 Qwen3-Omni 描述中，对于音频输入与视频中抽取的音频，会重采样至 16 kHz，并转换为 128 通道 mel-spectrogram（25 ms 窗口、10 ms hop）。这对应重叠帧，如 0–25 ms、10–35 ms、20–45 ms 等。常见计算流程为：
    - 将波形切分为重叠帧（例如 25 ms 窗口、10 ms hop）；
    - 对每帧做 FFT（或等价 STFT）得到幅度/功率谱；
    - 应用 mel 滤波器组，将谱能量聚合到 mel 频带；
    - 可选做对数压缩（实践中常用，但论文中未必总会显式写出）。
- 进入 attention 层之前，音频滤波器组特征经 Conv2D 模块下采样 8 倍，从 100 Hz（10ms）到 12.5 Hz。对应地，每帧音频表示约覆盖原始信号的 80 ms。
- 训练数据构成为：80% 中文与英文伪标注 ASR 数据，10% 其他语言 ASR 数据，10% 音频理解数据。
- 为平衡实时 prefill cache 效率与离线音频任务表现，AuT 使用动态注意力窗口的 flash attention，可覆盖 1 到 8 秒的注意力查询模式。

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
  <iframe src="../../assets/images/dflash-accepted-length.pdf" title="Accepted Length of DFlash vs. Eagle 3" width="700" height="500"></iframe>
</div>
