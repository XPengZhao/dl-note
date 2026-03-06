# Models

## LLMs Timeline

[LLMs Timeline](https://llm-timeline.com/)

## Qwen3-Omni

### Overview

<div align="center">
  <img src="../../assets/images/qwen3-omni-architecture.png" alt="Architecture of Qwen3-Omni" width="600">
</div>


Main features of Qwen3-Omni:

-  Both the Thinker and Talker adopt Mixture-of-Experts (MoE) architectures.
-  Talker no longer consumes the Thinker’s high-level **text representations** and conditions only on audio and visual multimodal features.
   - for textual content, discrete tokens and embeddings are effectively information-equivalent.
- Since textual representations are decoupled, the Thinker and Talker can use distinct system prompts, independently controlling the Thinker’s response style and the Talker’s audio style.
- The Talker adopts a multi-codebook autoregressive scheme: Talker generates one codec frame per step, while the MTP module produces the remaining residual codebooks.
- The Code2Wav is implemented as a lightweight causal ConvNet, simplifying the final stage of audio synthesis.

### Audio Transformer (AuT)

<div align="center">
  <img src="../../assets/images/aut-architecture.png" alt="Architecture of Audio Transformer" width="600">
</div>

- Audio Transformer (AuT) is an attention-encoder-decoder model, trained from scratch on 20 million hours of supervised audio data.
- AuT Encoder contains approximately 0.6B parameters.
- Filter bank features are time–frequency representations derived from short-time analysis of the waveform. A 10 ms frame shift (hop) means the analysis window advances by 10 ms each step.
- In Qwen3-Omni’s description, for audio inputs and audio extracted from video, we resample to 16 kHz and convert the raw waveform into a 128 channel mel-spectrogram with a 25 ms window and a 10 ms hop. Practically, this corresponds to overlapping frames such as 0–25 ms, 10–35 ms, 20–45 ms, etc. A common way to compute a mel-spectrogram is:
    - segment the waveform into overlapping frames (e.g., 25 ms window, 10 ms hop),
    - apply an FFT (or equivalent STFT) per frame to obtain a magnitude/power spectrum,
    - apply a mel-scaled filter bank to aggregate spectral energy into mel bands,
    - optionally apply log-compression (often used in practice, though not always explicitly stated).
- The filter bank features of the audio are downsampled 8 times using Conv2D blocks before the attention layers, from 100 Hz(10ms) to 12.5 Hz. Each frame of the audio representation corresponds to approximately an 80 ms segment of the original audio signal.
- Specifically, the training data includes 80% Chinese and English pseudo-labeled ASR data, 10% ASR data from other languages, and 10% audio understanding data.
- To balance the efficiency of real-time prefill caching with the performance for offline audio tasks, AuT utilizes flash attention with dynamic attention window sizes, covering attention query patterns ranging from 1 to 8 seconds.


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

Under the same MTBench serving configuration (80 prompts, 1024-token output cap, request rate and max concurrency both set to 12, OpenAI-compatible sglang backend with continuous usage statistics enabled), we compared speculative decoding between DFlash (draft block size = 16) and EAGLE-3 (layer-16 speculation). The position-wise acceptance distributions are similar at shallow draft positions, but DFlash consistently maintains higher acceptance mass across deeper positions, with a slower decay beyond mid-range indices. <b>It shows a higher average accepted length for DFlash (3.78) relative to EAGLE-3 (3.49)</b>.

<div align="center">
  <iframe src="../../assets/images/dflash-accepted-length.pdf" title="Accepted Length of DFlash vs. Eagle 3" width="700" height="500"></iframe>
</div>
