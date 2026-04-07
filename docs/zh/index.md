# DL Notes

一个聚焦于现代深度学习系统的知识库，覆盖硬件基础、LLM 推理服务与神经图形学。

## 建议阅读顺序

如果你是第一次阅读本笔记，建议按以下顺序：

1. [硬件](hardware.md) - GPU/NPU 基础、显存类型与 GPU 配置
2. [张量操作](tensor.md) - 常见张量维度操作、激活函数与 CUDA Graph
3. [AI Infra 总览](llms/overview.md) - 端到端推理系统的分层结构
4. [推理请求生命周期](llms/request-lifecycle.md) - 一条请求如何穿过整个在线系统
5. [AI Infra 指标](llms/metrics.md) - TTFT、TPOT 与 TPS
6. [KV Cache](llms/kv-cache.md) - KV 内存语义、paged blocks 与 prefix 复用
7. [推理运行时](llms/serving-runtime.md) - chunked prefill、准入控制与运行时稳定性
8. [并行策略](llms/parallelism.md) - 从显存、吞吐与通信理解 DP/TP
9. [解码与采样](llms/decoding.md) - 采样策略与 speculative decoding
10. [训练目标](llms/training-objective.md) - 自回归预训练目标
11. [模型笔记](llms/models.md) - Qwen3-Omni 与 DFlash 的结构和实践命令
12. [神经图形](neural-graphics.md) - NeRF 与 Flow Matching 基础

## 文档目录

### 系统与基础设施

- [硬件](hardware.md)
- [张量操作](tensor.md)

### AI Infra

- [总览](llms/overview.md)
- [请求生命周期](llms/request-lifecycle.md)
- [指标](llms/metrics.md)
- [KV Cache](llms/kv-cache.md)
- [推理运行时](llms/serving-runtime.md)
- [并行策略](llms/parallelism.md)
- [解码与采样](llms/decoding.md)
- [训练目标](llms/training-objective.md)

### 模型

- [模型笔记](llms/models.md)

### 图形与生成建模

- [神经图形](neural-graphics.md)

## 说明

本项目强调实践导向：

- 用简洁理论解释核心机制
- 提供可复用的代码片段与配置示例
- 关注真实推理负载下的性能与工程权衡
