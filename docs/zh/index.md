# DL Notes

一个聚焦于现代深度学习系统的知识库，覆盖硬件基础、LLM 推理服务与神经图形学。

## 建议阅读顺序

如果你是第一次阅读本笔记，建议按以下顺序：

1. [硬件](hardware.md) - GPU/NPU 基础、显存类型与 GPU 配置
2. [张量操作](tensor.md) - 常见张量维度操作、激活函数与 CUDA Graph
3. [LLM 基础](llms/preliminary.md) - 推理延迟/吞吐指标、KV Cache 与采样机制
4. [模型笔记](llms/models.md) - Qwen3-Omni 与 DFlash 的结构和实践命令
5. [神经图形](neural-graphics.md) - NeRF 与 Flow Matching 基础

## 文档目录

### 系统与基础设施

- [硬件](hardware.md)
- [张量操作](tensor.md)

### 大语言模型

- [LLM 基础](llms/preliminary.md)
- [模型笔记](llms/models.md)

### 图形与生成建模

- [神经图形](neural-graphics.md)

## 说明

本项目强调实践导向：

- 用简洁理论解释核心机制
- 提供可复用的代码片段与配置示例
- 关注真实推理负载下的性能与工程权衡
