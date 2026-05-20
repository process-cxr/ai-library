---
title: "Architecture"
---

大模型架构设计，从 Transformer 基础结构到注意力变体、位置编码、模型家族、稀疏高效架构和多模态架构。

## Core Modules

- [[architecture/transformer/|Transformer]] — 标准 Transformer 结构、decoder-only、FFN、归一化和残差。
- [[architecture/attention/|Attention]] — Self-Attention、MHA、MQA、GQA、滑动窗口注意力。
- [[architecture/positional-encoding/|Positional Encoding]] — 绝对位置、正弦位置、RoPE、ALiBi、YaRN。

## Model and Architecture Families

- [[architecture/model-families/|Model Families]] — GPT、LLaMA、Qwen、DeepSeek、Mistral、Gemma 等模型家族。
- [[architecture/sparse-and-efficient/|Sparse and Efficient Architectures]] — MoE、Mamba、SSM、Linear Attention 等高效路线。
- [[architecture/multimodal/|Multimodal]] — VLM、CLIP、LLaVA、Qwen-VL 和多模态 projector。
