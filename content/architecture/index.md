---
title: "Architecture"
created: 2026-01-18
published: 2026-01-18
modified: 2026-05-31
---

大模型架构设计，从 Transformer 基础结构到注意力变体、位置编码、模型家族、稀疏高效架构和多模态架构。

## Recommended Path

1. [[architecture/transformer/transformer|Transformer]]：先理解 block、residual、normalization、FFN 和 decoder-only 结构。
2. [[architecture/attention/attention|Attention]]：再理解 self-attention、MHA、MQA/GQA/MLA 和长上下文 attention 变体。
3. [[architecture/positional-encoding/positional-encoding|Positional Encoding]]：理解 RoPE、ALiBi、YaRN 等位置机制如何影响 attention。
4. [[architecture/sparse-and-efficient/moe|Mixture of Experts]]：理解 dense FFN 到 sparse expert FFN 的容量/计算取舍。
5. [[architecture/model-families/|Model Families]]：最后把 GPT、LLaMA、Qwen、DeepSeek 等看作基础机制的组合案例。

## Core Modules

- [[architecture/transformer/|Transformer]] — 标准 Transformer 结构、decoder-only、FFN、归一化和残差。
- [[architecture/attention/|Attention]] — Self-Attention、MHA、MQA、GQA、滑动窗口注意力。
- [[architecture/positional-encoding/|Positional Encoding]] — 绝对位置、正弦位置、RoPE、ALiBi、YaRN。

## Model and Architecture Families

- [[architecture/model-families/|Model Families]] — GPT、LLaMA、Qwen、DeepSeek、Mistral、Gemma 等模型家族。
- [[architecture/sparse-and-efficient/|Sparse and Efficient Architectures]] — MoE、Mamba、SSM、Linear Attention 等高效路线。
- [[architecture/multimodal/|Multimodal]] — VLM、CLIP、LLaVA、Qwen-VL 和多模态 projector。
- [[architecture/world-models/|World Models]] — observation、latent state、dynamics、reward 和未来状态预测。

## From Text LM to World Model

现有 Transformer、Attention、Position 和 Model Families 页面构成 Text LM Core。Multimodal 不是替换这条主线，而是在已有 sequence modeling 基础上增加视觉、音频和视频的 representation、alignment 与 fusion；World Model 则进一步引入 state、action、transition 和 planning。

```text
Text token prediction
  -> multimodal representation and alignment
  -> observation understanding
  -> state and dynamics modeling
  -> action-conditioned prediction
  -> planning in an environment
```

建议沿着 `architecture/multimodal/` 学习 encoder、projector 和 fusion，再沿着 `architecture/world-models/` 学习 latent state、dynamics 和 model-based planning。
