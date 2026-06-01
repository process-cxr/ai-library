---
title: Transformer
created: 2026-01-18
published: 2026-01-18
modified: 2026-05-31
---

Transformer 模块负责整理标准 Transformer 架构及其核心组成，包括 block 结构、attention、FFN、残差、归一化和 decoder-only 变体。

## Reading Path

1. [[architecture/transformer/transformer|Transformer]]：整体结构和训练/推理关系。
2. [[architecture/transformer/decoder-only-transformer|Decoder-only Transformer]]：当前 LLM 的主流架构形态。
3. [[architecture/attention/self-attention|Self-Attention]] 与 [[architecture/attention/multi-head-attention|Multi-Head Attention]]：block 中的信息交互机制。
4. [[architecture/transformer/feed-forward|Feed Forward Network]]：逐 token 非线性变换和 MoE 的基础。
5. [[architecture/transformer/normalization|Normalization in Transformer]] 与 [[architecture/transformer/residual|Residual in Transformer]]：深层训练稳定性的结构条件。

## Notes

- [[architecture/transformer/transformer|Transformer]]
- [[architecture/transformer/decoder-only-transformer|Decoder-only Transformer]]
- [[architecture/transformer/encoder-decoder-transformer|Encoder-Decoder Transformer]]
- [[architecture/transformer/feed-forward|Feed Forward Network]]
- [[architecture/transformer/normalization|Normalization in Transformer]]
- [[architecture/transformer/residual|Residual in Transformer]]

## Related Source TODOs

- [[sources/papers/2017-attention-is-all-you-need|Attention Is All You Need]]
- [[sources/papers/2020-language-models-are-few-shot-learners|Language Models are Few-Shot Learners]]
- [[sources/papers/2020-glu-variants-improve-transformer|GLU Variants Improve Transformer]]
- [[sources/papers/2016-layer-normalization|Layer Normalization]]
- [[sources/papers/2019-rmsnorm|Root Mean Square Layer Normalization]]
- [[sources/papers/2015-deep-residual-learning-for-image-recognition|Deep Residual Learning for Image Recognition]]
