---
title: Multimodal Training
created: 2026-09-09
published: 2026-09-09
modified: 2026-09-09
type: topic
status: seed
area: training
tags:
  - multimodal
  - training
  - data
---

Multimodal Training 研究如何让模型同时学习不同模态的表示、跨模态对齐和联合生成。它需要把 modality-specific data pipeline 与 Text LM 的 language modeling objective 连接起来。

## Training Path

```text
single-modality encoder pretraining
  -> image-text / audio-text alignment
  -> multimodal language modeling
  -> multimodal instruction tuning
  -> preference, reward or tool-use optimization
```

## 核心问题

- image-text、audio-text、video-text 和 interleaved data 如何构造；
- contrastive、captioning、masked prediction 和 autoregressive loss 如何配合；
- projector、connector 和 cross-attention 如何获得稳定梯度；
- 如何平衡模态能力、语言能力和多模态 reasoning；
- visual token、video token 和文本 token 的数量差异如何影响训练预算。

## Related Notes

- [[architecture/multimodal/|Multimodal]]
- [[training/pretraining/data-mix|Data Mix]]
- [[training/data-engineering/|Data Engineering]]
- [[inference/multimodal/|Multimodal Inference]]
