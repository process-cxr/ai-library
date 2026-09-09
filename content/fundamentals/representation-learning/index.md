---
title: Representation Learning
created: 2026-09-09
published: 2026-09-09
modified: 2026-09-09
type: topic
status: seed
area: fundamentals
tags:
  - representation-learning
  - multimodal
  - fundamentals
---

Representation Learning 研究模型如何把原始输入转换为能够支持预测、对齐、检索和决策的内部表示。它是从 Text LM 进入 Multimodal Model 和 World Model 的共同基础。

## 核心问题

- token、embedding、continuous feature 和 latent state 分别表示什么；
- 不同模态如何从 pixels、waveform、frames 或 sensor signals 变成 sequence；
- representation 是否保留语义、空间、时间和控制相关信息；
- contrastive learning、masked modeling 和 autoregressive modeling 如何塑造表示空间；
- 表示对齐后，模型是否真的获得了跨模态推理能力。

## Reading Path

```text
Raw signal
  -> encoder
  -> feature / embedding
  -> alignment or fusion
  -> prediction, retrieval or action
```

## Related Notes

- [[fundamentals/information-theory/cross-entropy|Cross Entropy]]
- [[architecture/multimodal/clip|CLIP]]
- [[architecture/multimodal/vision-language-model|Vision-Language Model]]
- [[architecture/world-models/|World Models]]
