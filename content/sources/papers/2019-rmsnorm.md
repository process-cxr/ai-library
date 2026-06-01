---
title: "Root Mean Square Layer Normalization"
created: 2026-06-01
published: 2026-06-01
modified: 2026-06-01
type: source
status: seed
source_type: paper
area: sources
tags:
  - source
  - paper
  - normalization
  - rmsnorm
source_url: https://arxiv.org/abs/1910.07467
---

## 基本信息

- Title: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- Source type: paper
- Related topic notes: [[architecture/transformer/normalization|Normalization in Transformer]], [[fundamentals/neural-network-basics/normalization|Normalization]]

## TODO

- 阅读论文原文，整理 RMSNorm 与 LayerNorm 的公式差异和效率动机。
- 回填 RMSNorm 不做均值中心化、只控制 RMS 尺度的机制。
- 结合 LLaMA/Qwen/DeepSeek 等现代 LLM，整理 RMSNorm 在 decoder-only block 中的常见位置。
