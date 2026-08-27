---
title: "Training Deep Nets with Sublinear Memory Cost"
created: 2026-05-31
published: 2026-05-31
modified: 2026-05-31
type: source
status: seed
source_type: paper
area: sources
tags:
  - source
  - paper
  - training-optimization
  - memory
  - activation-checkpointing
source_url: https://arxiv.org/abs/1604.06174
paper_date: "2016-04"
paper_order: "06174"
---

## 基本信息

- Title: [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
- Source type: paper
- Related topic notes: [[training/optimization/gradient-checkpointing|Gradient Checkpointing]], [[training/optimization/training-memory-estimation|Training Memory Estimation]]

## TODO

- 阅读论文原文，整理 activation checkpointing / rematerialization 如何用重算换显存。
- 回填 sublinear memory cost 的理论形式、计算开销和适用边界。
- 补充该思想在 Transformer / LLM 训练中的工程化变化。
