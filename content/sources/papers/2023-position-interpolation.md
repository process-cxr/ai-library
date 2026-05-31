---
title: "Position Interpolation"
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
  - long-context
  - positional-encoding
  - rope
source_url: https://arxiv.org/abs/2306.15595
---

## 基本信息

- Title: [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)
- Source type: paper
- Related topic notes: [[training/mid-training/long-context-training|Long Context Training]], [[architecture/positional-encoding/rope|RoPE]]

## TODO

- 阅读论文原文，整理 Position Interpolation 的动机、方法和长度外推设定。
- 回填位置插值为何比直接外推更稳定，以及它对长上下文 continued pretraining 的意义。
- 补充该方法的适用边界、短上下文保持和评测方式。
