---
title: "GPipe"
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
  - distributed-training
  - pipeline-parallel
source_url: https://arxiv.org/abs/1811.06965
---

## 基本信息

- Title: [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- Source type: paper
- Related topic notes: [[training/distributed-training/pipeline-parallel|Pipeline Parallel]], [[training/optimization/training-memory-estimation|Training Memory Estimation]]

## TODO

- 阅读论文原文，整理 pipeline parallelism、micro-batch 和 pipeline bubble 的机制。
- 回填 layer partition 对显存、吞吐和训练时延的影响。
- 补充 GPipe 与后续 1F1B / Megatron pipeline parallel 的关系。
