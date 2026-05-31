---
title: "Megatron-LM"
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
  - tensor-parallel
  - megatron
source_url: https://arxiv.org/abs/1909.08053
---

## 基本信息

- Title: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- Source type: paper
- Related topic notes: [[training/distributed-training/megatron|Megatron]], [[training/distributed-training/tensor-parallel|Tensor Parallel]], [[training/optimization/training-memory-estimation|Training Memory Estimation]]

## TODO

- 阅读论文原文，整理 Megatron-LM 的 tensor parallel 切分方式。
- 回填 attention / MLP 线性层切分对显存、通信和吞吐的影响。
- 补充 Megatron 与 3D parallelism、ZeRO/FSDP 的组合关系。
