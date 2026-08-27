---
title: "ZeRO-Infinity"
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
  - zero
  - offload
source_url: https://arxiv.org/abs/2104.07857
paper_date: "2021-04"
paper_order: "07857"
---

## 基本信息

- Title: [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)
- Source type: paper
- Related topic notes: [[training/distributed-training/zero|ZeRO]], [[training/optimization/training-memory-estimation|Training Memory Estimation]], [[training/scaling/training-budget|Training Budget]]

## TODO

- 阅读论文原文，整理 CPU/NVMe offload 如何扩展可训练模型规模。
- 回填异构内存层级、带宽瓶颈和调度策略。
- 补充 offload 与 GPU 显存、训练吞吐、checkpoint 存储之间的 trade-off。
