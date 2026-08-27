---
title: "GShard"
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
  - moe
  - distributed-training
source_url: https://arxiv.org/abs/2006.16668
paper_date: "2020-06"
paper_order: "16668"
---

## 基本信息

- Title: [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- Source type: paper
- Related topic notes: [[architecture/sparse-and-efficient/moe|Mixture of Experts]], [[training/distributed-training/megatron|Megatron]]

## TODO

- 阅读论文原文，整理 conditional computation、expert routing 和 automatic sharding 的系统设计。
- 回填 MoE 训练中 expert parallel、通信和负载均衡的关键问题。
- 对照 Switch Transformer 和 DeepSeekMoE，梳理大规模 MoE 系统演进。
