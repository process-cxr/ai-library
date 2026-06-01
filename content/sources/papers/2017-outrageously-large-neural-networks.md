---
title: "Outrageously Large Neural Networks"
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
  - sparse-model
source_url: https://arxiv.org/abs/1701.06538
---

## 基本信息

- Title: [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- Source type: paper
- Related topic notes: [[architecture/sparse-and-efficient/moe|Mixture of Experts]]

## TODO

- 阅读论文原文，整理 sparsely-gated MoE layer 的 routing、top-k experts 和 load balancing。
- 回填 MoE 中 total parameters、active computation、expert capacity 与负载均衡的基本概念。
- 对照 GShard、Switch Transformer、DeepSeekMoE 等后续路线，梳理 MoE 在 LLM 中的演进。
