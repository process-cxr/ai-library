---
title: "RoFormer"
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
  - positional-encoding
  - rope
source_url: https://arxiv.org/abs/2104.09864
paper_date: "2021-04"
paper_order: "09864"
---

## 基本信息

- Title: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Source type: paper
- Related topic notes: [[architecture/positional-encoding/rope|RoPE]], [[architecture/positional-encoding/positional-encoding|Positional Encoding]], [[architecture/attention/attention|Attention]]

## TODO

- 阅读论文原文，整理 Rotary Position Embedding 的旋转形式和相对位置性质。
- 回填 RoPE 作用于 Q/K、影响 attention score 而不是直接加到 token embedding 的机制。
- 对照 YaRN、Position Interpolation 和 LongRoPE，梳理 RoPE 长上下文扩展路线。
