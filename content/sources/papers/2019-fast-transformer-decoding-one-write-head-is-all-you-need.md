---
title: "Fast Transformer Decoding"
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
  - attention
  - mqa
  - kv-cache
source_url: https://arxiv.org/abs/1911.02150
---

## 基本信息

- Title: [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- Source type: paper
- Related topic notes: [[architecture/attention/multi-query-attention|Multi-Query Attention]], [[inference/kv-cache-and-memory/kv-cache|KV Cache]]

## TODO

- 阅读论文原文，整理 Multi-Query Attention 如何减少 autoregressive decoding 中的 K/V cache 写入和读取成本。
- 回填 MHA 与 MQA 在 KV heads、memory bandwidth 和质量取舍上的差异。
- 对照 GQA，梳理从 MQA 到 grouped sharing 的折中路线。
