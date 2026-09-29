---
title: "QLoRA"
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
  - qlora
  - quantization
  - parameter-efficient-finetuning
source_url: https://arxiv.org/abs/2305.14314
paper_date: "2023-05"
paper_order: "14314"
---

## 基本信息

- 标题：[QLoRA](https://arxiv.org/abs/2305.14314)
- 发表：arXiv:2305.14314
- 日期：2023-05
- 相关主题：[[training/post-training/sft|SFT]]、[[training/optimization/training-memory-estimation|Training Memory Estimation]]、[[inference/quantization/quantization|Quantization]]
## TODO

- 阅读论文原文，整理 4-bit base model、LoRA adapters、NF4、double quantization 和 paged optimizer。
- 回填 QLoRA 显存估算中 base weights、quantization metadata、LoRA states 和 activation 的组成。
- 补充 QLoRA 与 full fine-tuning、LoRA、post-training quantization 的边界。
