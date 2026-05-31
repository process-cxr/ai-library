---
title: Pretraining
created: 2026-02-21
published: 2026-02-21
modified: 2026-05-31
---

预训练阶段从大规模通用语料中学习基础语言建模能力，是 base model 能力形成的主要阶段。

这个目录可以按“目标 -> 数据 -> 表示 -> 规模”的顺序阅读：

1. [[training/pretraining/pretraining|Pretraining]]：预训练整体流程和它与后训练的关系。
2. [[training/pretraining/objective|Training Objective]]：next-token prediction 及其边界。
3. [[training/pretraining/data-mix|Data Mix]]：训练分布如何决定能力分布。
4. [[training/pretraining/tokenizer|Tokenizer]]：文本如何变成 token，以及 tokenizer 为什么是模型契约。
5. [[training/pretraining/compute-optimal|Compute Optimal]]：把 scaling 原理落到真实预训练计划。

## Notes

- [[training/pretraining/pretraining|预训练]]
- [[training/pretraining/objective|训练目标]]
- [[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]]
- [[training/pretraining/data-mix|数据混合]]
- [[training/pretraining/tokenizer|Tokenizer]]
- [[training/pretraining/compute-optimal|Compute Optimal]]

## Related Source TODOs

- [[sources/papers/2016-neural-machine-translation-of-rare-words-with-subword-units|Neural Machine Translation of Rare Words with Subword Units]]
- [[sources/papers/2018-sentencepiece|SentencePiece]]
- [[sources/papers/2021-the-pile|The Pile]]
- [[sources/papers/2023-a-pretrainers-guide-to-training-data|A Pretrainer's Guide to Training Data]]
- [[sources/papers/2023-refinedweb|RefinedWeb]]
- [[sources/papers/2024-datacomp-lm|DataComp-LM]]
