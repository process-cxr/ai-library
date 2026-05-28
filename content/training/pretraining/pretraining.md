---
title: Pretraining
created: 2026-02-21
published: 2026-02-21
modified: 2026-05-28
tags:
  - training
  - pretraining
---

Pretraining 是 base model 能力形成的主要阶段。模型在大规模文本、代码、数学、网页、书籍和其他语料上学习语言建模、世界知识、模式补全和初步推理能力。

## 基本流程

典型 decoder-only LLM 的预训练流程包括：

1. 准备和清洗大规模语料，完成去重、质量过滤、语言识别和安全过滤。
2. 用 tokenizer 把文本转成 token 序列。
3. 用 [[training/pretraining/objective|next-token prediction objective]] 训练模型，根据上下文预测下一个 token。
4. 通过学习率 warmup、cosine decay、梯度裁剪、混合精度和 checkpointing 保持训练稳定。
5. 定期评估困惑度、下游 benchmark、数据污染和训练异常。

## 预训练学到什么

Next-token prediction 看似简单，但由于训练数据覆盖大量文本结构，模型会隐式学习：

- 词法、句法和篇章结构；
- 常识和领域知识；
- 代码、数学、表格和格式模式；
- 指令、问答、解释和推理痕迹；
- 多步问题中的中间模式。

这些能力未必以“可控技能”的形式出现，但它们构成后续 SFT、RLHF、RLVR 和应用系统的基础。

## 目标函数不只是工程细节

预训练目标决定模型被奖励的行为。标准 NTP 奖励模型压缩前缀信息并预测真实 token，但不显式奖励探索或推理。因此，近年来出现了一些把 reasoning signal 前移的路线，例如 [[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]]。

[[sources/papers/2025-rlp-reinforcement-as-a-pretraining-objective|RLP]] 的例子表明，可以在预训练末期让模型先生成 thought，再根据 thought 是否提升真实 next token 的 log-likelihood 给 dense reward。这类方法仍然服务于语言建模，但把“思考是否有用”变成了训练信号。

## 与后训练的关系

预训练不是最终 assistant 行为的全部来源。SFT 会塑造对话格式和指令跟随，RLHF / RLVR 会进一步优化偏好或可验证任务表现。但预训练阶段决定了模型能力的底座：如果 base model 已经形成更强的推理表示，后训练通常更容易叠加收益。

## 相关概念

- [[training/pretraining/objective|Training Objective]]
- [[training/pretraining/data-mix|Data Mix]]
- [[training/pretraining/tokenizer|Tokenizer]]
- [[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]]
- [[training/mid-training/continued-pretraining|Continued Pretraining]]
