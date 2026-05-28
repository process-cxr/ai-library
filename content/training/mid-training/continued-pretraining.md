---
title: Continued Pretraining
created: 2026-02-28
published: 2026-02-28
modified: 2026-05-28
type: topic
status: growing
area: training
tags:
  - mid-training
  - continued-pretraining
---

Continued Pretraining，简称 CPT，是在已有 base model 或 intermediate checkpoint 上继续做预训练。它通常仍使用语言建模目标，但更换数据、阶段、学习率或训练预算，用来注入领域知识、延长训练、改善某类能力或完成预训练末期退火。

## 目标与使用场景

CPT 常见目标包括：

- 领域适配：继续训练法律、医疗、金融、代码、数学等领域语料。
- 能力注入：提高代码、数学、长上下文、多语言或科学问答能力。
- 数据修正：用更高质量或更干净的数据改善早期训练带来的噪声。
- 训练末期增强：在 base model 接近收敛后，用小学习率和高质量数据做能力整理。

## 与从零预训练的区别

从零预训练需要从随机初始化开始学习词法、语法、知识和基本模式，训练 token 量巨大。CPT 则继承已有模型能力，通常使用更小学习率、更少 token 和更聚焦的数据。

因此，CPT 的风险不是“学不会基础语言”，而是：

- 领域数据过窄导致灾难性遗忘；
- 数据配比不当导致通用能力下降；
- 学习率过高破坏已有表示；
- 数据污染影响评测可信度。

## 与 SFT 的区别

SFT 训练的是指令响应格式和 assistant 行为，数据通常是 prompt-response。CPT 训练的是语言建模或预训练式目标，数据可以是原始文档、代码、教材、论文、网页或混合语料。

简单说：

- CPT 更像“继续读书”；
- SFT 更像“学习如何按用户请求回答”。

## 与 Reinforcement Pretraining 的关系

传统 CPT 通常仍使用 next-token prediction。[[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]] 则在 continued pretraining 阶段改变目标函数。

例如 [[sources/papers/2025-rlp-reinforcement-as-a-pretraining-objective|RLP]] 会让模型在预测下一 token 前先生成 thought，并根据 thought 对真实 token log-likelihood 的提升获得 reward。它和 CPT 的差异不只是训练数据，而是训练信号：CPT 直接最大化 token likelihood，RLP 优化的是 thought 的信息增益。

## 相关概念

- [[training/pretraining/pretraining|Pretraining]]
- [[training/pretraining/objective|Training Objective]]
- [[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]]
- [[training/mid-training/domain-adaptation|Domain Adaptation]]
- [[training/mid-training/capability-injection|Capability Injection]]
