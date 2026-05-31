---
title: Continued Pretraining
created: 2026-02-28
published: 2026-02-28
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - mid-training
  - continued-pretraining
---

Continued Pretraining，简称 CPT，是在已有 base model 或 intermediate checkpoint 上继续做预训练。它通常仍使用语言建模目标，但更换数据、阶段、学习率或训练预算，用来注入领域知识、延长训练、改善某类能力或完成预训练末期退火。

CPT 处在 [[training/pretraining/pretraining|Pretraining]] 与 [[training/post-training/sft|SFT]] 之间。它继承 base model 的通用能力，又比 SFT 更接近预训练式的数据和目标，因此常被用作能力塑形、领域迁移和训练末期整理的中间层。

## 目标与使用场景

CPT 常见目标包括：

- 领域适配：继续训练法律、医疗、金融、代码、数学等领域语料。
- 能力注入：提高代码、数学、长上下文、多语言或科学问答能力。
- 数据修正：用更高质量或更干净的数据改善早期训练带来的噪声。
- 训练末期增强：在 base model 接近收敛后，用小学习率和高质量数据做能力整理。
- 上下文扩展：在已有模型上加入长文档、代码仓库、多轮对话或合成长上下文数据。
- 知识更新：补充较新网页、论文、文档和领域资料，但需要控制遗忘和污染。

## 与从零预训练的区别

从零预训练需要从随机初始化开始学习词法、语法、知识和基本模式，训练 token 量巨大。CPT 则继承已有模型能力，通常使用更小学习率、更少 token 和更聚焦的数据。

因此，CPT 的风险不是“学不会基础语言”，而是：

- 领域数据过窄导致灾难性遗忘；
- 数据配比不当导致通用能力下降；
- 学习率过高破坏已有表示；
- 数据污染影响评测可信度。

## 训练目标与数据形态

CPT 通常继续使用 [[training/pretraining/objective|next-token prediction objective]]：

$$
\mathcal{L}_{\mathrm{CPT}}
= -\sum_t \log p_\theta(x_t \mid x_{<t})
$$

数据可以是：

- 原始文档：书籍、网页、论文、教材、手册；
- 代码仓库：源码、README、issue、测试、API 文档；
- 数学数据：题目、解答、证明、推理轨迹；
- 长上下文数据：长文档、多文档 QA、代码仓库、长对话；
- 高质量混合数据：训练末期 annealing 使用的 curated mix；
- 合成数据：teacher-generated reasoning、long-context QA、tool traces。

CPT 不是必须使用 prompt-response 格式。如果数据已经变成 instruction / answer 对，它更接近 SFT。实际项目中二者可以混合，但应清楚区分 loss mask、chat template 和训练目标。

## 关键超参

CPT 对已有模型继续优化，超参通常比从零预训练更保守：

| 超参 | 影响 |
|---|---|
| learning rate | 过高会破坏已有表示，过低则难以注入新能力 |
| warmup steps | 缓解切换数据分布后的 loss spike |
| token budget | 决定新领域注入强度和遗忘风险 |
| data mix | 控制目标能力与通用能力之间的平衡 |
| sequence length | 决定长上下文学习和 activation 成本 |
| replay ratio | 保留通用数据以降低遗忘 |
| checkpoint cadence | 便于选择最佳能力/遗忘折中点 |

常见做法是使用较低 learning rate、较短 warmup、明确的 held-out domain validation，并保留多个 checkpoint 做回退。

## 遗忘与能力迁移

CPT 的核心风险是 catastrophic forgetting。领域数据过窄时，模型可能在目标领域表现提升，但通用语言、代码、数学、多语言或指令跟随能力下降。

缓解方式包括：

- 在领域数据中混入通用高质量 replay data；
- 控制 CPT token budget；
- 使用较小 learning rate；
- 分阶段逐步提高领域比例；
- 保留多领域 validation；
- 在后续 SFT/RLHF 前评估 base capability；
- 对关键 benchmark 做 contamination 检查。

能力迁移也可能是正向的。例如，代码 CPT 可能提升结构化推理，数学 CPT 可能改善符号操作，长上下文 CPT 可能提升多文档整合。但这类迁移依赖数据质量和模型容量，不能假设自动发生。

## 与 SFT 的区别

SFT 训练的是指令响应格式和 assistant 行为，数据通常是 prompt-response。CPT 训练的是语言建模或预训练式目标，数据可以是原始文档、代码、教材、论文、网页或混合语料。

简单说：

- CPT 更像“继续读书”；
- SFT 更像“学习如何按用户请求回答”。

更形式化地说，CPT 主要改变模型对文本分布的建模能力，SFT 主要改变模型在交互协议中的条件响应行为。CPT 做得好，通常会让 SFT 更容易；但 CPT 不能替代 SFT 的 instruction following、safety policy 和 assistant style。

## 与 Reinforcement Pretraining 的关系

传统 CPT 通常仍使用 next-token prediction。[[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]] 则在 continued pretraining 阶段改变目标函数。

例如 [[sources/papers/2025-rlp-reinforcement-as-a-pretraining-objective|RLP]] 会让模型在预测下一 token 前先生成 thought，并根据 thought 对真实 token log-likelihood 的提升获得 reward。它和 CPT 的差异不只是训练数据，而是训练信号：CPT 直接最大化 token likelihood，RLP 优化的是 thought 的信息增益。

## 评测

CPT 至少需要同时评估：

- CPT 目标领域 validation loss；
- 通用 validation loss；
- 关键 benchmark；
- 长上下文或领域专项 benchmark；
- 训练数据 contamination；
- 与原始 base model 的差异；
- 后续 SFT 后能力是否保留。

如果只看目标领域提升，容易忽略通用能力退化。如果只看 benchmark，容易被污染或过拟合误导。

## 常见失败模式

- **领域数据过窄**：模型变得更像领域语料补全器，而不是通用 base。
- **学习率过高**：短期 loss 下降，但已有能力被破坏。
- **没有 replay data**：通用能力和多语言能力快速下降。
- **数据污染**：目标 benchmark 被 CPT 数据泄漏。
- **合成数据比例过高**：模型继承 teacher 风格和错误。
- **把 CPT 当 SFT**：没有训练 assistant 行为，却期待模型会更会对话。

## 相关概念

- [[training/pretraining/pretraining|Pretraining]]
- [[training/pretraining/objective|Training Objective]]
- [[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]]
- [[training/mid-training/domain-adaptation|Domain Adaptation]]
- [[training/mid-training/capability-injection|Capability Injection]]
- [[training/mid-training/long-context-training|Long Context Training]]
- [[training/mid-training/annealing|Annealing]]
- [[training/pretraining/data-mix|Data Mix]]
