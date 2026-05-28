---
title: Training Objective
created: 2026-02-22
published: 2026-02-22
modified: 2026-05-28
type: topic
status: growing
area: training
tags:
  - pretraining
  - objective
---

预训练目标定义了模型在大规模数据上“为什么被奖励”。它不仅决定 loss 怎么算，也决定模型会被显式鼓励形成哪些行为。

## Next-token Prediction

Decoder-only LLM 最常见的预训练目标是 next-token prediction，也叫 causal language modeling。给定 token 序列 $x_1,\dots,x_T$，模型在每个位置根据前缀 $x_{<t}$ 预测真实 token $x_t$：

$$
\mathcal{L}_{NTP}(\theta)=-\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})
$$

从概率角度看，这等价于最大化训练语料的条件似然；从信息论角度看，它是在最小化数据分布 $p^*$ 和模型分布 $p_\theta$ 之间的 [[fundamentals/information-theory/cross-entropy|Cross Entropy]]。

## 为什么这个目标强大

Next-token prediction 的优势是极强的可扩展性：

- 数据不需要人工标注，只要有文本序列即可。
- 每个 token 都提供一个监督信号，训练信号密集。
- 目标与自回归生成过程一致，适合 decoder-only 模型。
- 同一个目标可以覆盖语法、事实、代码、推理痕迹和对话模式。

这也是现代 base model 能从海量语料中获得通用能力的原因。

## 目标的边界

NTP 虽然能隐式学习推理模式，但它并不显式奖励模型“先思考再预测”。如果训练语料中没有清晰推理轨迹，模型只能从最终文本分布中间接学习。即使语料包含 Chain-of-Thought，NTP 也只是模仿这些 token，而不是直接判断某段 thought 是否真的提升预测或解决问题。

因此，很多推理能力通常在后训练阶段通过 [[training/post-training/sft|SFT]]、[[training/post-training/rlhf|RLHF]]、[[training/post-training/grpo|GRPO]] 或 RLVR 被进一步强化。

## Reinforcement-style Pretraining Objective

[[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]] 是对 NTP 边界的一种探索。[[sources/papers/2025-rlp-reinforcement-as-a-pretraining-objective|RLP]] 把 Chain-of-Thought 看成预测下一 token 前的 action，并用信息增益作为 reward：

$$
r(c_t)=\log p_\theta(x_t \mid x_{<t}, c_t)-\log \bar{p}_\phi(x_t \mid x_{<t})
$$

如果 thought $c_t$ 让模型对真实 next token 的概率高于 no-think EMA baseline，它就获得正奖励。这个目标仍然依赖普通文本和 teacher forcing，但它优化的是“生成有预测价值的 thought”，而不只是直接最大化真实 token likelihood。

这类目标说明：预训练目标不一定只能是 likelihood。更一般地，它可以把普通文本转化为密集 reward，用来激励探索、推理或更好的内部表示。

## 相关概念

- [[training/pretraining/pretraining|Pretraining]]
- [[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]]
- [[fundamentals/information-theory/cross-entropy|Cross Entropy]]
- [[fundamentals/probability/maximum-likelihood|Maximum Likelihood]]
- [[training/post-training/grpo|GRPO]]
