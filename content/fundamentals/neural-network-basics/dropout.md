---
title: Dropout
created: 2026-01-31
published: 2026-01-31
modified: 2026-02-07
type: topic
status: seed
area: fundamentals
tags:
  - neural-network
  - regularization
aliases:
  - Dropout
---

## 概念界定

Dropout 是训练时随机将部分神经元或激活置零的正则化方法，用于降低模型对特定路径的依赖，缓解过拟合。

## 背景与问题

神经网络参数很多时，可能记住训练数据中的偶然模式。Dropout 通过训练时随机丢弃部分激活，让模型不能过度依赖某些特征组合，从而提高泛化能力。

## 结构与机制

训练时，对激活 `x` 采样 mask：

```text
m_i ~ Bernoulli(1 - p)
```

然后：

```text
y = x ⊙ m / (1 - p)
```

其中 `p` 是 dropout probability。

推理时通常不再随机丢弃，而是使用完整网络。

## 直观解释

Dropout 像是在训练时随机关闭一部分通路，迫使网络学到更稳健的表示，而不是依赖某几个固定神经元。

## 基本性质

- Dropout 只在训练时启用，推理时通常关闭。
- Dropout 会引入随机性。
- 大模型中 dropout 使用比例与模型规模、数据量和训练配方有关。
- 数据足够大、模型规模足够大时，有些 LLM 训练会使用较低 dropout 或不用 dropout。

## 常见误解

- 误解：Dropout 一定能提升大模型效果。
  - 正确理解：它是否有用取决于数据规模、模型规模和训练设置。
- 误解：Dropout 推理时也会随机关闭神经元。
  - 正确理解：标准 dropout 推理时关闭随机失活。

## 相关概念

- [[fundamentals/linear-algebra/elementwise-broadcasting|逐元素运算与广播]] — dropout mask 是逐元素乘法。
- [[training/pretraining/pretraining|预训练]] — 大规模预训练中 dropout 使用需要结合训练配方。
