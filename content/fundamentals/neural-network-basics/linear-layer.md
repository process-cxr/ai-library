---
title: Linear Layer
created: 2026-02-08
published: 2026-02-08
modified: 2026-02-15
type: topic
status: growing
area: fundamentals
tags:
  - neural-network
  - linear-layer
  - transformer
aliases:
  - Linear Layer
  - Fully Connected Layer
---

## 概念界定

线性层是神经网络中最基本的可学习变换模块，通常将输入向量通过权重矩阵和偏置映射到新的表示空间。在大模型中，Q/K/V 投影、MLP 升维降维和输出头都依赖线性层。

## 背景与问题

大模型需要不断改变 token 的表示，使其更适合上下文理解和下一个 token 预测。线性层提供了最基本的表示投影能力，是 Transformer 中高频出现的计算模块。

## 结构与机制

线性层通常写作：

```text
Y = XW + b
```

如果：

```text
X: [B, T, D_in]
W: [D_in, D_out]
b: [D_out]
```

则：

```text
Y: [B, T, D_out]
```

## 直观解释

线性层可以理解为一个可学习的坐标变换。它把输入表示中的信息重新组合，投影到新的空间中。

## 基本性质

- 线性层本身不引入非线性。
- 多个纯线性层连续堆叠可以合并成一个线性层。
- 线性层参数量为 `D_in × D_out`，如果有 bias 还要加 `D_out`。
- 大模型中的主要计算量往往来自大规模线性层和矩阵乘法。

## 示例

Attention 中的 Q/K/V 投影：

```text
Q = XW_q
K = XW_k
V = XW_v
```

MLP 中的升维与降维：

```text
[B, T, D] -> [B, T, D_ff] -> [B, T, D]
```

## 常见误解

- 误解：线性层只能表达线性关系，所以不重要。
  - 正确理解：线性层与非线性激活、attention、归一化和残差组合后可以构成复杂函数。
- 误解：bias 总是必须的。
  - 正确理解：许多大模型结构中会省略部分 bias，具体取决于架构设计。
- 误解：线性层只是分类头。
  - 正确理解：Transformer 内部大量投影都属于线性层。

## 相关概念

- [[fundamentals/linear-algebra/matrix-multiplication|矩阵乘法]] — 线性层的核心计算。
- [[fundamentals/linear-algebra/linear-transformation|线性变换]] — 线性层的数学解释。
- [[architecture/attention/attention|Attention 机制]] — Q/K/V 线性投影。
- [[fundamentals/neural-network-basics/feed-forward-network|前馈网络]] — 多个线性层和激活函数的组合。
