---
title: 残差连接
type: topic
status: growing
area: fundamentals
tags:
  - neural-network
  - residual
  - transformer
aliases:
  - Residual Connection
  - Skip Connection
---

## 概念界定

残差连接是把子层输出与原始输入相加的结构，使网络学习的是对输入的增量更新，而不是完全重新生成表示。它是深层 Transformer 可训练性的关键结构之一。

## 背景与问题

随着网络加深，梯度传播和表示保持会变得困难。残差连接提供了一条直接的信息通路和梯度通路，使深层模型更容易优化。

## 结构与机制

基本形式：

```text
y = x + F(x)
```

其中 `F(x)` 是某个子层，例如 Attention 或 FFN。

Pre-Norm Transformer 中常见：

```text
x = x + Attention(Norm(x))
x = x + FFN(Norm(x))
```

## 直观解释

残差连接让每一层不必从零开始构造新表示，而是在已有表示上做修改。模型可以选择保留原信息，也可以通过子层增加新信息。

## 基本性质

- 残差连接要求相加的两个张量 shape 一致。
- 它改善深层网络的梯度传播。
- 它让网络更容易学习接近恒等映射的函数。
- 与归一化、初始化和学习率共同影响训练稳定性。

## 示例

如果 Attention 子层暂时学不到有用信息，残差结构仍允许：

```text
x_new ≈ x
```

这比强迫每层完全重写表示更稳定。

## 常见误解

- 误解：残差连接只是把输入复制一份。
  - 正确理解：它提供可学习增量更新和梯度通路。
- 误解：有残差连接就一定不会训练不稳定。
  - 正确理解：残差需要和归一化、初始化、优化器等共同配合。
- 误解：残差连接只属于 ResNet。
  - 正确理解：Transformer block 中也大量使用残差连接。

## 相关概念

- [[fundamentals/neural-network-basics/normalization|归一化]] — 与残差连接共同影响稳定性。
- [[fundamentals/linear-algebra/elementwise-broadcasting|逐元素运算与广播]] — 残差相加要求 shape 兼容。
- [[architecture/transformer/transformer|Transformer]] — 每个 block 中常有多个残差路径。
