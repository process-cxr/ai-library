---
title: Residual in Transformer
created: 2026-01-25
published: 2026-01-25
modified: 2026-05-31
type: topic
status: mature
area: architecture
tags:
  - transformer
  - residual
---

Residual Connection 是深层 Transformer 能稳定训练的关键结构。它把子层输出作为对输入表示的增量修改，而不是让每一层完全重写 hidden state。Transformer block 中 attention 和 FFN 通常都包在 residual branch 中：

$$
H_{out}=H+F(H)
$$

其中 $F$ 可以是 attention 子层或 FFN 子层。

## 核心作用

Residual connection 的基本思想是保留一条直接信息通道。没有 residual 时，多层网络的输出是很多非线性变换连续复合，梯度需要穿过每一层的子模块；有 residual 后，梯度可以沿 identity path 更直接地传播。

这带来几个好处：

- 保留底层 token 表示，不让每层完全覆盖输入；
- 让每层学习“增量修正”而不是完整映射；
- 缓解深层网络中的梯度传播困难；
- 与 normalization 配合提高训练稳定性；
- 让模型可以堆叠更多层。

## 在 Transformer Block 中的位置

现代 decoder-only LLM 常用 Pre-Norm 结构：

$$
H' = H + \mathrm{Attention}(\mathrm{Norm}(H))
$$

$$
H_{out}=H' + \mathrm{FFN}(\mathrm{Norm}(H'))
$$

也就是：

```text
input H
  -> Norm -> Attention -> add back to H
  -> Norm -> FFN       -> add back to H'
output
```

每个 block 有两条主要 residual additions：一条包 attention，一条包 FFN。这样每层既能交换上下文信息，也能做逐 token 非线性变换，同时保留原有表示。

## Pre-Norm 与 Residual Path

Pre-Norm 的重要性在于 residual 主路径更接近 identity：

$$
H' = H + F(\mathrm{Norm}(H))
$$

这里 $H$ 可以直接传到下一层。相比 Post-Norm：

$$
H' = \mathrm{Norm}(H+F(H))
$$

Pre-Norm 通常更利于深层模型训练，因为梯度沿 residual path 传播时不必每次都先穿过 normalization。现代 LLM 大多采用 Pre-Norm 或其变体。

## 表示层面的直觉

可以把每一层 Transformer 看成对 token 表示做一次局部更新：

```text
new representation = old representation + learned update
```

Attention 的 update 来自上下文聚合；FFN 的 update 来自逐 token 非线性变换。Residual connection 让这些 update 叠加到原始表示上，而不是完全替换表示。

这对语言建模很重要：低层可能保留词形、局部语法和位置模式，高层逐渐加入语义、任务和推理相关信息。Residual path 让不同层级的信息可以更容易共存。

## 与尺度控制的关系

Residual addition 会把多个分支的信号相加。如果没有 normalization、初始化策略或 residual scaling，hidden state 尺度可能随层数增长而漂移。深层模型中常见的稳定化设计包括：

- [[architecture/transformer/normalization|Pre-Norm]]；
- RMSNorm / LayerNorm；
- 合适的 weight initialization；
- residual scaling；
- learning rate warmup；
- gradient clipping。

因此 residual connection 不是单独发挥作用，而是与 normalization 和优化 recipe 共同形成稳定训练条件。

## 与 Training Stability 的关系

训练很深的 Transformer 时，如果 residual/normalization 设计不合适，可能出现：

- loss spike；
- 梯度爆炸或消失；
- attention / FFN 输出尺度异常；
- mixed precision 下 NaN；
- 深层表示退化。

这也是为什么架构论文和模型报告经常强调 normalization placement、RMSNorm、residual scaling 等细节。它们看起来不像 attention 那样显眼，但直接影响模型能否稳定扩展。

## 常见误解

- **误解：Residual 只是把输入加回来，没什么信息。** 它为深层网络提供直接信息和梯度通道，是可训练性的核心条件。
- **误解：Residual 可以替代 normalization。** 二者作用不同，residual 保留路径，normalization 控制尺度。
- **误解：每层都必须大幅改变表示才有用。** 很多层可能只做小幅增量更新，这正是 residual 结构的自然用法。
- **误解：Residual 只影响训练，不影响表示。** 它也影响不同层级信息如何保留和组合。

## 相关概念

- [[architecture/transformer/transformer|Transformer]]
- [[architecture/transformer/normalization|Normalization in Transformer]]
- [[architecture/attention/attention|Attention]]
- [[architecture/transformer/feed-forward|Feed Forward Network]]
- [[training/optimization/training-stability|Training Stability]]
- [[training/optimization/loss-spike|Loss Spike]]
- [[fundamentals/neural-network-basics/residual-connection|Residual Connection]]

## 经典论文与资料

- [[sources/papers/2015-deep-residual-learning-for-image-recognition|Deep Residual Learning for Image Recognition]]
- [[sources/papers/2017-attention-is-all-you-need|Attention Is All You Need]]
