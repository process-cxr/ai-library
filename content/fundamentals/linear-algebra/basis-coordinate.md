---
title: 基与坐标
type: topic
status: growing
area: fundamentals
tags:
  - math
  - linear-algebra
  - representation
aliases:
  - Basis
  - Coordinate
---

## 概念界定

基是一组能够表示空间中任意向量的参考向量，坐标是某个向量在给定基下的数值表示。在大模型中，embedding 和 hidden state 可以看作高维表示空间中的坐标。

## 背景与问题

当我们说一个 token 被表示为向量时，容易误以为每个维度都有清晰的人类语义。实际上，向量坐标依赖表示空间的基，而这个空间通常由训练过程形成。理解基与坐标，有助于避免把 embedding 的单个维度过度解释为固定语义。

## 定义与记号

如果一组向量 `{e_1, e_2, ..., e_d}` 构成空间的基，那么任意向量 `x` 可以表示为：

```text
x = a_1 e_1 + a_2 e_2 + ... + a_d e_d
```

其中 `[a_1, a_2, ..., a_d]` 是 `x` 在这组基下的坐标。

## 直观解释

坐标不是对象本身，而是对象在某套参考方向下的表示。同一个几何向量，换一组基后坐标会变化，但它代表的空间对象没有变化。

对 embedding 来说，每个 token 的向量可以理解为模型在高维空间中给它分配的位置。但要注意，输入层的 token embedding 只是这个 token 的初始表示；经过多层 Transformer 之后，同一个 token 会因为上下文不同而形成不同的 hidden state。

例如同一个 `bank`：

```text
I sat by the bank of the river.
I deposited money in the bank.
```

如果 `bank` 对应同一个 token，那么它一开始查 embedding table 得到的 token embedding 是同一个向量：

```text
embedding("bank") = 同一个初始向量
```

但进入 Transformer 后，模型会通过 Attention 融合上下文信息。第一句里的 `bank` 会更多关联 `river`，深层 hidden state 会偏向“河岸”；第二句里的 `bank` 会更多关联 `money` 和 `deposited`，深层 hidden state 会偏向“银行”。因此：

```text
hidden_state("bank" | river context) ≠ hidden_state("bank" | money context)
```

所以，token embedding 是由 token id 决定的相对静态表示，而深层 hidden state 是由 token、上下文和层数共同决定的动态表示。

## 基本性质

- 同一向量在不同基下可以有不同坐标。
- 语义通常分布在多个方向上，而不是绑定到单个维度。
- 表示空间可以通过线性变换改变坐标结构。
- 深层 hidden state 是上下文相关表示，不等同于静态 token embedding。

## 示例

一个 token embedding：

```text
x_token ∈ R^{D}
```

可以看作在 `D` 维表示空间中的坐标。训练不会显式规定第 17 维表示“性别”或第 83 维表示“时态”；这些属性如果存在，往往体现为空间中的某些方向或子空间。

## 常见误解

- 误解：embedding 的每个维度都有独立可解释含义。
  - 正确理解：多数语义是分布式表示，单维解释通常不稳定。
- 误解：坐标接近就一定语义相同。
  - 正确理解：接近关系依赖训练目标、相似度度量和上下文。
- 误解：token embedding 和所有层的 hidden state 是同一类语义对象。
  - 正确理解：embedding 是输入表示，hidden state 是经过上下文和层变换后的动态表示。

## 相关概念

- [[fundamentals/neural-network-basics/embedding|Embedding]] — token 的向量表示。
- [[fundamentals/linear-algebra/dot-product-similarity|内积与相似度]] — 表示空间中的相似性度量。
- [[fundamentals/linear-algebra/linear-transformation|线性变换]] — 改变表示坐标的核心操作。
