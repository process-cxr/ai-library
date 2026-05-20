---
title: Vector Matrix and Tensor
created: 2026-01-17
published: 2026-01-17
modified: 2026-01-24
type: topic
status: growing
area: fundamentals
tags:
  - math
  - linear-algebra
  - tensor
aliases:
  - Vector
  - Matrix
  - Tensor
---

## 概念界定

向量、矩阵与张量是深度学习中表示数据、参数和中间计算结果的基本对象。对大模型而言，token embedding、hidden states、attention weights、logits、梯度和优化器状态都可以看作不同形状的张量。

## 背景与问题

自然语言本身无法直接参与神经网络计算。模型需要先把文本转换成 token id，再把 token id 映射为向量，并在多层网络中不断更新这些向量表示。理解这些对象的层级关系，是读懂模型结构、调试 shape 错误和估算显存占用的前提。

## 定义与记号

常见对象包括：

| 对象 | 记号示例 | 含义 | 大模型例子 |
|---|---|---|---|
| 标量 | `x` | 单个数值 | loss、learning rate |
| 向量 | `x ∈ R^d` | 一维数值数组 | 单个 token embedding |
| 矩阵 | `X ∈ R^{m×n}` | 二维数值数组 | 一个序列的 hidden states |
| 张量 | `X ∈ R^{B×T×D}` | 多维数值数组 | 一个 batch 的 hidden states |

典型 NLP 张量流：

```text
input_ids:     [B, T]
embeddings:    [B, T, D]
hidden states: [B, T, D]
logits:        [B, T, V]
```

其中 `B` 是 batch size，`T` 是 sequence length，`D` 是 hidden size，`V` 是 vocabulary size。

## 直观解释

可以把向量理解为“一个对象在多个数值维度上的表示”。一个 token embedding 不是 token 本身，而是模型用一组连续数值对这个 token 的表示。矩阵则可以看作一组向量的排列，例如一个句子中每个 token 一个向量，组合起来就是 `[T, D]` 的矩阵。张量是在矩阵之上继续增加 batch、head、layer 等轴。

## 基本性质

- 张量的语义由每个轴共同决定，不能只看元素个数。
- 同一个 shape 的张量可能表示完全不同的对象，例如 hidden states 和 embeddings 都可能是 `[B, T, D]`。
- 大模型中的参数也是张量，参数量就是这些权重张量元素数量的总和。
- 训练和推理的显存占用不仅来自参数，也来自激活、梯度、KV cache 和优化器状态。

## 示例

一个 Transformer 层的输入通常是：

```text
X: [B, T, D]
```

经过 Q/K/V 投影后得到：

```text
Q: [B, H, T, Dh]
K: [B, H, T, Dh]
V: [B, H, T, Dh]
```

这里 `H * Dh = D` 通常成立。理解这个变化，是理解 [[architecture/attention/attention|Attention 机制]] 的前提。

## 常见误解

- 误解：向量就是一个词的语义。
  - 正确理解：向量只是模型内部的数值表示，语义来自训练目标、上下文和表示空间结构。
- 误解：张量维度越高，模型一定越强。
  - 正确理解：维度提供表示容量，但效果取决于数据、结构、优化和训练目标。
- 误解：只要知道公式就能读懂模型。
  - 正确理解：很多模型理解和工程排错都首先依赖 shape 追踪。

## 相关概念

- [[fundamentals/linear-algebra/shape-dimension|形状与维度]] — 进一步区分张量轴和隐藏维度。
- [[fundamentals/neural-network-basics/embedding|Embedding]] — token 如何映射为向量表示。
- [[architecture/transformer/transformer|Transformer]] — 张量如何在模型层中流动。
