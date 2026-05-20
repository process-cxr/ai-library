---
title: Embedding
type: topic
status: growing
area: fundamentals
tags:
  - neural-network
  - representation
  - nlp
aliases:
  - Token Embedding
  - Word Embedding
---

## 概念界定

Embedding 是把离散对象映射为连续向量表示的方法。在 NLP 和大模型中，token embedding 将 token id 映射为向量，使离散文本可以进入神经网络进行计算。

## 背景与问题

神经网络不能直接处理字符串。文本需要先经过 tokenizer 转成 token id，再通过 embedding table 查表得到连续向量。Embedding 是语言模型从离散符号世界进入连续表示空间的入口。

## 结构与机制

假设词表大小为 `V`，隐藏维度为 `D`，embedding table 是一个矩阵：

```text
E ∈ R^{V × D}
```

给定 token id `i`，对应 embedding 为：

```text
x_i = E[i]
```

如果输入是一个序列：

```text
input_ids: [B, T]
```

查表后得到：

```text
embeddings: [B, T, D]
```

## 直观解释

Embedding 可以理解为给每个 token 分配一个可学习的坐标。训练过程中，模型会调整这些坐标，使它们更适合预测下一个 token。

但 embedding 不是词义词典。它只是输入层的初始表示，后续 Transformer 层会结合上下文不断更新 hidden state。

## 基本性质

- Embedding 是可学习参数。
- Token embedding 通常只由 token id 决定，是相对静态的输入表示。
- Hidden state 是经过上下文和多层网络更新后的动态表示。
- Position embedding 或 RoPE 等位置机制用于补充顺序信息。
- Embedding table 的参数量通常是 `V × D`。

## 示例

输入：

```text
"I love AI"
```

经过 tokenizer：

```text
[40, 3021, 9552]
```

查 embedding table：

```text
E[40], E[3021], E[9552]
```

得到：

```text
X: [3, D]
```

加上 batch 后：

```text
X: [B, T, D]
```

## 常见误解

- 误解：embedding 的每个维度都有固定可解释语义。
  - 正确理解：语义通常分布在多个方向和子空间中。
- 误解：embedding 就是模型最终理解的语义。
  - 正确理解：embedding 是初始表示，深层 hidden state 才融合上下文。
- 误解：词表越大 embedding 一定越好。
  - 正确理解：词表大小会影响 token 粒度、参数量和建模方式，但不是单调越大越好。

## 相关概念

- [[fundamentals/linear-algebra/vector-matrix|向量、矩阵与张量]] — embedding 是向量表示。
- [[fundamentals/linear-algebra/basis-coordinate|基与坐标]] — embedding 空间的坐标视角。
- [[fundamentals/neural-network-basics/logits-output-head|Logits 与输出头]] — embedding 与输出头可能共享权重。
- [[architecture/transformer/transformer|Transformer]] — embedding 是 Transformer 输入表示。
