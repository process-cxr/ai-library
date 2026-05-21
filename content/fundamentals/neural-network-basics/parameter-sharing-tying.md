---
title: Parameter Sharing and Weight Tying
created: 2026-01-10
published: 2026-01-10
modified: 2026-01-10
type: topic
status: seed
area: fundamentals
tags:
  - neural-network
  - parameter-sharing
  - language-modeling
aliases:
  - Parameter Sharing
  - Weight Tying
---

## 概念界定

参数共享是让模型的不同部分使用同一组参数，权重绑定是参数共享的一种形式。在语言模型中，常见做法是将输入 embedding 矩阵和输出 LM Head 权重绑定。

## 背景与问题

语言模型既需要把 token id 映射为输入向量，也需要把 hidden state 映射回词表 logits。这两个过程都和词表表示有关，因此可以共享权重，减少参数并增强输入输出表示的一致性。

## 结构与机制

输入 embedding：

```text
E: [V, D]
```

输出头通常可写为：

```text
W_vocab: [D, V]
```

权重绑定时，可以使用：

```text
W_vocab = E^T
```

## 直观解释

输入阶段问：“这个 token 应该映射到哪个向量？”输出阶段问：“这个 hidden state 更像哪个 token 的向量？”权重绑定让这两个问题共享同一套 token 表示空间。

## 基本性质

- 权重绑定可以减少参数量。
- 它可能改善输入输出表示的一致性。
- 是否绑定取决于模型架构和训练设计。
- 当词表很大时，embedding / output head 参数量非常可观。

## 常见误解

- 误解：权重绑定只是为了省参数。
  - 正确理解：它也体现了输入 token 表示和输出 token 分类之间的结构联系。
- 误解：所有语言模型都必须绑定权重。
  - 正确理解：这是常见设计之一，但不是必然要求。

## 相关概念

- [[fundamentals/neural-network-basics/embedding|Embedding]] — 输入 token 表示。
- [[fundamentals/neural-network-basics/logits-output-head|Logits 与输出头]] — 输出词表分类层。
- [[fundamentals/information-theory/perplexity|困惑度]] — 输出分布影响语言建模评估。
