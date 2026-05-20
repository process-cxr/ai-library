---
title: Conditional Probability
created: 2026-01-21
published: 2026-01-21
modified: 2026-01-28
type: topic
status: growing
area: fundamentals
tags:
  - math
  - probability
  - language-modeling
aliases:
  - Conditional Probability
---

## 概念界定

条件概率描述在某个条件已经发生或已知的情况下，另一个事件发生的概率。在语言模型中，下一个 token 的概率总是在已有上下文条件下计算的。

## 背景与问题

自然语言具有强上下文依赖。同一个 token 在不同上下文中的合理性不同。语言模型的核心任务不是学习孤立 token 的概率，而是学习：给定前文后，下一个 token 应该如何分布。

## 定义与记号

条件概率定义为：

```text
p(A | B) = p(A, B) / p(B)
```

其中 `p(A | B)` 表示在 `B` 已知的条件下，`A` 发生的概率。

自回归语言模型通常建模：

```text
p(x_t | x_<t)
```

其中 `x_<t` 表示第 `t` 个 token 之前的所有上下文。

## 直观解释

条件概率表达“上下文改变概率”。例如：

```text
I deposited money in the ____
I sat by the river ____
```

同一个候选 token `bank` 在两个上下文中的概率不同。语言模型通过上下文 hidden state 来产生这个条件分布。

## 基本性质

- 条件不同，概率分布可以完全不同。
- 条件概率不是因果关系，只表示在给定条件下的概率关系。
- 自回归模型把生成过程写成逐 token 的条件概率序列。
- 上下文越长，可用条件信息越多，但推理成本也通常越高。

## 示例

给定上下文：

```text
x_<t = "The capital of France is"
```

模型估计：

```text
p(X_t = "Paris" | x_<t)
```

这个概率应当远高于无条件情况下随机出现 `Paris` 的概率。

## 常见误解

- 误解：条件概率表示因果关系。
  - 正确理解：条件概率只表示概率关系，因果需要额外假设。
- 误解：语言模型记住了固定答案，所以不需要概率。
  - 正确理解：模型总是在上下文条件下产生一个分布。
- 误解：上下文越长总是越好。
  - 正确理解：长上下文带来更多信息，也带来注意力成本、噪声和位置泛化问题。

## 相关概念

- [[fundamentals/probability/chain-rule|概率链式法则]] — 序列概率由条件概率分解而来。
- [[architecture/attention/attention|Attention 机制]] — 模型融合上下文信息的重要机制。
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]] — 自回归条件生成的推理缓存机制。
