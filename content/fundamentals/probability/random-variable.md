---
title: Random Variable
created: 2025-12-14
published: 2025-12-14
modified: 2025-12-14
type: topic
status: growing
area: fundamentals
tags:
  - math
  - probability
  - language-modeling
aliases:
  - Random Variable
---

## 概念界定

随机变量是把随机试验的结果映射为数值或符号对象的函数。在大模型中，token、标签、序列、采样结果和模型预测都可以从随机变量的角度理解。

## 背景与问题

语言生成不是一次性输出固定文本，而是逐步预测下一个 token。每个位置上的 token 都可以看作一个随机变量，模型要学习这些随机变量在上下文条件下的概率分布。

## 定义与记号

随机变量通常记为大写字母，例如：

```text
X, Y, X_t
```

它的一个具体取值用小写表示：

```text
X = x
X_t = x_t
```

在语言模型中，可以把第 `t` 个 token 表示为随机变量：

```text
X_t ∈ Vocabulary
```

给定前文 `x_<t`，模型估计：

```text
p(X_t = x_t | X_<t = x_<t)
```

## 直观解释

随机变量不是“随机的数”这么简单，而是对不确定结果的形式化表示。对于语言模型来说，下一个 token 不是唯一确定的：在同一个上下文后面，可能接不同 token，只是概率不同。

例如：

```text
The capital of France is ____
```

`Paris` 的概率应当很高，但模型仍然会给其他 token 分配较小概率。

## 基本性质

- 随机变量可以是离散的，也可以是连续的。
- NLP 中 token 通常是离散随机变量。
- Embedding 是 token 取值后的连续向量表示，不是 token 随机变量本身。
- 序列可以看作多个随机变量组成的随机过程。

## 示例

一句话可以表示为随机变量序列：

```text
X_1, X_2, ..., X_T
```

自回归语言模型逐步建模：

```text
p(X_1, X_2, ..., X_T)
```

并通过链式法则分解为多个条件概率。

## 常见误解

- 误解：随机变量必须是数字。
  - 正确理解：随机变量可以表示类别，token 可以看作离散类别取值。
- 误解：模型生成时采样了 token，就说明模型内部所有东西都是随机的。
  - 正确理解：模型前向计算通常是确定的，随机性主要来自解码采样策略。
- 误解：embedding 向量就是随机变量本身。
  - 正确理解：token 是离散随机变量的取值，embedding 是该取值进入模型后的连续表示。

## 相关概念

- [[fundamentals/probability/distribution|概率分布]] — 随机变量取不同值的概率。
- [[fundamentals/probability/conditional-probability|条件概率]] — 上下文条件下的 token 概率。
- [[fundamentals/neural-network-basics/embedding|Embedding]] — token 取值后的向量表示。
