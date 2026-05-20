---
title: JS 散度
type: topic
status: seed
area: fundamentals
tags:
  - math
  - information-theory
  - divergence
aliases:
  - Jensen-Shannon Divergence
  - JS Divergence
---

## 概念界定

JS 散度是一种基于 KL 散度构造的分布差异度量。相比 KL 散度，它是对称的，并且通常更平滑。

## 定义与记号

给定两个分布 `p` 和 `q`，先定义混合分布：

```text
m = (p + q) / 2
```

JS 散度定义为：

```text
JS(p || q) = 1/2 KL(p || m) + 1/2 KL(q || m)
```

## 直观解释

JS 散度不是直接比较 `p` 和 `q`，而是分别比较它们与中间分布 `m` 的差异，再取平均。这使它比 KL 更对称，也避免某些 KL 极端情况。

## 基本性质

- JS 散度是对称的。
- JS 散度非负。
- 当两个分布相同时，JS 散度为 0。
- JS 散度常用于需要平滑比较两个分布的场景。

## 常见误解

- 误解：JS 散度和 KL 散度完全一样。
  - 正确理解：JS 基于 KL 构造，但具有对称性和更平滑的性质。
- 误解：JS 是大模型训练中最常见的损失。
  - 正确理解：大模型主线中更常见的是交叉熵、NLL 和 KL，JS 更多作为扩展理解。

## 相关概念

- [[fundamentals/information-theory/kl-divergence|KL 散度]] — JS 散度的构造基础。
- [[fundamentals/probability/distribution|概率分布]] — 分布差异的比较对象。
