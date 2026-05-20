---
title: Joint and Conditional Entropy
created: 2026-01-28
published: 2026-01-28
modified: 2026-02-04
type: topic
status: seed
area: fundamentals
tags:
  - math
  - information-theory
  - entropy
aliases:
  - Joint Entropy
  - Conditional Entropy
---

## 概念界定

联合熵度量多个随机变量整体的不确定性，条件熵度量在已知某些变量后剩余的不确定性。它们用于分析上下文信息如何减少预测不确定性。

## 定义与记号

联合熵：

```text
H(X, Y) = -Σ_x Σ_y p(x, y) log p(x, y)
```

条件熵：

```text
H(Y | X) = -Σ_x p(x) Σ_y p(y | x) log p(y | x)
```

也可以写为：

```text
H(Y | X) = H(X, Y) - H(X)
```

## 直观解释

条件熵回答：如果已经知道 `X`，那么 `Y` 还剩多少不确定性。对于语言模型，给定上下文后，下一个 token 的不确定性通常会降低。

## 示例

无上下文时，下一个 token 的可能性很多；给定上下文：

```text
The capital of France is
```

下一个 token 的不确定性显著降低，条件熵变小。

## 相关概念

- [[fundamentals/information-theory/entropy|熵]] — 单变量不确定性。
- [[fundamentals/information-theory/mutual-information|互信息]] — 条件信息减少了多少不确定性。
- [[fundamentals/probability/conditional-probability|条件概率]] — 条件熵依赖条件分布。
