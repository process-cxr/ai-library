---
title: Cross Entropy and KL Divergence
created: 2026-01-22
published: 2026-01-22
modified: 2026-01-29
type: topic
status: growing
area: fundamentals
tags:
  - math
  - information-theory
  - loss-function
aliases:
  - Cross Entropy KL Relation
---

## 概念界定

交叉熵可以分解为真实分布自身的熵和真实分布到模型分布的 KL 散度。因此，在真实分布固定时，最小化交叉熵等价于最小化 `KL(p || q)`。

## 背景与问题

深度学习中常说“最小化交叉熵就是让模型分布接近真实分布”。这句话背后的信息论解释就是交叉熵与 KL 的分解关系。

## 定义与记号

交叉熵定义为：

```text
H(p, q) = -Σ_x p(x) log q(x)
```

KL 散度定义为：

```text
KL(p || q) = Σ_x p(x) log(p(x) / q(x))
```

展开 KL：

```text
KL(p || q)
= Σ_x p(x) log p(x) - Σ_x p(x) log q(x)
= -H(p) + H(p, q)
```

因此：

```text
H(p, q) = H(p) + KL(p || q)
```

## 直观解释

交叉熵包含两部分：

- `H(p)`：真实分布本身的不确定性。
- `KL(p || q)`：模型分布 `q` 相对真实分布 `p` 的额外代价。

训练时数据分布 `p` 固定，`H(p)` 不随模型参数变化。因此，优化模型只能降低 KL 部分。

## 基本性质

- 当 `q = p` 时，KL 为 0，交叉熵达到最小值 `H(p)`。
- 交叉熵不可能低于真实分布自身的熵。
- 如果真实标签是 one-hot，交叉熵退化为真实类别的负 log probability。
- 这个关系解释了交叉熵为什么可以作为分布拟合目标。

## 示例

在 next-token prediction 中，真实分布通常由数据样本给出。模型最小化：

```text
-log q_θ(x_t | x_<t)
```

从分布角度看，就是让模型条件分布 `q_θ(. | x_<t)` 接近数据条件分布。

## 常见误解

- 误解：交叉熵越低就可以低于真实熵。
  - 正确理解：理论上最优模型的交叉熵下界是真实分布熵。
- 误解：KL 是训练中直接算出来的。
  - 正确理解：很多监督训练中只计算交叉熵；KL 分解是解释目标的方式。
- 误解：真实分布 `p` 总是已知。
  - 正确理解：实践中通常用训练数据的经验分布近似真实分布。

## 相关概念

- [[fundamentals/information-theory/entropy|熵]] — 真实分布自身的不确定性。
- [[fundamentals/information-theory/cross-entropy|交叉熵]] — 训练中常直接优化的目标。
- [[fundamentals/information-theory/kl-divergence|KL 散度]] — 分布差异项。
- [[fundamentals/probability/empirical-distribution|经验分布]] — 实践中用于近似真实分布。
