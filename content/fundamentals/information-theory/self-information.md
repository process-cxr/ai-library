---
title: Self-Information
created: 2025-12-21
published: 2025-12-21
modified: 2025-12-21
type: topic
status: growing
area: fundamentals
tags:
  - math
  - information-theory
aliases:
  - Self Information
  - Surprisal
---

## 概念界定

自信息用于度量单个事件发生时带来的信息量，也可以理解为事件的“意外程度”。事件概率越低，发生时带来的信息量越大；事件概率越高，发生时带来的信息量越小。

## 背景与问题

语言模型训练和评估经常关心真实 token 对模型来说有多“意外”。如果模型给真实 token 分配高概率，那么这个 token 的负对数概率较低；如果模型给真实 token 分配低概率，那么它对模型来说很意外，损失也会更高。

## 定义与记号

事件 `x` 的自信息定义为：

```text
I(x) = -log p(x)
```

如果使用以 2 为底的对数，单位是 bit；如果使用自然对数，单位是 nat。

## 直观解释

自信息可以理解为“知道这件事发生后，我获得了多少信息”。一个几乎必然发生的事件没有太多信息量；一个很罕见的事件一旦发生，就会提供更多信息。

## 基本性质

- `p(x)` 越小，`I(x)` 越大。
- `p(x)` 越大，`I(x)` 越小。
- 自信息非负。
- 自信息是交叉熵和负对数似然的基本组成。

## 示例

如果模型给真实 token 的概率是：

```text
p(x_t | x_<t) = 0.8
```

则该 token 的 surprisal 较低。如果概率是：

```text
p(x_t | x_<t) = 0.001
```

则该 token 对模型来说非常意外，训练损失会很大。

## 常见误解

- 误解：自信息表示事件本身的语义重要性。
  - 正确理解：自信息只由概率决定，不直接等于人类语义重要性。
- 误解：低概率事件一定有价值。
  - 正确理解：低概率只表示意外，不代表有用、正确或重要。

## 相关概念

- [[fundamentals/information-theory/entropy|熵]] — 自信息在分布下的期望。
- [[fundamentals/information-theory/negative-log-likelihood|负对数似然]] — 真实样本的自信息作为损失。
- [[fundamentals/information-theory/cross-entropy|交叉熵]] — 真实分布下模型自信息的平均值。
