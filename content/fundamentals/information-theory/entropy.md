---
title: Entropy
created: 2025-12-27
published: 2025-12-27
modified: 2025-12-27
type: topic
status: growing
area: fundamentals
tags:
  - math
  - information-theory
  - uncertainty
aliases:
  - Entropy
  - Shannon Entropy
---

## 概念界定

熵是随机变量不确定性的度量，也可以理解为事件自信息在整个概率分布下的期望。分布越均匀，熵通常越高；分布越集中，熵通常越低。

## 背景与问题

语言模型输出的是下一个 token 的概率分布。这个分布可能很确定，也可能很分散。熵提供了一种衡量模型在当前上下文下“不确定程度”的方式。

## 定义与记号

离散随机变量 `X` 的熵定义为：

```text
H(X) = -Σ_x p(x) log p(x)
```

也可以写成自信息的期望：

```text
H(X) = E_{x~p}[-log p(x)]
```

## 直观解释

如果一个分布几乎把全部概率放在一个 token 上，模型很确定，熵低。如果概率分散在很多 token 上，模型不确定，熵高。

例如：

```text
A: 0.95, B: 0.03, C: 0.02  -> 低熵
A: 0.34, B: 0.33, C: 0.33  -> 高熵
```

## 基本性质

- 熵非负。
- 对固定候选集合，均匀分布的熵最大。
- 分布越集中，熵越低。
- 熵是分布自身的不确定性，不需要另一个分布作为参照。

## 示例

在生成任务中，如果某一步上下文是：

```text
The capital of France is
```

模型可能对 `Paris` 给出很高概率，分布熵较低。

如果上下文是：

```text
Once upon a time
```

后续合理 token 很多，分布可能更分散，熵更高。

## 常见误解

- 误解：熵越低越好。
  - 正确理解：低熵表示分布更确定，但如果模型确定地预测错误 token，效果并不好。
- 误解：熵高说明模型更聪明。
  - 正确理解：高熵只表示不确定性更高，可能来自开放问题，也可能来自模型困惑。
- 误解：熵等于交叉熵。
  - 正确理解：熵只涉及真实分布自身，交叉熵涉及真实分布和模型分布。

## 相关概念

- [[fundamentals/information-theory/self-information|自信息]] — 熵是自信息的期望。
- [[fundamentals/information-theory/cross-entropy|交叉熵]] — 用模型分布编码真实分布的平均代价。
- [[fundamentals/probability/distribution|概率分布]] — 熵定义在概率分布上。
- [[fundamentals/probability/calibration-uncertainty|校准与不确定性]] — 熵可辅助理解不确定性但不等于校准。
