---
title: Probability Distribution
created: 2025-12-14
published: 2025-12-14
modified: 2025-12-14
type: topic
status: growing
area: fundamentals
tags:
  - math
  - probability
  - distribution
aliases:
  - Probability Distribution
---

## 概念界定

概率分布描述随机变量在不同取值上的概率分配。在语言模型中，模型对下一个 token 的预测就是一个定义在词表上的离散概率分布。

## 背景与问题

大模型不是只输出一个 token，而是为词表中每个候选 token 计算一个概率。解码策略再根据这个分布选择或采样下一个 token。因此，理解分布是理解生成、多样性、置信度和训练损失的基础。

## 定义与记号

对于离散随机变量 `X`，概率分布可以写作：

```text
p(X = x)
```

要求：

```text
p(x) ≥ 0
Σ_x p(x) = 1
```

语言模型输出 logits 后，通过 softmax 得到分布：

```text
logits: [V]
probabilities = softmax(logits): [V]
```

其中 `V` 是 vocabulary size。

## 直观解释

概率分布可以理解为模型对所有候选 token 的“质量分配”。如果某个 token 概率很高，说明在当前上下文中模型认为它更合理；但这不等于它一定真实正确。

## 基本性质

- 分布必须非负且总和为 1。
- 分布可以尖锐，也可以平坦。
- softmax 会把任意实数 logits 转换为概率分布。
- 分布的形状会影响生成结果的确定性和多样性。

## 示例

上下文：

```text
The capital of France is
```

模型可能输出：

```text
p(Paris) = 0.82
p(Lyon) = 0.03
p(the) = 0.01
...
```

贪心解码会选择概率最高的 `Paris`；采样解码则可能按概率随机选择。

## 常见误解

- 误解：概率最高的 token 就是模型“知道”的答案。
  - 正确理解：它只是当前模型分布下最可能的输出。
- 误解：softmax 后的概率天然可靠。
  - 正确理解：模型可能未校准，概率大小不一定等于真实正确率。
- 误解：分布越尖锐越好。
  - 正确理解：尖锐分布更确定，但可能降低多样性并放大过度自信。

## 相关概念

- [[fundamentals/probability/random-variable|随机变量]] — 分布描述随机变量取值概率。
- [[fundamentals/neural-network-basics/softmax|Softmax]] — logits 到概率分布的转换。
- [[fundamentals/probability/sampling|采样]] — 从分布中生成具体 token。
- [[fundamentals/probability/calibration-uncertainty|校准与不确定性]] — 分布概率与真实可靠性的关系。
