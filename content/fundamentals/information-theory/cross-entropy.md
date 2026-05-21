---
title: Cross Entropy
created: 2025-12-27
published: 2025-12-27
modified: 2025-12-27
type: topic
status: growing
area: fundamentals
tags:
  - math
  - information-theory
  - loss-function
aliases:
  - Cross Entropy
  - Cross-entropy Loss
---

## 概念界定

交叉熵衡量当真实数据服从分布 `p`，却使用模型分布 `q` 来编码或预测时的平均代价。在深度学习中，交叉熵是分类任务和 next-token prediction 最常见的训练损失之一。

## 背景与问题

语言模型训练时，真实数据给出了每个位置上的目标 token。模型输出的是词表上的概率分布。我们需要一个损失函数，让模型给真实 token 更高概率，同时惩罚给真实 token 低概率的情况。交叉熵正好提供了这样的目标。

## 定义与记号

真实分布 `p` 和模型分布 `q` 的交叉熵定义为：

```text
H(p, q) = -Σ_x p(x) log q(x)
```

如果真实标签是 one-hot 分布，即真实 token 为 `y`，则交叉熵变为：

```text
H(p, q) = -log q(y)
```

这就是分类任务中的 negative log likelihood。

## 直观解释

交叉熵可以理解为：真实答案出现时，模型有多意外。如果模型给真实 token 的概率高，`-log q(y)` 小；如果概率低，损失大。

## 基本性质

- 交叉熵越低，说明模型分布对真实样本的预测越好。
- 当真实分布固定时，最小化交叉熵等价于最小化 `KL(p || q)`。
- one-hot 标签下，交叉熵只直接读取真实类别对应的模型概率。
- 交叉熵不直接衡量生成文本的事实性、推理能力或指令跟随能力。

## 示例

给定上下文：

```text
The capital of France is
```

真实下一个 token 是 `Paris`。如果模型给出：

```text
q(Paris) = 0.9
```

损失较小：

```text
-log 0.9
```

如果模型给出：

```text
q(Paris) = 0.01
```

损失很大：

```text
-log 0.01
```

## 与 next-token prediction 的关系

自回归语言模型通常最小化每个位置的交叉熵：

```text
L = -Σ_t log q_θ(x_t | x_<t)
```

这等价于最大化训练序列在模型下的似然。

## 常见误解

- 误解：交叉熵只是一个经验上好用的分类 loss。
  - 正确理解：它有明确的信息论和最大似然解释。
- 误解：交叉熵低就说明回答质量一定高。
  - 正确理解：交叉熵衡量 token 预测，不直接等价于复杂任务能力。
- 误解：交叉熵和熵是同一个东西。
  - 正确理解：熵只涉及真实分布，交叉熵涉及真实分布和模型分布。

## 相关概念

- [[fundamentals/information-theory/entropy|熵]] — 真实分布自身的不确定性。
- [[fundamentals/information-theory/kl-divergence|KL 散度]] — 交叉熵与分布差异的关系。
- [[fundamentals/information-theory/cross-entropy-kl-relation|交叉熵与 KL 的关系]] — 进一步解释二者分解。
- [[fundamentals/probability/maximum-likelihood|最大似然估计]] — 交叉熵与最大似然等价。
- [[training/pretraining/pretraining|预训练]] — next-token 交叉熵的主要应用。
