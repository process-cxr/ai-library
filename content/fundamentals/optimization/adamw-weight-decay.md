---
title: AdamW and Weight Decay
created: 2026-02-06
published: 2026-02-06
modified: 2026-02-13
type: topic
status: growing
area: fundamentals
tags:
  - optimization
  - adamw
  - regularization
aliases:
  - AdamW
  - Weight Decay
---

## 概念界定

AdamW 是 Adam 的常用变体，它将 weight decay 与 Adam 的自适应梯度更新解耦。Weight decay 是一种通过衰减权重规模来控制模型复杂度的正则化方法。

## 背景与问题

在 SGD 中，L2 正则化和 weight decay 在形式上关系紧密；但在 Adam 这类自适应优化器中，直接把 L2 正则项加入梯度会与自适应缩放耦合，行为不再等价。AdamW 通过解耦权重衰减，使正则化效果更清晰。

## 定义与记号

AdamW 的更新可以粗略理解为两部分：

```text
θ_{t+1} = θ_t - η · adam_update - η · λ θ_t
```

其中：

- `adam_update` 来自 Adam 的自适应梯度更新。
- `λ` 是 weight decay 系数。
- `η · λ θ_t` 是对参数本身的衰减项。

## 直观解释

Adam 负责根据梯度更新参数，weight decay 负责让权重不要无限变大。AdamW 的关键是把这两件事分开做，而不是把 weight decay 混进梯度里再被 Adam 自适应缩放。

## 基本性质

- AdamW 是大模型预训练和微调中常见优化器。
- Weight decay 通常不应用于所有参数，例如 bias、LayerNorm/RMSNorm 参数常被排除。
- Weight decay 可以帮助控制权重规模，但不是万能正则化。
- 衰减过强可能损害模型表达能力或训练收敛。

## 示例

常见参数分组：

```text
apply weight decay: linear weights, attention weights, MLP weights
no weight decay: bias, norm weights
```

具体分组取决于模型结构和训练配方。

## 常见误解

- 误解：AdamW 只是 Adam 加 L2 正则。
  - 正确理解：AdamW 的重点是 decoupled weight decay，即权重衰减与自适应梯度更新解耦。
- 误解：所有参数都应该 weight decay。
  - 正确理解：norm 和 bias 参数通常不做 weight decay。
- 误解：weight decay 越大泛化越好。
  - 正确理解：过强衰减会限制模型容量和收敛。

## 相关概念

- [[adam|Adam]] — AdamW 的基础优化器。
- [[regularization|正则化]] — weight decay 是正则化方法之一。
- [[fundamentals/neural-network-basics/normalization|归一化]] — norm 参数常排除 weight decay。
