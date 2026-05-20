---
title: Gradient Descent
created: 2026-02-05
published: 2026-02-05
modified: 2026-02-12
type: topic
status: growing
area: fundamentals
tags:
  - math
  - optimization
aliases:
  - Gradient Descent
---

## 概念界定

梯度下降是一类利用损失函数梯度更新参数的优化方法。它的核心思想是：沿着损失函数下降最快的方向移动参数，使损失逐步减小。

## 背景与问题

神经网络训练的目标通常是最小化损失函数 `L(θ)`。模型参数 `θ` 数量巨大，无法靠人工搜索找到最优值。梯度提供了局部方向信息，让模型可以通过迭代更新逐步改善。

## 定义与记号

基本更新公式：

```text
θ_{t+1} = θ_t - η ∇_θ L(θ_t)
```

其中：

- `θ_t`：第 `t` 步的参数。
- `η`：学习率。
- `∇_θ L(θ_t)`：损失对参数的梯度。

如果使用 mini-batch 估计梯度：

```text
g_t = (1/B) Σ_i ∇_θ L_i(θ_t)
θ_{t+1} = θ_t - η g_t
```

## 直观解释

可以把损失函数想象成高维地形，参数是地形上的一个点。梯度指向当前点损失上升最快的方向，因此负梯度方向就是局部下降最快方向。

## 基本性质

- 学习率过大可能震荡或发散。
- 学习率过小会导致训练很慢。
- mini-batch 梯度是对真实梯度的随机估计。
- 深度学习优化通常是高维非凸问题，不保证找到全局最优。
- 梯度方向只反映局部信息。

## 示例

语言模型预训练中，参数更新可以粗略写成：

```text
loss = CrossEntropy(model(input), target)
gradients = backward(loss)
parameters = parameters - learning_rate * gradients
```

实际训练中通常不会直接使用最朴素的梯度下降，而是使用 AdamW 等优化器。

## 常见误解

- 误解：梯度下降每一步都会让整体训练更好。
  - 正确理解：单步更新可能受 batch 噪声影响，loss 会波动。
- 误解：梯度为 0 就一定达到最优。
  - 正确理解：也可能是鞍点、平坦区域或局部极值。
- 误解：梯度下降只适用于小模型。
  - 正确理解：大模型训练仍然基于梯度优化，只是配合自适应优化器和分布式系统。

## 相关概念

- [[backpropagation|反向传播]] — 计算梯度的方法。
- [[learning-rate|学习率]] — 控制更新步长。
- [[sgd-momentum|SGD 与 Momentum]] — 梯度下降的随机和动量变体。
- [[adam|Adam]] — 大模型训练常用的自适应优化器。
