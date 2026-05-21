---
title: Loss Landscape
created: 2026-01-18
published: 2026-01-18
modified: 2026-01-18
type: topic
status: seed
area: fundamentals
tags:
  - optimization
  - loss-landscape
  - non-convex
aliases:
  - Loss Landscape
  - Optimization Landscape
---

## 概念界定

Loss landscape 指损失函数在参数空间中的几何形状，包括斜率、曲率、鞍点、局部极小值和平坦区域。深度学习优化通常是在高维非凸 loss landscape 中寻找较低损失区域。

## 背景与问题

大模型参数空间极高维，损失函数通常非凸。优化过程不只是简单“下山”，还会遇到曲率变化、平坦区域、尖锐区域和噪声梯度。理解 loss landscape 有助于理解学习率、batch size、泛化和训练稳定性。

## 定义与记号

损失函数：

```text
L(θ)
```

梯度描述局部斜率：

```text
∇L(θ)
```

Hessian 描述局部曲率：

```text
∇²L(θ)
```

## 直观解释

可以把 loss landscape 想象成高维地形。优化器在地形上移动，梯度告诉当前位置哪里上坡最快，学习率决定每次走多远，动量和自适应缩放会改变移动方式。

## 基本性质

- 深度网络 loss landscape 通常非凸。
- 梯度为 0 的点可能是局部极小值、局部极大值或鞍点。
- 平坦极小值常被认为可能与泛化有关，但解释需要谨慎。
- batch 噪声、学习率和优化器都会影响最终落到哪个区域。

## 常见误解

- 误解：深度学习训练目标有一个容易找到的全局最优。
  - 正确理解：实际训练是在高维非凸空间中寻找足够好的低损失区域。
- 误解：loss landscape 可以像二维图一样完整画出来。
  - 正确理解：二维可视化只是投影或切片，不能完整代表高维结构。

## 相关概念

- [[gradient-descent|梯度下降]] — 在 loss landscape 上移动的基本方法。
- [[learning-rate|学习率]] — 决定每一步移动距离。
- [[batch-size-gradient-noise|Batch Size 与梯度噪声]] — 噪声影响优化轨迹。
