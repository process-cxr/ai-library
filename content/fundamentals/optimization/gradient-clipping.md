---
title: 梯度裁剪
type: topic
status: growing
area: fundamentals
tags:
  - optimization
  - gradient-clipping
  - training-stability
aliases:
  - Gradient Clipping
---

## 概念界定

梯度裁剪是在梯度过大时对其进行缩放或截断的稳定训练技术。它常用于防止梯度爆炸和训练初期 loss spike。

## 背景与问题

深层网络或长序列训练中，某些 batch 可能产生异常大的梯度。如果直接用这些梯度更新参数，模型可能出现数值不稳定甚至发散。梯度裁剪用于限制单步更新的最大强度。

## 定义与记号

常见方式是按全局范数裁剪：

```text
if ||g|| > c:
    g = c · g / ||g||
```

其中：

- `g` 是所有参数梯度组成的整体梯度。
- `c` 是裁剪阈值。
- `||g||` 通常是 L2 norm。

## 直观解释

梯度裁剪像给参数更新加了安全阀。正常梯度不受影响，异常大的梯度会被缩小到可接受范围。

## 基本性质

- 梯度裁剪不改变 loss 定义，只改变更新过程。
- 裁剪阈值过小会抑制正常学习。
- 裁剪阈值过大则无法有效防止异常更新。
- 大模型训练中常配合 mixed precision、AdamW 和 warmup 使用。

## 示例

训练日志中如果出现突然的 loss spike 或 grad norm 爆炸，可以检查是否需要梯度裁剪、调整学习率或排查数据异常。

## 常见误解

- 误解：梯度裁剪能修复所有训练不稳定。
  - 正确理解：它只能限制梯度幅度，无法解决数据错误、学习率过大或数值溢出等根因。
- 误解：梯度裁剪越强越安全。
  - 正确理解：过强裁剪会让有效学习信号变弱。

## 相关概念

- [[fundamentals/linear-algebra/norm|范数]] — 梯度裁剪通常基于梯度范数。
- [[backpropagation|反向传播]] — 裁剪对象是反向传播得到的梯度。
- [[numerical-stability-optimization|优化中的数值稳定性]] — 梯度溢出和 loss scaling 相关。
