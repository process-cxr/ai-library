---
title: Numerical Stability in Optimization
created: 2026-02-13
published: 2026-02-13
modified: 2026-02-20
type: topic
status: seed
area: fundamentals
tags:
  - optimization
  - numerical-stability
  - mixed-precision
aliases:
  - Optimization Numerical Stability
  - Loss Scaling
---

## 概念界定

优化中的数值稳定性关注低精度训练、梯度缩放、溢出检测和参数更新中的数值问题。它是混合精度大模型训练能否稳定运行的重要基础。

## 背景与问题

大模型训练为了节省显存和提升吞吐，常使用 fp16 或 bf16。低精度会带来溢出、下溢和舍入误差。优化过程中的梯度尤其容易受到这些问题影响。

## 定义与记号

Loss scaling 常用于 fp16 训练：

```text
scaled_loss = loss · scale
```

先对放大的 loss 反向传播，再在更新前把梯度缩放回来：

```text
grad = scaled_grad / scale
```

这样可以减少梯度下溢风险。

## 直观解释

如果梯度太小，fp16 可能表示不出来，直接变成 0。loss scaling 就像先把信号放大，计算完梯度后再缩回去。

## 基本性质

- bf16 范围比 fp16 更接近 fp32，通常更稳定。
- fp16 常需要 loss scaling。
- 梯度溢出时通常跳过当前 step 或降低 scale。
- 某些累加和优化器状态仍常使用 fp32。

## 常见误解

- 误解：低精度训练只影响速度和显存。
  - 正确理解：低精度也会影响稳定性和误差累积。
- 误解：loss scaling 改变了训练目标。
  - 正确理解：正确缩放回梯度后，目标不变，只是改善数值表示。

## 相关概念

- [[fundamentals/linear-algebra/numerical-stability|数值稳定性]] — 更一般的数值稳定背景。
- [[training/optimization/mixed-precision|混合精度训练]] — 低精度训练实践。
- [[gradient-clipping|梯度裁剪]] — 控制梯度异常幅度。
