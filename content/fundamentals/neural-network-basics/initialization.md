---
title: Initialization
created: 2026-01-10
published: 2026-01-10
modified: 2026-01-10
type: topic
status: seed
area: fundamentals
tags:
  - neural-network
  - initialization
  - training-stability
aliases:
  - Initialization
  - Weight Initialization
---

## 概念界定

参数初始化是训练开始前为模型权重设置初始值的方法。初始化会影响激活尺度、梯度传播和训练稳定性，尤其在深层网络和大模型中非常重要。

## 背景与问题

如果初始化尺度过大，激活和梯度可能爆炸；如果尺度过小，信号可能迅速衰减。深层 Transformer 需要合理初始化，才能在训练初期保持数值稳定。

## 结构与机制

常见初始化会根据输入输出维度控制权重方差，例如：

```text
Var(W) 与 fan_in / fan_out 相关
```

目标是让前向激活和反向梯度在层间保持合适尺度。

## 直观解释

初始化像是给训练选一个起点。好的起点不会直接给出好模型，但能让优化过程一开始不至于数值失控。

## 基本性质

- 初始化影响早期训练稳定性。
- 初始化需要与激活函数、归一化和残差结构配合。
- 大模型常使用特定初始化缩放策略来适应深层结构。
- 初始化不是越大或越随机越好。

## 常见误解

- 误解：有了 Adam，初始化就不重要。
  - 正确理解：优化器不能完全修复不合理初始化导致的尺度问题。
- 误解：所有层都应该用同样尺度初始化。
  - 正确理解：不同层、残差路径和输出层可能需要不同初始化策略。

## 相关概念

- [[fundamentals/neural-network-basics/normalization|归一化]] — 初始化和归一化共同影响尺度。
- [[fundamentals/neural-network-basics/residual-connection|残差连接]] — 深层残差网络需要稳定初始化。
- [[training/pretraining/pretraining|预训练]] — 大规模训练对初始化敏感。
