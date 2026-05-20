---
title: Adam
created: 2026-02-04
published: 2026-02-04
modified: 2026-02-11
type: topic
status: growing
area: fundamentals
tags:
  - optimization
  - adam
  - training
aliases:
  - Adam Optimizer
---

## 概念界定

Adam 是一种自适应优化器，它同时维护梯度的一阶矩估计和二阶矩估计，用于调整每个参数的更新方向和更新尺度。Adam 及其变体 AdamW 是大模型训练中最常见的优化器之一。

## 背景与问题

普通 SGD 对所有参数使用相同学习率，但不同参数的梯度尺度可能差异很大。Adam 通过估计梯度均值和平方梯度均值，为每个参数自适应调整更新幅度，从而提高训练稳定性和收敛速度。

## 定义与记号

给定梯度 `g_t`：

```text
m_t = β1 m_{t-1} + (1 - β1) g_t
v_t = β2 v_{t-1} + (1 - β2) g_t^2
```

其中：

- `m_t`：一阶矩估计，类似 Momentum。
- `v_t`：二阶矩估计，估计梯度平方的尺度。

偏差修正：

```text
m_hat_t = m_t / (1 - β1^t)
v_hat_t = v_t / (1 - β2^t)
```

参数更新：

```text
θ_{t+1} = θ_t - η · m_hat_t / (sqrt(v_hat_t) + ε)
```

## 直观解释

Adam 会记住“梯度长期往哪走”和“这个参数的梯度通常有多大”。如果某个参数梯度尺度很大，Adam 会相对缩小它的更新；如果梯度尺度小，则相对放大。

## 基本性质

- `β1` 控制一阶动量平滑程度。
- `β2` 控制二阶矩平滑程度。
- `ε` 防止除以 0，并影响数值稳定性。
- Adam 需要保存额外优化器状态，显存开销大。
- 大模型训练中常使用 AdamW 而不是原始 Adam。

## 示例

Adam 优化器状态通常包括：

```text
parameter θ
first moment m
second moment v
```

如果使用 fp32 master weights，还会额外保存 fp32 参数副本。这也是大模型训练中优化器状态显存占用很高的原因之一。

## 常见误解

- 误解：Adam 自动解决所有学习率问题。
  - 正确理解：Adam 仍然需要合适的全局学习率、warmup 和 decay。
- 误解：Adam 不需要 weight decay。
  - 正确理解：大模型训练常用 AdamW，将 weight decay 与自适应梯度更新解耦。
- 误解：Adam 状态很小。
  - 正确理解：Adam 至少为每个参数保存 `m` 和 `v`，显存成本很高。

## 相关概念

- [[sgd-momentum|SGD 与 Momentum]] — Adam 的一阶矩与 Momentum 相关。
- [[adamw-weight-decay|AdamW 与 Weight Decay]] — 大模型常用优化器形式。
- [[learning-rate|学习率]] — Adam 仍依赖全局学习率。
- [[training/distributed-training/fsdp|FSDP 分布式训练]] — 分片优化器状态以降低显存压力。
