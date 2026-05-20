---
title: Learning Rate
created: 2026-02-07
published: 2026-02-07
modified: 2026-02-14
type: topic
status: growing
area: fundamentals
tags:
  - optimization
  - learning-rate
  - training
aliases:
  - Learning Rate
---

## 概念界定

学习率是控制参数更新步长的超参数。它决定模型每次根据梯度移动多远，是影响训练速度、稳定性和最终效果的关键因素。

## 背景与问题

大模型训练通常非常昂贵，学习率设置不当可能导致训练发散、收敛过慢或最终效果不佳。即使使用 AdamW，仍然需要仔细设计学习率大小和调度策略。

## 定义与记号

基本更新中：

```text
θ_{t+1} = θ_t - η g_t
```

`η` 就是学习率。

对于 Adam 类优化器：

```text
θ_{t+1} = θ_t - η · normalized_update
```

学习率仍然控制全局更新幅度。

## 直观解释

学习率像走路步长。步子太大容易越过低谷甚至摔出去；步子太小虽然稳，但训练成本很高。

## 基本性质

- 学习率过大可能导致 loss spike 或训练发散。
- 学习率过小会导致训练慢或陷入低效区域。
- 合适学习率与 batch size、模型规模、优化器和 warmup 有关。
- 大模型训练通常使用 warmup 后再 decay 的策略。

## 示例

一个常见训练阶段：

```text
warmup: 学习率从 0 增加到 peak lr
decay: 学习率从 peak lr 逐步降低
```

warmup 可以避免训练初期参数和优化器状态尚不稳定时更新过大。

## 常见误解

- 误解：学习率越大训练越快，所以越好。
  - 正确理解：过大学习率会导致不稳定或发散。
- 误解：Adam 会自动选择学习率。
  - 正确理解：Adam 自适应缩放各参数更新，但仍需要全局学习率。
- 误解：同一个学习率适用于所有规模模型。
  - 正确理解：学习率通常需要随模型规模、batch size 和训练 token 数调整。

## 相关概念

- [[lr-scheduler|学习率调度]] — 学习率如何随训练步数变化。
- [[adam|Adam]] — 学习率控制 Adam 更新的全局尺度。
- [[batch-size-gradient-noise|Batch Size 与梯度噪声]] — batch size 会影响学习率选择。
- [[training/scaling/scaling-law|Scaling Law]] — 规模变化会影响训练配方。
