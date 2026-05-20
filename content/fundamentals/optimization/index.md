---
title: Optimization
created: 2026-01-03
published: 2026-01-03
modified: 2026-01-10
type: topic
status: growing
area: fundamentals
tags:
  - optimization
  - training
  - deep-learning
aliases:
  - Optimization
---

## 概念界定

优化研究如何调整模型参数，使损失函数逐步降低。在深度学习和大模型中，优化不仅包括梯度下降和反向传播，还包括优化器、学习率调度、正则化、梯度裁剪、混合精度稳定性和大 batch 训练等问题。

这一节只关注理解大模型训练主线所必需的优化基础，不展开完整凸优化理论。

## 背景与问题

大模型包含数十亿到万亿级参数，训练目标通常是最小化 next-token prediction 的交叉熵损失。优化过程要解决的问题是：如何用有限 batch 的梯度估计，在高维非凸参数空间中稳定地更新参数，同时避免梯度爆炸、数值不稳定、学习率失控和过拟合。

优化基础主要回答：

- 梯度为什么指示损失上升最快方向？
- 反向传播如何高效计算所有参数的梯度？
- SGD、Momentum、Adam、AdamW 分别解决什么问题？
- 学习率、warmup、decay 为什么会影响训练稳定性？
- 梯度裁剪、weight decay、mixed precision 为什么在大模型训练中重要？
- batch size 和梯度噪声如何影响优化行为？

## 知识结构

### 梯度与计算

- [[fundamentals/optimization/gradient-descent|梯度下降]] — 基础参数更新思想。
- [[fundamentals/optimization/backpropagation|反向传播]] — 通过链式法则高效计算梯度。
- [[fundamentals/optimization/computational-graph|计算图与自动微分]] — 深度学习框架如何组织前向和反向计算。

### 优化器

- [[fundamentals/optimization/sgd-momentum|SGD 与 Momentum]] — 随机梯度和动量加速。
- [[fundamentals/optimization/adam|Adam]] — 自适应一阶/二阶矩估计。
- [[fundamentals/optimization/adamw-weight-decay|AdamW 与 Weight Decay]] — 大模型训练中的常见优化器配置。

### 学习率与稳定性

- [[fundamentals/optimization/learning-rate|学习率]] — 参数更新步长的核心超参数。
- [[fundamentals/optimization/lr-scheduler|学习率调度]] — warmup、cosine decay、linear decay 等调度策略。
- [[fundamentals/optimization/gradient-clipping|梯度裁剪]] — 控制梯度范数，避免训练不稳定。
- [[fundamentals/optimization/numerical-stability-optimization|优化中的数值稳定性]] — loss scaling、低精度训练和梯度溢出。

### 泛化与规模

- [[fundamentals/optimization/regularization|正则化]] — 降低过拟合和控制模型复杂度。
- [[fundamentals/optimization/batch-size-gradient-noise|Batch Size 与梯度噪声]] — batch size、噪声尺度和大规模训练。
- [[fundamentals/optimization/loss-landscape|Loss Landscape]] — 非凸优化、局部极小值、鞍点和平坦极小值。

## 推荐阅读顺序

1. [[fundamentals/optimization/gradient-descent|梯度下降]]
2. [[fundamentals/optimization/backpropagation|反向传播]]
3. [[fundamentals/optimization/computational-graph|计算图与自动微分]]
4. [[fundamentals/optimization/sgd-momentum|SGD 与 Momentum]]
5. [[fundamentals/optimization/adam|Adam]]
6. [[fundamentals/optimization/adamw-weight-decay|AdamW 与 Weight Decay]]
7. [[fundamentals/optimization/learning-rate|学习率]]
8. [[fundamentals/optimization/lr-scheduler|学习率调度]]
9. [[fundamentals/optimization/gradient-clipping|梯度裁剪]]
10. [[fundamentals/optimization/batch-size-gradient-noise|Batch Size 与梯度噪声]]

## 与大模型主线的关系

- [[training/pretraining/pretraining|预训练]] 依赖大规模优化过程最小化 next-token 交叉熵。
- [[training/optimization/mixed-precision|混合精度训练]] 需要处理低精度下的梯度溢出和 loss scaling。
- [[training/distributed-training/fsdp|FSDP 分布式训练]] 会改变参数、梯度和优化器状态的存储与同步方式。
- [[training/scaling/scaling-law|Scaling Law]] 讨论模型规模、数据规模、计算量和 loss 的关系。
- [[fundamentals/information-theory/cross-entropy|交叉熵]] 是语言模型训练中最常见的损失目标。

## 常见误解

- 误解：优化就是选一个 Adam 然后训练。
  - 正确理解：优化还包括学习率、调度、初始化、归一化、梯度裁剪、数值精度和 batch size 等系统性设计。
- 误解：loss 下降就说明模型所有能力都提升。
  - 正确理解：loss 主要反映训练目标，和指令跟随、事实性、安全性并不完全等价。
- 误解：大 batch 一定更好。
  - 正确理解：大 batch 提高吞吐和稳定性，但可能改变梯度噪声、泛化和学习率需求。
