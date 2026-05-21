---
title: Batch Size and Gradient Noise
created: 2026-01-17
published: 2026-01-17
modified: 2026-01-17
type: topic
status: growing
area: fundamentals
tags:
  - optimization
  - batch-size
  - gradient-noise
aliases:
  - Batch Size
  - Gradient Noise
---

## 概念界定

Batch size 决定每一步用多少样本估计梯度，梯度噪声描述 mini-batch 梯度相对真实梯度的随机波动。二者共同影响训练稳定性、吞吐、泛化和学习率选择。

## 背景与问题

大模型训练需要高吞吐，因此常使用很大的 global batch size。但 batch 变大后，梯度估计更稳定、硬件利用率更高，同时也会改变优化动态和学习率需求。

## 定义与记号

真实梯度：

```text
g = E_{x~p_data}[∇L(x)]
```

mini-batch 梯度：

```text
g_B = (1/B) Σ_i ∇L(x_i)
```

梯度噪声可以理解为：

```text
g_B - g
```

## 直观解释

小 batch 像根据少量样本判断方向，方向更抖；大 batch 像综合更多样本再判断方向，方向更稳定。但过度稳定并不总是最好，因为梯度噪声也可能影响探索和泛化。

## 基本性质

- batch size 越大，梯度估计方差通常越小。
- 大 batch 可以提高并行吞吐，但需要更多显存或梯度累积。
- batch size 增大后，学习率常需要重新调整。
- gradient accumulation 可以在显存有限时模拟更大的 global batch。
- 大 batch 训练可能需要 warmup 来稳定初期更新。

## 示例

如果每张 GPU 的 micro batch 是 4，使用 8 张 GPU，梯度累积 16 步：

```text
global batch size = 4 × 8 × 16 = 512
```

优化动态更接近 batch size 512 的训练。

## 常见误解

- 误解：batch 越大越好。
  - 正确理解：大 batch 提升吞吐和稳定性，但可能影响泛化和学习率选择。
- 误解：gradient accumulation 和真实大 batch 完全没有差别。
  - 正确理解：在梯度同步和随机层行为等细节上可能存在差异。
- 误解：batch size 只影响显存。
  - 正确理解：它也影响梯度噪声、优化路径和训练配方。

## 相关概念

- [[gradient-descent|梯度下降]] — mini-batch 梯度是随机估计。
- [[learning-rate|学习率]] — batch size 改变常需要调整学习率。
- [[lr-scheduler|学习率调度]] — 大 batch 常配合 warmup。
- [[training/distributed-training/fsdp|FSDP 分布式训练]] — 分布式训练中的 global batch 计算。
