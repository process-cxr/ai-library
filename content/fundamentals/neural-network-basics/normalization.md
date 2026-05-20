---
title: 归一化
type: topic
status: growing
area: fundamentals
tags:
  - neural-network
  - normalization
  - transformer
aliases:
  - Normalization
  - LayerNorm
  - RMSNorm
---

## 概念界定

归一化是调整激活值尺度和分布的技术，用于提高训练稳定性和优化效率。在大模型中，LayerNorm、RMSNorm、Pre-Norm 和 Post-Norm 是理解 Transformer 稳定训练的关键概念。

## 背景与问题

深层神经网络中，激活值和梯度的尺度可能随着层数变化而不稳定。Transformer 通常堆叠几十甚至上百层，如果没有合适的归一化和残差结构，训练很容易不稳定。

## 结构与机制

LayerNorm 对单个 token 的 hidden vector 做归一化：

```text
LN(x) = γ · (x - mean(x)) / sqrt(var(x) + ε) + β
```

RMSNorm 不减均值，只使用均方根尺度：

```text
RMSNorm(x) = γ · x / RMS(x)
```

其中：

```text
RMS(x) = sqrt(mean(x^2) + ε)
```

## 直观解释

归一化可以理解为控制 hidden state 的数值尺度，让每层输入处在更稳定的范围内。它不是为了改变语义目标，而是为了让深层模型更容易训练。

## 基本性质

- LayerNorm 常用于 Transformer，因为它不依赖 batch 统计量。
- RMSNorm 比 LayerNorm 更简单，省去均值中心化，在 LLaMA 等模型中常用。
- Pre-Norm 把 Norm 放在子层前，通常更利于深层训练稳定。
- Post-Norm 把 Norm 放在残差之后，早期 Transformer 使用较多。
- 归一化参数 `γ` 和 `β` 通常是可学习的。

## 示例

Pre-Norm Transformer block：

```text
x = x + Attention(Norm(x))
x = x + FFN(Norm(x))
```

Post-Norm Transformer block：

```text
x = Norm(x + Attention(x))
x = Norm(x + FFN(x))
```

现代大模型更常见 Pre-Norm 或其变体。

## 常见误解

- 误解：归一化只是把数值缩放到 0 到 1。
  - 正确理解：LayerNorm/RMSNorm 是按 hidden 维度调整尺度，不是 min-max scaling。
- 误解：归一化一定提升最终能力。
  - 正确理解：它主要改善训练稳定性和优化条件，效果取决于整体架构。
- 误解：LayerNorm 和 BatchNorm 可以随意替换。
  - 正确理解：序列模型和大模型通常更适合 LayerNorm/RMSNorm，因为 batch 统计不稳定且不方便自回归推理。

## 相关概念

- [[fundamentals/linear-algebra/norm|范数]] — RMSNorm 和尺度控制的数学基础。
- [[fundamentals/neural-network-basics/residual-connection|残差连接]] — 与归一化共同支撑深层训练。
- [[architecture/transformer/transformer|Transformer]] — 归一化是 Transformer block 的核心结构。
- [[architecture/model-families/llama|LLaMA 系列]] — RMSNorm 是 LLaMA 的典型设计。
