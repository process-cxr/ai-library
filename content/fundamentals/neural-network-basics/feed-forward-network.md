---
title: 前馈网络
type: topic
status: growing
area: fundamentals
tags:
  - neural-network
  - ffn
  - transformer
aliases:
  - Feed Forward Network
  - FFN
  - MLP
---

## 概念界定

前馈网络是由线性层、激活函数和线性层组成的神经网络模块。在 Transformer 中，FFN/MLP 通常对每个 token 的 hidden state 独立进行非线性变换。

## 背景与问题

Attention 负责在 token 之间交换信息，但每个 token 的表示还需要更复杂的非线性加工。FFN 提供这种逐 token 的表示变换能力，是 Transformer block 中与 Attention 并列的重要模块。

## 结构与机制

经典 Transformer FFN：

```text
FFN(x) = W_2 activation(W_1 x + b_1) + b_2
```

常见 shape：

```text
x: [B, T, D]
W_1: [D, D_ff]
W_2: [D_ff, D]
```

现代 LLM 常用门控 MLP，例如 SwiGLU：

```text
FFN(x) = W_down( SiLU(W_gate x) ⊙ W_up x )
```

## 直观解释

如果 Attention 是“从上下文中取信息”，FFN 就是“对每个位置取到的信息进行加工”。它不直接混合不同 token，而是更新每个 token 自己的表示。

## 基本性质

- FFN 通常逐 token 独立作用，不改变序列长度。
- `D_ff` 往往大于 `D`，因此 FFN 会先升维再降维。
- FFN 通常贡献 Transformer 中大量参数和计算。
- 门控 FFN 可以选择性控制信息通过。

## 示例

一个 Transformer block 可以粗略写成：

```text
x = x + Attention(Norm(x))
x = x + FFN(Norm(x))
```

其中 FFN 是每层中负责非线性变换的重要部分。

## 常见误解

- 误解：FFN 只是 Attention 后面的附属层。
  - 正确理解：FFN 对模型容量和表达能力非常重要，通常参数量很大。
- 误解：FFN 会在 token 之间传递信息。
  - 正确理解：标准 FFN 对每个 token 独立作用，token 间信息混合主要来自 Attention。
- 误解：`D_ff` 越大一定越好。
  - 正确理解：更大 `D_ff` 增加容量，也增加参数、计算和显存成本。

## 相关概念

- [[fundamentals/neural-network-basics/linear-layer|线性层]] — FFN 的主要组成。
- [[fundamentals/neural-network-basics/activation-functions|激活函数]] — FFN 引入非线性的方式。
- [[architecture/transformer/transformer|Transformer]] — FFN 是 Transformer block 的核心模块。
