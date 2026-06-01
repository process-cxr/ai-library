---
title: Normalization in Transformer
created: 2026-01-24
published: 2026-01-24
modified: 2026-05-31
type: topic
status: mature
area: architecture
tags:
  - transformer
  - normalization
---

Normalization in Transformer 指 Transformer block 中用于稳定 hidden state 尺度和训练动态的归一化机制。它和 [[architecture/transformer/residual|Residual Connection]] 共同决定深层 Transformer 是否容易训练。现代 LLM 中最常见的是 LayerNorm、RMSNorm，以及 Pre-Norm block 结构。

## 为什么需要 Normalization

Transformer 是多层堆叠结构。每一层都会经过 attention、FFN、残差相加和非线性变换。如果 hidden state 的尺度在层间不断漂移，训练会出现：

- 梯度过大或过小；
- attention score 尺度不稳定；
- mixed precision 下更容易 NaN/Inf；
- 深层模型难以收敛；
- learning rate 和 initialization 更敏感。

Normalization 的作用不是提高模型容量，而是控制表示尺度，使优化更稳定。

## LayerNorm

LayerNorm 对单个 token 的 hidden dimension 做归一化。给定：

$$
x \in \mathbb{R}^{d}
$$

LayerNorm 计算：

$$
\mu = \frac{1}{d}\sum_{i=1}^{d}x_i
$$

$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

$$
\mathrm{LayerNorm}(x)_i
= \gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i
$$

其中 $\gamma,\beta$ 是可学习参数。LayerNorm 不跨 batch 统计，因此适合变长序列和自回归模型。

## RMSNorm

RMSNorm 是现代 LLM 中很常见的替代方案。它不减去均值，只用 root mean square 控制尺度：

$$
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}
$$

$$
\mathrm{RMSNorm}(x)_i=\gamma_i\frac{x_i}{\mathrm{RMS}(x)}
$$

相比 LayerNorm，RMSNorm 更简单，少了均值中心化和 bias 项，计算上更轻一些。许多 decoder-only LLM 采用 RMSNorm，因为它在大规模训练中通常足够稳定且效率更好。

## Pre-Norm 与 Post-Norm

Normalization 放在 residual branch 的不同位置，会显著影响训练稳定性。

### Post-Norm

原始 Transformer 常见写法是：

$$
H' = \mathrm{Norm}(H+\mathrm{SubLayer}(H))
$$

这种结构在较浅模型中可用，但深层训练时梯度更容易不稳定。

### Pre-Norm

现代 LLM 更常用 Pre-Norm：

$$
H' = H+\mathrm{SubLayer}(\mathrm{Norm}(H))
$$

Pre-Norm 的直觉是：残差主路径保持相对直接，子层在归一化后的输入上工作。这样梯度可以更容易沿 residual path 传播，使深层模型训练更稳定。

典型 decoder-only block：

```text
H
  -> Norm -> Causal Self-Attention -> Residual Add
  -> Norm -> FFN / MLP -> Residual Add
```

## 与 Attention 的关系

Normalization 会影响进入 Q/K/V 投影的 hidden state 尺度。如果尺度过大，attention score：

$$
\frac{QK^T}{\sqrt{d_k}}
$$

可能变得过大，使 softmax 过于尖锐，梯度不稳定。Pre-Norm 可以让 attention 子层看到更稳定的输入分布。

Normalization 也影响 FFN 输入。如果 FFN 输入尺度漂移，激活函数可能进入饱和或极端区域，导致训练变慢或数值问题。

## 与 Mixed Precision 的关系

在 [[training/optimization/mixed-precision|Mixed Precision Training]] 中，normalization 是数值敏感模块。很多实现会对 normalization 的统计、reduction 或部分计算使用更高精度，以避免 underflow、overflow 或累积误差。

实践中，LayerNorm/RMSNorm 的 fused kernel、dtype 配置和 epsilon 都会影响训练稳定性。大模型出现 loss spike 或 NaN 时，normalization 是需要检查的模块之一。

## 设计取舍

| 设计 | 优势 | 代价 |
|---|---|---|
| LayerNorm | 稳定、经典、表达灵活 | 计算稍重 |
| RMSNorm | 简洁、高效，现代 LLM 常用 | 不做均值中心化 |
| Post-Norm | 原始 Transformer 结构清晰 | 深层训练更难 |
| Pre-Norm | 深层模型更稳定 | 输出尺度可能需要额外控制 |

## 常见误解

- **误解：Normalization 只是实现细节。** 它直接影响深层训练稳定性和可扩展性。
- **误解：LayerNorm 和 BatchNorm 类似。** LayerNorm 沿 hidden dimension 归一化，不依赖 batch 统计。
- **误解：RMSNorm 一定比 LayerNorm 更好。** RMSNorm 是效率和稳定性的常见选择，但具体效果依赖模型规模和 recipe。
- **误解：用了 normalization 就不会 loss spike。** 它降低风险，但不能替代合适的 learning rate、初始化、precision 和数据处理。

## 相关概念

- [[architecture/transformer/transformer|Transformer]]
- [[architecture/transformer/residual|Residual in Transformer]]
- [[architecture/attention/attention|Attention]]
- [[training/optimization/training-stability|Training Stability]]
- [[training/optimization/mixed-precision|Mixed Precision Training]]
- [[fundamentals/neural-network-basics/normalization|Normalization]]
- [[fundamentals/linear-algebra/numerical-stability|Numerical Stability]]

## 经典论文与资料

- [[sources/papers/2016-layer-normalization|Layer Normalization]]
- [[sources/papers/2019-rmsnorm|Root Mean Square Layer Normalization]]
