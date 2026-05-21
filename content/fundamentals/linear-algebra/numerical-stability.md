---
title: Numerical Stability
created: 2025-12-13
published: 2025-12-13
modified: 2025-12-13
type: topic
status: growing
area: fundamentals
tags:
  - math
  - linear-algebra
  - numerical-stability
aliases:
  - Numerical Stability
---

## 概念界定

数值稳定性研究在有限精度计算中，如何避免误差、溢出、下溢和舍入问题破坏计算结果。在大模型中，它影响 softmax、交叉熵、混合精度训练、量化推理和长序列 attention 的可靠性。

## 背景与问题

数学公式通常假设实数可以无限精确表示，但计算机只能使用有限位宽表示数值。大模型为了提升吞吐、节省显存，常使用 bf16、fp16、int8、int4 等低精度格式。这会让原本等价的数学表达，在实际计算中出现不同稳定性。

## 定义与记号

常见数值问题：

- **Overflow 溢出**：数值过大，超出可表示范围。
- **Underflow 下溢**：数值过小，被近似为 0。
- **Rounding Error 舍入误差**：有限位宽无法精确表示实数。
- **Cancellation 相消误差**：两个接近数相减导致有效精度损失。

常见数值格式：

| 格式 | 特点 |
|---|---|
| fp32 | 精度和范围较好，成本较高 |
| fp16 | 精度和范围较低，速度快但更易不稳定 |
| bf16 | 范围接近 fp32，精度较低，训练中常用 |
| int8/int4 | 推理压缩常用，需要量化尺度 |

## 直观解释

数值稳定性并不总是改变数学目标，很多时候只是换一种等价但更安全的计算方式。例如 softmax 先减最大值，数学结果不变，但能避免指数溢出。

## 基本性质

- 数值稳定性依赖数据范围、计算顺序和数值格式。
- 低精度会改变可表示范围和舍入行为。
- 累加操作通常比单次乘法更容易积累误差。
- 稳定实现常通过重排公式、缩放、裁剪或使用更高精度累加完成。

## 示例

稳定 softmax：

```text
softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

Log-Sum-Exp 技巧：

```text
m = max(x)
log(Σ_i exp(x_i)) = m + log(Σ_i exp(x_i - m))
```

这类技巧在交叉熵、语言模型 loss 和 FlashAttention 中很常见。

## 常见误解

- 误解：公式正确，代码就一定正确。
  - 正确理解：直接实现数学公式可能导致溢出、下溢或精度损失。
- 误解：低精度只是数值更粗糙。
  - 正确理解：它还会改变范围、舍入模式和误差累积方式。
- 误解：数值稳定技巧都是近似。
  - 正确理解：很多稳定技巧在数学上等价，只是计算路径更安全。

## 相关概念

- [[fundamentals/neural-network-basics/softmax|Softmax]] — softmax 需要稳定实现。
- [[training/optimization/mixed-precision|混合精度训练]] — 低精度训练的核心问题。
- [[inference/quantization/quantization|量化]] — 更低精度推理中的尺度管理。
- [[inference/attention-acceleration/flash-attention|FlashAttention]] — 分块 attention 中需要维护稳定 softmax。
