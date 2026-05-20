---
title: 范数
type: topic
status: growing
area: fundamentals
tags:
  - math
  - linear-algebra
  - normalization
aliases:
  - Norm
  - L2 Norm
---

## 概念界定

范数是衡量向量或矩阵大小的函数。在深度学习中，范数用于描述向量长度、距离、权重规模、梯度规模和数值尺度，是理解归一化、正则化、梯度裁剪和相似度计算的重要基础。

## 背景与问题

大模型训练中，表示和梯度的尺度会直接影响稳定性。如果激活或梯度过大，训练可能发散；如果过小，信号可能消失。推理和检索中，向量长度也会影响相似度计算。因此需要用范数刻画“大小”和“尺度”。

## 定义与记号

常见向量范数：

```text
L1: ||x||_1 = Σ_i |x_i|
L2: ||x||_2 = sqrt(Σ_i x_i^2)
L∞: ||x||_∞ = max_i |x_i|
```

矩阵也有范数，例如 Frobenius 范数：

```text
||A||_F = sqrt(Σ_i Σ_j A_{ij}^2)
```

## 直观解释

L2 范数可以理解为向量在欧几里得空间中的长度。L1 范数更关注绝对值总量，常与稀疏性相关。无穷范数关注最大元素，适合观察是否存在异常大值。

## 基本性质

- 范数非负，只有零向量的范数为 0。
- 范数满足齐次性：`||αx|| = |α| ||x||`。
- L2 范数会影响内积大小，因此会影响 dot-product similarity。
- 向量归一化后，内积可以等价于余弦相似度。

## 示例

余弦相似度：

```text
cos(x, y) = (x · y) / (||x|| ||y||)
```

梯度裁剪：

```text
if ||g|| > threshold:
    g = threshold * g / ||g||
```

这可以避免单次更新过大导致训练不稳定。

## 常见误解

- 误解：范数越大说明表示越好。
  - 正确理解：过大的范数可能表示尺度失控，未必有更好语义。
- 误解：余弦相似度完全消除了尺度问题。
  - 正确理解：它消除了向量长度影响，但仍依赖表示空间本身是否有效。
- 误解：正则化和归一化是一回事。
  - 正确理解：正则化约束模型或参数，归一化调整激活或表示尺度。

## 相关概念

- [[fundamentals/linear-algebra/dot-product-similarity|内积与相似度]] — 范数参与余弦相似度定义。
- [[fundamentals/neural-network-basics/normalization|归一化]] — 控制激活尺度。
- [[training/distributed-training/fsdp|FSDP 分布式训练]] — 大规模训练中常关注梯度范数和裁剪。
- [[inference/quantization/quantization|量化]] — 低精度表示依赖数值范围和尺度。
