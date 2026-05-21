---
title: Matrix Multiplication
created: 2025-12-07
published: 2025-12-07
modified: 2025-12-07
type: topic
status: growing
area: fundamentals
tags:
  - math
  - linear-algebra
  - matrix-multiplication
aliases:
  - MatMul
  - GEMM
---

## 概念界定

矩阵乘法是将一个矩阵的行与另一个矩阵的列进行内积，并生成新矩阵的运算。在大模型中，线性层、Q/K/V 投影、Attention score、MLP 和输出 logits 都大量依赖矩阵乘法。

## 背景与问题

大模型的大部分计算量来自矩阵乘法。GPU、TPU、Tensor Core、FlashAttention、量化和张量并行等优化，很多都围绕如何更快、更省地执行大规模矩阵乘法展开。因此，理解矩阵乘法不仅是数学基础，也是理解大模型系统性能的入口。

## 定义与记号

若：

```text
A ∈ R^{m×n}
B ∈ R^{n×p}
```

则：

```text
C = A B ∈ R^{m×p}
```

其中：

```text
C_{ij} = Σ_k A_{ik} B_{kj}
```

shape 规则：

```text
[m, n] @ [n, p] -> [m, p]
```

中间维度 `n` 必须一致。

## 直观解释

矩阵乘法可以理解为批量计算多个向量之间的内积，也可以理解为把输入表示投影到新的坐标空间。神经网络中的权重矩阵不是静态查表，而是一个可学习的表示变换。

## 基本性质

- 矩阵乘法通常不满足交换律：`A B` 一般不等于 `B A`。
- 矩阵乘法会改变 shape 的最后一个语义维度，例如 `[B, T, D] @ [D, D_out] -> [B, T, D_out]`。
- 批量矩阵乘法会在 batch/head 等前缀维度上并行执行。
- 大模型的计算瓶颈常由矩阵大小、内存访问和硬件利用率共同决定。

## 示例

线性层：

```text
X: [B, T, D]
W: [D, D_out]
Y = X W: [B, T, D_out]
```

Attention score：

```text
Q:   [B, H, T, Dh]
K^T: [B, H, Dh, T]
S = QK^T: [B, H, T, T]
```

MLP：

```text
[B, T, D] -> [B, T, D_ff] -> [B, T, D]
```

## 常见误解

- 误解：矩阵乘法就是对应位置相乘。
  - 正确理解：对应位置相乘是逐元素乘法；矩阵乘法是行列内积。
- 误解：公式简单，所以计算成本可以忽略。
  - 正确理解：在大模型中，矩阵乘法通常是主要 FLOPs 来源。
- 误解：参数量相同，计算效率就相同。
  - 正确理解：不同矩阵 shape、batch 大小和硬件布局会导致不同吞吐。

## 相关概念

- [[fundamentals/linear-algebra/dot-product-similarity|内积与相似度]] — 矩阵乘法的局部计算形式。
- [[fundamentals/linear-algebra/linear-transformation|线性变换]] — 矩阵乘法在表示空间中的含义。
- [[architecture/attention/attention|Attention 机制]] — QKᵀ 是矩阵乘法的关键应用。
- [[inference/quantization/quantization|量化]] — 常围绕矩阵乘法权重和激活进行低精度优化。
