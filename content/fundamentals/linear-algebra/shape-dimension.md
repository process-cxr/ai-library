---
title: Shape and Dimension
created: 2025-12-06
published: 2025-12-06
modified: 2025-12-06
type: topic
status: growing
area: fundamentals
tags:
  - math
  - linear-algebra
  - tensor
aliases:
  - Shape
  - Dimension
---

## 概念界定

形状描述张量在每个轴上的大小，维度既可以指张量轴的数量，也可以指某个表示空间的宽度。在大模型中，shape 是连接数学公式、模型结构和代码实现的核心线索。

## 背景与问题

许多大模型概念表面上是算法问题，实际理解时首先要回答 shape 问题。例如 Attention 为什么有二次复杂度，KV Cache 为什么随序列长度增长，Tensor Parallel 为什么能切分权重矩阵，都需要先看清张量的轴和大小。

## 定义与记号

常见记号：

| 记号 | 含义 | 典型位置 |
|---|---|---|
| `B` | batch size | 一次并行处理的样本数 |
| `T` | sequence length | token 序列长度 |
| `D` | hidden size / d_model | 模型内部表示宽度 |
| `H` | number of heads | attention head 数量 |
| `Dh` | head dimension | 每个 head 的表示维度 |
| `V` | vocabulary size | 词表大小 |

常见张量：

```text
X:      [B, T, D]
Q/K/V:  [B, H, T, Dh]
scores: [B, H, T, T]
logits: [B, T, V]
```

## 直观解释

shape 可以看作张量的“地址系统”。例如 `[B, T, D]` 表示：先选择 batch 中的第几个样本，再选择序列中的第几个 token，最后选择这个 token 向量中的第几个坐标。

读懂 shape 后，模型中的很多操作会更清楚：矩阵乘法改变最后一维，softmax 常在某个语义轴上归一化，mask 通过广播扩展到 attention score 的形状。

## 基本性质

- shape 合法是计算能运行的必要条件，但不是语义正确的充分条件。
- 同一个维度位置不同，含义可能完全不同，例如 `[B, T, D]` 和 `[B, D, T]`。
- `D = H * Dh` 是多头注意力中常见的维度关系。
- Attention score 的 `[T, T]` 维度解释了长上下文计算和显存压力。

## 示例

单层 self-attention 的 shape 变化：

```text
X:             [B, T, D]
linear QKV:    [B, T, 3D]
split heads:   [B, H, T, Dh]
QK^T scores:   [B, H, T, T]
weighted sum:  [B, H, T, Dh]
merge heads:   [B, T, D]
```

这个过程说明 Attention 不是抽象黑盒，而是一组可追踪的张量变换。

## 常见误解

- 误解：维度只表示张量有几个轴。
  - 正确理解：在深度学习语境中，dimension 也常指 hidden size 这类表示宽度。
- 误解：shape 对了就说明模型逻辑对了。
  - 正确理解：广播、mask 或 transpose 方向错误时，shape 可能仍然合法。
- 误解：sequence length 只影响输入。
  - 正确理解：生成阶段中，KV Cache、attention score 和显存都会随上下文长度变化。

## 相关概念

- [[fundamentals/linear-algebra/vector-matrix|向量、矩阵与张量]] — shape 所描述的对象。
- [[fundamentals/linear-algebra/elementwise-broadcasting|逐元素运算与广播]] — shape 兼容规则。
- [[architecture/attention/attention|Attention 机制]] — shape 分析最典型的应用场景。
