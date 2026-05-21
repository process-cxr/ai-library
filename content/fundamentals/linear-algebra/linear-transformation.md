---
title: Linear Transformation
created: 2025-12-07
published: 2025-12-07
modified: 2025-12-07
type: topic
status: growing
area: fundamentals
tags:
  - math
  - linear-algebra
  - representation
aliases:
  - Linear Transformation
  - Linear Projection
---

## 概念界定

线性变换是保持向量加法和数乘结构的映射，通常可以由矩阵表示。在神经网络中，线性层、Q/K/V 投影、MLP 投影和输出分类头都可以理解为可学习的表示空间变换。

## 背景与问题

大模型的每一层都在不断改写 token 的表示。理解线性变换，可以把权重矩阵从“参数表”理解为“表示空间的变换规则”：它决定哪些方向被放大、压缩、组合或投影到新的空间中。

## 定义与记号

线性变换 `T` 满足：

```text
T(x + y) = T(x) + T(y)
T(αx) = αT(x)
```

矩阵形式：

```text
y = W x
```

深度学习中常见线性层写作：

```text
Y = X W + b
```

严格说，加入 bias 后是仿射变换；但工程语境中通常仍称为 linear layer。

## 直观解释

线性变换可以旋转、拉伸、压缩、剪切或投影向量空间。训练过程就是调整这些变换，使 token 表示逐层变得更适合预测、分类、检索或生成。

## 基本性质

- 线性变换本身不引入非线性，但多层线性变换夹杂激活函数后可以表达复杂函数。
- 权重矩阵的 shape 决定输入空间和输出空间的维度关系。
- 同一个输入 hidden state 可以通过不同矩阵投影到不同语义角色空间。
- 输出空间维度可以大于、小于或等于输入空间维度。

## 示例

Attention 中的 Q/K/V 投影：

```text
Q = X W_q
K = X W_k
V = X W_v
```

它们通常来自同一个输入 `X`，但被投影到不同空间：

- Query 空间：表示当前位置想要查询什么。
- Key 空间：表示当前位置提供什么匹配线索。
- Value 空间：表示当前位置被关注后贡献什么信息。

MLP 中的升维和降维：

```text
X: [B, T, D]
XW_up: [B, T, D_ff]
XW_down: [B, T, D]
```

## 常见误解

- 误解：线性层越多仍然只能表达线性函数。
  - 正确理解：如果中间没有非线性，多个线性层可合并；但加入激活、Attention、归一化后整体不再是简单线性函数。
- 误解：Q/K/V 是三份不同输入。
  - 正确理解：self-attention 中它们通常是同一输入的三个不同线性投影。
- 误解：权重矩阵只是存储知识。
  - 正确理解：它更像对表示空间进行变换的参数化规则。

## 相关概念

- [[fundamentals/linear-algebra/matrix-multiplication|矩阵乘法]] — 线性变换的计算形式。
- [[fundamentals/linear-algebra/basis-coordinate|基与坐标]] — 理解表示空间的坐标视角。
- [[architecture/attention/attention|Attention 机制]] — Q/K/V 投影的应用场景。
- [[architecture/transformer/transformer|Transformer]] — 线性变换反复组合的模型结构。
