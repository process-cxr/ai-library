---
title: Elementwise Operations and Broadcasting
created: 2025-12-13
published: 2025-12-13
modified: 2025-12-13
type: topic
status: growing
area: fundamentals
tags:
  - math
  - linear-algebra
  - tensor
aliases:
  - Elementwise Operation
  - Broadcasting
---

## 概念界定

逐元素运算是在张量对应位置的元素之间执行相同操作；广播是在满足特定 shape 规则时，让较小张量沿某些轴自动扩展参与计算。二者共同构成深度学习中 bias、mask、gate、residual 和 normalization 的基础运算规则。

## 背景与问题

大模型代码中并非所有计算都是矩阵乘法。大量操作是逐元素加法、乘法、比较、指数、归一化和 mask。许多 shape 错误、mask 错误和隐性 bug 都来自对广播语义理解不清。

## 定义与记号

逐元素加法示例：

```text
A: [B, T, D]
B: [B, T, D]
C = A + B: [B, T, D]
```

广播加法示例：

```text
X: [B, T, D]
b: [D]
Y = X + b: [B, T, D]
```

Attention mask 示例：

```text
scores: [B, H, T, T]
mask:   [B, 1, 1, T]
```

`mask` 可以广播到 head 维度和 query position 维度。

## 直观解释

广播可以理解为“在缺失的轴上复用同一组数值”。例如 bias `[D]` 会被加到每个 batch、每个 token 的 hidden vector 上。它通常不是物理复制数据，而是通过张量视图和 stride 规则完成。

## 基本性质

- 广播要求对应维度相等，或其中一个维度为 1。
- 广播合法不等于语义正确，尤其是 mask 和 attention score 场景。
- 逐元素乘法不同于矩阵乘法，不会做行列内积。
- residual connection 通常要求两个张量 shape 完全一致。

## 示例

SwiGLU 中的 gate 结构可以看作逐元素乘法：

```text
gate:  [B, T, D_ff]
value: [B, T, D_ff]
out = gate * value
```

Causal mask 通常通过给非法位置加一个极小值实现：

```text
masked_scores = scores + mask
```

其中 mask 在非法位置上接近 `-inf`，使 softmax 后这些位置权重接近 0。

## 常见误解

- 误解：广播会真的复制出一个大张量。
  - 正确理解：多数框架会用视图语义避免真实复制，但后续操作可能触发物化。
- 误解：shape 能广播就说明代码正确。
  - 正确理解：轴的语义可能错位，尤其是 `[B, T, 1]` 与 `[B, 1, T]`。
- 误解：`*` 和 `@` 只是写法不同。
  - 正确理解：`*` 通常是逐元素乘法，`@` 通常是矩阵乘法。

## 相关概念

- [[fundamentals/linear-algebra/shape-dimension|形状与维度]] — 广播规则依赖 shape 理解。
- [[fundamentals/neural-network-basics/activation-functions|激活函数]] — 常作为逐元素函数作用在张量上。
- [[fundamentals/neural-network-basics/normalization|归一化]] — 依赖逐元素中心化和缩放。
- [[architecture/attention/attention|Attention 机制]] — mask 广播的典型场景。
