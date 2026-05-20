---
title: 激活函数
type: topic
status: growing
area: fundamentals
tags:
  - neural-network
  - activation
  - transformer
aliases:
  - Activation Function
  - ReLU
  - GELU
  - SiLU
  - SwiGLU
---

## 概念界定

激活函数是在神经网络中引入非线性的函数。没有非线性激活，多层线性层最终仍可以合并为一个线性变换，模型表达能力会严重受限。

## 背景与问题

Transformer 中的 Attention 负责混合 token 间信息，MLP/FFN 则负责对每个 token 的表示进行非线性变换。激活函数是 MLP 能够表达复杂模式的关键。

## 结构与机制

常见激活函数包括 ReLU、Sigmoid、GELU、SiLU / Swish 和 SwiGLU。为了避免符号悬空，这里先给出相关基础函数。

### Sigmoid

Sigmoid 把实数压缩到 `(0, 1)` 区间：

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

它的特点是：

- `x` 很大时，`sigmoid(x)` 接近 1。
- `x` 很小时，`sigmoid(x)` 接近 0。
- 在门控结构中，它常用来控制信息通过比例。

### ReLU

ReLU 是最简单常见的分段线性激活：

```text
ReLU(x) = max(0, x)
```

也就是：

```text
ReLU(x) = x, if x > 0
ReLU(x) = 0, if x <= 0
```

它计算简单，但负半轴梯度为 0，可能出现部分神经元长期不更新的问题。

### GELU

GELU 的全称是 Gaussian Error Linear Unit。它可以写作：

```text
GELU(x) = x · Φ(x)
```

其中 `Φ(x)` 是标准正态分布 `N(0, 1)` 的累积分布函数：

```text
Φ(x) = P(Z <= x),  Z ~ N(0, 1)
```

展开为积分形式：

```text
Φ(x) = ∫_{-∞}^{x} (1 / sqrt(2π)) exp(-t^2 / 2) dt
```

实际实现中常用近似公式：

```text
GELU(x) ≈ 0.5x · (1 + tanh(sqrt(2/π) · (x + 0.044715x^3)))
```

也有实现使用 sigmoid 近似：

```text
GELU(x) ≈ x · sigmoid(1.702x)
```

直观上，GELU 不是像 ReLU 那样硬截断负数，而是根据输入大小给它一个平滑的通过概率。

### SiLU / Swish

SiLU，也常称为 Swish，定义为：

```text
SiLU(x) = x · sigmoid(x)
```

代入 sigmoid 可写成：

```text
SiLU(x) = x / (1 + exp(-x))
```

它是平滑激活函数。和 ReLU 相比，SiLU 在负半轴不会直接变成 0，而是保留较小的负值输出。

### GLU 与 SwiGLU

GLU 的全称是 Gated Linear Unit，核心思想是用一个 gate 控制另一条 value 分支的信息通过：

```text
GLU(a, b) = a ⊙ sigmoid(b)
```

其中 `⊙` 表示逐元素乘法。

SwiGLU 是把 gate 分支中的 sigmoid 换成 SiLU / Swish 的变体，现代 LLM 中常见写法是：

```text
SwiGLU(x) = SiLU(xW_gate) ⊙ (xW_up)
```

然后再经过降维投影：

```text
FFN(x) = W_down( SiLU(xW_gate) ⊙ (xW_up) )
```

这里 `xW_gate` 产生门控分支，`xW_up` 产生内容分支，逐元素相乘后表示“哪些信息允许通过”。

## 直观解释

激活函数像一个非线性开关或门控机制。它让模型不仅能做线性组合，还能根据输入大小选择性放大、抑制或过滤信息。

## 基本性质

- ReLU 简单高效，但负半轴梯度为 0。
- Sigmoid 输出在 `(0, 1)`，适合作为门控，但在极大或极小输入处容易梯度饱和。
- GELU 在 BERT、GPT 等模型中常见，过渡更平滑，可以看作对输入进行概率式软门控。
- SiLU / Swish 是平滑激活，负半轴也保留连续输出。
- SwiGLU 将激活和门控结合，在 LLaMA 等现代大模型中常见。
- 激活函数会影响训练稳定性、表达能力、梯度传播和计算成本。

## 示例

Transformer FFN 的经典结构：

```text
FFN(x) = W_down activation(W_up x)
```

SwiGLU 风格结构：

```text
FFN(x) = W_down( SiLU(W_gate x) ⊙ W_up x )
```

SwiGLU 通过 gate 控制哪些信息通过 MLP。

## 常见误解

- 误解：激活函数只是让数值变得非负。
  - 正确理解：只有 ReLU 这类函数有明显截断效果；GELU、SiLU、SwiGLU 更强调平滑变换和门控。
- 误解：Transformer 主要靠 Attention，MLP 和激活不重要。
  - 正确理解：MLP 通常贡献大量参数和计算，对模型能力非常关键。
- 误解：所有激活函数可以随意替换。
  - 正确理解：激活函数与初始化、归一化、学习率和模型结构都有耦合。

## 相关概念

- [[fundamentals/neural-network-basics/feed-forward-network|前馈网络]] — 激活函数的主要应用位置。
- [[fundamentals/linear-algebra/elementwise-broadcasting|逐元素运算与广播]] — 激活函数通常逐元素作用。
- [[architecture/model-families/llama|LLaMA 系列]] — SwiGLU 是 LLaMA MLP 的关键设计之一。
