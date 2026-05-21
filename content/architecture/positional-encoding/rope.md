---
title: RoPE
created: 2026-02-01
published: 2026-02-01
modified: 2026-02-01
type: topic
status: growing
area: architecture
tags:
  - positional-encoding
  - rope
  - attention
aliases:
  - Rotary Position Embedding
  - Rotary Positional Embedding
---

## 问题背景

RoPE，全称 Rotary Position Embedding，是一种把位置信息注入 [[architecture/attention/attention|Attention]] 的方法。它在现代 decoder-only 大语言模型中非常常见，例如 LLaMA、Qwen、部分 DeepSeek 系列等模型都使用了 RoPE 或其扩展变体。

Transformer 的 attention 本身是 permutation-invariant 的：如果只看 token embedding，不额外加入位置机制，模型无法知道 token 的先后顺序。对语言模型来说，顺序非常关键：

```text
狗咬人
人咬狗
```

两个句子包含相同 token，但意义完全不同。模型必须知道每个 token 出现在什么位置，以及 token 之间相隔多远。

传统做法可以给每个位置加一个 position embedding，但 RoPE 的思路不同：它不直接把位置向量加到 token embedding 上，而是在 attention 的 query 和 key 空间中，通过旋转变换编码位置信息。

## 核心直觉

RoPE 的核心思想是：

> 用位置决定旋转角度，把 query 和 key 按各自位置旋转；旋转后的 dot product 会自然包含相对位置信息。

可以把每个 token 的某些表示维度看成二维平面上的向量。位置 $m$ 对应一个旋转角度 $m\theta$。位置越靠后，旋转角度越大。

如果两个 token 分别在位置 $m$ 和 $n$，它们的 query/key 向量会被旋转不同角度。做 dot product 时，结果会依赖二者的角度差：

$$
(m\theta - n\theta) = (m-n)\theta
$$

因此，attention score 不仅取决于 token 内容，也取决于两个 token 的相对距离 $m-n$。

这就是 RoPE 的关键：它通过旋转让 attention score 感知相对位置。

## 二维旋转回顾

在二维平面中，向量 $(x_1, x_2)$ 旋转角度 $\phi$ 后变成：

$$
\begin{bmatrix}
x_1' \\
x_2'
\end{bmatrix}
=
\begin{bmatrix}
\cos \phi & -\sin \phi \\
\sin \phi & \cos \phi
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

也就是：

$$
x_1' = x_1\cos\phi - x_2\sin\phi
$$

$$
x_2' = x_1\sin\phi + x_2\cos\phi
$$

旋转不会改变向量长度，只改变方向。RoPE 正是利用这种性质，把“位置”变成“方向变化”。

## RoPE 的基本形式

设某个位置为 $m$，query 或 key 向量为 $x \in \mathbb{R}^{d}$。RoPE 会把向量按维度两两分组：

$$
(x_1, x_2), (x_3, x_4), \dots, (x_{d-1}, x_d)
$$

每一组二维向量使用不同频率的角度进行旋转：

$$
\phi_i(m) = m\theta_i
$$

其中 $\theta_i$ 是第 $i$ 个维度对的频率。通常不同维度对使用不同频率，使模型可以同时感知短距离和长距离位置关系。

对第 $i$ 个二维分组：

$$
\begin{bmatrix}
x_{2i}' \\
x_{2i+1}'
\end{bmatrix}
=
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
\begin{bmatrix}
x_{2i} \\
x_{2i+1}
\end{bmatrix}
$$

这个操作会应用到 query 和 key 上：

$$
q_m' = R_m q_m
$$

$$
k_n' = R_n k_n
$$

其中 $R_m$ 和 $R_n$ 是由位置 $m$、$n$ 决定的旋转矩阵。

## 为什么能表达相对位置

attention score 需要计算 query 和 key 的 dot product：

$$
(q_m')^T k_n' = (R_m q_m)^T (R_n k_n)
$$

利用旋转矩阵的性质：

$$
(R_m q_m)^T (R_n k_n) = q_m^T R_m^T R_n k_n
$$

二维旋转矩阵满足：

$$
R_m^T R_n = R_{n-m}
$$

所以：

$$
(q_m')^T k_n' = q_m^T R_{n-m} k_n
$$

这说明旋转后的 attention score 只通过 $n-m$ 这一相对距离感知位置关系，而不是只记住绝对位置编号。

这也是 RoPE 相比普通绝对位置 embedding 的重要优势：它和 attention score 的形式结合得更自然，让 query-key 匹配直接依赖相对位置。

## RoPE 放在 Transformer 的哪里

在 decoder-only Transformer 中，RoPE 通常应用在 self-attention 的 Q/K 上，而不是 V 上：

```text
hidden states H
  -> linear projection: Q, K, V
  -> apply RoPE to Q and K
  -> compute attention score: Q_rot K_rot^T
  -> softmax and causal mask
  -> weighted sum over V
```

为什么不旋转 V？因为 attention score 由 Q/K 决定，位置关系应该影响“关注谁”；而 V 是被读取的内容。RoPE 通过修改 Q/K 的匹配方式，就能影响 attention 权重，不需要直接改变 value 内容。

这也解释了为什么 RoPE 和 [[architecture/attention/attention|Attention]] 绑定很紧：它不是一个单独加到 embedding 上的位置向量，而是 query-key 匹配过程的一部分。

## 频率设计

RoPE 使用多组频率 $\theta_i$。常见形式类似 sinusoidal position encoding：

$$
\theta_i = base^{-\frac{2i}{d}}
$$

其中：

- $i$ 是维度对编号。
- $d$ 是 head dimension。
- $base$ 通常是一个较大的常数，例如 10000 或模型配置中的 `rope_theta`。

不同频率对应不同尺度的位置感知：

- 高频维度对短距离变化敏感。
- 低频维度对长距离变化更平滑。

这有点像用多种波长同时表示位置，使模型既能区分近邻顺序，也能保留较长距离的信息。

## RoPE 与绝对位置编码的区别

| 机制 | 做法 | 位置信息进入哪里 | 特点 |
|---|---|---|---|
| Absolute Position Embedding | token embedding 加上位置向量 | 输入表示 | 简单直接，但更偏绝对位置 |
| Sinusoidal Position Encoding | 加固定正弦/余弦位置向量 | 输入表示 | 不学习参数，可外推一定长度 |
| RoPE | 旋转 Q/K | attention score | 相对位置自然进入 query-key 匹配 |
| ALiBi | 给 attention score 加距离 bias | attention score | 直接偏置远近关系，外推友好 |

RoPE 的特点是：位置信息不是作为额外内容加到 hidden state，而是改变 query-key 的几何关系。这样 attention 在判断“该关注谁”时就已经考虑了相对位置。

## 一个直观例子

假设 token A 在位置 10，token B 在位置 13。RoPE 会让它们的 query/key 旋转不同角度。二者做 dot product 时，score 会包含位置差：

$$
13 - 10 = 3
$$

如果 token A 和 token B 内容相似，但相隔 3 个位置，与相隔 300 个位置时，RoPE 后的 dot product 会不同。模型因此可以学到“相同内容在不同距离下意义不同”。

例如在代码中：

```text
for i in range(n):
    total += i
```

变量 `i` 的不同出现位置之间有强关系，但这种关系和距离、缩进、局部结构都有关。RoPE 让 attention 在匹配变量引用时能同时考虑内容相似性和相对位置。

## 长上下文问题

RoPE 常被认为比普通 learned absolute position embedding 更适合长度外推，但它并不是天然解决所有长上下文问题。

训练时模型通常只见过最大长度 $L_{train}$。如果推理时扩展到远大于训练长度的位置，旋转角度会进入模型没有充分见过的范围，可能出现：

- attention pattern 失真。
- 高频维度旋转过快，远距离区分变得不稳定。
- 模型虽然能接收更长上下文，但不一定能有效利用远处信息。

因此，长上下文模型通常还需要 RoPE scaling 或额外训练。

常见思路包括：

- 调整 `rope_theta`，改变频率分布。
- Position interpolation，把长位置压缩到训练过的位置范围内。
- NTK-aware scaling，调整频率以改善外推。
- [[architecture/positional-encoding/yarn|YaRN]] 等更系统的长上下文扩展方法。

所以，“使用 RoPE”不等于“模型自动拥有无限长上下文能力”。

## 与 KV Cache 的关系

RoPE 对推理也有影响。自回归推理时，历史 token 的 K/V 会被缓存到 [[inference/kv-cache-and-memory/kv-cache|KV Cache]] 中。

因为 RoPE 应用在 K 上，缓存中的 K 通常已经包含了对应位置的旋转信息。生成新 token 时，只需要对新 token 的 Q 使用当前位置的 RoPE，再和历史 K 做 attention。

这意味着推理系统必须正确维护 position id：

- prefix cache 复用时，位置编号要一致。
- padding、batching、continuous batching 中，position id 不能错位。
- 长上下文扩展时，RoPE scaling 配置必须和模型训练/加载配置一致。

如果 position id 或 RoPE 配置错了，模型可能不会直接报错，但输出质量会明显异常。

## 设计取舍

| 设计点 | 优势 | 代价 |
|---|---|---|
| 旋转 Q/K | 相对位置进入 attention score，和机制结合自然 | 实现需要正确处理维度配对和 position id |
| 多频率设计 | 同时表达短距离和长距离信息 | 长上下文外推需要调频或 scaling |
| 不作用于 V | 保持 value 内容不被位置直接旋转 | 位置信息主要影响“关注谁”，不直接改变读出的内容 |
| 相对位置性质 | 更适合表达距离关系 | 不代表自动解决所有长度外推问题 |
| 与 KV Cache 兼容 | 推理时可缓存旋转后的 K | prefix/cache 复用时必须维护位置一致性 |

## 常见误解

### 误解一：RoPE 是加到 token embedding 上的位置向量

RoPE 不是把位置向量加到 token embedding 上，而是对 Q/K 做旋转。它改变的是 attention score 的计算方式。

### 误解二：RoPE 只表示绝对位置

RoPE 的旋转角度由绝对位置决定，但两个位置的 query-key dot product 会体现相对位置差。因此它同时使用绝对位置来构造旋转，并在 attention score 中体现相对位置信息。

### 误解三：用了 RoPE 就能无限扩展上下文

RoPE 比 learned absolute position embedding 更适合一定程度的外推，但模型能否有效处理长上下文，还取决于训练长度、数据分布、RoPE scaling、attention 机制和推理系统。

### 误解四：RoPE 只影响训练，不影响推理系统

RoPE 和推理系统关系很强。KV Cache、position id、prefix cache、batch 调度和长上下文配置都必须和 RoPE 处理方式一致。

### 误解五：RoPE 的数学只是实现细节

RoPE 的数学形式解释了为什么它能把相对距离放进 attention score。理解这一点有助于理解长上下文扩展、position interpolation 和 RoPE scaling。

## 与 Attention 的关系

RoPE 的位置编码不是独立于 attention 的附加模块，而是直接进入 Q/K 匹配过程：

$$
\mathrm{score}_{mn} = (R_m q_m)^T(R_n k_n)
$$

因此，RoPE 决定了 attention 如何感知“内容相关性 + 位置距离”。内容相似但位置关系不同，attention score 会不同。

这也解释了为什么 RoPE 通常只讨论在 Transformer attention 中的作用，而不是作为通用 embedding 技巧单独存在。

## 与训练的关系

训练时，RoPE 让模型在 next-token prediction 中学习不同位置关系的统计规律。例如：

- 局部 token 顺序。
- 长距离依赖。
- 文档结构。
- 代码缩进和作用域。
- 对话中的轮次关系。

如果训练数据和训练长度不足，模型即使数学上能计算更长位置，也不一定学会如何使用这些远距离关系。

## 与推理的关系

推理时，RoPE 的关键要求是 position id 和缓存一致。生成第 $t$ 个 token 时，query 必须使用位置 $t$ 的旋转；历史 key 必须使用它们各自位置的旋转。

如果一个 serving 系统做 prefix cache，它复用的不只是 token 内容，还包括历史 K 的位置语义。只要位置偏移处理错误，attention score 就会基于错误距离计算，导致模型输出退化。

因此，RoPE 是一个横跨 architecture 和 inference 的机制：它既决定模型如何表示位置，也影响长上下文 serving 的正确性。

## 相关概念

- [[architecture/attention/attention|Attention]] — RoPE 通过 Q/K 旋转改变 attention score。
- [[architecture/transformer/transformer|Transformer]] — RoPE 是现代 Transformer LLM 中常见的位置机制。
- [[architecture/positional-encoding/positional-encoding|Positional Encoding]] — 位置编码总览。
- [[architecture/positional-encoding/absolute-position|Absolute Position]] — 绝对位置 embedding 的对比对象。
- [[architecture/positional-encoding/sinusoidal-position|Sinusoidal Position]] — RoPE 频率设计的相关背景。
- [[architecture/positional-encoding/alibi|ALiBi]] — 另一类直接作用于 attention score 的位置方法。
- [[architecture/positional-encoding/yarn|YaRN]] — RoPE 长上下文扩展方法之一。
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]] — RoPE 在推理缓存中需要保持 position id 一致。
- [[fundamentals/linear-algebra/linear-transformation|Linear Transformation]] — 理解旋转矩阵和表示空间变换的基础。
