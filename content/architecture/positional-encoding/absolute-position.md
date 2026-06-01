---
title: Absolute Position Embedding
created: 2026-02-01
published: 2026-02-01
modified: 2026-05-31
type: topic
status: mature
area: architecture
tags:
  - positional-encoding
  - absolute-position
---

Absolute Position Embedding 是最直接的位置编码方法：为序列中的每个绝对位置学习或定义一个位置向量，并把它加到 token embedding 上。它回答的问题是：“这个 token 位于第几个位置？”

## 基本形式

给定 token embedding：

$$
e_i \in \mathbb{R}^{d_{model}}
$$

以及第 $i$ 个位置的位置向量：

$$
p_i \in \mathbb{R}^{d_{model}}
$$

输入 Transformer 的初始表示为：

$$
h_i^{(0)}=e_i+p_i
$$

如果使用 learned absolute position embedding，所有 $p_i$ 组成一个可学习表：

$$
P \in \mathbb{R}^{L_{max}\times d_{model}}
$$

其中 $L_{max}$ 是模型支持的最大位置数。

## 为什么可行

[[architecture/attention/attention|Attention]] 本身不天然知道顺序。把 position embedding 加到 token embedding 后，同一个 token 出现在不同位置会得到不同初始表示：

$$
e_{\text{dog}}+p_3 \neq e_{\text{dog}}+p_{10}
$$

这样后续 attention 和 FFN 就能利用位置差异。

## 优势

- 实现简单，只需要查表并相加；
- 可学习，能适配训练数据中的位置分布；
- 对固定最大长度任务有效；
- 与标准 Transformer block 解耦，不需要修改 attention score 公式。

早期 GPT、BERT 等模型都使用或接近这种路线。

## 局限

Absolute position embedding 的主要问题是长度外推较弱。因为位置表只训练了 $1,\dots,L_{max}$ 范围内的位置，超过最大长度后没有自然定义的向量。即使扩展位置表，新位置也缺少训练。

它还更容易让模型记住绝对位置模式，而不是相对距离。例如“两个 token 相隔 3 个位置”在位置 10/13 和 300/303 上对应不同的绝对向量组合，模型需要自己学会这种相对关系。

## 与 RoPE / ALiBi 的区别

| 方法 | 位置信息进入哪里 | 主要特点 |
|---|---|---|
| Absolute Position Embedding | 输入 embedding | 简单、可学习、固定长度友好 |
| [[architecture/positional-encoding/rope|RoPE]] | Q/K 旋转 | 相对距离进入 attention score |
| [[architecture/positional-encoding/alibi|ALiBi]] | attention score bias | 用距离偏置引入位置关系 |

现代 decoder-only LLM 更常用 RoPE 或其扩展，因为它和 query-key 匹配过程结合更紧密，也更适合长上下文扩展。

## 常见误解

- **误解：absolute position embedding 不能表示顺序。** 它能表示绝对顺序，只是相对距离和长度外推不如某些方法自然。
- **误解：位置向量只在第一层有用。** 它加到输入后会通过多层计算影响后续 hidden states。
- **误解：扩大位置表就等于长上下文模型。** 新位置向量需要训练，长上下文能力还需要数据、attention 机制和推理系统支持。

## 相关概念

- [[architecture/positional-encoding/positional-encoding|Positional Encoding]]
- [[architecture/positional-encoding/sinusoidal-position|Sinusoidal Position Encoding]]
- [[architecture/positional-encoding/rope|RoPE]]
- [[architecture/positional-encoding/alibi|ALiBi]]
- [[architecture/attention/attention|Attention]]
