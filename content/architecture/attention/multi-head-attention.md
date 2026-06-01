---
title: Multi-Head Attention
created: 2026-01-25
published: 2026-01-25
modified: 2026-05-31
type: topic
status: mature
area: architecture
tags:
  - attention
  - multi-head-attention
---

Multi-Head Attention，简称 MHA，是 Transformer 中最经典的 attention 结构。它把 hidden state 投影到多个 attention heads，让模型在多个子空间中并行计算信息路由，再把各个 head 的输出拼接并融合。相比单头 attention，MHA 允许不同 head 学习不同类型的 token 关系。

## 为什么需要多个 Head

单个 attention head 只能用一组 $Q/K/V$ 投影计算相关性。语言和代码中的关系却有很多种：局部语法、长距离依赖、实体指代、格式标记、括号匹配、对话轮次等。如果只用一个投影空间，所有关系都要压到同一组 attention pattern 中。

MHA 的直觉是：

> 把表示分成多个子空间，让不同 head 并行学习不同的信息路由方式。

这并不意味着每个 head 都有清晰的人类语义标签，但多头结构为模型提供了更丰富的关系建模能力。

## 结构与形状

给定 hidden states：

$$
H \in \mathbb{R}^{B \times T \times d_{model}}
$$

设 head 数量为 $n_h$，每个 head 的维度为 $d_h$，通常：

$$
d_{model}=n_h d_h
$$

MHA 先投影：

$$
Q=HW_Q,\quad K=HW_K,\quad V=HW_V
$$

然后 reshape 为多头形式：

$$
Q,K,V \in \mathbb{R}^{B \times n_h \times T \times d_h}
$$

每个 head 独立计算 attention：

$$
head_i=\mathrm{Attention}(Q_i,K_i,V_i)
$$

最后拼接所有 head，并通过输出投影融合：

$$
\mathrm{MHA}(H)=\mathrm{Concat}(head_1,\dots,head_{n_h})W_O
$$

其中 $W_O$ 把拼接后的表示重新映射回 $d_{model}$。

## Output Projection 的作用

如果只是拼接多个 head，每个 head 的信息仍然相对分离。输出投影 $W_O$ 负责在 head 之间重新混合信息，让模型学习不同 head 输出如何组合。

因此 MHA 不只是“多个 attention 并排算完”：它包含 head-specific attention 和跨 head 的线性融合。

## 与 Self-Attention 的关系

[[architecture/attention/self-attention|Self-Attention]] 描述的是 Q/K/V 来自同一序列；Multi-Head Attention 描述的是 attention 的多头组织方式。二者可以组合：

- encoder self-attention 可以是 multi-head；
- decoder causal self-attention 可以是 multi-head；
- encoder-decoder cross-attention 也可以是 multi-head。

在 decoder-only LLM 中，常见的是 causal multi-head self-attention。

## 参数量与计算

标准 MHA 的 Q/K/V/O 投影参数量大致为：

$$
4d_{model}^2
$$

其中 Q、K、V 三个投影各约 $d_{model}^2$，输出投影 $W_O$ 约 $d_{model}^2$。在固定 $d_{model}$ 下，增加 head 数通常会减少每个 head 的 $d_h$，不一定显著改变投影参数总量；但 head 数会影响 attention pattern、KV Cache 组织和 kernel 实现。

标准 MHA 的 attention score 计算仍然与序列长度二次相关：

$$
O(T^2 d_{model})
$$

## KV Cache 成本

自回归推理时，每层需要缓存历史 token 的 K/V。标准 MHA 中，每个 query head 都有自己的 K/V head，因此 KV Cache 大小随 $n_h$ 增长：

$$
M_{\mathrm{KV}}
\propto
2 \times L \times T \times n_h \times d_h
$$

其中：

- $2$ 表示 key 和 value；
- $L$ 是层数；
- $T$ 是上下文长度；
- $n_h$ 是 heads 数；
- $d_h$ 是 head dimension。

由于 $n_h d_h=d_{model}$，标准 MHA 的每 token KV 表示维度接近 $2d_{model}$。长上下文和大 batch 下，这会成为 serving 的核心瓶颈。

这也是 [[architecture/attention/multi-query-attention|MQA]]、[[architecture/attention/grouped-query-attention|GQA]] 和 [[architecture/attention/multi-head-latent-attention|MLA]] 出现的重要原因：它们都试图降低 K/V 表示或缓存成本。

## Head 的可解释性边界

一些 head 可能呈现可解释模式，例如关注分隔符、局部邻居、复制位置或实体引用。但不能把每个 head 都理解成独立、稳定、可命名的“功能模块”。原因包括：

- 多个 head 可能冗余；
- head 功能可能随层数变化；
- FFN 和 residual 会重组 head 输出；
- 相同任务能力可能分布在多个 head 和多层之间。

因此，head visualization 可以作为诊断工具，但不应直接等同于模型解释。

## 设计取舍

| 设计点 | 优势 | 代价 |
|---|---|---|
| 多个 query heads | 多子空间建模关系 | KV Cache 和实现复杂度增加 |
| 较大 head dimension | 单 head 表达更强 | head 数减少或总宽度增加 |
| 较多 heads | attention pattern 更丰富 | 每 head 维度过小时可能受限 |
| 输出投影 | 融合不同 head 信息 | 增加参数和计算 |
| MHA | 表达能力强，结构标准 | 推理 KV Cache 成本最高 |

## 与 MQA / GQA / MLA 的关系

MHA 是这些变体的基准：

- [[architecture/attention/multi-query-attention|MQA]]：保留多个 Q heads，但所有 heads 共享一组 K/V。
- [[architecture/attention/grouped-query-attention|GQA]]：多个 Q heads 分组共享 K/V。
- [[architecture/attention/multi-head-latent-attention|MLA]]：把 K/V 压缩到 latent representation，再恢复用于 attention。

这些方法主要针对推理时 KV Cache 和 memory bandwidth 成本，而不是否定 MHA 的表达能力。

## 常见误解

- **误解：head 越多一定越好。** head 数、head dimension、模型宽度和训练规模要一起考虑。
- **误解：每个 head 都有固定语义。** head 可以有可解释倾向，但功能通常是分布式的。
- **误解：MHA 只影响训练。** 推理时 MHA 直接决定 KV Cache 规模。
- **误解：MQA/GQA 只是实现优化。** 它们改变了 K/V 参数共享方式，也可能影响表达能力。

## 相关概念

- [[architecture/attention/attention|Attention]]
- [[architecture/attention/self-attention|Self-Attention]]
- [[architecture/attention/multi-query-attention|Multi-Query Attention]]
- [[architecture/attention/grouped-query-attention|Grouped-Query Attention]]
- [[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]]
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]]
- [[inference/attention-acceleration/flash-attention|FlashAttention]]
