---
title: Self-Attention
created: 2026-01-25
published: 2026-01-25
modified: 2026-05-31
type: topic
status: mature
area: architecture
tags:
  - attention
  - self-attention
---

Self-Attention 是 Transformer 中让同一序列内部 token 彼此读取信息的机制。它的输入是一组 token hidden states，输出仍是一组 hidden states，但每个位置的表示已经聚合了上下文信息。[[architecture/transformer/transformer|Transformer]] 之所以能并行建模长距离依赖，关键就在于 self-attention 不需要像 RNN 那样逐步传递状态，而是让每个位置直接和其他位置计算关系。

## 问题背景

序列中的 token 含义通常依赖上下文。例如代词指代、括号匹配、代码变量引用、问答中的约束条件，都要求某个位置读取其他位置的信息。固定窗口或递归状态很难灵活覆盖所有关系。

Self-attention 的目标是：

> 对序列中每个 token，根据当前内容动态决定应该关注哪些其他 token，并从它们那里聚合信息。

这里“self”表示 query、key、value 都来自同一条序列。

## Q/K/V 投影

给定 hidden states：

$$
H \in \mathbb{R}^{B \times T \times d_{model}}
$$

self-attention 先通过三个线性投影得到：

$$
Q = HW_Q,\quad K = HW_K,\quad V = HW_V
$$

其中：

- $Q$，query，表示当前位置想查询什么；
- $K$，key，表示每个位置提供什么可匹配索引；
- $V$，value，表示每个位置真正被读取的内容；
- $W_Q,W_K,W_V$ 是可学习投影矩阵。

同一个 token 的 hidden state 会被映射到三种角色中。这样，一个 token 既可以作为当前位置提出 query，也可以作为上下文被其他位置通过 key/value 读取。

## Scaled Dot-Product Attention

Self-attention 的核心计算是：

$$
\mathrm{Attention}(Q,K,V)
=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

分解来看：

1. 计算 score：

$$
S=QK^T
$$

2. 缩放并做 softmax：

$$
A=\mathrm{softmax}\left(\frac{S}{\sqrt{d_k}}\right)
$$

3. 按权重聚合 value：

$$
O=AV
$$

第 $t$ 个位置的输出可以写成：

$$
o_t=\sum_{j=1}^{T}A_{tj}v_j
$$

这表示当前位置从所有可见位置的 value 中加权读取信息。

## Causal Self-Attention

Decoder-only LLM 使用 causal self-attention。为了预测下一个 token，第 $t$ 个位置只能看到当前位置及之前的 token，不能看到未来：

$$
A_{tj}=0 \quad \text{for } j>t
$$

实现时通常在 softmax 前对未来位置加 $-\infty$：

$$
S_{tj}=-\infty \quad \text{for } j>t
$$

这让训练可以并行处理整段序列，但每个位置仍遵守自回归约束。没有 causal mask，next-token prediction 会泄漏未来信息。

## Bidirectional Self-Attention

Encoder-style 模型通常使用 bidirectional self-attention，也就是每个位置可以关注整段输入。它更适合理解任务，例如分类、检索或编码输入句子。

区别在于 mask：

| 类型 | 可见范围 | 常见模型 |
|---|---|---|
| Bidirectional self-attention | 全部输入位置 | BERT / encoder |
| Causal self-attention | 当前及历史位置 | GPT / LLaMA / Qwen / DeepSeek |

两者公式类似，关键差异是 attention mask 和训练目标。

## 为什么需要位置编码

Self-attention 本身根据内容相似度计算权重，不天然知道 token 顺序。如果没有 [[architecture/positional-encoding/positional-encoding|Positional Encoding]]，模型很难区分：

```text
dog bites man
man bites dog
```

因此 self-attention 通常配合 absolute position embedding、[[architecture/positional-encoding/rope|RoPE]] 或 [[architecture/positional-encoding/alibi|ALiBi]]。位置机制让 attention score 不只依赖 token 内容，也依赖顺序和距离。

## 复杂度

标准 self-attention 需要计算：

$$
QK^T \in \mathbb{R}^{T \times T}
$$

因此时间复杂度通常是：

$$
O(T^2 d)
$$

attention score/probability 的中间表示也与 $T^2$ 相关。训练长上下文时，这会带来显著的计算和显存压力。

推理时，decoder-only 模型会用 [[inference/kv-cache-and-memory/kv-cache|KV Cache]] 保存历史 K/V，避免每步重复计算历史 token 的 K/V。但每个新 token 仍需要读取历史 K/V，所以长上下文下 memory bandwidth 和 cache 管理仍是瓶颈。

## 表示能力

Self-attention 的价值在于内容相关的信息路由。它可以让某个 token 直接读取远处 token，而不需要信息一步步传递。例如：

```text
The trophy did not fit in the suitcase because it was too large.
```

处理 `it` 时，模型需要结合 `trophy`、`suitcase` 和 `large` 判断指代关系。Self-attention 允许 `it` 对这些位置分配较高权重，并把相关 value 聚合进当前 hidden state。

但注意：attention weight 只是某一层某个 head 的聚合权重，不等同于完整解释。最终表示还会经过多头融合、FFN、residual、normalization 和多层堆叠。

## 设计取舍

| 设计点 | 优势 | 代价 |
|---|---|---|
| 全局 self-attention | 任意 token 可直接交互 | $O(T^2)$ 成本高 |
| Causal mask | 支持自回归训练 | 不能读取未来 token |
| Bidirectional mask | 输入理解能力强 | 不适合直接自回归生成 |
| Q/K/V 分离 | 查询、索引、内容角色可学习 | 参数和实现复杂度增加 |
| KV Cache | 推理复用历史 K/V | 显存随上下文和 batch 增长 |

## 常见误解

- **误解：self-attention 就是关注所有 token。** 它会计算所有可见位置的权重，但不同位置权重不同。
- **误解：self-attention 天然知道顺序。** 顺序来自位置编码或 attention bias，不是 attention 公式自带的。
- **误解：causal mask 让训练不能并行。** 训练仍可以并行计算所有位置，只是 mask 限制可见范围。
- **误解：attention weight 是完整解释。** 它只是信息聚合的一部分，不能直接等同于模型因果解释。

## 相关概念

- [[architecture/attention/attention|Attention]]
- [[architecture/attention/multi-head-attention|Multi-Head Attention]]
- [[architecture/transformer/decoder-only-transformer|Decoder-only Transformer]]
- [[architecture/positional-encoding/positional-encoding|Positional Encoding]]
- [[architecture/positional-encoding/rope|RoPE]]
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]]
- [[fundamentals/neural-network-basics/softmax|Softmax]]
