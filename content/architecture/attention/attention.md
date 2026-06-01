---
title: Attention
created: 2026-01-25
published: 2026-01-25
modified: 2026-05-31
type: topic
status: mature
area: architecture
tags:
  - architecture
  - attention
aliases:
  - Attention Mechanism
  - Scaled Dot-Product Attention
---

## 问题背景

Attention 是一种让模型在处理当前位置时，动态选择并聚合其他位置的信息的机制。在大语言模型中，它是 [[architecture/transformer/transformer|Transformer]] 的核心组件，负责 token 之间的信息交互。

如果没有 attention，一个 token 的表示只能依赖局部窗口、固定卷积、递归状态或已经压缩过的上下文。这样会带来一个问题：不同 token 之间的关系并不是固定的。比如在一句话中，代词可能需要关联到很远处的实体，代码中的变量使用可能需要关联到前面的定义，问答任务中答案 token 需要关注问题中的关键约束。

Attention 的核心目标是解决这个问题：

> 对每个 token，根据当前内容动态决定应该从哪些 token 读取信息，以及读取多少。

因此，attention 可以看作一种内容相关的信息路由机制。它不是预先规定“只看左边 5 个 token”或“只看固定位置”，而是通过相似度计算决定哪些位置更重要。

## 核心直觉

Attention 可以用“查询资料”的过程来理解。

假设当前位置是一个 query，它想从上下文里找有用信息。上下文中每个 token 都提供一个 key 和一个 value：

- **Query**：当前位置提出的问题，表示“我想找什么”。
- **Key**：每个上下文位置的索引，表示“我这里有什么特征可匹配”。
- **Value**：每个上下文位置真正提供的内容，表示“如果你关注我，可以读走什么信息”。

模型先用 query 和每个 key 做匹配，得到 attention score；再把 score 转成权重；最后用这些权重对 value 加权求和。

所以 attention 并不是简单地“看所有 token”，而是：

1. 计算相关性。
2. 把相关性变成概率分布。
3. 按概率分布聚合信息。

## Scaled Dot-Product Attention

Transformer 中最常见的是 Scaled Dot-Product Attention：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：

- $Q \in \mathbb{R}^{n_q \times d_k}$：query 矩阵。
- $K \in \mathbb{R}^{n_k \times d_k}$：key 矩阵。
- $V \in \mathbb{R}^{n_k \times d_v}$：value 矩阵。
- $n_q$：query 的数量。
- $n_k$：可被关注的 key/value 数量。
- $d_k$：query/key 的维度。
- $d_v$：value 的维度。

计算分三步。

### 1. 计算 attention score

$$
S = QK^T
$$

$S_{ij}$ 表示第 $i$ 个 query 和第 $j$ 个 key 的匹配程度。如果二者方向相近，dot product 较大，表示 query 更应该关注该 key 对应的位置。

### 2. 缩放并 softmax

$$
A = \mathrm{softmax}\left(\frac{S}{\sqrt{d_k}}\right)
$$

$A$ 是 attention weight。每一行通常是一个概率分布，表示某个 query 对所有 key/value 位置的关注比例。

为什么要除以 $\sqrt{d_k}$？如果 $d_k$ 很大，dot product 的数值方差会变大，softmax 容易进入饱和区间：最大值权重接近 1，其余接近 0，梯度变得不稳定。缩放可以让 score 的量级更适合 softmax。

### 3. 加权聚合 value

$$
O = AV
$$

输出 $O$ 中的每个位置都是 value 的加权和。注意：attention weight 来自 query-key 匹配，但真正被聚合的是 value。

## 从 hidden states 到 Q/K/V

在 Transformer 的 self-attention 中，$Q$、$K$、$V$ 都来自同一组 hidden states：

$$
H \in \mathbb{R}^{B \times n \times d_{model}}
$$

通过三个线性投影得到：

$$
Q = HW_Q, \quad K = HW_K, \quad V = HW_V
$$

其中：

- $W_Q$ 学习如何把 token 表示变成“查询”。
- $W_K$ 学习如何把 token 表示变成“索引”。
- $W_V$ 学习如何把 token 表示变成“可读取内容”。

这三个投影很重要，因为同一个 token 在不同角色下需要不同表示。例如一个词既可以作为当前位置提出查询，也可以作为上下文被其他位置读取。Query、Key、Value 分离，让模型可以学习不同的信息匹配和信息传递方式。

## Self-Attention

[[architecture/attention/self-attention|Self-Attention]] 指 $Q$、$K$、$V$ 来自同一个序列。也就是说，序列内部的 token 彼此交互。

在语言模型中，第 $t$ 个 token 的表示可以通过 self-attention 聚合其他 token 的信息：

$$
h_t' = \sum_{j=1}^{n} A_{tj} v_j
$$

其中：

- $h_t'$ 是第 $t$ 个位置聚合后的表示。
- $A_{tj}$ 是第 $t$ 个位置对第 $j$ 个位置的 attention weight。
- $v_j$ 是第 $j$ 个位置的 value 向量。

如果是 encoder-style self-attention，每个位置通常可以看见所有位置。如果是 decoder-only language model，需要 causal mask。

## Causal Mask

在自回归语言模型中，模型训练目标是预测下一个 token。第 $t$ 个位置不能看到未来 token，否则训练会泄漏答案。

因此 decoder-only Transformer 使用 causal mask：

$$
A_{tj} = 0 \quad \text{for } j > t
$$

实际实现中通常是在 softmax 前，把未来位置的 score 加上 $-\infty$：

$$
S_{tj} = -\infty \quad \text{for } j > t
$$

这样 softmax 后未来位置的权重就是 0。

直观地看：

```text
位置 t 只能关注：x_1, x_2, ..., x_t
位置 t 不能关注：x_{t+1}, x_{t+2}, ...
```

这就是为什么 decoder-only Transformer 可以并行训练但仍保持自回归约束：训练时所有位置一起计算，但每个位置的 attention 被 mask 限制在合法上下文内。

## Multi-Head Attention

单个 attention head 只能在一个投影空间里计算 query-key 相似度。[[architecture/attention/multi-head-attention|Multi-Head Attention]] 会把表示拆成多个 head，让模型在多个子空间中并行学习不同关系。

$$
head_i = \mathrm{Attention}(HW_Q^{(i)}, HW_K^{(i)}, HW_V^{(i)})
$$

$$
\mathrm{MHA}(H) = \mathrm{Concat}(head_1, \dots, head_m)W_O
$$

其中：

- $m$ 是 head 数量。
- 每个 head 有自己的 $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$。
- $W_O$ 是输出投影，用来融合所有 head 的结果。

多头机制的价值不是简单地“多算几遍”，而是让不同 head 可以学习不同类型的信息路由模式。例如：

- 局部语法关系。
- 长距离依赖。
- 分隔符或格式 token。
- 实体指代。
- 代码中的括号、缩进、变量引用。

但需要注意：并不是每个 head 都一定对应清晰的人类可解释功能。head 的作用可能是冗余、组合式或随层数变化的。

## MHA、MQA 与 GQA

标准 MHA 中，每个 query head 都有自己的 key/value head。推理时为了避免重复计算历史 token 的 K/V，模型会保存 [[inference/kv-cache-and-memory/kv-cache|KV Cache]]。如果 head 很多、上下文很长，KV Cache 会占用大量显存。

因此出现了 MQA 和 GQA。

| 机制 | Query heads | Key/Value heads | 主要目标 |
|---|---:|---:|---|
| MHA | 多个 | 多个 | 表达能力强，标准多头注意力 |
| MQA | 多个 | 1 组共享 | 极大减少 KV Cache，但可能损失表达能力 |
| GQA | 多个 | 多组共享 | 在 MHA 和 MQA 之间折中 |

[[architecture/attention/multi-query-attention|Multi-Query Attention]] 让多个 query head 共享同一组 K/V。这样推理时 KV Cache 大幅减少，但 key/value 表达能力可能下降。

[[architecture/attention/grouped-query-attention|Grouped-Query Attention]] 把 query heads 分组，每组共享一组 K/V。它比 MQA 更保留多样性，又比 MHA 更省显存。很多现代 LLM 会使用 GQA 作为推理效率和模型效果之间的折中。

## Sliding Window Attention

标准 causal attention 中，每个 token 可以关注前面所有 token，复杂度和 KV Cache 都随上下文长度增长。[[architecture/attention/sliding-window-attention|Sliding Window Attention]] 限制每个 token 只关注最近一段窗口。

如果窗口大小为 $w$，第 $t$ 个 token 只能看：

$$
x_{t-w+1}, \dots, x_t
$$

这样 attention 的计算从全局 $O(n^2)$ 降低到近似 $O(nw)$。代价是模型不能在每一层直接访问窗口外的信息。为了保留长距离能力，一些模型会结合全局 token、层间传播或特殊位置策略。

## Attention 与位置编码

Attention 本身根据内容相似度计算权重，但不天然知道 token 的顺序。如果没有位置信息，序列被重新排列后，attention 很难区分“谁在前谁在后”。

因此 Transformer 必须引入位置机制，例如：

- [[architecture/positional-encoding/absolute-position|Absolute Position]]：给每个位置加一个绝对位置向量。
- [[architecture/positional-encoding/sinusoidal-position|Sinusoidal Position]]：使用正弦/余弦函数编码位置。
- [[architecture/positional-encoding/rope|RoPE]]：通过旋转 query/key 表示注入相对位置信息。
- [[architecture/positional-encoding/alibi|ALiBi]]：在 attention score 中加入位置偏置。

在现代 LLM 中，RoPE 很常见，因为它和 query-key dot product 结合紧密，能较自然地表达相对位置信息，并支持一定程度的长上下文扩展。

## 复杂度与瓶颈

标准 attention 的主要瓶颈来自 attention matrix：

$$
QK^T \in \mathbb{R}^{n \times n}
$$

对序列长度 $n$，计算复杂度通常是：

$$
O(n^2 d)
$$

attention score 的空间复杂度也与 $n^2$ 相关。虽然实际实现会通过 kernel 优化避免完整 materialize 某些中间矩阵，但长序列仍然是 attention 的主要压力来源。

在训练中，瓶颈主要是：

- attention score 计算量大。
- softmax 和 dropout 等中间激活占显存。
- 反向传播需要保存或重算中间状态。

在推理中，瓶颈主要是：

- 每生成一个 token 都要对历史 K/V 做 attention。
- KV Cache 随 batch size、层数、head 数、head dimension、sequence length 增长。
- 长上下文下 memory bandwidth 和 cache management 变得非常关键。

这也是为什么 [[inference/attention-acceleration/flash-attention|FlashAttention]]、[[inference/attention-acceleration/flash-decoding|FlashDecoding]]、PagedAttention、MQA/GQA 等工作都围绕 attention 展开。

## 最小例子

句子：

```text
The trophy would not fit in the suitcase because it was too large.
```

当模型处理 `it` 时，可能需要判断 `it` 指代 `trophy` 还是 `suitcase`。如果后面是 `too large`，语义上更可能是 trophy 太大，导致放不进 suitcase。

在 attention 中，`it` 对应位置的 query 会和前面 token 的 key 做匹配。理想情况下，它会给 `trophy`、`suitcase`、`large` 等相关位置较高权重，然后从这些位置的 value 中聚合信息。最终 `it` 的 hidden state 不再只是代词本身，而是包含上下文指代关系的表示。

这就是 attention 在语言理解中的作用：它让每个 token 的表示根据上下文动态改变。

## 设计取舍

| 设计点 | 优势 | 代价 |
|---|---|---|
| Dot-product attention | 计算简单，适合矩阵乘法加速 | 对尺度敏感，需要 $\sqrt{d_k}$ 缩放 |
| Softmax 权重 | 提供归一化的信息聚合分布 | 可能过于尖锐，且需要数值稳定实现 |
| Multi-Head | 多子空间建模不同关系 | 参数、计算和 KV Cache 增加 |
| Causal Mask | 保证自回归训练合法 | 不能直接访问未来信息 |
| MQA/GQA | 降低推理 KV Cache | 可能牺牲部分表达能力 |
| Sliding Window | 降低长上下文计算成本 | 长距离依赖需要额外机制补偿 |

Attention 的价值在于灵活的信息路由，但它的代价是复杂度和显存压力。现代大模型架构和推理系统的很多优化，都是在保留 attention 表达能力的同时降低它的系统成本。

## 常见误解

### 误解一：Attention weight 就是模型解释

Attention weight 能显示某个 head 在某层如何分配聚合权重，但不能直接等同于完整因果解释。模型输出还受到 FFN、residual、normalization、多层堆叠和其他 head 的影响。

### 误解二：Attention 会自动学到顺序

Attention 本身不包含顺序信息。顺序必须由 positional encoding、RoPE、ALiBi 或其他位置机制提供。

### 误解三：更多 head 一定更好

更多 head 增加表达空间，但也增加参数、计算和 KV Cache。实际效果取决于模型规模、head dimension、数据和训练设置。

### 误解四：MQA/GQA 只是工程优化

MQA/GQA 确实主要服务推理效率，但它们也改变了 attention 的参数共享方式，因此可能影响模型表达能力和训练动态。

### 误解五：Attention 解决了所有长上下文问题

标准 attention 允许直接访问长上下文，但并不意味着模型一定能有效利用全部上下文。长上下文还涉及位置外推、检索能力、训练数据分布、KV Cache 管理和 serving 成本。

## 与训练的关系

训练时，attention 决定每个位置如何聚合上下文信息。对 decoder-only LLM 来说，causal attention 保证第 $t$ 个位置只能使用 $x_{<t}$ 来预测 $x_t$。这与 [[training/pretraining/objective|Training Objective]]、[[fundamentals/information-theory/cross-entropy|Cross Entropy]] 和 [[fundamentals/probability/maximum-likelihood|Maximum Likelihood Estimation]] 直接相关。

Attention 的数值稳定性也会影响训练。比如 softmax 前的 score 过大可能导致权重过尖，混合精度训练中还需要注意溢出、underflow 和 kernel 实现细节。

## 与推理的关系

推理时，attention 的关键问题是如何避免重复计算历史信息。对于 decoder-only LLM，历史 token 的 K/V 可以缓存下来，新 token 只计算自己的 Q，然后和历史 K/V 做 attention。

这就是 [[inference/kv-cache-and-memory/kv-cache|KV Cache]] 的来源。KV Cache 让自回归推理从“每步重算全部历史”变成“每步复用历史 K/V”，大幅节省计算。但它也把瓶颈转移到了显存容量、显存带宽和 cache 管理。

因此，attention 不只是模型结构问题，也是推理系统问题。理解 attention 后，才能理解为什么 vLLM、PagedAttention、continuous batching、prefix cache、GQA 等系统设计会出现。

## 经典论文与资料

- [[sources/papers/2017-attention-is-all-you-need|Attention Is All You Need]]
- [[sources/papers/2019-fast-transformer-decoding-one-write-head-is-all-you-need|Fast Transformer Decoding]]
- [[sources/papers/2023-gqa|Grouped-Query Attention]]

## 相关概念

- [[architecture/transformer/transformer|Transformer]] — 以 attention 为核心的信息交互架构。
- [[architecture/attention/self-attention|Self-Attention]] — Q/K/V 来自同一序列的 attention。
- [[architecture/attention/multi-head-attention|Multi-Head Attention]] — 多个 head 并行建模不同关系子空间。
- [[architecture/attention/multi-query-attention|Multi-Query Attention]] — 多个 query head 共享一组 K/V，降低 KV Cache。
- [[architecture/attention/grouped-query-attention|Grouped-Query Attention]] — 在 MHA 与 MQA 之间折中。
- [[architecture/attention/sliding-window-attention|Sliding Window Attention]] — 限制注意力窗口以降低长序列成本。
- [[architecture/positional-encoding/rope|RoPE]] — 在 Q/K 中注入相对位置信息。
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]] — 自回归推理中保存历史 K/V 的缓存机制。
- [[inference/attention-acceleration/flash-attention|FlashAttention]] — 面向 attention 的 IO-aware 加速实现。
