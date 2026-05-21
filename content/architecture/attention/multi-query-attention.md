---
title: Multi-Query Attention
created: 2026-01-31
published: 2026-01-31
modified: 2026-01-31
type: topic
status: growing
area: architecture
tags:
  - attention
  - mqa
  - kv-cache
aliases:
  - MQA
---

## 核心问题

Multi-Query Attention，简称 MQA，是一种减少自回归推理中 [[inference/kv-cache-and-memory/kv-cache|KV Cache]] 成本的 attention 变体。它要解决的问题是：标准 [[architecture/attention/multi-head-attention|Multi-Head Attention]] 中每个 query head 都有自己的 key/value head，推理时需要为所有 heads 保存历史 K/V，导致显存和 memory bandwidth 压力很大。

MQA 的做法很直接：保留多个 query heads，但让所有 query heads 共享同一组 key/value heads。

一句话概括：MQA 让查询仍然多头，但把 K/V 合并成单头共享，从而显著减少 KV Cache。

## 从 MHA 到 MQA

在标准 MHA 中，每个 head 都有独立的 query、key、value：

```text
Q heads: Q1 Q2 Q3 Q4 ... Qh
K heads: K1 K2 K3 K4 ... Kh
V heads: V1 V2 V3 V4 ... Vh
```

第 $i$ 个 query head 会和第 $i$ 个 key/value head 配对：

```text
Qi attends to Ki, Vi
```

MQA 改成：

```text
Q heads: Q1 Q2 Q3 Q4 ... Qh
K heads: K_shared
V heads: V_shared
```

也就是说：

```text
Q1 attends to K_shared, V_shared
Q2 attends to K_shared, V_shared
Q3 attends to K_shared, V_shared
...
Qh attends to K_shared, V_shared
```

这样 query 仍然可以有多个 heads，从不同子空间生成不同 attention scores；但历史 token 的 K/V 只需要保存一份。

## 公式视角

设输入 hidden state 是 $X$，标准 MHA 通常会生成：

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

然后 reshape 成多个 heads：

$$
Q \in \mathbb{R}^{T \times n_h \times d_h}
$$

$$
K \in \mathbb{R}^{T \times n_h \times d_h}
$$

$$
V \in \mathbb{R}^{T \times n_h \times d_h}
$$

在 MQA 中，$Q$ 仍然有 $n_h$ 个 heads，但 $K$ 和 $V$ 只有一组：

$$
Q \in \mathbb{R}^{T \times n_h \times d_h}
$$

$$
K \in \mathbb{R}^{T \times 1 \times d_h}
$$

$$
V \in \mathbb{R}^{T \times 1 \times d_h}
$$

attention 计算时，多个 query heads 共享同一份 K/V。

## 为什么能节省 KV Cache

自回归生成时，每产生一个新 token，都需要把它对应的 K/V 加入缓存。之后生成后续 token 时，模型会读取历史 K/V 来计算 attention。

标准 MHA 的 KV Cache 规模大致和 head 数成正比：

$$
\mathrm{KVCache}_{MHA} \propto 2 \times T \times n_h \times d_h
$$

其中 $2$ 表示 key 和 value 两份缓存。

MQA 因为只保存一组 K/V，所以变成：

$$
\mathrm{KVCache}_{MQA} \propto 2 \times T \times 1 \times d_h
$$

如果 $n_h = 32$，那么仅从 K/V heads 数量看，MQA 的 KV Cache 可以接近减少到 MHA 的 $1/32$。实际系统收益还会受到实现、batch size、量化、显存布局和 kernel 的影响，但方向非常明确：MQA 能显著降低 KV Cache 显存和带宽压力。

## 推理吞吐为什么会提升

LLM serving 中，decode 阶段经常受到 memory bandwidth 限制。生成每个 token 时，模型需要读取所有历史 token 的 K/V。如果 KV Cache 很大，GPU 大量时间会花在搬运缓存上，而不是纯矩阵计算。

MQA 减少 K/V heads 后：

- 每个请求的 KV Cache 变小。
- 同一张 GPU 能容纳更长上下文或更大 batch。
- decode 阶段读取历史 K/V 的带宽压力下降。
- serving 系统更容易提升吞吐。

所以 MQA 的收益在长上下文、多并发、batch decoding 场景中尤其明显。

## 表达能力的 trade-off

MQA 的代价是 K/V 表达能力下降。标准 MHA 中，每个 head 都有自己的 K/V 子空间；不同 heads 可以关注不同类型的信息。MQA 让所有 query heads 共享同一组 K/V，意味着 K/V 的多样性减少。

这可能带来几个影响：

- 某些任务上 attention 表达能力下降。
- 模型需要在训练中适应共享 K/V 的约束。
- 小模型或某些复杂任务可能更容易受影响。
- 对已有 MHA 模型直接改成 MQA 可能导致质量下降，需要专门训练或转换。

因此，MQA 不是“无损压缩”。它是推理效率和模型表达能力之间的取舍。

## MQA、GQA、MLA 的位置

MQA、[[architecture/attention/grouped-query-attention|Grouped-Query Attention]] 和 [[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]] 都是围绕 KV Cache 成本设计的 attention 变体，但它们采取的策略不同。

| 方法 | K/V 组织方式 | 直觉 | 取舍 |
|---|---|---|---|
| MHA | 每个 query head 有独立 K/V | 表达能力强 | KV Cache 最大 |
| MQA | 所有 query heads 共享一组 K/V | 最大限度减少 KV Cache | K/V 表达能力压缩最强 |
| GQA | 一组 query heads 共享一组 K/V | MHA 和 MQA 的折中 | 需要选择合适 group 数 |
| MLA | K/V 压缩成 latent representation | 用低维 latent cache 降低 KV Cache | 结构更复杂 |

可以把它们看成一条效率-表达能力的连续谱：

```text
MHA  ->  GQA  ->  MQA
表达能力更强      KV Cache 更小
```

MLA 不完全在这条线上，因为它不是简单减少 K/V heads 数量，而是改变 K/V 的缓存表示形式。

## 什么时候适合 MQA

MQA 更适合以下场景：

- 推理吞吐比极致模型质量更重要。
- 上下文较长，KV Cache 成本明显。
- serving 系统需要支持大 batch 或高并发。
- 模型从训练阶段就按 MQA 设计，而不是事后硬改。

对于需要极强表达能力的模型，或者对质量损失非常敏感的场景，GQA 可能是更稳的折中。因此很多现代 LLM 更常采用 GQA，而不是完全 MQA。

## 常见误解

### 误解一：MQA 会减少 query heads

不对。MQA 的 query heads 仍然是多个。它减少的是 key/value heads，让多个 query heads 共享同一组 K/V。

### 误解二：MQA 只减少显存，不影响速度

不对。KV Cache 变小不仅减少显存，也会减少 decode 阶段读取历史 K/V 的 memory bandwidth 压力，因此常常能提升推理吞吐。

### 误解三：MQA 是无损优化

不严谨。MQA 压缩了 K/V 表达能力，可能影响模型质量。它需要在训练中适配，不能简单认为和 MHA 完全等价。

### 误解四：有了 MQA 就不需要 KV Cache 优化

不对。MQA 只是降低 KV Cache 大小的一种结构设计。长上下文 serving 仍然需要 PagedAttention、KV Cache 管理、量化、batching 和调度优化。

## Related Notes

- [[architecture/attention/attention|Attention]]
- [[architecture/attention/multi-head-attention|Multi-Head Attention]]
- [[architecture/attention/grouped-query-attention|Grouped-Query Attention]]
- [[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]]
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]]
- [[inference/serving-systems/vllm|vLLM]]
