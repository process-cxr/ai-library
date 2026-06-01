---
title: Grouped-Query Attention
created: 2026-01-31
published: 2026-01-31
modified: 2026-05-31
type: topic
status: mature
area: architecture
tags:
  - attention
  - gqa
  - kv-cache
aliases:
  - GQA
---

## 核心问题

Grouped-Query Attention，简称 GQA，是介于 [[architecture/attention/multi-head-attention|Multi-Head Attention]] 和 [[architecture/attention/multi-query-attention|Multi-Query Attention]] 之间的 attention 变体。它要解决的问题是：MHA 表达能力强，但 KV Cache 太大；MQA 的 KV Cache 很小，但所有 query heads 共享一组 K/V，表达能力压缩过强。有没有一个中间方案？

GQA 的答案是：把 query heads 分成若干组，每一组 query heads 共享一组 key/value heads。

一句话概括：GQA 不是让所有 query heads 共享同一组 K/V，而是按组共享 K/V，在推理效率和表达能力之间折中。

## 从 MHA 和 MQA 到 GQA

标准 MHA 中，每个 query head 都有自己的 K/V：

```text
Q heads: Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8
K heads: K1 K2 K3 K4 K5 K6 K7 K8
V heads: V1 V2 V3 V4 V5 V6 V7 V8
```

MQA 中，所有 query heads 共享一组 K/V：

```text
Q heads: Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8
K heads: K_shared
V heads: V_shared
```

GQA 在中间：

```text
Q heads: Q1 Q2 Q3 Q4 | Q5 Q6 Q7 Q8
K heads: K_group1    | K_group2
V heads: V_group1    | V_group2
```

也就是说，多个 query heads 组成一个 group，同一组内共享 K/V，不同组使用不同 K/V。

如果有 32 个 query heads，8 个 KV heads，那么每 4 个 query heads 共享 1 个 KV head。这比 MHA 少很多 K/V，也比 MQA 保留更多 K/V 多样性。

## 公式和维度

设：

- $n_q$：query heads 数量。
- $n_{kv}$：key/value heads 数量。
- $d_h$：每个 head 的维度。
- $T$：上下文长度。

在 MHA 中：

$$
n_{kv} = n_q
$$

在 MQA 中：

$$
n_{kv} = 1
$$

在 GQA 中：

$$
1 < n_{kv} < n_q
$$

通常要求 $n_q$ 能被 $n_{kv}$ 整除。每个 KV head 对应：

$$
\frac{n_q}{n_{kv}}
$$

个 query heads。

KV Cache 大小大致和 $n_{kv}$ 成正比：

$$
\mathrm{KVCache} \propto 2 \times T \times n_{kv} \times d_h
$$

因此，GQA 可以通过减少 $n_{kv}$ 显著降低 KV Cache。

## 为什么 GQA 是折中

GQA 介于 MHA 和 MQA 之间：

| 方法 | Query Heads | KV Heads | KV Cache | 表达能力 |
|---|---:|---:|---|---|
| MHA | 多个 | 与 query heads 相同 | 最大 | 最强 |
| GQA | 多个 | 少于 query heads | 中等 | 中等到较强 |
| MQA | 多个 | 1 组 | 最小 | 压缩最强 |

这个折中非常适合大模型 serving。相比 MHA，GQA 可以明显降低 KV Cache；相比 MQA，它不会把所有 query heads 都压到同一组 K/V 上，因此通常更容易保持模型质量。

## KV Cache 收益

GQA 的主要收益来自减少 key/value heads。

假设一个模型有 32 个 query heads：

- MHA：32 个 KV heads。
- GQA：例如 8 个 KV heads。
- MQA：1 个 KV head。

那么相对 MHA：

- GQA 的 KV Cache 大约是 $8/32 = 1/4$。
- MQA 的 KV Cache 大约是 $1/32$。

GQA 不像 MQA 那样极致压缩，但它常常更稳。对于长上下文和高并发推理来说，1/4 的 KV Cache 已经能带来很大 serving 收益。

## 为什么现代 LLM 常用 GQA

很多现代 LLM 选择 GQA，是因为它适合大模型部署中的实际约束：

- 大模型需要保持较强质量，不能过度压缩 K/V。
- 长上下文需要降低 KV Cache。
- Serving 需要更高 batch size 和吞吐。
- GQA 的实现比 MLA 这类 latent compression 更简单。
- 从 MHA 迁移到 GQA 的质量风险通常比直接到 MQA 更可控。

LLaMA / Llama 系列、Qwen 等模型中都可以看到 GQA 的使用。尤其在长上下文模型中，GQA 已经成为非常常见的推理效率设计。

## 与 LLaMA 的关系

[[architecture/model-families/llama|LLaMA]] 早期版本主要沿着 dense decoder-only Transformer 路线发展。随着模型变大、上下文变长，KV Cache 成本越来越重要。Llama 2 在较大模型中引入 GQA，Llama 3 / 3.1 也继续使用 GQA。

这说明 GQA 的作用主要不是让模型“更会推理”，而是让大模型更容易推理部署。它降低 K/V heads 数量，从而减少显存和带宽成本。

理解 LLaMA 的推理效率时，GQA 是一个关键点：同样的参数规模和上下文长度，如果 K/V heads 更少，serving 的显存压力会明显下降。

## 与 MQA 的区别

GQA 可以看作 MQA 的放松版本。

MQA 的共享范围最大：所有 query heads 共享同一组 K/V。GQA 则只在组内共享。这个差别带来两个结果：

1. GQA 的 KV Cache 比 MQA 大。
2. GQA 的 K/V 表达能力通常比 MQA 更强。

因此，如果目标是极致压缩，MQA 更激进；如果目标是在质量和效率之间取平衡，GQA 更常见。

## 与 MLA 的区别

[[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]] 也在降低 KV Cache，但思路不同。

GQA 减少的是 K/V heads 数量。它的问题仍然可以用“几个 query heads 共享几个 K/V heads”来描述。

MLA 则把 K/V 联合压缩成 latent representation，缓存的是低维 latent vector 和 RoPE 相关部分。它不是简单减少 KV heads，而是改变 K/V 的表示和缓存形式。

所以：

- GQA：head sharing。
- MLA：latent compression。

二者都服务于推理效率，但机制不同，工程复杂度也不同。

## 设计选择

选择 GQA 时，一个关键超参数是 $n_{kv}$，也就是 KV heads 数量。

- $n_{kv}$ 越大，越接近 MHA，表达能力更强，但 KV Cache 更大。
- $n_{kv}$ 越小，越接近 MQA，KV Cache 更小，但表达能力压缩更强。

因此 GQA 不是一个单点设计，而是一组连续选择。不同模型会根据规模、训练数据、上下文长度和 serving 目标选择不同的 KV head 数。

## 常见误解

### 误解一：GQA 会减少 query heads

不对。GQA 通常保留多个 query heads，减少的是 key/value heads。query heads 仍然可以从多个子空间计算 attention scores。

### 误解二：GQA 和 MQA 一样

不一样。MQA 是所有 query heads 共享一组 K/V；GQA 是分组共享。GQA 是 MHA 与 MQA 之间的折中。

### 误解三：GQA 只影响显存，不影响速度

不对。KV Cache 变小也会降低 decode 阶段读取 K/V 的 memory bandwidth 压力，从而影响吞吐和延迟。

### 误解四：GQA 是无损优化

不严谨。GQA 通常比 MQA 更稳，但仍然减少了 K/V heads。它是一种 trade-off，效果依赖模型规模、训练方式和任务。

## Related Notes

- [[architecture/attention/attention|Attention]]
- [[architecture/attention/multi-head-attention|Multi-Head Attention]]
- [[architecture/attention/multi-query-attention|Multi-Query Attention]]
- [[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]]
- [[architecture/model-families/llama|LLaMA]]
- [[architecture/model-families/deepseek|DeepSeek]]
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]]
- [[inference/serving-systems/vllm|vLLM]]

## 经典论文与资料

- [[sources/papers/2023-gqa|Grouped-Query Attention]]
- [[sources/papers/2019-fast-transformer-decoding-one-write-head-is-all-you-need|Fast Transformer Decoding]]
