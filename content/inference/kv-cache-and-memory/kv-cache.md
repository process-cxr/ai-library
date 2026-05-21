---
title: KV Cache
created: 2026-04-04
published: 2026-04-04
modified: 2026-04-04
type: topic
status: growing
area: inference
tags:
  - inference
  - memory-optimization
  - kv-cache
aliases:
  - Key-Value Cache
  - K/V Cache
---

## 核心问题

KV Cache 是自回归大语言模型推理中最重要的缓存机制之一。它要解决的问题是：生成第 $t$ 个 token 时，模型需要关注前面所有 token；如果每一步都重新计算所有历史 token 的 key/value states，计算会大量重复，推理会非常慢。

KV Cache 的做法是：在生成过程中，把每一层 attention 里历史 token 的 key 和 value 保存下来。下一步生成时，只为新 token 计算 query/key/value，然后复用历史 token 的 cached K/V。

一句话概括：KV Cache 用显存换计算，避免每个 decoding step 重复计算历史 token 的 K/V。

## 为什么自回归解码需要缓存

[[architecture/transformer/decoder-only-transformer|Decoder-only Transformer]] 按自回归方式生成文本：

```text
prompt -> token1 -> token2 -> token3 -> ...
```

生成每个新 token 时，模型都要基于已有上下文计算 attention。假设当前已经有 $T$ 个 token，如果没有缓存，第 $T+1$ 步就需要重新处理前面 $T$ 个 token 的所有 attention K/V。

这会造成大量重复：

- 第 1 步计算 token 1 的 K/V。
- 第 2 步又计算 token 1、2 的 K/V。
- 第 3 步又计算 token 1、2、3 的 K/V。
- 后面每一步都重复处理历史 token。

KV Cache 避免这种重复。每个历史 token 的 K/V 只计算一次，然后保存在缓存中供后续步骤读取。

## Attention 中缓存的是什么

在 [[architecture/attention/attention|Attention]] 中，每个 token 的 hidden state 会投影成 query、key、value：

$$
q_t = h_t W_Q
$$

$$
k_t = h_t W_K
$$

$$
v_t = h_t W_V
$$

生成当前 token 时，query 只来自当前 token；但 key/value 需要包含所有历史 token：

$$
\mathrm{Attention}(q_t, K_{\le t}, V_{\le t})
$$

其中：

- $q_t$ 是当前 token 的 query。
- $K_{\le t}$ 是从第 1 个 token 到第 $t$ 个 token 的 key。
- $V_{\le t}$ 是从第 1 个 token 到第 $t$ 个 token 的 value。

KV Cache 缓存的就是每一层、每个历史 token 的 $K$ 和 $V$。

## Prefill 与 Decode

LLM 推理通常可以分成两个阶段：prefill 和 decode。

### Prefill

Prefill 阶段处理用户输入的 prompt。假设 prompt 有 $T$ 个 token，模型会一次性并行处理这 $T$ 个 token，并为每一层生成对应的 K/V cache。

Prefill 的特点是：

- 并行度高。
- 计算量大。
- attention 需要处理 prompt 内部所有 token 的关系。
- 输出是最后一个位置的 logits，以及后续 decode 需要的 KV Cache。

### Decode

Decode 阶段每次生成一个新 token。每一步只计算当前 token 的 query/key/value，然后把新的 K/V append 到缓存里。

Decode 的特点是：

- 串行依赖强，因为 token 必须一个一个生成。
- 每步计算量相对小，但要读取越来越长的 KV Cache。
- 容易受到 memory bandwidth 限制。
- batch size、上下文长度和 KV Cache 管理会显著影响吞吐。

KV Cache 对 decode 阶段尤其关键。没有 KV Cache，decode 会反复重算历史；有 KV Cache，decode 主要变成“读取历史 K/V + 计算当前 token”。

## 显存占用如何估算

KV Cache 显存大致由以下因素决定：

- Transformer 层数 $L$。
- batch size $B$。
- sequence length $T$。
- KV heads 数量 $n_{kv}$。
- 每个 head 的维度 $d_h$。
- key 和 value 两份缓存。
- 每个元素的字节数，例如 FP16/BF16 是 2 bytes。

一个常见估算式是：

$$
\mathrm{KVCacheBytes} \approx 2 \times L \times B \times T \times n_{kv} \times d_h \times \mathrm{bytes}
$$

其中最前面的 $2$ 表示 K 和 V 两份。

这个公式解释了为什么长上下文很贵：KV Cache 和上下文长度 $T$ 线性相关。如果 context 从 4K 增加到 128K，KV Cache 也会大约增加 32 倍。

## 一个简单例子

假设一个模型有：

- $L = 32$ 层；
- $B = 1$；
- $T = 32768$ tokens；
- $n_{kv} = 32$；
- $d_h = 128$；
- BF16，每个元素 2 bytes；

则 KV Cache 约为：

$$
2 \times 32 \times 1 \times 32768 \times 32 \times 128 \times 2
$$

约等于 17GB。

这只是 batch size 为 1 的情况。如果 batch size 增大，或者上下文继续变长，KV Cache 会迅速成为主要显存开销。

## MHA、MQA、GQA、MLA 对 KV Cache 的影响

不同 attention 结构对 KV Cache 的影响主要体现在 $n_{kv}$ 或缓存表示方式上。

### MHA

[[architecture/attention/multi-head-attention|Multi-Head Attention]] 中，每个 query head 都有自己的 K/V head。此时：

$$
n_{kv} = n_q
$$

表达能力强，但 KV Cache 最大。

### MQA

[[architecture/attention/multi-query-attention|Multi-Query Attention]] 中，所有 query heads 共享一组 K/V。此时：

$$
n_{kv} = 1
$$

KV Cache 最小，但 K/V 表达能力压缩最强。

### GQA

[[architecture/attention/grouped-query-attention|Grouped-Query Attention]] 中，多个 query heads 分组共享 K/V。此时：

$$
1 < n_{kv} < n_q
$$

它是 MHA 和 MQA 之间的折中，也是现代 LLM 中非常常见的设计。

### MLA

[[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]] 不只是减少 KV heads 数量，而是把 K/V 联合压缩成 latent representation。它缓存的是压缩后的 latent vector 和位置相关部分，从表示方式上降低 KV Cache。

### Hybrid Attention

[[architecture/attention/hybrid-attention|Hybrid Attention]] 更关注超长上下文场景下的 sequence-level compression 和 sparse/dense 组合。它不是只改变每个 token 的 K/V head 数，而是处理 1M context 下如何压缩、选择和访问长序列历史。

## KV Cache 与吞吐

KV Cache 影响的不只是显存，也影响吞吐。

在 decode 阶段，每生成一个 token，模型都要读取历史 K/V。随着上下文变长，读取 K/V 的 memory bandwidth 成本越来越高。很多 serving 场景下，decode 不是纯 compute-bound，而是 memory-bound。

因此，减少 KV Cache 可以带来多重收益：

- 更低显存占用。
- 更高 batch capacity。
- 更低 memory bandwidth 压力。
- 更高 decode throughput。
- 更容易支持长上下文。

这也是为什么 MQA、GQA、MLA、PagedAttention 等技术都围绕 KV Cache 展开。

## KV Cache 的生命周期

一个请求的 KV Cache 通常经历以下过程：

```text
prefill prompt
  -> allocate KV blocks
  -> write K/V for prompt tokens
  -> decode token by token
  -> append new K/V each step
  -> request finished
  -> release cache memory
```

如果是多轮对话，系统可能保留前面轮次的 KV Cache，避免每次都重新 prefill 全部历史。这就引出了 prefix cache、context caching 和 cache eviction 等问题。

## Prefix Cache

[[inference/kv-cache-and-memory/prefix-cache|Prefix Cache]] 指复用相同前缀对应的 KV Cache。

例如多个请求共享同一个 system prompt、工具说明、文档前缀或 few-shot examples，那么这些前缀的 K/V 可以只计算一次，后续请求复用。这样可以减少 prefill 成本。

Prefix cache 的难点是：

- 如何判断前缀完全一致；
- 如何管理不同请求共享的 cache；
- 如何处理 cache 命中、淘汰和权限隔离；
- 如何避免缓存占用过多显存。

## PagedAttention

[[inference/kv-cache-and-memory/paged-attention|PagedAttention]] 是 vLLM 中提出的 KV Cache 管理思想。它借鉴操作系统分页，把 KV Cache 切成 blocks/pages，而不是为每个请求连续分配一大块显存。

它要解决的是 KV Cache 的内存碎片和动态增长问题：

- 不同请求长度不同。
- Decode 过程中 cache 持续增长。
- 请求完成时间不同。
- 连续显存分配容易造成碎片和浪费。

PagedAttention 通过 block table 管理逻辑 token 到物理 cache blocks 的映射，让 serving 系统能更灵活地分配、共享和释放 KV Cache。

## Cache Eviction

[[inference/kv-cache-and-memory/cache-eviction|Cache Eviction]] 是指当缓存资源不足时，系统决定丢弃哪些 KV Cache。

在有限显存下，不可能无限保留所有请求和所有前缀的 cache。系统需要根据策略淘汰缓存，例如：

- 最近最少使用；
- 低复用概率；
- 低价值 prefix；
- 已完成请求；
- 超过最大上下文长度的早期 tokens。

Cache eviction 的难点是平衡显存利用率、命中率、延迟和结果正确性。不能随便丢弃当前 generation 必须依赖的 K/V，否则 attention 会出错。

## 常见误解

### 误解一：KV Cache 会减少模型参数显存

不对。KV Cache 是推理时为请求动态保存的 activation-like memory，不是模型权重。它不会减少参数显存，而是额外占用显存来减少重复计算。

### 误解二：有了 KV Cache，生成每个 token 就不需要看历史了

不对。KV Cache 不是让模型忽略历史，而是把历史 token 的 K/V 预先保存下来。生成新 token 时仍然要读取历史 K/V 并计算 attention。

### 误解三：KV Cache 越大越好

不一定。更大的 KV Cache 允许更长上下文和更多并发状态，但也会占用显存、降低 batch capacity、增加带宽压力。Serving 系统需要管理 cache，而不是无限扩张。

### 误解四：MQA/GQA/MLA 只是模型结构优化，和推理系统无关

不对。这些结构直接改变 KV Cache 大小和读取方式，会影响 serving 系统的显存规划、batching、throughput 和 latency。

### 误解五：长上下文只要扩展 RoPE 就够了

不对。[[architecture/positional-encoding/rope|RoPE]] 影响位置建模，但长上下文推理还需要处理 KV Cache 显存、attention 计算、训练长度、数据分布和系统调度。

## Related Notes

- [[architecture/attention/attention|Attention]]
- [[architecture/attention/multi-head-attention|Multi-Head Attention]]
- [[architecture/attention/multi-query-attention|Multi-Query Attention]]
- [[architecture/attention/grouped-query-attention|Grouped-Query Attention]]
- [[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]]
- [[architecture/attention/hybrid-attention|Hybrid Attention]]
- [[architecture/positional-encoding/rope|RoPE]]
- [[inference/kv-cache-and-memory/paged-attention|PagedAttention]]
- [[inference/kv-cache-and-memory/prefix-cache|Prefix Cache]]
- [[inference/kv-cache-and-memory/cache-eviction|Cache Eviction]]
- [[inference/serving-systems/vllm|vLLM]]
