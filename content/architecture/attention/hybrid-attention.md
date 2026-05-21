---
title: Hybrid Attention
created: 2026-01-31
published: 2026-01-31
modified: 2026-01-31
type: topic
status: growing
area: architecture
tags:
  - attention
  - hybrid-attention
  - long-context
  - deepseek
aliases:
  - Hybrid Attention Architecture
  - DeepSeek Hybrid Attention
---

## 核心问题

Hybrid Attention 是 DeepSeek V4 技术文档中提出的长上下文 attention 方向。它要解决的问题是：当上下文长度扩展到 1M tokens 级别时，标准 dense attention 和完整 KV Cache 的计算、显存和带宽成本都很难承受；单一压缩或单一稀疏策略又可能损失重要信息。

因此，DeepSeek V4 使用 Hybrid Attention Architecture，把两类 attention 路径结合起来：

- Compressed Sparse Attention，简称 CSA。
- Heavily Compressed Attention，简称 HCA。

它的核心直觉是：长上下文里并不是所有 token 都需要以完整精度、完整密度参与 attention。模型需要一部分机制处理稀疏但重要的信息，也需要一部分机制保留压缩后的全局信息。

## 为什么长上下文需要新的 Attention

标准 [[architecture/attention/attention|Attention]] 的计算复杂度通常随序列长度二次增长。即使在自回归推理中使用 [[inference/kv-cache-and-memory/kv-cache|KV Cache]] 避免重复计算历史 K/V，模型仍然需要在每一步读取大量历史 key/value。

当 context length 从 4K、32K、128K 扩展到 1M，瓶颈会变得非常明显：

- Prefill 阶段需要处理很长的输入序列。
- Decode 阶段每个新 token 都要访问长历史的 KV Cache。
- KV Cache 显存占用随层数、head 数、head dimension 和上下文长度增长。
- Memory bandwidth 可能比矩阵乘法本身更限制吞吐。
- 很长上下文中，大量 token 对当前预测可能并不重要。

所以长上下文模型常常需要组合多种策略：压缩、稀疏、分块、检索、滑动窗口、全局 token、KV Cache 管理和系统级优化。Hybrid Attention 就属于这种组合式思路。

## DeepSeek V4 中的定义

根据 DeepSeek V4 technical documentation，Hybrid Attention 由 CSA 和 HCA 组成：

```text
Hybrid Attention
  = Compressed Sparse Attention
  + Heavily Compressed Attention
```

文档给出的描述是：

- CSA compresses KV caches along the sequence dimension and applies DeepSeek Sparse Attention。
- HCA applies heavier compression with dense attention。

这两句话是当前公开资料中最关键的定义。由于文档不是完整论文，很多细节还没有完全展开，因此写作时应避免把 CSA/HCA 过度具体化成未公开的公式。

## Compressed Sparse Attention

Compressed Sparse Attention，简称 CSA，可以拆成两个关键词：compressed 和 sparse。

### Compressed

Compressed 表示它会沿 sequence dimension 压缩 KV cache。这里压缩的不是模型权重，而是长上下文中的历史 token 表示。目标是减少需要保存、读取和参与 attention 的序列信息。

直觉上，1M tokens 中很多信息并不需要以 token-level 原始形式参与每一步计算。压缩可以把一段上下文的信息汇聚成更紧凑的表示，从而减少 memory 和 bandwidth 成本。

### Sparse

Sparse 表示它不会对所有历史位置做完整 dense attention，而是应用 DeepSeek Sparse Attention。稀疏 attention 的基本思想是：当前 token 只关注历史上下文中一部分位置或块，而不是所有位置。

稀疏化的收益是显然的：如果每个 token 只和较少历史位置交互，计算和读取成本都会降低。代价也很明显：如果稀疏模式选错，模型可能错过关键信息。

因此 CSA 的关键难点在于：如何在压缩后仍然选择到重要上下文。

## Heavily Compressed Attention

Heavily Compressed Attention，简称 HCA。它的描述是 heavier compression with dense attention。

这和 CSA 形成互补：

- CSA 偏向压缩后做稀疏选择。
- HCA 偏向更强压缩后保留 dense attention。

可以把 HCA 理解为一种全局信息保留路径。它把长上下文压缩得更厉害，让 dense attention 的成本变得可控。虽然压缩会丢失细节，但 dense attention 能让当前 token 访问更全局的压缩信息。

这种设计试图避免单纯 sparse attention 的风险：如果只依赖稀疏选择，模型可能漏掉远处但重要的信息；如果保留一条强压缩的 dense 路径，模型至少可以看到某种全局摘要。

## 为什么要 Hybrid

Hybrid Attention 的关键不是 CSA 或 HCA 单独哪个更强，而是两者互补。

长上下文中的信息可以粗略分成几类：

- 局部上下文：离当前 token 很近，通常高度相关。
- 稀疏关键证据：距离很远，但对当前任务非常重要。
- 全局背景：不一定需要逐 token 精确访问，但需要保留整体语义。
- 噪声或低价值上下文：存在于输入中，但对当前预测帮助有限。

CSA 更适合处理“稀疏关键证据”：通过 sparse attention 在长序列中选择重要位置。HCA 更适合处理“全局背景”：通过更强压缩，让 dense attention 仍能访问整体信息。

因此 Hybrid Attention 的目标是让模型同时具备两种能力：

1. 在长上下文中找到关键局部/远程证据。
2. 在压缩表示中保留全局语义背景。

## 与 MLA 的关系

[[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]] 和 Hybrid Attention 都服务于 attention efficiency，但它们的侧重点不同。

MLA 的核心是 low-rank joint compression for keys and values。它主要解决的是：如何减少推理时每个历史 token 需要缓存的 K/V 表示。

Hybrid Attention 的核心是长上下文下的 sequence-level compression 和 sparse/dense 组合。它主要解决的是：当上下文达到 1M tokens 时，如何降低对超长历史序列的访问成本，同时保留关键和全局信息。

可以粗略理解为：

| 机制 | 主要压缩对象 | 核心目标 |
|---|---|---|
| MLA | 每个 token 的 K/V 表示维度 | 减少 KV Cache 表示大小 |
| Hybrid Attention | 长序列上的历史上下文 | 降低 1M context 下的计算和内存成本 |

二者并不是互斥关系。DeepSeek-V3 强调 MLA，DeepSeek-V4 强调 Hybrid Attention，说明 DeepSeek 的长上下文路线从“压缩 K/V 表示”进一步扩展到“压缩和稀疏化长序列访问”。

## 与 Sliding Window Attention 的区别

[[architecture/attention/sliding-window-attention|Sliding Window Attention]] 通常让 token 只关注固定窗口内的邻近 token。这非常适合降低成本，但它天然更偏局部。如果任务需要访问很远处的信息，纯滑动窗口可能不够。

Hybrid Attention 的目标更复杂。它不只是限制在局部窗口，而是试图通过压缩、稀疏选择和全局压缩表示来服务超长上下文。

简单说：

- Sliding window：主要靠局部窗口降成本。
- Hybrid Attention：靠压缩 + 稀疏 + dense compressed path 组合降成本。

## 与 KV Cache 的关系

Hybrid Attention 和 KV Cache 的关系非常直接。DeepSeek V4 文档明确说 CSA 会压缩 KV caches along the sequence dimension。这意味着它不仅关心 attention score 的计算量，也关心历史 K/V 如何存储和读取。

在长上下文推理中，KV Cache 的压力有两个维度：

- representation dimension：每个 token 的 K/V 表示有多大。
- sequence length：要为多少历史 token 保存和访问 K/V。

MLA 更偏向减少 representation dimension；Hybrid Attention 更偏向处理 sequence length 维度上的压缩和选择。

这也是为什么 V4 的 1M context 需要 Hybrid Attention 支撑。如果仍然用普通 dense attention + 完整 KV Cache，成本会非常高。

## 优势

### 1. 支撑 1M Context

Hybrid Attention 的直接目标是让 1M context 在官方服务中更可行。它通过压缩和稀疏化降低长上下文的 compute 和 memory 压力。

### 2. 兼顾稀疏证据与全局背景

CSA 可以帮助模型在长上下文中选择关键位置，HCA 可以在压缩表示上保留更全局的信息。二者结合比单一路径更灵活。

### 3. 更贴近 Agent 场景

Agentic coding、多文档处理、长任务规划都可能产生非常长的上下文。Hybrid Attention 这样的机制让模型更可能在长上下文中保持可用成本。

## 风险与边界

### 1. 压缩可能丢失细节

任何压缩都会带来信息损失。HCA 的 heavier compression 尤其需要关注：它可能保留全局语义，但丢掉细粒度证据。

### 2. 稀疏选择可能漏掉关键信息

Sparse attention 的难点是选对位置。如果稀疏模式没覆盖关键证据，模型可能在长上下文中表现不稳定。

### 3. 公开资料细节有限

当前 DeepSeek V4 technical documentation 对 CSA/HCA 的描述还比较高层。没有完整公式、kernel 实现、ablation 和失败案例时，不应把它写成完全确定的机制。

### 4. 1M Context 不等于可靠推理 1M Tokens

支持 1M context 只是能力上限或系统接口能力。真实任务效果仍然需要看 long-context retrieval、multi-hop reasoning、needle-in-haystack、代码库理解、agent benchmark 等评测。

## 常见误解

### 误解一：Hybrid Attention 就是 Sliding Window Attention

不对。Sliding window 主要限制局部窗口；Hybrid Attention 是压缩、稀疏和 dense compressed path 的组合，更面向超长上下文。

### 误解二：Hybrid Attention 解决了所有长上下文问题

不对。它降低 attention 和 KV Cache 成本，但长上下文还依赖训练数据、位置编码、推理系统、prompt 结构和评测方式。

### 误解三：压缩后的 dense attention 一定等价于完整 dense attention

不严谨。压缩后的 dense attention 可以保留全局信息，但它不是无损替代。压缩强度越大，细节损失风险越高。

### 误解四：稀疏 attention 一定会损失效果

也不一定。长上下文中很多 token 对当前预测并不重要。合理的稀疏选择可能在大幅降低成本的同时保持效果，关键在于选择机制和训练方式。

## Related Notes

- [[architecture/attention/attention|Attention]]
- [[architecture/attention/sliding-window-attention|Sliding Window Attention]]
- [[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]]
- [[architecture/model-families/deepseek|DeepSeek]]
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]]
- [[application/agents/|Agents]]
