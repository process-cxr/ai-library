---
title: Attention
created: 2026-01-25
published: 2026-01-25
modified: 2026-10-20
---

Attention 模块负责整理注意力机制及其现代变体，包括 self-attention、multi-head attention、MQA、GQA、滑动窗口与块稀疏注意力，以及面向 GPU IO 的 FlashAttention。

## Reading Path

1. [[architecture/attention/attention|Attention 机制]]：Q/K/V、attention score、mask 和复杂度。
2. [[architecture/attention/self-attention|Self-Attention]]：同一序列内部的信息交互。
3. [[architecture/attention/multi-head-attention|Multi-Head Attention]]：多头子空间和 KV Cache 基线。
4. [[architecture/attention/multi-query-attention|MQA]]、[[architecture/attention/grouped-query-attention|GQA]]、[[architecture/attention/multi-head-latent-attention|MLA]]：降低 KV Cache 的结构路线。
5. [[architecture/attention/sliding-window-attention|Sliding Window Attention]]：从局部窗口理解因果与双向可见范围，再理解 dilation 和全局中转。
6. [[architecture/attention/sparse-attention|Sparse and Block-Sparse Attention]]：组合局部、全局、随机连接，并以 block 组织计算。
7. [[architecture/attention/hybrid-attention|Hybrid Attention]]：进一步压缩历史表示，并组合稀疏访问与全局交互。
8. [[inference/attention-acceleration/flash-attention|FlashAttention]]：IO-aware kernel 如何执行 attention。

## Notes

- [[architecture/attention/attention|Attention 机制]]
- [[architecture/attention/self-attention|Self-Attention]]
- [[architecture/attention/multi-head-attention|Multi-Head Attention]]
- [[architecture/attention/multi-query-attention|Multi-Query Attention]]
- [[architecture/attention/grouped-query-attention|Grouped-Query Attention]]
- [[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]]
- [[architecture/attention/sliding-window-attention|Sliding Window Attention]]
- [[architecture/attention/sparse-attention|Sparse and Block-Sparse Attention]]
- [[architecture/attention/hybrid-attention|Hybrid Attention]]

## Long-Context Attention

局部窗口降低每层的直接访问量，但远程证据需要额外传播路径。Longformer 用少量全局位置连接窗口，使问题和文档能够在多层中交换信息；双向窗口、独立投影与迁移训练详见 [[sources/papers/2020-longformer|Longformer 原文解析]]。

BigBird 将这类问题放入稀疏图的框架，分析局部、随机、全局连接的分工，并用 block 化适配硬件。其变体、表达能力证明与有限深度的限制见 [[sources/papers/2020-big-bird|BigBird 原文解析]]。两者都应同时考察可见范围、信息路径和实际 kernel，主题层的比较见 [[architecture/attention/sparse-attention|稀疏与块稀疏注意力]]。

## Implementation and IO

- [[inference/attention-acceleration/flash-attention|FlashAttention]]：通过 tiling、片上复用和 online softmax 减少 attention 的显存 IO。
- [[inference/attention-acceleration/attention-kernels|Attention Kernels]]：attention kernel 的通用实现问题。
- [[inference/attention-acceleration/flash-decoding|FlashDecoding]]：decode 阶段的并行化与缓存读取优化。

## Related Source Notes

- [[sources/papers/2017-attention-is-all-you-need|Attention Is All You Need]]
- [[sources/papers/2019-fast-transformer-decoding-one-write-head-is-all-you-need|Fast Transformer Decoding]]
- [[sources/papers/2023-gqa|Grouped-Query Attention]]
- [[sources/papers/2024-deepseek-v2|DeepSeek-V2]]
- [[sources/papers/2020-longformer|Longformer]]
- [[sources/papers/2020-big-bird|Big Bird]]
- [[sources/papers/2022-flashattention|FlashAttention]]
- [[sources/papers/2023-flashattention-2|FlashAttention-2]]
- [[sources/papers/2024-flashattention-3|FlashAttention-3]]
