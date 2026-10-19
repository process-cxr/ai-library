---
title: Attention
created: 2026-01-25
published: 2026-01-25
modified: 2026-05-31
---

Attention 模块负责整理注意力机制及其现代变体，包括 self-attention、multi-head attention、MQA、GQA、滑动窗口注意力，以及面向 GPU IO 的 FlashAttention。

## Reading Path

1. [[architecture/attention/attention|Attention 机制]]：Q/K/V、attention score、mask 和复杂度。
2. [[architecture/attention/self-attention|Self-Attention]]：同一序列内部的信息交互。
3. [[architecture/attention/multi-head-attention|Multi-Head Attention]]：多头子空间和 KV Cache 基线。
4. [[architecture/attention/multi-query-attention|MQA]]、[[architecture/attention/grouped-query-attention|GQA]]、[[architecture/attention/multi-head-latent-attention|MLA]]：降低 KV Cache 的结构路线。
5. [[architecture/attention/sliding-window-attention|Sliding Window Attention]] 与 [[architecture/attention/hybrid-attention|Hybrid Attention]]：长上下文下的稀疏、压缩和混合 attention。
6. [[inference/attention-acceleration/flash-attention|FlashAttention]]：不改变 dense attention 数学目标的 IO-aware kernel 路线。

## Notes

- [[architecture/attention/attention|Attention 机制]]
- [[architecture/attention/self-attention|Self-Attention]]
- [[architecture/attention/multi-head-attention|Multi-Head Attention]]
- [[architecture/attention/multi-query-attention|Multi-Query Attention]]
- [[architecture/attention/grouped-query-attention|Grouped-Query Attention]]
- [[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]]
- [[architecture/attention/sliding-window-attention|Sliding Window Attention]]
- [[architecture/attention/hybrid-attention|Hybrid Attention]]

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
