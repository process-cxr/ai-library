---
title: Attention Acceleration
created: 2026-04-05
published: 2026-04-05
modified: 2026-10-20
---

Attention 加速模块负责整理 FlashAttention、FlashDecoding、attention kernel 和长序列 attention 的系统优化。

## Notes

- [[inference/attention-acceleration/flash-attention|FlashAttention]]
- [[inference/attention-acceleration/flash-decoding|FlashDecoding]]
- [[inference/attention-acceleration/attention-kernels|Attention Kernels]]

## Related Papers

- [[sources/papers/2022-flashattention|FlashAttention]]：IO-aware exact attention 的原始论文。
- [[sources/papers/2023-flashattention-2|FlashAttention-2]]：并行划分和 warp 工作分配改进。
- [[sources/papers/2024-flashattention-3|FlashAttention-3]]：Hopper 异步执行与 FP8 attention。
- [[sources/papers/2026-dosa-dots-sparse-attention|DoSA: Dots Sparse Attention]]：利用合法 score bounds 跳过不可能入选的完整多头评分，保持原始稀疏 indexer 的 Top-K 选择。
