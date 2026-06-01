---
title: Sparse and Efficient Architectures
created: 2026-02-14
published: 2026-02-14
modified: 2026-05-31
---

稀疏与高效架构模块负责整理 MoE、Mamba、SSM、Linear Attention 等用于提升扩展性、效率或长序列建模能力的架构路线。

## Reading Path

1. [[architecture/sparse-and-efficient/moe|Mixture of Experts]]：通过 sparse FFN 扩大 total capacity。
2. [[architecture/sparse-and-efficient/linear-attention|Linear Attention]] 与 [[architecture/sparse-and-efficient/efficient-transformer|Efficient Transformer]]：降低 attention 长序列复杂度的路线。
3. [[architecture/sparse-and-efficient/ssm|State Space Model]] 与 [[architecture/sparse-and-efficient/mamba|Mamba / SSM]]：用状态空间结构替代或补充 attention 的路线。

## Notes

- [[architecture/sparse-and-efficient/moe|Mixture of Experts]]
- [[architecture/sparse-and-efficient/mamba|Mamba / SSM]]
- [[architecture/sparse-and-efficient/ssm|State Space Model]]
- [[architecture/sparse-and-efficient/linear-attention|Linear Attention]]
- [[architecture/sparse-and-efficient/efficient-transformer|Efficient Transformer]]

## Related Source TODOs

- [[sources/papers/2017-outrageously-large-neural-networks|Outrageously Large Neural Networks]]
- [[sources/papers/2020-gshard|GShard]]
- [[sources/papers/2021-switch-transformer|Switch Transformer]]
- [[sources/papers/2024-deepseekmoe|DeepSeekMoE]]
- [[sources/papers/2024-deepseek-v2|DeepSeek-V2]]
- [[sources/papers/2024-deepseek-v3|DeepSeek-V3]]
