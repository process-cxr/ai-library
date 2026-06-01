---
title: Positional Encoding
created: 2026-02-01
published: 2026-02-01
modified: 2026-05-31
---

位置编码模块负责整理序列位置信息的建模方式，包括绝对位置、正弦位置编码、RoPE、ALiBi、YaRN 和长上下文扩展。

## Reading Path

1. [[architecture/positional-encoding/positional-encoding|位置编码]]：总览 attention 为什么需要顺序信息。
2. [[architecture/positional-encoding/absolute-position|Absolute Position Embedding]] 与 [[architecture/positional-encoding/sinusoidal-position|Sinusoidal Position Encoding]]：输入层位置向量路线。
3. [[architecture/positional-encoding/rope|RoPE]]：现代 decoder-only LLM 常用的 Q/K 旋转路线。
4. [[architecture/positional-encoding/alibi|ALiBi]]：attention score bias 路线。
5. [[architecture/positional-encoding/yarn|YaRN]]：RoPE 长上下文扩展路线。

## Notes

- [[architecture/positional-encoding/positional-encoding|位置编码]]
- [[architecture/positional-encoding/absolute-position|Absolute Position Embedding]]
- [[architecture/positional-encoding/sinusoidal-position|Sinusoidal Position Encoding]]
- [[architecture/positional-encoding/rope|RoPE]]
- [[architecture/positional-encoding/alibi|ALiBi]]
- [[architecture/positional-encoding/yarn|YaRN]]

## Related Source TODOs

- [[sources/papers/2017-attention-is-all-you-need|Attention Is All You Need]]
- [[sources/papers/2021-roformer|RoFormer]]
- [[sources/papers/2021-alibi|ALiBi]]
- [[sources/papers/2023-position-interpolation|Position Interpolation]]
- [[sources/papers/2023-yarn|YaRN]]
- [[sources/papers/2024-longrope|LongRoPE]]
