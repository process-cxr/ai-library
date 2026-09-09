---
title: Multimodal
created: 2026-02-15
published: 2026-02-15
modified: 2026-02-15
---

多模态架构模块负责整理不同模态如何被编码、对齐并接入 language model。当前以 Vision-Language Model 为主线，逐步扩展到 Audio、Video、Interleaved Multimodal 和 action-conditioned perception。

这条路线建立在 Text LM Core 之上：文本模型负责 sequence modeling，模态 encoder 负责把图像、音频或视频转换成可处理的 representation，projector / connector 负责空间映射，fusion 机制负责跨模态交互。

## Reading Path

```text
Modality representation
  -> vision / audio / video encoder
  -> cross-modal alignment
  -> projector / connector
  -> fusion and multimodal context
  -> multimodal language generation
```

## Notes

- [[architecture/multimodal/vision-language-model|Vision-Language Model]]
- [[architecture/multimodal/clip|CLIP]]
- [[architecture/multimodal/llava|LLaVA]]
- [[architecture/multimodal/qwen-vl|Qwen-VL]]
- [[architecture/multimodal/multimodal-projector|Multimodal Projector]]
- [[architecture/world-models/|World Models]] — 从静态多模态理解进一步进入状态演化与环境预测。
