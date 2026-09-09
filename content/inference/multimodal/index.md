---
title: Multimodal Inference
created: 2026-09-09
published: 2026-09-09
modified: 2026-09-09
type: topic
status: seed
area: inference
tags:
  - multimodal
  - inference
  - serving
---

Multimodal Inference 关注图像、音频和视频进入模型后的推理成本、上下文组织与 serving 行为。它不是把文本推理简单加上一个 encoder，而是要处理不同模态的 token 数量、时序和缓存方式。

## 核心问题

- visual token 与 text token 如何共同进入 context window；
- image / video token compression 如何影响质量和 latency；
- multimodal KV Cache 如何组织，哪些 feature 可以复用；
- streaming audio / video 如何与 autoregressive decoding 协同；
- multimodal serving 的 batching、memory 和 failure mode 有什么不同。

## Related Notes

- [[architecture/multimodal/|Multimodal]]
- [[inference/kv-cache-and-memory/|KV Cache and Memory]]
- [[inference/performance/|Performance]]
- [[inference/serving-systems/|Serving Systems]]
