---
title: Multimodal Applications
created: 2026-09-09
published: 2026-09-09
modified: 2026-09-09
type: topic
status: seed
area: application
tags:
  - multimodal
  - evaluation
  - application
---

Multimodal Applications 关注模型如何在真实任务中使用图像、音频、视频和文本信息。知识组织从单模态感知开始，逐步进入 grounding、reasoning、工具调用和多轮环境交互。

## Task Path

```text
perception
  -> recognition and extraction
  -> grounding and spatial / temporal reasoning
  -> multimodal answer or transformation
  -> tool use and environment interaction
```

## 任务范围

- visual question answering 和 image understanding；
- OCR、document understanding、chart / table reasoning；
- visual grounding、spatial reasoning 和 temporal reasoning；
- audio understanding、speech interaction 和 video understanding；
- multimodal RAG、tool use 和 GUI interaction。

## 评测关注

不能只看生成文本是否流畅，还需要区分 perception accuracy、grounding accuracy、reasoning correctness、hallucination、latency 和真实任务成功率。

## Related Notes

- [[architecture/multimodal/|Multimodal]]
- [[training/multimodal/|Multimodal Training]]
- [[application/evaluation/|Evaluation]]
- [[application/embodied-agents/|Embodied Agents]]
