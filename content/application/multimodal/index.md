---
title: Multimodal Applications
created: 2026-09-09
published: 2026-09-09
modified: 2026-10-08
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
  -> text / image / audio / video answer or transformation
  -> tool use and environment interaction
```

应用层需要把“模型能处理什么”拆成输入和输出两件事。一个视觉问答系统可能接收 image + text、只输出 text；一个图像编辑系统接收 image + text、输出 image；一个实时语音系统接收 audio + text、输出 audio + text；一个 VLA 系统接收 image + language、输出 action。系统支持多模态输入输出，也不代表所有模态由同一个模型原生生成。

## 任务范围

- visual question answering 和 image understanding；
- OCR、document understanding、chart / table reasoning；
- visual grounding、spatial reasoning 和 temporal reasoning；
- audio understanding、speech interaction 和 video understanding；
- multimodal RAG、tool use 和 GUI interaction。

## 评测关注

不能只看生成文本是否流畅，还需要按输入输出任务区分 perception accuracy、grounding accuracy、generation quality、reasoning correctness、hallucination、latency 和真实任务成功率。对于图像、音频和视频输出，还要单独检查内容一致性、时间或空间一致性、可控性和编辑保真度；对于 action 输出，则要检查环境中的执行结果，而不是只看动作格式是否正确。

## Related Notes

- [[architecture/multimodal/|Multimodal]]
- [[training/multimodal/|Multimodal Training]]
- [[application/evaluation/|Evaluation]]
- [[application/embodied-agents/|Embodied Agents]]
