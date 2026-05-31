---
title: Mid-training
created: 2026-02-22
published: 2026-02-22
modified: 2026-05-31
---

中训练阶段介于预训练和后训练之间，通常用于继续预训练、领域能力注入、长上下文扩展和训练末期退火。

建议阅读顺序：

1. [[training/mid-training/continued-pretraining|Continued Pretraining]]：理解在已有 checkpoint 上继续训练的基本框架。
2. [[training/mid-training/domain-adaptation|Domain Adaptation]]：理解领域数据注入、replay 和遗忘风险。
3. [[training/mid-training/capability-injection|Capability Injection]]：理解数学、代码、工具、多语言等能力塑形。
4. [[training/mid-training/long-context-training|Long Context Training]]：理解位置编码扩展、长文档数据和长上下文评测。
5. [[training/mid-training/annealing|Annealing]]：理解训练末期高质量数据和 learning rate 收敛整理。

## Notes

- [[training/mid-training/continued-pretraining|Continued Pretraining]]
- [[training/mid-training/domain-adaptation|领域适配]]
- [[training/mid-training/long-context-training|长上下文训练]]
- [[training/mid-training/annealing|Annealing]]
- [[training/mid-training/capability-injection|能力注入]]

## Related Source TODOs

- [[sources/papers/2023-position-interpolation|Position Interpolation]]
- [[sources/papers/2023-yarn|YaRN]]
- [[sources/papers/2023-longlora|LongLoRA]]
- [[sources/papers/2024-longrope|LongRoPE]]
