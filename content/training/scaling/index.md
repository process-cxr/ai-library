---
title: Scaling
created: 2026-03-28
published: 2026-03-28
modified: 2026-05-31
---

Scaling 模块关注模型参数量、训练 token、计算量、loss 和训练成本之间的关系。它是预训练规划的上游：先理解规模变量如何影响性能，再决定如何分配预算、估算显存、组织数据和设置训练方案。

阅读顺序建议：

1. [[training/scaling/scaling-law|Scaling Law]]：理解 loss 随规模变化的经验规律，以及 Kaplan / Chinchilla 的差异。
2. [[training/scaling/model-data-compute|Model Data and Compute]]：建立 $N$、$D$、$C$、batch tokens 和 wall-clock time 的基本估算。
3. [[training/scaling/compute-optimal|Compute Optimal]]：理解固定 compute 下参数量和训练 token 的分配问题。
4. [[training/scaling/training-budget|Training Budget]]：把理论规模转成 GPU、显存、时间、checkpoint 和评测预算。

## Notes

- [[training/scaling/scaling-law|Scaling Law]]
- [[training/scaling/model-data-compute|模型、数据与算力]]
- [[training/scaling/training-budget|训练预算]]
- [[training/scaling/compute-optimal|Compute Optimal]]

## Related Source TODOs

- [[sources/papers/2020-scaling-laws-for-neural-language-models|Scaling Laws for Neural Language Models]]
- [[sources/papers/2022-training-compute-optimal-large-language-models|Training Compute-Optimal Large Language Models]]
