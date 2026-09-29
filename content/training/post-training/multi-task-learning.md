---
title: Multi-Task Learning
created: 2026-09-23
published: 2026-09-23
modified: 2026-09-23
type: topic
status: seed
area: training
tags:
  - post-training
  - multi-task-learning
  - capability-integration
---

Multi-Task Learning（MTL，多任务学习）让一个共享模型同时学习多个任务。核心问题是共享可迁移表示，同时控制不同任务之间的 gradient interference、数据不平衡和能力遗忘。

## 主要路线

- **Joint training**：混合多个 task datasets，直接优化共享模型。
- **Sequential training**：依次训练不同任务，需要控制 catastrophic forgetting。
- **Multi-teacher distillation**：先训练 specialist teachers，再把能力蒸馏到统一 student。
- **Model merging**：分别训练 task-specific models，再在 parameter space 合并。

不同路线分别把冲突放在 gradient、training order、teacher policy 或 task-vector geometry 中处理。

## 与 Model Merging 的关系

Model merging 可以看作 MTL 的一种参数整合路线：它不重新集中所有 task data，而是组合已经训练好的 task models。[[sources/papers/2026-learn-to-merge-meta-learning-adaptive-multi-task-model-merging|MetaMerging]] 进一步将后续 task-specific adapter training 纳入 merging coefficient 的优化目标。

## 相关知识

- [[training/post-training/model-merging|Model Merging]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/on-policy-kd|On-policy KD]]
- [[sources/papers/2025-distac-conditioning-task-vectors-via-distillation-for-robust-model-merging|DisTaC]]
