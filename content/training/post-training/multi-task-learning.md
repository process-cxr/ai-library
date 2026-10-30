---
title: Multi-Task Learning
created: 2026-09-23
published: 2026-09-23
modified: 2026-10-29
type: topic
status: growing
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

## 梯度冲突：比较对象决定它表示什么

对共享参数，设两个 loss 的梯度为 $g_a,g_b$。若采用普通梯度步 $-\eta g_a$，第二个 loss 的一阶变化为：

$$
\Delta L_b\approx-\eta\langle g_a,g_b\rangle.
$$

负内积表示该候选方向会局部增加 $L_b$，但必须先明确两个梯度来自什么数据与监督：

- **独立任务比较**：任务 A 在自己的数据上得到 $g_A$，任务 B 在自己的数据上得到 $g_B$，用于检查两个任务更新方向是否相容。
- **同轨迹跨教师比较**：对同一批任务 A 的学生回答，分别使用 A 教师与 B 教师构造监督，得到两个方向，用于检查当前行为上的修正是否相容。

这两种比较对应不同的局部目标。独立任务梯度平均对齐，并不排除同轨迹上的具体修正存在冲突；同轨迹辅助 loss 上升，也不直接等价于任务 B 的真实评测能力下降。

[[sources/papers/2026-roadmopd-cross-teacher-guarding|ROADMOPD]] 使用第二种比较，并让辅助梯度只定义投影约束，不将其直接作为额外任务更新合入共享梯度。

## 冲突投影与主任务信号保留

当目标梯度 $g$ 与辅助方向 $g'$ 冲突时，全投影可以写成：

$$
g_{\mathrm{full}}=g-
\frac{\langle g,g'\rangle}{\|g'\|_2^2}g'.
$$

它去掉沿辅助方向的冲突分量。但两个方向几乎反向时，全投影也可能几乎擦除整个主梯度，因此减少冲突与保留学习信号需要权衡。

一种做法是在原梯度和全投影之间选择部分投影，约束 $\|\widetilde g\|\ge m\|g\|$，在允许的范数削减范围内尽量减少冲突。这里 $m$ 是梯度范数比例，不是参数保留率或能力保留率；部分投影仍可留下负内积。

若连续对多个辅助方向投影，需要约束相对于**最初梯度**的统一预算。每一步只保留上一步的 $m$，两步后可能只剩 $m^2$；后一次投影也可能破坏之前的对齐，顺序处理不能自动证明同时消除了所有冲突。

## 分布式训练中的度量范围

参数 tensor 被分片到不同设备后，一个完整 tensor 的梯度内积与范数必须聚合相应 shards，再决定共同的投影系数。独立按本地 shard 计算系数，一般不等价于完整 tensor 的投影。

还需要分清约束成立在哪一层：逐 tensor 投影、领域梯度聚合、global norm clipping、优化器动量与预条件、weight decay。投影阶段的范数下界不保证最终共享更新有同样的范数，也不保证实际 optimizer step 后所有任务都不退化。

## 多任务结果的评估口径

单任务参照应说明训练数据、预算和 checkpoint；把不同单任务模型各自的域内分数放在一行，不代表一个统一模型达到了全部分数。

联合结果同时保留逐任务和聚合分数，说明聚合按领域、benchmark 还是问题加权。总体平均提高可能来自弱域的大幅恢复，也可能伴随另一领域的小幅回退，不能仅凭平均分声称逐任务无损。

比较不同梯度协调算法时，还需固定或记录 rollout budget、optimizer updates、领域采样、loss reduction 与额外 backward 成本；相同生成样本数不等于相同训练算力。

## 与 Model Merging 的关系

Model merging 可以看作 MTL 的一种参数整合路线：它不重新集中所有 task data，而是组合已经训练好的 task models。[[sources/papers/2026-learn-to-merge-meta-learning-adaptive-multi-task-model-merging|MetaMerging]] 进一步将后续 task-specific adapter training 纳入 merging coefficient 的优化目标。

## 相关知识

- [[training/post-training/model-merging|Model Merging]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/on-policy-kd|On-policy KD]]
- [[training/distributed-training/fsdp|FSDP]]
- [[sources/papers/2026-roadmopd-cross-teacher-guarding|ROADMOPD: Cross-Teacher Guarding for MOPD]]
- [[sources/papers/2025-distac-conditioning-task-vectors-via-distillation-for-robust-model-merging|DisTaC]]
