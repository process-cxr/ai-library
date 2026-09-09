---
title: World Model Training
created: 2026-09-09
published: 2026-09-09
modified: 2026-09-09
type: topic
status: seed
area: training
tags:
  - world-model
  - dynamics
  - model-based-rl
  - training
---

World Model Training 研究如何从视频、轨迹、传感器或交互数据中学习环境 dynamics，并把预测能力转化为 planning 或 policy improvement。

## Training Path

```text
observation and action trajectories
  -> representation learning
  -> forward / inverse dynamics
  -> reward and termination prediction
  -> latent rollout or imagination
  -> policy learning and environment evaluation
```

## 核心问题

- offline trajectory、online interaction 和 synthetic rollout 如何组合；
- pixel prediction 与 latent dynamics 的训练取舍；
- action-conditioned prediction 如何区别于无条件 video prediction；
- model error、uncertainty 和 compounding error 如何控制；
- prediction loss 的改善是否能够转化为任务成功率的改善。

## Related Notes

- [[architecture/world-models/|World Models]]
- [[fundamentals/dynamical-systems/|Dynamical Systems]]
- [[fundamentals/control-and-planning/|Control and Planning]]
- [[training/post-training/grpo|GRPO]]
- [[application/world-models/|World Model Applications]]
