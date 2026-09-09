---
title: World Models
created: 2026-09-09
published: 2026-09-09
modified: 2026-09-09
type: topic
status: seed
area: architecture
tags:
  - world-model
  - dynamics
  - planning
  - architecture
---

World Model 通过学习环境状态、状态转移和 action 结果，为 agent 提供可用于预测与规划的内部环境模型。它与 Multimodal Model 有连续关系，但不能简单等同于“能够看图或生成视频的模型”。

## Architecture Map

```text
observation encoder
  -> latent state
  -> dynamics model
  -> predicted future state
  -> reward / termination model
  -> planner or policy
```

## 核心问题

- observation 如何压缩成保留控制信息的 latent state；
- dynamics model 是否显式条件化于 action；
- 模型预测的是 pixel、feature、latent state 还是 reward；
- prediction model、simulator 和 planner 的边界如何划分；
- 长时域 rollout 中的误差如何累积。

## Reading Path

1. [[fundamentals/dynamical-systems/|Dynamical Systems]]：理解 state、observation 和 transition。
2. [[fundamentals/control-and-planning/|Control and Planning]]：理解预测如何服务于 action selection。
3. [[architecture/multimodal/|Multimodal]]：理解视觉、视频和其他 observation encoder。
4. [[training/world-models/|World Model Training]]：理解 dynamics、reward 和 prediction objective。
5. [[application/world-models/|World Model Applications]]：观察 simulation、planning 和 embodied task 中的实际作用。

## Related Notes

- [[application/agents/planning|Planning]]
- [[inference/world-models/|World Model Inference]]
- [[sources/papers/2026-internalizing-the-future-world-model-agentic-training|Internalizing the Future]]
