---
title: World Model Inference
created: 2026-09-09
published: 2026-09-09
modified: 2026-09-09
type: topic
status: seed
area: inference
tags:
  - world-model
  - inference
  - planning
---

World Model Inference 关注模型如何在执行 action 前进行 rollout、simulation 和 imagination，并把预测结果交给 planner 或 policy。它的主要瓶颈通常不是一次 forward，而是多步预测中的 latency、memory 和误差累积。

## Inference Path

```text
current observation
  -> latent state
  -> candidate action rollout
  -> predicted future states / rewards
  -> planner selects action
  -> real environment feedback
```

## 核心问题

- rollout horizon 如何影响 planning quality 和计算成本；
- imagined trajectory 与真实 environment trajectory 如何区分；
- model uncertainty 如何传递到 action selection；
- 如何缓存 state、reuse prefix 并并行执行 candidate rollouts；
- 预测性能、规划性能和最终任务成功率是否一致。

## Related Notes

- [[architecture/world-models/|World Models]]
- [[fundamentals/control-and-planning/|Control and Planning]]
- [[application/agents/planning|Planning]]
- [[application/world-models/|World Model Applications]]
