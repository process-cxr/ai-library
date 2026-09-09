---
title: Control and Planning
created: 2026-09-09
published: 2026-09-09
modified: 2026-09-09
type: topic
status: seed
area: fundamentals
tags:
  - control
  - planning
  - world-models
  - fundamentals
---

Control and Planning 研究如何根据目标、当前状态和未来结果选择 action。它是连接 Agent、World Model 和 embodied intelligence 的基础模块。

## 核心问题

- policy、value、reward 和 planner 分别承担什么作用；
- model-free 与 model-based decision 有什么差异；
- 如何使用 learned dynamics 做 imagination 或 look-ahead planning；
- planning horizon、uncertainty 和 model error 如何影响最终 action；
- 为什么短期预测准确不一定能带来长期任务成功。

## Reading Path

```text
goal and current state
  -> candidate actions
  -> predicted future states
  -> value / reward comparison
  -> action execution
  -> feedback and replanning
```

## Related Notes

- [[application/agents/planning|Planning]]
- [[architecture/world-models/|World Models]]
- [[training/post-training/grpo|GRPO]]
