---
title: Dynamical Systems
created: 2026-09-09
published: 2026-09-09
modified: 2026-09-09
type: topic
status: seed
area: fundamentals
tags:
  - dynamical-systems
  - world-models
  - fundamentals
---

Dynamical Systems 关注状态如何随时间变化。它为 World Model 提供最基本的语言：模型不只预测当前输入的下一个 token，还要描述 action 作用后环境可能进入的下一个 state。

## 核心记号

$$
s_{t+1}\sim T(s_{t+1}\mid s_t,a_t)
$$

- $s_t$：环境在时刻 $t$ 的 state；
- $a_t$：模型或 agent 在时刻 $t$ 采取的 action；
- $s_{t+1}$：action 之后的下一个 state；
- $T$：描述状态转移的 transition dynamics。

现实任务中，模型通常只能看到 observation $o_t$，而不能直接访问完整的 state。于是还需要研究 observation model、partial observability、memory 和 state estimation。

## Reading Path

```text
state
  -> observation
  -> transition
  -> action-conditioned future
  -> planning and control
```

## Related Notes

- [[fundamentals/probability/conditional-probability|Conditional Probability]]
- [[fundamentals/control-and-planning/|Control and Planning]]
- [[architecture/world-models/|World Models]]
- [[application/agents/agent|Agent]]
