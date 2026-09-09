---
title: Embodied Agents
created: 2026-09-09
published: 2026-09-09
modified: 2026-09-09
type: topic
status: seed
area: application
tags:
  - embodied-agent
  - multimodal
  - world-model
  - application
---

Embodied Agents 研究能够感知环境并采取动作的 agent。它可以运行在机器人、游戏、GUI、模拟器或其他具有连续反馈的环境中，是从一般 Multimodal Agent 进入 World Model 和 physical interaction 的过渡模块。

## Interaction Loop

```text
observation
  -> state estimation and memory
  -> planning / policy
  -> action
  -> environment feedback
  -> next observation
```

## 核心问题

- observation 与真实 state 之间如何建立对应；
- language instruction 如何转换为可执行 action；
- agent 如何处理 partial observability、long horizon 和 failure recovery；
- World Model 如何辅助 imagination、planning 和 policy learning；
- 模拟环境中的能力能否迁移到真实环境。

## Related Notes

- [[application/agents/|Agents]]
- [[application/agents/planning|Planning]]
- [[architecture/world-models/|World Models]]
- [[application/world-models/|World Model Applications]]
