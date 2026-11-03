---
title: Embodied Agents
created: 2026-09-09
published: 2026-09-09
modified: 2026-11-03
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

## 工程代理与执行策略

具身系统中，负责组织训练的语言模型代理和实际控制机器人的策略可以是不同组件。工程代理阅读代码、构建环境、调整奖励并分析训练结果；执行策略根据观测输出动作，由强化学习或模仿学习更新参数。部署时还可能需要独立的感知、状态估计和低层控制器。

[[sources/papers/2026-harbor-harness-agentic-robot-reinforcement-learning|HARBOR]] 属于前一种自动化：让语言模型代理组织机器人强化学习工作流。它训练出的机器人策略与负责工程工作的语言模型分属不同层次，不能将系统自动化等同于语言模型直接控制机器人。

## 仿真到真实的验证层次

Sim-to-real 指把仿真中获得的策略迁移到真实环境。不同阶段回答的问题不同：

1. **接口是否正确**：观测顺序、动作单位、控制频率、重置和终止条件是否符合约定。
2. **行为是否满足目标**：除训练回报外，检查独立成功标准和行为轨迹，避免只学会获取高奖励。
3. **策略是否稳健**：在质量、摩擦、初始状态和观测噪声等变化下是否仍能完成任务。
4. **真实系统是否成立**：部署感知、延迟、控制接口和真实接触条件是否与训练假设相容。

域随机化通过覆盖一组环境参数提高稳健性，系统辨识则利用真实观测约束仿真参数；二者作用不同，也都不能替代真实任务评测。仿真成功率与真实成功率需要分别报告，真实轨迹、人工校准和反馈也属于迁移条件。

## Related Notes

- [[application/agents/|Agents]]
- [[application/agents/planning|Planning]]
- [[architecture/world-models/|World Models]]
- [[application/world-models/|World Model Applications]]
