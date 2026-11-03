---
title: Workflow Agent
created: 2026-05-23
published: 2026-05-23
modified: 2026-11-03
type: topic
status: growing
area: application
tags:
  - agent
  - workflow
---

Workflow Agent 指由预定义流程、状态机或图编排约束行动顺序的 agent 系统。它通常不会让模型完全自由地决定下一步，而是把任务拆成若干阶段，每个阶段只暴露有限工具、输入和输出格式。

与更开放的 [[application/agents/agent|Agent]] 相比，workflow agent 的优势是可控性、可观测性和可验证性更强。缺点是流程刚性更高，遇到需要反复探索、跨阶段回退或非典型路径的问题时，灵活性不如端到端多轮 agent。

## 固定流程与动态交互

| 维度 | Workflow Agent | Open-ended Agent |
|---|---|---|
| 行动空间 | 按阶段受限 | 动态选择工具和步骤 |
| 评测 | 每个阶段更容易局部验证 | 通常依赖最终任务结果 |
| 稳定性 | 高，适合生产流程 | 高度依赖模型和 scaffold |
| 灵活性 | 受流程限制 | 更适合未知路径探索 |
| 训练难度 | 可拆成单步任务 | 长程 credit assignment 更难 |

在 software engineering 场景中，Agentless 方法是典型 workflow：先定位文件，再生成 patch，再生成测试或用测试筛选 patch。SWE-Agent 则允许模型在仓库中多轮搜索、读取、编辑、运行测试和反思。

## 以产物和验证组织长流程

预定义阶段不要求阶段内部的每一步都固定。一个工作流可以约束“先验证环境，再生成任务，再运行训练”，同时允许阶段内部检索代码、尝试不同实现和反复修复。

长流程需要显式管理三类信息：

- **持久产物**：代码、配置、指标、视频与修改记录，保存当前状态和重现结果所需的细节。
- **验证条件**：决定阶段能否继续的可执行检查，并在失败时返回检查项、实际观测和错误信息。
- **决策摘要**：记录采用或放弃某个方向的原因，供主控和后续阶段使用。

[[sources/papers/2026-harbor-harness-agentic-robot-reinforcement-learning|HARBOR]] 将这一结构用于机器人强化学习工程：代理生成环境、奖励和训练配置，执行阶段检查，再保留产物并进入后续流程。候选试验在隔离目录与上下文中运行，主控基于摘要选择下一轮方向。

这些设计分别解决状态丢失、错误传播、试验相互覆盖和上下文反复读取的问题。验证仍有覆盖边界：程序退出成功不代表视频有效，视频存在不代表任务完成，任务在仿真中完成也不代表能够真实部署。

流程中的“经验学习”也要按实际机制理解。将失败诊断和有效配置写入可检索记忆，是外部知识积累；只有确实用这些记录更新模型参数，才构成模型训练。两者可以组合，但不能仅凭经验库增长就推断模型权重发生了学习。

## 训练价值

Workflow 不只是一种部署形态，也可以是一种训练数据组织方式。固定流程把长程任务拆成可验证子技能，例如 localization、code edit、test writing 和 self-reflection，使每个子技能更容易构造 verifier、SFT 样本或 RL reward。

因此，workflow training 可以作为 open-ended agent 的 skill prior。[[sources/papers/2025-kimi-dev-agentless-training-as-skill-prior-for-swe-agents|Kimi-Dev]] 表明，Agentless workflow 训练得到的 BugFixer、TestWriter 和 reflection 能力，可以通过少量 SWE-Agent trajectories 迁移到多轮 agent 框架。

## 相关概念

- [[application/agents/agent|Agent]]
- [[application/agents/planning|Planning]]
- [[application/tool-use/tool-calling|Tool Calling]]
- [[training/mid-training/capability-injection|Capability Injection]]
