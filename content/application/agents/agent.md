---
title: Agent
created: 2026-05-16
published: 2026-05-16
modified: 2026-06-30
type: topic
status: growing
area: application
tags:
  - application
  - agent
  - reasoning
---

Agent 指能够围绕目标自主组织行动的模型系统。与单轮问答不同，agent 通常需要维护任务状态、制定计划、调用工具、读取环境反馈，并根据反馈调整后续行动。典型形式是 action-observation loop：

```text
observe state → reason/plan → act with tools → receive observation → revise plan
```

在 LLM 应用中，agent 不一定是单个模型。它通常由模型、prompt/scaffold、工具接口、记忆、检索、执行环境、评测器和安全约束共同组成。模型负责决策和语言生成，scaffold 负责组织可见上下文、工具格式和执行循环。

## 核心能力

| 能力 | 说明 |
|---|---|
| Planning | 将目标拆解为可执行步骤，并在失败后重规划 |
| Tool Use | 选择工具、生成参数、解析结果 |
| Memory / State | 保留任务历史、环境状态和中间结论 |
| Reflection | 根据错误、测试结果或外部反馈修正行动 |
| Termination | 判断任务何时完成并提交最终结果 |

对 code agent 而言，常见动作包括搜索文件、读取代码、编辑文件、运行测试、分析错误日志和提交 patch。其能力不只是代码生成，而是跨多轮环境交互完成软件工程任务。

## Agent 与 Workflow

Agentic framework 强调多轮自主决策，workflow framework 则将任务拆成固定或半固定步骤。二者并非互斥：workflow 可以作为部署时的稳定编排，也可以作为训练时的结构化技能注入方式。对于软件工程任务，localization、code edit、test writing 和 verification 等 workflow 子任务可以先被单独训练，再迁移到更自由的 SWE-Agent 框架中。相关案例见 [[sources/papers/2025-kimi-dev-agentless-training-as-skill-prior-for-swe-agents|Kimi-Dev]]。

## 相关概念

- [[application/agents/workflow-agent|Workflow Agent]]
- [[application/agents/planning|Planning]]
- [[application/tool-use/tool-calling|Tool Calling]]
- [[application/evaluation/benchmark|Benchmark]]
