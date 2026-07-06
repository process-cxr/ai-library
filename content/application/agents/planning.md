---
title: Planning
created: 2026-05-17
published: 2026-05-17
modified: 2026-06-30
type: topic
status: seed
area: application
tags:
  - agent
  - planning
---

Planning 是 agent 在行动前或行动过程中选择任务路线、分解目标、安排工具调用并根据反馈调整策略的能力。它不只是生成一个静态 checklist，而是在当前状态、历史 observation、可用工具和目标约束之间做连续决策。

## 基本形态

Agent planning 常见形态包括：

| 形态 | 说明 |
|---|---|
| Task decomposition | 将用户目标拆成子任务或阶段 |
| Stepwise planning | 在每一步根据当前 observation 决定下一步 |
| Dynamic replanning | 工具失败、检索不充分、测试失败后修改路线 |
| Look-ahead planning | 在执行前预测某条路线可能带来的后续状态 |
| Confidence estimation | 估计当前计划成功概率或不确定性 |

## World Model Planning

World model planning 指 agent 在执行 action 前预测可能发生的后续状态或结果。传统 model-based RL 通常显式学习 transition model 和 reward model；LLM agent 中更常见的做法是把未来状态压缩成语义摘要，例如未来路径、信息缺口、关键里程碑和成功概率估计。

这种机制可以帮助 agent 在行动前比较路线、提前发现信息缺口，并避免盲目执行低成功率计划。但它也容易产生 hallucinated foresight：模型生成看似合理的未来模拟，实际并不被环境反馈支持。因此，look-ahead planning 通常需要 grounding、真实执行反馈或 confidence calibration。相关案例见 [[sources/papers/2026-internalizing-the-future-world-model-agentic-training|Internalizing the Future]]。

## 相关概念

- [[application/agents/agent|Agent]]
- [[application/tool-use/tool-calling|Tool Calling]]
- [[application/agents/memory|Memory]]
- [[training/mid-training/continued-pretraining|Continued Pretraining]]
