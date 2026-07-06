---
title: Benchmark
created: 2026-05-23
published: 2026-05-23
modified: 2026-06-30
type: topic
status: seed
area: application
tags:
  - evaluation
  - benchmark
---

Benchmark 是用于比较模型、系统或方法能力的标准化任务集合。应用评测中的 benchmark 不只是“题目和分数”，还包括任务分布、输入输出协议、指标、评测环境、数据泄漏控制、人工或自动判分规则，以及分数与真实应用表现之间的外部效度。

## 构造要点

一个可靠的 benchmark 通常需要明确以下问题：

| 维度 | 需要说明的问题 |
|---|---|
| 任务覆盖 | benchmark 覆盖哪些能力、场景和难度层级，哪些能力明确不覆盖 |
| 数据来源 | 任务来自真实用户、公开数据集、合成数据、专家设计还是模型生成 |
| 标注与答案 | 正确答案如何获得，是否存在多解，是否需要人工验证 |
| 指标 | 使用 accuracy、EM、ROUGE、pass@k、reward、judge score 或人工偏好 |
| 环境 | 是否依赖工具、网页、代码执行、Docker、浏览器或外部 API |
| 泄漏控制 | 训练数据、公开题库、参考实现、官方答案和评测脚本是否可能被模型见过 |
| 可复现性 | 评测环境、随机性、版本、prompt、scaffold 和 judge 是否固定 |

## Agentic Potential Benchmark

Agent 系统的端到端评测通常需要 post-trained model、工具调用协议、执行环境和多轮 scaffold。对于 base model 或 pre-training checkpoint，这类评测往往不可直接运行，因为模型尚未具备稳定的指令跟随和工具格式控制能力。

一种折中方法是将成功 agent trajectory 转换为 base model 可以完成的静态评测题。例如，给定任务描述和已有轨迹前缀，让模型选择下一步计划、补全下一条工具调用、判断正确 patch，或选择哪些报告陈述被证据支持。这类 benchmark 评估的是 base model 在后续 agent post-training 前已经具备的 **agentic potential**，而不是完整 agent 执行能力。

这种设计的优势是成本低、可用于训练早期监控，并且比通用静态 benchmark 更接近真实 agent 行为分布。主要风险是它仍然只是代理指标：multiple-choice 或 text completion 不能替代真实 rollout；下一步行动可能存在多条合理路径；负选项质量会影响难度；长上下文处理能力也可能与 agentic decision-making 混在一起。

相关案例见 [[sources/papers/2025-aptbench-benchmarking-agentic-potential-of-base-llms|APTBench: Benchmarking Agentic Potential of Base LLMs During Pre-Training]]。
