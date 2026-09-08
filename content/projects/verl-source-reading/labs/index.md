---
title: verl Source Reading Labs
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-02
type: project
status: seed
tags:
  - projects
  - source-reading
  - verl
  - experiments
---

Labs 用于把源码阅读结论转成可复现的观察。每个实验记录环境、source revision、resolved config、执行命令、关键 trace、结果和结论；不把一次运行中的偶然现象直接提升为通用知识。

## 实验列表

- [[projects/verl-source-reading/labs/01-trace-config-and-batch|Lab 01: Trace Config and Batch]]：跟踪 resolved config 和 batch field lifecycle。
- [[projects/verl-source-reading/labs/02-custom-reward|Lab 02: Custom Reward]]：实现并验证一个 function-based reward。
- [[projects/verl-source-reading/labs/03-compare-grpo-ppo|Lab 03: Compare GRPO and PPO]]：在控制变量下比较两条 codepath。
- [[projects/verl-source-reading/labs/04-multiturn-tool-agent|Lab 04: Multi-turn Tool Agent]]：验证多轮 tool trajectory、mask 和 reward。

## 实验记录规范

```markdown
## Objective
## Environment
## Source Revision
## Resolved Config
## Command
## Instrumentation
## Observations
## Failure Analysis
## Conclusion
## Follow-up
```

大型训练前先完成 CPU unit test、config resolution 和小 batch trace。实验重点是验证字段和控制流，不追求 benchmark 分数。

