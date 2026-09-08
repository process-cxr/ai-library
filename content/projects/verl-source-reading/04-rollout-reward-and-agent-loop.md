---
title: 04 Rollout, Reward and AgentLoop
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-02
type: project
status: seed
tags:
  - projects
  - source-reading
  - verl
  - rollout
  - reward
  - agent
---

本页连接三类执行对象：rollout backend 负责生成，AgentLoop 负责多轮交互，RewardLoop 负责把结果转换成训练信号。

## 本页问题

1. LLMServerManager 如何创建并路由 rollout replicas？
2. single-turn 与 multi-turn rollout 在数据结构和终止条件上有何不同？
3. rule-based reward、model-based reward 和 remote reward 如何接入统一数据流？
4. reward 是 sequence-level、token-level 还是 trajectory-level，如何写回 `rm_scores`？
5. tool failure、timeout、truncation 和 invalid response 如何影响 sample status 与最终 reward？

## Source Anchors

- [LLMServerManager](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/llm_server.py)
- [rollout base](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/base.py)
- [vLLM rollout](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/vllm_rollout)
- [SGLang rollout](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/sglang_rollout)
- [AgentLoop](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/agent_loop.py)
- [RewardLoop](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/reward_loop/reward_loop.py)
- [reward score functions](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/reward_score)

## Expected Flow

```text
raw prompt
  -> sampling request
  -> assistant tokens
  -> optional tool call / observation loop
  -> terminal response or failure
  -> rule / verifier / reward model
  -> rm_scores and reward metadata
```

## Completion Criteria

- 分别画出 single-turn 和 tool-use trajectory；
- 说明 rollout log-prob 的生成位置与 temperature 处理；
- 记录 reward 写回 TransferQueue 的字段与时机；
- 解释 validation reward 与 training reward 的差异；
- 与 [[training/post-training/reward-model|Reward Model]] 和 [[application/tool-use/tool-calling|Tool Calling]] 建立边界清晰的回链。

