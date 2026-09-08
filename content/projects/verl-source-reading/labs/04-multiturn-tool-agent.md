---
title: Lab 04 - Multi-turn Tool Agent
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-02
type: project
status: seed
tags:
  - projects
  - verl
  - experiment
  - agent
  - tool-use
---

## Objective

运行一个最小 multi-turn tool Agent，导出完整 trajectory，并验证 assistant action、tool observation、mask、reward 和 termination 的边界。

## Planned Trace

```text
user query
  -> assistant tool call
  -> parsed arguments
  -> tool execution
  -> tool result
  -> assistant continuation
  -> final answer
  -> terminal reward
```

## Required Artifacts

- raw messages and chat template output；
- token ids grouped by role and turn；
- `attention_mask`、`response_mask` 或等价 loss mask；
- tool latency、status 和 truncated response；
- trajectory reward and failure metadata；
- rollout/model version identifiers。

## Source Anchors

- [AgentLoop](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/agent_loop.py)
- [AgentLoop tutorial](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/tutorial/agent_loop_get_started)
- [multi-turn config](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/config/rollout/rollout.yaml#L176)

## Acceptance Criteria

- observation token 不产生 policy gradient；
- tool failure 和 normal termination 可区分；
- turn-by-turn tokenization 与完整 conversation tokenization 的差异可观测；
- 结果回填 [[projects/verl-source-reading/11-agentic-rl-extension|11 Agentic RL Extension]]。

