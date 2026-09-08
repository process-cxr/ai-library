---
title: 11 Agentic RL Extension
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-02
type: project
status: seed
tags:
  - projects
  - source-reading
  - verl
  - agentic-rl
  - tool-use
---

本页研究 verl 如何从 single-turn response RL 扩展到 multi-turn Agent trajectory。重点不是某一种工具协议，而是 state-action-observation 边界如何进入 rollout、mask、reward 与 credit assignment。

## 本页问题

1. AgentLoop 如何保存 conversation state、request id 与 turn limit？
2. tool call、tool result 和 assistant generation 如何拼接并 tokenization？
3. 哪些 token 进入 `response_mask` 和 policy loss？
4. trajectory-level reward 如何分配到多个 assistant turns？
5. tool timeout、invalid arguments、environment failure 和 truncation 如何编码？
6. 同一个环境任务的多条 rollout 如何形成 group-relative advantage？

## Source Anchors

- [AgentLoop implementation](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/agent_loop.py)
- [V1 AgentLoop and TransferQueue](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py)
- [multi-turn rollout config](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/config/rollout/rollout.yaml#L176)
- [agent examples](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/tutorial/agent_loop_get_started)
- [tool data preprocessing](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/data_preprocess)

## Trajectory Ledger

```text
system / user context       conditioning only
assistant reasoning/action  policy output
tool call arguments         policy output
tool result                 environment observation
next assistant turn         policy output conditioned on observation
terminal state              environment / task outcome
```

实际 mask 语义必须以源码和实验验证为准，不能仅依据 role name 推断。

## Completion Criteria

- 跑通一个最小 multi-turn tool example；
- 导出完整 token ids、roles、turns、masks、reward 和 trajectory status；
- 验证 observation token 不产生 policy gradient；
- 记录长轨迹截断、跨 turn credit 和 environment nondeterminism 的风险；
- 与 [[application/tool-use/tool-calling|Tool Calling]] 建立通用机制回链。

