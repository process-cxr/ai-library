---
title: Lab 02 - Custom Reward
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-02
type: project
status: seed
tags:
  - projects
  - verl
  - experiment
  - reward
---

## Objective

实现一个最小 function-based reward，验证 data source routing、response decoding、score shape、extra metadata、failure handling 和 validation behavior。

## Planned Cases

1. 正确答案返回正 reward；
2. 格式错误与答案错误得到可区分结果；
3. 超时、异常和空 response 有明确 fallback；
4. batch 内不同长度 response 的 score 正确写回；
5. reward 不依赖本不应可见的 future information。

## Source Anchors

- [reward preparation guide](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/docs/preparation/reward_function.rst)
- [custom reward example](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/rewards/custome_reward_fn)
- [reward score implementations](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/reward_score)
- [RewardLoop](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/reward_loop/reward_loop.py)

## Acceptance Criteria

- reward function 有单元测试；
- score 与 extra info 可在 rollout dump 中审计；
- 异常不会静默变成高 reward；
- 结果回填 [[projects/verl-source-reading/04-rollout-reward-and-agent-loop|04 Rollout, Reward and AgentLoop]]。

