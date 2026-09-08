---
title: 09 PPO and GRPO Codepath
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-02
type: project
status: seed
tags:
  - projects
  - source-reading
  - verl
  - ppo
  - grpo
---

本页在同一 V1 trainer 中对比 PPO 与 GRPO。比较对象不是算法名，而是实际启用的 role、batch field、forward pass、loss 和 resource cost。

## Shared Path

```text
prompt
  -> rollout
  -> reward
  -> old/reference log-prob
  -> advantage
  -> actor update
```

## Expected Differences

```text
PPO:
  critic role
  -> values
  -> GAE / returns
  -> critic update
  -> actor update

GRPO:
  repeated responses per prompt
  -> group reward statistics
  -> group-relative advantages
  -> actor update without critic
```

## Source Anchors

- [need_critic / need_reference_policy](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/utils.py)
- [_step_once branches](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L540)
- [advantage estimators](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/core_algos.py)
- [critic update](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1711)
- [actor update](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1734)

## Comparison Dimensions

- required models and GPU memory；
- rollout count and prompt grouping；
- advantage bias/variance；
- reward scale sensitivity；
- sequence/token normalization；
- compute and communication cost；
- metrics required for debugging；
- suitability for long-horizon Agent trajectories。

## Completion Criteria

- 使用同一模型和数据构造两份最小 resolved config；
- 生成 role/field/call/metric 四张差异表；
- 运行 [[projects/verl-source-reading/labs/03-compare-grpo-ppo|Lab 03]]；
- 将算法级结论回填 [[training/post-training/ppo|PPO]] 和 [[training/post-training/grpo|GRPO]]。

