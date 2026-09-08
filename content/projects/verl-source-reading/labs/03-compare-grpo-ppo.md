---
title: Lab 03 - Compare GRPO and PPO
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-02
type: project
status: seed
tags:
  - projects
  - verl
  - experiment
  - ppo
  - grpo
---

## Objective

在相同 base model、prompt data、rollout backend、reward 和 sequence limits 下，比较 PPO 与 GRPO 的 runtime roles、batch fields、显存、step timing 和 metric behavior。

## Controlled Variables

- model checkpoint；
- prompt dataset and split；
- rollout temperature and response length；
- reward implementation；
- actor optimizer and learning rate；
- hardware and training backend。

算法本身要求不同的 rollout count、critic 和 advantage estimator，这些差异作为实验变量明确记录。

## Measurements

```text
enabled roles
peak memory by role
rollout / reward / advantage / update timing
advantage mean and std
response length
reward distribution
policy KL and clip fraction
critic loss and explained variance, if present
```

## Acceptance Criteria

- 两份 resolved config 可直接 diff；
- 调用链差异与测量结果一致；
- 不用单次 reward 高低直接判断算法优劣；
- 结果回填 [[projects/verl-source-reading/09-ppo-and-grpo-codepath|09 PPO and GRPO Codepath]]。

