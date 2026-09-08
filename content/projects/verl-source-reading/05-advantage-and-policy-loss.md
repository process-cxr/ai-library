---
title: 05 Advantage and Policy Loss
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-02
type: project
status: seed
tags:
  - projects
  - source-reading
  - verl
  - advantage
  - policy-loss
---

本页把算法公式映射到 verl 的 tensor 字段、mask、normalization 和 loss registry。第一轮只覆盖 GAE、GRPO outcome advantage、vanilla PPO loss、value loss 与 KL；其他 loss variant 作为扩展对照。

## 本页问题

1. `rm_scores` 如何变成 `token_level_rewards`？
2. GAE 与 GRPO 分别依赖哪些字段，在哪个维度做 baseline 和 normalization？
3. `old_log_probs`、current `log_probs` 与 `advantages` 如何构成 policy ratio 和 clipped objective？
4. `loss_agg_mode` 如何改变 token、sequence 和 batch 的权重？
5. KL 放在 reward 中和放在 actor loss 中有何实现差异？

## Source Anchors

- [advantage dispatch](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/ray_trainer.py#L187)
- [core_algos.py](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/core_algos.py)
- [policy loss registry](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/core_algos.py#L50)
- [vanilla policy loss](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/core_algos.py#L1285)
- [ppo_loss](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/utils/losses.py#L57)
- [value_loss](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/utils/losses.py#L147)

## Algorithm-to-Field Mapping

```text
reward / score       -> rm_scores / token_level_rewards
behavior policy      -> old_log_probs
current policy       -> model_output.log_probs
reference policy     -> ref_log_prob
credit signal        -> advantages
critic target        -> returns
valid action tokens  -> response_mask
```

## Completion Criteria

- 对每个公式给出 tensor shape 与 reduction dimension；
- 用 CPU test 验证正负 advantage 下 clipping 行为；
- 比较至少三种 `loss_agg_mode` 对长短 response 权重的影响；
- 明确 GRPO 方差来源与 group normalization 的适用边界；
- 将通用结论回填 [[training/post-training/ppo|PPO]] 与 [[training/post-training/grpo|GRPO]]。

