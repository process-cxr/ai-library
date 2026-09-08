---
title: 10 Async Trainer and Off-policy
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-02
type: project
status: seed
tags:
  - projects
  - source-reading
  - verl
  - asynchronous-training
  - off-policy
---

本页比较 V1 `sync`、`colocate_async` 与 `separate_async`，重点分析异步执行如何改变 trajectory 的 policy version、训练等待和 off-policy correction，而不只比较吞吐数字。

## 本页问题

1. 三种 trainer mode 的 GPU placement 和执行时间线有何不同？
2. partial rollout 为什么可能让一条 trajectory 跨越多个 model versions？
3. ReplayBufferAsync 如何处理 pending、running、finished、failed 和 stale groups？
4. `drop`、`wait`、refill 与 checkpoint recovery 各自保持什么不变量？
5. decoupled PPO 中 rollout policy、old policy 和 current policy 如何分离？

## Source Anchors

- [sync trainer](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_sync.py)
- [colocate async trainer](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_colocate_async.py)
- [separate async trainer](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_separate_async.py)
- [ReplayBufferAsync](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/replay_buffer.py#L497)
- [rollout correction](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/rollout_corr_helper.py)
- [V1 async guide](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/docs/advance/v1_async_trainer.md)

## Metrics to Track

```text
timing_s/gen
timing_s/update_actor
timing_s/update_weights
trajectory version span
trajectory staleness
rollout/actor log-prob mismatch
sample drop / refill reason
trainer idle time
```

## Completion Criteria

- 为三种 mode 画出资源图和时间线；
- 定义 trajectory-level 与 token-level off-policy 来源；
- 解释 old log-prob recomputation、bypass mode 和 rollout correction 的关系；
- 给出性能收益与训练分布风险并列的评估框架。

