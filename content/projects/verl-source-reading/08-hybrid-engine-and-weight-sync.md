---
title: 08 Hybrid Engine and Weight Sync
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-02
type: project
status: seed
tags:
  - projects
  - source-reading
  - verl
  - hybrid-engine
  - weight-sync
---

本页研究同一 actor 如何在 training runtime 与 rollout runtime 之间切换，以及更新后的训练权重如何传输到 vLLM/SGLang replicas。

## 本页问题

1. actor training model 与 rollout model 是同一个对象、同一份显存，还是两个运行时副本？
2. sleep、wake、abort、resume 分别释放或保留哪些状态？
3. full weight、sharded weight 与 delta weight 如何导出和传输？
4. checkpoint engine 如何记录并推进 policy version？
5. weight sync 的通信、显存和停顿成本如何进入 trainer mode 选择？

## Source Anchors

- [CheckpointEngineManager](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/checkpoint_engine)
- [ActorRolloutRefWorker.update_weights](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine_workers.py#L727)
- [LLMServerManager](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/llm_server.py)
- [vLLM weight update](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/vllm_rollout)
- [delta weight sync](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/checkpoint_engine/delta_sync)

## Version Ledger

完整笔记需要持续区分：

```text
rollout policy version
old policy version
current actor version
reference policy version
teacher policy version
```

这些版本在 sync 模式中可能彼此接近，但在 async、partial rollout 和 decoupled PPO 中会产生真实差异。

## Completion Criteria

- 给出 sync trainer 一次 update_weights 的时间线；
- 记录参数从 FSDP/Megatron shard 到 rollout layout 的转换；
- 标明 KV cache、CUDA graph、optimizer state 和 model weights 的生命周期；
- 解释 weight sync 失败或滞后如何表现为 off-policy。

