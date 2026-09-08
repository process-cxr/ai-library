---
title: 07 Model Engine and Backend
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-02
type: project
status: seed
tags:
  - projects
  - source-reading
  - verl
  - fsdp
  - megatron
  - model-engine
---

本页研究统一 worker contract 如何落到不同 training backend。第一遍以 FSDP 为主，第二遍再映射 Megatron，不同时展开所有 engine implementation。

## 本页问题

1. `TrainingWorker` 与 `ActorRolloutRefWorker` 对 trainer 暴露哪些稳定接口？
2. BaseEngine 如何统一 initialize、forward/backward、optimizer、checkpoint 和 parameter export？
3. FSDP 与 Megatron 在 batch dispatch、loss scaling、parallel group 和 checkpoint 上有哪些不可隐藏差异？
4. actor、critic、reference model 是否复用相同 engine abstraction？
5. backend-specific config 如何进入 engine registry？

## Source Anchors

- [engine_workers.py](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine_workers.py)
- [BaseEngine](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/base.py#L30)
- [EngineRegistry](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/base.py#L339)
- [FSDP engine](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp)
- [Megatron engine](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/megatron)
- [engine configs](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/config/engine)

## Comparison Frame

```text
trainer request
  -> worker method
  -> engine contract
  -> backend-specific batch layout
  -> forward/backward schedule
  -> optimizer step
  -> metrics / parameter shard
```

## Completion Criteria

- 对同一个 actor update 画出 FSDP 和 Megatron 调用链；
- 记录 global batch、mini-batch、micro-batch 与 DP size 的关系；
- 说明 mixed precision、gradient accumulation 和 loss normalization 在哪个层处理；
- 与 [[training/distributed-training/fsdp|FSDP]]、[[training/distributed-training/megatron|Megatron 与 3D 并行]] 互相回链。

