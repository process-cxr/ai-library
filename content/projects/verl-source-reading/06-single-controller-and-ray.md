---
title: 06 Single Controller and Ray
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-02
type: project
status: seed
tags:
  - projects
  - source-reading
  - verl
  - ray
  - distributed-systems
---

本页研究 verl 如何让 controller 侧的一次 Python method call 变成多个 Ray actors 上的分布式执行。重点是 method registration、input dispatch、remote execution、result collection 和 resource placement。

## 本页问题

1. `@register` 在 worker method 上附加了什么 metadata？
2. WorkerGroup 如何在初始化时动态绑定远程方法？
3. `ONE_TO_ALL`、`DP_COMPUTE` 等 dispatch mode 如何拆分输入？
4. ResourcePool 如何把 role 映射到 node/GPU 资源？
5. colocated roles 如何共享进程与设备，又如何保持独立 method namespace？

## Source Anchors

- [register and dispatch](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/single_controller/base/decorator.py)
- [WorkerGroup](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/single_controller/base/worker_group.py)
- [Worker](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/single_controller/base/worker.py)
- [ResourcePoolManager](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/single_controller/ray/base.py#L185)
- [RayWorkerGroup](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/single_controller/ray/base.py#L418)

## Running Example

完整笔记以一次 `actor_wg.compute_log_prob(batch)` 或 `actor_wg.update_actor(batch)` 为例：

```text
controller method
  -> dispatch metadata
  -> DataProto / KVBatchMeta split
  -> remote worker invocation
  -> backend execution
  -> collect and merge
  -> controller result
```

## Completion Criteria

- 给出一个 method 从 decoration 到 invocation 的完整调用链；
- 解释 world size、DP rank 和 role prefix 的关系；
- 记录 blocking/non-blocking 与 future materialization 的语义；
- 说明 Ray 负责什么、`torch.distributed`/backend 又负责什么。

