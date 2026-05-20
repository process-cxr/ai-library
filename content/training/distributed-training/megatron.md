---
title: Megatron-LM 与 3D 并行
tags:
  - distributed-training
  - megatron
  - parallelism
---

> Megatron-LM 是 NVIDIA 开源的大规模预训练框架，核心贡献是 3D 并行（TP + PP + DP）的工程实现。

## TODO

- [ ] Tensor Parallelism（张量并行）原理与实现
- [ ] Pipeline Parallelism（流水线并行）调度策略（1F1B、Interleaved）
- [ ] 3D 并行组网：TP × PP × DP 如何划分 GPU 网格
- [ ] Sequence Parallelism（序列并行）
- [ ] Megatron-Core 与 Megatron-LM 的关系
- [ ] 与 [[training/distributed-training/fsdp|FSDP]] 的配合与对比

> [!note] Lab Status
> 占位中，待填充。
