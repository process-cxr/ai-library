---
title: Distributed Training
created: 2026-03-21
published: 2026-03-21
modified: 2026-07-06
---

分布式训练负责把大模型训练拆分到多 GPU / 多节点执行，包括数据并行、张量并行、流水线并行、sequence/context parallel、ZeRO、FSDP 和 Megatron。

建议阅读顺序：

1. [[training/distributed-training/torch-distributed|torch.distributed]]：理解 process group、rank、backend 和通信原语。
2. [[training/distributed-training/data-parallel|Data Parallel]]：理解最基础的数据切分和梯度同步。
3. [[training/distributed-training/zero|ZeRO]] 与 [[training/distributed-training/fsdp|FSDP]]：理解 data parallel 冗余状态如何切分。
4. [[training/distributed-training/tensor-parallel|Tensor Parallel]]：理解单层矩阵和 attention/MLP 如何切分。
5. [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]：理解层级切分、micro-batch 和 pipeline bubble。
6. [[training/distributed-training/sequence-parallel|Sequence Parallel]] 与 [[training/distributed-training/context-parallel|Context Parallel]]：理解 activation 与长上下文 attention 如何沿 sequence 维度切分。
7. [[training/distributed-training/megatron|Megatron 与 3D 并行]]：理解 TP × PP × DP 及长上下文并行的组合。

## Notes

- [[training/distributed-training/data-parallel|Data Parallel]]
- [[training/distributed-training/torch-distributed|torch.distributed]]
- [[training/distributed-training/tensor-parallel|Tensor Parallel]]
- [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]
- [[training/distributed-training/sequence-parallel|Sequence Parallel]]
- [[training/distributed-training/context-parallel|Context Parallel]]
- [[training/distributed-training/zero|ZeRO]]
- [[training/distributed-training/fsdp|FSDP 分布式训练]]
- [[training/distributed-training/megatron|Megatron 与 3D 并行]]
