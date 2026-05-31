---
title: Pipeline Parallel
created: 2026-03-22
published: 2026-03-22
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - distributed-training
  - pipeline-parallel
---

Pipeline Parallelism, PP，是把模型层切分到不同 GPU 或 GPU 组上，让不同设备负责不同层段。它解决的是模型深度和总参数量太大，单个 GPU 或单个 tensor-parallel group 无法容纳完整模型的问题。

与 [[training/distributed-training/tensor-parallel|Tensor Parallel]] 切分单层内部矩阵不同，pipeline parallel 切分的是层序列；与 [[training/distributed-training/data-parallel|Data Parallel]] 不同，每个 pipeline stage 只保存模型的一部分层。

## Layer Partition

设模型有 $L$ 层，pipeline parallel size 为 $P$。最简单做法是每个 stage 放约 $L/P$ 层：

```text
Stage 0: layers 0 ... k
Stage 1: layers k+1 ... m
Stage 2: layers m+1 ... n
...
```

forward 时 activation 从前一 stage 传到后一 stage；backward 时 gradient 从后一 stage 传回前一 stage。

如果层计算量不均匀，需要手动或自动 balance。embedding、LM head、MoE layers、长上下文 attention 都可能造成 stage 不均衡。

## Micro-batch 与 Pipeline Bubble

如果整个 batch 一次性通过 pipeline，后面的 stage 会在开始时空闲，前面的 stage 会在结束时空闲。这些空闲称为 pipeline bubble。

为提高利用率，通常把 global batch 切成多个 micro-batches，让不同 micro-batch 同时处在不同 stage：

```text
time →
Stage 0: mb1 F  mb2 F  mb3 F  ...
Stage 1:        mb1 F  mb2 F  ...
Stage 2:               mb1 F  ...
```

micro-batch 数越多，bubble 占比通常越低，但 activation buffering、调度复杂度和优化约束增加。

## GPipe 与 1F1B

常见调度策略：

- **GPipe-style**：先完成所有 micro-batch forward，再做 backward。实现直观，但需要保存较多 activation。
- **1F1B**：warmup 后每个 stage 交替做 one forward / one backward，减少 activation peak。
- **Interleaved 1F1B**：把每个设备上的层再切成 virtual pipeline stages，提高负载均衡并减少 bubble。

1F1B 是大模型训练中常见策略，因为它在显存和吞吐之间更平衡。

## 通信

PP 的通信主要是相邻 stage 之间传 activation 和 activation gradient。通信量与 micro-batch size、sequence length、hidden size 相关：

$$
\mathrm{ActivationComm}
\propto B_{\mu} \cdot S \cdot H
$$

相比 TP 的层内高频 collective，PP 通信更局部，但会引入 pipeline latency 和调度复杂度。

## 与 Gradient Accumulation

Pipeline training 通常需要 gradient accumulation。多个 micro-batches 通过 pipeline 后，累积梯度，再做一次 optimizer step。

global batch size 仍然满足：

$$
B_{\mathrm{global}}
= B_{\mu} \times \mathrm{num\_microbatches} \times DP
$$

其中 $B_{\mu}$ 是每个 micro-batch 的 token 或 sample 数。PP 配置会影响可用 micro-batch 数，从而影响 global batch 和优化 recipe。

## 适用场景

PP 适合：

- 模型层数多或总参数量很大；
- TP 后仍无法放下完整模型；
- 大规模预训练；
- GPU 数足够多，可以组合 TP/PP/DP；
- 愿意接受更复杂的调度和 checkpoint。

对中小模型或后训练，FSDP/ZeRO 往往更简单；PP 的复杂度不一定值得。

## 常见失败模式

- **pipeline bubble 大**：micro-batch 太少，GPU 利用率低。
- **stage imbalance**：某些 stage 计算更慢，拖累整体吞吐。
- **activation memory 高**：调度不当保存过多 micro-batch activation。
- **跨节点 stage 通信慢**：相邻 stage 放在低带宽链路上。
- **debug 困难**：错误可能只在特定 stage 或 micro-batch 出现。
- **checkpoint 复杂**：模型按 stage 分片保存和恢复。

## 相关概念

- [[training/distributed-training/megatron|Megatron and 3D Parallelism]]
- [[training/distributed-training/tensor-parallel|Tensor Parallel]]
- [[training/distributed-training/data-parallel|Data Parallel]]
- [[training/optimization/gradient-checkpointing|Gradient Checkpointing]]
- [[sources/papers/2018-gpipe|GPipe]]
