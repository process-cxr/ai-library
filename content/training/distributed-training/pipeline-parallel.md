---
title: Pipeline Parallel
created: 2026-03-22
published: 2026-03-22
modified: 2026-07-06
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

## DualPipe 与 MoE 通信重叠

DeepSeek-V3 在 16-way Pipeline Parallelism 和 64-way Expert Parallelism 上使用 DualPipe，目标是把 cross-node expert parallelism 的 all-to-all 通信隐藏在 forward / backward 计算中。

一个 forward / backward chunk 会被拆成 attention、all-to-all dispatch、MLP、all-to-all combine，以及 backward for input、backward for weights 和 PP communication。DualPipe 重新安排一对 forward / backward chunks，并采用 bidirectional pipeline，同时从 pipeline 两端输入 micro-batches，使计算、expert dispatch / combine 和相邻 stage 通信尽量重叠。

它的意义不只是减少 pipeline bubble。MoE 中 token 会被动态发送到远端 experts，如果 dispatch 与 MLP 串行执行，sparse computation 的收益会被网络延迟抵消。DualPipe 配合 node-limited routing、定制 all-to-all kernels、IB / NVLink 分层转发和独立 communication stream，可以把通信从显式等待变成计算期间的后台工作。

相应的取舍是：

- 比 1F1B 需要更多 activation buffering；
- 需要保存两份 model parameters；
- 调度与 debug 复杂度更高；
- 需要根据 communication / computation ratio 手工配置资源；
- 性能高度依赖网络拓扑和 kernel 实现。

因此，DualPipe 适合超大规模、通信占比较高的 MoE 训练，不应被理解为所有 pipeline training 的默认替代方案。

## GPipe 的同步更新语义

[[sources/papers/2018-gpipe|GPipe]] 的关键设计是：把一个 mini-batch 切成多个 micro-batches，先让它们流水通过所有 stages，再累积所有 micro-batch 的梯度，最后统一执行一次 optimizer update。

```text
固定参数 W
  -> mb_1 forward / backward
  -> mb_2 forward / backward
  -> ...
  -> 累积梯度
  -> 一次更新得到 W'
```

因此，同一个 mini-batch 内的 micro-batch 使用同一组参数。它与异步 pipeline 的主要差异在于不会因为 stage 更新时序不同而产生 weight staleness，但代价是需要等待整个 mini-batch 的梯度完成。

GPipe 还结合 rematerialization：forward 阶段主要保存 pipeline partition 边界的 activation，backward 阶段重新计算 cell 内部的 forward，以降低 activation memory。论文实验中，当 micro-batch 数量达到 partition 数量的约 4 倍时，pipeline bubble 通常已经较小。

GPipe 的通信只发生在相邻 stage 边界，主要传递 activation 和 activation gradient；它不负责切分单个 layer 内部的矩阵。后者属于 [[training/distributed-training/tensor-parallel|Tensor Parallel]]，也是 Megatron-LM 的主要方向。

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

在包含 TP/PP/DP/CP 的训练栈中，pipeline stage 只保存模型的一部分层。checkpoint、dataloader 和日志都需要记录 pipeline rank。否则恢复训练或导出完整权重时，容易出现层顺序错位、stage 缺失或 optimizer state 对不上参数的问题。

## 常见失败模式

- **pipeline bubble 大**：micro-batch 太少，GPU 利用率低。
- **stage imbalance**：某些 stage 计算更慢，拖累整体吞吐。
- **activation memory 高**：调度不当保存过多 micro-batch activation。
- **跨节点 stage 通信慢**：相邻 stage 放在低带宽链路上。
- **debug 困难**：错误可能只在特定 stage 或 micro-batch 出现。
- **checkpoint 复杂**：模型按 stage 分片保存和恢复。

## 相关概念

- [[training/distributed-training/megatron|Megatron and 3D Parallelism]]
- [[training/distributed-training/torch-distributed|torch.distributed]]
- [[training/distributed-training/tensor-parallel|Tensor Parallel]]
- [[training/distributed-training/data-parallel|Data Parallel]]
- [[training/optimization/gradient-checkpointing|Gradient Checkpointing]]
- [[training/optimization/checkpoint-sharding|Checkpoint Sharding]]
- [[sources/papers/2018-gpipe|GPipe]]
