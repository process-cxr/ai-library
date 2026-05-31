---
title: Data Parallel
created: 2026-03-21
published: 2026-03-21
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - distributed-training
  - data-parallel
---

Data Parallelism 是最基础的分布式训练方式：每张 GPU 持有一份完整模型副本，处理不同 mini-batch 数据，反向传播后同步梯度，使所有副本保持一致。

它解决的是吞吐问题，而不是单卡显存装不下模型的问题。普通 data parallel 中，parameters、gradients 和 optimizer states 在每张 GPU 上完整复制；如果这些状态过大，需要 [[training/distributed-training/zero|ZeRO]] 或 [[training/distributed-training/fsdp|FSDP]] 做切分。

## 基本流程

设有 $D$ 张 GPU，每张 GPU 处理 micro-batch size $b$。一次训练 step：

1. 每张 GPU 拿到不同数据 shard；
2. 每张 GPU 用完整模型做 forward；
3. 每张 GPU 本地 backward 得到梯度 $g_i$；
4. 所有 GPU 通过 AllReduce 求平均梯度：

$$
g = \frac{1}{D}\sum_{i=1}^{D}g_i
$$

5. 每张 GPU 用相同梯度更新自己的模型副本。

更新后所有模型副本保持一致。

## Global Batch Size

Data parallel 会扩大 global batch size：

$$
B_{\mathrm{global}}
= b_{\mu} \times a \times D
$$

其中：

- $b_{\mu}$ 是每张 GPU 的 micro-batch size；
- $a$ 是 gradient accumulation steps；
- $D$ 是 data parallel world size。

global batch size 会影响优化动态。扩大 GPU 数时，如果保持每卡 batch 不变，global batch 会增大，可能需要调整 learning rate、warmup、gradient clipping 和训练 token schedule。

## 通信开销

普通 data parallel 的主要通信是 gradient AllReduce。每个 step 需要同步所有参数对应的梯度，通信量大致与参数量成正比。

优化方式包括：

- overlap gradient communication with backward；
- bucket gradients；
- 使用高速互联，例如 NVLink / InfiniBand；
- gradient accumulation，减少同步频率；
- ZeRO/FSDP，减少冗余状态并改变通信模式。

当模型变大或跨节点训练时，通信可能成为瓶颈。Data parallel 扩展效率不只取决于 GPU FLOPs，也取决于网络带宽和拓扑。

## 与 DDP

PyTorch DDP, DistributedDataParallel，是常用 data parallel 实现。它会在 backward 过程中按 bucket 异步 AllReduce 梯度，从而尽量把通信和计算重叠。

DDP 的优势：

- 实现成熟；
- 语义接近单机训练；
- 适合中小模型和后训练；
- 调试相对容易。

DDP 的限制：

- 每卡保存完整模型状态；
- optimizer states 冗余；
- 大模型显存压力高；
- 跨节点通信开销大。

## 与 ZeRO/FSDP 的关系

ZeRO/FSDP 可以看作在 data parallel 维度上减少冗余：

- DDP：每张 GPU 完整保存 parameters、gradients、optimizer states。
- ZeRO-1：切分 optimizer states。
- ZeRO-2：切分 optimizer states 和 gradients。
- ZeRO-3 / FSDP FULL_SHARD：切分 parameters、gradients 和 optimizer states。

因此，ZeRO/FSDP 不是替代 data parallel，而是在 data parallel 的基础上切分冗余模型状态。

## 适用场景

Data parallel 适合：

- 模型单卡能放下；
- 希望提高训练吞吐；
- SFT、DPO、RLHF 等后训练；
- 小到中等规模 CPT；
- 需要实现简单和调试方便。

当单卡放不下模型状态时，需要 FSDP/ZeRO；当单层计算太大时，需要 [[training/distributed-training/tensor-parallel|Tensor Parallel]]；当层数太多或模型总规模太大时，需要 [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]。

## 常见失败模式

- **global batch 被无意放大**：GPU 数增加后优化动态改变。
- **通信成为瓶颈**：AllReduce 时间抵消并行收益。
- **数据 shard 不均匀**：不同 rank 处理长度或难度差异大，造成 straggler。
- **随机性不一致**：seed、dropout、data loader 状态导致复现实验困难。
- **只用 DP 训练超大模型**：每卡完整状态导致 OOM。

## 相关概念

- [[training/distributed-training/zero|ZeRO]]
- [[training/distributed-training/fsdp|FSDP]]
- [[training/distributed-training/tensor-parallel|Tensor Parallel]]
- [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]
- [[training/scaling/training-budget|Training Budget]]
