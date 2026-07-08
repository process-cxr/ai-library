---
title: Sequence Parallel
created: 2026-07-06
published: 2026-07-06
modified: 2026-07-06
type: topic
status: growing
area: training
tags:
  - distributed-training
  - sequence-parallel
  - activation-memory
---

Sequence Parallelism 是在模型并行训练中沿 sequence dimension 切分部分 activation 的方法。它通常与 [[training/distributed-training/tensor-parallel|Tensor Parallel]] 配合使用，用来降低每张 GPU 保存的 activation memory，并让 tensor-parallel group 内的计算和通信布局更加高效。

它的核心动机是：tensor parallel 已经把 hidden dimension 或 attention heads 切到多张 GPU 上，但许多非矩阵乘模块仍会产生按完整 sequence 保存的 activation。对长上下文训练而言，这部分 activation 会随 sequence length 线性增长，成为 ZeRO / FSDP 无法直接解决的显存瓶颈。

## 基本思想

Transformer 训练中的 hidden states 常可表示为：

$$
X \in \mathbb{R}^{B \times S \times H}
$$

其中 $B$ 是 micro-batch size，$S$ 是 sequence length，$H$ 是 hidden size。

Tensor parallel 通常切 hidden dimension 或 heads：

```text
GPU 0: hidden shard 0
GPU 1: hidden shard 1
...
```

Sequence parallel 则在部分模块中切 sequence dimension：

```text
GPU 0: token positions 0 ... k
GPU 1: token positions k+1 ... m
...
```

这样每张 GPU 只保存一部分 token positions 对应的 activation。若 tensor-parallel size 为 $T$，理想情况下某些 activation 的单卡保存量可从：

$$
B \cdot S \cdot H
$$

下降到近似：

$$
\frac{B \cdot S \cdot H}{T}
$$

实际收益取决于哪些模块可以 sequence-shard、通信是否增加、以及框架实现是否能避免额外 full activation materialization。

## 与 Tensor Parallel 的关系

Sequence parallel 通常不是独立替代 tensor parallel，而是 tensor parallel 的补充。

在 Megatron-style Transformer 中，tensor parallel 负责切分计算最重的线性层：

- QKV projection；
- attention output projection；
- MLP up / gate projection；
- MLP down projection。

LayerNorm、Dropout、residual connection 等模块通常按 token position 独立处理，不需要每个 rank 同时持有完整 sequence。Sequence parallel 利用这一点，在 tensor-parallel group 内把某些 activation 改为沿 sequence dimension 分布，从而减少 activation replication。需要注意，LayerNorm / RMSNorm 仍需要单个 token 的完整 hidden vector 或等价的正确聚合实现，因此 sequence parallel 的具体 layout transition 由框架负责。

二者的分工可概括为：

```text
Tensor Parallel:
  切 hidden / heads / MLP intermediate dimension
  主要降低单层矩阵计算和参数压力

Sequence Parallel:
  切 sequence dimension 上的部分 activation
  主要降低长序列训练中的 activation memory
```

在实现上，sequence parallel 往往需要在 tensor parallel 原有的 AllReduce 位置改用 ReduceScatter / AllGather 组合。ReduceScatter 将聚合结果同时切分到不同 rank；后续需要完整张量的模块再通过 AllGather 恢复。

## 典型作用位置

Sequence parallel 常用于不需要同时持有完整 sequence activation 的部分，例如：

- LayerNorm / RMSNorm 输入输出；
- Dropout；
- residual add；
- tensor-parallel MLP 或 attention 之间的中间 activation；
- 某些可以按 token 独立计算的 elementwise operations。

这些操作的共同点是：每个 token position 上的计算相对独立，不需要像 self-attention 那样直接访问所有 token 的 K/V。

Self-attention 本身通常不能简单地只靠 sequence parallel 解决，因为每个 query token 可能需要 attend 到历史或全局 context。长上下文 attention 的进一步切分通常由 [[training/distributed-training/context-parallel|Context Parallel]] 或专门的 distributed attention 实现处理。

## 通信模式

Sequence parallel 的关键通信包括：

- **ReduceScatter**：把原本 AllReduce 得到的完整结果，改为聚合后按 sequence shard 分发；
- **AllGather**：在需要完整 activation 或完整 hidden 表示时重新收集；
- **AllToAll**：某些高级实现中用于在 hidden-sharded 与 sequence-sharded layout 之间转换。

通信收益来自减少本地 activation 保存，但代价是引入更多 layout transition。是否划算取决于：

- sequence length 是否足够长；
- tensor-parallel size 是否较大；
- network bandwidth 是否充足；
- kernel 是否支持 sharded layout；
- AllGather / ReduceScatter 是否能与计算重叠。

## 与 Context Parallel 的区别

Sequence parallel 和 context parallel 都会提到切 sequence dimension，但目标和作用位置不同。

Sequence parallel 通常指在 tensor-parallel group 内切分部分 activation，尤其是 LayerNorm、Dropout、residual 等 token-wise 模块。它主要是 activation memory optimization。

Context parallel 更直接面向长上下文 attention，把 attention computation 所需的 context / K/V 沿 sequence dimension 分布到多张 GPU 上。它需要处理跨 shard attention、K/V 通信、causal mask 和 attention output 聚合，通信语义更复杂。

简化区分：

```text
Sequence Parallel:
  切部分 activation
  常与 TP 绑定
  重点是减少 activation replication

Context Parallel:
  切长上下文 attention 的 context
  重点是让更长 sequence 可训练
  需要处理跨 token shard 的 attention 通信
```

实际训练栈可能同时开启二者。

## 与 ZeRO / FSDP 的关系

[[training/distributed-training/zero|ZeRO]] 和 [[training/distributed-training/fsdp|FSDP]] 主要切分 parameters、gradients 和 optimizer states。它们对模型状态显存非常有效，但不自动把每张 GPU 在 forward/backward 中产生的 activation 等比例切分。

Sequence parallel 作用于 activation memory，因此常与 ZeRO/FSDP 形成互补：

- ZeRO/FSDP：减少模型状态冗余；
- sequence parallel：减少部分 activation 冗余；
- [[training/optimization/gradient-checkpointing|Gradient Checkpointing]]：通过重算减少 activation 保存；
- FlashAttention：减少 attention kernel 中间矩阵和内存访问。

长上下文训练通常需要这些方法组合使用。

## 适用场景

Sequence parallel 适合：

- 已经使用 tensor parallel；
- sequence length 较长，activation memory 成为瓶颈；
- tensor-parallel group 内通信带宽较好；
- Megatron-style 训练栈或支持 sequence parallel 的框架；
- 希望在不显著改变全局训练目标的前提下降低 activation 峰值。

它不适合被视为通用显存万能解。若 OOM 主要来自 optimizer states，sequence parallel 帮助有限；若模型没有使用 tensor parallel，单独引入 sequence parallel 的工程复杂度通常较高。

## 配置关注点

使用 sequence parallel 时需要关注：

- tensor-parallel size；
- sequence length 和 micro-batch size；
- 是否与 activation checkpointing 同时开启；
- 是否使用支持 sharded activation layout 的 fused kernels；
- communication overlap 是否生效；
- checkpoint 保存的参数 layout 与 activation layout 是否混淆；
- debug / profiler 中是否能正确解释每个 rank 的 activation shape。

在源码阅读中，sequence parallel 常出现在 tensor-parallel layer、normalization、dropout、parallel state 和 communication primitive 附近。读 Megatron 相关代码时，应重点追踪 tensor layout 如何在 hidden-sharded 与 sequence-sharded 之间转换。

## 常见失败模式

- **误认为切分了全部 activation**：sequence parallel 只作用于部分 activation，不会自动切分所有中间张量。
- **与 context parallel 混淆**：前者偏 activation layout，后者偏 distributed attention / long context。
- **通信抵消收益**：序列较短或 TP degree 过大时，AllGather / ReduceScatter 开销可能超过显存收益。
- **kernel 不兼容**：某些 fused kernel 可能要求完整或特定布局的 input。
- **debug 形状困难**：同一个张量在不同模块可能采用不同 shard layout，shape 需要结合 parallel group 解释。
- **checkpoint 认知错误**：sequence parallel 通常不改变权重切分方式，不能把 activation sharding 与 model checkpoint sharding 混为一谈。

## 相关概念

- [[training/distributed-training/tensor-parallel|Tensor Parallel]]
- [[training/distributed-training/torch-distributed|torch.distributed]]
- [[training/distributed-training/context-parallel|Context Parallel]]
- [[training/distributed-training/megatron|Megatron 与 3D 并行]]
- [[training/optimization/training-memory-estimation|Training Memory Estimation]]
- [[training/optimization/gradient-checkpointing|Gradient Checkpointing]]
- [[training/mid-training/long-context-training|Long Context Training]]
