---
title: Megatron-LM and 3D Parallelism
created: 2026-03-22
published: 2026-03-22
modified: 2026-07-06
type: topic
status: mature
area: training
tags:
  - distributed-training
  - megatron
  - parallelism
---

Megatron-LM 是 NVIDIA 开源的大规模 Transformer 训练框架，其核心价值在于系统化实现 model parallel training，尤其是 tensor parallelism、pipeline parallelism、data parallelism 组合而成的 3D parallelism。Megatron-Core 则是更模块化的底层库，用于在不同训练框架中复用这些并行组件。

3D parallelism 的目标是把一个超大模型训练拆到大量 GPU 上，同时控制单卡显存、通信开销和 pipeline bubble。

## 3D Parallelism

3D parallelism 组合三个基本维度：

| 维度 | 切分对象 | 主要解决问题 |
|---|---|---|
| Tensor Parallel, TP | 单层内部矩阵、attention heads、MLP hidden dimension | 单层计算和参数太大 |
| Pipeline Parallel, PP | 模型层序列 | 模型深度/总参数太大 |
| Data Parallel, DP | 数据 batch | 提高吞吐，并复制或切分模型状态 |

GPU 总数满足：

$$
W = TP \times PP \times DP
$$

例如 $W=1024$，若 $TP=8$、$PP=8$，则 $DP=16$。这意味着：

- 每个 tensor-parallel group 有 8 张 GPU，共同计算一个 stage 内的层；
- pipeline 有 8 个 stage，按层切分模型；
- data parallel 有 16 份 pipeline/tensor 组合副本处理不同数据。

长上下文训练中还可能引入 [[training/distributed-training/sequence-parallel|Sequence Parallel]] 或 [[training/distributed-training/context-parallel|Context Parallel]]。前者通常作为 TP 的 activation 优化存在；后者会把同一长序列的 context 切到多个 ranks 上。此时 world size 的分解可能扩展为：

$$
W = TP \times PP \times CP \times DP
$$

其中 CP 是否出现、是否与 TP 共享通信组、以及 loss normalization 如何处理，取决于具体训练框架实现。

## GPU 网格与通信拓扑

不同并行维度通信模式不同：

- TP：层内频繁 AllReduce / AllGather，通信密集，最好放在同节点 NVLink 内。
- PP：相邻 stage 传 activation 和 gradient，通信较局部，但受 pipeline 调度影响。
- DP：同步梯度或 sharded states，通信与参数量或状态量相关，常跨节点。

因此，合理组网通常把通信最密集的 TP 放在高速互联范围内，把 PP stage 按拓扑顺序排列，再把 DP 扩展到更多节点。并行配置不是纯数学分解，还要匹配硬件拓扑。

## Tensor Parallel in Megatron

Megatron 的 tensor parallelism 经典设计包括：

- attention QKV projection 按 heads / hidden dimension 切分；
- MLP up projection 使用 column parallel；
- MLP down projection 使用 row parallel；
- 配对 column/row parallel 减少中间 AllGather；
- 在必要位置使用 AllReduce 合并 partial outputs。

这使每张 GPU 只计算部分矩阵和 heads，但每层会引入 collective communication。TP degree 过大时，每卡矩阵变小，kernel efficiency 下降，通信占比上升。

## Pipeline Parallel in Megatron

Megatron 支持 pipeline parallelism，将 Transformer layers 分到不同 stages。常见调度：

- 1F1B：warmup 后每个 stage 交替 forward/backward；
- interleaved pipeline：每个 physical stage 拆成多个 virtual stages，减少 bubble；
- micro-batch scheduling：通过更多 micro-batches 提高 pipeline 利用率。

PP 的关键是平衡 stage 计算量。如果 embedding、LM head、MoE 或长上下文 attention 集中在某些 stage，会造成 straggler。

## Sequence Parallelism

[[training/distributed-training/sequence-parallel|Sequence Parallelism]] 是 Megatron 中常与 TP 搭配的优化。TP 沿 hidden dimension 切分，而某些 activation 可以沿 sequence dimension 切分，从而减少每卡 activation memory。

它尤其适用于：

- LayerNorm；
- Dropout；
- residual 相关张量；
- 长上下文训练；
- TP degree 较大时的 activation 压力。

Sequence parallel 不等于 [[training/distributed-training/context-parallel|Context Parallel]]。前者通常切某些 activation 的 sequence dimension；后者更直接切长上下文 attention 的 context 维度，具体实现和通信模式不同。

## Context Parallelism

Context parallelism 面向超长 sequence training。它把同一条 sequence 的 token span 分布到多个 ranks 上，并在 attention 中通过 K/V 交换或 distributed attention kernel 完成跨 shard 计算。

在代码阅读中，context parallel 通常会影响：

- attention mask 和 causal mask 构造；
- position ids / RoPE offset；
- batch 中 packed sequence 的 boundary metadata；
- loss 按有效 token 数归一化；
- distributed checkpoint 中的 parallel metadata。

因此，CP 不只是一个显存优化开关，也会进入数据管线、mask、attention kernel 和 checkpoint 设计。

## 与 ZeRO / FSDP 的关系

Megatron-style 3D parallelism 可以与 ZeRO/FSDP 类思想组合，但需要明确维度：

- TP/PP 属于 model parallel；
- DP/ZeRO/FSDP 作用在 data parallel 维度；
- ZeRO/FSDP 切分 data parallel 副本之间的冗余 parameters、gradients、optimizer states。

在大规模预训练中，常见组合是 TP + PP + DP，再在 DP 维度使用 optimizer state sharding 或 distributed optimizer。对中小规模后训练，FSDP/ZeRO 单独使用可能更简单。

## Megatron-LM 与 Megatron-Core

可以粗略理解为：

- **Megatron-LM**：完整训练框架和脚本生态，面向大规模 Transformer 预训练。
- **Megatron-Core**：更模块化的并行、Transformer layer、optimizer、distributed checkpoint 等核心组件。

实际项目可能直接使用 Megatron-LM，也可能通过 NeMo、Megatron-Core 或其他训练框架间接使用这些组件。

## 源码阅读分层

读 Megatron 源码时，应把“通信库”和“并行训练系统”分开。Megatron 并不是从零实现 GPU 间网络通信，而是在 [[training/distributed-training/torch-distributed|torch.distributed]] / NCCL 之上组织大模型并行训练语义。

一条稳定的源码阅读分层如下：

```text
第一层：torch.distributed / NCCL
  提供通信原语
  - process group
  - rank / world size
  - all_reduce
  - all_gather
  - reduce_scatter
  - broadcast
  - send / recv
  - all_to_all

第二层：Megatron parallel_state
  建立并管理并行通信组
  - tensor model parallel group
  - pipeline model parallel group
  - data parallel group
  - context parallel group
  - expert parallel group
  - embedding / position embedding group

第三层：Megatron parallel modules and distributed optimizer
  用通信原语实现并行训练语义
  - tensor_parallel layers
  - sequence/context parallel communication
  - pipeline schedules
  - custom DDP
  - distributed optimizer
  - communication overlap

第四层：training loop
  串联完整训练过程
  - batch / dataloader
  - forward step
  - backward step
  - gradient sync
  - optimizer step
  - lr scheduler
  - checkpoint save/load
  - evaluation / logging
```

这个分层的关键是：`torch.distributed` 只提供“如何通信”，Megatron 决定“为什么通信、谁和谁通信、什么时候通信、通信前后的 tensor layout 是什么”。

### 第一层：通信原语

第一层是 PyTorch 与 NCCL 提供的分布式通信能力。Megatron 中的 AllReduce、AllGather、ReduceScatter、Broadcast、AllToAll 和 send/recv 最终都会落到这一层。

该层需要重点理解：

- 每个 collective operation 的输入输出语义；
- process group 决定通信范围；
- NCCL backend 决定 GPU 间实际通信实现；
- collective 调用顺序必须在所有参与 ranks 上一致；
- 异步通信需要正确 `wait` 或与计算安全重叠。

读源码时看到 `torch.distributed.all_reduce(...)`，不能只看 API 名字，还要确认它在哪个 group 上执行。例如 AllReduce 在 data parallel group 中通常表示梯度同步；在 tensor parallel group 中可能表示 row-parallel linear 的 partial output 合并。

### 第二层：parallel_state

Megatron 的 `parallel_state` 负责把全局 ranks 切分为多个并行维度的 process groups。它相当于 Megatron 并行系统的坐标系。

典型问题包括：

- 当前 rank 的 tensor parallel rank 是多少；
- 当前 rank 属于哪个 pipeline stage；
- 哪些 ranks 构成同一个 data parallel group；
- context parallel 或 expert parallel 是否启用；
- embedding layer 和 output layer 是否需要特殊通信组；
- virtual pipeline rank 如何影响 interleaved schedule。

这一层决定后续通信的作用范围。如果 process group 构造错误，后面的 tensor parallel、pipeline parallel 和 optimizer sharding 都会失去正确语义。

### 第三层：并行模块与优化器语义

第三层把通信原语组织成模型层、调度器和优化器。

典型模块包括：

- **tensor_parallel layers**：ColumnParallelLinear、RowParallelLinear、parallel embedding 等；
- **tensor_parallel mappings**：在 forward/backward 中执行 copy、gather、reduce、scatter 等 layout 转换；
- **pipeline schedules**：控制 micro-batch 如何在 pipeline stages 间流动；
- **distributed data parallel / distributed optimizer**：控制梯度 bucket、参数同步、optimizer state sharding 和通信重叠；
- **sequence/context parallel**：控制 activation 或长上下文 attention 在 ranks 之间的分布。

这一层的核心问题是 tensor layout。每个 tensor 在通信前后都需要回答：

```text
当前 tensor 是 replicated 还是 sharded？
沿哪个 dimension sharded？
属于哪个 parallel group？
forward 需要什么 layout？
backward 后 gradient 应该保留完整还是分片？
```

例如，RowParallelLinear 的 forward 可能需要对 partial output 做 AllReduce；ColumnParallelLinear 的 output 可以保持 hidden dimension shard，直到后续模块需要 gather 或 reduce。FSDP/ZeRO-style optimizer 则会围绕参数 shard、梯度 shard 和 optimizer state shard 组织 AllGather / ReduceScatter。

### 第四层：训练循环

第四层是训练 loop。它把数据、模型、loss、反向传播、优化器、checkpoint 和日志串联起来。

读训练循环时需要关注：

- batch 从 dataloader 进入模型前是否已经按 TP/PP/CP 需要处理；
- forward step 是否只返回 loss，还是还包含额外 metadata；
- backward step 何时触发梯度同步；
- gradient accumulation 与 pipeline micro-batch 如何交互；
- optimizer step 是否依赖 distributed optimizer；
- checkpoint 保存的是 sharded training state 还是可导出的推理权重；
- consumed tokens / samples 如何记录并用于 resume。

训练 loop 本身看似是普通 forward / backward / step，但在 Megatron 中每一步都会牵涉底层并行状态。源码阅读时不要只沿 Python 调用栈向下看，还要同时追踪当前 rank 的并行坐标和 tensor layout。

### 阅读顺序

较稳妥的源码阅读顺序是：

1. 先读 parallel state 初始化，明确 TP/PP/DP/CP/EP groups 如何构造；
2. 再读 tensor parallel mappings，理解 copy / gather / reduce / scatter 的 autograd 语义；
3. 再读 ColumnParallelLinear、RowParallelLinear 和 parallel embedding；
4. 再读 pipeline schedule，理解 micro-batch 如何跨 stages 流动；
5. 再读 custom DDP / distributed optimizer，理解梯度同步和 optimizer state sharding；
6. 最后读 training loop 和 checkpoint，理解完整训练过程如何被串起来。

这个顺序比从 `pretrain_gpt.py` 一路深挖更稳定，因为它先建立“通信组和 tensor layout”的坐标系，再看训练主循环。

## 配置思路

选择 TP/PP/DP 时通常考虑：

1. 单层是否能放下：决定 TP 是否需要增大。
2. 模型总层数和显存：决定 PP 是否需要增大。
3. global batch size 和吞吐：决定 DP 和 gradient accumulation。
4. 硬件拓扑：TP 尽量在高速互联内。
5. pipeline bubble：PP 越大越需要足够 micro-batches。
6. activation memory：长上下文可能需要 sequence/context parallel 和 checkpointing。
7. checkpoint：TP/PP/DP/CP 坐标和 optimizer state 是否能可靠保存、恢复和转换。

没有单一最优配置。并行策略需要和模型结构、sequence length、batch size、网络拓扑、checkpoint 策略一起调。

## 常见失败模式

- **只按 GPU 数分解**：忽略 TP/PP/DP 的通信差异和硬件拓扑。
- **TP 过大**：通信和小矩阵低效率抵消收益。
- **PP 过大**：pipeline bubble 增加，stage balance 困难。
- **DP 过大**：global batch size 过大，优化 recipe 失效。
- **checkpoint 复杂**：TP/PP/DP 分片 checkpoint 合并、加载和迁移困难。
- **调试困难**：错误可能只发生在某个 parallel group 或 stage。

## 相关概念

- [[training/distributed-training/tensor-parallel|Tensor Parallel]]
- [[training/distributed-training/torch-distributed|torch.distributed]]
- [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]
- [[training/distributed-training/data-parallel|Data Parallel]]
- [[training/distributed-training/sequence-parallel|Sequence Parallel]]
- [[training/distributed-training/context-parallel|Context Parallel]]
- [[training/distributed-training/zero|ZeRO]]
- [[training/distributed-training/fsdp|FSDP]]
- [[training/optimization/checkpoint-sharding|Checkpoint Sharding]]
- [[training/optimization/training-memory-estimation|Training Memory Estimation]]
- [[sources/papers/2019-megatron-lm|Megatron-LM]]
