---
title: Megatron-LM and 3D Parallelism
created: 2026-03-22
published: 2026-03-22
modified: 2026-05-31
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

3D parallelism 组合三个维度：

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

Sequence Parallelism 是 Megatron 中常与 TP 搭配的优化。TP 沿 hidden dimension 切分，而某些 activation 可以沿 sequence dimension 切分，从而减少每卡 activation memory。

它尤其适用于：

- LayerNorm；
- Dropout；
- residual 相关张量；
- 长上下文训练；
- TP degree 较大时的 activation 压力。

Sequence parallel 不等于 context parallel。前者通常切某些 activation 的 sequence dimension；后者更直接切长上下文 attention 的 context 维度，具体实现和通信模式不同。

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

## 配置思路

选择 TP/PP/DP 时通常考虑：

1. 单层是否能放下：决定 TP 是否需要增大。
2. 模型总层数和显存：决定 PP 是否需要增大。
3. global batch size 和吞吐：决定 DP 和 gradient accumulation。
4. 硬件拓扑：TP 尽量在高速互联内。
5. pipeline bubble：PP 越大越需要足够 micro-batches。
6. activation memory：长上下文可能需要 sequence/context parallel 和 checkpointing。

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
- [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]
- [[training/distributed-training/data-parallel|Data Parallel]]
- [[training/distributed-training/zero|ZeRO]]
- [[training/distributed-training/fsdp|FSDP]]
- [[training/optimization/training-memory-estimation|Training Memory Estimation]]
- [[sources/papers/2019-megatron-lm|Megatron-LM]]
