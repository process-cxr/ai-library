---
title: Tensor Parallel
created: 2026-03-22
published: 2026-03-22
modified: 2026-07-06
type: topic
status: mature
area: training
tags:
  - distributed-training
  - tensor-parallel
---

Tensor Parallelism, TP，是把单层内部的张量计算切分到多张 GPU 上。它不同于 [[training/distributed-training/data-parallel|Data Parallel]] 的“每张 GPU 都有完整模型”，也不同于 [[training/distributed-training/pipeline-parallel|Pipeline Parallel]] 的“不同 GPU 负责不同层”。TP 关注的是同一层里的矩阵乘法、attention heads 或 MLP hidden dimension 如何拆分。

TP 主要解决单层参数和激活太大、单卡无法高效计算的问题，是大规模预训练和 Megatron-style 训练的核心组件。

## 线性层切分

Transformer 中大量计算来自线性层：

$$
Y = XW
$$

其中 $X \in \mathbb{R}^{B\times H}$，$W \in \mathbb{R}^{H\times K}$。Tensor parallel 可以按列或按行切分 $W$。

### Column Parallel

按输出维度切分：

$$
W = [W_1, W_2, \dots, W_p]
$$

每张 GPU 计算：

$$
Y_i = XW_i
$$

最后 $Y=[Y_1,\dots,Y_p]$。如果下一层可以消费分片输出，就不必立即 AllGather。

### Row Parallel

按输入维度切分：

$$
W =
\begin{bmatrix}
W_1 \\
W_2 \\
\dots \\
W_p
\end{bmatrix},
\quad
X=[X_1,\dots,X_p]
$$

每张 GPU 计算 partial output：

$$
Y_i = X_iW_i
$$

然后通过 AllReduce 求和：

$$
Y = \sum_i Y_i
$$

Megatron-LM 的 MLP 常用 column-parallel + row-parallel 配对，以减少不必要通信。

## Attention 与 MLP 切分

在 Transformer 中：

- attention 的 Q/K/V projection 可以按 heads 或 hidden dimension 切分；
- attention output projection 可以用 row parallel；
- MLP 的 up/gate projection 常用 column parallel；
- MLP 的 down projection 常用 row parallel。

这种设计让每个 GPU 只处理部分 heads 或部分 intermediate dimension，降低单卡计算和参数压力。

## 通信模式

TP 的通信发生在层内部，通常比 data parallel 更频繁。常见通信：

- AllReduce：合并 row-parallel partial outputs；
- AllGather：收集分片 activation；
- ReduceScatter：合并并分发结果；
- broadcast / scatter：准备输入分片。

因为 TP 通信频繁，最好放在同节点高速互联 GPU 上，例如 NVLink。跨节点做大 TP 往往通信开销很高。

## Sequence Parallel

[[training/distributed-training/sequence-parallel|Sequence Parallelism]] 常与 tensor parallel 配合，把某些 activation 沿 sequence dimension 切分，减少 activation memory。它通常作用于 LayerNorm、Dropout、residual 等 token-wise 模块附近：这些模块按 token position 独立处理，不需要同时持有完整 sequence，但 LayerNorm / RMSNorm 仍需要该 token 的完整 hidden vector 或等价的正确聚合实现。

直观上：

- TP 切 hidden / heads / MLP dimension；
- sequence parallel 切 sequence dimension；
- 二者配合降低单卡 activation 和通信压力。

具体实现依赖框架，Megatron-Core 中 sequence parallel 是重要优化。需要注意，sequence parallel 不等于 [[training/distributed-training/context-parallel|Context Parallel]]：前者主要切分部分 activation layout，后者直接面向长上下文 attention 的 context / K/V 跨 shard 计算。

## 与其他并行的关系

TP 解决单层太大的问题；PP 解决层数/模型深度分布问题；DP/FSDP/ZeRO 解决数据并行冗余状态问题。

常见组合：

$$
\mathrm{WorldSize}
= TP \times PP \times DP
$$

如果一个 1024 GPU 训练使用 $TP=8, PP=8$，则 $DP=16$。每个并行维度有不同通信组。

## 适用场景

TP 适合：

- 单层矩阵乘法太大；
- attention heads 很多；
- hidden size / MLP intermediate size 很大；
- 希望提高单 step 计算并行度；
- 大规模预训练中同节点 GPU 高速互联充足。

TP 不适合过小模型或低带宽跨节点环境，因为通信可能超过计算收益。

## 常见失败模式

- **TP degree 过大**：每卡矩阵太小，kernel efficiency 下降。
- **跨节点 TP**：频繁通信导致严重瓶颈。
- **通信未重叠**：AllReduce/AllGather 等待时间变长。
- **与 PP/DP 组网不匹配**：GPU 拓扑没有按通信密集度布置。
- **实现细节错误**：分片权重、checkpoint 合并和随机种子处理复杂。

## 相关概念

- [[training/distributed-training/torch-distributed|torch.distributed]]
- [[training/distributed-training/megatron|Megatron and 3D Parallelism]]
- [[training/distributed-training/sequence-parallel|Sequence Parallel]]
- [[training/distributed-training/context-parallel|Context Parallel]]
- [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]
- [[training/distributed-training/data-parallel|Data Parallel]]
- [[training/distributed-training/zero|ZeRO]]
- [[training/optimization/training-memory-estimation|Training Memory Estimation]]
- [[sources/papers/2019-megatron-lm|Megatron-LM]]
