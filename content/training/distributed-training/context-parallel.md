---
title: Context Parallel
created: 2026-07-06
published: 2026-07-06
modified: 2026-07-06
type: topic
status: growing
area: training
tags:
  - distributed-training
  - context-parallel
  - long-context
---

Context Parallelism 是面向长上下文训练的分布式并行方法。它沿 sequence / context dimension 切分输入 token 和 attention 相关状态，使多个 GPU 共同处理同一条长序列，从而降低单卡 activation、attention workspace 和 K/V 相关显存压力。

它的目标不是扩大 batch，也不是简单切分模型参数，而是让原本单卡或单个 tensor-parallel group 难以承载的超长 sequence 进入训练。对长文档、长代码仓库、多轮工具轨迹和 agent trajectory 训练而言，context parallel 是理解 Megatron 等训练栈时必须掌握的概念。

## 为什么需要 Context Parallel

Transformer 训练中，长上下文带来的压力主要来自三部分：

- hidden activation 随 sequence length 近似线性增长；
- attention computation 需要 query 与 key/value 交互；
- naive attention 的 score / probability 中间量可能随 $S^2$ 增长。

设 hidden states 为：

$$
X \in \mathbb{R}^{B \times S \times H}
$$

当 $S$ 从 4K 增长到 128K 或 256K 时，仅靠减小 micro-batch、开启 activation checkpointing 或使用 FlashAttention 可能仍不够。此时需要把同一条长序列本身拆到多个 GPU 上。

Context parallel 将 sequence length $S$ 切为 $C$ 份：

```text
GPU 0: tokens 0 ... s1
GPU 1: tokens s1+1 ... s2
GPU 2: tokens s2+1 ... s3
...
```

每张 GPU 只长期持有一部分 context 对应的 activation。attention 需要跨 shard 访问 K/V 时，再通过通信完成必要的信息交换。

## 基本机制

在 causal self-attention 中，某个 query token 只能 attend 到它之前的 tokens。若 sequence 被切到多个 context-parallel ranks 上，后面 shard 的 query 可能需要前面 shard 的 K/V。

因此 context parallel 的核心问题是：

```text
local query shard 如何访问 remote key/value shard
```

常见实现思路包括：

- 在 context-parallel ranks 之间交换 K/V blocks；
- 使用 ring attention 类通信，让每个 rank 分阶段看到其他 shard 的 K/V；
- 对 attention output 做必要的 reduce / gather；
- 保持 causal mask、position id 和 sequence boundary 正确；
- 将通信与 attention block 计算重叠。

与 tensor parallel 的层内矩阵切分相比，context parallel 更关注 token shard 之间的 attention dependency。

## 与 Sequence Parallel 的区别

Context parallel 和 [[training/distributed-training/sequence-parallel|Sequence Parallel]] 都涉及 sequence dimension，但二者不应混用。

Sequence parallel 主要在 tensor-parallel group 内切分部分 activation，常用于 LayerNorm、Dropout、residual 等 token-wise 模块。它减少的是部分 activation replication。

Context parallel 直接处理长上下文 attention：同一条 sequence 的不同 token span 分布在不同 GPU 上，attention 需要跨 span 通信。它解决的是长 sequence 本身放不下或 attention 计算过重的问题。

二者的差异可概括为：

```text
Sequence Parallel:
  activation layout optimization
  常服务于 TP

Context Parallel:
  long-context attention parallelism
  常服务于超长 sequence training
```

## 与 Tensor Parallel / Pipeline Parallel / Data Parallel 的关系

Context parallel 可与其他并行维度组合。总 world size 可扩展为：

$$
W = TP \times PP \times CP \times DP
$$

其中：

- TP 切分单层内部 hidden / heads / MLP dimension；
- PP 切分层序列；
- CP 切分同一训练样本的 sequence / context；
- DP 复制或切分多份训练样本和模型状态。

引入 CP 后，数据并行的含义需要更谨慎理解。同一条样本可能被 CP group 内多张 GPU 共同处理；不同 CP groups 再处理不同样本 shard。训练日志中的 batch size、tokens per step、sequence length、global batch 和 loss normalization 都需要按并行组关系重新核对。

## Long Context Attention 中的通信

Context parallel 的通信通常比普通 sequence sharding 更复杂，因为 attention 不是纯 token-wise operation。一个 query block 的 attention 可能需要多个 key/value blocks。

通信设计要处理：

- K/V block 的发送和接收；
- causal mask 在跨 shard 场景下的正确性；
- attention softmax 的数值稳定性，尤其是分块 softmax 的 max / sum 归约；
- dropout / RNG 在分布式 attention 中的一致性；
- attention output 的聚合；
- backward 中对 Q/K/V gradient 的反向通信。

成熟实现通常会把 attention 按 block 计算，并在通信过程中维护 softmax 的 running max 和 normalization term，以避免直接 materialize 完整 attention matrix。

## 对长轨迹 Agent 数据的意义

Agent 轨迹、长文档和代码仓库数据经常包含 64K、128K 甚至更长的上下文。若训练目标需要模型学习长程状态维持、跨步骤一致性、工具协议和历史反馈利用，训练时就不能只依赖短窗口片段。

Context parallel 的价值在于提供工程可行性：让完整或更长的 trajectory window 能进入训练，而不是全部依赖截断。它并不自动保证长程信息被有效利用；数据侧仍需要设计合理的 packing、trajectory boundary、loss mask、采样权重和评测方法。

因此，CP 解决的是“能不能训这么长”的工程问题；长程能力是否提升，还取决于数据是否真的包含需要远距离依赖的监督信号。

## 与 FlashAttention 和 Checkpointing 的关系

Context parallel 常与以下技术同时使用：

- [[training/optimization/gradient-checkpointing|Activation Checkpointing]]：减少保存的 activation；
- FlashAttention / memory-efficient attention：降低 attention kernel 中间量；
- [[training/distributed-training/sequence-parallel|Sequence Parallel]]：减少部分 activation replication；
- tensor parallel：切分 hidden / heads；
- packed sequence 或 document boundary mask：保证样本边界语义正确。

这些技术作用层次不同：

```text
FlashAttention:
  优化单设备或局部 attention kernel

Activation Checkpointing:
  用重算减少 activation 保存

Sequence Parallel:
  切分部分 token-wise activation

Context Parallel:
  将长 sequence attention 分布到多张 GPU
```

实践中，长上下文训练往往需要组合，而不是只依赖单项技术。

## 配置关注点

使用 context parallel 时，需要重点检查：

- context parallel size；
- sequence length 是否能被合理切分；
- position ids 和 RoPE / ALiBi 等位置编码是否正确；
- packed samples 的 document boundary 是否跨 shard 保留；
- causal mask 是否在跨 shard attention 中正确；
- loss normalization 是否按全局 token 数计算；
- CP group 与 TP/PP/DP group 是否正确构建；
- checkpoint 是否保存了足够的 parallel metadata；
- eval / generation 是否支持相同或兼容的长上下文布局。

对源码阅读而言，context parallel 往往出现在 attention module、parallel state、position embedding、mask construction、distributed attention kernel 和 batch preparation 附近。

## 常见失败模式

- **只解决长度，不解决数据质量**：能训练 128K sequence 不代表模型一定学到有效长程依赖。
- **mask 错误**：跨 shard causal mask 或 document boundary mask 错误会造成未来泄漏或跨样本污染。
- **position id 错误**：长上下文位置编码在 shard 边界处不连续或重复，会破坏训练信号。
- **loss 归一化错误**：不同 CP rank 上有效 token 数不同，若归一化不当会改变优化目标。
- **通信瓶颈**：长上下文 attention 的 K/V 交换可能成为主要耗时。
- **与 packing 冲突**：packed sequence 的样本边界被切到不同 shard 后，mask 和 metadata 必须同步维护。
- **checkpoint / resume 困难**：改变 CP degree 可能需要额外的 checkpoint 转换或重新分片。

## 适用场景

Context parallel 适合：

- 训练 32K 以上长上下文模型；
- 长文档、代码仓库、多轮 agent trajectory、长工具链数据；
- 单卡 activation / attention workspace 成为瓶颈；
- 已有成熟 TP/PP/DP 训练栈，并支持 distributed attention；
- 需要在 mid-training 中保留较长原始样本结构。

它不适合用来弥补普通短序列训练中的数据不足，也不应替代数据工程中的长程依赖构造、质量过滤和评测设计。

## 相关概念

- [[training/distributed-training/sequence-parallel|Sequence Parallel]]
- [[training/distributed-training/torch-distributed|torch.distributed]]
- [[training/distributed-training/tensor-parallel|Tensor Parallel]]
- [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]
- [[training/distributed-training/megatron|Megatron 与 3D 并行]]
- [[training/optimization/training-memory-estimation|Training Memory Estimation]]
- [[training/optimization/gradient-checkpointing|Gradient Checkpointing]]
- [[training/data-engineering/packing|Packing]]
- [[training/mid-training/long-context-training|Long Context Training]]
