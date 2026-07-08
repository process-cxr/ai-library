---
title: Gradient Checkpointing
created: 2026-03-15
published: 2026-03-15
modified: 2026-07-06
type: topic
status: mature
area: training
tags:
  - training-optimization
  - memory
---

Gradient Checkpointing，也常称 Activation Checkpointing，是一种用计算换显存的训练技术。普通反向传播需要保存前向传播中的大量 activations，以便 backward 计算梯度；checkpointing 只保存少量边界 activation，在 backward 时重新执行部分 forward 来恢复中间 activation。

它主要降低 [[training/optimization/training-memory-estimation|Training Memory Estimation]] 中的 activation memory，而不是降低 parameters、gradients 或 optimizer states。

## 核心思想

普通训练：

```text
forward:  保存每层中间激活
backward: 直接读取保存的激活计算梯度
```

Activation checkpointing：

```text
forward:  只保存 checkpoint 边界
backward: 从边界重新计算该段 forward，再计算梯度
```

因此它的 trade-off 是：

- 显存下降；
- backward 需要额外 forward recomputation；
- 训练 step time 增加；
- 实现需要保证 recomputation 与原 forward 一致。

从自动微分角度看，checkpointing 改变的是 backward 所需中间值的保存策略，而不是数学上的 loss 或 gradient 定义。只要 recomputation 与原 forward 完全一致，梯度应与不使用 checkpointing 时一致；差异主要来自随机性、数值非确定性和 kernel 实现。

## 为什么对 LLM 重要

Transformer 训练中 activation 与 micro-batch、sequence length、layers、hidden size 强相关：

$$
M_{\text{act}}
\propto B_{\mu} \cdot S \cdot L \cdot H
$$

当训练长上下文或大 batch 时，activation 可能成为显存瓶颈。ZeRO/FSDP 能切分模型状态，但不能自动切分每个 GPU 在本地 forward/backward 中产生的 activation。因此，checkpointing 经常与 FSDP、tensor parallel、sequence/context parallel 和 FlashAttention 一起使用。

需要区分两类显存：

- **模型状态**：parameters、gradients、optimizer states，主要随参数量增长；
- **activation**：forward 中为 backward 保存的中间值，主要随 micro-batch、sequence length 和层数增长。

Checkpointing 主要作用于后者。如果 OOM 来自 AdamW optimizer state，checkpointing 只能释放一部分空间，不能替代 ZeRO/FSDP 或参数高效训练。

## Checkpoint 粒度

常见粒度包括：

- 每个 Transformer block 一个 checkpoint；
- 每几个 blocks 一个 checkpoint；
- 只 checkpoint attention 或 MLP；
- selective activation checkpointing；
- framework 自动策略。

粒度越细，保存的 activation 越少，但 recomputation 越多。粒度越粗，速度损失较小，但显存节省也较弱。

实际选择粒度时通常从 Transformer block 级别开始，因为它与模型结构边界自然对应，易于调试，也能获得较稳定的显存收益。更细粒度的 selective checkpointing 适合在 profiler 已经确认某些子模块 activation 特别昂贵时使用，例如 MLP 中间激活或 attention 部分中间量。

## 内存与计算权衡

理论上，activation checkpointing 可以把深层网络的 activation 保存量从随层数线性增长，降低到更接近分段边界数量的增长，但代价是 backward 时重新计算分段内部 forward。实践中常见现象是：

- activation memory 显著下降；
- 每 step 训练时间增加；
- recomputation FLOPs 增加，但总训练 FLOPs 不一定简单翻倍；
- 显存释放后可以增大 micro-batch 或 sequence length，从而部分抵消吞吐损失。

因此它不是单纯“训练更慢”，而是把不可运行的配置变成可运行，或用额外计算换更好的硬件利用率。是否划算取决于原始瓶颈：如果 GPU 因显存只能用很小 micro-batch，checkpointing 反而可能提高有效吞吐；如果原本已经 compute-bound 且显存充足，它只会增加开销。

## 与 Gradient Accumulation 的区别

Gradient accumulation 通过多次 micro-batch 累积梯度来扩大 effective batch size，不直接减少单个 micro-batch 的 activation 峰值。

Activation checkpointing 直接减少单次 forward 需要保存的 activation。二者可以组合：

- 用 gradient accumulation 控制 global batch size；
- 用 checkpointing 控制每个 micro-batch 是否放得下。

## 与 FlashAttention 的区别

FlashAttention 通过 IO-aware tiling 避免显式 materialize 完整 attention matrix，减少 attention 部分的显存和带宽开销。Activation checkpointing 则通过 backward recomputation 少存中间激活。

二者解决的问题相邻但不同：

- FlashAttention：优化 attention kernel 的内存访问和中间矩阵；
- Checkpointing：减少 autograd 保存的 activation。

实践中常同时使用。

## 实现注意事项

- **RNG state**：dropout、stochastic depth 等随机 op 必须在 recomputation 时复现相同随机性。
- **Side effect**：checkpointed forward 中不应包含不可重复副作用，例如写日志、更新缓存或修改全局状态。
- **Non-reentrant 实现**：现代框架常提供 reentrant / non-reentrant 两类 checkpoint 实现，二者在 autograd graph、性能和兼容性上可能不同。
- **Fused kernel**：某些 fused attention 或 MLP kernel 已经内置 recomputation 策略，和外层 checkpointing 叠加时需要 profiler 验证。
- **Debugging**：如果只在开启 checkpointing 后出现 NaN，应检查随机性、custom autograd function 和 dtype cast。

## 失败模式与边界

- **训练变慢**：recomputation 增加 FLOPs，吞吐下降。
- **随机性不一致**：dropout 等随机 op 需要正确保存 RNG state，否则 backward 重算与 forward 不一致。
- **调试困难**：activation 不完整保存，排查中间层数值问题更麻烦。
- **收益不均匀**：如果显存瓶颈来自 optimizer states 而不是 activation，checkpointing 帮助有限。
- **实现粒度敏感**：checkpoint 太细可能过慢，太粗可能省不下显存。

## 适用场景

- 长上下文训练；
- 大 micro-batch 训练；
- full fine-tuning 或 continued pretraining；
- activation 显存占比高；
- 已经使用 ZeRO/FSDP 但仍 OOM；
- 愿意用训练速度换更大模型或更长序列。

## 经典论文与资料

- [[sources/papers/2016-training-deep-nets-with-sublinear-memory-cost|Training Deep Nets with Sublinear Memory Cost]]

## 相关概念

- [[training/optimization/training-memory-estimation|Training Memory Estimation]]
- [[training/optimization/mixed-precision|Mixed Precision Training]]
- [[training/distributed-training/fsdp|FSDP]]
- [[training/distributed-training/zero|ZeRO]]
- [[training/distributed-training/sequence-parallel|Sequence Parallel]]
- [[training/distributed-training/context-parallel|Context Parallel]]
- [[inference/attention-acceleration/flash-attention|FlashAttention]]
