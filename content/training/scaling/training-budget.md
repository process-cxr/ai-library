---
title: Training Budget
created: 2026-03-28
published: 2026-03-28
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - scaling
  - budget
---

Training Budget 指训练一个模型所需的总资源预算，包括算力、显存、训练时间、数据处理、checkpoint 存储、评测和失败重跑成本。它回答的问题不是“模型最终 loss 会是多少”，而是“在给定目标和约束下，训练是否可执行、需要多少资源、风险在哪里”。

预算估算的核心是把目标拆成几类硬约束：总 FLOPs 决定训练成本下界，单卡显存决定配置是否能跑，wall-clock time 决定项目周期，数据和 checkpoint 决定工程吞吐，评测与失败重跑决定真实预算余量。

## 预算组成

| 预算项 | 主要变量 | 相关笔记 |
|---|---|---|
| Compute | 参数量、token 数、FLOPs/token、训练步数 | [[training/scaling/model-data-compute|Model Data Compute]] |
| GPU Memory | 参数、梯度、optimizer states、activation、并行策略 | [[training/optimization/training-memory-estimation|Training Memory Estimation]] |
| Wall-clock Time | GPU 数、MFU、通信效率、checkpoint 频率 | [[training/distributed-training/|Distributed Training]] |
| Data | 数据收集、清洗、去重、过滤、packing | [[training/data-engineering/|Data Engineering]] |
| Checkpoint Storage | 权重、optimizer state、保存频率、保留策略 | [[training/optimization/training-memory-estimation|Training Memory Estimation]] |
| Evaluation | benchmark、人工评测、模型比较、回归测试 | [[application/evaluation/evaluation|Evaluation]] |

## Compute Budget

对 dense decoder-only Transformer，训练 FLOPs 的常用粗估是：

$$
C_{\text{train}} \approx 6ND
$$

其中：

- $N$ 是非 embedding 近似参数量；
- $D$ 是训练 token 数；
- 系数 6 来自 forward 与 backward 的粗略 FLOPs 估算；
- 实际值会受 attention、sequence length、embedding、MoE、activation recomputation 和实现细节影响。

例如 7B 模型训练 1T tokens：

$$
C_{\text{train}}\approx 6 \times 7\times10^9 \times 10^{12}
=4.2\times10^{22}\text{ FLOPs}
$$

这个数字只描述数学计算量，不代表实际 GPU 时间。实际训练时间还要乘上硬件利用率、通信开销、数据加载、checkpoint 保存和故障恢复成本。

## Wall-clock Time 与 MFU

设集群共有 $G$ 张 GPU，每张 GPU 的理论峰值为 $F_{\text{peak}}$ FLOPs/s，模型 FLOPs 利用率为 MFU，则理想 wall-clock time 可粗估为：

$$
T_{\text{wall}}
\approx
\frac{C_{\text{train}}}
{G \cdot F_{\text{peak}} \cdot \text{MFU}}
$$

MFU，Model FLOPs Utilization，反映训练实际完成的模型计算量占硬件理论峰值的比例。它受 kernel、batch size、sequence length、通信、并行策略和数据管线影响。大模型训练中，MFU 往往比理论峰值更能说明预算现实：同样 GPU 数量下，MFU 从 0.25 提升到 0.40，训练周期会明显缩短。

需要注意：

- activation checkpointing 会增加 recomputation FLOPs，降低 step throughput，但可能允许更大 batch 或更长上下文；
- ZeRO-3 / FSDP 降低显存，但 all-gather / reduce-scatter 会增加通信时间；
- 小 batch、短 sequence 或过细并行切分可能导致 GPU 利用率偏低；
- 多节点训练还要考虑网络拓扑、straggler、故障率和 checkpoint 恢复时间。

## 与显存估算的关系

显存预算是 training budget 的约束条件之一。如果单卡显存不足，训练可能需要改变：

- micro-batch size；
- sequence length；
- precision；
- activation checkpointing；
- ZeRO / FSDP stage；
- tensor / pipeline parallel；
- full fine-tuning vs LoRA / QLoRA；
- checkpoint 保存策略。

详细公式和示例放在 [[training/optimization/training-memory-estimation|Training Memory Estimation]]，这里保留总体预算视角。

显存预算与算力预算经常相互牵制。例如减小 micro-batch 可以避免 OOM，但可能降低 MFU；启用 activation checkpointing 可以训练更长上下文，但会增加每步计算；采用 ZeRO-3 可以训练更大模型，但通信峰值和 checkpoint 复杂度会上升。因此预算不能只看“总 FLOPs 是否够”，也不能只看“单卡显存是否够”。

## 数据与 I/O 预算

训练预算还必须包含数据工程成本。大规模训练通常需要：

- 原始语料存储；
- 清洗、去重、质量过滤的中间产物；
- tokenizer 后的 tokenized dataset；
- packing / sharding 后的训练样本；
- 数据加载缓存和多节点分发；
- 数据版本、比例和过滤规则的追踪。

数据预算不是纯存储问题。若数据管线吞吐不足，GPU 会等待 dataloader；若清洗规则不可复现，实验结果无法解释；若数据版本与 checkpoint 不对应，训练回滚和 ablation 会变得困难。

## Checkpoint 与恢复预算

Checkpoint 预算包括磁盘容量、保存时间、加载时间和保留策略。完整训练 checkpoint 往往包含 model weights、optimizer states、scheduler、random states、dataloader position 和 sharded metadata。对于 AdamW full training，optimizer checkpoint 可能远大于推理权重。

常见策略：

- 高频保存 lightweight checkpoint，用于短期故障恢复；
- 低频保存 full training state，用于长期 resume；
- 单独导出 inference weights，用于评测和对外发布；
- 对旧 checkpoint 做 retention policy，避免存储无限增长；
- 在大规模训练前演练一次从 checkpoint 恢复。

如果训练需要数周，checkpoint 不只是保险，而是预算的一部分：保存太频繁会拖慢训练，保存太少会放大故障损失。

## Evaluation 与 Ablation 预算

评测预算经常被低估。一次训练不应只预算主 run，还应预算：

- tokenizer、data mix、learning rate、batch size 的小规模 ablation；
- validation loss 和 downstream benchmark 的周期性评测；
- safety、toxicity、memorization、copyright 或 domain-specific eval；
- 不同 checkpoint 的模型选择；
- 失败 run 的诊断和重跑。

从项目管理角度看，主训练最好只占总预算的一部分。剩余预算用于试跑、消融、评测、故障恢复和后训练，否则主 run 一旦暴露配置问题，就没有资源修正。

## 快速估算顺序

1. 先用 scaling / compute 估计目标参数量和 token 数。
2. 估计训练 FLOPs 和可用 GPU 总算力。
3. 用 MFU 粗估 wall-clock time。
4. 用显存公式判断每卡是否能放下模型状态和 activation。
5. 估算 checkpoint、数据缓存和日志存储。
6. 预留失败重跑、ablation 和评测成本。

训练预算不是一次性算准，而是随着模型规模、数据质量、并行策略和训练稳定性不断修正。

## 常见误区

- **只按参数量预算**：参数量只决定部分显存和部分 FLOPs，token 数、sequence length、optimizer、activation 和并行策略同样关键。
- **把理论峰值当实际吞吐**：GPU peak FLOPs 不能直接换算训练天数，必须乘以 MFU。
- **忽略失败成本**：长时间训练的故障率、checkpoint 损坏、数据 bug 和 loss spike 都可能吞掉大量预算。
- **忽略评测成本**：没有稳定评测，训练完成也难以判断模型是否真的更好。
- **把预算视为静态数字**：训练预算应随 pilot run 和 profiler 结果更新，而不是一次性写死。

## 相关概念

- [[training/optimization/training-memory-estimation|Training Memory Estimation]]
- [[training/scaling/model-data-compute|Model Data Compute]]
- [[training/scaling/compute-optimal|Compute Optimal]]
- [[training/distributed-training/fsdp|FSDP]]
- [[training/distributed-training/zero|ZeRO]]
