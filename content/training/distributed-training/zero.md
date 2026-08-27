---
title: ZeRO
created: 2026-03-22
published: 2026-03-22
modified: 2026-07-06
type: topic
status: mature
area: training
tags:
  - distributed-training
  - zero
---

ZeRO，Zero Redundancy Optimizer，是一种减少 data parallel 训练中冗余模型状态的分布式训练方法。普通 DDP 中，每张 GPU 都复制完整 parameters、gradients 和 optimizer states；ZeRO 将这些状态沿 data parallel 维度切分，使每张 GPU 只保存一部分，从而显著降低单卡显存。

ZeRO 的核心问题是：**数据并行需要复制模型状态，但复制并不是计算所必需的。** 如果能在需要时通过通信恢复完整状态，就可以用通信换显存。

它解决的是 data parallel 维度上的冗余，而不是直接改变模型结构或数学目标。模型仍然执行同样的 forward/backward，区别在于参数、梯度和 optimizer state 在不同 GPU 上如何驻留、同步和恢复。

这套设计来自 [[sources/papers/2019-zero|ZeRO: Memory Optimizations Toward Training Trillion Parameter Models]]。论文将完整方案写成 `Pos`、`Pg`、`Pp` 三个累积阶段，后来通常分别称为 ZeRO-1、ZeRO-2、ZeRO-3。论文评测中的 `ZeRO-100B` 主要启用 `Pos+g` 与 ZeRO-R，不能把它直接等同于完整参数分片阶段的实验结果。

## 显存背景

在 bf16/fp16 + AdamW + fp32 master weights 的 full fine-tuning 中，模型状态常可粗略估为 16 bytes / parameter：

- parameter：2 bytes；
- gradient：2 bytes；
- fp32 master weight：4 bytes；
- Adam first moment：4 bytes；
- Adam second moment：4 bytes。

这些显存组成详见 [[training/optimization/training-memory-estimation|Training Memory Estimation]] 和 [[training/optimization/optimizer-state|Optimizer State]]。

普通 DDP 每卡都保存完整状态：

$$
M_{\text{DDP}} \approx N(b_p+b_g+b_o)
$$

ZeRO 按 stage 逐步切分这些状态。

## ZeRO-1：切分 Optimizer States

ZeRO-1 只切分 optimizer states：

$$
M_{\text{ZeRO-1}}
\approx N(b_p + b_g + \frac{b_o}{D})
$$

其中 $D$ 是 data parallel world size。

特点：

- parameters 每卡完整复制；
- gradients 每卡完整复制；
- optimizer states 被切分；
- 通信复杂度相对较低；
- 适合 optimizer state 是显存瓶颈但参数和梯度仍放得下的场景。

AdamW 的 optimizer states 很大，因此 ZeRO-1 已经能节省明显显存。

## ZeRO-2：切分 Optimizer States 与 Gradients

ZeRO-2 切分 optimizer states 和 gradients：

$$
M_{\text{ZeRO-2}}
\approx N(b_p + \frac{b_g}{D} + \frac{b_o}{D})
$$

特点：

- parameters 每卡完整复制；
- gradients 被 reduce-scatter 后分片保存；
- optimizer states 被切分；
- 比 ZeRO-1 更省显存；
- 仍需要每卡放得下完整参数。

论文将 gradient partitioning 描述为 reduce-scatter 语义：反向传播产生 gradient 后，每个参数区间被 reduce 到负责更新该区间的 rank，而不是让所有 rank 都保留完整 reduced gradient。实际实现会按目标 partition 做 bucketization，以增大通信消息并尝试和 backward 计算重叠。

如果模型参数本身已经接近单卡显存上限，ZeRO-2 可能仍不够。

## ZeRO-3：切分 Parameters、Gradients 和 Optimizer States

ZeRO-3 进一步切分 parameters：

$$
M_{\text{ZeRO-3}}
\approx N\frac{b_p+b_g+b_o}{D}
$$

特点：

- parameters 分片存储；
- forward/backward 需要时 all-gather 当前层参数；
- 用完后释放或 reshard；
- 单卡长期驻留模型状态最小；
- 通信和实现复杂度最高。

ZeRO-3 是训练超大模型或在较少 GPU 上微调大模型的关键技术之一。但它不意味着显存严格除以 $D$，因为 activation、通信 buffer、临时 full parameter、allocator overhead 仍会造成峰值。

ZeRO-3 的运行时可以理解为“分片驻留、按需聚合、用后释放”。这也是它比 ZeRO-1/2 更复杂的原因：每一层何时 all-gather、是否 prefetch、backward 后是否立即 reshard、通信是否与计算重叠，都会影响速度和峰值显存。

从论文的通信账本看，标准 data parallel 的 gradient all-reduce 可以理解为一次 reduce-scatter 加一次 all-gather，总 data movement 约为 $2N$（$N$ 为参数元素数量）。ZeRO-2 用 gradient reduce-scatter 加 updated parameter all-gather，通信量仍约为 $2N$；ZeRO-3 还需要在 forward 和 backward 按需聚合参数，论文给出的总量约为 $3N$，即 baseline 的 1.5 倍。这里是带宽级别的理想化分析，真实耗时还取决于 bucket、拓扑和通信计算重叠。

## 与 FSDP 的关系

[[training/distributed-training/fsdp|FSDP]] 可以看作 PyTorch 生态中与 ZeRO-3 思想高度相近的 fully sharded data parallel 实现。二者都通过切分参数、梯度和 optimizer states 降低 data parallel 冗余。

常见对照：

| ZeRO / DeepSpeed | PyTorch FSDP 概念 | 切分内容 |
|---|---|---|
| ZeRO-1 | 无完全等价常用模式 | optimizer states |
| ZeRO-2 | `SHARD_GRAD_OP` 类似 | gradients + optimizer states |
| ZeRO-3 | `FULL_SHARD` 类似 | parameters + gradients + optimizer states |

实际选型取决于训练框架、生态、checkpoint 格式、offload 支持、调试便利性和团队经验。

一个实践差异是生态入口不同：DeepSpeed ZeRO 通常通过 DeepSpeed engine 和配置文件管理优化器、offload、checkpoint 等能力；FSDP 则更贴近 PyTorch 原生模块包装、state dict 和 distributed API。知识层面可以把二者放在同一个“sharded data parallel”家族中理解，工程层面则需要按训练栈选择。

## ZeRO-Offload 与 ZeRO-Infinity

ZeRO 系列还包括 offload 思路：把部分 optimizer states、parameters 或 activation 迁移到 CPU/NVMe，以突破 GPU 显存限制。这样可以训练更大模型，但会引入 PCIe/NVMe 带宽瓶颈和复杂调度。

offload 的本质是把瓶颈从 GPU memory 转移到异构内存层级：

- GPU HBM：快但贵且容量小；
- CPU DRAM：容量更大但带宽/延迟较差；
- NVMe：容量更大但更慢。

因此 offload 适合“显存容量是硬瓶颈、吞吐可以下降”的场景。若训练目标是最大化大集群吞吐，过度 offload 可能让 GPU 等待数据搬运，反而降低整体效率。

论文中的 ZeRO-R 还处理了 model states 之外的 residual memory：`partitioned activation checkpointing` 沿 model-parallel group 分片保存 activation checkpoint，在 backward 重计算前通过 all-gather 恢复；constant-size buffer 避免 fused buffer 随模型规模增长；预分配连续区域并搬运 checkpoint/gradient，减少 memory fragmentation。这些机制与 ZeRO-DP 正交，也说明 ZeRO/FSDP 解决 model-state memory 后，activation 和临时 workspace 仍需单独预算。

## 与其他并行策略的关系

ZeRO 主要作用在 data parallel 维度。超大规模训练常组合：

- tensor parallel：切分单层矩阵计算；
- pipeline parallel：切分模型层；
- ZeRO / FSDP：切分 data parallel 冗余状态。

这类组合常称为 3D parallelism，相关内容见 [[training/distributed-training/megatron|Megatron]]。

与 tensor parallel / pipeline parallel 相比，ZeRO 不切分单层矩阵乘的数学计算，也不把层分配到不同 pipeline stage。它更像是让 data parallel 不再为每张卡保存完整训练状态。实际大模型训练常用 TP 解决单层计算和参数过大问题，用 PP 解决层数和显存分布问题，再用 ZeRO/FSDP 解决 DP 复制冗余问题。

## Checkpoint 与恢复

ZeRO 训练通常使用 [[training/optimization/checkpoint-sharding|sharded checkpoint]]。优点是每个 rank 只写自己负责的状态，保存和恢复更接近训练时布局；代价是 checkpoint 与 world size、并行策略和框架版本可能耦合。

需要提前明确：

- 是否要保存 optimizer states 以便继续训练；
- 是否需要合并成 inference weights；
- 是否支持改变 GPU 数量 resume；
- sharded checkpoint 的 metadata 是否可靠保存；
- 多节点故障时是否能部分恢复。

大规模训练中，checkpoint 策略应和 ZeRO stage 一起设计，而不是训练结束后再补。

## 失败模式与边界

- **通信开销增加**：状态切分越彻底，需要的 all-gather / reduce-scatter 越多。
- **峰值显存不等于长期状态**：ZeRO-3 仍有临时 full layer parameter 和 buffer。
- **Activation 不自动切分**：activation memory 仍需 checkpointing、sequence parallel 或减小 micro-batch。
- **Checkpoint 复杂**：sharded checkpoint 保存、加载、合并和迁移需要额外处理。
- **小模型收益有限**：模型不大时，通信开销可能抵消显存收益。

论文还展示了一个容易误读的现象：`Pos+g` 随 data parallel degree 增大而降低单卡模型状态显存，使单卡可以容纳更大的 batch，从而在特定受显存约束的区间出现 super-linear speedup。这不是普遍的扩展规律；当 batch 已超过优化上的合适范围，继续增大 batch 可能不再带来吞吐和收敛收益。

## 经典论文与资料

- [[sources/papers/2019-zero|ZeRO: Memory Optimizations Toward Training Trillion Parameter Models]]
- [[sources/papers/2021-zero-infinity|ZeRO-Infinity]]

## 相关概念

- [[training/optimization/training-memory-estimation|Training Memory Estimation]]
- [[training/optimization/optimizer-state|Optimizer State]]
- [[training/distributed-training/torch-distributed|torch.distributed]]
- [[training/distributed-training/fsdp|FSDP]]
- [[training/distributed-training/data-parallel|Data Parallel]]
- [[training/distributed-training/tensor-parallel|Tensor Parallel]]
- [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]
- [[training/optimization/checkpoint-sharding|Checkpoint Sharding]]
