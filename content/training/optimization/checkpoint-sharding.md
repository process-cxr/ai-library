---
title: Checkpoint Sharding
created: 2026-07-06
published: 2026-07-06
modified: 2026-07-06
type: topic
status: growing
area: training
tags:
  - training-optimization
  - checkpoint
  - distributed-training
---

Checkpoint Sharding 指在分布式训练中把 checkpoint 按 rank、并行维度或张量切片保存，而不是把完整模型和训练状态合并成一个单体文件。它是大模型训练工程中的基础能力，因为参数、梯度、optimizer states、scheduler、random states 和 dataloader 状态的总量常远超单机内存或单文件管理的舒适范围。

需要区分两个问题：

- **训练能否恢复**：checkpoint 必须保存足够的训练状态和并行布局信息；
- **权重能否发布或推理**：通常只需要模型权重，且可能需要合并或转换格式。

Sharded checkpoint 主要服务前者；发布权重通常需要额外 export / merge。

## Checkpoint 保存什么

完整训练 checkpoint 通常包括：

- model parameters；
- optimizer states，例如 AdamW 的 $m$、$v$ 和 master weights；
- gradients 或 gradient accumulation 状态，取决于框架；
- lr scheduler state；
- mixed precision scaler state；
- random states，包括 Python、NumPy、PyTorch CPU/GPU RNG；
- dataloader / sampler position；
- consumed samples / consumed tokens；
- tokenizer、数据版本和训练 config；
- distributed parallel metadata；
- framework / kernel version 相关 metadata。

若只保存 model weights，则无法严格恢复训练。原因是 optimizer state、学习率进度、随机状态和数据读取位置都会影响后续更新轨迹。

## 为什么需要 Sharding

以 bf16 full training with AdamW 为例，模型参数可能只占 $2N$ bytes，但训练状态可能达到 $12N$ 到 $16N$ bytes。对 70B 模型，完整训练态 checkpoint 可能达到 TB 级别。

单体 checkpoint 会带来多重问题：

- 保存时需要将各 rank 状态聚合到少数进程，造成内存峰值；
- 单文件写入慢，失败后恢复成本高；
- 多节点训练中网络聚合成为瓶颈；
- optimizer states 远大于推理权重；
- 改变并行策略或 world size 时需要昂贵转换。

Sharded checkpoint 让每个 rank 保存自己负责的状态分片，避免在保存路径上重建完整训练状态。

## 常见分片维度

Checkpoint sharding 可能沿多个并行维度组织：

- **Data Parallel / ZeRO / FSDP shard**：按 data-parallel ranks 切分 parameters、gradients 和 optimizer states；
- **Tensor Parallel shard**：保存按 hidden dimension、heads 或 MLP dimension 切开的权重；
- **Pipeline Parallel shard**：不同 pipeline stage 保存不同层；
- **Expert Parallel shard**：MoE 中不同 expert 分布到不同 ranks；
- **Context Parallel metadata**：通常不直接切权重，但可能影响训练状态、position / mask 相关配置和数据布局。

在 Megatron-style 训练中，一个 rank 的 checkpoint 往往同时对应 TP rank、PP rank 和 DP rank。恢复时必须知道每个 shard 属于哪个并行坐标。

## 训练态 Checkpoint 与推理权重

训练态 checkpoint 和推理权重的目标不同。

训练态 checkpoint 关注：

- 能否从同一步继续训练；
- optimizer states 是否完整；
- random states 是否恢复；
- dataloader position 是否一致；
- 并行分片是否与当前 world size 匹配；
- 保存和恢复是否足够快。

推理权重关注：

- 是否只包含 model parameters；
- 是否合并 TP/PP/FSDP shards；
- dtype 是否符合部署要求；
- tokenizer 和 config 是否完整；
- 是否兼容目标 serving 框架。

因此，大规模训练通常同时设计两条路径：

```text
training checkpoint:
  sharded, complete, resumable

inference export:
  merged or deployment-sharded, lightweight, portable
```

把二者混用容易造成恢复失败或部署格式不兼容。

## 与 ZeRO / FSDP 的关系

[[training/distributed-training/zero|ZeRO]] 和 [[training/distributed-training/fsdp|FSDP]] 会在训练时切分模型状态。checkpoint 保存通常也保持这种 sharded layout：

- ZeRO-1：optimizer states 分片；
- ZeRO-2：optimizer states 和 gradients 分片；
- ZeRO-3 / FSDP full shard：parameters、gradients 和 optimizer states 分片。

该保存方式高效，但会引入耦合：

- checkpoint 可能依赖原 world size；
- 改变 ZeRO stage / FSDP strategy 可能需要转换；
- 从 sharded state dict 导出 full state dict 可能需要聚合；
- optimizer state 的 key、flattening 和 parameter order 必须稳定。

PyTorch FSDP、DeepSpeed ZeRO、Megatron Distributed Checkpoint 等实现对 state dict 和 metadata 的约定不同。跨框架迁移通常不能只复制文件，需要显式转换。

## 与 Tensor / Pipeline Parallel 的关系

TP 和 PP 会改变权重本身的组织方式。

Tensor parallel 下，一个线性层权重可能被按 column 或 row 切分：

```text
W = [W_0, W_1, ..., W_{T-1}]
```

每个 TP rank 保存自己的 slice。导出推理权重时，可能需要按正确维度 concat 或转换为 serving 框架的 tensor-parallel 格式。

Pipeline parallel 下，不同 stage 保存不同层。恢复训练时，每个 stage 只加载自己负责的层；导出完整模型时需要按层序合并。

TP/PP checkpoint 的关键是保存并行坐标：

```text
global rank
  → tensor parallel rank
  → pipeline parallel rank
  → data parallel rank
```

缺失这些 metadata 会让 checkpoint 难以可靠恢复。

## Dataloader 与随机状态

严格 resume 不只需要模型和 optimizer。数据读取位置同样重要。

需要保存或可推导：

- 当前 consumed samples / tokens；
- dataset shard；
- sampler epoch；
- shuffle seed；
- worker seed；
- packed data offset；
- streaming dataset cursor；
- gradient accumulation 内部进度。

如果 dataloader 状态不一致，恢复后的训练会看到不同 batch。短期可能不报错，但 ablation、loss spike 排查和精确复现实验都会变困难。

对大规模 mid-training，应以 consumed tokens 作为核心进度指标，并让 checkpoint、日志、数据版本和评测结果都能对齐到同一 token step。

## 保存频率与保留策略

Checkpoint 频率需要在可靠性和成本之间权衡：

- 保存过频：浪费训练时间、存储和 I/O；
- 保存过稀：故障后重跑成本高；
- 只保留最新：无法回滚到健康状态；
- 保留过多：存储成本不可控。

常见策略：

- 高频保存 lightweight checkpoint；
- 低频保存 full training checkpoint；
- 保留最近 $k$ 个 checkpoint；
- 额外保留关键里程碑 checkpoint；
- loss spike 或数据切换前后保留 checkpoint；
- 定期导出推理权重用于评测。

大规模训练中，checkpoint I/O 可能显著影响 MFU / tokens per day，因此保存策略是训练 recipe 的一部分，而不是训练后的杂项配置。

## Checkpoint 转换与迁移

常见转换包括：

- sharded training checkpoint → merged inference weights；
- TP size $T_1$ → TP size $T_2$；
- PP size $P_1$ → PP size $P_2$；
- FSDP full shard → full state dict；
- ZeRO checkpoint → HuggingFace weights；
- bf16 training weights → fp16 / fp32 / quantized serving weights；
- optimizer state discard，用于只保留模型权重。

转换时必须验证：

- tensor shape；
- concat / split dimension；
- tied embeddings / LM head 是否一致；
- MoE expert order；
- tokenizer 和 config；
- 数值 checksum 或小 batch logits 对齐。

对源码学习而言，checkpoint 转换代码是理解训练框架权重布局的关键入口。

## 常见失败模式

- **只保存 model weights**：无法恢复 optimizer、scheduler 和数据位置。
- **metadata 不完整**：不知道 shard 属于哪个 TP/PP/DP rank。
- **world size 改变后直接 resume**：并行布局不匹配，导致 shape 或 optimizer state 错误。
- **optimizer state 丢失或错位**：训练能启动，但 loss 曲线异常。
- **dataloader position 未恢复**：实验不可复现，数据重复或跳过。
- **checkpoint I/O 阻塞训练**：保存时间过长，降低整体吞吐。
- **推理导出未验证**：合并维度错误但文件能加载，最终 logits 不一致。
- **版本耦合**：框架升级后 state dict key 或 flattening 策略变化。

## 实践检查清单

训练前应明确：

- 保存 full training checkpoint 还是 model-only checkpoint；
- checkpoint 是否 sharded；
- sharding metadata 保存在哪里；
- 是否支持改变 world size resume；
- 是否需要定期导出推理权重；
- optimizer state 是否必须保留；
- dataloader / sampler 状态如何恢复；
- 保存频率、保留数量和存储预算；
- checkpoint 写入失败时训练如何处理；
- 是否有小规模 restore test。

真正可靠的 checkpoint 策略需要在训练早期就做 restore 演练。等到首次故障后再验证 checkpoint，往往已经太晚。

## 相关概念

- [[training/optimization/training-memory-estimation|Training Memory Estimation]]
- [[training/optimization/optimizer-state|Optimizer State]]
- [[training/distributed-training/zero|ZeRO]]
- [[training/distributed-training/fsdp|FSDP]]
- [[training/distributed-training/tensor-parallel|Tensor Parallel]]
- [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]
- [[training/distributed-training/megatron|Megatron 与 3D 并行]]
- [[training/data-engineering/distributed-dataloader|Distributed Dataloader]]
- [[training/scaling/training-budget|Training Budget]]
