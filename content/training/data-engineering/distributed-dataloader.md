---
title: Distributed Dataloader
created: 2026-07-06
published: 2026-07-06
modified: 2026-07-06
type: topic
status: growing
area: training
tags:
  - data-engineering
  - distributed-training
  - dataloader
---

Distributed Dataloader 指在多 GPU / 多节点训练中把数据稳定、高吞吐、可复现地送入每个训练 rank 的数据加载系统。它不仅是 PyTorch `DataLoader` 的并行版本，还包括 dataset sharding、shuffle、packing、batching、worker seed、resume position、数据版本和吞吐监控等训练工程问题。

大模型训练中，dataloader 的质量会直接影响训练效率和实验可信度。GPU 等待数据会降低 tokens/sec；数据 shard 不均会造成 straggler；resume position 错误会导致样本重复或跳过；packing mask 错误会污染 loss。

## 基本目标

Distributed dataloader 需要同时满足四个目标：

1. **无重复或可控重复**：同一训练 step 内，不同 data-parallel ranks 不应意外读取同一批样本。
2. **负载均衡**：各 rank 的 batch 计算量尽量接近，避免长样本或复杂 packing 造成 straggler。
3. **高吞吐**：数据解压、tokenization、packing、host-to-device copy 不应让 GPU 长时间等待。
4. **可恢复**：checkpoint resume 后，训练应从正确数据位置继续。

这些目标之间存在权衡。例如，按长度 bucketing 能提高吞吐和减少 padding，但可能改变 batch 内数据分布；严格全局 shuffle 提升随机性，但会增加索引和状态管理成本。

## 数据并行下的 Sharding

在 [[training/distributed-training/data-parallel|Data Parallel]] 中，每个 data-parallel rank 处理不同数据。设 data parallel world size 为 $D$，全局 batch 可写为：

$$
B_{\mathrm{global}} = B_{\mu} \times G \times D
$$

其中 $B_{\mu}$ 是每个 rank 的 micro-batch size，$G$ 是 gradient accumulation steps。

Dataloader 需要保证每个 optimizer step 中，$D$ 个 ranks 读到不同的数据 shard。常见方式包括：

- map-style dataset 使用 `DistributedSampler`；
- iterable / streaming dataset 按 rank 做 stride 或 contiguous partition；
- 预先把数据切成多个 shards，每个 rank 读取不同 shard；
- 使用全局样本 ID 或 token offset 追踪 consumed data。

当同时存在 TP / PP / CP 时，只有 data-parallel 维度处理不同样本。TP/PP/CP group 内的多个 ranks 通常共同处理同一 micro-batch 的不同模型或序列分片，因此 dataloader sharding 必须基于 DP group，而不是简单基于 global rank。

## Shuffle 与随机性

Shuffle 的目标是打散数据顺序，降低局部相关性。分布式训练中需要保证：

- 各 rank 的 shuffle 一致地覆盖全局数据；
- 不同 epoch 或 token cycle 的顺序可控变化；
- resume 后不会从错误位置重新 shuffle；
- worker-level random augmentation 或 packing 随机性可复现。

常见状态包括：

- global seed；
- epoch；
- sampler position；
- worker id；
- rank id；
- dataset shard id；
- consumed samples / tokens。

若这些状态没有进入 checkpoint，训练恢复后可能读到不同数据。对大规模训练而言，此类差异常常不会立刻报错，但会让 loss 曲线、ablation 和问题定位变得不可解释。

## Map-style 与 Streaming Dataset

Map-style dataset 有明确索引：

```text
sample = dataset[i]
```

它便于随机访问、全局 shuffle、去重记录和精确 resume，但需要索引文件和本地或共享存储支持。

Streaming dataset 按顺序读取数据流：

```text
for sample in dataset:
    ...
```

它适合超大规模数据、远程对象存储和持续扩展数据源，但 resume、shuffle 和去重更复杂。常见做法是把数据预切为 shards，并在 shard 级别 shuffle，再在 shard 内顺序读取或局部 shuffle。

对百 B 级 token 训练，常见工程形态是：

```text
raw data
  → cleaned / filtered documents
  → tokenized shards
  → indexed binary dataset or streaming shards
  → distributed sampler
  → packing / batching
  → GPU batch
```

## Tokenization 与 Packing 位置

Dataloader 可在不同阶段处理 tokenization 和 [[training/data-engineering/packing|Packing]]。

离线 tokenization 的优点：

- 训练时 CPU 压力低；
- token 数和样本长度可提前统计；
- 易于做 packing、bucketing 和数据配比；
- resume 可基于 token offset。

在线 tokenization 的优点：

- 数据处理灵活；
- 便于动态模板、augmentation 或 synthetic data；
- 不需要为每个 tokenizer 保存完整副本。

大规模 pre-training / mid-training 通常更偏向离线 tokenization，因为训练吞吐和可复现性更重要。SFT / RL 场景可能更多使用在线模板渲染和动态 batch。

Packing 可离线完成，也可在线完成。离线 packing 吞吐稳定，但灵活性较低；在线 packing 能动态适配 sequence length、数据 mixture 和过滤规则，但需要更强 CPU / host pipeline。

## Length Bucketing 与负载均衡

变长样本会造成 padding waste 和 rank 间计算不均。常见策略包括：

- 按长度 bucketing；
- best-fit packing；
- dynamic batching；
- 限制每 batch 最大 token 数；
- 将超长样本单独分桶；
- 对 agent trajectory 保留边界后再做长度分桶。

对 Transformer 训练，batch 的真实成本更接近 token 数，而不是样本数。对长上下文或 agent 轨迹，单条样本可能达到数十万 tokens，若直接按样本数分配，会造成严重 straggler。

因此，distributed dataloader 通常需要记录：

- tokens per rank；
- padding ratio；
- packed sequence length；
- number of samples per packed sequence；
- domain / quality / length distribution；
- 每个 rank 的 data loading time 和 step time。

## Agent 轨迹数据的特殊性

Agent 轨迹数据包含 system instruction、tool schema、assistant reasoning、tool call、observation、代码片段、网页内容、日志和最终回答。它比普通文档更依赖边界和角色结构。

Distributed dataloader 处理这类数据时需要特别关注：

- 不要随机从轨迹中间切出缺失 tool schema 的片段；
- role token、tool call wrapper 和 observation boundary 必须保留；
- loss mask 是否覆盖正确的 assistant / reasoning / action tokens；
- 超长 observation 是否截断、压缩或降采样；
- trajectory id、step id 和数据质量标签是否进入 metadata；
- 按 task domain、workflow type、quality level 做 mixture 控制；
- 长轨迹样本是否造成 rank 间负载不均。

对 agentic mid-training，dataloader 不只是喂 token 的组件，还承担保持 trajectory 结构、控制 mixture 和提供可追踪 metadata 的职责。

## Resume 与 Checkpoint 对齐

分布式 dataloader 必须与 [[training/optimization/checkpoint-sharding|Checkpoint Sharding]] 协同。

需要保存或可恢复：

- consumed samples；
- consumed tokens；
- current epoch / shard cycle；
- sampler state；
- streaming cursor；
- random seed；
- packing buffer state；
- data mixture schedule；
- dataset version。

如果使用 gradient accumulation，还要明确 checkpoint 保存发生在 accumulation 边界还是中间。中间保存需要恢复已经累积但尚未 step 的梯度状态，否则应只在 optimizer step 边界保存。

对超大规模训练，常用 consumed tokens 作为主进度单位，因为样本长度差异很大，sample count 不能稳定代表训练量。

## 吞吐瓶颈

Dataloader 可能成为瓶颈的原因包括：

- 数据存储带宽不足；
- 小文件过多；
- 解压或反序列化慢；
- 在线 tokenization CPU 不足；
- packing 算法过重；
- host-to-device copy 未 overlap；
- worker 数配置不合理；
- 多节点同时读取导致对象存储限流；
- rank 间样本长度不均造成 straggler。

诊断时应同时看：

- GPU utilization；
- data loading time；
- batch preparation time；
- host memory；
- disk / network throughput；
- per-rank step time；
- tokens/sec 和 effective tokens/sec。

只看 raw tokens/sec 不够，还要结合 padding ratio、loss mask ratio 和有效训练 token 数。

## 常见失败模式

- **按 global rank 错误切数据**：TP/PP ranks 读了不同样本，导致模型并行组输入不一致。
- **resume 后数据重复或跳过**：sampler / cursor / seed 未保存完整。
- **长度分布不均**：某些 rank 长期处理更长样本，形成 straggler。
- **packing mask 错误**：样本之间 attention leakage 或 loss mask 错误。
- **数据 mixture 漂移**：按长度 bucketing 后 domain 或 quality 分布被改变。
- **在线处理拖慢训练**：CPU tokenization / packing 跟不上 GPU。
- **metadata 丢失**：loss spike 后无法定位数据来源。
- **streaming shuffle 不充分**：局部数据分布过于集中，影响训练稳定性。

## 实践检查清单

训练前应确认：

- DP group 如何映射到数据 shard；
- TP/PP/CP ranks 是否共享同一 micro-batch；
- tokenization 是离线还是在线；
- packing 是离线还是在线；
- batch 按 sample count 还是 token count 控制；
- length bucketing 是否改变数据分布；
- loss mask 和 attention mask 是否有单元测试；
- dataloader state 是否进入 checkpoint；
- 是否记录 sample id / shard id / token offset；
- 是否有 per-rank throughput 监控；
- 是否能从任意 checkpoint 做 restore test。

## 相关概念

- [[training/data-engineering/data-engineering|Data Engineering]]
- [[training/data-engineering/packing|Packing]]
- [[training/pretraining/data-mix|Data Mix]]
- [[training/distributed-training/data-parallel|Data Parallel]]
- [[training/distributed-training/tensor-parallel|Tensor Parallel]]
- [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]
- [[training/distributed-training/context-parallel|Context Parallel]]
- [[training/optimization/checkpoint-sharding|Checkpoint Sharding]]
- [[training/scaling/training-budget|Training Budget]]
