---
title: 03 Data Protocol and Batch Lifecycle
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-03
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - dataflow
  - batching
---

本页把 verl V1 中一次 sync GRPO update 看成一条字段生命周期：数据集中的 prompt 如何进入 TransferQueue，AgentLoop 如何产生 trajectory，reward、log-prob 和 advantage 如何逐步写回，以及 actor update 完成后这些字段如何被消费和清理。

重点不是罗列所有字段，而是区分四种职责：训练数据协议、TransferQueue 的存储与状态管理、算法字段的逐步补全，以及 dispatch 到 model engine 时的 batch 形态转换。

## Data Representation

### Dataset row

`RLHFDataset` 读取 parquet、json 或 jsonl 数据，并在 `__getitem__` 中返回面向 agent loop 的原始消息，而不是提前完成最终 chat-template tokenization。一个 row 主要可能包含：

```text
raw_prompt       原始 message list
data_source      数据来源标识
extra_info       index / tools_kwargs / interaction_kwargs 等运行参数
dummy_tensor     兼容旧 DataProto.batch 的占位 tensor
multimodal data  图像、视频或音频等可选输入
```

超长 prompt 的过滤和长度计算发生在 dataset 预处理阶段；具体 prompt 如何套用 chat template 则由 AgentLoop / processor 路径决定。`collate_fn` 将 tensor 字段沿 batch 维 stack，将非 tensor 字段转为 object array。

源码入口：

- [`RLHFDataset` 初始化与文件读取](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/dataset/rl_dataset.py#L72)
- [`RLHFDataset.__getitem__`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/dataset/rl_dataset.py#L386)
- [`collate_fn`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/dataset/rl_dataset.py#L41)

### DataProto、TensorDict 和 KVBatchMeta

- `DataProto` 是通用函数间数据协议，包含 `batch: TensorDict`、`non_tensor_batch` 和 `meta_info`；
- `TensorDict` 是 tensor 容器，具有相同 batch size 的张量放在其中，可以整体索引、切分和移动；
- `KVBatchMeta` 是 V1 controller 侧的轻量批次描述，主要记录 TransferQueue 的 `partition_id`、trajectory `keys` 和 `tags`；
- TransferQueue 将 tag 与 value 分开存储：tag 用于状态、长度、global step 和版本元数据，value 承载 `prompts`、`responses`、mask、log-prob 等真实字段。

因此 V1 中常见的路径不是一个 batch 对象贯穿所有阶段，而是：controller 持有 `KVBatchMeta`，计算阶段按 keys 从 TQ 取出 `TensorDict`，必要时转换成 `DataProto` 或 padded tensor，再把新增字段写回 TQ。

源码入口：

- [`DataProto`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/protocol.py#L318)
- [`ReplayBuffer` 数据模型](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/replay_buffer.py#L63)
- [TransferQueue 工具封装](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/transferqueue_utils.py)

### Protocol 与 backend 的分界

从 dataset 到 advantage 写回的主要字段 contract 不依赖 actor 使用 FSDP 还是 Megatron：`uid`、trajectory key、`response_mask`、reward、old/reference log-prob 和 advantage 的 producer / consumer 关系由 trainer、AgentLoop 和 TransferQueue 决定。

```text
backend-independent:
  dataset -> rollout -> TQ -> reward -> advantage

backend-dependent:
  KVBatchMeta -> worker dispatch -> model engine
              -> micro-batches -> forward/backward
              -> gradient synchronization -> optimizer step
```

这条边界位于 `ActorRolloutRefWorker.update_actor -> TrainingWorker.train_mini_batch -> actor.train_batch`。同一组 trajectory fields 可以送入不同 backend，但本地 batch 布局、需要参与同一次计算的 ranks 和通信方式会改变。

## Field Lineage

### 总体字段链

```text
dataset row
  -> raw_prompt / extra_info / index
  -> uid
  -> prompt tag in TransferQueue
  -> trajectory keys: {uid}_{session_id}_{index}
  -> prompts / responses / response_mask / loss_mask
  -> input_ids / position_ids / attention context
  -> rm_scores
  -> rollout_log_probs / old_log_probs / entropy
  -> ref_log_prob
  -> token_level_scores / token_level_rewards
  -> advantages / returns
  -> actor mini-batch
  -> metrics
  -> tq.kv_clear
```

### Dataset 到 prompt metadata

`_fetch_one_gen_batch` 取出 dataloader batch 后，为每个 raw prompt 创建 UUID 并写入 `uid`。`_submit_batch_to_rollout` 将这些 uid 作为 prompt key 写入 `train` partition 的 TransferQueue，并添加：

```text
is_prompt=True
status=pending
global_steps=current trainer step
```

sync 模式只把 prompt tag 写入 TQ，完整 prompt fields 随调用传给 AgentLoop；async 模式还会把可用于恢复的 prompt fields 存入 TQ。这体现了 sync / async 在数据持久化和恢复语义上的差异。

源码入口：

- [`_fetch_one_gen_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1373)
- [`_submit_batch_to_rollout`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1403)

### AgentLoop 到 trajectory value

数据层级需要先固定为：

```text
uid                         一个 dataset prompt group
uid + session_id            该 prompt 的一次独立 rollout / episode
uid + session_id + index    该 session 返回的一个 AgentLoopOutput
```

`rollout.n` 决定每个 uid 的 session 数。一个 tool-agent session 可以包含多轮模型生成和环境 observation，但仍是一条 rollout；当前内置 single-turn / tool AgentLoop 通常各返回一个完整 output，多个 `index` 主要服务于可返回 output list 的自定义实现。

AgentLoop output 的基础字段为：

```text
prompt_ids
response_ids
response_mask
response_logprobs (optional)
reward_score (optional)
num_turns
extra_fields
```

TQ postprocess 为每个 output 构造：

```text
prompts       prompt_ids tensor
responses     response_ids tensor
input_ids     concat(prompts, responses)
position_ids  derived from input_ids / attention / multimodal inputs
loss_mask     response_mask 的当前实现别名
multi_modal_inputs
```

每个 output 的 tag 还记录 `prompt_len`、`response_len`、`seq_len`、生成时的 `global_steps`，以及 trajectory 的 `min_global_steps` / `max_global_steps`。后两类字段为后续 off-policy staleness 判断提供版本边界。

源码入口：

- [`AgentLoopOutput` 字段定义](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/agent_loop.py#L88)
- [`_agent_loop_postprocess` 字段写入](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L150)

### response mask 的语义

在多轮 tool-use trajectory 中，`responses` 是完整交互 token 序列，但不等于所有 token 都由 policy 产生：

```text
response_ids:  LLM generation + tool response / observation
response_mask: 1 for LLM generation
               0 for tool response / observation / padding
loss_mask:     当前实现直接等于 response_mask
```

它至少影响三处：

1. actor policy loss 只应在有效 policy token 上累计；
2. GRPO 的 scalar advantage broadcast 后需要乘 response mask；
3. padded batch 转成 dense tensor 时不能把 observation 或 padding 误当成 action。

源码入口：

- [`AgentLoopOutput.response_mask`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/agent_loop.py#L91)
- [`generate_sequences` 的 mask 说明](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/agent_loop.py#L537)
- [`loss_mask = response_mask`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L198)

### reward、log-prob 与 advantage

在 rule-based reward 路径中，`RewardLoopWorker.compute_score` 调用注册的 reward manager，并返回 trajectory-level `reward_score`；`RewardLoopManager.compute_rm_score` 将结果组装为 `rm_scores`，trainer 再写回原 trajectory keys。

`_compute_old_log_prob` 有两种语义：

```text
normal mode:
  actor engine recompute -> log_probs -> old_log_probs + entropy

bypass mode:
  rollout_log_probs -> old_log_probs
```

`_compute_ref_log_prob` 通过独立 reference worker 或 actor no-LoRA 路径写入 `ref_log_prob`。`_compute_advantage` 随后读取 `uid`、`response_mask`、`rm_scores`、`rollout_log_probs`、`old_log_probs`、`ref_log_prob` 和可选 `values`，在 padded `DataProto` 中计算，再将 `advantages`、`returns` 转换回 nested representation，写回 TQ。

源码入口：

- [`RewardLoopWorker` 的 reward 分支](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/reward_loop/reward_loop.py#L93)
- [`RewardLoopManager.compute_rm_score`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/reward_loop/reward_loop.py#L343)
- [`_compute_old_log_prob`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1541)
- [`_compute_advantage`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1650)

## TransferQueue Lifecycle

### Prompt 状态

prompt key 与 trajectory key 是两套对象。prompt key 只用于跟踪一个 GRPO group 是否完成：

```text
pending  已取出，尚未启动 session
running  该 uid 的 session 正在执行
finished 所有 session 成功结束
failure  至少一个 session 出错
```

`ReplayBuffer` 周期性同步 TQ metadata，将 prompt 分入上述状态集合；只有 terminal group 才能进入 sampleable 集合。trajectory key 携带具体 output 的 fields 和长度 / version tags。

### sync sample 的实际语义

`ReplayBuffer.sample` 在 sync mode 下：

1. 同步 TQ metadata；
2. 处理 terminal group 的过滤或失败状态；
3. 等待可采样 prompt group 数量达到 `batch_size`；
4. 按 prompt global step 选择 group，并物化对应 trajectory keys；
5. 返回 `KVBatchMeta`，而不是将所有 trajectory fields 复制到 controller 返回值中。

因此 sync mode 虽然使用 TransferQueue 和后台 AgentLoop，但没有 async replay backlog 的训练含义：trainer 仍等待当前生成批次形成足够的 terminal groups。TQ 在这里主要解决跨 worker 的数据存储、状态管理和字段回写。

源码入口：

- [`ReplayBuffer` 的状态和 key 约定](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/replay_buffer.py#L63)
- [`_sync_metadata_from_transfer_queue`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/replay_buffer.py#L188)
- [`ReplayBuffer.sample`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/replay_buffer.py#L404)

## Batch Shape Transitions

AgentLoop output 的 response 长度可以不同，因此 TQ 中的 `prompts`、`responses`、`rm_scores` 和 log-prob 通常保留为 jagged / nested representation。进入 reward、advantage 或 engine dispatch 时，trainer 按阶段需要转成 padded tensor；写回 TQ 时再使用有效长度转换回 nested representation。

```text
per-trajectory variable length
  -> nested TensorDict in TransferQueue
  -> padded DataProto for algorithm / model call
  -> nested fields written back by trajectory key
```

以下长度不能混淆：

```text
prompt_len   prompt token 数
response_len response 容器长度，可能含 observation token
seq_len      prompt_len + response_len
policy_len   response_mask 中有效 policy token 数
```

对于 agent 数据，`response_len` 较大并不意味着同样数量的 token 会产生 policy gradient；trace 需要同时记录 `response_mask.sum()`。

### DP batch balance 与 padding

`_balance_batch` 先依据 actor data-parallel size 计算下游需要的 batch multiple，再通过 padding sample 使 batch 可被 mini-batch 和 DP 结构整除，随后依据 `seq_len` 估计 workload，重新排列样本，让各 DP rank 获得更接近的 token workload。

该步骤改变的是 dispatch 顺序，不是 group 关系。padding sample 由 tag 标识，metrics 统计通过 `is_padding` 排除。

源码入口：

- [`_balance_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1496)
- [`_compute_metrics` 的 non-padding 过滤](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1775)

### Training backend 中的 batch mapping

设配置中的 `ppo_mini_batch_size=M`、`rollout.n=G`，trainer 传给 actor 的全局 PPO mini-batch 为 `M × G` 条 trajectory。worker 使用 model engine 报告的 DP size 划分这批样本，但 DP 的来源因 backend 而异。

FSDP baseline 中：

```text
world_size = trainer.nnodes × trainer.n_gpus_per_node
actor DP   = world_size / Ulysses SP
```

`fsdp_size=-1` 表示使用全部可用 ranks 作为 FSDP shard group；设置更小的 `fsdp_size` 会形成二维 DDP × FSDP mesh，但当前 FSDP engine 的 actor batch dispatch 仍通过 `get_data_parallel_size()` 按 `world_size / Ulysses SP` 计算 logical DP。

Megatron 的 process groups 由 MCore 使用 TP、PP、CP、EP 等配置建立。关闭 dynamic context parallel 时，普通 actor DP 可写为：

```text
actor DP = world_size / (TP × PP × CP)
```

一个 DP replica 内的 TP / PP / CP ranks 共同处理同一份样本，其中 PP 还要把本地 batch 组织成 pipeline micro-batches。EP 会进一步改变 MoE expert 参数使用的通信组；dynamic context parallel 则在 verl 中显式返回 logical `DP=1`，不能继续套用静态公式。

因此 `M × G / DP` 表示每个 DP replica 的逻辑样本份额，而不是所有 backend 下“每张 GPU 独立拥有的样本数”。rollout backend 的 TP 只决定生成副本如何占用 GPU，也不参与 actor DP 的推导。

源码入口：

- [`TrainingWorker.train_mini_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine_workers.py#L241)
- [FSDP device mesh 与 DP](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L230)
- [Megatron model-parallel 初始化](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/megatron/transformer_impl.py#L239)
- [Megatron pipeline forward-backward schedule](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/megatron/transformer_impl.py#L930)

## Batch Cleanup

actor update、metrics 和可选 rollout dump 完成后，`fit` 使用本次返回的 `KVBatchMeta.keys` 调用 `tq.kv_clear`，清除当前 partition 的 trajectory fields 和 tags。

```text
sample keys
  -> reward / log-prob / advantage write-back
  -> actor update
  -> metrics read
  -> tq.kv_clear(keys, partition_id)
```

如果真实运行中出现字段不存在、旧 batch 污染下一轮或 TQ 存储持续增长，应首先检查 cleanup 是否覆盖了合并后的 `batch.keys`，以及 AgentLoop 是否仍有未完成任务向已清理 group 写回。

源码入口：

- [`fit` 中的 TQ cleanup](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L488)

## Field Invariants

- 每条 trajectory 的 `prompts`、`responses`、`response_mask` 和 `loss_mask` 在有效长度上对齐；
- `input_ids` 的长度等于 `prompt_len + response_len`，`position_ids` 与其形状兼容；
- 同一 prompt 的 sampled trajectories 共享稳定 `uid`，trajectory key 的 session / index 能还原层级关系；
- `response_mask=0` 的 observation / padding token 不被当作 policy action；
- padding sample 有明确 `is_padding` tag，并从有效 reward、metrics 和样本计数中排除；
- batch reorder 不改变 uid、trajectory key 和 version tags；
- 全局 PPO mini-batch 能被 backend 报告的 logical DP size 正确划分；
- topology 记录区分 actor TP / PP / CP / SP 与 rollout TP，不用 rollout 并行度推导 actor DP；
- nested -> padded -> nested 转换不丢失每条样本的真实长度；
- `advantages` 和 `returns` 的有效区域与 response mask 对齐；
- cleanup 使用完整的合并 key 集合，不把 trajectory 残留到下一轮。

## Evidence Status

- **Source fact**：字段名称、TQ key / tag 格式、状态集合、nested / padded 转换和 cleanup 位置来自固定 revision 的源码。
- **Tested behavior**：尚未运行真实 example，因此 FSDP 与 Megatron 下的实际 dtype、device、nested tensor 布局、DP dispatch 和动态 batch shape 仍待分别 trace 确认。
- **Engineering interpretation**：关于 `response_mask` 对 policy token 统计的影响、sync TQ 的 bufferless 语义和 cleanup 竞态风险，是基于字段 lineage 的工程解释。

## Trace Entry

最小 trace 只需对一个 prompt group 记录：

```text
stage
field
producer
consumer
shape / nested length
dtype / device
valid length
response_mask.sum()
uid / trajectory key
global_steps / min_global_steps / max_global_steps
is_padding
```

实验设计见 [[projects/verl-source-reading/labs/01-trace-config-and-batch|Lab 01: Trace Config and Batch]]；主运行时序见 [[projects/verl-source-reading/01-sync-grpo-run-overview|01 Sync GRPO Run Overview]]。

## 相关知识

- [[training/post-training/grpo|GRPO]]
- [[training/post-training/ppo|PPO]]
- [[training/distributed-training/fsdp|FSDP]]
- [[projects/megatron-lm-source-reading/01-training-run-overview|Megatron training run]]
