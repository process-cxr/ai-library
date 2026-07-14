---
title: Megatron-LM Source Reading Roadmap
created: 2026-07-08
published: 2026-07-08
modified: 2026-07-08
type: project
status: growing
tags:
  - projects
  - source-reading
  - megatron
---

这份路线图用于组织 Megatron-LM / Megatron-Core 源码阅读。当前阶段不区分 pre-training 与 mid-training 的训练循环差异：二者在 Megatron 工程主链路上通常共享同一套 causal LM training loop，差别主要体现在 checkpoint 起点、数据 mixture、sequence length、packing、learning-rate schedule 和评测目标。

因此，本项目的首要目标是读懂一次 **Megatron GPT training run**。

## 入口选择

源码仓库：[process-cxr/Megatron-LM-study](https://github.com/process-cxr/Megatron-LM-study)，Megatron 主目录见 [megatron/](https://github.com/process-cxr/Megatron-LM-study/tree/main/megatron)。

两个入口承担不同角色：

- [examples/run_simple_mcore_train_loop.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py)：最小 Megatron-Core 训练闭环，适合建立第一遍工程链路。
- [pretrain_gpt.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py)：正式 GPT pretrain / SFT 入口，适合映射真实训练脚本、数据构造和 Megatron-LM training loop。

先读 simple loop，再读正式入口。simple loop 的价值在于它显式展示了初始化、模型构造、dataloader、forward/backward、gradient sync、optimizer step 和 checkpoint；正式入口则展示真实 GPT 训练中更多配置、数据类型、packing、CP/TP batch 分发和训练框架封装。

## 第一轮：训练主链路

第一轮只回答一个问题：

```text
一次 GPT causal LM training step 是如何完成的？
```

推荐顺序：

1. [examples/run_simple_mcore_train_loop.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py)
2. [pretrain_gpt.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py)
3. [megatron/training/training.py::pretrain](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L1005)
4. [megatron/training/initialize.py::initialize_megatron](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/initialize.py#L42)
5. [megatron/training/training.py::setup_model_and_optimizer](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2000)
6. [megatron/training/training.py::train_step](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2280)
7. [megatron/training/checkpointing.py::save_checkpoint](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/checkpointing.py#L510)
8. [megatron/training/checkpointing.py::load_checkpoint](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/checkpointing.py#L1718)

每个函数阅读时记录：

- 输入是什么；
- 输出是什么；
- 调用了哪些 Megatron-Core 组件；
- 是否涉及 process group；
- 是否涉及 tensor shard / batch shard；
- 是否改变训练状态；
- 与 simple loop 中哪一步对应。

## 第二轮：并行机制

第二轮只回答一个问题：

```text
Megatron 如何把一个模型和一个 batch 分布到多卡上？
```

推荐顺序：

1. [megatron/core/parallel_state.py::initialize_model_parallel](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/core/parallel_state.py#L547)
2. [megatron/core/tensor_parallel/mappings.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/core/tensor_parallel/mappings.py)
3. [megatron/core/tensor_parallel/layers.py::ColumnParallelLinear](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/core/tensor_parallel/layers.py#L778)
4. [megatron/core/tensor_parallel/layers.py::RowParallelLinear](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/core/tensor_parallel/layers.py#L1142)
5. [megatron/core/pipeline_parallel/schedules.py::get_forward_backward_func](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/core/pipeline_parallel/schedules.py#L48)
6. [megatron/core/distributed/distributed_data_parallel.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/core/distributed/distributed_data_parallel.py)
7. [megatron/core/distributed/finalize_model_grads.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/core/distributed/finalize_model_grads.py)
8. [megatron/core/optimizer/](https://github.com/process-cxr/Megatron-LM-study/tree/main/megatron/core/optimizer)

重点不是背 API，而是追踪 tensor layout：

```text
replicated tensor
  -> tensor-parallel shard
  -> sequence/context shard
  -> pipeline stage local activation
  -> data-parallel gradient sync
  -> optimizer state shard
```

每次遇到通信操作都记录：

- API：AllReduce / AllGather / ReduceScatter / Send / Recv / AllToAll；
- group：TP / PP / DP / CP / EP；
- 通信前 tensor shape；
- 通信后 tensor shape；
- 语义：同步梯度、聚合参数、合并 partial output、分发 activation 还是 dispatch token。

## 第三轮：训练系统专题

第三轮把主链路拆成专题，服务后续 mid-training 工程理解。

### 数据专题

核心文件：

- [pretrain_gpt.py::get_batch](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L92)
- [pretrain_gpt.py::core_gpt_dataset_config_from_args](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L371)
- [pretrain_gpt.py::train_valid_test_datasets_provider](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L438)
- [megatron/core/datasets/](https://github.com/process-cxr/Megatron-LM-study/tree/main/megatron/core/datasets)
- [megatron/training/datasets/](https://github.com/process-cxr/Megatron-LM-study/tree/main/megatron/training/datasets)

关注：

- GPTDatasetConfig 如何由 args 构造；
- BlendedMegatronDatasetBuilder 如何构造 train/valid/test；
- batch 中 `tokens`、`labels`、`loss_mask`、`attention_mask`、`position_ids` 如何进入 forward；
- TP / CP 下 batch 如何分发到当前 rank；
- packed sequence 下 `cu_seqlens` 和 `max_seqlen` 如何影响 attention。

### 模型专题

核心文件：

- [model_provider.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/model_provider.py)
- [gpt_builders.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/gpt_builders.py)
- [megatron/core/models/gpt/](https://github.com/process-cxr/Megatron-LM-study/tree/main/megatron/core/models/gpt)
- [megatron/core/transformer/](https://github.com/process-cxr/Megatron-LM-study/tree/main/megatron/core/transformer)

关注：

- GPTModel 如何由 config 和 layer spec 构造；
- TransformerConfig 如何承接 CLI args；
- layer spec 如何决定 attention / MLP / normalization / MoE；
- tensor parallel layer 如何嵌入模型结构。

### 优化器专题

核心文件：

- [megatron/training/training.py::setup_model_and_optimizer](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2000)
- [megatron/training/training.py::train_step](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2280)
- [megatron/core/optimizer/](https://github.com/process-cxr/Megatron-LM-study/tree/main/megatron/core/optimizer)
- [megatron/core/distributed/](https://github.com/process-cxr/Megatron-LM-study/tree/main/megatron/core/distributed)

关注：

- optimizer 如何创建；
- mixed precision 和 grad scaler 如何处理；
- distributed optimizer 如何切分 optimizer states；
- gradient clipping 和 grad norm 如何跨 ranks 计算；
- overlap grad reduce / param gather 如何接入训练 step。

### Checkpoint 专题

核心文件：

- [megatron/training/checkpointing.py::save_checkpoint](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/checkpointing.py#L510)
- [megatron/training/checkpointing.py::load_checkpoint](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/checkpointing.py#L1718)
- [megatron/core/dist_checkpointing/](https://github.com/process-cxr/Megatron-LM-study/tree/main/megatron/core/dist_checkpointing)
- [examples/run_simple_mcore_train_loop.py::save_distributed_checkpoint](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L169)
- [examples/run_simple_mcore_train_loop.py::load_distributed_checkpoint](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L193)

关注：

- 保存的是 sharded training state 还是 model-only weights；
- state dict 如何与 TP / PP / DP / CP 坐标关联；
- optimizer state、scheduler、random state、dataloader progress 是否保存；
- world size 或并行策略变化时如何转换。

## 每篇源码阅读笔记模板

每篇笔记采用固定结构：

```markdown
## 阅读目标

## 源码位置

## 调用链

## 关键对象

## Tensor / Process Group 语义

## 机制解释

## 与训练主链路的关系

## 与基础知识的连接

## 遗留问题
```

记录重点不是复述代码，而是解释“这段代码在训练系统中承担什么职责”。

## 第一批笔记规划

优先完成：

1. [[projects/megatron-lm-source-reading/01-training-run-overview|01 Training Run Overview]]
2. `02-distributed-initialization`
3. `03-data-and-batch-flow`
4. `04-model-provider-and-gpt-model`
5. `05-forward-loss-backward`
6. `06-gradient-sync-and-optimizer`
7. `07-checkpointing`

完成第一轮后，再进入并行机制专题。
