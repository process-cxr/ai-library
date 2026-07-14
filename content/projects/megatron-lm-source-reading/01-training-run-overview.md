---
title: 01 Training Run Overview
created: 2026-07-08
published: 2026-07-08
modified: 2026-07-09
type: project
status: growing
tags:
  - projects
  - source-reading
  - megatron
---

这篇笔记建立 Megatron GPT training run 的第一层全局图。当前不区分 pre-training 与 mid-training 的训练循环差异，而是先读懂 causal LM training run 的工程骨架：

```text
入口脚本
  -> 参数与配置
  -> distributed / parallel state 初始化
  -> model / optimizer / scheduler
  -> dataset / dataloader
  -> train loop
  -> forward / loss / backward schedule
  -> gradient sync / optimizer step
  -> checkpoint / eval / logging
```

现在知识库页面已经支持右侧 SourcePanel，所以正文不再摘录大段源码。正文只保留精确源码锚点，点击链接后在右侧看源码；正文部分负责解释每个位置在训练系统中的角色、边界和后续阅读方向。

## 阅读目标

第一轮阅读要回答五个问题：

1. 一批 token 从哪里进入训练循环？
2. 模型、optimizer 和 scheduler 在哪里被构造？
3. `forward_step`、`loss_func` 和 pipeline schedule 的接口关系是什么？
4. gradient synchronization 和 optimizer step 在什么边界发生？
5. checkpoint 保存的是普通权重，还是完整训练状态？

## 源码入口

源码仓库：[process-cxr/Megatron-LM-study](https://github.com/process-cxr/Megatron-LM-study)，Megatron 主目录见 [megatron/](https://github.com/process-cxr/Megatron-LM-study/tree/main/megatron)。

本轮主要读两条入口：

- [examples/run_simple_mcore_train_loop.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py)：最小 Megatron-Core 训练闭环。
- [pretrain_gpt.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py)：正式 GPT pretrain / SFT 入口。

辅助入口：

- [megatron/training/training.py::pretrain](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L1005)：正式训练主控函数。
- [megatron/training/training.py::setup_model_and_optimizer](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2000)：模型、optimizer、scheduler 构造入口。
- [megatron/training/training.py::train](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L3248)：正式训练循环。
- [megatron/training/training.py::train_step](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2280)：单个 training step 的核心执行入口。
- [megatron/training/checkpointing.py::save_checkpoint](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/checkpointing.py#L510) / [load_checkpoint](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/checkpointing.py#L1718)：正式 checkpoint 入口。

## 两个入口的关系

[run_simple_mcore_train_loop.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py) 是最小可运行训练系统。它把 distributed init、model provider、DDP wrapper、data iterator、forward/backward schedule、gradient finalize、optimizer step 和 checkpoint 都显式写在一个文件里。它适合用来建立训练闭环的最小心智模型。

[pretrain_gpt.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py) 是正式 GPT 入口。它不直接手写完整训练循环，而是定义 GPT 训练需要的 hook 和配置，然后把控制权交给 [training.py::pretrain](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L1005)。

两者的关系可以理解为：

```text
simple loop:
  把训练闭环拆开写，适合看懂基本工程骨架

formal GPT entry:
  把模型、数据、forward/loss 作为 hook 接入 Megatron 通用训练系统
```

simple loop 帮助理解“训练系统至少需要哪些组件”；正式入口帮助理解“真实 Megatron 如何把这些组件组织成可扩展的大规模训练系统”。

## 主链路对照

| 训练环节                  | simple loop 源码锚点                                                                                                                                                                                                                                                            | 正式 GPT 入口源码锚点                                                                                                                                                                                                                          |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| distributed 初始化        | [initialize_distributed](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L29)                                                                                                                                                | [initialize_megatron](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L1081)                                                                                                                          |
| model parallel group      | [parallel_state.initialize_model_parallel](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/core/parallel_state.py#L547)                                                                                                                                     | [training.py::pretrain 初始化阶段](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L1081)                                                                                                             |
| 模型构造                  | [simple model_provider](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L57)                                                                                                                                                 | [setup_model_and_optimizer](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2000)                                                                                                                    |
| dataset / dataloader      | [get_train_data_iterator](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L82)                                                                                                                                               | [train_valid_test_datasets_provider](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L438)                                                                                                                          |
| batch 获取                | [forward_step_func](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L124) 中取 batch                                                                                                                                         | [get_batch](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L92)                                                                                                                                                    |
| forward / loss            | [forward_step_func](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L124)                                                                                                                                                    | [forward_step](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L277) / [loss_func](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L204)                                                 |
| forward/backward schedule | [get_forward_backward_func](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/core/pipeline_parallel/schedules.py#L48)                                                                                                                                        | [train 中选择 schedule](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L3516)                                                                                                                        |
| 单步训练                  | [simple loop iteration](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L249)                                                                                                                                                | [train_step](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2280)                                                                                                                                   |
| checkpoint                | [save_distributed_checkpoint](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L169) / [load_distributed_checkpoint](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L193) | [save_checkpoint](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/checkpointing.py#L510) / [load_checkpoint](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/checkpointing.py#L1718) |

## Simple Loop 的作用

simple loop 不是正式训练系统的替代品，它的价值是把 Megatron-Core 训练闭环压缩到最小可读形态。阅读时重点看这些边界：

- distributed 初始化从 [run_simple_mcore_train_loop.py#L29](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L29) 开始，先建立 `torch.distributed`，再初始化 Megatron model-parallel groups。
- 小模型构造从 [run_simple_mcore_train_loop.py#L57](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L57) 开始，用一个很小的 `TransformerConfig` 和 `GPTModel` 展示模型创建过程。
- 数据迭代器从 [run_simple_mcore_train_loop.py#L82](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L82) 开始，保留了 dataset builder 和 dataloader 的基本形态。
- forward step 从 [run_simple_mcore_train_loop.py#L124](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L124) 开始，展示 Megatron schedule 需要的 `output_tensor, loss_func` 接口。
- checkpoint 保存和加载分别从 [run_simple_mcore_train_loop.py#L169](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L169) 和 [run_simple_mcore_train_loop.py#L193](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L193) 开始，展示 `sharded_state_dict` 与 `dist_checkpointing.save/load` 的关系。
- 主训练闭环从 [run_simple_mcore_train_loop.py#L221](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L221) 开始，核心 iteration 在 [run_simple_mcore_train_loop.py#L249](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L249) 附近。

这一段读完后，应建立一个最小 schema：

```text
data iterator
  -> forward_step_func
  -> forward_backward_func
  -> finalize_model_grads
  -> optimizer.step
  -> checkpoint save/load
```

后面读正式入口时，不需要重新理解“训练是什么”，而是看正式系统如何把这个 schema 扩展到多并行维度、完整配置、复杂 checkpoint 和训练监控。

## 正式 GPT 入口调用链

[pretrain_gpt.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py) 的职责不是执行每一个训练细节，而是为 GPT 任务定义 Megatron 通用训练系统需要的接口。它主要提供四类东西：

1. 数据接口：`train_valid_test_datasets_provider` 和 `get_batch`。
2. 计算接口：`forward_step` 和 `loss_func`。
3. 模型配置：`gpt_config_from_args(args)` 生成 GPT model config。
4. 训练配置容器：`pretrain_cfg_container_from_args(args, model_cfg)` 组装 training / optimizer / DDP / checkpoint / tokenizer 等配置。

主入口在 [pretrain_gpt.py#L489](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L489)。真正把控制权交给训练系统的是 [pretrain_gpt.py#L508-L515](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L508-L515)。

这一版源码里有一个需要特别注意的变化：`pretrain_gpt.py` 虽然在 [pretrain_gpt.py#L64](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L64) import 了 `model_provider`，但主入口处调用 `pretrain(...)` 时并没有像旧版 Megatron 教程那样显式传入 `model_provider`。模型构造优先走配置容器路线：

- [gpt_config_from_args](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/argument_utils.py#L430) 把 args 转成 `GPTModelConfig`。
- [pretrain_cfg_container_from_args](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/argument_utils.py#L517) 把 `model_cfg` 放入 `PretrainConfigContainer`。
- [setup_model_and_optimizer](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2000) 在 [training.py#L2020-L2043](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2020-L2043) 优先使用 `cfg_container.model` 的 builder 构造分布式模型；只有没有 model config 时才回退到 `model_provider_func`。

因此，当前正式 GPT 入口的主线更准确地写成：

```text
pretrain_gpt.py
  -> parse_and_validate_args
  -> gpt_config_from_args
  -> pretrain_cfg_container_from_args
  -> training.py::pretrain
      -> initialize_megatron
      -> setup_model_and_optimizer
      -> build_train_valid_test_data_iterators
      -> train
          -> get_forward_backward_func
          -> train_step
      -> eval / checkpoint / finalize
```

这里的关键不是“入口文件里是否直接构造模型”，而是入口文件把 GPT-specific 的部分交给配置和 hook，把训练系统的通用部分交给 `megatron/training/training.py`。

## `training.py::pretrain` 的外层控制流

[training.py::pretrain](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L1005) 是正式训练的外层主控函数。它负责把 GPT 入口提供的 hook 和配置变成完整训练任务。

主要阶段如下：

- 初始化 Megatron：入口在 [training.py#L1081-L1093](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L1081-L1093)。这里会完成参数、timers、distributed process groups、parallel state、随机种子等初始化。
- 构造模型、optimizer 和 scheduler：入口在 [training.py#L1217-L1234](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L1217-L1234)，具体进入 [setup_model_and_optimizer](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2000)。
- 构造 train / valid / test iterators：入口在 [training.py#L1330-L1357](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L1330-L1357)。这里会调用入口脚本提供的 `train_valid_test_dataset_provider`。
- 进入正式训练：入口在 [training.py#L1405-L1428](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L1405-L1428)，核心训练循环在 [train](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L3248)。
- 训练结束后的验证与测试：验证入口在 [training.py#L1452-L1481](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L1452-L1481)，测试入口在 [training.py#L1483-L1498](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L1483-L1498)。

读 `pretrain` 时不要陷入所有日志、fault tolerance、profiling、RL 分支。第一轮只需要把它看成一个外层 orchestrator：它先把训练环境、模型和数据准备好，再把每个 step 交给 `train_step`。

## Hook Contract：入口文件到底交给训练系统什么

正式 GPT 入口和通用训练系统之间的接口主要是三个 hook。

第一个是 dataset provider：[train_valid_test_datasets_provider](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L438)。它根据 args 选择 `SFTDataset`、`MockGPTDataset`、`GPTFIMDataset` 或 `GPTDataset`，再通过 [BlendedMegatronDatasetBuilder](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L465) 构造 train / valid / test datasets。正式 dataloader 的构造不在入口脚本里完成，而是在 `training.py::pretrain` 中调用 dataset provider 后继续完成。

第二个是 forward step：[forward_step](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L277)。它的职责不是执行完整 backward，而是给 pipeline schedule 提供一次 micro-batch forward 的行为。它先从 `get_batch` 获取当前 rank 应该看到的 batch，然后调用 model，最后返回 `output_tensor` 和绑定了 `loss_mask` 的 `loss_func`。

第三个是 loss function：[loss_func](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L204)。它把 model 输出的 per-token loss 和 `loss_mask` 结合起来，得到当前 micro-batch 的 loss、token 计数和用于日志汇报的 loss report。它还包含 cached logits / ModelOpt distillation / NaN / Inf / spiky loss 等扩展路径，但第一轮只要抓住“masked NTP loss + report”这条主线。

这三个 hook 共同把 GPT 任务接入 Megatron 通用训练系统：

```text
dataset provider:
  定义数据从哪里来

forward_step:
  定义一个 micro-batch 如何进入 model

loss_func:
  定义 model output 如何变成可训练 loss
```

## Batch Flow

正式 `pretrain_gpt.py` 的 batch flow 比 simple loop 复杂，因为 batch 获取已经嵌入 TP、CP、PP 和 packed sequence 的并行语义。

关键源码锚点：

- batch 字段顺序定义在 [pretrain_gpt.py#L76-L89](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L76-L89)。
- `get_batch` 入口在 [pretrain_gpt.py#L92](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L92)。
- tensor-parallel group 内，`tp_rank == 0` 读取 batch 并搬到 CUDA，见 [pretrain_gpt.py#L118-L126](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L118-L126)。
- TP 内 batch 分发由 [get_batch_on_this_tp_rank](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L128-L143) 完成。
- packed sequence 的 flatten 在 [pretrain_gpt.py#L145](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L145)。
- 某些非首尾 pipeline stage 在特定 packed sequence 场景下只需要 sequence metadata，见 [pretrain_gpt.py#L147-L160](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L147-L160)。
- context parallel 切分由 [get_batch_on_this_cp_rank](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L162-L168) 完成。

这意味着正式 Megatron 数据入口不是“每个 rank 都读一份完整 batch”。更准确地说：

```text
data iterator
  -> TP src rank 读取 batch
  -> TP group 内分发当前 rank 所需字段
  -> packed sequence 整理
  -> CP rank 切分上下文
  -> forward_step unpack 固定字段
```

这部分后续适合单独写成 `03-data-and-batch-flow`，重点追 `get_batch_on_this_tp_rank`、`get_batch_on_this_cp_rank` 和 packed sequence metadata 如何进入 attention。

## Forward / Loss / Backward Flow

[forward_step](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L277) 的主线是：

```text
get_batch
  -> optional PackedSeqParams
  -> model(...)
  -> return output_tensor, partial(loss_func, loss_mask, model=model)
```

对应源码锚点：

- 获取 batch 在 [pretrain_gpt.py#L288-L303](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L288-L303)。
- packed sequence 参数构造在 [pretrain_gpt.py#L305-L328](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L305-L328)。
- model forward 在 [pretrain_gpt.py#L332-L352](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L332-L352)。
- loss function 入口在 [pretrain_gpt.py#L204](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L204)，基础 masked loss 路径在 [pretrain_gpt.py#L236-L241](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L236-L241)。

Megatron 的重要接口习惯是：`forward_step` 不直接完成完整训练 step，它返回 `output_tensor` 和 loss closure。真正的 forward/backward 调度由 pipeline schedule 接管：

- schedule 选择入口在 [get_forward_backward_func](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/core/pipeline_parallel/schedules.py#L48)。
- 正式 train loop 中获取 schedule 的位置在 [training.py#L3516-L3518](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L3516-L3518)。
- 单个 step 调度入口是 [train_step](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2280)。
- train loop 调用 `train_step` 的位置在 [training.py#L3758-L3762](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L3758-L3762)。

因此，第一轮不应该把 `forward_step` 理解成完整训练 step。它只是 schedule 的一个可调用单元。pipeline parallel、microbatch、activation checkpointing、gradient accumulation 等复杂行为都在 schedule 和 `train_step` 里展开。

## Model / Optimizer Flow

正式模型构造主线在 [setup_model_and_optimizer](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2000)。

当前源码优先使用配置容器中的 model config：

- `cfg_container.model` 路径在 [training.py#L2020-L2038](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2020-L2038)。
- fallback 到 `model_provider_func` 的路径在 [training.py#L2039-L2041](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2039-L2041)。
- optimizer 构造在 [training.py#L2071-L2093](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2071-L2093)。

这和 simple loop 有明显不同。simple loop 里可以直观看到 `model_provider()`、`DistributedDataParallel(...)`、`Adam(...)`。正式入口里这些都被配置系统、builder、DDP config、optimizer config 包起来了。

需要特别区分两层模型构造方式：

- 旧式 hook 风格：传入 `model_provider_func`，由 `get_model` 调用 provider。
- 新式 config / builder 风格：入口脚本先构造 `GPTModelConfig`，`setup_model_and_optimizer` 根据 config 的 builder 构造 distributed models。

当前 `pretrain_gpt.py` 主线走后者。`model_provider.py` 和 `gpt_builders.py` 仍然有阅读价值，尤其是理解 GPTModel 如何选择 layer spec：

- [model_provider.py::model_provider](https://github.com/process-cxr/Megatron-LM-study/blob/main/model_provider.py#L20)。
- [gpt_builders.py::gpt_builder](https://github.com/process-cxr/Megatron-LM-study/blob/main/gpt_builders.py#L25)。
- GPT layer spec 选择在 [gpt_builders.py#L32-L55](https://github.com/process-cxr/Megatron-LM-study/blob/main/gpt_builders.py#L32-L55)。
- `GPTModel` 构造在 [gpt_builders.py#L86-L105](https://github.com/process-cxr/Megatron-LM-study/blob/main/gpt_builders.py#L86-L105)。

## Train Loop

正式训练循环入口是 [training.py::train](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L3248)。这一层比 `pretrain` 更接近“训练运行时”。

第一轮只追这些位置：

- 函数入口与职责说明在 [training.py#L3248-L3268](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L3248-L3268)。
- model 切到 training mode 在 [training.py#L3398-L3400](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L3398-L3400)。
- gradient / param sync hook 配置在 [training.py#L3431-L3452](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L3431-L3452)。
- forward/backward schedule 获取在 [training.py#L3516-L3518](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L3516-L3518)。
- 主 while loop 从 [training.py#L3625](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L3625) 开始。
- microbatch 更新与因 microbatch 改变触发 checkpoint 的逻辑在 [training.py#L3654-L3683](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L3654-L3683)。
- `train_step` 调用在 [training.py#L3758-L3762](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L3758-L3762)。
- step 后 checkpoint 触发在 [training.py#L3773-L3782](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L3773-L3782)。

这一层要建立的理解是：`train` 管 iteration 生命周期，`train_step` 管一次训练 step 的计算和更新，schedule 管 forward/backward 在 microbatch 和 pipeline stage 上如何展开。

## Gradient Sync / Optimizer Flow

simple loop 里梯度同步边界很显式：[finalize_model_grads](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L259) 后接 `optim.step`。正式训练中，这个边界被包装得更深：

- gradient sync hook 的配置在 [training.py#L3431-L3452](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L3431-L3452)。
- `finalize_model_grads` 默认 hook 设定在 [training.py#L3450-L3452](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L3450-L3452)。
- `train_step` 负责把 forward/backward、grad sync、optimizer step 和 scheduler step 串起来，入口在 [training.py#L2280](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2280)。

后续阅读这里时，需要把通信概念和训练语义对应起来：

- TP 下哪些梯度需要跨 tensor-parallel ranks 同步；
- DP 下哪些梯度需要 reduce-scatter / all-reduce；
- distributed optimizer 如何切分 optimizer state；
- overlap grad reduce 如何把通信放进 backward 的时间线；
- gradient clipping / grad norm 如何跨 ranks 得到一致结果。

这部分后续适合单独写 `06-gradient-sync-and-optimizer`。

## Checkpoint Flow

simple loop 的 checkpoint 展示的是模型权重层面的 distributed checkpoint：

- 保存入口：[run_simple_mcore_train_loop.py#L169](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L169)。
- 加载入口：[run_simple_mcore_train_loop.py#L193](https://github.com/process-cxr/Megatron-LM-study/blob/main/examples/run_simple_mcore_train_loop.py#L193)。
- 关键对象是 `model.sharded_state_dict(prefix="")` 和 `dist_checkpointing.save/load`。

正式 checkpoint 是完整训练态 checkpoint：

- 保存入口：[checkpointing.py::save_checkpoint](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/checkpointing.py#L510)。
- 加载入口：[checkpointing.py::load_checkpoint](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/checkpointing.py#L1718)。
- train loop 中的保存包装是 [save_checkpoint_and_time](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2937)。
- `pretrain` 结束前补保存的入口在 [training.py#L1432-L1441](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L1432-L1441)。
- train loop 内 step 后触发保存的位置在 [training.py#L3773-L3782](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L3773-L3782)。

正式 checkpoint 不只包含 model weights，还要处理 optimizer、scheduler、iteration、random state、distributed checkpoint metadata、并行布局和异步保存。后续阅读时要区分：

```text
training checkpoint:
  用于 resume，保存完整训练态

inference / export weights:
  用于部署或评测，通常只关心模型权重
```

## 当前结论

本轮只建立主链路，不深入每个模块内部。当前可以确认：

- simple loop 是最小训练闭环，适合建立 Megatron-Core training run 的基本骨架；
- 正式 `pretrain_gpt.py` 不是手写训练循环，而是把 GPT-specific 数据、forward、loss 和配置交给通用训练系统；
- 当前源码的正式模型构造优先走 `PretrainConfigContainer -> model config -> builder` 路线，而不是旧式显式传 `model_provider` 的路线；
- `training.py::pretrain` 是外层 orchestrator，`training.py::train` 管 iteration 生命周期，`train_step` 管单步训练，pipeline schedule 管 forward/backward 展开；
- 后续阅读要沿着“入口 hook -> 通用训练系统 -> schedule / optimizer / checkpoint 子系统”的顺序走。

## 下一轮阅读顺序

1. `02-distributed-initialization`：从 [initialize_megatron](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L1081) 追到 process group 和 parallel state。
2. `03-data-and-batch-flow`：从 [get_batch](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L92) 追 TP / CP / packed sequence。
3. `04-model-building`：从 [gpt_config_from_args](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/argument_utils.py#L430) 和 [setup_model_and_optimizer](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2000) 追 model config、builder、DDP wrapper。
4. `05-forward-backward-schedule`：从 [forward_step](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py#L277) 和 [get_forward_backward_func](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/core/pipeline_parallel/schedules.py#L48) 追 pipeline schedule。
5. `06-gradient-sync-and-optimizer`：从 [train_step](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py#L2280) 追 grad sync、distributed optimizer、scheduler step。
6. `07-checkpoint-flow`：从 [save_checkpoint](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/checkpointing.py#L510) 和 [load_checkpoint](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/checkpointing.py#L1718) 追完整训练态 checkpoint。
