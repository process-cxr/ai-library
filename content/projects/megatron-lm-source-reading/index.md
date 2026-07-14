---
title: Megatron-LM Source Reading
created: 2026-07-08
published: 2026-07-08
modified: 2026-07-09
type: project
status: growing
tags:
  - projects
  - source-reading
  - megatron
  - training
---

Megatron-LM 是一个面向大模型训练的工程系统。它不是单一的 model implementation，也不是单纯的 training script，而是一套把数据、模型并行、pipeline 调度、分布式优化器和 checkpoint 组织在一起的训练栈。

这组源码阅读笔记的主线是一次 GPT causal LM training run：token 如何从数据集进入 batch，batch 如何被切分到不同 parallel ranks，模型如何完成 forward / backward，梯度如何同步，optimizer 如何更新，checkpoint 如何保存和恢复。把这条链路读通之后，再去看 tensor parallel、pipeline parallel、context parallel、distributed optimizer 这些模块，才不会被目录结构拖着走。

源码仓库：[process-cxr/Megatron-LM-study](https://github.com/process-cxr/Megatron-LM-study)，Megatron 主目录见 [megatron/](https://github.com/process-cxr/Megatron-LM-study/tree/main/megatron)。

## 项目结构

从工程职责看，Megatron 可以先拆成四层。

第一层是训练入口。这里包括 [pretrain_gpt.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/pretrain_gpt.py)、[model_provider.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/model_provider.py)、[gpt_builders.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/gpt_builders.py) 和 examples 里的最小训练脚本。它们负责把“训练 GPT”这件事具体化为数据 provider、model config、forward step 和 loss function。

第二层是训练主控。核心入口在 [megatron/training/training.py](https://github.com/process-cxr/Megatron-LM-study/blob/main/megatron/training/training.py)，它负责初始化、构造模型与 optimizer、构造 dataloader、进入 train loop、触发 eval 和 checkpoint。阅读这一层时要抓住 `pretrain -> setup_model_and_optimizer -> train -> train_step` 这条主调用链。

第三层是 Megatron-Core。核心目录在 [megatron/core/](https://github.com/process-cxr/Megatron-LM-study/tree/main/megatron/core)。这里实现真正的并行训练语义：parallel state、tensor parallel layer、pipeline schedule、distributed DDP、optimizer、datasets、transformer block 和 distributed checkpoint。训练入口把任务交给这一层，Megatron-Core 再把任务展开到多卡、多机和多种 parallel groups 上。

第四层是底层通信与运行时。Megatron 并不绕开 `torch.distributed` / NCCL，而是在它们之上构造 TP / PP / DP / CP / EP process groups，并用 collective / point-to-point communication 实现并行训练。理解这一层的重点不是背通信 API，而是知道每次通信在训练语义上是在同步梯度、聚合 partial output、传递 pipeline activation，还是切分 optimizer state。

## 训练链路

一次正式 GPT training run 可以先按下面这条链路理解：

```text
pretrain_gpt.py
  -> argument / config
  -> training.py::pretrain
  -> initialize_megatron
  -> setup_model_and_optimizer
  -> build train / valid / test data iterators
  -> training.py::train
  -> training.py::train_step
  -> pipeline forward/backward schedule
  -> gradient sync / optimizer step
  -> checkpoint / eval / logging
```

这条链路也是本项目第一轮源码阅读的骨架。入口脚本并不负责把所有训练细节写在一个文件里，它只提供 GPT-specific 的部分；通用训练流程由 `megatron/training` 接管；真正的模型并行、通信、optimizer 和 checkpoint 机制落在 `megatron/core`。

因此，读 Megatron 不能只问“这个函数在哪里”，更要问“这段代码属于哪一层”：是 GPT 任务入口、训练生命周期控制、Megatron-Core 并行语义，还是底层通信封装。

## 关键模块

`megatron/training` 是正式训练系统的控制层。这里有参数解析、初始化、训练循环、checkpoint、日志、评测和 fault tolerance 相关逻辑。它的代码量很大，第一轮只追主链路，不急着读完所有分支。

`megatron/core/parallel_state.py` 是并行拓扑入口。它把 global ranks 划分成 tensor-parallel、pipeline-parallel、data-parallel、context-parallel、expert-parallel 等 process groups。后面所有“在哪个 group 上通信”的问题，都要回到这里理解。

`megatron/core/tensor_parallel` 实现 tensor parallel 的基础算子、通信 mapping 和并行 linear layer。ColumnParallelLinear / RowParallelLinear 的权重切分方式，是理解 Megatron 张量并行的核心入口。

`megatron/core/pipeline_parallel` 实现 pipeline schedule。这里决定 micro-batches 如何流过 pipeline stages，forward / backward 如何交错，activation 和 gradient 如何在相邻 stage 之间发送。

`megatron/core/distributed` 和 `megatron/core/optimizer` 共同支撑数据并行、梯度同步、mixed precision、distributed optimizer 和 optimizer state sharding。这里是理解大规模训练内存效率和通信重叠的关键。

`megatron/core/datasets` 与 `megatron/training/datasets` 负责把 token 数据组织成训练 batch。对于 pre-training / mid-training 来说，数据 mixture、sequence length、packing、loss mask 和 sample index 的处理都在这一层体现。

`megatron/core/dist_checkpointing` 与 `megatron/training/checkpointing.py` 负责分布式 checkpoint。它保存的不只是普通模型权重，还要处理 sharded state dict、optimizer state、scheduler、iteration、random state 和并行布局相关 metadata。

## 阅读入口

当前已经完成的第一篇笔记是 [[projects/megatron-lm-source-reading/01-training-run-overview|01 Training Run Overview]]。它不深入每个模块内部，而是先把最小 Megatron-Core 训练闭环和正式 GPT 入口对应起来。

后续阅读顺序放在 [[projects/megatron-lm-source-reading/roadmap|Megatron-LM Source Reading Roadmap]]。Roadmap 只负责安排阅读路径；本页负责提供项目概览和工程地图。

## 源码链接约定

笔记中的源码引用统一指向 GitHub 文件或目录链接；函数级引用使用 `#Lx` 行号锚点。页面支持右侧 SourcePanel：点击源码链接会在右侧打开源码预览，正文不再粘贴大段实现。

代码块只用于表达调用链、数据流或 tensor layout，不作为源码复刻。完整实现以对应 GitHub 链接为准。

## 相关基础知识

- [[training/distributed-training/torch-distributed|torch.distributed]]
- [[training/distributed-training/megatron|Megatron 与 3D 并行]]
- [[training/distributed-training/tensor-parallel|Tensor Parallel]]
- [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]
- [[training/distributed-training/data-parallel|Data Parallel]]
- [[training/distributed-training/sequence-parallel|Sequence Parallel]]
- [[training/distributed-training/context-parallel|Context Parallel]]
- [[training/data-engineering/distributed-dataloader|Distributed Dataloader]]
- [[training/optimization/checkpoint-sharding|Checkpoint Sharding]]
