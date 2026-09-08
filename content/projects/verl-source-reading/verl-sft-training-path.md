---
title: verl SFT Training Path
created: 2026-09-03
published: 2026-09-03
modified: 2026-09-03
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - supervised-fine-tuning
  - post-training
---

verl 的主要阅读主线仍然是 RL：prompt 经过 rollout 和环境交互生成 trajectory，再由 reward、advantage 和 policy loss 驱动 actor update。本页保留一条 SFT 对照路径，用来说明监督训练如何复用 verl 的数据、worker 和 model engine，而不改变项目的 RL 组织方式。

## SFT 在 verl 中的位置

SFT 的训练目标是给定完整的监督序列，直接优化目标 token 的 next-token likelihood。它没有 rollout、reward、advantage、old policy 或 policy ratio，运行链更接近普通的 distributed language-model training：

```text
messages / trajectory records
  -> MultiTurnSFTDataset
  -> input_ids + loss_mask + attention_mask
  -> SFTTensorCollator
  -> data-parallel batch
  -> TrainingWorker
  -> FSDP / Megatron / AutoModel engine
  -> masked next-token loss
  -> optimizer step
  -> checkpoint
```

源码入口是 [`examples/sft/`](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/sft) 和 [`verl/trainer/sft_trainer.py`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/sft_trainer.py)。与 RL 主线通过 `main_ppo.py` 进入 V1 trainer 不同，SFT 直接通过 `torchrun -m verl.trainer.sft_trainer` 启动。

## 数据如何变成训练字段

当前 SFT dataset 以 parquet 文件为输入，默认读取 `messages` 字段，也可以从配置中指定 message、tool、image、video 和 thinking 相关字段。`SFTTrainer._build_dataset` 创建 `MultiTurnSFTDataset`，随后由 dataset 的 `__getitem__` 逐条处理完整对话。

```text
parquet row
  -> messages / tools / multimodal inputs
  -> tokenize each message with chat template
  -> concatenate turns
  -> input_ids
  -> attention_mask
  -> loss_mask
```

对应源码：

- [`SFTTrainer._build_dataset`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/sft_trainer.py#L200)
- [`MultiTurnSFTDataset._process_single_message`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/dataset/multiturn_sft_dataset.py#L187)
- [`MultiTurnSFTDataset.__getitem__`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/dataset/multiturn_sft_dataset.py#L291)

每条 message 先独立套用 tokenizer 的 chat template，再拼接成一个训练序列。assistant message 默认产生有效的 `loss_mask`，同时屏蔽 assistant generation prompt；system、user 等非 assistant message 的 mask 为零。这样模型可以看到完整上下文，却只对监督目标部分承担 loss。

这一点对 agent trajectory 尤其重要：如果一条样本包含 user request、assistant reasoning、tool call、tool result 和下一轮 assistant response，是否对 tool result 或其他 environment observation 计算 loss，不由“它出现在一条序列里”自动决定，而取决于数据 message role、chat template 和最终 `loss_mask`。因此，使用长 agent trajectory 做 SFT 前，需要先验证每种 role 的 token 边界。

## Padding、截断与长序列

SFT dataset 支持 `right` 和 `no_padding` 两种模式。

```text
right padding
  -> 固定 max_length
  -> padding tokens 的 attention_mask / loss_mask 为 0

no padding
  -> 每条样本保留实际长度
  -> 使用 nested tensor 表示 batch
```

`max_length` 超限时，`truncation` 可以设为 `left`、`right` 或 `error`。对于长程 agent trajectory，截断方向不是纯粹的内存配置：left truncation 可能丢失早期任务背景，right truncation 可能丢失末尾 action 或 final answer，而 `error` 则能把超长样本显式暴露出来。训练前应将序列长度分布、有效 loss token 数和截断比例作为数据诊断的一部分。

源码锚点：[`padding and truncation`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/dataset/multiturn_sft_dataset.py#L364)。

## Loss 与 RL 的边界

SFT loss 使用 model output 中的 token log-prob，并根据 `loss_mask` 或 `response_mask` 过滤不参与监督的 token：

```python
log_prob = model_output["log_probs"]
loss_mask = data["loss_mask"]
loss_mask = shift_left(loss_mask)
loss = -masked_sum(log_prob, loss_mask) / batch_num_tokens * dp_size
```

在 `no_padding` 模式下，verl 将 nested tensor 展平后对齐 shifted mask；在 padding 模式下，则直接使用 `response_mask`。loss 会按照 batch token 数归一化，并结合 data-parallel size 适配分布式执行。这里没有 RL 中的 `advantages`、`old_log_probs`、`ref_log_prob` 或 KL penalty。

源码：[`sft_loss`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/utils/losses.py#L28)。

因此需要区分两种看似相近的训练：

- SFT 是 response 或指定 message token 上的 masked NTP；
- mid-training 常见的全 token NTP 则通常不依赖 assistant-only `loss_mask`，并可能把完整序列作为语言建模文本处理。

verl 的 SFT 路径适合监督式 post-training、multi-turn conversation 和带明确 target 的 agent trajectory。它不能仅凭 trainer 名称替代面向百 B 级别语料吞吐优化的通用 pre-training / mid-training pipeline。

## Batch 与 TrainingWorker

SFT 的 global batch 由 `data.train_batch_size` 指定。`SFTTrainer` 从当前 engine 读取 data-parallel rank 和 size，通过 `DistributedSampler` 为每个 DP rank 分配数据，并使用：

```text
train_batch_size_per_dp = train_batch_size / dp_size
```

单卡上的 `micro_batch_size_per_gpu`、`max_token_len_per_gpu` 和 `use_dynamic_bsz` 决定实际 engine 如何组织 forward/backward；它们不改变 global batch 的定义。相比 RL 路径，SFT 每个 dataloader batch 直接进入一次 `training_client.train_batch`，不存在 PPO 的 rollout group、PPO mini-batch 和多 epoch policy update 语义。

源码锚点：

- [`SFTTrainer._build_dataloader`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/sft_trainer.py#L224)
- [`SFTTrainer.fit`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/sft_trainer.py#L308)
- [`TrainingWorker.train_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine_workers.py#L340)

`TrainingWorker` 是 SFT 与 RL 之间重要的共用边界。它接收 TensorDict，注入 remove padding、dynamic batch size 和 token length 等工程参数，然后将 batch 与指定 loss function 交给 engine。SFT trainer 注入的是 `sft_loss`；RL actor worker 则注入 `ppo_loss` 或对应的 policy loss。因此，两条路径可以共享 engine contract，但训练字段和优化语义不同。

## Backend 选择

SFT 的入口和 loss 与 backend 解耦，具体执行由 `engine` 和 `optim` 配置决定。

```text
verl SFT trainer
  -> TrainingWorker
       ├─ engine=fsdp
       ├─ engine=megatron
       └─ engine=automodel
```

FSDP 适合较直接的 data-parallel sharding 与快速实验；Megatron backend 提供 TP、PP、EP 等更完整的大模型并行执行；AutoModel 是针对特定模型和 engine 的另一条实现路径。相同的 `input_ids`、mask 和 global batch 进入不同 backend 后，本地 batch layout、micro-batch schedule、通信、optimizer state 和 checkpoint layout 仍然可能不同。

一个可直接阅读的 Megatron SFT 配置是 [`run_qwen_megatron_fsdp.sh`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/sft/gsm8k/run_qwen_megatron_fsdp.sh)，其中 `engine=megatron`、TP/PP 参数和 `verl.trainer.sft_trainer` 共同出现。这说明“使用 verl 跑 SFT”和“使用 Megatron 执行 SFT”可以同时成立。

## Checkpoint 与 RL 衔接

SFT trainer 通过 `CheckpointHandler` 保存 model、optimizer、extra 等内容，并支持从已有 checkpoint 恢复。SFT 输出的 checkpoint 可以作为后续 RL 的 actor initialization，但真正衔接时还需要重新确认：

```text
SFT checkpoint
  -> model format / tokenizer compatibility
  -> RL actor engine loading
  -> rollout backend weight loading
  -> reference / critic initialization
  -> RL-specific optimizer and scheduler state
```

SFT checkpoint 能被加载，不等于它已经包含 RL 运行所需的 rollout、reference、critic 或 policy-version 状态。模型权重兼容性和训练状态兼容性需要分开检查。

## 与 RL 主线的关系

```text
SFT path:
  supervised messages -> masked likelihood -> checkpoint

RL path:
  prompt -> rollout / environment -> reward -> advantage -> policy update
```

SFT 页面在本项目中的作用，是提供一条可复用的监督训练基线：当需要判断某个收益来自数据质量、assistant mask、模型初始化还是 RL objective 时，可以先用相同 engine 和相近数据做 SFT 对照，再回到 RL 主线分析 rollout、reward 和 policy update 的额外贡献。

## 相关知识

- [[projects/verl-source-reading/|verl Source Reading]]
- [[projects/verl-source-reading/roadmap|verl Source Reading Roadmap]]
- [[projects/verl-source-reading/07-model-engine-and-backend|07 Model Engine and Backend]]
- [[training/post-training/ppo|PPO]]
- [[training/post-training/grpo|GRPO]]
- [[training/distributed-training/fsdp|FSDP]]
- [[training/distributed-training/megatron|Megatron 与 3D 并行]]

