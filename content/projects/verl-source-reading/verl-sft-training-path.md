---
title: verl SFT Training Path
created: 2026-09-03
published: 2026-09-03
modified: 2026-11-03
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - supervised-fine-tuning
  - post-training
---

在 RL 主线中，P0 进入数据集时只有问题和用于评分的标准答案。模型需要先生成四条回答，再通过规则奖励和 GRPO 决定怎样更新。若直接把一条正确解答放进数据集，训练会发生什么变化？本页沿着同一道题走一遍 verl 的 SFT 路径，看监督序列如何变成 token loss，再进入已经熟悉的 FSDP engine。

源码仍固定为 [`5b79827b04cee6e4b7b5ff737e047f1d72d50433`](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433)。下面沿用 2 节点 × 4 GPU 的资源规模，并给出便于比较的 SFT 配置和手算示例；没有实际执行这套多机训练，不能把它视为已验证的启动配方。

## 把同一道题改成监督样本

P0 的问题和正确答案可以保存成一条 parquet 记录中的 `messages`：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "盒中有 3 袋铅笔，每袋 4 支，一共多少支？请把最终答案写在 #### 后面。"
    },
    {
      "role": "assistant",
      "content": "3 × 4 = 12。#### 12"
    }
  ]
}
```

这条 assistant message 就是监督目标。训练的 forward 会看到给定的整段序列，并在每个位置预测下一个 token；不会先自由生成一条回答，再把答案与 `12` 比较。`####` 在这里是目标文本的一部分，不触发 GSM8K reward scorer。

P0 依然便于识别样本，但 SFT 不依靠 RL 中的 `uid` 进行组内比较，也不创建 `P0_0_0` 到 `P0_3_0` 四条 session。原来错误的 `7`、`16` 两条 rollout 不会自动成为 SFT 数据。若把它们作为 assistant target 写入训练集，普通 SFT 也会学习它们，除非在数据准备阶段另行筛选或设计目标。

本页保留相同模型与机器规模，具体比较基线为：

| 配置                            | 本页 SFT 示例  | 对应含义                                  |
| ------------------------------- | -------------- | ----------------------------------------- |
| 模型                            | Qwen3-4B       | 与 RL 主线使用相同模型系列与 tokenizer    |
| 训练资源                        | 2 节点 × 4 GPU | FSDP，SP=1，训练 DP=8                     |
| `data.train_batch_size`         | 128            | 每次更新合计消费 128 条监督序列           |
| `data.micro_batch_size_per_gpu` | 4              | 每个 rank 一次 forward/backward 处理 4 条 |
| `data.use_dynamic_bsz`          | `false`        | 固定每个 micro-batch 的样本数             |
| `data.pad_mode`                 | `no_padding`   | Dataset / collator 保留变长序列           |
| `model.use_remove_padding`      | `true`         | 模型执行使用有效 token 的 packed 表示     |
| `data.max_length`               | 2048           | user 与 assistant 拼接后的总长度上限      |
| `data.truncation`               | `error`        | 超长样本显式报错，避免示例中静默截掉答案  |

SFT 使用独立的配置树，不能把 RL 的 `actor_rollout_ref.*` 参数原样搬过来。这里也不启动 rollout TP=2 的四个生成副本：这 8 张卡直接用于监督训练。两条路径可以共享模型和 engine，但配置字段的所属对象、每个 batch 的含义都需要重新确定。

## 八个进程怎样开始读取数据

入口是 [`verl.trainer.sft_trainer`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/sft_trainer.py#L457)，仓库 [Qwen3 SFT 示例](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/sft/gsm8k/run_qwen3_8b_fsdp.sh) 使用 `torchrun -m verl.trainer.sft_trainer` 启动它。多机示例还需要正确设置 rendezvous、节点数和每节点进程数；单节点脚本的 `--standalone` 不能直接代表本页两节点设置。

在这条入口中，八个进程执行相同的程序：初始化全局 process group，构造 `SFTTrainer`，再进入 `fit()`。它与第六章 RL driver 通过 Ray 调度 worker 的主路径有所不同，`TrainingWorker` 在这里由各进程直接创建并调用。

[`SFTTrainer.__init__`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/sft_trainer.py#L48) 先解析 model、engine、optimizer 等配置，创建 dataset 和 engine，然后根据 engine 提供的 data-parallel rank / size 创建 dataloader。先知道 DP 拓扑，才能决定这个进程应该读取哪一份样本。

`DistributedSampler` 在 DP 组内划分样本；本例每个 rank 的 dataloader batch 大小为：

$$
B_{\mathrm{local}}=\frac{\texttt{data.train\_batch\_size}}{D}
=\frac{128}{8}=16.
$$

128 能被 8 整除是本例固定的条件。源码这里使用整除，不能把任意不能整除的全局值都解释成恰好相同的消费量。sampler 和 dataloader 还都设置了 `drop_last=True`，尾部不足完整批次的数据不会以一个较小 batch 继续这条训练路径。

八个 rank 各拿到 16 条监督记录，合起来是 128 条。P0 只属于其中一个 rank；以下继续放大这个 rank 如何处理它。

## `messages` 怎样形成 `input_ids` 与 `loss_mask`

[`_build_dataset`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/sft_trainer.py#L200) 默认创建 `MultiTurnSFTDataset`。它按 `data.messages_key` 读取对话，默认字段名就是 `messages`。`__getitem__` 逐条 message 调用 `_process_single_message`，经 chat template 生成 token，然后把各轮拼接起来。

对 user message，`loss_mask` 全为 0；对 assistant message，先把 mask 置为 1，再把 assistant generation prompt 对应的前缀清零。模板里哪些结束标记被纳入目标，取决于实际 tokenizer 输出，因此不能只按人眼看到的答案文字数 token。

为了与 RL 主线保持直观对应，把 P0 缩成六个上下文 token、三个 assistant target token。这里三个 target 可以理解成包含结束标记在内的教学符号，它们不是上面字符串的真实 tokenization：

```text
位置           0  1  2  3  4  5 | 6  7  8
input_ids      p0 p1 p2 p3 p4 p5 | a0 a1 a2
attention_mask 1  1  1  1  1  1 | 1  1  1
loss_mask      0  0  0  0  0  0 | 1  1  1
```

六个上下文位置包括 user 内容和不作为监督目标的模板前缀。模型需要通过 attention 读取这些位置，才能预测正确答案；它们不直接作为 loss target。于是 `attention_mask` 与 `loss_mask` 可以在同一位置取不同值，两者分别回答“能否作为上下文参与计算”和“该 token 是否承担监督目标”。

[`_process_single_message`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/dataset/multiturn_sft_dataset.py#L187) 还会去掉后续 message 套模板时重复出现的默认 system prompt，避免每轮重复插入。拼接完成后，`sanity_check` 比较“逐条套模板再拼接”和“对完整对话套模板”的 token 是否一致。某些 thinking 模型模板会根据是否最后一轮插入不同内容，因此这个检查关系到真实训练序列，不能把忽略 mismatch 当作纯日志选项。

如果 P0 后面再接一轮 tool call 和 tool result，assistant token 会继续进入监督区域，tool role 的内容则作为上下文被 mask 掉。多轮 SFT 的目标来自 message role 和模板，而不是“序列中所有 token 都是模型应该学会生成的东西”。

## 四条监督序列怎样组成一个 micro-batch

`no_padding` 模式下，dataset 返回每条样本实际长度的 `input_ids`、`position_ids`、`loss_mask`。上面的 `attention_mask` 有助于解释概念，但这一分支不会把它作为最终输出字段；有效长度本身保存在变长结构里，engine 按执行方式使用或重建 attention 所需信息。

[`SFTTensorCollator`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/dataset/dataset_utils.py#L32) 将一个 rank 的 16 条记录组成 nested tensor。接下来固定 micro-batch size=4，把它分成四组。选取包含 P0 的一组，另外三条是不同问题 P1、P2、P3 的监督记录，并不是 P0 的另外三次 rollout。

假设这四条教学序列的上下文都是 6 个 token，target 长度为 `[3,2,4,1]`：

```text
样本              P0          P1          P2           P3
完整长度           9           8          10            7
有效 target 数     3           2           4            1

input_ids.values()：共 34 个 token
input_ids.offsets()：[0,9,17,27,34]

loss_mask.values()：同样 34 个位置，其中 10 个值为 1
position_ids：各样本内部从 0 开始
```

FSDP remove-padding 路径会把这个 micro-batch 的 token 输入整理为 `[1,34]`，student logits 的概念形状为 `[1,34,V]`。这里 V 是词表大小，34 是完整输入的有效 token 数，10 才是最终参与 loss 的 token 数。

物理上拼成一行，并不允许 P1 看到 P0 的内容。变长 attention 必须根据序列边界隔开不同样本；源码会保留各样本重置的 position IDs，并按模型实现传递或构造所需的序列长度信息。把 flatten 后的 token 直接按一段普通长文本做 attention，会改变监督问题本身。

静态 micro-batch 与 remove-padding 也不冲突：前者规定每次有 4 条序列，后者规定这 4 条序列在模型内部如何表示。即使每组都是 4 条，token 总数仍可不同。动态分批则会进一步按 token 预算调整每组的样本数，本页先固定它以便核算更新次数。

## 模型在第 5 个位置，为什么监督第 6 个 token

P0 的第一个目标 token 为 `a0`，它位于 `input_ids[6]`。因果语言模型在读完位置 5 的上下文后预测它，所以 `a0` 对应的 log-prob 位于模型输出的第 5 个位置。

这个关系可以逐列展开：

```text
当前位置          0   1   2   3   4   5   6   7   8
input_ids         p0  p1  p2  p3  p4  p5  a0  a1  a2
预测目标          p1  p2  p3  p4  p5  a0  a1  a2  无监督目标
原 loss_mask      0   0   0   0   0   0   1   1   1
左移后的 mask     0   0   0   0   0   1   1   1   0
```

engine 用左移后的输入 token IDs 抽取 next-token log-prob，[`sft_loss`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/utils/losses.py#L28) 则对 mask 做相同方向的移动。以下是 no-padding 分支的关键代码节选：

```python
log_prob_flatten = model_output["log_probs"].values()
loss_mask_flatten = data["loss_mask"].values()
loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
```

在前面的四条 packed 样本中，`torch.roll` 会把下一条样本的第一个 mask 移到上一条的最后位置。正常对话样本从非监督上下文开始，该值为 0，因此跨样本边界上的预测不会进入 loss；整个 packed 末尾回卷到起点也是同样的处理。attention 边界与 loss mask 仍是两套不同的保证：前者阻止读到另一条样本，后者阻止训练一个跨样本的 next-token target。

假设 P0 三个目标位置的 log-prob 分别为 `[-0.2,-0.7,-0.1]`，其负对数似然总和为 1.0。如果只评估 P0 这一个样本，token mean loss 是 `1.0 / 3`。但实际训练不能在每条样本或每个 micro-batch 内各自取平均后简单相加，因为各组有效 target 数量不同。

## 从一个 token 的 loss 到八卡的一次 optimizer update

进入 micro-batch 循环前，[`FSDPEngine.forward_backward_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L719) 对这个 rank 的整个本地 batch 计算 `loss_mask.sum()`，再在 DP 组执行求和。得到的是本次全局 128 条监督序列的有效 target 总数 `batch_num_tokens`。

为了演算，设八个 rank 合计有 1024 个有效 target token。这个数字是假设的监督 token 数，不是模型的序列长度上限。前面 P0 对其所在 rank 的 backward loss 的贡献为：

$$
\frac{1.0}{1024}\times8=0.0078125.
$$

这里乘 DP size 是为补偿随后 DP 梯度平均的缩放。八个 rank 的梯度平均后，P0 对全局 token mean 目标的贡献就回到 `1.0 / 1024`。每个 micro-batch 使用同一全局分母，因此切成几块不会把长回答和短回答人为赋予同样的权重。

对本页固定配置，一次更新的完整算术为：

```text
128 条监督序列
  → 8 个 DP rank，每个 16 条
  → 每个 rank 顺序执行 4 个 micro-batch，每个 4 条
  → 每次 backward 正常跨 DP 同步，并累计本次更新的梯度
  → 四次完成后，裁剪梯度并执行一次 optimizer step
```

[`BaseEngine.train_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/base.py#L113) 的顺序是 `zero_grad → forward_backward_batch → optimizer_step`，所以 4 个 micro-batch 不等于 4 次参数更新。SFT 配置同样加载 [`engine/fsdp.yaml`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/config/engine/fsdp.yaml#L61)，最终 `use_no_sync_for_gradient_accumulation=false`：每次 micro-batch backward 正常同步，累计到 mini-batch 末尾才更新参数。只有显式设置 `engine.use_no_sync_for_gradient_accumulation=true`，才会将梯度同步延后到最后一个 micro-batch；这项通信优化不改变全局 batch=128 的含义。

回到主线的 RL 设置，可以把数量关系并排比较：

| 阶段                             | 本页 SFT           | 主线 GRPO：B=128、G=4、M=32、E=1 |
| -------------------------------- | ------------------ | -------------------------------- |
| 数据入口                         | 128 条完整监督序列 | 128 个待生成回答的 prompt        |
| 更新使用的序列                   | 128 条             | 512 条 student trajectory        |
| 一次全局更新消费                 | 128 条             | 128 条 trajectory，即 M×G        |
| 每个 rank 每次更新               | 16 条              | 16 条 trajectory                 |
| 每次更新的本地 micro-batch       | 4 组，每组 4 条    | 4 组，每组 4 条                  |
| 一次 dataloader batch 对应更新数 | 1 次               | 4 次                             |

两边每次更新都恰好消费 128 条，这是所选数字形成的对照，不表示 `data.train_batch_size` 在 SFT 和 RL 中有相同的后续展开过程。SFT 没有 `rollout.n`、PPO mini-batch 和 PPO epochs；多次 dataset epoch 也不是 PPO 对同一批已生成 trajectory 的多遍优化。

## 相同的 TrainingWorker，为什么会优化不同目标

[`SFTTrainer._build_engine`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/sft_trainer.py#L157) 创建 `TrainingWorker` 后，通过 `set_loss_fn` 注入 `sft_loss`。RL actor 则注入 `ppo_loss`，启用蒸馏时还会切换到 `distillation_ppo_loss`。engine 负责模型、micro-batch、backward、通信和 optimizer，loss function 决定当前输出该怎样变成优化目标。

所以 SFT 需要的是 `input_ids`、位置及目标 mask；GRPO 还要读取 `old_log_probs`、`advantages`、可选的 reference log-prob。不能因为两者最后都拿到 token log-prob，就认为 loss 中只差一个函数名。

本页沿 FSDP 执行，其他 engine 也接在这个边界后。使用 Megatron 时，训练 DP size 要从实际 TP / PP 等进程组推导，不能再无条件用总 GPU 数 8 当作 DP；pipeline schedule 还会影响 micro-batch loss 的缩放。使用 AutoModel 等实现时，也需要核对它实际接受的布局与 checkpoint 格式。共享接口提供了比较的起点，具体张量和通信仍属于各后端的实现。

## 长度和 checkpoint 怎样影响后续学习

本页使用 `no_padding + truncation=error`，完整序列超过 2048 就报错。其他截断选项要结合实际分支理解：[`MultiTurnSFTDataset.__getitem__`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/dataset/multiturn_sft_dataset.py#L364) 的 right-padding 分支分别实现了 left、right、error；no-padding 分支在不报错时统一取 `[:max_length]`，即保留前部。即使配置写 `left`，这个分支也不会执行保留末尾的切片。

对 P0 而言，如果目标恰好处于被截掉的后部，模型可能只剩上下文或不完整答案可学。真实数据准备中应检查目标 token 数，而不只统计完整序列长度。右补齐模式下，补齐位置的 attention mask 和 loss mask 都为零；本页的变长模式则通过实际长度避免把补齐 token 放进存储。

一次更新完成后，`fit()` 推进 scheduler，按频率做验证或调用 `CheckpointHandler.save_checkpoint`。默认保存内容为 model、optimizer、extra；handler 同时关联 `StatefulDataLoader`，恢复时还需要对齐数据进度。八个 rank 的 sharded checkpoint 是训练恢复格式的一部分，不应默认等同于一个可直接交给任意推理服务的完整 Hugging Face 模型目录。

若用 SFT 的结果初始化 RL，先确认模型权重、tokenizer 和模板能够被 RL actor 及 rollout 后端一致加载。然后 RL trainer 才建立自己的 reference、rollout 权重版本和 optimizer / scheduler 状态。SFT checkpoint 本身不包含刚才四条 P0 rollout，也不提供后续 RL 需要的 reward、advantage 或 old-policy 状态。

现在回看 P0，三种训练的差别就落在具体数据上：SFT 学习预先给定的正确答案 token；GRPO 比较 student 四条回答的任务得分；[[projects/verl-source-reading/12-on-policy-distillation-and-mopd|12 On-Policy Distillation and MOPD]] 让 teacher 评价 student 已经走过的 token 前缀。它们可以复用同一模型执行底座，但监督信号从哪里来、在哪个位置对齐、怎样聚合，决定了模型实际学到的目标。
