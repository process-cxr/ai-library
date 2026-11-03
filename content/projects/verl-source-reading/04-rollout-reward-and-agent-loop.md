---
title: 04 Reward and Training Batch
created: 2026-09-02
published: 2026-09-02
modified: 2026-11-03
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - reward
  - dataflow
  - batching
---

P0 的 session 0 已经生成了“`3 × 4 = 12。#### 12`”，返回的 `AgentLoopOutput` 还没有 reward。现在要解决三个连续的问题：规则函数怎样给这条回答打分，分数怎样与 token 一起写回 TransferQueue，以及 trainer 怎样收齐本轮需要的 512 条 trajectory。

本章接续 [[projects/verl-source-reading/03-data-protocol-and-batch-lifecycle|03 From Prompts to Trajectories]] 的同一批样本。仍然使用 2 个节点、每节点 4 张 GPU，actor DP=8；128 个 prompt 各生成 4 条回答，PPO mini-batch 参数为 32。先沿所有 session 成功、每个 session 一个 output 的单轮路径讲完，再讨论失败和 reward model 分支。调用关系固定于项目首页记录的 revision，数值与缩尺 token 均为教学示例。

## 先给这一条回答评分，不必等满 512 条

V1 的 [`_agent_loop_postprocess`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L150) 收到 session 0 的 output 后，先把它规范为列表，再执行：

```python
outputs = output if isinstance(output, list) else [output]
await self._compute_score(outputs, kwargs=kwargs)
```

这里 `outputs` 的长度是 1。它表示**当前 session 的输出**，不包含 P0 的另外三个 session，更不包含全部 512 条 trajectory。支持列表是为了容纳一个 session 返回多个输出的 AgentLoop；不能把这个列表维度直接认成 GRPO 的 group 维度。

基线关闭 reward model，使用 `naive` reward manager 和默认规则函数。根据 [`RewardLoopManager.reward_loop_worker_handles`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/reward_loop/reward_loop.py#L313)，这种配置会向 AgentLoop 暴露 reward worker handles，评分可以与其他尚未结束的生成请求并行。

[`_compute_score`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/agent_loop.py#L937) 检查 `final_output.reward_score is None`，随后把 tokens 暂时装进一个 `DataProto`，发送给选中的 reward worker。使用上一章的缩尺长度，当前单输出 session 的临时张量为：

| 字段             | 内容示意           | 形状    |
| ---------------- | ------------------ | ------- |
| `prompts`        | `p0 … p5`          | `[1,6]` |
| `responses`      | `a0 a1 a2`         | `[1,3]` |
| `input_ids`      | `p0 … p5 a0 a1 a2` | `[1,9]` |
| `attention_mask` | 9 个 1             | `[1,9]` |
| `position_ids`   | `0 … 8`            | `[1,9]` |

这些张量在后处理端构建，尚不是 actor 的 GPU micro-batch。函数用 `pad_sequence` 为一个 session 的 outputs 建 batch；当前只有一条输出，所以并没有额外 pad token。`data_source`、`reward_model`、`raw_prompt`、`uid` 等非张量信息通过 `non_tensor_batch` 同时传入。

这里用的是 attention mask：它说明输入序列中哪些位置是真实 token。当前单轮案例的 response 全由模型生成，因此 response 区域的 attention mask 和 `response_mask` 都是 1。到了工具调用，这两个概念就会分开：工具返回的文字是真实上下文，attention 需要看到它，policy loss 却不应把它当成模型采样的动作。

## 从 `a0 a1 a2` 还原回答，再选择 GSM8K 评分规则

reward worker 在关闭 reward model 的分支调用 [`NaiveRewardManager.run_single`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/reward_loop/reward_manager/naive.py#L34)。其关键操作依次是：

```python
data = data[-1:]
data_item = data[0]
response_ids = data_item.batch["responses"]
response_length = response_ids.shape[-1]
valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
valid_response_ids = response_ids[:valid_response_length]

data_source = data_item.non_tensor_batch["data_source"]
ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
```

第一行意味着这个 manager 对一个 session 的多输出只依据最后一条评分。主线只有一个 output，因此选中的仍是 session 0。接着按 attention mask 取有效 response token，并通过 tokenizer decode，得到文本“`3 × 4 = 12。#### 12`”。

需要注意，缩尺的 `[a0,a1,a2]` 只示意 token 位置；并没有声称这段中文文本真的编码成 3 个 token。实际评分器接收的始终是模型真实 token 序列 decode 后的字符串。

此时调用评分函数所需的主要参数已经完整：

```python
data_source = "openai/gsm8k"
solution_str = "3 × 4 = 12。#### 12"
ground_truth = "12"
```

[`default_compute_score`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/reward_score/__init__.py#L44) 根据 `data_source` 进入 `gsm8k.compute_score`。GSM8K 的默认模式是 strict，它在回答尾部匹配 `#### 数字`，取最后一个匹配结果，去除逗号后与字符串 ground truth 比较。源码中的关键判断是：

```python
answer = extract_solution(solution_str=solution_str, method=method)
if answer is None:
    return 0
else:
    if answer == ground_truth:
        return score
    else:
        return format_score
```

默认 `score=1.0、format_score=0.0`。因此我们的四条回答分别得到：

| trajectory                          | 解析出的答案 | `ground_truth` | reward |
| ----------------------------------- | ------------ | -------------- | -----: |
| `P0_0_0`：`3 × 4 = 12。#### 12`     | `"12"`       | `"12"`         |      1 |
| `P0_1_0`：`3 + 4 = 7。#### 7`       | `"7"`        | `"12"`         |      0 |
| `P0_2_0`：`4 + 4 + 4 = 12。#### 12` | `"12"`       | `"12"`         |      1 |
| `P0_3_0`：`4 × 4 = 16。#### 16`     | `"16"`       | `"12"`         |      0 |

这个函数核对的是答案格式与答案字符串，没有逐步检查推理过程。即使文字中出现了正确的 12，缺少要求的 `#### 12` 也可能得 0 分；`"12.0"` 与 `"12"` 也不是自动按数学等价比较。理解实际评分规则，才能理解 policy 最终在优化什么。

完整逻辑见 [`gsm8k.extract_solution / compute_score`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/reward_score/gsm8k.py#L20)。其中只检查末尾 300 个字符，这是字符限制，并非精确的 tokenizer token 限制。

NaiveRewardManager 把标量返回成 `{"reward_score": 1, "reward_extra_info": {"acc": 1}}`。如果自定义规则返回字典，则要求其中有 `score`，其余信息也会保留。结果回到 AgentLoop 后，被写入 `final_output.reward_score` 与 `final_output.extra_fields["reward_extra_info"]`。回答本身没有被改写，只增加了评分结果。

## 标量 1 怎样变成 `[0,0,1]`

评分完成后，V1 后处理继续把 `AgentLoopOutput` 转成准备存储的字段。这里真正产生 token reward 的位置是 [`AgentLoopOutput.as_dict`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/agent_loop.py#L114)：

```python
output["prompts"] = torch.tensor(output.pop("prompt_ids"), dtype=torch.int64)
output["responses"] = torch.tensor(output.pop("response_ids"), dtype=torch.int64)
output["response_mask"] = torch.tensor(output.pop("response_mask"), dtype=torch.int64)

reward_score = output.pop("reward_score", None)
if reward_score is not None:
    rm_scores = torch.zeros_like(output["response_mask"], dtype=torch.float32)
    rm_scores[-1] = reward_score
    output["rm_scores"] = rm_scores
```

session 0 的 response 长度为 3，因而先生成长度 3 的浮点零向量，再把最后一项设为 1：

```text
response_mask [3] = [1, 1, 1]
reward_score      = 1
rm_scores     [3] = [0, 0, 1]
```

源码使用的是 `[-1]`，即当前**未 padding response 的最后一个位置**。主线单轮生成中，它也是最后一个有效 policy token。不能把这段实现推广成“寻找 response_mask 中最后一个 1”；在带 observation 的变体里，这两个位置未必相同。

`rm_scores` 这个字段名也不表示一定运行了神经网络 reward model。我们刚刚只做了字符串匹配，得到的规则奖励照样写进 `rm_scores`。

为了观察四条回答的关系，下面暂时把 P0 的四条 response 单独补到最大长度 5。**这是局部教学矩阵，不是前面评分 RPC 的实际 batch 形状，也不是 TQ 强制采用的存储形状。**

```text
                    response_mask [4,5]        rm_scores [4,5]
P0_0_0, length 3     1 1 1 0 0                 0 0 1 0 0
P0_1_0, length 5     1 1 1 1 1                 0 0 0 0 0
P0_2_0, length 4     1 1 1 1 0                 0 0 0 1 0
P0_3_0, length 2     1 1 0 0 0                 0 0 0 0 0
```

第二条回答的 reward 是 0，整行 `rm_scores` 因而为 0，但它仍然有 5 个有效 token。有效长度必须由长度信息或 mask 确定，不能通过“找到 reward 的非零位置”反推。

把 `rm_scores` 沿最后一维求和，会恢复每条回答的分数 `[1,0,1,0]`。05 的 GRPO 将用这四个分数做组内比较，再把结果传播到有效 policy token。当前只完成了奖励的表示，还没有生成 advantage。

## 一条回答写入 TQ 时，有字段，也有可查询的 tag

回到 [`AgentLoopWorkerTQ._agent_loop_postprocess`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L177)。它用 prompt 与 response 拼出完整输入，并计算 position ids：

```python
prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
responses = torch.tensor(output.response_ids, dtype=torch.int64)
input_ids = torch.cat([prompts, responses], dim=0)
attention_mask = torch.ones_like(input_ids, dtype=torch.int64)

keys.append(f"{uid}_{session_id}_{i}")
field = output.as_dict()
field.update(kwargs)
field["loss_mask"] = field["response_mask"]
field["input_ids"] = input_ids
field["position_ids"] = position_ids
```

节选省略了 position ids 的调用和多模态分支；对本例纯文本，位置从 0 连续编号到 8。这里的 attention mask 是计算 position ids 的临时变量，不能据此认定后处理一定把一个同名完整 mask 字段写入 TQ。

`field.update(kwargs)` 保留原始 `uid`、`session_id`、dataset 附加信息和 step。于是同一条存储记录既有回答，也有回答属于哪道题的信息。主要字段与 tag 可以分开理解：

| `P0_0_0` 的 fields            | 缩尺内容与类型                  |
| ----------------------------- | ------------------------------- |
| `prompts` / `responses`       | `int64 [6]` / `int64 [3]`       |
| `input_ids` / `position_ids`  | `int64 [9]` / `int64 [9]`       |
| `response_mask` / `loss_mask` | `int64 [3]`，此处相同           |
| `rm_scores`                   | `float32 [3] = [0,0,1]`         |
| `rollout_log_probs`           | 若开启，`float32 [3]`           |
| `uid` / `session_id`          | `P0` / `0`                      |
| `extra_fields`                | reward 附加信息、生成版本与统计 |

tag 则保存 `status="success"`、`prompt_len=6`、`response_len=3`、`seq_len=9`，以及 step/version 信息。`global_steps` 来自这道题进入生成流程时的 step，`min_global_steps/max_global_steps` 来自生成侧返回的权重版本。同步主线通常只有一个生成版本，但字段含义不能互换，异步时它们会成为重要差别。

fields 存数据，tags 允许 controller 先查询状态、长度和版本，而不把全部 token 搬过来。最终调用是：

```python
await tq.async_kv_batch_put(
    keys=keys,
    fields=list_of_dict_to_tensordict(fields),
    tags=tags,
    partition_id="train" if not validate else "val",
)
```

对当前 session，这次写入只有 `P0_0_0`。另外三个 session 会分别写入自己的 key；它们不需要按 session 编号顺序完成。

## 变长存储与临时矩阵为什么会同时出现

四条回答的真实长度不同，TQ 不要求它们在写入前一律补到 1024。调用端的 [`list_of_dict_to_tensordict`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/tensordict_utils.py#L918) 会对同形张量 `stack`，不同长度的张量用 jagged nested tensor 表示；一次只写一个 output 时，同形判断自然成立。后续 V1 从 TQ 批量读取序列字段时，按 nested 接口取长度和内容。

所以更准确的理解是“按 key 保存每条序列的内容和边界”，而不是“每次 put 必定已经构造四行 nested tensor”。当把 P0 的四条 response 组织成 jagged 表示时，可以画成：

```text
values = [a0 a1 a2 | b0 b1 b2 b3 b4 | c0 c1 c2 c3 | d0 d1]
offsets = [0, 3, 8, 12, 14]
lengths = offsets[1:] - offsets[:-1] = [3, 5, 4, 2]
```

`offsets[1]=3` 表示第一条序列结束，`offsets[2]=8` 表示第二条结束。14 是这些 response 的有效 token 总数，没有把 padding 存成第五个位置的需要。

同样，完整 `input_ids` 的四条长度是 `[9,11,10,8]`，对应 offsets `[0,9,20,30,38]`。两个 offsets 数组不同，因为一个只记录 response，另一个还包含 prompt。05 中从完整序列的 log-prob 提取 response 段，必须尊重这个差别。

算法需要规则矩阵时再调用 `to_padded_tensor()`。P0 单独演示是 `[4,5]`；实际训练批次如果包含 512 条 response，补齐形状为 `[512,R_max]`，其中 `R_max` 取决于**这次读取的整批**最大有效 response 长度。配置上限 1024、小例子的 5 和该批实际 `R_max` 是三种不同的量。

## P0 什么时候才能被 trainer 取走

一条 trajectory 的 tag 是 `success`，还不足以说明它所属的 prompt group 已经完成。外层 [`_run_prompt`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L131) 等待四个 session，包括评分和 TQ 写回，全部 settle：

```python
session_errors = await _settle_session_tasks(tasks)
if session_errors:
    status = "failure"
else:
    status = "finished"
await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})
```

正常情况下，P0 的状态变为 `finished` 时，`P0_0_0 … P0_3_0` 已经写好。这里的等待还保护了失败清理：不能刚发现一个 session 异常，就把整个 group 清掉，而其余 session 随后又往已清理的 group 写入。

trainer 在 [`_step_once`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L540) 中调用 `ReplayBuffer.sample`。当前要收集的 `batch_size=128` 指 prompt groups。ReplayBuffer 查询 TQ 元信息，等待足够的可采样 group，选择 128 个 prompt uid，然后收集属于这些 uid 的 trajectory keys。

在本例所有生成成功、每 session 单输出的前提下，数量因此是：

```text
选择的 prompt groups   128
每组 session 数          4
每 session outputs 数    1
返回 trajectory keys    512
```

[`_materialize_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/replay_buffer.py#L377) 返回 `KVBatchMeta(partition_id, keys, tags)`。名字中的 materialize 指组装这份 batch 描述，它没有在这里读取 512 条回答的所有 tensor fields。

函数还清理已选中的 **prompt keys**，例如 `P0`，因为 group 已经被消费；`P0_0_0` 等 trajectory keys 仍保留给概率计算、advantage 和训练。后面训练与指标计算结束才清理 trajectory 数据，这两次清理不是同一件事。

## 在送入 8 个训练 rank 之前，为什么要重排样本

512 条 trajectory 可以平均分成 8 份，每份 64 条。但每份样本数相同，不表示 token 工作量相同：如果某个 rank 恰好拿到大量长回答，其他 rank 可能在同步时等它。

因此 `_step_once` 取到 keys 后调用 [`_balance_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1515)。它先查询 **actor 的 DP 映射**，得到本例的 8；rollout 的 TP=2 不参与这个计算。随后执行数量补齐和重排：

```python
batch_multiple = self._get_required_batch_multiple(dp_size)
batch = upsample_batch_to_divisible_size(batch, batch_multiple, self.tokenizer.eos_token_id)
global_seqlen_lst = torch.tensor([tag["seq_len"] for tag in batch.tags], dtype=torch.int64)
workload_lst = calculate_workload(global_seqlen_lst)
global_partition_lst = get_seqlen_balanced_partitions(
    workload_lst, k_partitions=dp_size, equal_size=True
)
batch.reorder([j for partition in global_partition_lst for j in partition])
```

无 critic 的当前配置要求兼顾 DP 和 actor mini-batch：全局 PPO mini-batch 消费 `32 × 4 = 128` 条 trajectory，因此需要的倍数是 `lcm(8,128)=128`。512 已经整除，不添加 synthetic sample。这里的数量补齐与序列内部补 pad token 是两种不同操作。

`calculate_workload` 使用 `24576 × L + L²` 作为相对工作量，其中 `L` 是 `seq_len`。这个近似式来自 dense Transformer 的线性项和 attention 二次项，不是对 Qwen3-4B 实测出的 FLOPs。使用它是为了让各 rank 的预计工作量接近。

重排操作同步改变 key 与 tag 的顺序。例如 P0 的四条 trajectory 原本相邻，重排后可能散到不同位置。下面约定一组从 0 开始计数的示意位置，05、06 继续沿用；**它不是没有其他 508 条长度信息时就能算出的真实分配结果**：

| 示意 batch 位置 | key      | 字段中的 uid | reward |
| --------------: | -------- | ------------ | -----: |
|              16 | `P0_0_0` | P0           |      1 |
|             144 | `P0_1_0` | P0           |      0 |
|             272 | `P0_2_0` | P0           |      1 |
|             400 | `P0_3_0` | P0           |      0 |

位置变化不会把 P0 的 1 分写到另一道题上，因为读取与回写都按同一批 keys 进行；GRPO 分组依据 `uid`，也不依赖四行紧挨着。后面将整批分给 8 个 rank 时，每份 64 条，表中四条分别落到 rank 0、2、4、6，各自位于本地第 16 行。本例不 shuffle，因此它们在本地第二个 mini-batch 的开头。再约定全批第 17、18、19 行为 `P17_2_0`、`P42_1_0`、`P85_3_0`，05 就可以沿 rank 0 的这四条样本讲一次具体 micro-batch。

## 不满足主线假设时，哪些地方会变化

理解正常路径后，可以准确界定“128 个 prompt 产生 512 条 trajectory”的适用条件。

**失败状态不等于自动丢弃。** 当前 sync ReplayBuffer 的默认可采样终态包含 `finished` 和 `failure`，再扣除启用的淘汰策略。一个 group 中部分 session 失败时，已经成功写回的 trajectory 仍可能进入训练，因此实际数量可能少于 512。后续 padding 补的是 mask 为零的 synthetic samples，以满足整除要求；它没有替模型补出真正的回答。若选中的所有 group 都没有任何可物化 trajectory，源码会报错。

`sync_refill_failed_groups` 是额外开启的策略，用于替换没有可物化 trajectory 的失败 group，且该 revision 要求 `gen_batch_size=1`。它不是本例默认启用的行为。DAPO group filtering、异步 staleness filtering 也各有条件，不能仅凭出现 `ReplayBuffer` 就推断它们都会发生。相关选择逻辑见 [`_terminal_eviction_reasons / _sampleable_terminal_keys`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/replay_buffer.py#L300)。

**reward model 共用训练资源时，评分时机后移。** 此时 AgentLoop 拿不到可并行的 reward worker handles，trajectory 可以先写入；trainer 收齐后通过 [`_compute_reward_colocate`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1436) 读取数据，依据 prompt/response 长度重建 attention mask，调用 `compute_rm_score`，再把 `rm_scores` 写回原 keys。这条批处理分支才调用 reward manager 的 `assemble_rm_scores`；主线规则评分使用的是前面讲过的 `AgentLoopOutput.as_dict`。

**自定义 reward 与多输出 AgentLoop 会改变评分内容。** custom reward function 优先于默认 dataset 路由；已有 `reward_score` 的 output 可以跳过重复计算。一个 session 返回多个 outputs 时，默认 NaiveRewardManager 看最后一条，TQ 后处理再把最终 reward 复制到前面的 outputs。此时每个 session 不再必然对应一条训练记录，不能继续机械使用 `128 × 4 = 512`。多轮工具交互与多输出也不是同一个概念：默认 ToolAgentLoop 可把多个 turn 合在一条 output 中，见 [[projects/verl-source-reading/11-agentic-rl-extension|11 Agentic RL Extension]]。

## 现在训练端拿到了什么

沿主线走到这里，trainer 持有的是重排后的 512 个 trajectory keys 及其长度 tags。每个 key 对应的 TQ 数据已有 tokens、mask、规则 reward 和可选 rollout log-probs；P0 的四个 reward 仍然是 `[1,0,1,0]`，与它们处于哪个 rank 或列表位置无关。

接下来 [`_step_once`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L562) 会先计算 `old_log_probs`，再计算本例启用的 `ref_log_prob`，然后计算 advantage。因为 `use_kl_in_reward=false`，规则 reward 不会在这里先扣除 reference KL；本例的 KL 约束在 actor loss 中以系数 0.001 加入。

[[projects/verl-source-reading/05-advantage-and-policy-loss|05 Advantage and Policy Loss]] 继续使用这些 keys，解释 token 概率怎样与 response 对齐，以及 `[1,0,1,0]` 为什么最终能让正确回答的 token 获得正的训练信号。
