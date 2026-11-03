---
title: 01 Sync GRPO Run Overview
created: 2026-09-02
published: 2026-09-02
modified: 2026-11-03
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - grpo
---

假设要训练一个模型回答数学问题。数据集提供题目和标准答案，模型先为每道题尝试四种回答，评分函数检查结果，再用这些回答更新模型。下一轮生成应当使用更新后的参数。verl 把这个循环分配给数据加载器、生成服务、奖励 worker 和训练 worker；阅读源码时，需要把它们处理的同一批样本认出来。

本篇跟踪一轮同步 GRPO。先建立贯穿后文的机器配置和样本，再沿 trainer 的实际执行顺序走完一次训练 step。02–06 分别深入这条路径的局部，继续使用这里的配置、样本标识和数值。

## 贯穿全文的案例

源码固定在 [revision 5b79827](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433)。以下是基于这份源码选择的教学配置，未执行多机 GPU 训练。配置数量用于推导控制流；后面的短 token 序列是便于手算的示意，不是 Qwen tokenizer 的实测输出。

| 设置        | 固定值                                                                 | 对这轮训练的影响                                              |
| ----------- | ---------------------------------------------------------------------- | ------------------------------------------------------------- |
| 模型、输入  | Qwen3-4B，纯文本数学问答                                               | 从同一个初始模型创建 actor 和 reference                       |
| trainer     | V1 `sync`，GRPO                                                        | 生成完成后训练，再同步权重                                    |
| 机器        | 2 个节点，每节点 4 张 GPU                                              | 全部使用同一个 8-GPU 训练资源池                               |
| 训练        | FSDP，`fsdp_size=-1`，Ulysses SP=1                                     | actor logical DP=8；8 个 rank 处理不同样本并协作维护模型      |
| 生成        | vLLM，TP=2、DP=1、PP=1，无 prefill/decode 分离                         | 4 个生成 replica，每个使用 2 张 GPU；与 actor 共用上述资源池  |
| 会话调度    | `rollout.agent.num_workers=8`                                          | 8 个 AgentLoop worker 各接收 16 道题，不代表 8 个模型副本     |
| 取样        | `train_batch_size=128`，`gen_batch_size=null`                          | 每轮一次取出 128 个 prompt                                    |
| 每题采样    | `rollout.n=4`                                                          | 单轮内置 AgentLoop 下产生 512 条 trajectory                   |
| actor 更新  | `ppo_mini_batch_size=32`，`ppo_epochs=1`，`shuffle=false`              | 全局每次更新消费 128 条 trajectory，共 4 次 optimizer update  |
| micro-batch | actor、old/ref log-prob 的本地 micro-batch 均为 4；关闭 dynamic batch  | 固定每次 forward/backward 的样本数                            |
| 梯度同步    | `use_no_sync_for_gradient_accumulation=false`                          | 每个 micro-batch 正常同步，累计四次 backward 后再更新一次参数 |
| token 表示  | `use_remove_padding=true`，`pad_to_length=false`                       | 模型使用有效 token 的 packed 输入，不额外补齐到固定 bucket    |
| reward      | GSM8K 规则评分，无 reward model                                        | CPU reward worker 评分，无额外 GPU 奖励模型                   |
| loss        | GRPO 按组标准差归一化；vanilla、token-mean、clip=0.2、dual-clip=3      | 同组回答互相提供 baseline                                     |
| KL 与采样   | temperature=1；actor KL=0.001，`low_var_kl`；entropy 系数=0            | reference 参与 actor loss，不向 task reward 加 KL             |
| 其他分支    | 无 critic、teacher、rollout correction、group filtering、失败组 refill | 先追踪全部 session 正常结束的路径                             |

`rollout.n=4` 和“4 个 vLLM replica”恰好都是 4。前者是每道题的采样次数，后者是服务容量，负载均衡器不保证把同一道题的四次请求各分配给一个 replica。

这里固定的每卡样本数也不保证任意 GPU 都有足够显存。显存取决于设备型号、序列长度、模型状态和实现；这套数字的用途是让每章的分批关系一致。完整配置入口见 [[projects/verl-source-reading/02-config-and-hydra|02 Launch and First Batch]]。

## 从 128 道题里选出 P0

选定这条人为构造的输入，使用与 GSM8K 数据相同的字段：

```python
{
    "prompt": [{
        "role": "user",
        "content": "盒中有3袋铅笔，每袋4支，一共多少支？请把最终答案写在 #### 后面。"
    }],
    "data_source": "openai/gsm8k",
    "reward_model": {"style": "rule", "ground_truth": "12"},
    "extra_info": {"index": 0}
}
```

模型看到题目，评分函数读取标准答案。`ground_truth` 不会自动拼到发给模型的 prompt 中。trainer 取出这一行时会生成 UUID；后文将该 UUID 简写为 `P0`。P0 是本轮样本身份，不是数据集永不变化的主键。

四个 session 的示例结果固定如下：

| trajectory key 的简写 | 示例回答                  | 规则得分 | 画图时的 response token 数 |
| --------------------- | ------------------------- | -------- | -------------------------- |
| `P0_0_0`              | `3 × 4 = 12。#### 12`     | 1        | 3                          |
| `P0_1_0`              | `3 + 4 = 7。#### 7`       | 0        | 5                          |
| `P0_2_0`              | `4 + 4 + 4 = 12。#### 12` | 1        | 4                          |
| `P0_3_0`              | `4 × 4 = 16。#### 16`     | 0        | 2                          |

表中最后一列只给后续张量图规定行长，不是对前一列字符串的真实分词结果。画图时再规定 prompt 有 6 个符号 token，因此四条完整序列长度为 `[9,11,10,8]`。

这些 key 分别表示 `{uid}_{session_id}_{output_index}`。单轮 AgentLoop 一次 session 返回一个 output，所以最后一位均为 0。四个独立 session 共享同一个 uid，后续 GRPO 据此找回一组回答。

## 启动阶段还没有回答

[main_ppo.main](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L166) 接收 Hydra 组合好的配置，检查 micro-batch 和角色需求，再选择 `TaskRunnerV1`。Ray 把这个 TaskRunner 放到远程 actor 进程，里面创建 `PPOTrainerSync`。

trainer 初始化时先准备 tokenizer、数据加载器和资源池，再初始化 actor/reference、reward worker、vLLM 服务和权重同步管理器。到这里模型已经可调用，但 P0 的回答还不存在。

reference 与 actor 从同一模型初始化，在本例中 reference 保持固定，actor 参与优化。rollout 是 actor 权重在生成 runtime 中的另一份执行表示：FSDP 训练出的参数需要被同步到 vLLM，生成服务才会使用它们。初始化末尾的 `PPOTrainerSync.on_init_end` 先做一次同步，避免第一轮从未加载的权重开始生成。

这一部分的代码在 [PPOTrainer.\_setup](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L232) 与 [同步 trainer hooks](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_sync.py#L24)。多节点进程和资源池的细节留到 06；下面先进入第一轮训练。

## 先把 128 个 prompt 交给生成侧

`prepare_step` 调用 `_next_train_batch`，得到 128 行原始消息和 metadata。此时是 128 个问题，还没有 512 个回答。`_submit_batch_to_rollout` 的 sync 分支做两件事：

```python
tq.kv_batch_put(
    keys=list(batch["uid"]),
    partition_id="train",
    tags=tags,
)
self.agent_loop_manager.generate_sequences(batch)
```

这是 [源码 sync 分支的节选](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1403)。第一处给每个问题登记状态，P0 对应 `is_prompt=True, status=pending`；第二处把包含实际消息的 batch 交给 AgentLoop。

TransferQueue（下文简称 TQ）在这里既有“问题是否完成”的状态，又会保存后续 trajectory 的真实张量。两者有不同的 key：`P0` 跟踪整组状态，`P0_0_0` 保存第一条回答及其训练字段。controller 不必把每个新字段都通过一次 Python 返回值传回来，只需持续持有这些 key。

## 每道题开始四次独立生成

生成侧的 `_run_prompt` 读取 `rollout.n=4`，为同一道题创建四个 session task。每个 task 进入 `SingleTurnAgentLoop`，将原始消息转换成 prompt token，向 vLLM 发起生成，再把输出包装成 `AgentLoopOutput`。

```text
P0
 ├─ session 0 → prompt token → generate → response 0
 ├─ session 1 → prompt token → generate → response 1
 ├─ session 2 → prompt token → generate → response 2
 └─ session 3 → prompt token → generate → response 3
```

这个图描述逻辑任务，不规定它们实际的完成顺序或 server 分配。整批 128 个 prompt 同样展开为 512 个 session；本例每个 session 生成一个 output。

生成结果进入后处理时，规则 reward worker 解码回答，取出 `####` 后的数字，与 `ground_truth="12"` 比较，P0 得到 `[1,0,1,0]`。随后 `AgentLoopOutput.as_dict` 将 score 写到 response 长度的最后一个位置，再将各条 output 的字段写入 TQ。此时已有：

```text
P0_0_0:
  prompts          6 个示意 token
  responses        3 个示意 token
  input_ids        9 个示意 token
  response_mask    [1,1,1]
  rm_scores        [0,0,1]
```

四个 session 的任务都 settle 后，P0 的状态才成为 `finished`。这里先假设全部成功；有失败任务时状态与可采样轨迹的关系需要额外处理，04 会展开。

生成本身见 [[projects/verl-source-reading/03-data-protocol-and-batch-lifecycle|03 From Prompts to Trajectories]]，评分、写回和组成训练 batch 见 [[projects/verl-source-reading/04-rollout-reward-and-agent-loop|04 Reward and Training Batch]]。

## Trainer 收齐数据后，先算概率再算 advantage

`ReplayBuffer.sample` 等待本轮所需的 prompt group，返回 512 条 trajectory 的 `KVBatchMeta`。它描述 partition、keys 和 tags；计算所需的 tensor 由下游按 key 从 TQ 取出。

[\_step_once](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L540) 在本例实际走的主干可节选为：

```python
batch = self._balance_batch(batch, metrics=metrics)
batch = self._compute_old_log_prob(batch, metrics=metrics)
batch = self._compute_ref_log_prob(batch, metrics=metrics)
batch = self._compute_advantage(batch, metrics=metrics)
batch = self._update_actor(batch, metrics=metrics)
```

这里省略了计时和条件判断：reference 分支因 actor KL 开启而执行，critic 分支因本例关闭而跳过。rule reward 已经在生成后处理阶段完成；只有奖励模型与训练资源共用时，trainer 才要另外调用 colocated reward 分支。

`_balance_batch` 按序列长度重排，使八个训练 rank 的 token 工作量更接近。P0 的四条回答可能因此不再相邻。接着 actor engine 在更新前重算 sampled tokens 的概率并保存为 `old_log_probs`，reference 计算 `ref_log_prob`，最后才计算 advantage。

为什么 reward 已有 0/1 还要 advantage？因为 GRPO 要问的是某条回答相对于同题其他回答表现如何。P0 的组内均值为 0.5，源码默认使用样本标准差，四个 advantage 约为：

```text
[+0.866, -0.866, +0.866, -0.866]
```

每条回答的值再扩展到它的有效 policy token。05 会从 `rm_scores` 的矩阵开始手算，并解释它怎样与 `old_log_probs`、current log-prob 组成 loss。

## 512 条 trajectory 怎样变成四次参数更新

`_update_actor` 中有两行决定更新粒度：

```python
ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
```

因此配置的 32 变为传给 worker 的全局 `mini_batch_size=128`。这说明一次逻辑优化消费 128 条 trajectory，却不表示 controller 发起四次各带 128 条数据的调用。

实际顺序是整批 dispatch，再在 worker 内切分：

```text
controller: update_actor(512 条 trajectory 的 metadata)
  │
  ├─ rank 0: 64 条 → 4 个本地 mini-batch，每个 16 条
  ├─ rank 1: 64 条 → 4 个本地 mini-batch，每个 16 条
  ├─ ...
  └─ rank 7: 64 条 → 4 个本地 mini-batch，每个 16 条

同一 mini-batch 迭代：
  每个 rank 的 16 条 → 4 个 micro-batch，每个 4 条
  八个 rank 合计消费 8 × 16 = 128 条
  累积梯度，并通过 FSDP 通信完成一次逻辑 optimizer update
```

整个 step 有四次这样的更新。每个 rank 上调用四次 optimizer，并不等于全局独立更新 32 次；它们共同更新同一个分布式 actor。

P0 的四条回答用于 advantage 分组时必须能关联，但 actor 训练不要求它们留在同一个 mini-batch。advantage 已经按 key 写回，训练时每条 trajectory 携带自己需要的数值。不同 mini-batch 之间 current actor 参数会变，预先保存的 old log-prob、reference log-prob 和 advantage 不变。

具体代码见 [\_update_actor](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1734) 与 [TrainingWorker.train_mini_batch](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine_workers.py#L241)。

## 更新完训练模型，生成服务还需要换权重

用 `θ₀` 表示本 step 开始时 actor 的参数，四次 optimizer update 得到 `θ₁、θ₂、θ₃、θ₄`。这些下标只用于本文说明更新次数，不是源码中的 `global_steps`：

| 时刻                         | actor 参数 | 本轮训练数据                     | rollout              |
| ---------------------------- | ---------- | -------------------------------- | -------------------- |
| 生成阶段                     | θ₀         | 逐步产生 512 条 trajectory       | 使用与 θ₀ 同步的权重 |
| old/reference/advantage 计算 | θ₀         | 逐步补齐字段                     | sample 结束后 sleep  |
| 第一次 mini-batch 更新后     | θ₁         | 保留原 old log-prob 和 advantage | 仍处于暂停阶段       |
| 第四次 mini-batch 更新后     | θ₄         | 本 step 的优化完成               | 尚需同步新权重       |
| `on_step_end` 后             | θ₄         | 统计指标并清理本批字段           | 使用新权重准备下一轮 |

[PPOTrainerSync.on_sample_end](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_sync.py#L39) 让 rollout replicas sleep，`on_step_end` 则调用 checkpoint manager 同步权重。本例的 sync trainer 使用 naive 同步路径。传入同步管理器的 step 标记不等于本 step 内四个 optimizer 更新的编号。

最后，[fit](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L389) 在必要的 checkpoint、权重同步和验证之后读取指标、按配置导出 rollout，再用本批 trajectory keys 清理 TQ。样本的内容生命周期到这里结束；更新后的模型参数则留给下一轮。

下一篇 [[projects/verl-source-reading/02-config-and-hydra|02 Launch and First Batch]] 回到这轮运行的起点：把上面的配置落实到 Hydra，读取 P0 所在的第一批数据，并解释为什么提交给 AgentLoop 时仍然是 128 个问题。
