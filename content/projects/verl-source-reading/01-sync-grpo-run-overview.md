---
title: 01 Sync GRPO Run Overview
created: 2026-09-02
published: 2026-09-02
modified: 2026-10-10
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - grpo
---

这篇文章只讲一件事：在当前固定 revision 中，一条 prompt 怎样经过 `sync GRPO + FSDP + vLLM + rule-based reward`，最后变成一次 actor update。

可以把它当成后面所有专题的底图。先看懂这条最小闭环，再去理解 PPO、MOPD、Agentic RL 或 async trainer，会容易很多。文中涉及的调用关系来自固定源码；GPU shape、显存和耗时等运行现象，需要实际运行后再确认。

## 从哪里开始

第一轮使用 [examples/grpo_trainer/run_qwen3_4b_fsdp.sh](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/grpo_trainer/run_qwen3_4b_fsdp.sh) 作为入口。脚本通过环境变量接收模型、数据和设备配置，再组织成 Hydra overrides。与本页相关的配置意图包括：

- `algorithm.adv_estimator=grpo`：使用 GRPO advantage estimator；
- `data.train_batch_size`、`data.max_prompt_length`、`data.max_response_length`：控制 prompt 取样规模和长度边界；
- `data.filter_overlong_prompts=True`、`data.truncation=error`：超长 prompt 在数据阶段过滤，超过配置边界的序列不静默截断；
- `actor_rollout_ref.actor.use_kl_loss=True`：actor update 中保留 reference-policy KL loss，因此需要 reference log-prob；
- `actor_rollout_ref.actor.fsdp_config.*_offload=False`：主 actor FSDP 路径不启用参数和 optimizer offload；
- `actor_rollout_ref.actor.use_dynamic_bsz=True`：训练侧按 token workload 使用动态 batch 控制；
- `actor_rollout_ref.rollout.name=vllm`、`rollout.n`：由 vLLM 生成每个 prompt 的多条候选 trajectory；
- `actor_rollout_ref.rollout.free_cache_engine=True`：同步权重前后配合 rollout replica 的 sleep / wake 生命周期；
- `actor_rollout_ref.ref.fsdp_config.param_offload=True`：reference 路径可以使用参数 offload。

这些参数告诉我们这次阅读关注哪条路径。真正运行时的值还要以 Hydra resolve 后的完整 config 为准。

### 这一轮有哪些角色

```text
Hydra driver
  └─ Ray TaskRunnerV1
       ├─ PPOTrainerSync                 training orchestration
       ├─ ActorRolloutRefWorker          actor / reference model engine
       ├─ AgentLoopWorkerTQ              vLLM-backed agent generation
       ├─ RewardLoopWorker               rule-based reward
       └─ CheckpointEngineManager        actor -> rollout weight lifecycle
```

`sync` 描述的是训练节奏和权重更新顺序，不代表所有步骤都在同一个 Python 进程中执行。driver、Ray actor、训练 worker、rollout worker 和 TransferQueue 各自承担不同的工作。

### 为什么先固定 FSDP

为了先把主链看清楚，这里固定使用 FSDP。`actor_rollout_wg.update_actor(batch)` 之前的 prompt、trajectory、reward 和 advantage 属于 trainer 的公共语义；进入 model engine 之后，batch dispatch、forward/backward、梯度同步、optimizer 和 checkpoint 则按 FSDP 的执行方式展开。

```text
train_batch_size                     prompt group 数
train_batch_size × rollout.n         常规情况下的 trajectory 数
ppo_mini_batch_size × rollout.n      actor 的全局 PPO mini-batch
```

进入 model engine 后，同一个全局 PPO mini-batch 会依据 backend topology 形成不同的本地执行布局：

```text
FSDP:
  actor DP = world_size / Ulysses SP

Megatron, static CP:
  actor DP = world_size / (TP × PP × CP)
```

FSDP 的 `fsdp_size` 控制参数 shard group，不能直接等同于 actor 的 logical DP；Megatron 的 TP、PP、CP 会占用 model-parallel 维度，EP 还会建立 expert-specific groups。`actor_rollout_ref.rollout.tensor_model_parallel_size` 则属于 rollout backend，不参与上述 actor DP 计算。FSDP 与 Megatron 的完整对照放在 [[projects/verl-source-reading/07-model-engine-and-backend|07 Model Engine and Backend]]。

源码入口：

- [FSDP `get_data_parallel_size`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L646)
- [Megatron process-group 初始化](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/megatron/transformer_impl.py#L239)
- [Megatron `get_data_parallel_size`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/megatron/transformer_impl.py#L750)

## 一次训练是怎样启动的

### 启动与 trainer 构造

```text
run_qwen3_4b_fsdp.sh
  -> python3 -m verl.trainer.main_ppo
  -> @hydra.main(config_path="config", config_name="ppo_trainer")
  -> main(config)
  -> validate_config(...)
  -> run_ppo(config, TaskRunnerV1)
  -> ray.init(...)
  -> TaskRunnerV1.remote().run(config)
  -> get_trainer_cls(config.trainer.v1.trainer_mode)
  -> PPOTrainerSync(config)
```

`main` 先检查当前配置是否需要 reference policy 和 critic，再根据 `trainer.use_v1` 选择 V1 或 legacy trainer。当前主线进入 V1：`run_ppo` 初始化 Ray，随后由 Ray actor 执行 `TaskRunnerV1.run`。

源码入口：

- [`main_ppo.py::run_ppo`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L34)
- [`main_ppo.py::TaskRunnerV1.run`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L133)
- [`main_ppo.py::main`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L166)

### 初始化时会准备什么

启动之后，系统依次准备 tokenizer、dataloader、资源池、模型 worker、reward loop、rollout server 和 checkpoint engine：

```text
tq.init(config.transfer_queue)
  -> trainer = PPOTrainerSync(config)
  -> trainer.init()
       -> _init_tokenizer()
       -> _init_dataloader()
       -> create_resource_pool()
       -> spawn worker groups
       -> init actor / reference engine
       -> init RewardLoopManager
       -> init LLMServerManager
       -> init CheckpointEngineManager
       -> sleep rollout replicas
       -> load checkpoint
  -> init_agent_loop_manager()
  -> trainer.fit(agent_loop_manager)
```

`PPOTrainer._setup()` 先准备 tokenizer 和 dataloader，再建立 resource pool 与 worker group。actor / rollout worker 随后初始化 model engine；如果当前配置需要 reference policy，它会复用 colocated worker 或使用独立 reference worker。最后创建 reward loop、LLM server 和 checkpoint engine。

sync trainer 将 checkpoint backend 强制设为 `naive`。初始化结束后，`PPOTrainerSync.on_init_end()` 立即把 actor checkpoint 的权重同步给 rollout replica。

源码入口：

- [`TaskRunnerV1.run`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L133)
- [`PPOTrainer.init / _setup`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L219)
- [`PPOTrainer._setup` 的 checkpoint backend](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L352)
- [`PPOTrainerSync.on_init_end`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_sync.py#L24)

## 一轮 update 怎样运行

```text
fit
  -> step
       -> prepare_step
            -> _add_batch_to_generate
                 -> _next_train_batch
                 -> _submit_batch_to_rollout
       -> _step_once
            -> ReplayBuffer.sample
            -> reward (when colocated path is active)
            -> _balance_batch
            -> _compute_old_log_prob
            -> _compute_ref_log_prob
            -> _compute_advantage
            -> _update_actor
       -> save checkpoint (if scheduled)
       -> on_step_end
            -> update rollout weights
       -> metrics / rollout dump
       -> tq.kv_clear
```

`prepare_step` 先把下一批 prompt 交给 AgentLoop。它不会在这里同步等待所有 response，而是先把 prompt 注册到 TransferQueue，再让后台 rollout 逐步写回结果；`_step_once` 随后从 ReplayBuffer 取得已经结束的 prompt group。

训练和 validation 都通过 `agent_loop_manager.generate_sequences(batch)` 提交生成，但默认并不是多轮工具调用。默认 `agent_name` 是 `single_turn_agent`，普通单轮 response 也会经过 AgentLoop；只有数据或配置指定 `tool_agent` 时，才会进入 tool call 和 observation 的多轮交互。

源码入口：

- [`PPOTrainer.fit`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L389)
- [`PPOTrainer.step`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L511)
- [`PPOTrainer._step_once`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L540)
- [`prepare_step / _submit_batch_to_rollout`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1403)
- [`TaskRunnerV1.init_agent_loop_manager`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L111)
- [`AgentLoopWorkerTQ.generate_sequences` 的默认 agent](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L59)

## Prompt 如何变成一组 trajectory

`_fetch_one_gen_batch` 从 `RLHFDataset` 取出一个 dataloader batch，为每个 `raw_prompt` 生成一个 UUID 作为 `uid`。`_submit_batch_to_rollout` 把这个 uid 注册到 TransferQueue，并写入 `is_prompt=True`、`status=pending` 和 `global_steps`，随后调用 `AgentLoopManagerTQ.generate_sequences`。

```text
dataset row
  -> raw_prompt + extra_info
  -> uid = UUID per prompt
  -> prompt key in TransferQueue
  -> AgentLoopManagerTQ
  -> AgentLoopWorkerTQ
  -> rollout.n parallel sessions
```

在 `AgentLoopWorkerTQ._run_prompt` 中，同一个 uid 的多个 session 会并行启动，每个 session 由 `session_id` 标识。所有 session 结束后，prompt tag 才会变为 `finished`；出现错误时则变为 `failure`。因此，GRPO group 由一个 `uid` 以及这个 uid 下的 `rollout.n` 个 session 构成。

一个 session 就是同一 prompt 的一次独立 rollout / episode。它可以只生成一次 response，也可以经历多轮“生成、工具执行、环境反馈”。`rollout.n=G` 表示每个 uid 启动 G 个独立 session，而不是让一个 session 重复生成 G 次。

源码入口：

- [`_fetch_one_gen_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1373)
- [`AgentLoopManagerTQ.generate_sequences`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L243)
- [`AgentLoopWorkerTQ._run_prompt`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L107)

### 一条 trajectory 如何表示

AgentLoop 的每个输出写入 TransferQueue 时使用：

```text
{uid}_{session_id}_{index}
```

`index` 用来区分一个 session 返回的多个 `AgentLoopOutput`。当前 revision 的内置 `SingleTurnAgentLoop` 和 `ToolAgentLoop` 通常每个 session 返回一个 output，所以常见 key 是 `{uid}_{session_id}_0`。它不表示同一 session 内又做了一次独立 rollout。每条 output 至少会转换成：

```text
prompt_ids + response_ids -> input_ids
response_mask            -> loss_mask
input_ids + attention    -> position_ids
AgentLoopOutput.extra_fields -> trajectory metadata
```

`response_ids` 可能同时包含模型生成 token 和 tool response / observation token。`response_mask` 用 1 标出由 policy 生成、可以进入 policy loss 的 token，用 0 标出 observation 或 padding token。

源码入口：

- [`AgentLoopOutput`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/agent_loop.py#L88)
- [`AgentLoopOutput` 后处理与 TQ 写入](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L150)
- [`generate_sequences` 的 response / response_mask 约定](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/agent_loop.py#L537)
- [内置 `SingleTurnAgentLoop`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/single_turn_agent_loop.py#L28)
- [内置 `ToolAgentLoop`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/tool_agent_loop.py#L98)

## Reward 怎样变成 actor update

当 ReplayBuffer 取出一组已经完成的 trajectory 后，`_step_once` 按下面的顺序处理：

```text
1. ReplayBuffer.sample()
2. optional colocated reward
3. _balance_batch()
4. _compute_old_log_prob()
5. optional _compute_ref_log_prob()
6. optional _compute_values()
7. _compute_advantage()
8. optional _update_critic()
9. _update_actor()
```

GRPO 配置通常不需要 critic，因此 `values`、GAE 和 `update_critic` 分支会被跳过。若开启 actor KL loss，reference policy 仍可能存在：它服务于 actor loss 的 KL 项，不等同于 GRPO 的 group-relative baseline。

在 rule-based reward 路径中，`RewardLoopWorker.compute_score` 调用注册的 reward manager，再把结果写回 output。`_compute_advantage` 将 `rm_scores` 变成 token-level reward；如果打开 `use_kl_in_reward`，还会把 KL penalty 加进这条 reward 路径。本入口脚本关闭了 `algorithm.use_kl_in_reward`，因此这里的任务 reward 与 actor loss 中的 policy KL 是两条不同的路径。

普通 GRPO outcome advantage 先为每条 response 得到一个 scalar score，再按 uid 计算 group mean 和可选 std，把相对分数 broadcast 到 response token，最后乘上 `response_mask`。V1 的 multi-trajectory helper 以每个 session 的最终 output 参与 group 比较，再把这个 session 的结果广播回它的 output。

如果自定义 AgentLoop 把一个 session 表示成多个 output，helper 会用最大 `index` 对应的最终结果参与同一 uid 下的 session 比较，再把 session-level advantage 回填给这个 session 的所有 output。默认内置 AgentLoop 每个 session 只有一个 output，因此多轮对话不会被额外切成多个 group；完整 trajectory 中的有效 policy token 会共享同一个 outcome advantage。

源码入口：

- [`RewardLoopWorker.compute_score`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/reward_loop/reward_loop.py#L145)
- [`PPOTrainer._compute_advantage`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1650)
- [`compute_grpo_outcome_advantage`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/core_algos.py#L267)
- [`compute_advantage_for_multi_trajectories`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/utils.py#L148)

### 三个 policy 角色

`_compute_old_log_prob` 默认让 actor engine 对 sampled batch 重新计算 log-prob，并将结果写为 `old_log_probs`；只有 rollout correction 的 bypass mode 才会直接把 `rollout_log_probs` 作为 `old_log_probs`。`_compute_ref_log_prob` 则通过独立 reference worker 或 actor no-LoRA 路径写入 `ref_log_prob`。

可以这样区分：

- `old_log_probs` 是 PPO ratio 的 proximal anchor；
- `ref_log_prob` 是 reference KL 的参照；
- current actor 在 `train_mini_batch` 中更新；
- rollout policy 负责产生 trajectory，但不必与 old policy 完全相同。

### actor 怎样更新

`_update_actor` 将 actor 的 `ppo_mini_batch_size` 乘以 `rollout.n`，把 mini-batch size、epoch、shuffle、temperature 等信息写入 `batch.extra_info`，再调用 `ActorRolloutRefWorker.update_actor`。worker 侧的 `TrainingWorker.train_mini_batch` 按 DP size 拆分 batch，按 epoch 构造 iterator，逐个执行 `train_batch`，最后聚合 metrics。

设 `B=train_batch_size`、`M=ppo_mini_batch_size`、`G=rollout.n`、`E=ppo_epochs`。普通 sync GRPO 且每个 session 只有一个 output 时：

```text
一次 step 的 prompt groups       = B
一次 step 的 trajectories        = B × G
一个全局 PPO mini-batch          = M × G trajectories
每个 PPO epoch 的 mini-batches   = B / M
一次 step 的 actor mini-batch updates = E × B / M
```

因此两个配置值都以 prompt-group 规模表达，`rollout.n` 在进入 actor 前将它们扩展为 trajectory 数。`train_batch_size` 控制本轮采样和 advantage 计算的数据规模，`ppo_mini_batch_size` 控制 advantage 固定后每次 actor optimization 消费的数据规模；通常应满足 `M <= B`，并使 `B` 能被 `M` 整除。

`update_actor` 是 trainer 把公共 batch 交给 training backend 的位置。FSDP 下，一个 DP rank 对应一个 FSDP worker rank，必要时再与 Ulysses SP 协同；Megatron 下，一个 DP replica 由 TP / PP / CP ranks 共同组成，batch 还会进入 pipeline forward-backward schedule。因此 `mini_batch_size_per_gpu = mini_batch_size / DP` 更准确地表示每个 DP replica 的样本份额，不能简单理解成每张 GPU 都独立拿到一份样本。

源码入口：

- [`PPOTrainer._compute_old_log_prob`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1541)
- [`PPOTrainer._compute_ref_log_prob`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1602)
- [`PPOTrainer._update_actor`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1734)
- [`TrainingWorker.train_mini_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine_workers.py#L241)

## 更新后的权重怎样回到 rollout

```text
on_sample_end
  -> sleep rollout replicas
  -> ReplayBuffer returns terminal trajectories
  -> actor update
  -> on_step_end
       -> update_weights(..., mode="naive")
       -> actor worker directly updates colocated rollout
       -> next rollout uses new actor weights
```

`PPOTrainerSync.on_sample_end` 让 rollout replica 进入休眠；`on_step_end` 调用 checkpoint manager。sync 模式的 `naive` backend 直接调用 actor worker group 的 `update_weights`，worker 侧恢复必要权重、更新 rollout 参数、清理 KV cache，再恢复下一轮生成。

这给出了 sync 的版本边界：一轮采样得到的 batch 先完成 actor update，actor update 结束后才把新权重交给下一轮 rollout。它与 async trainer 允许 rollout 跨越多个 actor version 的语义不同。

源码入口：

- [`PPOTrainerSync.on_sample_end / on_step_end`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_sync.py#L35)
- [`CheckpointEngineManager.update_weights`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/checkpoint_engine/base.py#L505)
- [`ActorRolloutRefWorker.update_weights`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine_workers.py#L726)

## 读完这条路径应该记住什么

### sync 的关键顺序

AgentLoop 可以在后台并行推进，TransferQueue 也保存生成过程中的状态；但对 trainer 来说，一轮 update 仍然要等当前 prompt group 完成，并在 actor update 后统一同步新权重。

### group、trajectory 和 token

- `uid` 连接同一 prompt 的 sampled trajectories；
- `{uid}_{session_id}_{index}` 连接某条具体 output；
- `response_mask` 区分 response 容器中的 policy token 与 observation / padding token。

### GRPO 少了哪一部分

GRPO 省略 value inference、GAE 和 critic update，但仍然需要 reward、old log-prob、可选 reference log-prob、advantage、actor mini-batch 和权重同步。

## 证据范围

- 调用链、字段名称、sync hooks、TQ key 格式和分支判断来自 revision `5b79827b04cee6e4b7b5ff737e047f1d72d50433`。
- 当前还没有在真实 GPU、vLLM、数据集和完整 Ray / TransferQueue 依赖下运行这个 example，因此吞吐、显存、实际 shape 和 runtime timing 仍不能从本文得出。
- 关于同步版本边界、mask 对 agent policy objective 的意义、TQ 的职责和每个 DP replica 的 batch 含义，属于基于源码关系做出的解释；Lab 01 和后续 backend 对照可以继续检验这些解释。

## 下一步看什么

下一步可以用 Lab 01 保存 resolved Hydra config，并跟踪一个 prompt group 的：

```text
uid
  -> trajectory keys
  -> prompt_len / response_len / seq_len
  -> response_mask.sum()
  -> reward / rm_scores
  -> old_log_probs / ref_log_prob
  -> advantages / returns
  -> actor mini-batch metadata
  -> update_weights global step
```

实验入口见 [[projects/verl-source-reading/labs/01-trace-config-and-batch|Lab 01: Trace Config and Batch]]；字段级说明见 [[projects/verl-source-reading/03-data-protocol-and-batch-lifecycle|03 Data Protocol and Batch Lifecycle]]。

## 相关知识

- [[training/post-training/grpo|GRPO]]
- [[training/post-training/ppo|PPO]]
- [[training/distributed-training/fsdp|FSDP]]
- [[inference/serving-systems/vllm|vLLM]]
