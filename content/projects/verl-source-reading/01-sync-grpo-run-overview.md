---
title: 01 Sync GRPO Run Overview
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-03
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - grpo
---

本页以固定 revision 的一个训练入口为主线，追踪一次 `sync GRPO + FSDP + vLLM + rule-based reward` update 如何从 prompt 进入 rollout，经过 reward 与 advantage 计算，最后完成 actor update 和 rollout 权重同步。

这里记录的是静态源码追踪结果。当前环境尚未完成 GPU、模型、数据集和完整依赖条件下的端到端运行，因此文中将源码事实、待运行确认的行为和工程解读分开书写。

## Runtime Context

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

这些是入口脚本表达的配置意图，实际运行时仍应以 Hydra resolve 后打印出的完整 config 为准。

### 参与者和执行边界

```text
Hydra driver
  └─ Ray TaskRunnerV1
       ├─ PPOTrainerSync                 training orchestration
       ├─ ActorRolloutRefWorker          actor / reference model engine
       ├─ AgentLoopWorkerTQ              vLLM-backed agent generation
       ├─ RewardLoopWorker               rule-based reward
       └─ CheckpointEngineManager        actor -> rollout weight lifecycle
```

`sync` 描述训练节奏和权重版本关系，不意味着所有步骤在同一个 Python 进程中执行。driver、Ray actor、训练 worker、rollout worker 和 TransferQueue 之间仍有独立边界。

### Training backend 边界

本页选择 FSDP 作为第一条可追踪基线，因此从 `actor_rollout_wg.update_actor(batch)` 继续向下的 batch dispatch、forward / backward、gradient synchronization、optimizer 和 checkpoint 语义均以 FSDP 为准。进入 model engine 之前，以下关系属于 trainer 层的公共语义：

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

FSDP 的 `fsdp_size` 控制参数 shard group，不能直接等同于 actor 的 logical DP；Megatron 的 TP、PP、CP 会占用 model-parallel 维度，EP 还会建立 expert-specific groups。`actor_rollout_ref.rollout.tensor_model_parallel_size` 则属于 rollout backend，不参与上述 actor DP 计算。完整 backend 对照留在 [[projects/verl-source-reading/07-model-engine-and-backend|07 Model Engine and Backend]] 展开。

源码入口：

- [FSDP `get_data_parallel_size`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L646)
- [Megatron process-group 初始化](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/megatron/transformer_impl.py#L239)
- [Megatron `get_data_parallel_size`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/megatron/transformer_impl.py#L750)

## Call Chain

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

`main` 先根据配置检查 reference policy 和 critic 是否需要，再根据 `trainer.use_v1` 选择 V1 或 legacy trainer。V1 路径由 `run_ppo` 初始化 Ray，随后在 Ray actor 中执行 `TaskRunnerV1.run`。

源码入口：

- [`main_ppo.py::run_ppo`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L34)
- [`main_ppo.py::TaskRunnerV1.run`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L133)
- [`main_ppo.py::main`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L166)

### 初始化顺序

`TaskRunnerV1.run` 的初始化顺序为：

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

`PPOTrainer._setup()` 先准备 tokenizer 和 dataloader，再建立 resource pool 与 worker group。actor / rollout worker 初始化 model engine；如果 `need_reference_policy(config)` 为真，则 reference policy 复用 colocated worker 或使用独立 reference worker。之后创建 reward loop、LLM server 和 checkpoint engine。

sync trainer 将 checkpoint backend 强制设为 `naive`。初始化结束后，`PPOTrainerSync.on_init_end()` 立即把 actor checkpoint 的权重同步给 rollout replica。

源码入口：

- [`TaskRunnerV1.run`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L133)
- [`PPOTrainer.init / _setup`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L219)
- [`PPOTrainer._setup` 的 checkpoint backend](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L352)
- [`PPOTrainerSync.on_init_end`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_sync.py#L24)

### `fit` 中的一次 training step

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

`prepare_step` 先提交下一批 prompt。它不直接等待完整 response，而是把 prompt 注册到 TransferQueue 后触发 AgentLoop；随后 `_step_once` 从 ReplayBuffer 等待并采样已经结束的 prompt group。

当前 V1 的训练和 validation 都通过 `agent_loop_manager.generate_sequences(batch)` 提交生成，但这不等于所有样本都执行多轮工具交互。默认 `agent_name` 是 `single_turn_agent`，普通单轮 response 也被统一包装进 AgentLoop 协议；数据指定 `tool_agent` 时才进入多轮 LLM generation、tool call 和 observation。还可以通过 `agent_loop_manager_class` 替换 manager，只要实现 `generate_sequences` 并将结果写入 TransferQueue。

源码入口：

- [`PPOTrainer.fit`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L389)
- [`PPOTrainer.step`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L511)
- [`PPOTrainer._step_once`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L540)
- [`prepare_step / _submit_batch_to_rollout`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1403)
- [`TaskRunnerV1.init_agent_loop_manager`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L111)
- [`AgentLoopWorkerTQ.generate_sequences` 的默认 agent](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L59)

## Rollout Boundary

`_fetch_one_gen_batch` 从 `RLHFDataset` 取出 dataloader batch，并为每个 `raw_prompt` 生成一个 UUID，写入 `uid`。`_submit_batch_to_rollout` 为每个 uid 写入 `is_prompt=True`、`status=pending` 和 `global_steps` 标签，然后调用 `AgentLoopManagerTQ.generate_sequences`。

```text
dataset row
  -> raw_prompt + extra_info
  -> uid = UUID per prompt
  -> prompt key in TransferQueue
  -> AgentLoopManagerTQ
  -> AgentLoopWorkerTQ
  -> rollout.n parallel sessions
```

在 `AgentLoopWorkerTQ._run_prompt` 中，同一个 uid 的多个 session 并行启动；每个 session 由 `session_id` 标识。所有 session 成功后，prompt tag 才变为 `finished`；任一 session 出错则变为 `failure`。因此 GRPO group 由稳定的 `uid` 和 `rollout.n` 建立。

这里的 session 就是同一 prompt 的一次独立 rollout / episode：它从初始 prompt 出发，可以包含一次单轮生成，也可以包含多轮“生成、工具执行、环境反馈”，直到终止。`rollout.n=G` 表示每个 uid 启动 `G` 个彼此独立的 session，而不是在一个 session 内再产生 `G` 次 rollout。

源码入口：

- [`_fetch_one_gen_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1373)
- [`AgentLoopManagerTQ.generate_sequences`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L243)
- [`AgentLoopWorkerTQ._run_prompt`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L107)

### trajectory key 与 multi-turn output

AgentLoop 的每个输出写入 TransferQueue 时使用：

```text
{uid}_{session_id}_{index}
```

其中 `index` 区分一个 session 返回的多个 `AgentLoopOutput`。在当前 revision 中，内置 `SingleTurnAgentLoop` 和 `ToolAgentLoop` 都将一次完整 session 返回为单个 output，因此常见 key 是 `{uid}_{session_id}_0`；多个 index 主要是为返回 `list[AgentLoopOutput]` 的自定义实现保留，不应理解为同一 session 下的多次独立 rollout。每条 output 至少被转换为：

```text
prompt_ids + response_ids -> input_ids
response_mask            -> loss_mask
input_ids + attention    -> position_ids
AgentLoopOutput.extra_fields -> trajectory metadata
```

`response_ids` 可能同时包含模型生成 token 和 tool response / observation token；`response_mask` 用 `1` 标出 LLM 生成 token，用 `0` 标出 observation 或 padding token。

源码入口：

- [`AgentLoopOutput`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/agent_loop.py#L88)
- [`AgentLoopOutput` 后处理与 TQ 写入](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L150)
- [`generate_sequences` 的 response / response_mask 约定](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/agent_loop.py#L537)
- [内置 `SingleTurnAgentLoop`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/single_turn_agent_loop.py#L28)
- [内置 `ToolAgentLoop`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/tool_agent_loop.py#L98)

## Update Boundary

`_step_once` 的源码顺序是：

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

在 rule-based reward 路径中，`RewardLoopWorker.compute_score` 转到 registered reward manager 的 `run_single`。`_compute_advantage` 将 `rm_scores` 作为 `token_level_scores`；若开启 `use_kl_in_reward`，还会把 KL penalty 纳入 token-level reward。本入口脚本明确将 `algorithm.use_kl_in_reward=False`，因此 reward 与 policy KL 是两条不同路径。

普通 GRPO outcome advantage 对每条 response 汇总 scalar score，按 `uid` 计算 group mean 和可选 std，再把相对分数 broadcast 到 response token，并乘以 `response_mask`。V1 的 multi-trajectory helper 只用每个 session 的最终 output 参与 group 计算，再把该 session 的结果广播回其 output。

更准确地说，GRPO 的比较单位仍是 session：若自定义 AgentLoop 把同一个 session 表示为多个 output，helper 只用最大 `index` 对应的最终结果参与同一 uid 下的 session 间比较，再把得到的 session-level advantage 回填给该 session 的所有 output。默认内置 AgentLoop 每个 session 只有一个 output，因此该逻辑不会额外切分多轮对话，完整 trajectory 中的有效 policy token 都获得同一个 outcome advantage。

源码入口：

- [`RewardLoopWorker.compute_score`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/reward_loop/reward_loop.py#L145)
- [`PPOTrainer._compute_advantage`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1650)
- [`compute_grpo_outcome_advantage`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/core_algos.py#L267)
- [`compute_advantage_for_multi_trajectories`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/utils.py#L148)

### old / reference / current policy

`_compute_old_log_prob` 默认让 actor engine 对 sampled batch 重新计算 log-prob，并将结果写为 `old_log_probs`；只有 rollout correction 的 bypass mode 才会直接把 `rollout_log_probs` 作为 `old_log_probs`。`_compute_ref_log_prob` 则通过独立 reference worker 或 actor no-LoRA 路径写入 `ref_log_prob`。

因此：

- `old_log_probs` 是 PPO ratio 的 proximal anchor；
- `ref_log_prob` 是 reference KL 的参照；
- current actor 在 `train_mini_batch` 中更新；
- rollout policy 负责产生 trajectory，但不必与 old policy 完全相同。

### actor update

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

`update_actor` 是 controller 公共数据流进入 training backend 的明确分叉点。FSDP 下，一个 DP rank 对应一个 FSDP worker rank，必要时再与 Ulysses SP rank 协同；Megatron 下，一个 DP replica 由 TP / PP / CP ranks 共同组成，batch 还会进入 pipeline forward-backward schedule。因此 worker 源码中的 `mini_batch_size_per_gpu = mini_batch_size / DP` 更准确地应理解为“每个 DP replica 的样本份额”，不能跨 backend 简化成每张 GPU 独立持有不同样本。

源码入口：

- [`PPOTrainer._compute_old_log_prob`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1541)
- [`PPOTrainer._compute_ref_log_prob`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1602)
- [`PPOTrainer._update_actor`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1734)
- [`TrainingWorker.train_mini_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine_workers.py#L241)

## Sync Weight Lifecycle

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

`PPOTrainerSync.on_sample_end` 让 rollout replica 休眠；`on_step_end` 调用 checkpoint manager。sync 模式的 `naive` backend 直接调用 actor worker group 的 `update_weights`，worker 侧恢复必要权重、更新 rollout 参数、清理 KV cache，并恢复下一轮生成状态。

这给出了 sync 的版本边界：一轮采样得到的 batch 先完成 actor update，actor update 结束后才把新权重交给下一轮 rollout。它与 async trainer 允许 rollout 跨越多个 actor version 的语义不同。

源码入口：

- [`PPOTrainerSync.on_sample_end / on_step_end`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_sync.py#L35)
- [`CheckpointEngineManager.update_weights`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/checkpoint_engine/base.py#L505)
- [`ActorRolloutRefWorker.update_weights`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine_workers.py#L726)

## First-Round Findings

### sync 的核心约束

AgentLoop 可以在后台任务中并行推进，TransferQueue 也承担异步状态管理；sync 的关键约束是 trainer 在采样时等待当前 prompt group 完成，并在 actor update 后统一同步权重。

### group、trajectory 和 token 的三层关系

- `uid` 连接同一 prompt 的 sampled trajectories；
- `{uid}_{session_id}_{index}` 连接某条具体 output；
- `response_mask` 区分 response 容器中的 policy token 与 observation / padding token。

### GRPO 省略的是 critic 分支

GRPO 省略 value inference、GAE 和 critic update，但仍然需要 reward、old log-prob、可选 reference log-prob、advantage、actor mini-batch 和权重同步。

## Evidence Status

- **Source fact**：调用链、字段名称、sync hooks、TQ key 格式和分支判断均来自 revision `5b79827b04cee6e4b7b5ff737e047f1d72d50433`。
- **Tested behavior**：当前尚未在真实 GPU、vLLM、数据集和完整 Ray / TransferQueue 依赖下运行该 example，因此没有把吞吐、显存、实际 shape 或 runtime timing 标为已验证。
- **Engineering interpretation**：关于同步版本边界、mask 对 agent policy objective 的意义、TQ 职责和每个 DP replica 的 batch 含义，是基于源码调用关系的解释，后续需要 Lab 01 的 FSDP trace 及 Megatron 对照实验进一步确认。

## Next Trace

第一轮最小实验应保存 resolved Hydra config，并至少记录一个 prompt group 的：

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
