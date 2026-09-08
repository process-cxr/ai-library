---
title: Lab 01 - Trace Config and Batch
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-03
type: project
status: growing
tags:
  - projects
  - verl
  - experiment
  - tracing
---

## Objective

在固定 revision 上对一个最小的 sync GRPO run 做字段级 trace，回答两个问题：

1. resolved Hydra config 如何映射到 trainer、rollout、reward 和 actor update 的 runtime object；
2. 一个 prompt group 如何从 dataset 经过 AgentLoop / TransferQueue，最终进入 advantage 和 actor mini-batch。

本 lab 当前完成的是运行设计和静态 source trace，尚未在真实 GPU、vLLM、Ray、TransferQueue 和模型数据条件下执行。因此本页的预期结果不标记为已测结果。

## Fixed Scope

- Repository：[process-cxr/verl-upstream](https://github.com/process-cxr/verl-upstream)
- Revision：`5b79827b04cee6e4b7b5ff737e047f1d72d50433`（[fixed source tree](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433)）
- Entry：[run_qwen3_4b_fsdp.sh](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/grpo_trainer/run_qwen3_4b_fsdp.sh)
- Algorithm：`grpo`
- Trainer mode：`sync`
- Rollout backend：`vllm`
- Training backend：`FSDP`
- Reward path：优先验证 reward model 关闭时的 rule-based reward

模型路径、数据路径、GPU 数量、Ray runtime 和实际 tokenizer 由运行环境提供，不在知识库中写死。

## Trace Stages

```text
config resolve
  -> dataset item / collated batch
  -> uid assignment
  -> prompt tag in TransferQueue
  -> rollout.n sessions
  -> trajectory fields and tags
  -> terminal prompt group
  -> ReplayBuffer.sample
  -> batch balance / optional padding
  -> rm_scores
  -> old_log_probs / entropy
  -> ref_log_prob
  -> advantages / returns
  -> actor mini-batch metadata
  -> metrics
  -> tq.kv_clear
```

这个顺序对应源码中的两条交错路径：`prepare_step` 先提交 prompt；`_step_once` 再从 TQ 等待可采样的 terminal group。trace 必须同时记录 prompt 提交时间和 sample 时间，才能看出生成等待与训练计算的边界。

## Configuration Trace

记录 resolved config 时，至少保存以下字段及其来源：

```text
trainer.use_v1
trainer.v1.trainer_mode
data.train_files
data.train_batch_size
data.gen_batch_size (if set)
data.max_prompt_length
data.max_response_length
data.filter_overlong_prompts
data.truncation
algorithm.adv_estimator
algorithm.use_kl_in_reward
algorithm.norm_adv_by_std_in_grpo
actor_rollout_ref.actor.ppo_mini_batch_size
actor_rollout_ref.actor.ppo_epochs
actor_rollout_ref.actor.use_kl_loss
actor_rollout_ref.actor.strategy
actor_rollout_ref.actor.fsdp_config.fsdp_size
actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size
actor_rollout_ref.rollout.name
actor_rollout_ref.rollout.n
actor_rollout_ref.rollout.tensor_model_parallel_size
actor_rollout_ref.rollout.calculate_log_probs
reward.reward_model.enable
reward.custom_reward_function.path
trainer.nnodes
trainer.n_gpus_per_node
transfer_queue.enable
```

逐项标注三个来源：

```text
script override
  -> Hydra config default
  -> resolved runtime value
```

尤其要记录 `use_kl_loss`、`algorithm.use_kl_in_reward` 和 `reward.reward_model.enable` 的最终值。它们分别影响 reference log-prob、reward shaping 和 reward 计算路径，不能用相近字段名替代。

源码入口：

- [`main` 的 Hydra 入口与 config validation](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L166)
- [`run_qwen3_4b_fsdp.sh` 的 overrides](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/grpo_trainer/run_qwen3_4b_fsdp.sh#L60)
- [`print_cfg.py`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/scripts/print_cfg.py)

## Parallel Topology Trace

本实验固定使用 FSDP，但不能只记录 GPU 数量。运行开始时需要保存：

```text
world_size
actor strategy
FSDP strategy / fsdp_size
Ulysses SP size
engine-reported actor DP size
rollout backend / rollout TP
rollout replica count
```

当前基线应满足：

```text
world_size = trainer.nnodes × trainer.n_gpus_per_node
actor DP   = world_size / Ulysses SP
```

同时验证 `rollout.tensor_model_parallel_size` 没有被用于计算 actor DP。以默认 8 GPU、Ulysses SP=1、rollout TP=2 为例，预期 topology 是 actor DP=8，而不是 4；rollout TP=2 描述的是生成侧副本布局。

后续用同一组 controller-level batch 参数切换 Megatron backend 时，应增加：

```text
Megatron TP / PP / CP / EP
dynamic_context_parallel
MCore-reported DP size
num_microbatches
pipeline stage / virtual pipeline stage
```

对照的目标不是要求两套 backend 产生相同的本地 shape，而是验证二者消费相同的全局 trajectory batch，并正确保持 uid、mask、reward 和 advantage 语义。

源码入口：

- [FSDP `get_data_parallel_size`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L646)
- [Megatron process groups](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/megatron/transformer_impl.py#L239)
- [Megatron DP accessor](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/megatron/transformer_impl.py#L750)

## Prompt Group Trace

选取一个尚未经过 padding 的有效 prompt，记录：

```text
dataset row index
raw_prompt message count
raw_prompt token length after final formatting
extra_info.index
tools_kwargs keys
interaction_kwargs keys
uid
global_steps
```

随后追踪同一个 uid 的所有 key：

```text
prompt key:       {uid}
trajectory keys:  {uid}_{session}_{index}
```

每个 trajectory 记录：

```text
session_id / output index
prompt_len
response_len
seq_len
response_mask.sum()
num_turns
reward_score
min_global_steps / max_global_steps
status
```

预期的静态关系是：同一 uid 下有 `rollout.n` 个 session；若一个 session 只返回一个 output，则通常对应一个 trajectory key。若 AgentLoop 返回多个 output，`index` 继续区分同一 session 内的 output，后续 V1 multi-trajectory GRPO helper 会按 session 的最终 output 参与 group advantage。

这里需要在 trace 中明确记录：session 是一次独立 rollout / episode，内部多轮 tool interaction 不增加 `session_id`；只有新的独立采样才产生新的 session。当前内置 single-turn / tool AgentLoop 预期每个 session 只有一个 `index=0` output，出现多个 index 时应进一步确认是否使用了自定义 AgentLoop，而不能直接计为额外 rollout。

源码入口：

- [`_fetch_one_gen_batch` 的 uid 生成](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1373)
- [`AgentLoopWorkerTQ` 的 session 创建](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L107)
- [`trajectory key / tag 写入`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L177)

## Field Trace

对同一组 keys，在每个阶段只打印摘要，不打印完整 prompt、tool result 和 token 内容：

```text
stage                  fields
dataset                raw_prompt, extra_info, index
prompt submit          uid, global_steps, prompt tag
rollout output         prompts, responses, response_mask, loss_mask
trajectory materialize input_ids, position_ids, num_turns, extra_fields
reward                 rm_scores, reward_extra_info
old log-prob           old_log_probs, entropy, rollout_log_probs (optional)
reference              ref_log_prob (optional)
advantage              token_level_rewards, advantages, returns
actor update           mini_batch_size, epochs, global_batch_size, metrics
cleanup                batch.keys, partition_id, remaining TQ keys
```

每个 tensor / nested field 的摘要格式建议为：

```text
field=<name>
producer=<stage / object>
consumer=<next stage>
shape=<padded shape or per-row lengths>
dtype=<dtype>
device=<device>
valid_length=<per-row length if applicable>
mask_sum=<response_mask.sum if applicable>
```

### 必须单独核对的字段

- `response_ids` / `responses`：确认 observation token 是否包含在 response 容器中；
- `response_mask`：确认 policy token 数，而不是只看 `response_len`；
- `loss_mask`：确认当前实现是否始终等于 `response_mask`；
- `rm_scores`：确认 outcome reward 被放在哪些 token 位置；
- `old_log_probs`、`rollout_log_probs`：确认是否走 bypass mode；
- `ref_log_prob`：确认 reference policy 是否启用以及是否独立 worker；
- `advantages` / `returns`：确认有效区域、padding 区域和 group uid；
- `tags.is_padding`：确认 padding 是否影响有效样本 metrics；
- `min_global_steps` / `max_global_steps`：确认当前 sync run 是否产生跨 version trajectory。

## Recommended Hooks

正式运行前，优先在以下位置加入临时 logging 或 debug hook。hook 只输出 metadata，避免把完整 prompt、tool result 和 token 序列写入日志：

1. `PPOTrainer._fetch_one_gen_batch`：记录 dataloader row 到 uid 的映射；
2. `PPOTrainer._submit_batch_to_rollout`：记录 prompt keys、partition 和 submit global step；
3. `AgentLoopWorkerTQ._agent_loop_postprocess`：记录 trajectory key、长度、mask sum 和 version tags；
4. `ReplayBuffer.sample` 返回前：记录选中的 prompt uids、trajectory keys 和 padding 情况；
5. `PPOTrainer._compute_reward_colocate` 或 reward loop：记录 reward field 的 shape / valid length；
6. `PPOTrainer._compute_advantage` 写回前后：记录 uid、reward summary、advantage summary；
7. `PPOTrainer._update_actor`：记录 mini-batch metadata 和 actor metrics；
8. `PPOTrainer.fit` 的 cleanup 后：记录本次 batch keys 和 partition 剩余 TQ keys；
9. `PPOTrainerSync.on_step_end`：记录 weight sync 的 `global_steps` 和 rollout replica 状态。

对应源码链接：

- [`trainer_base.py` 的 batch 提交、reward、advantage、actor 路径](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py)
- [`agent_loop_tq.py` 的 trajectory 写入](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py)
- [`replay_buffer.py` 的 sampling 状态机](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/replay_buffer.py)

## Expected Trace Record

建议把每个阶段记录成一行 JSONL 或结构化日志。示意如下：

```json
{
  "stage": "advantage",
  "uid": "<prompt-group-id>",
  "trajectory_key": "<uid>_<session>_<index>",
  "global_steps": 0,
  "topology": {
    "world_size": 0,
    "actor_backend": "<fsdp-or-megatron>",
    "actor_dp": 0,
    "rollout_tp": 0
  },
  "fields": {
    "response_len": 0,
    "policy_token_count": 0,
    "reward_sum": 0.0,
    "advantage_mean": 0.0,
    "advantage_std": 0.0
  },
  "is_padding": false
}
```

数字 `0` 只是占位，不是本次运行结果。真实 trace 还应记录 `dtype`、`device`、nested row lengths 和 stage timestamp，以便区分数据搬运、rollout 等待、reward 计算和 actor compute 的耗时。

## Acceptance Criteria

本 lab 只有满足以下条件，才能把运行结论回填为 tested behavior：

- resolved config 可以从日志或保存文件复查，并能区分 script override 与 default；
- 至少一个有效 prompt group 的 uid、session、trajectory key 全链路一致；
- `rollout.n`、实际 session 数和 terminal prompt 状态能够相互印证；
- 至少一条多轮或带 observation 的 trajectory 能区分 `response_len` 与 `response_mask.sum()`；
- reward、old/ref log-prob、advantage 的 producer 和 consumer 与源码一致；
- padding sample 不进入有效 reward、metrics 和 loss 统计；
- actor update 的 mini-batch metadata 可追溯到 `_update_actor`；
- engine 报告的 actor DP 能由本次 backend topology 解释，并与实际 batch dispatch 一致；
- actor parallel dimensions 与 rollout TP 分开记录，没有用 rollout TP 推导 actor DP；
- actor update 后的 weight sync global step 与下一轮 rollout 的版本信息一致；
- cleanup 后当前 batch keys 不再残留在对应 TQ partition；
- 至少保存一份完整 trace summary，而不是只保存最终 reward 或训练 loss。

## Current Status

- **已完成**：固定 revision、入口脚本、调用链、字段 lineage 和 hook 位置的静态追踪。
- **待运行**：resolved config、真实 tensor shape / dtype / device、实际 rollout session 数、TQ 状态变化、reward 数值、actor update metrics 和权重同步耗时。
- **待回填页面**：运行成功后，将字段级结果补入 [[projects/verl-source-reading/03-data-protocol-and-batch-lifecycle|03 Data Protocol and Batch Lifecycle]]，将真实时序和 performance 观察补入 [[projects/verl-source-reading/01-sync-grpo-run-overview|01 Sync GRPO Run Overview]]。

## 相关知识

- [[projects/verl-source-reading/|verl Source Reading]]
- [[training/post-training/grpo|GRPO]]
- [[training/post-training/ppo|PPO]]
- [[training/distributed-training/fsdp|FSDP]]
