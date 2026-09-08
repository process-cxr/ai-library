---
title: verl Source Reading Roadmap
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-03
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - roadmap
---

这份路线图按照训练数据流与工程依赖组织 verl 源码阅读。顺序不追随仓库目录，也不同时展开所有算法和 backend：先建立一条能够闭合的同步训练链，再沿着数据、算法、调度、执行后端和异步系统逐层深入。

项目定位、系统地图与固定源码版本见 [[projects/verl-source-reading/|verl Source Reading]]。

## 路线总览

```text
sync GRPO 闭环
  -> config 与 batch lifecycle
  -> rollout / reward / AgentLoop
  -> advantage / policy loss / PPO comparison
  -> Single Controller / Ray
  -> FSDP / Megatron backend
  -> hybrid engine / weight sync
  -> agentic RL
  -> async / off-policy
```

每一轮都沿用前一轮已经确认的对象和字段。新的专题只增加必要变量：从 GRPO 到 PPO 时增加 critic / value / GAE；从 FSDP 到 Megatron 时替换 worker-side execution；从 sync 到 async 时增加 policy version 与 replay semantics。

SFT 不作为 RL 阅读路线中的一轮，而作为一条辅助对照路径单独阅读：[[projects/verl-source-reading/verl-sft-training-path|verl SFT Training Path]]。它用于建立 `messages -> loss_mask -> sft_loss -> TrainingWorker -> engine -> checkpoint` 的监督训练链，并帮助区分通用 engine 能力与 RL 特有的 rollout、reward、advantage 逻辑。

## 第一轮：Sync GRPO 闭环

**核心问题**

```text
一次 sync GRPO update 如何从 prompt 开始，
经过 rollout、reward 和 advantage，最终完成 actor update？
```

**阅读入口**

1. [`run_qwen3_4b_fsdp.sh`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/grpo_trainer/run_qwen3_4b_fsdp.sh)
2. [`main_ppo.py`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py)
3. [`TaskRunnerV1`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L102)
4. [`PPOTrainer.init / fit / step`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L219)
5. [`PPOTrainerSync hooks`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_sync.py)

**阅读产物**

- [[projects/verl-source-reading/01-sync-grpo-run-overview|01 Sync GRPO Run Overview]]
- [[projects/verl-source-reading/03-data-protocol-and-batch-lifecycle|03 Data Protocol and Batch Lifecycle]]
- [[projects/verl-source-reading/labs/01-trace-config-and-batch|Lab 01: Trace Config and Batch]]

当前已经完成固定 revision 下的静态调用链、session / trajectory 层级、字段 lineage、batch-size 语义和 FSDP baseline 边界；下一步是用最小 GPU run 将实际 shape、device、timing 和 weight version 回填为 tested behavior。

## 第二轮：Config 与数据生命周期

**核心问题**

- shell overrides 如何经过 Hydra composition 变成最终 runtime config？
- `RLHFDataset`、`TensorDict`、`DataProto`、`KVBatchMeta` 各自处于哪一层？
- TransferQueue 为什么将 tag 与 value 分开？
- nested、padded、DP-balanced 和 micro-batch representation 如何转换？

**阅读顺序**

```text
ppo_trainer.yaml and config groups
  -> Hydra resolved config
  -> RLHFDataset / collate_fn
  -> prompt uid
  -> AgentLoopOutput
  -> TransferQueue fields
  -> ReplayBuffer materialization
  -> worker dispatch
```

**阅读产物**

- [[projects/verl-source-reading/02-config-and-hydra|02 Config and Hydra]]
- [[projects/verl-source-reading/03-data-protocol-and-batch-lifecycle|03 Data Protocol and Batch Lifecycle]]

这一轮完成后，应能从任意一个训练字段反向回答它的 producer、consumer、shape、mask、存储位置和生命周期。

## 第三轮：Rollout、Reward 与 AgentLoop

**核心问题**

- 为什么 V1 将普通单轮生成也统一放入 AgentLoop？
- `rollout.n` 如何形成同一 prompt 的多个独立 sessions？
- multi-turn trajectory 中哪些 token 是 policy action，哪些是 observation？
- rule-based reward、custom reward 和 reward model 分别在哪里执行？
- reward 是在 rollout 过程中并行计算，还是在采样后 colocated 计算？

**阅读顺序**

```text
AgentLoopManager
  -> AgentLoopWorker
  -> SingleTurnAgentLoop / ToolAgentLoop
  -> LLMServerManager
  -> AgentLoopOutput / response_mask
  -> RewardLoopWorker / RewardLoopManager
  -> rm_scores
```

**阅读产物**

- [[projects/verl-source-reading/04-rollout-reward-and-agent-loop|04 Rollout, Reward and AgentLoop]]
- [[projects/verl-source-reading/labs/02-custom-reward|Lab 02: Custom Reward]]

## 第四轮：Advantage 与 Policy Loss

**核心问题**

```text
rm_scores
  -> token_level_scores / rewards
  -> advantages / returns
  -> old / reference / current log-prob
  -> policy loss / KL loss
```

先完成 GRPO outcome advantage、group normalization、loss mask 和 vanilla PPO loss，再加入 critic、value inference、GAE 与 value loss。算法变体只有在 baseline 字段链明确之后再展开。

从 GRPO 切换到 PPO 时，只追踪新增分支：

```text
critic initialization
  -> value inference
  -> GAE and returns
  -> critic update
```

**阅读产物**

- [[projects/verl-source-reading/05-advantage-and-policy-loss|05 Advantage and Policy Loss]]
- [[projects/verl-source-reading/09-ppo-and-grpo-codepath|09 PPO and GRPO Codepath]]
- [[projects/verl-source-reading/labs/03-compare-grpo-ppo|Lab 03: Compare GRPO and PPO]]

## 第五轮：Single Controller 与 Ray

**核心问题**

```text
controller 上的一次 Python method call，
如何变成多个 GPU worker 上的分布式执行？
```

依次追踪 `@register`、dispatch / collect、WorkerGroup、ResourcePool、colocated worker class 和 Ray actor handle。重点不是罗列 Ray API，而是建立：

```text
logical role
  -> resource pool
  -> worker group
  -> dispatch mode
  -> DP / model-parallel rank
  -> returned metadata or tensors
```

**阅读产物**

- [[projects/verl-source-reading/06-single-controller-and-ray|06 Single Controller and Ray]]

## 第六轮：Model Engine 与 Training Backend

**核心问题**

- 公共 `TrainingWorker` 如何调用不同 model engine？
- 相同的全局 PPO mini-batch 如何映射到不同 process-group topology？
- FSDP 的 shard mesh、Ulysses SP 与 logical DP 如何关联？
- Megatron 的 TP / PP / CP / DP / EP 如何改变 batch、forward-backward 和 optimizer？
- checkpoint 与参数导出的 backend contract 是否一致？

先用 FSDP 解释公共 engine contract，再用同一组 controller-level 输入映射到 Megatron。对照时保持 rollout、reward 和 advantage 不变，只替换 worker-side execution，避免把算法差异与 backend 差异混在一起。

**阅读产物**

- [[projects/verl-source-reading/07-model-engine-and-backend|07 Model Engine and Backend]]

## 第七轮：Hybrid Engine 与 Weight Sync

**核心问题**

```text
actor training weights
  -> model-engine parameters
  -> checkpoint / direct / delta transport
  -> rollout replica
  -> next policy version
```

重点追踪 LLMServerManager、CheckpointEngineManager、sleep / wake、KV cache、abort / resume 和 `update_weights`。需要明确区分 rollout policy、old policy、current actor 和 reference policy，记录它们在每个时刻对应的版本。

**阅读产物**

- [[projects/verl-source-reading/08-hybrid-engine-and-weight-sync|08 Hybrid Engine and Weight Sync]]

## 第八轮：Agentic RL

**核心问题**

```text
assistant generation
  -> tool call
  -> tool execution
  -> observation
  -> next assistant generation
  -> termination / reward
```

这一轮集中检查 chat template、continuous token construction、response / loss mask、turn state、tool result truncation、failure handling、terminal reward 和 trajectory-level credit assignment。普通 language-model response 与 environment observation 必须在 token、mask 和 reward 三个层面保持边界一致。

**阅读产物**

- [[projects/verl-source-reading/11-agentic-rl-extension|11 Agentic RL Extension]]
- [[projects/verl-source-reading/labs/04-multiturn-tool-agent|Lab 04: Multi-turn Tool Agent]]

## 第九轮：Async 与 Off-policy

最后比较 V1 的 `sync`、`colocate_async` 和 `separate_async`：

- rollout 与 training 是否共享 GPU；
- generation 与 optimization 是否重叠；
- rollout 是否跨越多个 actor versions；
- ReplayBuffer 如何判断 trajectory 可训练性；
- stale group 如何 drop、wait 或 refill；
- old log-prob 如何作为 proximal anchor；
- rollout correction、partial rollout 和 parameter sync cadence 如何影响训练分布。

这一轮必须以已经完成的 sync 数据流为对照。异步系统增加的是时间和版本维度，不应重新发明 prompt、reward 和 advantage 的基本语义。

**阅读产物**

- [[projects/verl-source-reading/10-async-trainer-and-off-policy|10 Async Trainer and Off-policy]]

## 完成标准

完成这组源码阅读后，应能够独立回答：

1. 一个 prompt 如何形成 `rollout.n` 条独立 sessions；
2. session、AgentLoopOutput、trajectory key 和多轮 turn 分别是什么；
3. reward、advantage、old/reference/current log-prob 在哪里产生；
4. `train_batch_size`、`ppo_mini_batch_size`、`rollout.n` 和 DP 如何共同决定 batch；
5. GRPO 和 PPO 在相同 trainer 中经过哪些不同分支；
6. Ray WorkerGroup 如何把 controller 调用映射到分布式 workers；
7. FSDP 与 Megatron 为什么会给同一全局 batch 不同的本地执行布局；
8. actor update 后 rollout engine 如何获得新权重；
9. multi-turn observation 为什么不能进入 policy loss；
10. async trainer 的 off-policy 来自哪些 policy-version 差异；
11. 训练异常应从 rollout、reward、advantage、loss、调度、backend 还是 weight sync 哪一层定位。
