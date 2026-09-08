---
title: verl Source Reading
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-03
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - post-training
  - reinforcement-learning
---

verl（Volcano Engine Reinforcement Learning for LLMs）是一套面向大模型 post-training 的分布式 RL 训练框架。它并不只实现 PPO、GRPO 等算法公式，而是把 prompt data、rollout、环境交互、reward、advantage estimation、actor / critic update、分布式执行和参数同步组织成一个可以持续运行的系统。

这组源码阅读关注的核心不是“某个 API 怎样调用”，而是一次 policy update 如何跨越多种 runtime：

```text
prompt 从哪里进入系统？
  -> 谁负责生成 response 或 agent trajectory？
  -> reward 在哪里计算并写回？
  -> old / reference / current policy 如何区分？
  -> advantage 如何变成 actor / critic 的训练字段？
  -> batch 如何映射到 FSDP 或 Megatron workers？
  -> 更新后的参数如何回到 rollout engine？
```

## verl 的系统位置

普通 supervised training 的主链通常围绕 `data -> model -> loss -> backward -> optimizer` 展开。verl 也提供独立的 SFT 路径，见 [[projects/verl-source-reading/verl-sft-training-path|verl SFT Training Path]]；本项目的主线仍然聚焦 RL post-training，它在此之外增加了一个动态数据生产闭环：训练样本不是完全离线给定的，而是由当前或相近版本的 policy 在 rollout runtime 中生成，再经过 reward 和 advantage processing 进入 optimization runtime。

```text
                    ┌──────── rollout / environment ──────────┐
prompt dataset ───> policy generation ───> trajectory         │
                         │                    │               │
                         │                    v               │
                         │               reward / verifier    │
                         │                    │               │
                         └──── new weights <──┼───────────────┘
                                              v
                                  advantage / policy loss
                                              │
                                              v
                                      actor update
```

verl 的主要职责，就是管理这条闭环中的数据依赖、模型角色、分布式资源和版本关系。算法决定“优化什么”，training backend 决定“梯度如何计算”，rollout backend 决定“trajectory 如何生成”，controller 则把它们连接成一轮可执行的训练过程。

## 系统地图

从源码职责看，可以先把 verl 分成五层。

### 入口与配置

[`examples/grpo_trainer/`](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/grpo_trainer) 提供可运行配置，[`verl/trainer/main_ppo.py`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py) 负责 Hydra config、运行前校验、Ray 初始化和 TaskRunner 选择。阅读这一层需要建立“shell override、config default 和 runtime object”之间的对应关系。

### Trainer 与算法编排

[`verl/trainer/ppo/v1/`](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1) 负责一轮训练的控制流：提交 prompt、等待 trajectory、计算 reward、构造 log-prob 和 advantage、触发 actor / critic update、记录 metrics，并在 step 结束后清理或保留状态。

这一层表达的是 backend-independent 的训练语义。`train_batch_size`、`rollout.n`、prompt group、old policy 和 reference policy 都首先在这里获得含义。

### Trajectory、Reward 与状态存储

AgentLoop 把 single-turn generation 和 multi-turn tool interaction 统一为 trajectory；RewardLoop 对接 rule-based verifier、custom reward 或 reward model；TransferQueue 保存 prompt group 状态和逐步补齐的训练字段；ReplayBuffer 决定哪些 trajectory 可以进入当前 update。

这部分是 verl 从普通 model training framework 走向 agentic RL system 的关键：response 不再只是连续生成文本，还可能包含多轮 action、tool observation、失败状态和跨 policy version 的执行信息。

### Training backend

Actor、critic 和 reference model 可以由 FSDP、Megatron、TorchTitan、VeOmni 等 backend 执行。它们消费相同的 controller-level 字段，但 process-group topology、batch dispatch、forward-backward schedule、gradient communication、optimizer state 和 checkpoint layout 并不相同。

```text
global actor batch
  ├─ FSDP:    FSDP shard mesh + optional Ulysses SP
  └─ Megatron: TP / PP / CP / DP / EP + pipeline schedule
```

因此 `DP size` 不是一个脱离 backend 的固定数字，也不能由 rollout TP 推导。

### Rollout backend 与权重循环

vLLM、SGLang、TensorRT-LLM 或 HF rollout 负责生成；LLMServerManager 管理 rollout replicas；CheckpointEngineManager 负责 actor training weights 与 rollout weights 之间的切换和同步。sync、colocate async 与 separate async 的主要区别，也集中体现在 rollout 和 training 是否重叠、trajectory 会跨越多少 policy versions，以及何时同步新权重。

## 一次 V1 Training Run

当前源码主线使用 V1 trainer。一次 sync GRPO update 可以先压缩为：

```text
run script
  -> main_ppo.main
  -> TaskRunnerV1
  -> PPOTrainerSync.init
  -> AgentLoopManager
  -> PPOTrainer.fit
       -> prepare_step
       -> rollout.n sessions per prompt
       -> TransferQueue / ReplayBuffer
       -> reward
       -> old / reference log-prob
       -> GRPO advantage
       -> actor update
       -> rollout weight sync
       -> metrics and cleanup
```

这条链路是后续专题页共享的坐标。每篇笔记会向下展开其中一个局部，但最终都需要回答它与整轮训练的关系。

## 阅读地图

### 主运行链与数据协议

- [[projects/verl-source-reading/01-sync-grpo-run-overview|01 Sync GRPO Run Overview]]：从启动脚本追踪到 actor update 和 rollout weight sync。
- [[projects/verl-source-reading/02-config-and-hydra|02 Config and Hydra]]：解释 Hydra composition、CLI overrides 与 runtime config。
- [[projects/verl-source-reading/03-data-protocol-and-batch-lifecycle|03 Data Protocol and Batch Lifecycle]]：追踪 `DataProto`、`TensorDict`、`KVBatchMeta`、TransferQueue 和训练字段生命周期。
- [[projects/verl-source-reading/verl-sft-training-path|verl SFT Training Path]]：补充监督训练的数据、mask、loss、engine 和 checkpoint 对照链路。

### Rollout、Reward 与优化算法

- [[projects/verl-source-reading/04-rollout-reward-and-agent-loop|04 Rollout, Reward and AgentLoop]]：分析 single-turn、multi-turn、tool observation 与 reward 路径。
- [[projects/verl-source-reading/05-advantage-and-policy-loss|05 Advantage and Policy Loss]]：把 reward、advantage、KL 和 policy objective 映射到 tensor 字段。
- [[projects/verl-source-reading/09-ppo-and-grpo-codepath|09 PPO and GRPO Codepath]]：比较 PPO 与 GRPO 在相同 trainer 中的真实分支。

### 分布式控制与执行后端

- [[projects/verl-source-reading/06-single-controller-and-ray|06 Single Controller and Ray]]：解释 controller method 如何调度到 Ray workers。
- [[projects/verl-source-reading/07-model-engine-and-backend|07 Model Engine and Backend]]：比较公共 model-engine contract 与 FSDP / Megatron 执行语义。
- [[projects/verl-source-reading/08-hybrid-engine-and-weight-sync|08 Hybrid Engine and Weight Sync]]：分析训练态、生成态、显存切换和参数版本同步。

### Agentic RL 与异步训练

- [[projects/verl-source-reading/11-agentic-rl-extension|11 Agentic RL Extension]]：研究多轮 tool-use trajectory 的状态、mask、termination 和 credit assignment。
- [[projects/verl-source-reading/10-async-trainer-and-off-policy|10 Async Trainer and Off-policy]]：比较三种 V1 trainer mode、policy staleness 和 off-policy control。
- [[projects/verl-source-reading/labs/|verl Source Reading Labs]]：用 resolved config、字段 trace 和最小实验检验静态源码判断。

完整学习顺序见 [[projects/verl-source-reading/roadmap|verl Source Reading Roadmap]]。

## 阅读基线

- Repository：[process-cxr/verl-upstream](https://github.com/process-cxr/verl-upstream)
- Revision：[`5b79827b04cee6e4b7b5ff737e047f1d72d50433`](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433)
- Branch at reading start：`main`
- Reading start：2026-09-02
- Primary trainer：[`main_ppo.py -> TaskRunnerV1 -> ppo/v1`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L102)

源码链接固定到完整 commit，避免 upstream 演化导致函数、配置和行号与笔记失配。当前默认入口启用 V1 trainer，并支持 `sync`、`colocate_async` 和 `separate_async`；旧 `main_ppo_v0.py` 只用于理解历史教程和旧架构，不作为正文主线。

第一轮已经完成 `sync GRPO + FSDP + vLLM + rule-based reward` 的静态调用链和字段追踪。实际 GPU shape、runtime timing、显存行为和 backend performance 仍需通过 labs 验证，不能由静态源码直接替代。

## 与通用知识的连接

源码项目页保存 verl 特有的调用链、数据结构、调度和 backend contract。可迁移的算法与系统机制继续沉淀在通用主题页：

- [[training/post-training/ppo|PPO]]
- [[training/post-training/grpo|GRPO]]
- [[training/post-training/reward-model|Reward Model]]
- [[training/distributed-training/fsdp|FSDP]]
- [[training/distributed-training/megatron|Megatron 与 3D 并行]]
- [[training/distributed-training/torch-distributed|torch.distributed]]
- [[inference/serving-systems/vllm|vLLM]]
